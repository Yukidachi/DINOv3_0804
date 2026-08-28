"""R4 server entry point: relation-only diagnostic training.

R4 is deliberately a diagnostic ablation of the R group.  It keeps the
Cityscapes/K-group data, student, teacher, optimiser, DDP and evaluation
protocol, removes the pointwise A0 feature-MSE term, and enables exactly one
screened relation objective (R1 or R2)::

    L = L_seg + warmup(step) * lambda_rel * L_selected_relation

The selected relation is resolved from the accepted seed-42 R1/R2 artifacts
using the pre-registered ordering (mIoU, small-object mIoU, boundary F1,
then R1).  ``--relation`` and ``--lambda-rel`` can be supplied explicitly for
an auditable replay.  R4 always runs with seed=42 and is never a main model
candidate.

The implementation routes the audited R1 runner through the R4 loss contract.
The runner still computes the locked PCA projection for compatibility with the
shared K1 resource/bootstrap path, but ``lambda_feat`` is forced to zero and
the emitted R4 configuration explicitly records that feature KD is disabled.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

import dino as common
import dino_a0_server as a0
import dino_k0_server as k0
import dino_k1_server as k1
import dino_r0_server as r0
import dino_r1_server as r1
import dino_r2_server as r2
import dino_s2_0 as base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "R_MobileNetV2_RASPP_server"
K_GROUP_OUTPUT_DIR = SCRIPT_DIR / "result" / "K_MobileNetV2_RASPP_server"
EXPERIMENT = "R4"
EXPERIMENT_GROUP = "R_MobileNetV2_RASPP_server"
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_r4_relation_only"
ARTIFACT_FORMAT_VERSION = 1
FORMAL_SEEDS = (42,)

RELATION_EPSILON = 1e-6
LAMBDA_REL = 0.03
ALLOWED_LAMBDA_R1 = (0.015, 0.03, 0.06)
ALLOWED_LAMBDA_R2 = (0.015, 0.03, 0.06, 0.3)
GRADIENT_CE_STOP_RATIO = 2.0
GRADIENT_CE_STOP_CONSECUTIVE = 3
FIXED_GRADIENT_AUDIT_STEPS = (1, 4_000, 20_000, 40_000, 60_000, 80_000)

_ACTIVE_RELATION = "r1"
_ACTIVE_LAMBDA = LAMBDA_REL
_RELATION_SPEC: Optional[Dict[str, object]] = None
_REFERENCE_TESTS: Optional[Dict[str, object]] = None
_R0_GATE: Optional[Dict[str, object]] = None
_SELECTED_GATE: Optional[Dict[str, object]] = None
_ORIGINAL_R1_TRAIN_ONE_EPOCH = r1.train_one_epoch_r1


def _format_lambda(value: float) -> str:
    return format(float(value), ".12g")


def _candidate_path(output_dir: Path, relation: str, value: float) -> Path:
    if relation == "r1":
        return output_dir.resolve() / "R1" / "seed_42" / "metrics.json"
    return (
        output_dir.resolve()
        / "R2"
        / f"seed_42_lambda_{_format_lambda(value)}"
        / "metrics.json"
    )


def _read_json(path: Path) -> Dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _metric_number(metrics: Mapping[str, object], key: str) -> Optional[float]:
    best = metrics.get("best_dev_metrics")
    if not isinstance(best, Mapping) or best.get(key) is None:
        return None
    try:
        return float(best[key])
    except (TypeError, ValueError):
        return None


def _candidate_lambda(metrics: Mapping[str, object], relation: str) -> Optional[float]:
    loss = metrics.get("loss")
    if not isinstance(loss, Mapping):
        return None
    key = "lambda_r1" if relation == "r1" else "lambda_r2"
    value = loss.get(key)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _candidate_passes(metrics: Mapping[str, object], relation: str) -> bool:
    if metrics.get("experiment") != relation.upper():
        return False
    if metrics.get("test_local_evaluated") is not False:
        return False
    reference = metrics.get("relation_reference_tests")
    if not isinstance(reference, Mapping) or not bool(reference.get("passed")):
        return False
    gate = metrics.get("gradient_gate")
    if not isinstance(gate, Mapping) or not bool(gate.get("passed_target_at_any_record")):
        return False
    comparison = metrics.get(f"{relation}_vs_r0")
    if not isinstance(comparison, Mapping):
        return False
    delta = comparison.get(f"delta_{relation.upper()}_minus_R0")
    # Match the R-group screening rule: the primary path needs the registered
    # mIoU margin; the mechanism path accepts the pre-registered boundary or
    # small-object margin.  R4 cannot promote an unaccepted single relation.
    try:
        primary_pass = float(delta) >= 0.00219
    except (TypeError, ValueError):
        primary_pass = False
    best = metrics.get("best_dev_metrics")
    r0_best = metrics.get("r0_best_dev_metrics")
    mechanism_pass = False
    if isinstance(best, Mapping) and isinstance(r0_best, Mapping):
        try:
            mechanism_pass = (
                float(best.get("boundary_f1", float("-inf")))
                - float(r0_best.get("boundary_f1", float("inf")))
                >= 0.00613
                or float(best.get("small_object_mIoU", float("-inf")))
                - float(r0_best.get("small_object_mIoU", float("inf")))
                >= 0.00851
            )
        except (TypeError, ValueError):
            mechanism_pass = False
    # Existing R1/R2 artifacts store the R0 metrics in their comparison block
    # rather than at the top level, so primary_pass is the normal route.  The
    # explicit mechanism fallback is retained for future artifacts.
    if not (primary_pass or mechanism_pass):
        return False
    return _metric_number(metrics, "mIoU") is not None


def _scan_candidates(args: argparse.Namespace) -> List[Dict[str, object]]:
    values = {
        "r1": ALLOWED_LAMBDA_R1,
        "r2": ALLOWED_LAMBDA_R2,
    }
    candidates: List[Dict[str, object]] = []
    for relation, lambdas in values.items():
        for value in lambdas:
            path = _candidate_path(args.output_dir, relation, value)
            if not path.is_file():
                continue
            metrics = _read_json(path)
            if not _candidate_passes(metrics, relation):
                continue
            recorded = _candidate_lambda(metrics, relation)
            if recorded is not None and not math.isclose(recorded, value, abs_tol=1e-12):
                continue
            candidates.append(
                {
                    "relation": relation,
                    "lambda": value if recorded is None else recorded,
                    "metrics": metrics,
                    "path": path,
                }
            )
    # Stable sort implements the pre-registered dictionary ordering.  The
    # final relation key intentionally prefers R1 on a complete tie.
    candidates.sort(
        key=lambda item: (
            float(_metric_number(item["metrics"], "mIoU") or float("-inf")),
            float(_metric_number(item["metrics"], "small_object_mIoU") or float("-inf")),
            float(_metric_number(item["metrics"], "boundary_f1") or float("-inf")),
            1 if item["relation"] == "r1" else 0,
        ),
        reverse=True,
    )
    return candidates


def _resolve_selection(args: argparse.Namespace) -> Tuple[str, float, Optional[Dict[str, object]]]:
    requested_relation = str(args.relation).lower()
    requested_lambda = args.lambda_rel
    candidates = _scan_candidates(args)
    if requested_relation == "auto":
        if requested_lambda is not None:
            candidates = [
                c
                for c in candidates
                if math.isclose(float(c["lambda"]), float(requested_lambda), abs_tol=1e-12)
            ]
            if not candidates and not args.smoke_test:
                raise RuntimeError(
                    f"No accepted R1/R2 candidate is available at lambda_rel={requested_lambda}"
                )
        if candidates:
            selected = candidates[0]
            return str(selected["relation"]), float(selected["lambda"]), selected
        if not args.smoke_test:
            raise RuntimeError(
                "R4 requires an accepted R1/R2 seed-42 candidate; none was found"
            )
        return "r1", LAMBDA_REL if requested_lambda is None else float(requested_lambda), None

    if requested_relation not in {"r1", "r2"}:
        raise SystemExit("--relation must be auto, r1, or r2")
    allowed = ALLOWED_LAMBDA_R1 if requested_relation == "r1" else ALLOWED_LAMBDA_R2
    if requested_lambda is not None:
        if not any(math.isclose(float(requested_lambda), v, abs_tol=1e-12) for v in allowed):
            raise SystemExit(
                f"--lambda-rel must be one of {', '.join(_format_lambda(v) for v in allowed)} "
                f"for {requested_relation.upper()}"
            )
        selected_value = float(requested_lambda)
    else:
        matching = [c for c in candidates if c["relation"] == requested_relation]
        if matching:
            selected_value = float(matching[0]["lambda"])
        else:
            if not args.smoke_test:
                raise RuntimeError(
                    f"No accepted {requested_relation.upper()} candidate is available for R4"
                )
            selected_value = LAMBDA_REL
    selected = next(
        (
            c
            for c in candidates
            if c["relation"] == requested_relation
            and math.isclose(float(c["lambda"]), selected_value, abs_tol=1e-12)
        ),
        None,
    )
    return requested_relation, selected_value, selected


def parse_args() -> argparse.Namespace:
    """Reuse the locked K1 CLI and add the R4 relation selector."""

    saved_default = k1.DEFAULT_OUTPUT_DIR
    saved_argparse = k1.argparse
    saved_lambda_feat = k1.LAMBDA_FEAT

    class R4ArgparseProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(saved_argparse, name)

        @staticmethod
        def ArgumentParser(*parser_args: Any, **parser_kwargs: Any):
            parser_kwargs["description"] = (
                "R4 MobileNetV2+R-ASPP: hard-label CE plus one screened "
                "native relation objective, without pointwise feature KD."
            )
            parser = saved_argparse.ArgumentParser(*parser_args, **parser_kwargs)
            parser.add_argument(
                "--relation",
                choices=("auto", "r1", "r2"),
                default="auto",
                help="Relation to retain; auto applies the pre-registered selector.",
            )
            parser.add_argument(
                "--lambda-rel",
                type=float,
                default=None,
                help="Registered weight for the selected relation.",
            )
            return parser

    k1.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    k1.argparse = R4ArgparseProxy()
    # K1's parser locks its own default/validation to LAMBDA_FEAT=1.0.  R4
    # deliberately changes that parser contract to the relation-only value.
    k1.LAMBDA_FEAT = 0.0
    try:
        args = k1.parse_args()
    finally:
        k1.DEFAULT_OUTPUT_DIR = saved_default
        k1.argparse = saved_argparse
        k1.LAMBDA_FEAT = saved_lambda_feat

    # The R protocol fixes physical batch 4, accumulation 2, and effective
    # optimizer batch 8.  Keep K1's compatibility behaviour for omitted CLI.
    if not any(
        value == "--accumulation-steps"
        or value.startswith("--accumulation-steps=")
        for value in os.sys.argv[1:]
    ):
        args.accumulation_steps = 2
    if args.seed != 42:
        raise SystemExit("R4 is a single seed=42 diagnostic by protocol")
    args.lambda_feat = 0.0
    if not args.smoke_test:
        if args.max_steps != 80_000:
            raise SystemExit("Formal R4 is locked to exactly 80,000 optimizer steps")
        if args.eval_every_steps != 5_000:
            raise SystemExit("Formal R4 is locked to --eval-every-steps 5000")
        if args.gradient_log_steps != 500:
            raise SystemExit("Formal R4 is locked to --gradient-log-steps 500")
    if args.output_dir.resolve() == K_GROUP_OUTPUT_DIR.resolve():
        raise SystemExit("R4 output must use R_MobileNetV2_RASPP_server, not K output")

    relation, value, _selected = _resolve_selection(args)
    args.relation = relation
    args.lambda_rel = value
    # The audited R1 loop uses this private alias for its relation weight.
    args.lambda_r1 = value
    args.lambda_r2 = value
    return args


def r4_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    original = k1._ORIGINAL_K0_PATHS(output_dir, seed)
    run_dir = output_dir.resolve() / "R4" / f"seed_{seed}"
    return {
        key: run_dir if key == "run_dir" else run_dir / value.name
        for key, value in original.items()
    }


def _canonical_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _relation_spec_r4(
    args: argparse.Namespace, accumulation_steps: int, world_size: int
) -> Dict[str, object]:
    physical_batch = int(args.batch_size) * int(world_size)
    relation = str(args.relation).upper()
    spec: Dict[str, object] = {
        "enabled": True,
        "experiment": "R4",
        "active_relation_types": [f"{relation}_cross_image" if relation == "R1" else "R2_within_image_spatial"],
        "relation_only": True,
        "feature_kd_enabled": False,
        "logit_kd_enabled": False,
        "relation_feature_source": {
            "teacher": "native OS=4/8/16 features",
            "student": "native OS=4/8/16 features",
            "a0_projected_features_used_for_relation": False,
        },
        "epsilon": RELATION_EPSILON,
        "physical_relation_batch_size": physical_batch,
        "nominal_physical_relation_batch_size": physical_batch,
        "effective_optimizer_batch_size": physical_batch * int(accumulation_steps),
        "accumulated_batches_used_for_relation": False,
        "relation_warmup_steps": 4_000,
        "warmup_step_unit": "optimizer_step",
        "lambda_rel": float(args.lambda_rel),
        "lambda_feat": 0.0,
        "diagonal_policy": "keep" if relation == "R1" else "keep for valid tokens",
        "matrix_dtype": "float32",
        "gradient_gate": {
            "relation_to_ce_stop_ratio": GRADIENT_CE_STOP_RATIO,
            "consecutive_records_before_stop": GRADIENT_CE_STOP_CONSECUTIVE,
            "relation_to_feature_ratio": "not_applicable_feature_kd_removed",
        },
    }
    if relation == "R1":
        spec.update(
            {
                "relation_representation": "masked GAP then physical BxB signed cosine matrix",
                "mask_policy": "nearest-resized targets != 255",
                "reduction": "sum squared error divided by actual B^2",
                "gather": "differentiable student all_gather; detached teacher all_gather",
                "token_count": None,
            }
        )
    else:
        spec.update(
            {
                "relation_representation": "masked adaptive 8x16 tokens then per-image 128x128 signed cosine matrix",
                "mask_policy": "nearest-resized valid mask; adaptive-average valid fraction > 0",
                "reduction": "global valid-pair masked sum divided by global valid-pair count",
                "pool_size": [8, 16],
                "num_tokens": 128,
                "token_order": "8x16 row-major",
            }
        )
    return spec


def r4_relation_losses(
    student_features: Mapping[str, torch.Tensor],
    teacher_features: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    world_size: int,
):
    """Dispatch exactly one native relation implementation."""

    if _ACTIVE_RELATION == "r1":
        return r1.r1_relation_losses(student_features, teacher_features, targets, world_size)
    return r2.r2_relation_losses(student_features, teacher_features, targets, world_size)


def run_relation_reference_tests(device: torch.device, world_size: int) -> Dict[str, object]:
    if _ACTIVE_RELATION == "r1":
        return r1.run_relation_reference_tests(device, world_size)
    return r2.run_r2_reference_tests(device, world_size)


def _validate_selected_gate(args: argparse.Namespace) -> Dict[str, object]:
    path = _candidate_path(args.output_dir, args.relation, args.lambda_rel)
    if not path.is_file():
        if args.smoke_test:
            return {
                "required_for_formal_run": True,
                "checked": False,
                "passed": None,
                "reason": "selected R1/R2 artifact absent; protocol smoke is allowed",
                "metrics_path": str(path),
            }
        raise FileNotFoundError(f"R4 requires the accepted relation artifact: {path}")
    metrics = _read_json(path)
    failures: List[str] = []
    if not _candidate_passes(metrics, args.relation):
        failures.append("selected relation artifact did not pass its reference/effect/gradient gates")
    recorded = _candidate_lambda(metrics, args.relation)
    if recorded is None or not math.isclose(recorded, args.lambda_rel, abs_tol=1e-12):
        failures.append(f"selected lambda differs: artifact={recorded}, requested={args.lambda_rel}")
    result: Dict[str, object] = {
        "required_for_formal_run": True,
        "checked": True,
        "passed": not failures,
        "failures": failures,
        "relation": args.relation.upper(),
        "lambda": float(args.lambda_rel),
        "metrics_path": str(path),
        "metrics_sha256": common.sha256_file(path),
        "selected_mIoU": _metric_number(metrics, "mIoU"),
    }
    if failures and not args.smoke_test:
        raise RuntimeError("R4 selected-relation gate failed:\n- " + "\n- ".join(failures))
    return result


def build_config_r4(
    args: argparse.Namespace,
    accumulation_steps: int,
    world_size: int,
    device: torch.device,
    shared_init_state_sha256: str,
    shared_init_file_sha256: str,
) -> Dict[str, object]:
    global _RELATION_SPEC, _REFERENCE_TESTS
    # This compatibility bootstrap loads the locked teacher/PCA resources;
    # the loss and protocol below explicitly disable pointwise feature KD.
    config = r1._ORIGINAL_K1_BUILD_CONFIG(
        args, accumulation_steps, world_size, device,
        shared_init_state_sha256, shared_init_file_sha256,
    )
    _RELATION_SPEC = _relation_spec_r4(args, accumulation_steps, world_size)
    _REFERENCE_TESTS = run_relation_reference_tests(device, world_size)
    if not args.smoke_test:
        if world_size != 2:
            raise RuntimeError(f"Formal R4 requires world_size=2, got {world_size}")
        if _RELATION_SPEC["physical_relation_batch_size"] != 4:
            raise RuntimeError("Formal R4 requires physical relation batch size 4")
        if _RELATION_SPEC["effective_optimizer_batch_size"] != 8:
            raise RuntimeError("Formal R4 requires effective optimizer batch size 8")
    r1._RELATION_SPEC = _RELATION_SPEC
    r1._REFERENCE_TESTS = _REFERENCE_TESTS
    config.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "server_entry_point": str(Path(__file__).resolve()),
            "formal_seeds": list(FORMAL_SEEDS),
            "relation": copy.deepcopy(_RELATION_SPEC),
            "relation_spec_sha256": _canonical_sha256(_RELATION_SPEC),
            "r0_gate": copy.deepcopy(r1._R0_GATE or _R0_GATE),
            "selected_relation_gate": copy.deepcopy(_SELECTED_GATE),
            "loss": {
                "hard_label_ce": True,
                "feature_kd": False,
                "logit_kd": False,
                "relation_kd": True,
                "relation_r1": args.relation == "r1",
                "relation_r2": args.relation == "r2",
                "lambda_feat": 0.0,
                "lambda_rel": float(args.lambda_rel),
                "warmup_steps": 4_000,
                "total": "CE + warmup * lambda_rel * selected_relation",
            },
        }
    )
    config["pca"] = {
        "enabled": False,
        "used_in_loss": False,
        "reason": "R4 relation-only ablation; PCA is loaded only by the shared K1 bootstrap",
    }
    return config


def audit_shapes_r4(
    model: base.MobileNetV2RASPPStudent,
    device: torch.device,
    height: int,
    width: int,
    amp_enabled: bool,
) -> Dict[str, object]:
    audit = r1._ORIGINAL_K1_AUDIT_SHAPES(model, device, height, width, amp_enabled)
    audit["experiment"] = EXPERIMENT
    audit["relation"] = copy.deepcopy(_RELATION_SPEC) if _RELATION_SPEC else {
        "active_relation": _ACTIVE_RELATION
    }
    audit["feature_kd_enabled"] = False
    return audit


def build_best_checkpoint_r4(*args: Any, **kwargs: Any) -> Dict[str, object]:
    payload = r1._ORIGINAL_K1_BUILD_BEST_CHECKPOINT(*args, **kwargs)
    payload.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "relation": copy.deepcopy(_RELATION_SPEC),
            "relation_spec_sha256": None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC),
            "selected_relation_gate": copy.deepcopy(_SELECTED_GATE),
        }
    )
    return payload


def _patched_torch_save_atomic_r4(payload: object, path: Path) -> None:
    if isinstance(payload, Mapping) and payload.get("artifact_type") == ARTIFACT_TYPE:
        payload = dict(payload)
        payload.update(
            {
                "experiment": EXPERIMENT,
                "experiment_group": EXPERIMENT_GROUP,
                "relation": copy.deepcopy(_RELATION_SPEC),
                "relation_spec_sha256": None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC),
                "selected_relation_gate": copy.deepcopy(_SELECTED_GATE),
            }
        )
    k1._ORIGINAL_TORCH_SAVE_ATOMIC(payload, path)


def train_one_epoch_r4(*args: Any, **kwargs: Any):
    """Run the audited loop with the feature term mathematically removed."""

    active_args = k1._ACTIVE_ARGS
    if float(active_args.lambda_feat) != 0.0:
        raise RuntimeError("R4 train loop received a non-zero feature lambda")
    metrics, steps, records, first_batch = _ORIGINAL_R1_TRAIN_ONE_EPOCH(*args, **kwargs)
    raw_relation = metrics.get("relation_r1_loss")
    metrics.update(
        {
            "loss_schema": "hard_label_CE_plus_relation_MSE_relation_only",
            "feature_kd_enabled": False,
            "feature_loss": None,
            "feature_loss_by_layer": None,
            "relation_r1_loss": raw_relation if _ACTIVE_RELATION == "r1" else None,
            "relation_r2_loss": raw_relation if _ACTIVE_RELATION == "r2" else None,
            "relation_loss_weighted_at_last_warmup": float(active_args.lambda_rel)
            * float(metrics.get("warmup_weight", 0.0))
            * float(raw_relation or 0.0),
        }
    )
    # R1's generic audit names remain useful, but make the selected relation
    # explicit in every record and remove the inapplicable feature ratio.
    for record in records:
        record["experiment"] = EXPERIMENT
        record["relation_type"] = "R1_cross_image" if _ACTIVE_RELATION == "r1" else "R2_within_image_spatial"
        record["feature_kd_enabled"] = False
        record["relation_to_feature_effective_ratio_os16"] = None
    if first_batch is not None:
        first_batch["experiment"] = EXPERIMENT
        first_batch["feature_kd_enabled"] = False
        first_batch["relation_type"] = "R1_cross_image" if _ACTIVE_RELATION == "r1" else "R2_within_image_spatial"
    return metrics, steps, records, first_batch


def smoke_test_r4(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    amp_enabled: bool,
    rank: int,
) -> None:
    teacher, _projection = k1._require_resources()
    args = k1._ACTIVE_ARGS
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    images, targets, paths = next(iter(loader))
    images, targets = images.to(device), targets.to(device)
    model.train(); teacher.eval(); model.zero_grad(set_to_none=True)
    with common.autocast_context(device, amp_enabled):
        output = model(images)
        if not isinstance(output, Mapping):
            raise RuntimeError("R4 smoke forward did not expose features")
        with torch.no_grad():
            teacher_features = teacher.extract_features(images)
        relation, _layers, audit = r4_relation_losses(
            output["features"], teacher_features, targets, world_size
        )
        logits = output["logits"].float()
    valid = int((targets != common.IGNORE_INDEX).sum().item())
    if valid <= 0:
        raise RuntimeError("R4 smoke batch contains no valid Cityscapes pixels")
    ce = F.cross_entropy(logits, targets, ignore_index=common.IGNORE_INDEX, reduction="sum") / valid
    total = ce + (1.0 / r1._warmup_steps(args)) * args.lambda_rel * relation
    total.backward()
    if not all(bool(torch.isfinite(v).all().item()) for v in (ce, relation, total)):
        raise RuntimeError("R4 smoke test produced a non-finite loss")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("R4 smoke test found a teacher gradient")
    if rank == 0:
        print(
            f"[OK] R4 server smoke: relation={_ACTIVE_RELATION.upper()}, sample={paths[0]}, "
            f"logits={tuple(logits.shape)}, CE={ce.item():.6f}, "
            f"relation={relation.item():.6f}, total={total.item():.6f}, audit={audit}"
        )


def _postprocess_metrics_r4(args: argparse.Namespace) -> None:
    # The original K1 postprocessor creates the standard metrics/history files
    # and then this pass changes only the R4 scientific contract.
    r1._ORIGINAL_K1_POSTPROCESS(args)
    if int(os.environ.get("RANK", "0")) != 0:
        return
    metrics_path = r4_paths(args.output_dir, args.seed)["metrics"]
    if not metrics_path.is_file():
        return
    results = _read_json(metrics_path)
    relation = _RELATION_SPEC or results.get("config", {}).get("relation")
    results.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "protocol": "R4 relation-only diagnostic: hard-label CE plus one pre-screened native R1/R2 relation term, no pointwise feature KD, fixed 4000-step warm-up, fixed 80k budget, dev_local selection, and no test_local evaluation.",
            "relation": copy.deepcopy(relation),
            "relation_spec_sha256": None if not isinstance(relation, Mapping) else _canonical_sha256(relation),
            "feature_kd_enabled": False,
            "selected_relation_gate": copy.deepcopy(_SELECTED_GATE),
            "r0_gate": copy.deepcopy(_R0_GATE or r1._R0_GATE),
            "test_local_evaluated": False,
            "r4_is_main_candidate": False,
        }
    )
    results["loss"] = {
        "hard_label_ce": True,
        "feature_kd": False,
        "logit_kd": False,
        "relation_kd": True,
        "relation_r1": _ACTIVE_RELATION == "r1",
        "relation_r2": _ACTIVE_RELATION == "r2",
        "lambda_feat": 0.0,
        "lambda_rel": float(args.lambda_rel),
        "warmup_steps": 4_000,
        "total": "CE + warmup * lambda_rel * selected_relation",
    }
    results["pca"] = {"enabled": False, "used_in_loss": False, "reason": "relation-only diagnostic"}
    results["hashes"] = {
        **dict(results.get("hashes", {})),
        "relation_spec_sha256": results["relation_spec_sha256"],
        "r4_training_script_sha256": common.sha256_file(Path(__file__).resolve()),
    }
    common.write_json_atomic(metrics_path, results)


def run_training(args: argparse.Namespace) -> None:
    global _ACTIVE_RELATION, _ACTIVE_LAMBDA, _RELATION_SPEC, _REFERENCE_TESTS, _R0_GATE, _SELECTED_GATE
    _ACTIVE_RELATION = str(args.relation).lower()
    _ACTIVE_LAMBDA = float(args.lambda_rel)
    _R0_GATE = r1._validate_r0_gate(args)
    _SELECTED_GATE = _validate_selected_gate(args)
    _RELATION_SPEC = None
    _REFERENCE_TESTS = None

    saved = {
        "EXPERIMENT": r1.EXPERIMENT,
        "ARTIFACT_TYPE": r1.ARTIFACT_TYPE,
        "r1_paths": r1.r1_paths,
        "_relation_spec": r1._relation_spec,
        "r1_relation_losses": r1.r1_relation_losses,
        "run_relation_reference_tests": r1.run_relation_reference_tests,
        "build_config_r1": r1.build_config_r1,
        "build_best_checkpoint_r1": r1.build_best_checkpoint_r1,
        "train_one_epoch_r1": r1.train_one_epoch_r1,
        "smoke_test_r1": r1.smoke_test_r1,
        "_postprocess_metrics_r1": r1._postprocess_metrics_r1,
        "audit_shapes_r1": r1.audit_shapes_r1,
        "_patched_torch_save_atomic_r1": r1._patched_torch_save_atomic_r1,
        "_patched_evaluate_r1": r1._patched_evaluate_r1,
        "_r1_print": r1._r1_print,
        "_r1_tqdm": r1._r1_tqdm,
        "_aggregate_gradient_record": r1._aggregate_gradient_record,
        "_update_relation_stop_gate": r1._update_relation_stop_gate,
        "_REFERENCE_TESTS": r1._REFERENCE_TESTS,
        "_R0_GATE": r1._R0_GATE,
        "_FIRST_BATCH_BASE_EQUIVALENCE": r1._FIRST_BATCH_BASE_EQUIVALENCE,
    }
    try:
        r1.EXPERIMENT = EXPERIMENT
        r1.ARTIFACT_TYPE = ARTIFACT_TYPE
        r1.r1_paths = r4_paths
        r1._relation_spec = _relation_spec_r4
        r1.r1_relation_losses = r4_relation_losses
        r1.run_relation_reference_tests = run_relation_reference_tests
        r1.build_config_r1 = build_config_r4
        r1.build_best_checkpoint_r1 = build_best_checkpoint_r4
        r1.train_one_epoch_r1 = train_one_epoch_r4
        r1.smoke_test_r1 = smoke_test_r4
        r1._postprocess_metrics_r1 = _postprocess_metrics_r4
        r1.audit_shapes_r1 = audit_shapes_r4
        r1._patched_torch_save_atomic_r1 = _patched_torch_save_atomic_r4
        r1._patched_evaluate_r1 = r1._patched_evaluate_r1
        if _ACTIVE_RELATION == "r2":
            r1._aggregate_gradient_record = r2._aggregate_gradient_record_r2
            r1._update_relation_stop_gate = r2._update_relation_stop_gate_r2
        r1._r1_print = lambda *v, **kw: print(
            *tuple(x.replace("K0", EXPERIMENT).replace("K1", EXPERIMENT) if isinstance(x, str) else x for x in v),
            **kw,
        )
        r1._r1_tqdm = r1._r1_tqdm
        r1.run_training(args)
    finally:
        for name, value in saved.items():
            setattr(r1, name, value)
        _RELATION_SPEC = None
        _REFERENCE_TESTS = None


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
