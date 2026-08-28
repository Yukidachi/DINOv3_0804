"""H2 server entry point: late OS=16 Hardswish placement ablation.

H2 is the main H-group position hypothesis.  It keeps the locked R5
MobileNetV2+R-ASPP protocol and replaces only the expansion and depthwise
activations in blocks 14..17 (the fixed OS=16 late stage) with Hardswish.
The stem, final projection, R-ASPP head and all linear bottlenecks remain
ReLU6/no-activation exactly as in H0.

The runner is deliberately an adapter around :mod:`dino_r5_server`; this
keeps DDP, relation/logit losses, checkpointing, resume and metric code
identical to the other H experiments.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

import dino as common
import dino_h0_server as h0
import dino_k3_server as k3
import dino_r2_server as r2
import dino_r5_server as r5
import dino_s2_0 as base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "H_MobileNetV2_RASPP_server"
K_GROUP_OUTPUT_DIR = SCRIPT_DIR / "result" / "K_MobileNetV2_RASPP_server"
R_GROUP_OUTPUT_DIR = SCRIPT_DIR / "result" / "R_MobileNetV2_RASPP_server"

EXPERIMENT = "H2"
EXPERIMENT_GROUP = "H_MobileNetV2_RASPP_server"
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_h2_hardswish_blocks_14_17_r5"
ARTIFACT_FORMAT_VERSION = 1

FORMAL_SEEDS = (42, 3407, 260805)
SCREENING_SEED = 42
LAMBDA_FEAT = 1.0
LAMBDA_R2 = 0.3
LAMBDA_LOGIT = 0.5
TEMPERATURE = 4.0
AUXILIARY_WARMUP_RATIO = 0.05
GRADIENT_LOG_STEPS = 500

# These references must remain independent of any temporary hooks installed by
# the R5 runner.
_ORIGINAL_K3_BUILD_MODEL = k3.build_k3_model
_ORIGINAL_R5_BUILD_CONFIG = h0._ORIGINAL_R5_BUILD_CONFIG
_ORIGINAL_R5_AUDIT_SHAPES = h0._ORIGINAL_R5_AUDIT_SHAPES
_ORIGINAL_R5_BUILD_BEST_CHECKPOINT = h0._ORIGINAL_R5_BUILD_BEST_CHECKPOINT
_ORIGINAL_R5_TRAIN_ONE_EPOCH = h0._ORIGINAL_R5_TRAIN_ONE_EPOCH


def h2_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    """Return standard artifact paths under ``H2/seed_<seed>``."""

    original = k3._ORIGINAL_K0_PATHS(output_dir, seed)
    run_dir = output_dir.resolve() / EXPERIMENT / f"seed_{seed}"
    return {
        key: run_dir if key == "run_dir" else run_dir / value.name
        for key, value in original.items()
    } | {"activation_replacement": run_dir / "activation_replacement.json"}


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return common.sha256_bytes(payload)


def _module_type(module: nn.Module) -> str:
    return f"{module.__class__.__module__}.{module.__class__.__qualname__}"


def _candidate_sites(block_index: int):
    # MobileNetV2 block 1 has expansion ratio 1 and only exposes conv.0.2;
    # H2 starts at block 14, where both paths are present.
    return (("depthwise", 0),) if block_index == 1 else (
        ("expansion", 0),
        ("depthwise", 1),
    )


def _h2_activation_spec() -> Dict[str, object]:
    eligible = []
    replaced = []
    for block_index in range(1, 18):
        late = 14 <= block_index <= 17
        for path_type, sub_index in _candidate_sites(block_index):
            path = f"backbone.{block_index}.conv.{sub_index}.2"
            row = {
                "path": path,
                "block_index": block_index,
                "path_type": path_type,
                "exists_when_built": True,
                "original_activation": "ReLU6",
                "activation": "Hardswish" if late else "ReLU6",
                "replaced": late,
                "original_module_type": "torch.nn.modules.activation.ReLU6",
                "replacement_module_type": (
                    "torch.nn.modules.activation.Hardswish"
                    if late
                    else "torch.nn.modules.activation.ReLU6"
                ),
            }
            eligible.append(row)
            if late:
                replaced.append(path)
    return {
        "name": "Hardswish",
        "implementation": "torch.nn.Hardswish(inplace=True)",
        "formula": "h-swish(x)=x*ReLU6(x+3)/6",
        "placement_scope": (
            "inverted-residual expansion/depthwise activations in "
            "MobileNetV2 blocks 14..17 (OS=16 late stage)"
        ),
        "eligible_blocks": list(range(1, 18)),
        "replacement_blocks": list(range(14, 18)),
        "eligible_paths": eligible,
        "replaced_module_paths": replaced,
        "replacement_count": len(replaced),
        "replacement_list_sha256": _canonical_hash(replaced),
        "stem_path": "backbone.0.2",
        "final_path": "backbone.18.2",
        "head_path": "head.project.2",
        "linear_bottleneck_activation": False,
    }


def _fixed_activation_types(student: nn.Module) -> Dict[str, str]:
    backbone = student.backbone
    fixed = {
        "stem": backbone[0][2],
        "final": backbone[18][2],
        "raspp_head": student.head.project[2],
    }
    types = {name: _module_type(module) for name, module in fixed.items()}
    if not all(isinstance(module, nn.ReLU6) for module in fixed.values()):
        raise RuntimeError(f"H2 fixed activation contract failed: {types}")
    return types


def _check_linear_bottleneck(block: nn.Module, block_index: int) -> None:
    conv = getattr(block, "conv", None)
    projection = conv[2] if conv is not None and len(conv) > 2 else None
    if projection is None:
        raise RuntimeError(f"H2 block {block_index} has no linear bottleneck projection")
    if any(
        isinstance(module, (nn.ReLU6, nn.Hardswish, nn.SiLU, nn.GELU))
        for module in projection.modules()
    ):
        raise RuntimeError(
            f"H2 block {block_index} linear bottleneck contains an activation"
        )


def _replace_h2_activations(model: nn.Module) -> Dict[str, object]:
    """Replace exactly blocks 14..17 and verify all fixed boundaries."""

    student = model.module if hasattr(model, "module") else model
    if not hasattr(student, "backbone") or not hasattr(student, "head"):
        raise RuntimeError("H2 activation replacement received an unexpected student")
    backbone = student.backbone
    if len(backbone) != 19:
        raise RuntimeError(f"H2 expected 19 MobileNetV2 modules, got {len(backbone)}")

    replaced = []
    for block_index in range(1, 18):
        block = backbone[block_index]
        conv = getattr(block, "conv", None)
        if conv is None:
            raise RuntimeError(f"H2 block {block_index} has no conv sequential")
        for path_type, sub_index in _candidate_sites(block_index):
            path = f"backbone.{block_index}.conv.{sub_index}.2"
            try:
                previous = conv[sub_index][2]
            except (IndexError, KeyError, TypeError) as exc:
                raise RuntimeError(f"H2 missing eligible activation at {path}") from exc
            if not isinstance(previous, nn.ReLU6):
                raise RuntimeError(
                    f"H2 expected ReLU6 before replacement at {path}, "
                    f"got {_module_type(previous)}"
                )
            if 14 <= block_index <= 17:
                conv[sub_index][2] = nn.Hardswish(inplace=True)
                replaced.append(
                    {
                        "path": path,
                        "block_index": block_index,
                        "path_type": path_type,
                        "original_module_type": _module_type(previous),
                        "replacement_module_type": _module_type(conv[sub_index][2]),
                    }
                )
        _check_linear_bottleneck(block, block_index)

    fixed_types = _fixed_activation_types(student)
    expected = 8
    if len(replaced) != expected:
        raise RuntimeError(f"H2 expected {expected} replacements, observed {len(replaced)}")
    return {
        "experiment": EXPERIMENT,
        "activation_name": "Hardswish",
        "activation_formula": "x*min(max(x+3, 0), 6)/6",
        "eligible_module_count": 33,
        "eligible_module_paths": replaced,
        "fixed_paths": {
            "backbone.0.2": fixed_types["stem"],
            "backbone.18.2": fixed_types["final"],
            "head.project.2": fixed_types["raspp_head"],
        },
        "linear_bottleneck_activation": False,
        "replacement_count": len(replaced),
    }


def _activation_audit_h2(model: nn.Module) -> Dict[str, object]:
    """Numerically and structurally verify the Hardswish contract."""

    probe = torch.tensor([-12.0, -6.0, -3.0, -2.5, 0.0, 2.5, 3.0, 6.0, 12.0])
    reference = probe * torch.clamp(probe + 3.0, min=0.0, max=6.0) / 6.0
    observed = nn.Hardswish()(probe)
    reference_error = float((observed - reference).abs().max().item())
    if reference_error > 1e-7 or not bool(torch.isfinite(observed).all().item()):
        raise RuntimeError(
            f"H2 Hardswish reference test failed: max_abs_error={reference_error}"
        )
    audit = _replace_h2_activations_audit_only(model)
    audit["reference_test"] = {
        "max_abs_error": reference_error,
        "tolerance": 1e-7,
        "passed": True,
        "probe_values": probe.tolist(),
    }
    audit["activation_spec_sha256"] = _canonical_hash(_h2_activation_spec())
    audit["activation_audit_sha256"] = _canonical_hash(audit)
    return audit


def _replace_h2_activations_audit_only(model: nn.Module) -> Dict[str, object]:
    """Audit an already-built H2 model without mutating it."""

    student = model.module if hasattr(model, "module") else model
    if not hasattr(student, "backbone") or not hasattr(student, "head"):
        raise RuntimeError("H2 activation audit received an unexpected student")
    backbone = student.backbone
    if len(backbone) != 19:
        raise RuntimeError(f"H2 expected 19 MobileNetV2 modules, got {len(backbone)}")

    actual = []
    for block_index in range(1, 18):
        block = backbone[block_index]
        conv = getattr(block, "conv", None)
        if conv is None:
            raise RuntimeError(f"H2 block {block_index} has no conv sequential")
        for path_type, sub_index in _candidate_sites(block_index):
            path = f"backbone.{block_index}.conv.{sub_index}.2"
            try:
                activation = conv[sub_index][2]
            except (IndexError, KeyError, TypeError) as exc:
                raise RuntimeError(f"H2 missing activation at {path}") from exc
            expected_type = nn.Hardswish if 14 <= block_index <= 17 else nn.ReLU6
            if not isinstance(activation, expected_type):
                raise RuntimeError(
                    f"H2 activation mismatch at {path}: {_module_type(activation)}; "
                    f"expected {expected_type.__name__}"
                )
            actual.append(
                {
                    "path": path,
                    "block_index": block_index,
                    "path_type": path_type,
                    "module_type": _module_type(activation),
                    "replaced": 14 <= block_index <= 17,
                }
            )
        _check_linear_bottleneck(block, block_index)

    fixed_types = _fixed_activation_types(student)
    if len(actual) != 33:
        raise RuntimeError(f"H2 expected 33 eligible modules, observed {len(actual)}")
    return {
        "experiment": EXPERIMENT,
        "activation_name": "Hardswish",
        "activation_formula": "x*min(max(x+3, 0), 6)/6",
        "eligible_module_count": len(actual),
        "eligible_module_paths": actual,
        "fixed_paths": {
            "backbone.0.2": fixed_types["stem"],
            "backbone.18.2": fixed_types["final"],
            "head.project.2": fixed_types["raspp_head"],
        },
        "linear_bottleneck_activation": False,
        "replacement_count": sum(row["replaced"] for row in actual),
    }


def build_h2_model(head_channels: int, dropout: float):
    """Build the K3 tapped student and apply the H2 replacements."""

    model = _ORIGINAL_K3_BUILD_MODEL(head_channels, dropout)
    _replace_h2_activations(model)
    return model


def parse_args() -> argparse.Namespace:
    """Reuse H0's locked CLI while assigning the H2 identity/output."""

    saved_default = h0.DEFAULT_OUTPUT_DIR
    h0.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    try:
        args = h0.parse_args()
    finally:
        h0.DEFAULT_OUTPUT_DIR = saved_default

    if args.seed not in FORMAL_SEEDS:
        raise SystemExit(f"H2 seed must be one of {FORMAL_SEEDS}")
    forbidden = {
        K_GROUP_OUTPUT_DIR.resolve(),
        R_GROUP_OUTPUT_DIR.resolve(),
        (SCRIPT_DIR / "result" / "H0_MobileNetV2_RASPP_server").resolve(),
        (SCRIPT_DIR / "result" / "H1_MobileNetV2_RASPP_server").resolve(),
    }
    if args.output_dir.resolve() in forbidden:
        raise SystemExit("H2 output must use the separate H-group output directory")
    if not args.smoke_test:
        if args.max_steps != 80_000:
            raise SystemExit("Formal H2 is locked to exactly 80,000 optimizer steps")
        if args.eval_every_steps != 5_000:
            raise SystemExit("Formal H2 is locked to --eval-every-steps 5000")
        if args.gradient_log_steps != GRADIENT_LOG_STEPS:
            raise SystemExit("Formal H2 is locked to --gradient-log-steps 500")
    return args


def _read_json(path: Path) -> Dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _reference_rows(path: Path) -> Optional[Dict[int, Dict[str, object]]]:
    if not path.is_file():
        return None
    value = _read_json(path).get("per_rank")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    rows: Dict[int, Dict[str, object]] = {}
    for row in value:
        if isinstance(row, Mapping):
            rows[int(row.get("rank", -1))] = dict(row)
    return rows or None


def _validate_launch_gate_h2(args: argparse.Namespace) -> Dict[str, object]:
    """Require K3/R2 loader identity references, without an R5 expansion gate."""

    if args.smoke_test:
        r5._REFERENCE_FIRST_BATCH = {}
        return {"status": "skipped_for_smoke_test", "experiment": EXPERIMENT}
    k3_first = r5._ORIGINAL_K3_PATHS(K_GROUP_OUTPUT_DIR, args.seed)["first_batch_audit"]
    r2_first = r2.r2_paths(R_GROUP_OUTPUT_DIR, args.seed, LAMBDA_R2)["first_batch_audit"]
    k3_rows = _reference_rows(k3_first)
    r2_rows = _reference_rows(r2_first)
    if k3_rows is None or r2_rows is None:
        raise FileNotFoundError(
            "H2 requires matched K3 and R2 first-batch audits for the same seed: "
            f"K3={k3_first}, R2={r2_first}"
        )
    r5._REFERENCE_FIRST_BATCH = {"K3": k3_rows, "R2": r2_rows}
    return {
        "status": "matched_k3_r2_first_batch_loaded",
        "experiment": EXPERIMENT,
        "comparison_seed": args.seed,
        "K3_first_batch_audit": str(k3_first.resolve()),
        "R2_first_batch_audit": str(r2_first.resolve()),
        "r5_expansion_gate": "not used by H2",
    }


def _compare_first_batch_h2(
    audit: Mapping[str, object], rank: int
) -> Dict[str, object]:
    if not r5._REFERENCE_FIRST_BATCH:
        return {
            "passed": True,
            "status": "skipped_for_smoke_test",
            "comparison": "no reference rows loaded",
        }
    exact_fields = ("paths", "image_tensor_sha256", "target_tensor_sha256")
    mismatches = []
    references_checked = []
    for experiment in ("K3", "R2"):
        rows = r5._REFERENCE_FIRST_BATCH.get(experiment)
        if not isinstance(rows, Mapping) or rank not in rows:
            mismatches.append({"reference": experiment, "field": "rank", "rank": rank})
            continue
        reference = rows[rank]
        references_checked.append(experiment)
        for field in exact_fields:
            if audit.get(field) != reference.get(field):
                mismatches.append(
                    {
                        "reference": experiment,
                        "field": field,
                        "actual": audit.get(field),
                        "expected": reference.get(field),
                    }
                )
    return {
        "passed": not mismatches,
        "comparison": (
            "H2 loader identity only: paths/image/target hashes versus K3/R2; "
            "student-dependent losses are intentionally not compared"
        ),
        "references_checked": references_checked,
        "student_loss_scalar_comparison": "skipped_by_design_for_activation_ablation",
        "mismatches": mismatches,
    }


def _resource_hashes_h2() -> Dict[str, object]:
    return {
        **r5.k3._resource_hashes(),
        "r2_relation_source_sha256": common.sha256_file(Path(r2.__file__).resolve()),
        "h2_training_script_sha256": common.sha256_file(Path(__file__).resolve()),
        "relation_spec_sha256": (
            None if r5._RELATION_SPEC is None else r2._canonical_sha256(r5._RELATION_SPEC)
        ),
        "activation_spec_sha256": _canonical_hash(_h2_activation_spec()),
    }


def _base_h2_config(*args: Any, **kwargs: Any) -> Dict[str, object]:
    config = _ORIGINAL_R5_BUILD_CONFIG(*args, **kwargs)
    activation = _h2_activation_spec()
    config.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "server_entry_point": str(Path(__file__).resolve()),
            "formal_seeds": list(FORMAL_SEEDS),
            "screening_seed": SCREENING_SEED,
            "activation": activation,
            "activation_name": "Hardswish",
            "activation_formula": "x*ReLU6(x+3)/6",
            "activation_spec_sha256": _canonical_hash(activation),
            "activation_replacement": {
                "name": "Hardswish",
                "replacement_count": 8,
                "replacement_paths": activation["replaced_module_paths"],
                "scope": activation["placement_scope"],
            },
            "test_local_evaluated": False,
        }
    )
    config["loss"] = {
        **dict(config.get("loss", {})),
        "hard_label_ce": True,
        "feature_kd": True,
        "relation_kd": True,
        "relation_r1": False,
        "relation_r2": True,
        "logit_kd": True,
        "lambda_feat": LAMBDA_FEAT,
        "lambda_r2": LAMBDA_R2,
        "lambda_logit": LAMBDA_LOGIT,
        "temperature": TEMPERATURE,
        "warmup_steps": 4_000,
        "warmup_ratio": AUXILIARY_WARMUP_RATIO,
        "total": "CE + warmup * (1.0*feature + 0.3*R2 + 0.5*logit_KL_T2)",
    }
    return config


def build_config_h2(*args: Any, **kwargs: Any) -> Dict[str, object]:
    return _base_h2_config(*args, **kwargs)


def audit_shapes_h2(model: base.MobileNetV2RASPPStudent, *args: Any, **kwargs: Any):
    audit = _ORIGINAL_R5_AUDIT_SHAPES(model, *args, **kwargs)
    activation = _activation_audit_h2(model)
    audit.update(
        {
            "experiment": EXPERIMENT,
            "activation": activation,
            "activation_spec_sha256": activation["activation_spec_sha256"],
            "r5_compatible_loss": True,
        }
    )
    return audit


def build_best_checkpoint_h2(*args: Any, **kwargs: Any) -> Dict[str, object]:
    payload = _ORIGINAL_R5_BUILD_BEST_CHECKPOINT(*args, **kwargs)
    activation = _h2_activation_spec()
    payload.update(
        {
            "artifact_type": ARTIFACT_TYPE,
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "activation": activation,
            "activation_name": "Hardswish",
            "activation_formula": "x*ReLU6(x+3)/6",
            "activation_spec_sha256": _canonical_hash(activation),
            "initialization": "K-group shared scratch initialization; weights=None",
        }
    )
    payload["hashes"] = {**dict(payload.get("hashes", {})), **_resource_hashes_h2()}
    return payload


def train_one_epoch_h2(*args: Any, **kwargs: Any):
    metrics, steps, gradients, first_batch = _ORIGINAL_R5_TRAIN_ONE_EPOCH(*args, **kwargs)
    if isinstance(metrics, Mapping):
        metrics = dict(metrics)
        metrics["experiment"] = EXPERIMENT
    adjusted_gradients = []
    for record in gradients:
        adjusted = dict(record)
        adjusted["experiment"] = EXPERIMENT
        adjusted["activation_spec_sha256"] = _canonical_hash(_h2_activation_spec())
        adjusted_gradients.append(adjusted)
    if isinstance(first_batch, Mapping):
        first_batch = dict(first_batch)
        first_batch["experiment"] = EXPERIMENT
        first_batch["activation_spec_sha256"] = _canonical_hash(_h2_activation_spec())
    return metrics, steps, adjusted_gradients, first_batch


def _patched_torch_save_atomic_h2(payload: object, path: Path) -> None:
    if isinstance(payload, Mapping) and payload.get("artifact_type") == ARTIFACT_TYPE:
        payload = dict(payload)
        activation = _h2_activation_spec()
        payload.update(
            {
                "experiment": EXPERIMENT,
                "experiment_group": EXPERIMENT_GROUP,
                "activation": activation,
                "activation_spec_sha256": _canonical_hash(activation),
                "hashes": {**dict(payload.get("hashes", {})), **_resource_hashes_h2()},
            }
        )
    r5._ORIGINAL_TORCH_SAVE_ATOMIC(payload, path)


def _patched_evaluate_h2(*args: Any, **kwargs: Any):
    split_name = kwargs.get("split_name")
    if isinstance(split_name, str):
        kwargs["split_name"] = (
            split_name.replace("K0", EXPERIMENT)
            .replace("K3", EXPERIMENT)
            .replace("R5", EXPERIMENT)
        )
    return r5._ORIGINAL_EVALUATE(*args, **kwargs)


def _h2_print(*values: object, **kwargs: object) -> None:
    adjusted = tuple(
        value.replace("R5", EXPERIMENT)
        .replace("K3", EXPERIMENT)
        .replace("K0", EXPERIMENT)
        if isinstance(value, str)
        else value
        for value in values
    )
    print(*adjusted, **kwargs)


def _postprocess_metrics_h2(args: argparse.Namespace) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    r5._ORIGINAL_K3_POSTPROCESS(args)
    metrics_path = h2_paths(args.output_dir, args.seed)["metrics"]
    if not metrics_path.is_file():
        return
    results = _read_json(metrics_path)
    activation = _h2_activation_spec()
    results.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "protocol": (
                "H2 late-stage Hardswish ablation: K-group shared scratch "
                "MobileNetV2+R-ASPP, hard-label CE plus locked A0 feature MSE, "
                "native masked 8x16 R2 relation MSE, and frozen T1 full-resolution "
                "masked pixel KL (T=4); weights 1.0/0.3/0.5, shared 4000-step "
                "warm-up, 80k optimizer steps, dev_local selection, and no "
                "test_local evaluation. Only blocks 14..17 expansion/depthwise "
                "activations are Hardswish."
            ),
            "activation": activation,
            "activation_name": "Hardswish",
            "activation_formula": "x*ReLU6(x+3)/6",
            "activation_spec_sha256": _canonical_hash(activation),
            "activation_replacement": {
                "name": "Hardswish",
                "replacement_count": 8,
                "replacement_paths": activation["replaced_module_paths"],
                "scope": activation["placement_scope"],
            },
            "loss": {
                "hard_label_ce": True,
                "feature_kd": True,
                "feature_mechanism": "A0 fixed StandardScaler+PCA",
                "relation_kd": True,
                "relation_r1": False,
                "relation_r2": True,
                "lambda_feat": LAMBDA_FEAT,
                "lambda_r2": LAMBDA_R2,
                "relation_pool_size": list(r2.POOL_SIZE),
                "logit_kd": True,
                "logit_mechanism": "full-resolution masked pixel KL",
                "lambda_logit": LAMBDA_LOGIT,
                "temperature": TEMPERATURE,
                "temperature_squared_factor": True,
                "warmup_steps": 4_000,
                "warmup_ratio": AUXILIARY_WARMUP_RATIO,
                "shared_auxiliary_warmup": True,
                "total": "CE + warmup * (1.0*feature + 0.3*R2 + 0.5*logit_KL_T2)",
            },
            "launch_gate": copy.deepcopy(r5._REFERENCE_GATE),
            "test_local_evaluated": False,
            "hashes": {**dict(results.get("hashes", {})), **_resource_hashes_h2()},
        }
    )
    common.write_json_atomic(
        h2_paths(args.output_dir, args.seed)["activation_replacement"],
        {
            **activation,
            "experiment": EXPERIMENT,
            "artifact_type": "activation_replacement_audit",
            "activation_spec_sha256": _canonical_hash(activation),
        },
    )
    efficiency_path = h2_paths(args.output_dir, args.seed)["efficiency"]
    if efficiency_path.is_file():
        efficiency = _read_json(efficiency_path)
        efficiency["activation"] = {
            "name": "Hardswish",
            "replacement_count": 8,
            "operator_fusion_candidate": "Hardswish in inverted residual blocks 14..17",
            "activation_spec_sha256": _canonical_hash(activation),
        }
        common.write_json_atomic(efficiency_path, efficiency)
    common.write_json_atomic(metrics_path, results)


def run_training(args: argparse.Namespace) -> None:
    """Install H2 metadata/hooks and execute the shared R5 runner."""

    names = (
        "DEFAULT_OUTPUT_DIR",
        "EXPERIMENT",
        "EXPERIMENT_GROUP",
        "ARTIFACT_TYPE",
        "ARTIFACT_FORMAT_VERSION",
        "r5_paths",
        "_validate_launch_gate",
        "_resource_hashes",
        "build_config_r5",
        "audit_shapes_r5",
        "build_best_checkpoint_r5",
        "train_one_epoch_r5",
        "_patched_torch_save_atomic_r5",
        "_patched_evaluate_r5",
        "_r5_print",
        "_postprocess_metrics_r5",
        "print",
    )
    had_print = hasattr(r5, "print")
    saved_r5 = {name: getattr(r5, name, None) for name in names}
    saved_compare = r5._compare_first_batch
    saved_builder = k3.build_k3_model
    try:
        r5.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
        r5.EXPERIMENT = EXPERIMENT
        r5.EXPERIMENT_GROUP = EXPERIMENT_GROUP
        r5.ARTIFACT_TYPE = ARTIFACT_TYPE
        r5.ARTIFACT_FORMAT_VERSION = ARTIFACT_FORMAT_VERSION
        r5.r5_paths = h2_paths
        r5._validate_launch_gate = _validate_launch_gate_h2
        r5._resource_hashes = _resource_hashes_h2
        r5.build_config_r5 = build_config_h2
        r5.audit_shapes_r5 = audit_shapes_h2
        r5.build_best_checkpoint_r5 = build_best_checkpoint_h2
        r5.train_one_epoch_r5 = train_one_epoch_h2
        r5._patched_torch_save_atomic_r5 = _patched_torch_save_atomic_h2
        r5._patched_evaluate_r5 = _patched_evaluate_h2
        r5._r5_print = _h2_print
        r5._postprocess_metrics_r5 = _postprocess_metrics_h2
        r5.print = _h2_print
        r5._compare_first_batch = _compare_first_batch_h2
        k3.build_k3_model = build_h2_model
        r5.run_training(args)
    finally:
        k3.build_k3_model = saved_builder
        r5._compare_first_batch = saved_compare
        for name, value in saved_r5.items():
            if name == "print" and not had_print:
                if hasattr(r5, "print"):
                    delattr(r5, "print")
            else:
                setattr(r5, name, value)


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
