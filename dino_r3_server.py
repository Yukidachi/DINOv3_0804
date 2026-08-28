"""R3 server entry point: combined cross-image and within-image relation KD.

R3 keeps the locked K1/R0 training contract and combines the two registered
relation objectives without changing either reduction:

    L = L_seg + warmup * (lambda_feat * L_feat
                          + lambda_r1 * L_R1 + lambda_r2 * L_R2)

R3 also supports an independent diagnostic mode: R0/R1/R2 result artifacts
are optional paired references, while the locked K1 protocol/resources and
the selected seed's K shared initialization remain required.  The independent
mode deliberately does not perform relation-gradient review; it still checks
loss finiteness and teacher/projection freezing.

The entry point is intentionally separate from the R1/R2 scripts so its
output, hashes and loss schema cannot overwrite a single-relation run.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import dino as common
import dino_a0_server as a0
import dino_k0_server as k0
import dino_k1_server as k1
import dino_r0_server as r0
import dino_r1_server as r1
import dino_r2_server as r2
import dino_s2_0 as base
import dino_s2_0_server as server_base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "R_MobileNetV2_RASPP_server"
K_GROUP_OUTPUT_DIR = SCRIPT_DIR / "result" / "K_MobileNetV2_RASPP_server"
EXPERIMENT = "R3"
EXPERIMENT_GROUP = "R_MobileNetV2_RASPP_server"
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_r3_combined_relation_kd"
ARTIFACT_FORMAT_VERSION = 1
# All pre-registered seeds may be run independently for R3 diagnostics.
FORMAL_SEEDS = (42, 3407, 260805)

RELATION_EPSILON = 1e-6
LAMBDA_R1 = 0.03
LAMBDA_R2 = 0.03
ALLOWED_LAMBDA_R1 = (0.015, 0.03, 0.06)
# 0.3 is the accepted extended R2 calibration.  It is kept explicit rather
# than accepting arbitrary values so a formal R3 cannot silently bypass the
# screened single-relation candidate.
ALLOWED_LAMBDA_R2 = (0.015, 0.03, 0.06, 0.3)
GRADIENT_GATE_MIN = 0.05
GRADIENT_GATE_MAX = 0.20
GRADIENT_CE_STOP_RATIO = 2.0
GRADIENT_CE_STOP_CONSECUTIVE = 3
FIXED_GRADIENT_AUDIT_STEPS = (1, 4_000, 20_000, 40_000, 60_000, 80_000)
GRADIENT_REVIEW_ENABLED = False
POOL_SIZE = r2.POOL_SIZE
NUM_TOKENS = r2.NUM_TOKENS

_ORIGINAL_K_SHARED_INITIALIZATION = k1._ORIGINAL_ENSURE_SHARED_INITIALIZATION
_ORIGINAL_TQDM = r1._ORIGINAL_TQDM

_R0_GATE: Optional[Dict[str, object]] = None
_R1_GATE: Optional[Dict[str, object]] = None
_R2_GATE: Optional[Dict[str, object]] = None
_RELATION_SPEC: Optional[Dict[str, object]] = None
_REFERENCE_TESTS: Optional[Dict[str, object]] = None
_FIRST_BATCH_BASE_EQUIVALENCE: Optional[Dict[str, object]] = None
_RELATION_GATE_CONSECUTIVE_EXCESS = 0
_ACTIVE_LAMBDA_R1 = LAMBDA_R1
_ACTIVE_LAMBDA_R2 = LAMBDA_R2


def parse_args() -> argparse.Namespace:
    """Reuse K1's CLI and add the two pre-registered relation weights."""

    saved_default = k1.DEFAULT_OUTPUT_DIR
    saved_argparse = k1.argparse

    class R3ArgparseProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(saved_argparse, name)

        @staticmethod
        def ArgumentParser(*parser_args: Any, **parser_kwargs: Any):
            parser_kwargs["description"] = (
                "R3 MobileNetV2+R-ASPP: locked A0 feature KD plus combined "
                "cross-image R1 and within-image R2 relation KD."
            )
            parser = saved_argparse.ArgumentParser(*parser_args, **parser_kwargs)
            parser.add_argument("--lambda-r1", type=float, default=LAMBDA_R1)
            parser.add_argument("--lambda-r2", type=float, default=LAMBDA_R2)
            return parser

    k1.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    k1.argparse = R3ArgparseProxy()
    try:
        args = k1.parse_args()
    finally:
        k1.DEFAULT_OUTPUT_DIR = saved_default
        k1.argparse = saved_argparse

    if not any(
        value == "--accumulation-steps"
        or value.startswith("--accumulation-steps=")
        for value in sys.argv[1:]
    ):
        args.accumulation_steps = 2
    if args.seed not in FORMAL_SEEDS:
        raise SystemExit(f"R3 seed must be one of {FORMAL_SEEDS}")
    if not any(math.isclose(args.lambda_r1, value, abs_tol=1e-12) for value in ALLOWED_LAMBDA_R1):
        raise SystemExit("--lambda-r1 must be one of 0.015, 0.03, 0.06")
    if not any(math.isclose(args.lambda_r2, value, abs_tol=1e-12) for value in ALLOWED_LAMBDA_R2):
        raise SystemExit("--lambda-r2 must be one of 0.015, 0.03, 0.06, 0.3")
    if not args.smoke_test:
        if args.max_steps != 80_000:
            raise SystemExit("Formal R3 is locked to exactly 80,000 optimizer steps")
        if args.eval_every_steps != 5_000:
            raise SystemExit("Formal R3 is locked to --eval-every-steps 5000")
        # ``gradient_log_steps`` is retained for CLI compatibility but is
        # ignored because independent R3 disables gradient review.
    if args.output_dir.resolve() == K_GROUP_OUTPUT_DIR.resolve():
        raise SystemExit(
            "R3 output must not point at the K-group directory; use the separate "
            "R_MobileNetV2_RASPP output root"
        )
    return args


def r3_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    original = k1._ORIGINAL_K0_PATHS(output_dir, seed)
    run_dir = output_dir.resolve() / EXPERIMENT / _r3_run_name(
        seed, _ACTIVE_LAMBDA_R1, _ACTIVE_LAMBDA_R2
    )
    return {key: run_dir if key == "run_dir" else run_dir / value.name for key, value in original.items()}


def _format_lambda(value: float) -> str:
    """Return the stable directory token used by the R1/R2 calibration runs."""

    return format(float(value), ".12g")


def _single_relation_run_name(name: str, seed: int, value: float) -> str:
    default = LAMBDA_R1 if name == "R1" else LAMBDA_R2
    if math.isclose(float(value), default, rel_tol=0.0, abs_tol=1e-12):
        return f"seed_{seed}"
    return f"seed_{seed}_lambda_{_format_lambda(value)}"


def _relation_metrics_path(
    name: str, args: argparse.Namespace, lambda_key: str
) -> Path:
    value = float(getattr(args, lambda_key))
    return (
        args.output_dir.resolve()
        / name
        / _single_relation_run_name(name, args.seed, value)
        / "metrics.json"
    )


def _r3_run_name(seed: int, lambda_r1: float, lambda_r2: float) -> str:
    """Keep the legacy default path while isolating every calibrated R3 run."""

    if math.isclose(lambda_r1, LAMBDA_R1, abs_tol=1e-12) and math.isclose(
        lambda_r2, LAMBDA_R2, abs_tol=1e-12
    ):
        return f"seed_{seed}"
    return (
        f"seed_{seed}_lambda_r1_{_format_lambda(lambda_r1)}"
        f"_lambda_r2_{_format_lambda(lambda_r2)}"
    )


def _read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _canonical_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _r0_metrics_path(args: argparse.Namespace) -> Path:
    return args.output_dir.resolve() / "R0" / f"seed_{args.seed}" / "metrics.json"


def _candidate_lambda(metrics: Mapping[str, object], key: str) -> Optional[float]:
    for container_name in ("loss", "config"):
        container = metrics.get(container_name)
        if not isinstance(container, Mapping):
            continue
        if container_name == "config":
            container = container.get("loss", {})
            if not isinstance(container, Mapping):
                continue
        value = container.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _effect_gate(metrics: Mapping[str, object], args: argparse.Namespace) -> bool:
    comparison = metrics.get("r1_vs_r0") or metrics.get("r2_vs_r0")
    if isinstance(comparison, Mapping):
        for key in ("delta_R1_minus_R0", "delta_R2_minus_R0"):
            value = comparison.get(key)
            if value is not None and float(value) > 0.00219:
                return True
    r0_path = _r0_metrics_path(args)
    if not r0_path.is_file():
        return False
    r0_metrics = _read_json(r0_path)
    baseline = r0_metrics.get("best_dev_metrics", {})
    candidate = metrics.get("best_dev_metrics", {})
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        return False
    for field, threshold in (("boundary_f1", 0.00613), ("small_object_mIoU", 0.00851)):
        if baseline.get(field) is not None and candidate.get(field) is not None:
            if float(candidate[field]) - float(baseline[field]) >= threshold:
                return True
    return False


def _validate_relation_gate(
    name: str, args: argparse.Namespace, lambda_key: str, script_path: Path
) -> Dict[str, object]:
    path = _relation_metrics_path(name, args, lambda_key)
    if not path.is_file():
        if args.smoke_test:
            return {
                "required_for_formal_run": True,
                "checked": False,
                "passed": None,
                "reason": f"{name} metrics are absent; protocol smoke is allowed",
                "metrics_path": str(path),
            }
        raise FileNotFoundError(f"Formal R3 requires an accepted {name} result: {path}")
    metrics = _read_json(path)
    failures: List[str] = []
    if metrics.get("experiment") != name:
        failures.append(f"the artifact is not an {name} result")
    if metrics.get("test_local_evaluated") is not False:
        failures.append(f"{name} evaluated test_local")
    reference = metrics.get("relation_reference_tests")
    if not isinstance(reference, Mapping) or not bool(reference.get("passed")):
        failures.append(f"{name} relation reference tests did not pass")
    first = metrics.get(f"{name.lower()}_first_batch_base_equivalence")
    if not isinstance(first, Mapping) or not bool(first.get("passed")):
        failures.append(f"{name}/K1 first-batch equivalence did not pass")
    gradient = metrics.get("gradient_gate")
    if not isinstance(gradient, Mapping) or not bool(gradient.get("passed_target_at_any_record")):
        failures.append(f"{name} did not pass the relation/feature gradient gate")
    training = metrics.get("training")
    if not isinstance(training, Mapping) or int(training.get("optimizer_steps", -1)) != 80_000:
        failures.append(f"{name} did not complete the locked 80,000 optimizer steps")
    config = metrics.get("config")
    if not isinstance(config, Mapping):
        failures.append(f"{name} metrics has no auditable config")
        config = {}
    if int(config.get("world_size", -1)) != 2:
        failures.append(f"{name} was not trained with world_size=2")
    if int(config.get("global_batch_size", -1)) != 8:
        failures.append(f"{name} was not trained with global batch size 8")
    if config.get("amp") is not True or config.get("deterministic") is not True:
        failures.append(f"{name} did not preserve AMP+deterministic protocol")
    if not _effect_gate(metrics, args):
        failures.append(f"{name} did not pass the pre-registered effect gate versus R0")
    selected = _candidate_lambda(metrics, lambda_key)
    requested = float(getattr(args, lambda_key))
    if selected is not None and not math.isclose(selected, requested, abs_tol=1e-12):
        failures.append(f"{name} selected {lambda_key}={selected}, requested {requested}")
    hashes = metrics.get("hashes", {})
    expected_hash_key = f"{name.lower()}_training_script_sha256"
    if isinstance(hashes, Mapping) and hashes.get(expected_hash_key) != common.sha256_file(script_path):
        failures.append(f"{name} artifact was produced by a different {script_path.name}")
    if not isinstance(hashes, Mapping) or hashes.get("dataset_combined_manifest_sha256") != k1.EXPECTED_COMBINED_MANIFEST_SHA256:
        failures.append(f"{name} dataset manifest hash differs from the locked protocol")
    if not isinstance(hashes, Mapping) or hashes.get("teacher_checkpoint_sha256") != k1.EXPECTED_TEACHER_CHECKPOINT_SHA256:
        failures.append(f"{name} teacher checkpoint hash differs from the locked T1 teacher")
    r0_path = _r0_metrics_path(args)
    if r0_path.is_file() and isinstance(hashes, Mapping):
        r0_hashes = _read_json(r0_path).get("hashes", {})
        if isinstance(r0_hashes, Mapping):
            for field in (
                "student_init_state_sha256",
                "student_init_file_sha256",
                "pca_parameter_record_sha256",
            ):
                if hashes.get(field) != r0_hashes.get(field):
                    failures.append(f"{name} {field} differs from the accepted R0 anchor")
    best_path = path.parent / "best_checkpoint.pth"
    sidecar_path = best_path.with_suffix(best_path.suffix + ".sha256")
    if not best_path.is_file() or not sidecar_path.is_file():
        failures.append(f"{name} best checkpoint or SHA-256 sidecar is missing")
    else:
        expected_checkpoint_hash = sidecar_path.read_text(encoding="utf-8").split()[0].lower()
        actual_checkpoint_hash = common.sha256_file(best_path).lower()
        if expected_checkpoint_hash != actual_checkpoint_hash:
            failures.append(f"{name} best checkpoint SHA-256 sidecar does not match")
    result = {
        "required_for_formal_run": True,
        "checked": True,
        "passed": not failures,
        "failures": failures,
        "metrics_path": str(path),
        "metrics_sha256": common.sha256_file(path),
        "candidate_run_directory": str(path.parent),
        "candidate_checkpoint": str(best_path),
        "selected_lambda": selected,
        "effect_gate_passed": _effect_gate(metrics, args),
    }
    if failures and not args.smoke_test:
        raise RuntimeError(f"R3 {name}-gate validation failed:\n- " + "\n- ".join(failures))
    return result


def _read_optional_relation_reference(
    name: str, args: argparse.Namespace, lambda_key: str
) -> Dict[str, object]:
    """Read an R1/R2 artifact for provenance without making it a gate."""

    path = _relation_metrics_path(name, args, lambda_key)
    reference: Dict[str, object] = {
        "required_for_formal_run": False,
        "checked": False,
        "passed": None,
        "available": False,
        "comparison_only": True,
        "reason": f"R3 independent mode; {name} result is optional",
        "metrics_path": str(path),
        "requested_lambda": float(getattr(args, lambda_key)),
    }
    if not path.is_file():
        reference["reason"] = (
            f"{name} metrics are absent; continuing because independent R3 does "
            f"not require {name}"
        )
        return reference
    try:
        metrics = _read_json(path)
    except Exception as exc:  # pragma: no cover - defensive metadata path
        reference.update(
            {
                "checked": True,
                "reason": f"{name} metrics are unreadable; ignored in independent R3",
                "read_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return reference
    best = metrics.get("best_dev_metrics")
    is_expected_artifact = metrics.get("experiment") == name
    reference.update(
        {
            "checked": True,
            "available": True,
            "experiment": metrics.get("experiment"),
            "is_expected_artifact": is_expected_artifact,
            "selected_lambda": _candidate_lambda(metrics, lambda_key),
            "best_dev_mIoU": (
                best.get("mIoU")
                if is_expected_artifact and isinstance(best, Mapping)
                else None
            ),
            "metrics_sha256": common.sha256_file(path),
        }
    )
    if not is_expected_artifact:
        reference["reason"] = (
            f"An artifact exists at the {name} path but is not labelled {name}; "
            "it is retained for provenance only"
        )
    return reference


def _read_optional_r0_reference(args: argparse.Namespace) -> Dict[str, object]:
    """Read R0 metadata without making the baseline a launch prerequisite."""

    path = _r0_metrics_path(args)
    reference: Dict[str, object] = {
        "required_for_formal_run": False,
        "checked": False,
        "passed": None,
        "available": False,
        "comparison_only": True,
        "reason": "R3 independent mode; R0 result is optional",
        "metrics_path": str(path),
    }
    if not path.is_file():
        reference["reason"] = (
            "R0 metrics are absent; continuing because independent R3 does not "
            "require an R0 baseline"
        )
        return reference
    try:
        metrics = _read_json(path)
    except Exception as exc:  # pragma: no cover - defensive metadata path
        reference.update(
            {
                "checked": True,
                "reason": "R0 metrics are unreadable; ignored in independent R3",
                "read_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return reference
    best = metrics.get("best_dev_metrics")
    is_r0_artifact = metrics.get("experiment") == "R0"
    reference.update(
        {
            "checked": True,
            "available": True,
            "experiment": metrics.get("experiment"),
            "is_r0_artifact": is_r0_artifact,
            "r0_best_dev_mIoU": (
                best.get("mIoU") if is_r0_artifact and isinstance(best, Mapping) else None
            ),
            "metrics_sha256": common.sha256_file(path),
        }
    )
    if not is_r0_artifact:
        reference["reason"] = (
            "An artifact exists at the R0 path but is not labelled R0; it is "
            "retained for provenance only"
        )
    return reference


def _relation_spec_r3(args: argparse.Namespace, accumulation_steps: int, world_size: int) -> Dict[str, object]:
    physical_batch = int(args.batch_size) * int(world_size)
    return {
        "enabled": True,
        "active_relation_types": ["R1_cross_image", "R2_within_image_spatial"],
        "relation_feature_source": {
            "teacher": "native OS=4/8/16 features",
            "student": "native OS=4/8/16 features",
            "a0_projected_features_used_for_relation": False,
        },
        "epsilon": RELATION_EPSILON,
        "pool_size": list(POOL_SIZE),
        "num_tokens": NUM_TOKENS,
        "nominal_physical_relation_batch_size": physical_batch,
        "physical_relation_batch_size": physical_batch,
        "effective_optimizer_batch_size": physical_batch * int(accumulation_steps),
        "accumulated_batches_used_for_relation": False,
        "tail_batch_policy": "use the actual synchronized micro-batch; never cache, pad, or combine samples across optimizer steps",
        "layer_aggregation": "equal mean over native OS=4/8/16 for each relation type",
        "matrix_dtype": "float32",
        "normalization": "row L2 / (norm + epsilon); no matrix Frobenius normalization",
        "r1": {
            "enabled": True,
            "representation": "masked GAP then BxB signed cosine matrix",
            "diagonal_policy": "keep",
            "reduction": "sum squared error divided by actual B^2",
            "lambda": float(args.lambda_r1),
        },
        "r2": {
            "enabled": True,
            "representation": "per-image 128x128 signed token cosine matrix",
            "diagonal_policy": "keep for valid tokens",
            "reduction": "valid-pair masked sum divided by valid-pair count",
            "mask_policy": "nearest-resized valid mask, adaptive-average valid fraction, valid fraction > 0",
            "lambda": float(args.lambda_r2),
        },
        "relation_warmup_steps": 4_000,
        "relation_warmup_shared_with_feature_kd": True,
        "relation_gradient_gate": {
            "enabled": GRADIENT_REVIEW_ENABLED,
            "review_required": False,
            "target_relation_to_feature_effective_ratio": [GRADIENT_GATE_MIN, GRADIENT_GATE_MAX],
            "stop_if_combined_relation_to_ce_exceeds": None,
            "consecutive_records_before_stop": None,
            "lambda_is_fixed_during_formal_training": True,
            "reason": "Independent R3 mode does not perform gradient review",
        },
    }


def r3_relation_losses(
    student_features: Mapping[str, torch.Tensor],
    teacher_features: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, object], Dict[str, object]]:
    """Compute raw R1/R2 losses independently, then return their audits."""

    r1_loss, r1_layers, r1_audit = r1.r1_relation_losses(
        student_features, teacher_features, targets, world_size
    )
    r2_loss, r2_layers, r2_audit = r2.r2_relation_losses(
        student_features, teacher_features, targets, world_size
    )
    if int(r1_audit["physical_batch_size"]) != int(r2_audit["physical_batch_size"]):
        raise RuntimeError("R3 R1/R2 relation batches disagree")
    return r1_loss, r2_loss, r1_layers, r2_layers, r1_audit, r2_audit


def run_r3_reference_tests(device: torch.device, world_size: int) -> Dict[str, object]:
    r1_tests = r1.run_relation_reference_tests(device, world_size)
    r2_tests = r2.run_r2_reference_tests(device, world_size)
    return {"passed": bool(r1_tests.get("passed") and r2_tests.get("passed")), "r1": r1_tests, "r2": r2_tests}


def _ensure_locked_k_shared_initialization(
    model: base.MobileNetV2RASPPStudent,
    args: argparse.Namespace,
    _output_dir: Path,
    seed: int,
    rank: int,
    world_size: int,
) -> Tuple[str, str, Path]:
    path = k0._shared_init_path(K_GROUP_OUTPUT_DIR, seed)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"R3 requires the existing K shared initialization: {path}")
    return _ORIGINAL_K_SHARED_INITIALIZATION(model, args, K_GROUP_OUTPUT_DIR, seed, rank, world_size)


def build_config_r3(
    args: argparse.Namespace,
    accumulation_steps: int,
    world_size: int,
    device: torch.device,
    shared_init_state_sha256: str,
    shared_init_file_sha256: str,
) -> Dict[str, object]:
    global _RELATION_SPEC, _REFERENCE_TESTS
    config = r1._ORIGINAL_K1_BUILD_CONFIG(
        args, accumulation_steps, world_size, device, shared_init_state_sha256, shared_init_file_sha256
    )
    _RELATION_SPEC = _relation_spec_r3(args, accumulation_steps, world_size)
    _REFERENCE_TESTS = run_r3_reference_tests(device, world_size)
    r1._RELATION_SPEC = _RELATION_SPEC
    r1._REFERENCE_TESTS = _REFERENCE_TESTS
    if not args.smoke_test:
        if world_size != 2 or int(_RELATION_SPEC["physical_relation_batch_size"]) != 4:
            raise RuntimeError("Formal R3 requires two ranks and physical relation batch size 4")
        if int(_RELATION_SPEC["effective_optimizer_batch_size"]) != 8:
            raise RuntimeError("Formal R3 requires effective optimizer batch size 8")
    config.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "server_entry_point": str(Path(__file__).resolve()),
            "formal_seeds": list(FORMAL_SEEDS),
            "shared_initialization": {
                "source_group": "K_MobileNetV2_RASPP_server",
                "path": str(k0._shared_init_path(K_GROUP_OUTPUT_DIR, args.seed).resolve()),
                "state_sha256": shared_init_state_sha256,
                "file_sha256": shared_init_file_sha256,
                "r_specific_initialization_created": False,
            },
            "relation": copy.deepcopy(_RELATION_SPEC),
            "relation_spec_sha256": _canonical_sha256(_RELATION_SPEC),
            "relation_reference_tests": copy.deepcopy(_REFERENCE_TESTS),
            "r0_gate": copy.deepcopy(_R0_GATE),
            "r1_gate": copy.deepcopy(_R1_GATE),
            "r2_gate": copy.deepcopy(_R2_GATE),
            "k1_reference_validation": copy.deepcopy(r1._K1_REFERENCE_VALIDATION),
        }
    )
    loss = dict(config.get("loss", {}))
    loss.update(
        {
            "relation_kd": True,
            "relation_r1": True,
            "relation_r2": True,
            "lambda_r1": float(args.lambda_r1),
            "lambda_r2": float(args.lambda_r2),
            "total": "CE + warmup * (lambda_feat * feature + lambda_r1 * R1 + lambda_r2 * R2)",
        }
    )
    config["loss"] = loss
    return config


def audit_shapes_r3(*args: Any, **kwargs: Any) -> Dict[str, object]:
    audit = r1._ORIGINAL_K1_AUDIT_SHAPES(*args, **kwargs)
    audit["experiment"] = EXPERIMENT
    audit["relation"] = {
        "enabled": True,
        "types": ["R1_cross_image", "R2_within_image_spatial"],
        "native_teacher_student_taps": list(a0.A0_LAYER_ORDER),
        "r1_matrix_shape": ["physical_batch", "physical_batch"],
        "r2_matrix_shape": [NUM_TOKENS, NUM_TOKENS],
        "r2_pool_size": list(POOL_SIZE),
        "matrix_dtype": "float32",
    }
    return audit


def build_best_checkpoint_r3(*args: Any, **kwargs: Any) -> Dict[str, object]:
    payload = r1._ORIGINAL_K1_BUILD_BEST_CHECKPOINT(*args, **kwargs)
    payload.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "relation": copy.deepcopy(_RELATION_SPEC),
            "relation_spec_sha256": None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC),
            "r0_gate": copy.deepcopy(_R0_GATE),
            "r1_gate": copy.deepcopy(_R1_GATE),
            "r2_gate": copy.deepcopy(_R2_GATE),
        }
    )
    return payload


def _patched_torch_save_atomic_r3(payload: object, path: Path) -> None:
    if isinstance(payload, Mapping) and payload.get("artifact_type") == ARTIFACT_TYPE:
        payload = dict(payload)
        payload.update(
            {
                "experiment": EXPERIMENT,
                "experiment_group": EXPERIMENT_GROUP,
                "relation": copy.deepcopy(_RELATION_SPEC),
                "relation_spec_sha256": None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC),
                "r0_gate": copy.deepcopy(_R0_GATE),
                "r1_gate": copy.deepcopy(_R1_GATE),
                "r2_gate": copy.deepcopy(_R2_GATE),
                "hashes": {
                    **dict(payload.get("hashes", {})),
                    **k1._resource_hashes(),
                    "relation_spec_sha256": None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC),
                    "r1_gate_metrics_sha256": None if _R1_GATE is None else _R1_GATE.get("metrics_sha256"),
                    "r2_gate_metrics_sha256": None if _R2_GATE is None else _R2_GATE.get("metrics_sha256"),
                    "r3_training_script_sha256": common.sha256_file(Path(__file__).resolve()),
                },
                "pca_parameters_sha256_record": copy.deepcopy(k1._PCA_PARAMETER_RECORD),
            }
        )
    k1._ORIGINAL_TORCH_SAVE_ATOMIC(payload, path)


def _patched_evaluate_r3(*args: Any, **kwargs: Any):
    split_name = kwargs.get("split_name")
    if isinstance(split_name, str):
        kwargs["split_name"] = split_name.replace("K0", EXPERIMENT).replace("K1", EXPERIMENT)
    return k1._ORIGINAL_EVALUATE(*args, **kwargs)


def _r3_print(*values: object, **kwargs: object) -> None:
    adjusted = tuple(
        value.replace("K1", EXPERIMENT).replace("R1", EXPERIMENT).replace("R2", EXPERIMENT)
        if isinstance(value, str) else value
        for value in values
    )
    builtins.print(*adjusted, **kwargs)


def _r3_tqdm(*args: Any, **kwargs: Any):
    description = kwargs.get("desc")
    if isinstance(description, str):
        kwargs["desc"] = description.replace("K1", EXPERIMENT).replace("R1", EXPERIMENT)
    return _ORIGINAL_TQDM(*args, **kwargs)


def _gradient_l2(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm(2).item())


def _gradient_cosine(first: torch.Tensor, second: torch.Tensor) -> Optional[float]:
    a = first.detach().float().reshape(-1)
    b = second.detach().float().reshape(-1)
    denom = float(a.norm().item() * b.norm().item())
    return None if denom <= 1e-12 else float(torch.dot(a, b).item() / denom)


def _mean_std(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return {"mean": None, "sample_std": None}
    mean = sum(finite) / len(finite)
    std = 0.0 if len(finite) < 2 else math.sqrt(sum((v - mean) ** 2 for v in finite) / (len(finite) - 1))
    return {"mean": mean, "sample_std": std}


def _aggregate_gradient_record_r3(local_record: Dict[str, object], world_size: int) -> Dict[str, object]:
    rows: List[Optional[Dict[str, object]]] = [None for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather_object(rows, local_record)
    else:
        rows[0] = local_record
    valid = [row for row in rows if row is not None]
    if len(valid) != world_size:
        raise RuntimeError("R3 failed to gather all gradient audits")
    summary = dict(local_record)
    summary["rank_aggregation"] = "mean across ranks; sample_std and per_rank retained"
    summary["per_rank"] = valid
    fields = (
        "grad_l2_ce", "grad_l2_feature", "grad_l2_relation_r1_os4", "grad_l2_relation_r1_os8",
        "grad_l2_relation_r1_os16", "grad_l2_relation_r2_os4", "grad_l2_relation_r2_os8",
        "grad_l2_relation_r2_os16", "grad_l2_relation_effective_r1_os16",
        "grad_l2_relation_effective_r2_os16", "grad_l2_relation_effective_os16",
        "grad_l2_total_os16", "grad_l2_total_student", "relation_to_feature_effective_ratio_os16",
        "relation_to_ce_effective_ratio_os16", "cos_ce_feature_os16", "cos_ce_relation_r1_os16",
        "cos_ce_relation_r2_os16", "cos_feature_relation_r1_os16", "cos_feature_relation_r2_os16",
    )
    for field in fields:
        stats = _mean_std([row.get(field) for row in valid])
        summary[field] = stats["mean"]
        summary[f"{field}_sample_std"] = stats["sample_std"]
    return summary


def _update_relation_stop_gate_r3(record: Mapping[str, object]) -> None:
    global _RELATION_GATE_CONSECUTIVE_EXCESS
    ratio = record.get("relation_to_ce_effective_ratio_os16")
    if ratio is not None and float(ratio) > GRADIENT_CE_STOP_RATIO:
        _RELATION_GATE_CONSECUTIVE_EXCESS += 1
    else:
        _RELATION_GATE_CONSECUTIVE_EXCESS = 0
    if _RELATION_GATE_CONSECUTIVE_EXCESS >= GRADIENT_CE_STOP_CONSECUTIVE:
        raise RuntimeError("R3 combined relation gradient exceeded 2x CE for three consecutive records")


def _reduce_r3_statistics(
    layer_sums: Mapping[str, float], feature_sum: float, r1_sum: float, r2_sum: float,
    total_sum: float, token_sum: float, pair_sum: float, physical_sum: float,
    batch_count: int, min_physical: int, max_physical: int, device: torch.device, world_size: int,
) -> Tuple[Dict[str, float], float, float, float, float, float, float, float, int, int, int]:
    values = [layer_sums[layer] for layer in a0.A0_LAYER_ORDER]
    values.extend([feature_sum, r1_sum, r2_sum, total_sum, token_sum, pair_sum, physical_sum, float(batch_count)])
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    count = max(int(tensor[-1].item()), 1)
    means = {layer: float(tensor[i].item() / count) for i, layer in enumerate(a0.A0_LAYER_ORDER)}
    extrema = torch.tensor([min_physical, max_physical], device=device, dtype=torch.int64)
    if world_size > 1:
        low, high = extrema[:1].clone(), extrema[1:].clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        dist.all_reduce(high, op=dist.ReduceOp.MAX)
        min_physical, max_physical = int(low.item()), int(high.item())
    return (
        means,
        float(tensor[-8].item() / count), float(tensor[-7].item() / count),
        float(tensor[-6].item() / count), float(tensor[-5].item() / count),
        float(tensor[-4].item() / count), float(tensor[-3].item() / count),
        float(tensor[-2].item() / count), count, min_physical, max_physical,
    )


def train_one_epoch_r3(
    model: torch.nn.Module, loader: DataLoader, sampler: Optional[DistributedSampler],
    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler, device: torch.device, amp_enabled: bool,
    accumulation_steps: int, epoch: int, starting_optimizer_step: int,
    remaining_optimizer_steps: int, rank: int, world_size: int,
) -> Tuple[Dict[str, object], int, List[Dict[str, object]], Optional[Dict[str, object]]]:
    global _FIRST_BATCH_BASE_EQUIVALENCE
    teacher, projection = k1._require_resources()
    args = k1._ACTIVE_ARGS
    warmup_steps = k1._warmup_steps(args)
    if sampler is not None:
        sampler.set_epoch(epoch)
    model.train(); teacher.eval(); projection.eval(); optimizer.zero_grad(set_to_none=True)
    confusion = torch.zeros(common.NUM_CLASSES, common.NUM_CLASSES, dtype=torch.int64)
    ce_sum = 0.0; valid_pixels = 0; feature_sum = 0.0; r1_sum = 0.0; r2_sum = 0.0; total_sum = 0.0
    token_sum = 0.0; pair_sum = 0.0; physical_sum = 0.0; batch_count = 0
    min_physical = 1 << 30; max_physical = 0; optimizer_steps = 0; last_warmup = 0.0
    layer_sums = {layer: 0.0 for layer in a0.A0_LAYER_ORDER}
    gradient_records: List[Dict[str, object]] = []; first_batch_audit: Optional[Dict[str, object]] = None
    possible_steps = math.ceil(len(loader) / accumulation_steps)
    target_steps = min(possible_steps, remaining_optimizer_steps)
    max_batches = min(len(loader), target_steps * accumulation_steps)
    progress = _ORIGINAL_TQDM(loader, desc=f"Epoch {epoch} [R3 CE+feature+R1+R2]", disable=rank != 0)

    for batch_index, (images, targets, paths) in enumerate(progress):
        if batch_index >= max_batches:
            break
        group_position = batch_index % accumulation_steps
        if group_position == 0:
            group_size = min(accumulation_steps, max_batches - batch_index)
        sync_gradients = group_position + 1 == group_size
        images = images.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
        next_step = starting_optimizer_step + optimizer_steps + 1
        warmup = min(1.0, next_step / warmup_steps)
        sync_context = contextlib.nullcontext()
        if isinstance(model, DDP) and not sync_gradients:
            sync_context = model.no_sync()
        with sync_context:
            with common.autocast_context(device, amp_enabled):
                output = model(images)
                if not isinstance(output, Mapping):
                    raise RuntimeError("R3 training forward did not return features")
                logits = output["logits"]; student_features = output["features"]
                with torch.no_grad():
                    teacher_features = teacher.extract_features(images)
                feature_layers: Dict[str, torch.Tensor] = {}
                projected_shapes: Dict[str, List[int]] = {}
                for layer in a0.A0_LAYER_ORDER:
                    projected = projection[layer](teacher_features[layer].detach())
                    projected_shapes[layer] = list(projected.shape)
                    feature_layers[layer] = F.mse_loss(student_features[layer].float(), projected.float())
            logits_float = logits.float()
            ce_sum_batch = F.cross_entropy(logits_float, targets, ignore_index=common.IGNORE_INDEX, reduction="sum")
            batch_valid = int((targets != common.IGNORE_INDEX).sum().item())
            if batch_valid <= 0:
                raise RuntimeError("R3 training batch contains no valid Cityscapes pixels")
            loss_ce = ce_sum_batch / batch_valid
            loss_feat = sum(feature_layers.values()) / len(a0.A0_LAYER_ORDER)
            loss_r1, loss_r2, r1_layers, r2_layers, r1_audit, r2_audit = r3_relation_losses(
                student_features, teacher_features, targets, world_size
            )
            total_loss = loss_ce + warmup * (args.lambda_feat * loss_feat + args.lambda_r1 * loss_r1 + args.lambda_r2 * loss_r2)
            finite = [loss_ce, loss_feat, loss_r1, loss_r2, total_loss, *feature_layers.values(), *r1_layers.values(), *r2_layers.values()]
            if not all(bool(torch.isfinite(value).all().item()) for value in finite):
                raise RuntimeError("R3 produced a non-finite CE/feature/relation loss")

            # Independent R3 intentionally skips relation-gradient auditing.
            # Loss finiteness and teacher/projection freeze checks remain active.
            log_gradients = GRADIENT_REVIEW_ENABLED and sync_gradients and (
                next_step == 1 or next_step % args.gradient_log_steps == 0
            )
            local_record: Optional[Dict[str, object]] = None
            if log_gradients:
                per_layer: Dict[str, Dict[str, object]] = {}; gradients: Dict[str, Dict[str, torch.Tensor]] = {}
                for layer in a0.A0_LAYER_ORDER:
                    tap = student_features[layer]
                    grad_ce = torch.autograd.grad(loss_ce, tap, retain_graph=True)[0].detach().float()
                    grad_feat = torch.autograd.grad(loss_feat, tap, retain_graph=True)[0].detach().float()
                    grad_r1 = torch.autograd.grad(loss_r1, tap, retain_graph=True)[0].detach().float()
                    grad_r2 = torch.autograd.grad(loss_r2, tap, retain_graph=True)[0].detach().float()
                    gradients[layer] = {"ce": grad_ce, "feature": grad_feat, "r1": grad_r1, "r2": grad_r2}
                    combined = args.lambda_r1 * grad_r1 + args.lambda_r2 * grad_r2
                    effective = warmup * combined
                    per_layer[layer] = {
                        "tap_shape": list(tap.shape), "grad_l2_ce": _gradient_l2(grad_ce),
                        "grad_l2_feature": _gradient_l2(grad_feat), "grad_l2_relation_r1": _gradient_l2(grad_r1),
                        "grad_l2_relation_r2": _gradient_l2(grad_r2), "grad_l2_relation_effective": _gradient_l2(effective),
                        "cos_ce_feature": _gradient_cosine(grad_ce, grad_feat),
                        "cos_ce_relation_r1": _gradient_cosine(grad_ce, grad_r1),
                        "cos_ce_relation_r2": _gradient_cosine(grad_ce, grad_r2),
                        "cos_feature_relation_r1": _gradient_cosine(grad_feat, grad_r1),
                        "cos_feature_relation_r2": _gradient_cosine(grad_feat, grad_r2),
                    }
                os16 = gradients["os16"]
                effective_r1 = warmup * args.lambda_r1 * os16["r1"]
                effective_r2 = warmup * args.lambda_r2 * os16["r2"]
                effective_relation = effective_r1 + effective_r2
                feature_effective = warmup * args.lambda_feat * os16["feature"]
                ce_norm = _gradient_l2(os16["ce"]); feature_norm = _gradient_l2(feature_effective); relation_norm = _gradient_l2(effective_relation)
                local_record = {
                    "optimizer_step": next_step, "fixed_audit_step": next_step in FIXED_GRADIENT_AUDIT_STEPS,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]), "warmup_weight": warmup,
                    "lambda_feat": args.lambda_feat, "lambda_r1": args.lambda_r1, "lambda_r2": args.lambda_r2,
                    "relation_r1_loss_raw": float(loss_r1.detach().item()), "relation_r2_loss_raw": float(loss_r2.detach().item()),
                    "relation_r1_loss_weighted": float((warmup * args.lambda_r1 * loss_r1).detach().item()),
                    "relation_r2_loss_weighted": float((warmup * args.lambda_r2 * loss_r2).detach().item()),
                    "grad_l2_ce": ce_norm, "grad_l2_feature": _gradient_l2(os16["feature"]), "grad_l2_relation_r1_os4": _gradient_l2(gradients["os4"]["r1"]),
                    "grad_l2_relation_r1_os8": _gradient_l2(gradients["os8"]["r1"]), "grad_l2_relation_r1_os16": _gradient_l2(os16["r1"]),
                    "grad_l2_relation_r2_os4": _gradient_l2(gradients["os4"]["r2"]), "grad_l2_relation_r2_os8": _gradient_l2(gradients["os8"]["r2"]),
                    "grad_l2_relation_r2_os16": _gradient_l2(os16["r2"]), "grad_l2_relation_effective_r1_os16": _gradient_l2(effective_r1),
                    "grad_l2_relation_effective_r2_os16": _gradient_l2(effective_r2), "grad_l2_relation_effective_os16": relation_norm,
                    "grad_l2_total_os16": _gradient_l2(os16["ce"] + feature_effective + effective_relation),
                    "relation_to_feature_effective_ratio_os16": relation_norm / max(feature_norm, 1e-12),
                    "relation_to_ce_effective_ratio_os16": relation_norm / max(ce_norm, 1e-12),
                    "cos_ce_feature_os16": per_layer["os16"]["cos_ce_feature"], "cos_ce_relation_r1_os16": per_layer["os16"]["cos_ce_relation_r1"],
                    "cos_ce_relation_r2_os16": per_layer["os16"]["cos_ce_relation_r2"], "cos_feature_relation_r1_os16": per_layer["os16"]["cos_feature_relation_r1"],
                    "cos_feature_relation_r2_os16": per_layer["os16"]["cos_feature_relation_r2"], "relation_valid_token_count": r2_audit["valid_token_count"],
                    "relation_valid_pair_count": r2_audit["valid_pair_count"], "relation_physical_batch_size": r1_audit["physical_batch_size"],
                    "relation_finite": True, "layers": per_layer, "gradient_component_scope": "native student OS=4/8/16 taps",
                    "relation_spec_sha256": None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC),
                }
            scaler.scale(total_loss / group_size).backward()
        if sync_gradients:
            scaler.unscale_(optimizer); optimizer_steps += 1
            if local_record is not None:
                local_record["grad_l2_total_student"] = k0._gradient_l2_named(model)
                aggregated = _aggregate_gradient_record_r3(local_record, world_size)
                if GRADIENT_REVIEW_ENABLED:
                    _update_relation_stop_gate_r3(aggregated)
                if rank == 0:
                    gradient_records.append(aggregated)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()

        if first_batch_audit is None and starting_optimizer_step == 0 and batch_index == 0:
            global_paths: List[Optional[List[str]]] = [None for _ in range(world_size)]
            if world_size > 1:
                dist.all_gather_object(global_paths, list(paths))
            else:
                global_paths[0] = list(paths)
            first_batch_audit = {
                "rank": rank, "epoch": epoch, "micro_batch_index": 0, "paths": list(paths),
                "relation_global_path_order_by_rank": global_paths, "image_tensor_shape": list(images.shape), "target_tensor_shape": list(targets.shape),
                "image_tensor_sha256": k0._tensor_sha256(images), "target_tensor_sha256": k0._tensor_sha256(targets), "valid_pixels": batch_valid,
                "student_feature_shapes": {layer: list(student_features[layer].shape) for layer in a0.A0_LAYER_ORDER},
                "teacher_feature_shapes": {layer: list(teacher_features[layer].shape) for layer in a0.A0_LAYER_ORDER}, "projected_teacher_shapes": projected_shapes,
                "feature_loss_by_layer": {layer: float(feature_layers[layer].detach().item()) for layer in a0.A0_LAYER_ORDER}, "feature_loss": float(loss_feat.detach().item()), "ce_loss": float(loss_ce.detach().item()),
                "relation_r1_loss_by_layer": {layer: float(r1_layers[layer].detach().item()) for layer in a0.A0_LAYER_ORDER}, "relation_r1_loss": float(loss_r1.detach().item()),
                "relation_r2_loss_by_layer": {layer: float(r2_layers[layer].detach().item()) for layer in a0.A0_LAYER_ORDER}, "relation_r2_loss": float(loss_r2.detach().item()),
                "relation_r1": r1_audit, "relation_r2": r2_audit,
                "relation_loss_weighted": float((warmup * (args.lambda_r1 * loss_r1 + args.lambda_r2 * loss_r2)).detach().item()),
                "warmup_weight": warmup, "lambda_r1": args.lambda_r1, "lambda_r2": args.lambda_r2,
                "total_loss": float(total_loss.detach().item()), "relation_spec_sha256": None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC), **k1._resource_hashes(),
            }
            local_equivalence = r1._compare_first_batch_base_to_k1(first_batch_audit, rank)
            gathered: List[Optional[Dict[str, object]]] = [None for _ in range(world_size)]
            if world_size > 1:
                dist.all_gather_object(gathered, local_equivalence)
            else:
                gathered[0] = local_equivalence
            _FIRST_BATCH_BASE_EQUIVALENCE = {"passed": all(value is not None and bool(value.get("passed")) for value in gathered), "world_size": world_size, "comparison": "R3 CE+feature base versus locked K1", "per_rank": gathered}
            r1._FIRST_BATCH_BASE_EQUIVALENCE = _FIRST_BATCH_BASE_EQUIVALENCE
            first_batch_audit["r3_base_k1_equivalence"] = local_equivalence
            if not _FIRST_BATCH_BASE_EQUIVALENCE["passed"]:
                raise RuntimeError("R3 changed a locked K1 first-batch base field")

        predictions = logits_float.detach().argmax(dim=1)
        confusion += common.confusion_counts(predictions, targets); ce_sum += float(ce_sum_batch.detach().item()); valid_pixels += batch_valid
        feature_value = float(loss_feat.detach().item()); r1_value = float(loss_r1.detach().item()); r2_value = float(loss_r2.detach().item())
        feature_sum += feature_value; r1_sum += r1_value; r2_sum += r2_value; total_sum += float(total_loss.detach().item())
        token_sum += float(r2_audit["valid_token_count"]); pair_sum += float(r2_audit["valid_pair_count"]); physical = int(r1_audit["physical_batch_size"]); physical_sum += physical
        min_physical = min(min_physical, physical); max_physical = max(max_physical, physical); batch_count += 1; last_warmup = warmup
        for layer in a0.A0_LAYER_ORDER:
            layer_sums[layer] += float(feature_layers[layer].detach().item())
        if rank == 0:
            running = common.metrics_from_confusion(confusion, ce_sum, valid_pixels)
            progress.set_postfix({"CE": f"{running['loss']:.4f}", "feat": f"{feature_value:.4f}", "R1": f"{r1_value:.4f}", "R2": f"{r2_value:.4f}", "mIoU": f"{running['mIoU']:.4f}", "warm": f"{warmup:.3f}", "steps": optimizer_steps})

    if optimizer_steps != target_steps:
        raise RuntimeError(f"R3 optimizer-step accounting failed: actual={optimizer_steps}, expected={target_steps}")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("R3 training found a gradient on the frozen teacher")
    if list(projection.parameters()):
        raise RuntimeError("R3 projection became trainable during training")
    if batch_count == 0:
        raise RuntimeError("R3 epoch processed no micro-batches")
    metrics = server_base._reduce_train_metrics(confusion, ce_sum, valid_pixels, device, world_size)
    reduced = _reduce_r3_statistics(layer_sums, feature_sum, r1_sum, r2_sum, total_sum, token_sum, pair_sum, physical_sum, batch_count, min_physical, max_physical, device, world_size)
    layer_means, feature_mean, r1_mean, r2_mean, total_mean, token_mean, pair_mean, physical_mean, global_batches, global_min, global_max = reduced
    metrics.update({
        "loss_schema": "hard_label_CE_plus_A0_feature_MSE_plus_R1_relation_MSE_plus_R2_relation_MSE", "ce_loss": metrics["loss"],
        "feature_loss": feature_mean, "feature_loss_by_layer": layer_means, "logit_loss": None, "relation_enabled": True,
        "relation_r1_loss": r1_mean, "relation_r2_loss": r2_mean,
        "relation_loss_weighted_at_last_warmup": last_warmup * (args.lambda_r1 * r1_mean + args.lambda_r2 * r2_mean),
        "relation_valid_token_count": token_mean, "relation_valid_pair_count": pair_mean, "relation_physical_batch_size_mean": physical_mean,
        "relation_physical_batch_size_min": global_min, "relation_physical_batch_size_max": global_max, "total_loss_micro_batch_mean": total_mean,
        "warmup_weight": last_warmup, "micro_batches_global": global_batches,
    })
    return metrics, optimizer_steps, gradient_records, first_batch_audit


def smoke_test_r3(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp_enabled: bool, rank: int) -> None:
    global _REFERENCE_TESTS
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    _REFERENCE_TESTS = run_r3_reference_tests(device, world_size)
    teacher, projection = k1._require_resources(); args = k1._ACTIVE_ARGS
    model.train(); teacher.eval(); projection.eval(); images, targets, paths = next(iter(loader)); images = images.to(device); targets = targets.to(device); model.zero_grad(set_to_none=True)
    with common.autocast_context(device, amp_enabled):
        output = model(images)
        if not isinstance(output, Mapping):
            raise RuntimeError("R3 smoke forward did not expose features")
        with torch.no_grad():
            teacher_features = teacher.extract_features(images)
        student_features = output["features"]
        feature_layers = {layer: F.mse_loss(student_features[layer].float(), projection[layer](teacher_features[layer].detach()).float()) for layer in a0.A0_LAYER_ORDER}
        r1_loss, r2_loss, r1_layers, r2_layers, r1_audit, r2_audit = r3_relation_losses(student_features, teacher_features, targets, world_size)
    logits = output["logits"].float(); valid = int((targets != common.IGNORE_INDEX).sum().item())
    if valid <= 0:
        raise RuntimeError("R3 smoke batch contains no valid Cityscapes pixels")
    ce = F.cross_entropy(logits, targets, ignore_index=common.IGNORE_INDEX, reduction="sum") / valid; feat = sum(feature_layers.values()) / 3
    warmup = 1.0 / k1._warmup_steps(args); total = ce + warmup * (args.lambda_feat * feat + args.lambda_r1 * r1_loss + args.lambda_r2 * r2_loss)
    # Keep a single backward smoke check for trainability, but do not compute
    # or report relation-gradient ratios/norms in independent R3 mode.
    total.backward()
    if not all(bool(torch.isfinite(value).all().item()) for value in (ce, feat, r1_loss, r2_loss, total)):
        raise RuntimeError("R3 smoke test produced a non-finite loss")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("R3 smoke test found a teacher gradient")
    if rank == 0:
        _r3_print(f"[OK] R3 server smoke: sample={paths[0]}, logits={tuple(logits.shape)}, CE={ce.item():.6f}, feature={feat.item():.6f}, R1={r1_loss.item():.6f}, R2={r2_loss.item():.6f}, total={total.item():.6f}, relation_B={r1_audit['physical_batch_size']}, valid_tokens={r2_audit['valid_token_count']}, valid_pairs={r2_audit['valid_pair_count']}, gradient_review=disabled")


def _read_gradient_gate_summary_r3(path: Path) -> Dict[str, object]:
    if not GRADIENT_REVIEW_ENABLED:
        return {
            "enabled": False,
            "review_required": False,
            "records": 0,
            "target_ratio_range": [GRADIENT_GATE_MIN, GRADIENT_GATE_MAX],
            "passed_target_at_any_record": None,
            "reason": "Independent R3 mode does not perform gradient review",
        }
    if not path.is_file():
        return {"records": 0, "target_ratio_range": [GRADIENT_GATE_MIN, GRADIENT_GATE_MAX], "passed_target_at_any_record": False}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ratios = [float(record["relation_to_feature_effective_ratio_os16"]) for record in records if record.get("relation_to_feature_effective_ratio_os16") is not None]
    ce_ratios = [float(record["relation_to_ce_effective_ratio_os16"]) for record in records if record.get("relation_to_ce_effective_ratio_os16") is not None]
    return {"records": len(records), "fixed_audit_steps_expected": list(FIXED_GRADIENT_AUDIT_STEPS), "fixed_audit_steps_observed": [int(r["optimizer_step"]) for r in records if r.get("fixed_audit_step")], "target_ratio_range": [GRADIENT_GATE_MIN, GRADIENT_GATE_MAX], "relation_to_feature_effective_ratio_min": min(ratios) if ratios else None, "relation_to_feature_effective_ratio_max": max(ratios) if ratios else None, "passed_target_at_any_record": any(GRADIENT_GATE_MIN <= r <= GRADIENT_GATE_MAX for r in ratios), "relation_to_ce_effective_ratio_max": max(ce_ratios) if ce_ratios else None}


def _existing_first_batch_equivalence_r3(args: argparse.Namespace) -> Optional[Dict[str, object]]:
    path = r3_paths(args.output_dir, args.seed)["first_batch_audit"]
    if not path.is_file():
        return None
    audit = _read_json(path)
    rows = audit.get("per_rank")
    if not isinstance(rows, Sequence):
        return None
    values = [
        dict(row["r3_base_k1_equivalence"])
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("r3_base_k1_equivalence"), Mapping)
    ]
    if not values:
        return None
    return {
        "passed": all(bool(value.get("passed")) for value in values),
        "world_size": len(values),
        "comparison": "restored from existing R3 first_batch_audit.json",
        "per_rank": values,
    }


def _postprocess_metrics_r3(args: argparse.Namespace) -> None:
    r1._ORIGINAL_K1_POSTPROCESS(args)
    if int(os.environ.get("RANK", "0")) != 0:
        return
    metrics_path = r3_paths(args.output_dir, args.seed)["metrics"]
    results = _read_json(metrics_path)
    relation_spec = _RELATION_SPEC or results.get("config", {}).get("relation")
    relation_hash = None if not isinstance(relation_spec, Mapping) else _canonical_sha256(relation_spec)
    best = results.get("best_dev_metrics", {})
    r0_reference = _R0_GATE or _read_optional_r0_reference(args)
    r1_reference = _R1_GATE or _read_optional_relation_reference("R1", args, "lambda_r1")
    r2_reference = _R2_GATE or _read_optional_relation_reference("R2", args, "lambda_r2")
    def miou(value: Mapping[str, object]) -> Optional[float]:
        candidate = value.get("best_dev_metrics", {})
        return float(candidate["mIoU"]) if isinstance(candidate, Mapping) and candidate.get("mIoU") is not None else None
    r3_miou = miou(results)
    r0_miou_value = r0_reference.get("r0_best_dev_mIoU")
    r1_miou_value = r1_reference.get("best_dev_mIoU", r1_reference.get("r1_best_dev_mIoU"))
    r2_miou_value = r2_reference.get("best_dev_mIoU", r2_reference.get("r2_best_dev_mIoU"))
    r0_miou = float(r0_miou_value) if r0_miou_value is not None else None
    r1_miou = float(r1_miou_value) if r1_miou_value is not None else None
    r2_miou = float(r2_miou_value) if r2_miou_value is not None else None
    results.update({
        "experiment": EXPERIMENT, "experiment_group": EXPERIMENT_GROUP, "artifact_type": ARTIFACT_TYPE,
        "protocol": "R3 independent combined relation-KD run: K1-compatible seed-specific shared initialization, hard-label CE, locked A0 feature MSE, native masked-GAP R1 and masked 8x16 token R2 relation MSE, no R0/R1/R2 checkpoint or result-gate dependency, shared 4000-step warm-up, fixed 80k budget, dev_local selection, no test_local evaluation, and no relation-gradient review.",
        "relation": copy.deepcopy(relation_spec), "relation_spec_sha256": relation_hash, "relation_reference_tests": copy.deepcopy(_REFERENCE_TESTS),
        "r0_gate": copy.deepcopy(r0_reference), "r1_gate": copy.deepcopy(r1_reference), "r2_gate": copy.deepcopy(r2_reference),
        "r3_first_batch_base_equivalence": copy.deepcopy(_FIRST_BATCH_BASE_EQUIVALENCE or _existing_first_batch_equivalence_r3(args)),
        "r3_vs_r0": {"R3_mIoU": r3_miou, "R0_mIoU": r0_miou, "delta_R3_minus_R0": None if r3_miou is None or r0_miou is None else r3_miou - r0_miou},
        "r3_vs_r1": {"R3_mIoU": r3_miou, "R1_mIoU": r1_miou, "delta_R3_minus_R1": None if r3_miou is None or r1_miou is None else r3_miou - r1_miou},
        "r3_vs_r2": {"R3_mIoU": r3_miou, "R2_mIoU": r2_miou, "delta_R3_minus_R2": None if r3_miou is None or r2_miou is None else r3_miou - r2_miou},
        "physical_relation_batch_size": relation_spec.get("physical_relation_batch_size") if isinstance(relation_spec, Mapping) else None,
        "effective_optimizer_batch_size": relation_spec.get("effective_optimizer_batch_size") if isinstance(relation_spec, Mapping) else None,
        "test_local_evaluated": False,
    })
    loss = results.get("loss")
    if isinstance(loss, dict):
        loss.update({"relation_kd": True, "relation_r1": True, "relation_r2": True, "lambda_r1": float(args.lambda_r1), "lambda_r2": float(args.lambda_r2), "total": "CE + warmup * (lambda_feat * feature + lambda_r1 * R1 + lambda_r2 * R2)"})
    results["gradient_gate"] = _read_gradient_gate_summary_r3(r3_paths(args.output_dir, args.seed)["gradient_norms"])
    results["hashes"] = {**dict(results.get("hashes", {})), "relation_spec_sha256": relation_hash, "r1_gate_metrics_sha256": r1_reference.get("metrics_sha256"), "r2_gate_metrics_sha256": r2_reference.get("metrics_sha256"), "r3_training_script_sha256": common.sha256_file(Path(__file__).resolve())}
    common.write_json_atomic(metrics_path, results)


def run_training(args: argparse.Namespace) -> None:
    global _R0_GATE, _R1_GATE, _R2_GATE, _RELATION_SPEC, _REFERENCE_TESTS, _FIRST_BATCH_BASE_EQUIVALENCE, _RELATION_GATE_CONSECUTIVE_EXCESS, _ACTIVE_LAMBDA_R1, _ACTIVE_LAMBDA_R2
    _ACTIVE_LAMBDA_R1 = float(args.lambda_r1)
    _ACTIVE_LAMBDA_R2 = float(args.lambda_r2)
    old_k1_reference = r1._K1_REFERENCE
    old_k1_reference_validation = r1._K1_REFERENCE_VALIDATION
    old_r1_relation_spec = r1._RELATION_SPEC
    old_r1_reference_tests = r1._REFERENCE_TESTS
    old_r1_first_batch_equivalence = r1._FIRST_BATCH_BASE_EQUIVALENCE
    old_r1_comparator = r1._compare_first_batch_base_to_k1
    # Keep the locked K1 code/config/resource contract, but validate the
    # selected seed's own shared initialization.  R0/R1/R2 artifacts are
    # optional provenance in independent R3 mode.
    r1._K1_REFERENCE, r1._K1_REFERENCE_VALIDATION = r0._validate_k1_reference(
        args, allow_seed_specific_shared_init=True
    )
    _R0_GATE = _read_optional_r0_reference(args)
    _R1_GATE = _read_optional_relation_reference("R1", args, "lambda_r1")
    _R2_GATE = _read_optional_relation_reference("R2", args, "lambda_r2")
    _RELATION_SPEC = None; _REFERENCE_TESTS = None; _RELATION_GATE_CONSECUTIVE_EXCESS = 0
    _FIRST_BATCH_BASE_EQUIVALENCE = _existing_first_batch_equivalence_r3(args) if args.resume else None
    saved = {
        "__file__": k1.__file__, "EXPERIMENT": k1.EXPERIMENT, "ARTIFACT_TYPE": k1.ARTIFACT_TYPE, "ARTIFACT_FORMAT_VERSION": k1.ARTIFACT_FORMAT_VERSION,
        "k1_paths": k1.k1_paths, "build_config": k1.build_config, "build_best_checkpoint": k1.build_best_checkpoint, "train_one_epoch_k1": k1.train_one_epoch_k1,
        "smoke_test_k1": k1.smoke_test_k1, "_postprocess_metrics": k1._postprocess_metrics, "audit_k1_shapes": k1.audit_k1_shapes,
        "_patched_torch_save_atomic": k1._patched_torch_save_atomic, "_patched_evaluate": k1._patched_evaluate, "_k1_print": k1._k1_print,
        "tqdm": k1.tqdm, "_ORIGINAL_ENSURE_SHARED_INITIALIZATION": k1._ORIGINAL_ENSURE_SHARED_INITIALIZATION,
    }
    had_print = "print" in k1.__dict__; old_print = k1.__dict__.get("print"); old_loader = server_base.build_train_loader
    try:
        k1.__file__ = str(Path(__file__).resolve()); k1.EXPERIMENT = EXPERIMENT; k1.ARTIFACT_TYPE = ARTIFACT_TYPE; k1.ARTIFACT_FORMAT_VERSION = ARTIFACT_FORMAT_VERSION
        k1.k1_paths = r3_paths; k1.build_config = build_config_r3; k1.build_best_checkpoint = build_best_checkpoint_r3; k1.train_one_epoch_k1 = train_one_epoch_r3
        k1.smoke_test_k1 = smoke_test_r3; k1._postprocess_metrics = _postprocess_metrics_r3; k1.audit_k1_shapes = audit_shapes_r3; k1._patched_torch_save_atomic = _patched_torch_save_atomic_r3
        k1._patched_evaluate = _patched_evaluate_r3; k1._k1_print = _r3_print; k1.tqdm = _r3_tqdm; k1._ORIGINAL_ENSURE_SHARED_INITIALIZATION = _ensure_locked_k_shared_initialization; k1.print = _r3_print
        # Seed-specific independent R3 first-batch audit: compare only
        # invariant protocol/resource fields, not seed-42 images or losses.
        r1._compare_first_batch_base_to_k1 = r2._compare_first_batch_base_r2
        server_base.build_train_loader = r1.build_train_loader_r1
        k1.run_training(args)
    finally:
        for name, value in saved.items():
            setattr(k1, name, value)
        server_base.build_train_loader = old_loader
        if had_print:
            k1.print = old_print
        else:
            k1.__dict__.pop("print", None)
        _RELATION_SPEC = None; _REFERENCE_TESTS = None
        r1._K1_REFERENCE = old_k1_reference
        r1._K1_REFERENCE_VALIDATION = old_k1_reference_validation
        r1._RELATION_SPEC = old_r1_relation_spec
        r1._REFERENCE_TESTS = old_r1_reference_tests
        r1._FIRST_BATCH_BASE_EQUIVALENCE = old_r1_first_batch_equivalence
        r1._compare_first_batch_base_to_k1 = old_r1_comparator


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
