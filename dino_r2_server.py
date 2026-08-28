"""R2 server entry point: within-image spatial relation distillation.

R2 is deliberately a small extension of the audited K1/R1 server pipeline.
It keeps the locked teacher, A0 pointwise feature target, K shared
initialisation, Cityscapes protocol, optimiser and ordered server shutdown,
and adds only the registered 8x16 token-relation loss::

    L = L_seg + warmup(step) * (L_feat + lambda_r2 * L_R2)

The relation source is the native (unprojected) teacher/student feature at
OS=4/8/16.  Ignore pixels are removed before adaptive pooling; each image
therefore contributes a masked 128x128 signed-cosine matrix.  R2 can be run
independently of the R0/R1 result gates; an R0 or R1 result is recorded only
when available so that it can be used as an optional paired comparison after
training.  The locked K1 code/config contract, teacher/PCA resources, and
seed-specific K shared initialization remain required.
"""

from __future__ import annotations

import argparse
import builtins
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
from torch.utils.data import DataLoader, DistributedSampler

import dino as common
import dino_a0_server as a0
import dino_k0_server as k0
import dino_k1_server as k1
import dino_r0_server as r0
import dino_r1_server as r1
import dino_s2_0 as base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "R_MobileNetV2_RASPP_server"
K_GROUP_OUTPUT_DIR = SCRIPT_DIR / "result" / "K_MobileNetV2_RASPP_server"
R_GROUP_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
# R2 is allowed to run independently for each pre-registered formal seed.
# The seed list is a validation set, not a dependency on an earlier R1 run.
FORMAL_SEEDS = (42, 3407, 260805)
EXPERIMENT = "R2"
EXPERIMENT_GROUP = "R_MobileNetV2_RASPP_server"
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_r2_spatial_relation_kd"
ARTIFACT_FORMAT_VERSION = 1

RELATION_EPSILON = 1e-6
POOL_SIZE = (8, 16)
NUM_TOKENS = POOL_SIZE[0] * POOL_SIZE[1]
# The seed-42 screening run at 0.03 completed with a relation/feature
# effective-gradient ratio below the registered lower gate.  The next
# pre-registered R2 calibration therefore starts at 0.3 (ten times the
# original weight); the smaller values remain selectable for audit/replay.
LAMBDA_R2 = 0.3
ALLOWED_LAMBDA_R2 = (0.015, 0.03, 0.06, 0.3)
GRADIENT_GATE_MIN = 0.05
GRADIENT_GATE_MAX = 0.20
GRADIENT_CE_STOP_RATIO = 2.0
GRADIENT_CE_STOP_CONSECUTIVE = 3
FIXED_GRADIENT_AUDIT_STEPS = (1, 4_000, 20_000, 40_000, 60_000, 80_000)

_R1_GATE: Optional[Dict[str, object]] = None
_REFERENCE_TESTS: Optional[Dict[str, object]] = None
_RELATION_SPEC: Optional[Dict[str, object]] = None
_RELATION_GATE_CONSECUTIVE_EXCESS = 0
_ORIGINAL_AGGREGATE_GRADIENT_RECORD = r1._aggregate_gradient_record
_ORIGINAL_TRAIN_ONE_EPOCH_R1 = r1.train_one_epoch_r1
_ORIGINAL_K_SHARED_INITIALIZATION = k1._ORIGINAL_ENSURE_SHARED_INITIALIZATION
_ORIGINAL_R0_VALIDATE_K1_REFERENCE = r0._validate_k1_reference
_COLLECT_TRAIN_RELATION_STATS = False
_TRAIN_RELATION_STATS: Dict[str, float] = {}


def parse_args() -> argparse.Namespace:
    """Use the locked K1 CLI and add only the registered R2 weight."""

    saved_default = k1.DEFAULT_OUTPUT_DIR
    saved_argparse = k1.argparse

    class R2ArgparseProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(saved_argparse, name)

        @staticmethod
        def ArgumentParser(*parser_args: Any, **parser_kwargs: Any):
            parser_kwargs["description"] = (
                "R2 MobileNetV2+R-ASPP: hard-label CE plus locked A0 feature "
                "KD and masked 8x16 within-image spatial relation KD."
            )
            parser = saved_argparse.ArgumentParser(*parser_args, **parser_kwargs)
            parser.add_argument(
                "--lambda-r2",
                type=float,
                default=LAMBDA_R2,
                help="Fixed R2 relation weight; registered values only.",
            )
            return parser

    k1.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    k1.argparse = R2ArgparseProxy()
    try:
        args = k1.parse_args()
    finally:
        k1.DEFAULT_OUTPUT_DIR = saved_default
        k1.argparse = saved_argparse

    # K1's historical parser default is 4, while the accepted Cityscapes
    # protocol is physical batch 4 with accumulation 2.  Preserve an explicit
    # user value so the common K1 compatibility validation can reject it.
    if not any(
        value == "--accumulation-steps"
        or value.startswith("--accumulation-steps=")
        for value in sys.argv[1:]
    ):
        args.accumulation_steps = 2

    if args.seed not in FORMAL_SEEDS:
        raise SystemExit(f"R2 seed must be one of {FORMAL_SEEDS}")
    if not any(
        math.isclose(args.lambda_r2, value, rel_tol=0.0, abs_tol=1e-12)
        for value in ALLOWED_LAMBDA_R2
    ):
        raise SystemExit(
            "--lambda-r2 must be one of the registered values 0.015, 0.03, 0.06, 0.3"
        )
    if not args.smoke_test:
        if args.max_steps != 80_000:
            raise SystemExit("Formal R2 is locked to exactly 80,000 optimizer steps")
        if args.eval_every_steps != 5_000:
            raise SystemExit("Formal R2 is locked to --eval-every-steps 5000")
        if args.gradient_log_steps != 500:
            raise SystemExit("Formal R2 is locked to --gradient-log-steps 500")
    if args.output_dir.resolve() == K_GROUP_OUTPUT_DIR.resolve():
        raise SystemExit(
            "R2 output must not point at the K-group directory; use the separate "
            "R_MobileNetV2_RASPP output root"
        )
    # The audited R1 training loop is reused below.  Keep the alias private to
    # this process; it is never written as an R1 experiment setting.
    args.lambda_r1 = args.lambda_r2
    return args


def _lambda_path_component(lambda_r2: float) -> str:
    """Return a stable, human-readable lambda component for run directories."""

    value = float(lambda_r2)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"R2 lambda must be finite and positive, got {lambda_r2!r}")
    return format(value, ".12g")


def r2_paths(
    output_dir: Path,
    seed: int,
    lambda_r2: Optional[float] = None,
) -> Dict[str, Path]:
    if lambda_r2 is None:
        active_args = getattr(k1, "_ACTIVE_ARGS", None)
        if active_args is None or not hasattr(active_args, "lambda_r2"):
            raise RuntimeError(
                "R2 paths require lambda_r2 before the active training arguments "
                "have been initialized"
            )
        lambda_r2 = float(active_args.lambda_r2)
    original = k1._ORIGINAL_K0_PATHS(output_dir, seed)
    lambda_component = _lambda_path_component(lambda_r2)
    run_dir = (
        output_dir.resolve()
        / EXPERIMENT
        / f"seed_{seed}_lambda_{lambda_component}"
    )
    return {
        key: run_dir if key == "run_dir" else run_dir / value.name
        for key, value in original.items()
    }


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


def _r1_metrics_path(args: argparse.Namespace) -> Path:
    return args.output_dir.resolve() / "R1" / f"seed_{args.seed}" / "metrics.json"


def _validate_r1_gate(args: argparse.Namespace) -> Dict[str, object]:
    """Validate the historical R1 gate (kept for audit/replay only).

    This strict validator is intentionally no longer called by ``run_training``.
    R1 is not a launch prerequisite for the independent R2 mode.
    """

    path = _r1_metrics_path(args)
    if not path.is_file():
        if args.smoke_test:
            return {
                "required_for_formal_run": True,
                "checked": False,
                "passed": None,
                "reason": "R1 metrics are absent; protocol smoke is allowed",
                "metrics_path": str(path),
            }
        raise FileNotFoundError(
            "Formal R2 is gated on an accepted R1 seed result: " + str(path)
        )

    metrics = _read_json(path)
    failures: List[str] = []
    if metrics.get("experiment") != "R1":
        failures.append("the gate artifact is not an R1 result")
    if metrics.get("test_local_evaluated") is not False:
        failures.append("R1 does not record test_local_evaluated=false")
    reference = metrics.get("relation_reference_tests")
    if not isinstance(reference, Mapping) or not bool(reference.get("passed")):
        failures.append("R1 relation reference tests did not pass")
    first = metrics.get("r1_first_batch_base_equivalence")
    if not isinstance(first, Mapping) or not bool(first.get("passed")):
        failures.append("R1/K1 first-batch base equivalence did not pass")
    gradient_gate = metrics.get("gradient_gate")
    if not isinstance(gradient_gate, Mapping) or not bool(
        gradient_gate.get("passed_target_at_any_record")
    ):
        failures.append("R1 did not pass the relation/feature gradient gate")
    hashes = metrics.get("hashes", {})
    if not isinstance(hashes, Mapping):
        failures.append("R1 metrics has no hash mapping")
        hashes = {}
    local_r1_hash = common.sha256_file(Path(r1.__file__).resolve())
    if hashes.get("r1_training_script_sha256") != local_r1_hash:
        failures.append(
            "the current dino_r1_server.py differs from the accepted R1 run: "
            f"local={local_r1_hash}, recorded={hashes.get('r1_training_script_sha256')}"
        )

    # The pre-registered effect gate accepts either the primary mIoU route or
    # the mechanism route (boundary/small-object improvement).  Do not let an
    # unpaired single test value unlock R2.
    comparison = metrics.get("r1_vs_r0")
    effect_passed = False
    if isinstance(comparison, Mapping):
        delta = comparison.get("delta_R1_minus_R0")
        if delta is not None and float(delta) > 0.00219:
            effect_passed = True
    best = metrics.get("best_dev_metrics")
    r0_metrics_path = _r0_metrics_path(args)
    r0_metrics = _read_json(r0_metrics_path) if r0_metrics_path.is_file() else {}
    r0_best = r0_metrics.get("best_dev_metrics", {})
    if isinstance(best, Mapping) and isinstance(r0_best, Mapping):
        for field, threshold in (("boundary_f1", 0.00613), ("small_object_mIoU", 0.00851)):
            actual = best.get(field)
            baseline = r0_best.get(field)
            if actual is not None and baseline is not None and float(actual) - float(baseline) >= threshold:
                effect_passed = True
    if not effect_passed:
        failures.append("R1 did not pass the pre-registered effect gate versus R0")

    result = {
        "required_for_formal_run": True,
        "checked": True,
        "passed": not failures,
        "failures": failures,
        "metrics_path": str(path),
        "metrics_sha256": common.sha256_file(path),
        "r1_best_dev_mIoU": (
            best.get("mIoU") if isinstance(best, Mapping) else None
        ),
        "effect_gate_passed": effect_passed,
        "r1_training_script_sha256": local_r1_hash,
    }
    if failures and not args.smoke_test:
        raise RuntimeError("R2 R1-gate validation failed:\n- " + "\n- ".join(failures))
    return result


def _read_optional_r1_reference(args: argparse.Namespace) -> Dict[str, object]:
    """Read R1 metadata without making it a prerequisite for R2.

    A missing or malformed R1 artifact is represented in the run metadata and
    never aborts R2.  The optional result is useful for paired reporting, but
    it is not treated as an acceptance gate and no R1 checkpoint or loss is
    loaded by R2.
    """

    path = _r1_metrics_path(args)
    reference: Dict[str, object] = {
        "required_for_formal_run": False,
        "checked": False,
        "passed": None,
        "available": False,
        "comparison_only": True,
        "reason": "R2 independent mode; R1 result is optional",
        "metrics_path": str(path),
    }
    if not path.is_file():
        reference["reason"] = (
            "R1 metrics are absent; continuing because independent R2 does not "
            "require R1"
        )
        return reference

    try:
        metrics = _read_json(path)
    except Exception as exc:  # pragma: no cover - defensive metadata path
        reference.update(
            {
                "checked": True,
                "reason": "R1 metrics are unreadable; ignored in independent R2",
                "read_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return reference

    best = metrics.get("best_dev_metrics")
    is_r1_artifact = metrics.get("experiment") == "R1"
    reference.update(
        {
            "checked": True,
            "available": True,
            "experiment": metrics.get("experiment"),
            "is_r1_artifact": is_r1_artifact,
            "r1_best_dev_mIoU": (
                best.get("mIoU")
                if is_r1_artifact and isinstance(best, Mapping)
                else None
            ),
            "metrics_sha256": common.sha256_file(path),
        }
    )
    if metrics.get("experiment") != "R1":
        reference["reason"] = (
            "A metrics artifact exists at the R1 path but is not labelled R1; "
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
        "reason": "R2 independent mode; R0 result is optional",
        "metrics_path": str(path),
    }
    if not path.is_file():
        reference["reason"] = (
            "R0 metrics are absent; continuing because independent R2 does not "
            "require an R0 baseline"
        )
        return reference

    try:
        metrics = _read_json(path)
    except Exception as exc:  # pragma: no cover - defensive metadata path
        reference.update(
            {
                "checked": True,
                "reason": "R0 metrics are unreadable; ignored in independent R2",
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
                best.get("mIoU")
                if is_r0_artifact and isinstance(best, Mapping)
                else None
            ),
            "metrics_sha256": common.sha256_file(path),
        }
    )
    if not is_r0_artifact:
        reference["reason"] = (
            "A metrics artifact exists at the R0 path but is not labelled R0; "
            "it is retained for provenance only"
        )
    return reference


def _validate_k1_reference_r2(args: argparse.Namespace):
    """Use K1 checks with seed-specific shared-init support for independent R2."""

    return _ORIGINAL_R0_VALIDATE_K1_REFERENCE(
        args, allow_seed_specific_shared_init=True
    )


def _compare_first_batch_base_r2(
    row: Mapping[str, object], rank: int
) -> Dict[str, object]:
    """Audit seed-invariant first-batch fields without requiring seed-42 data.

    The locked K1 first-batch audit contains concrete paths, pixels and loss
    values for seed 42.  Those values must differ when an independent R2 run
    uses seed 3407/260805.  Resource, tensor-shape and protocol fields remain
    comparable and are still checked strictly.
    """

    reference = r1._reference_rank_row(rank)
    invariant_fields = (
        "image_tensor_shape",
        "target_tensor_shape",
        "student_feature_shapes",
        "teacher_feature_shapes",
        "projected_teacher_shapes",
        "teacher_checkpoint_sha256",
        "k0_shared_training_runner_sha256",
        "pca_parameter_record_sha256",
        "pca_parameter_sha256",
        "projection_parameter_sha256",
        "pca_sampling_manifest_sha256",
        "warmup_weight",
    )
    invariant_mismatches = {
        field: {"actual": row.get(field), "expected": reference.get(field)}
        for field in invariant_fields
        if row.get(field) != reference.get(field)
    }
    seed_specific_fields = (
        "paths",
        "image_tensor_sha256",
        "target_tensor_sha256",
        "valid_pixels",
        "feature_loss_by_layer",
        "feature_loss",
        "ce_loss",
    )
    ignored_mismatches = {
        field: {"actual": row.get(field), "reference_seed_42": reference.get(field)}
        for field in seed_specific_fields
        if row.get(field) != reference.get(field)
    }
    return {
        "rank": rank,
        "passed": not invariant_mismatches,
        "comparison": (
            "R2 independent first-batch invariant fields versus locked K1; "
            "seed-specific data and loss fields are informational"
        ),
        "reference": str((r1.K1_REFERENCE_DIR / "first_batch_audit.json").resolve()),
        "invariant_mismatches": invariant_mismatches,
        "ignored_seed_specific_mismatches": ignored_mismatches,
        "ignored_seed_specific_fields": list(seed_specific_fields),
        "checked_invariant_fields": list(invariant_fields),
    }


def _relation_spec_r2(
    args: argparse.Namespace, accumulation_steps: int, world_size: int
) -> Dict[str, object]:
    physical_batch = int(args.batch_size) * int(world_size)
    return {
        "enabled": True,
        "active_relation_types": ["R2_within_image_spatial"],
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
        "tail_batch_policy": (
            "use the actual synchronized micro-batch; never cache, pad, or combine "
            "samples across optimizer steps"
        ),
        "mask_policy": "nearest-resized valid mask, adaptive-average valid fraction, valid fraction > 0",
        "pooling": "adaptive_avg_pool2d(mask * feature) / adaptive_avg_pool2d(mask)",
        "token_order": "8x16 row-major",
        "layer_aggregation": "equal mean over native OS=4/8/16 relation losses",
        "matrix_dtype": "float32",
        "normalization": "token-row L2 / (norm + epsilon)",
        "r1": {"enabled": False, "lambda": 0.0},
        "r2": {
            "enabled": True,
            "representation": "per-image 128x128 signed token cosine matrix",
            "diagonal_policy": "keep for valid tokens",
            "reduction": "sum over valid token pairs divided by valid-pair count",
            "lambda": float(args.lambda_r2),
            "distributed_reduction": "global numerator/global valid-pair denominator",
        },
        "relation_warmup_steps": 4_000,
        "relation_warmup_shared_with_feature_kd": True,
        "estimated_relation_matrix_bytes_per_rank": int(
            int(args.batch_size) * NUM_TOKENS * NUM_TOKENS * 4 * 2 * 3
        ),
        "relation_gradient_gate": {
            "target_relation_to_feature_effective_ratio": [
                GRADIENT_GATE_MIN,
                GRADIENT_GATE_MAX,
            ],
            "stop_if_relation_to_ce_exceeds": GRADIENT_CE_STOP_RATIO,
            "consecutive_records_before_stop": GRADIENT_CE_STOP_CONSECUTIVE,
            "lambda_is_fixed_during_formal_training": True,
        },
    }


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"R2 {name} contains a non-finite value")


def _resize_valid_mask(targets: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    if targets.ndim != 3:
        raise RuntimeError(f"R2 targets must be [B,H,W], got {tuple(targets.shape)}")
    mask = (targets != common.IGNORE_INDEX).unsqueeze(1).to(dtype=torch.float32)
    return F.interpolate(mask, size=tuple(size), mode="nearest")


def masked_spatial_tokens(
    features: torch.Tensor,
    targets: torch.Tensor,
    pool_size: Tuple[int, int] = POOL_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return [B,128,C] masked tokens, validity and valid fractions."""

    if features.ndim != 4:
        raise RuntimeError(f"R2 features must be [B,C,H,W], got {tuple(features.shape)}")
    _assert_finite("native feature", features)
    mask = _resize_valid_mask(targets, features.shape[-2:])
    valid_fraction = F.adaptive_avg_pool2d(mask, pool_size)
    pooled_feature = F.adaptive_avg_pool2d(features.float() * mask, pool_size)
    valid = valid_fraction > 0.0
    if bool((valid.sum(dim=(1, 2, 3)) == 0).any().item()):
        raise RuntimeError("R2 spatial pooling found an image with no valid token")
    tokens = pooled_feature / valid_fraction.clamp_min(RELATION_EPSILON)
    tokens = torch.where(valid.expand_as(tokens), tokens, torch.zeros_like(tokens))
    tokens = tokens.permute(0, 2, 3, 1).reshape(features.shape[0], NUM_TOKENS, features.shape[1])
    valid_tokens = valid.reshape(features.shape[0], NUM_TOKENS)
    _assert_finite("masked spatial tokens", tokens)
    return tokens, valid_tokens, valid_fraction


def token_cosine_matrix(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise RuntimeError(f"R2 tokens must be [B,N,C], got {tuple(tokens.shape)}")
    vectors = tokens.float()
    _assert_finite("token vectors", vectors)
    norms = vectors.norm(2, dim=2, keepdim=True)
    normalized = vectors / (norms + RELATION_EPSILON)
    matrix = normalized @ normalized.transpose(1, 2)
    _assert_finite("token cosine matrix", matrix)
    return matrix


def _masked_matrix_loss(
    student_matrix: torch.Tensor,
    teacher_matrix: torch.Tensor,
    valid_pairs: torch.Tensor,
    world_size: int,
) -> Tuple[torch.Tensor, int, int]:
    if student_matrix.shape != teacher_matrix.shape or student_matrix.ndim != 3:
        raise RuntimeError("R2 student/teacher matrices must have equal [B,128,128] shapes")
    if valid_pairs.shape != student_matrix.shape or valid_pairs.dtype != torch.bool:
        raise RuntimeError("R2 valid-pair mask shape/dtype mismatch")
    local_numerator = (student_matrix - teacher_matrix.detach()).square().masked_select(valid_pairs).sum()
    local_denominator = valid_pairs.sum().to(dtype=torch.float64)
    if float(local_denominator.item()) <= 0.0:
        raise RuntimeError("R2 valid-pair denominator is zero")
    global_numerator = local_numerator.detach().to(dtype=torch.float64)
    global_denominator = local_denominator.detach().clone()
    if world_size > 1:
        dist.all_reduce(global_numerator, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
    denominator = max(float(global_denominator.item()), 1.0)
    if world_size > 1:
        # DDP averages rank gradients.  Scaling the local numerator by world
        # size makes that average equal the global valid-pair mean.  The value
        # correction keeps the logged loss identical on every rank.
        gradient_value = local_numerator * (float(world_size) / denominator)
        value = global_numerator.to(dtype=local_numerator.dtype) / denominator
        loss = gradient_value + (value - gradient_value.detach())
    else:
        loss = local_numerator / denominator
    _assert_finite("masked spatial relation loss", loss)
    return loss, int(round(float(global_denominator.item()))), int(local_denominator.item())


def r2_relation_losses(
    student_features: Mapping[str, torch.Tensor],
    teacher_features: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    world_size: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, object]]:
    global _TRAIN_RELATION_STATS
    physical_batch, per_rank_sizes = r1._synchronized_local_batch_size(
        int(targets.shape[0]), targets.device, world_size
    )
    layer_losses: Dict[str, torch.Tensor] = {}
    layer_audit: Dict[str, object] = {}
    global_token_total = 0
    global_pair_total = 0
    for layer in a0.A0_LAYER_ORDER:
        student_tokens, student_valid, student_fraction = masked_spatial_tokens(
            student_features[layer], targets
        )
        with torch.no_grad():
            teacher_tokens, teacher_valid, teacher_fraction = masked_spatial_tokens(
                teacher_features[layer].detach(), targets
            )
        if not torch.equal(student_valid, teacher_valid):
            raise RuntimeError(f"R2 {layer} student/teacher valid-token masks differ")
        student_matrix = token_cosine_matrix(student_tokens)
        teacher_matrix = token_cosine_matrix(teacher_tokens).detach()
        valid_pairs = student_valid.unsqueeze(2) & student_valid.unsqueeze(1)
        loss, global_pairs, local_pairs = _masked_matrix_loss(
            student_matrix, teacher_matrix, valid_pairs, world_size
        )
        local_tokens = int(student_valid.sum().item())
        token_count_tensor = torch.tensor(local_tokens, device=targets.device, dtype=torch.int64)
        if world_size > 1:
            dist.all_reduce(token_count_tensor, op=dist.ReduceOp.SUM)
        global_tokens = int(token_count_tensor.item())
        layer_losses[layer] = loss
        global_token_total += global_tokens
        global_pair_total += global_pairs
        layer_audit[layer] = {
            "student_token_shape": list(student_tokens.shape),
            "teacher_token_shape": list(teacher_tokens.shape),
            "matrix_shape": list(student_matrix.shape),
            "student_valid_token_count_local": local_tokens,
            "teacher_valid_token_count_local": int(teacher_valid.sum().item()),
            "valid_token_count_global": global_tokens,
            "valid_pair_count_local": local_pairs,
            "valid_pair_count_global": global_pairs,
            "valid_fraction_min": float(student_fraction.min().detach().item()),
            "valid_fraction_max": float(student_fraction.max().detach().item()),
            "loss": float(loss.detach().item()),
        }
    total = sum(layer_losses.values()) / len(a0.A0_LAYER_ORDER)
    _assert_finite("three-layer R2 relation loss", total)
    mean_valid_tokens = global_token_total / len(a0.A0_LAYER_ORDER)
    mean_valid_pairs = global_pair_total / len(a0.A0_LAYER_ORDER)
    if _COLLECT_TRAIN_RELATION_STATS:
        _TRAIN_RELATION_STATS["calls"] = _TRAIN_RELATION_STATS.get("calls", 0.0) + 1.0
        _TRAIN_RELATION_STATS["valid_tokens"] = (
            _TRAIN_RELATION_STATS.get("valid_tokens", 0.0) + mean_valid_tokens
        )
        _TRAIN_RELATION_STATS["valid_pairs"] = (
            _TRAIN_RELATION_STATS.get("valid_pairs", 0.0) + mean_valid_pairs
        )
    return total, layer_losses, {
        "physical_batch_size": physical_batch,
        "per_rank_batch_sizes": per_rank_sizes,
        "valid_token_count": int(round(mean_valid_tokens)),
        "valid_pair_count": int(round(mean_valid_pairs)),
        "valid_token_count_by_layer": {
            layer: value["valid_token_count_global"] for layer, value in layer_audit.items()
        },
        "valid_pair_count_by_layer": {
            layer: value["valid_pair_count_global"] for layer, value in layer_audit.items()
        },
        "layers": layer_audit,
    }


def _run_local_reference_tests(device: torch.device) -> Dict[str, object]:
    targets = torch.zeros((2, 8, 16), device=device, dtype=torch.long)
    targets[0, 0, 0] = common.IGNORE_INDEX
    targets[1, 4:, 8:] = common.IGNORE_INDEX
    values = torch.arange(1, 2 * 3 * 8 * 16 + 1, device=device, dtype=torch.float32)
    student_features = values.reshape(2, 3, 8, 16) / 31.0
    teacher_features = torch.sin(values.reshape(2, 3, 8, 16) / 11.0)
    tokens, valid, fractions = masked_spatial_tokens(student_features, targets)
    teacher_tokens, teacher_valid, _ = masked_spatial_tokens(teacher_features, targets)
    if list(tokens.shape) != [2, NUM_TOKENS, 3]:
        raise RuntimeError("R2 reference test: token shape mismatch")
    matrices = token_cosine_matrix(tokens)
    valid_pairs = valid.unsqueeze(2) & valid.unsqueeze(1)
    loss, pair_count, _ = _masked_matrix_loss(
        matrices, token_cosine_matrix(teacher_tokens[:, :, :2]), valid_pairs, 1
    )
    if not bool(torch.isfinite(loss).item()) or pair_count <= 0:
        raise RuntimeError("R2 reference test: invalid masked relation loss")

    ignored_changed = student_features.clone()
    ignore_mask = (targets == common.IGNORE_INDEX).unsqueeze(1).expand_as(ignored_changed)
    ignored_changed[ignore_mask] += 1000.0
    ignored_tokens, ignored_valid, _ = masked_spatial_tokens(ignored_changed, targets)
    ignored_matrix = token_cosine_matrix(ignored_tokens)
    ignored_loss, _, _ = _masked_matrix_loss(
        ignored_matrix, token_cosine_matrix(teacher_tokens), ignored_valid.unsqueeze(2) & ignored_valid.unsqueeze(1), 1
    )
    original_loss, _, _ = _masked_matrix_loss(
        matrices, token_cosine_matrix(teacher_tokens), valid_pairs, 1
    )
    if not torch.allclose(original_loss, ignored_loss, atol=1e-6, rtol=1e-6):
        raise RuntimeError("R2 reference test: ignore pixels changed relation loss")

    valid_changed = student_features.clone()
    valid_changed[0, :, 0, 1] += 3.0
    changed_tokens, changed_valid, _ = masked_spatial_tokens(valid_changed, targets)
    changed_loss, _, _ = _masked_matrix_loss(
        token_cosine_matrix(changed_tokens), token_cosine_matrix(teacher_tokens),
        changed_valid.unsqueeze(2) & changed_valid.unsqueeze(1), 1
    )
    if torch.allclose(original_loss, changed_loss, atol=1e-8, rtol=1e-8):
        raise RuntimeError("R2 reference test: valid-pixel change did not change loss")

    student = tokens.clone().requires_grad_(True)
    teacher = token_cosine_matrix(teacher_tokens).detach()
    grad_loss, _, _ = _masked_matrix_loss(
        token_cosine_matrix(student), teacher, valid_pairs, 1
    )
    grad_loss.backward()
    if student.grad is None or float(student.grad.norm().item()) <= 0:
        raise RuntimeError("R2 reference test: relation gradient did not reach student")
    row_norm = tokens.float().norm(dim=2)
    return {
        "passed": True,
        "token_shape": list(tokens.shape),
        "matrix_shape": list(matrices.shape),
        "pool_size": list(POOL_SIZE),
        "valid_pair_count": pair_count,
        "valid_token_count": int(valid.sum().item()),
        "valid_fraction_min": float(fractions.min().item()),
        "valid_fraction_max": float(fractions.max().item()),
        "ignore_invariance": True,
        "valid_pixel_sensitivity": True,
        "diagonal_kept": True,
        "teacher_detached": True,
        "finite_zero_norm": True,
        "token_norm_min": float(row_norm.min().item()),
    }


def run_r2_reference_tests(device: torch.device, world_size: int) -> Dict[str, object]:
    local = _run_local_reference_tests(device)
    distributed: Dict[str, object] = {"passed": True, "world_size": world_size}
    if world_size > 1:
        # All ranks must observe the same globally reduced scalar even when
        # valid-pair counts differ.  This is the R2 DDP reduction contract.
        rank = dist.get_rank()
        targets = torch.zeros((2, 8, 16), device=device, dtype=torch.long)
        if rank == 1:
            targets[:, 4:, 8:] = common.IGNORE_INDEX
        values = torch.arange(1, 2 * 3 * 8 * 16 + 1, device=device, dtype=torch.float32)
        student = values.reshape(2, 3, 8, 16).clone().requires_grad_(True)
        teacher = torch.cos(values.reshape(2, 3, 8, 16) / 7.0)
        student_tokens, valid, _ = masked_spatial_tokens(student, targets)
        teacher_tokens, _, _ = masked_spatial_tokens(teacher, targets)
        loss, global_pairs, _ = _masked_matrix_loss(
            token_cosine_matrix(student_tokens), token_cosine_matrix(teacher_tokens).detach(),
            valid.unsqueeze(2) & valid.unsqueeze(1), world_size
        )
        gathered = [torch.empty_like(loss.detach()) for _ in range(world_size)]
        dist.all_gather(gathered, loss.detach())
        if not all(torch.allclose(value, gathered[0], atol=1e-6, rtol=1e-6) for value in gathered):
            raise RuntimeError("R2 DDP reference test: ranks observed different losses")
        loss.backward()
        distributed = {
            "passed": True,
            "world_size": world_size,
            "global_valid_pair_count": global_pairs,
            "same_scalar_on_all_ranks": True,
            "student_gradient_finite": bool(torch.isfinite(student.grad).all().item()),
        }
    return {
        "passed": True,
        "local": local,
        "distributed": distributed,
        "formal_definition": {
            "pool_size": list(POOL_SIZE),
            "matrix_shape": [NUM_TOKENS, NUM_TOKENS],
            "diagonal": "kept for valid tokens",
            "reduction": "valid-pair masked sum / valid-pair count",
            "epsilon": RELATION_EPSILON,
            "teacher_target_detached": True,
        },
    }


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
        raise FileNotFoundError(
            "R2 requires the existing K-group shared initialization and will not "
            f"generate an R-specific replacement: {path}"
        )
    return _ORIGINAL_K_SHARED_INITIALIZATION(
        model, args, K_GROUP_OUTPUT_DIR, seed, rank, world_size
    )


def build_config_r2(
    args: argparse.Namespace,
    accumulation_steps: int,
    world_size: int,
    device: torch.device,
    shared_init_state_sha256: str,
    shared_init_file_sha256: str,
) -> Dict[str, object]:
    global _RELATION_SPEC, _REFERENCE_TESTS
    config = r1._ORIGINAL_K1_BUILD_CONFIG(
        args, accumulation_steps, world_size, device,
        shared_init_state_sha256, shared_init_file_sha256,
    )
    _RELATION_SPEC = _relation_spec_r2(args, accumulation_steps, world_size)
    _REFERENCE_TESTS = run_r2_reference_tests(device, world_size)
    # The reused epoch loop resolves these through dino_r1_server's globals.
    r1._RELATION_SPEC = _RELATION_SPEC
    r1._REFERENCE_TESTS = _REFERENCE_TESTS
    relation_hash = _canonical_sha256(_RELATION_SPEC)
    if not args.smoke_test:
        if world_size != 2:
            raise RuntimeError(f"Formal R2 requires world_size=2, got {world_size}")
        if _RELATION_SPEC["physical_relation_batch_size"] != 4:
            raise RuntimeError("Formal R2 requires physical relation batch size 4")
        if _RELATION_SPEC["effective_optimizer_batch_size"] != 8:
            raise RuntimeError("Formal R2 requires effective optimizer batch size 8")
    config.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "server_entry_point": str(Path(__file__).resolve()),
            "formal_seeds": list(FORMAL_SEEDS),
            "run_directory_name": (
                f"seed_{args.seed}_lambda_{_lambda_path_component(args.lambda_r2)}"
            ),
            "lambda_directory_component": _lambda_path_component(args.lambda_r2),
            "shared_initialization": {
                "source_group": "K_MobileNetV2_RASPP_server",
                "path": str(k0._shared_init_path(K_GROUP_OUTPUT_DIR, args.seed).resolve()),
                "state_sha256": shared_init_state_sha256,
                "file_sha256": shared_init_file_sha256,
                "r_specific_initialization_created": False,
            },
            "relation": copy.deepcopy(_RELATION_SPEC),
            "relation_spec_sha256": relation_hash,
            "relation_reference_tests": copy.deepcopy(_REFERENCE_TESTS),
            "r0_gate": copy.deepcopy(r1._R0_GATE),
            "r1_gate": copy.deepcopy(_R1_GATE),
            "r1_reference": copy.deepcopy(_R1_GATE),
            "k1_reference_validation": copy.deepcopy(r1._K1_REFERENCE_VALIDATION),
        }
    )
    loss = dict(config.get("loss", {}))
    loss.update(
        {
            "relation_kd": True,
            "relation_r1": False,
            "relation_r2": True,
            "lambda_r1": 0.0,
            "lambda_r2": float(args.lambda_r2),
            "total": "CE + warmup * (lambda_feat * feature + lambda_r2 * R2)",
        }
    )
    config["loss"] = loss
    return config


def audit_shapes_r2(
    model: base.MobileNetV2RASPPStudent,
    device: torch.device,
    height: int,
    width: int,
    amp_enabled: bool,
) -> Dict[str, object]:
    audit = r1._ORIGINAL_K1_AUDIT_SHAPES(model, device, height, width, amp_enabled)
    audit["experiment"] = EXPERIMENT
    audit["relation"] = {
        "enabled": True,
        "type": "R2_within_image_spatial",
        "native_teacher_student_taps": list(a0.A0_LAYER_ORDER),
        "a0_projection_used_only_by_pointwise_feature_anchor": True,
        "pool_size": list(POOL_SIZE),
        "token_count": NUM_TOKENS,
        "matrix_shape": [NUM_TOKENS, NUM_TOKENS],
        "matrix_dtype": "float32",
    }
    return audit


def build_best_checkpoint_r2(*args: Any, **kwargs: Any) -> Dict[str, object]:
    payload = r1._ORIGINAL_K1_BUILD_BEST_CHECKPOINT(*args, **kwargs)
    payload.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "relation": copy.deepcopy(_RELATION_SPEC),
            "relation_spec_sha256": None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC),
            "r0_gate": copy.deepcopy(r1._R0_GATE),
            "r1_gate": copy.deepcopy(_R1_GATE),
            "r1_reference": copy.deepcopy(_R1_GATE),
        }
    )
    return payload


def _patched_torch_save_atomic_r2(payload: object, path: Path) -> None:
    if isinstance(payload, Mapping) and payload.get("artifact_type") == ARTIFACT_TYPE:
        payload = dict(payload)
        payload.update(
            {
                "experiment": EXPERIMENT,
                "experiment_group": EXPERIMENT_GROUP,
                "relation": copy.deepcopy(_RELATION_SPEC),
                "relation_spec_sha256": None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC),
                "r0_gate": copy.deepcopy(r1._R0_GATE),
                "r1_gate": copy.deepcopy(_R1_GATE),
                "r1_reference": copy.deepcopy(_R1_GATE),
                "hashes": {
                    **dict(payload.get("hashes", {})),
                    **k1._resource_hashes(),
                    "relation_spec_sha256": None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC),
                    "r1_gate_metrics_sha256": None if _R1_GATE is None else _R1_GATE.get("metrics_sha256"),
                    "r1_reference_metrics_sha256": None if _R1_GATE is None else _R1_GATE.get("metrics_sha256"),
                    "r2_training_script_sha256": common.sha256_file(Path(__file__).resolve()),
                },
            }
        )
        payload["pca_parameters_sha256_record"] = copy.deepcopy(
            k1._PCA_PARAMETER_RECORD
        )
    k1._ORIGINAL_TORCH_SAVE_ATOMIC(payload, path)


def _patched_evaluate_r2(*args: Any, **kwargs: Any):
    split_name = kwargs.get("split_name")
    if isinstance(split_name, str):
        kwargs["split_name"] = split_name.replace("K0", EXPERIMENT).replace("K1", EXPERIMENT)
    return k1._ORIGINAL_EVALUATE(*args, **kwargs)


def _r2_print(*values: object, **kwargs: object) -> None:
    adjusted = tuple(
        value.replace("K0", EXPERIMENT).replace("K1", EXPERIMENT)
        if isinstance(value, str) else value
        for value in values
    )
    builtins.print(*adjusted, **kwargs)


def _r2_tqdm(*args: Any, **kwargs: Any):
    description = kwargs.get("desc")
    if isinstance(description, str):
        kwargs["desc"] = description.replace("K1", EXPERIMENT).replace("R1", EXPERIMENT)
    return r1._ORIGINAL_TQDM(*args, **kwargs)


def _rename_relation_record_r2(record: Mapping[str, object]) -> Dict[str, object]:
    renamed = copy.deepcopy(dict(record))
    scalar_aliases = {
        "lambda_r1": "lambda_r2",
        "relation_r1_loss_raw": "relation_r2_loss_raw",
        "relation_r1_loss": "relation_r2_loss",
        "relation_r1_loss_by_layer": "relation_r2_loss_by_layer",
        "r1_base_k1_equivalence": "r2_base_k1_equivalence",
    }
    for old, new in scalar_aliases.items():
        if old in renamed:
            renamed[new] = renamed.pop(old)
    gradient_aliases = {
        "grad_l2_relation_r1_os4": "grad_l2_relation_r2_os4",
        "grad_l2_relation_r1_os8": "grad_l2_relation_r2_os8",
        "grad_l2_relation_r1_os16": "grad_l2_relation_r2_os16",
        "grad_l2_relation_r1": "grad_l2_relation_r2",
    }
    for old, new in gradient_aliases.items():
        if old in renamed:
            renamed[new] = renamed.pop(old)
        std_key = f"{old}_sample_std"
        if std_key in renamed:
            renamed[f"{new}_sample_std"] = renamed.pop(std_key)
    layers = renamed.get("layers")
    if isinstance(layers, Mapping):
        renamed["layers"] = {
            str(layer): _rename_relation_record_r2(values)
            if isinstance(values, Mapping)
            else values
            for layer, values in layers.items()
        }
    per_rank = renamed.get("per_rank")
    if isinstance(per_rank, Sequence) and not isinstance(per_rank, (str, bytes)):
        renamed["per_rank"] = [
            _rename_relation_record_r2(value) if isinstance(value, Mapping) else value
            for value in per_rank
        ]
    renamed["relation_type"] = "R2_within_image_spatial"
    return renamed


def _aggregate_gradient_record_r2(local_record: Dict[str, object], world_size: int) -> Dict[str, object]:
    record = _ORIGINAL_AGGREGATE_GRADIENT_RECORD(local_record, world_size)
    return _rename_relation_record_r2(record)


def train_one_epoch_r2(*args: Any, **kwargs: Any):
    """Reuse R1's audited loop while exposing an R2-native artifact schema."""

    global _COLLECT_TRAIN_RELATION_STATS, _TRAIN_RELATION_STATS
    _TRAIN_RELATION_STATS = {"calls": 0.0, "valid_tokens": 0.0, "valid_pairs": 0.0}
    _COLLECT_TRAIN_RELATION_STATS = True
    try:
        metrics, steps, gradient_records, first_batch = _ORIGINAL_TRAIN_ONE_EPOCH_R1(
            *args, **kwargs
        )
    finally:
        _COLLECT_TRAIN_RELATION_STATS = False

    if isinstance(r1._FIRST_BATCH_BASE_EQUIVALENCE, Mapping):
        equivalence = copy.deepcopy(dict(r1._FIRST_BATCH_BASE_EQUIVALENCE))
        equivalence["comparison"] = (
            "R2 independent first-batch invariant fields versus locked K1; "
            "seed-specific data and loss fields are informational"
        )
        rows = equivalence.get("per_rank")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            for row in rows:
                if isinstance(row, dict):
                    row["comparison"] = (
                        "R2 independent first-batch invariant fields versus locked K1"
                    )
        r1._FIRST_BATCH_BASE_EQUIVALENCE = equivalence

    calls = max(_TRAIN_RELATION_STATS.get("calls", 0.0), 1.0)
    metrics["loss_schema"] = "hard_label_CE_plus_A0_feature_MSE_plus_R2_relation_MSE"
    metrics["relation_r2_loss"] = metrics.pop("relation_r1_loss", None)
    metrics["relation_r1_loss"] = None
    metrics["relation_valid_token_count"] = _TRAIN_RELATION_STATS.get(
        "valid_tokens", 0.0
    ) / calls
    metrics["relation_valid_pair_count"] = _TRAIN_RELATION_STATS.get(
        "valid_pairs", 0.0
    ) / calls
    metrics.pop("relation_valid_pair_count_nominal", None)
    metrics["relation_count_reduction"] = (
        "mean per physical micro-batch after equal OS=4/8/16 layer aggregation"
    )
    gradient_records = [
        _rename_relation_record_r2(record) for record in gradient_records
    ]
    if first_batch is not None:
        first_batch = _rename_relation_record_r2(first_batch)
    return metrics, steps, gradient_records, first_batch


def _update_relation_stop_gate_r2(record: Mapping[str, object]) -> None:
    global _RELATION_GATE_CONSECUTIVE_EXCESS
    ratio = record.get("relation_to_ce_effective_ratio_os16")
    if ratio is not None and float(ratio) > GRADIENT_CE_STOP_RATIO:
        _RELATION_GATE_CONSECUTIVE_EXCESS += 1
    else:
        _RELATION_GATE_CONSECUTIVE_EXCESS = 0
    if _RELATION_GATE_CONSECUTIVE_EXCESS >= GRADIENT_CE_STOP_CONSECUTIVE:
        raise RuntimeError(
            "R2 effective relation gradient exceeded 2x CE for three consecutive "
            "gradient records; inspect mask/reduction/lambda"
        )


def _restore_relation_gate_state_r2(args: argparse.Namespace) -> None:
    global _RELATION_GATE_CONSECUTIVE_EXCESS
    _RELATION_GATE_CONSECUTIVE_EXCESS = 0
    if not args.resume:
        return
    path = r2_paths(args.output_dir, args.seed, args.lambda_r2)["gradient_norms"]
    if not path.is_file():
        return
    for value in [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][-GRADIENT_CE_STOP_CONSECUTIVE:]:
        ratio = value.get("relation_to_ce_effective_ratio_os16")
        if ratio is not None and float(ratio) > GRADIENT_CE_STOP_RATIO:
            _RELATION_GATE_CONSECUTIVE_EXCESS += 1
        else:
            _RELATION_GATE_CONSECUTIVE_EXCESS = 0


def _existing_first_batch_equivalence_r2(args: argparse.Namespace) -> Optional[Dict[str, object]]:
    path = r2_paths(args.output_dir, args.seed, args.lambda_r2)["first_batch_audit"]
    if not path.is_file():
        return None
    audit = _read_json(path)
    rows = audit.get("per_rank")
    if not isinstance(rows, Sequence):
        return None
    values = [
        dict(row["r2_base_k1_equivalence"])
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("r2_base_k1_equivalence"), Mapping)
    ]
    if not values:
        return None
    return {
        "passed": all(bool(value.get("passed")) for value in values),
        "world_size": len(values),
        "comparison": "restored from existing R2 first_batch_audit.json",
        "per_rank": values,
    }


def _postprocess_metrics_r2(args: argparse.Namespace) -> None:
    r1._ORIGINAL_K1_POSTPROCESS(args)
    if int(os.environ.get("RANK", "0")) != 0:
        return
    metrics_path = r2_paths(args.output_dir, args.seed, args.lambda_r2)["metrics"]
    results = _read_json(metrics_path)
    relation_spec = _RELATION_SPEC
    if relation_spec is None:
        config = results.get("config", {})
        if isinstance(config, Mapping) and isinstance(config.get("relation"), Mapping):
            relation_spec = dict(config["relation"])
    relation_hash = None if relation_spec is None else _canonical_sha256(relation_spec)
    r2_best = results.get("best_dev_metrics", {})
    # R0 and R1 are optional paired comparisons in independent R2 mode.  Do
    # not make malformed or missing baseline artifacts abort post-processing.
    r0_reference = r1._R0_GATE or _read_optional_r0_reference(args)
    r1_reference = _R1_GATE or _read_optional_r1_reference(args)
    r2_miou = float(r2_best["mIoU"]) if isinstance(r2_best, Mapping) and "mIoU" in r2_best else None
    r0_miou_value = r0_reference.get("r0_best_dev_mIoU")
    r0_miou = float(r0_miou_value) if r0_miou_value is not None else None
    r1_miou_value = r1_reference.get("r1_best_dev_mIoU")
    r1_miou = float(r1_miou_value) if r1_miou_value is not None else None

    results.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "protocol": (
                "R2 independent relation-KD run: K1-compatible seed-specific shared "
                "initialization, hard-label CE, locked A0 feature MSE, and native "
                "teacher/student 8x16 masked within-image token cosine relation MSE; "
                "no R0/R1 checkpoint or loss dependency, 4000-step auxiliary warm-up, "
                "fixed 80k budget, dev_local selection, and no test_local evaluation."
            ),
            "relation": copy.deepcopy(relation_spec),
            "relation_spec_sha256": relation_hash,
            "relation_reference_tests": copy.deepcopy(_REFERENCE_TESTS),
            "r0_gate": copy.deepcopy(r0_reference),
            "r1_gate": copy.deepcopy(_R1_GATE),
            "r1_reference": copy.deepcopy(r1_reference),
            "r2_first_batch_base_equivalence": copy.deepcopy(
                r1._FIRST_BATCH_BASE_EQUIVALENCE or _existing_first_batch_equivalence_r2(args)
            ),
            "r2_vs_r0": {
                "R2_mIoU": r2_miou,
                "R0_mIoU": r0_miou,
                "delta_R2_minus_R0": None if r2_miou is None or r0_miou is None else r2_miou - r0_miou,
            },
            "r2_vs_r1": {
                "R2_mIoU": r2_miou,
                "R1_mIoU": r1_miou,
                "delta_R2_minus_R1": None if r2_miou is None or r1_miou is None else r2_miou - r1_miou,
            },
            "physical_relation_batch_size": None if relation_spec is None else relation_spec.get("physical_relation_batch_size"),
            "effective_optimizer_batch_size": None if relation_spec is None else relation_spec.get("effective_optimizer_batch_size"),
            "test_local_evaluated": False,
        }
    )
    loss = results.get("loss")
    if isinstance(loss, dict):
        loss.update({"relation_kd": True, "relation_r1": False, "relation_r2": True, "lambda_r1": 0.0, "lambda_r2": float(args.lambda_r2)})
    gradient_path = r2_paths(
        args.output_dir, args.seed, args.lambda_r2
    )["gradient_norms"]
    results["gradient_gate"] = r1._read_gradient_gate_summary(gradient_path)
    results["hashes"] = {
        **dict(results.get("hashes", {})),
        "relation_spec_sha256": relation_hash,
        "r1_gate_metrics_sha256": None if _R1_GATE is None else _R1_GATE.get("metrics_sha256"),
        "r1_reference_metrics_sha256": None if r1_reference is None else r1_reference.get("metrics_sha256"),
        "r2_training_script_sha256": common.sha256_file(Path(__file__).resolve()),
    }
    common.write_json_atomic(metrics_path, results)


def smoke_test_r2(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    rank: int,
) -> None:
    global _REFERENCE_TESTS
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    _REFERENCE_TESTS = run_r2_reference_tests(device, world_size)
    teacher, projection = k1._require_resources()
    args = k1._ACTIVE_ARGS
    model.train()
    teacher.eval()
    projection.eval()
    images, targets, paths = next(iter(loader))
    images, targets = images.to(device), targets.to(device)
    model.zero_grad(set_to_none=True)
    with common.autocast_context(device, amp_enabled):
        output = model(images)
        if not isinstance(output, Mapping):
            raise RuntimeError("R2 smoke forward did not expose features")
        with torch.no_grad():
            teacher_features = teacher.extract_features(images)
        student_features = output["features"]
        projected = {layer: projection[layer](teacher_features[layer].detach()) for layer in a0.A0_LAYER_ORDER}
        feature_loss = sum(F.mse_loss(student_features[layer].float(), projected[layer].float()) for layer in a0.A0_LAYER_ORDER) / 3
        relation_loss, _, relation_audit = r2_relation_losses(student_features, teacher_features, targets, world_size)
    logits = output["logits"].float()
    valid_pixels = int((targets != common.IGNORE_INDEX).sum().item())
    if valid_pixels <= 0:
        raise RuntimeError("R2 smoke batch contains no valid pixels")
    ce = F.cross_entropy(logits, targets, ignore_index=common.IGNORE_INDEX, reduction="sum") / valid_pixels
    warmup = 1.0 / r1._warmup_steps(args)
    total = ce + warmup * (args.lambda_feat * feature_loss + args.lambda_r2 * relation_loss)
    relation_grad = torch.autograd.grad(relation_loss, student_features["os16"], retain_graph=True)[0]
    total.backward()
    if not all(bool(torch.isfinite(value).all().item()) for value in (ce, feature_loss, relation_loss, total)):
        raise RuntimeError("R2 smoke test produced a non-finite loss")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("R2 smoke test found a teacher gradient")
    if rank == 0:
        _r2_print(
            f"[OK] R2 server smoke: sample={paths[0]}, logits={tuple(logits.shape)}, "
            f"CE={ce.item():.6f}, feature={feature_loss.item():.6f}, "
            f"R2={relation_loss.item():.6f}, total={total.item():.6f}, "
            f"valid_tokens={relation_audit['valid_token_count']}, "
            f"valid_pairs={relation_audit['valid_pair_count']}, "
            f"relation_grad_os16={float(relation_grad.float().norm().item()):.6e}"
        )


def run_training(args: argparse.Namespace) -> None:
    global _R1_GATE, _RELATION_SPEC, _REFERENCE_TESTS
    # R2 has no R1 checkpoint/loss dependency.  Keep a provenance record when
    # an R1 metrics file exists, but never block an independent R2 launch.
    _R1_GATE = _read_optional_r1_reference(args)
    _RELATION_SPEC = None
    _REFERENCE_TESTS = None

    saved = {
        "__file__": r1.__file__,
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
        "_ensure_locked_k_shared_initialization": r1._ensure_locked_k_shared_initialization,
        "_aggregate_gradient_record": r1._aggregate_gradient_record,
        "_update_relation_stop_gate": r1._update_relation_stop_gate,
        "_restore_relation_gate_state": r1._restore_relation_gate_state,
        "_existing_first_batch_equivalence": r1._existing_first_batch_equivalence,
        "_validate_r0_gate": r1._validate_r0_gate,
        "_compare_first_batch_base_to_k1": r1._compare_first_batch_base_to_k1,
    }
    saved_r0_validate_k1_reference = r0._validate_k1_reference
    try:
        # R2 independent mode still validates the locked K1 code/config
        # contract, but permits a valid shared initialization for the selected
        # seed instead of comparing every seed to K1's seed-42 file hash.
        r0._validate_k1_reference = _validate_k1_reference_r2
        # R0 is a useful baseline for paired reporting, not a launch
        # prerequisite for an independent R2 run.
        r1._validate_r0_gate = _read_optional_r0_reference
        # A seed-independent R2 audit checks protocol/resource invariants but
        # does not require seed-3407 images, pixels, or losses to equal the
        # locked K1 seed-42 first batch.
        r1._compare_first_batch_base_to_k1 = _compare_first_batch_base_r2
        r1.__file__ = str(Path(__file__).resolve())
        r1.EXPERIMENT = EXPERIMENT
        r1.ARTIFACT_TYPE = ARTIFACT_TYPE
        r1.r1_paths = r2_paths
        r1._relation_spec = _relation_spec_r2
        r1.r1_relation_losses = r2_relation_losses
        r1.run_relation_reference_tests = run_r2_reference_tests
        r1.build_config_r1 = build_config_r2
        r1.build_best_checkpoint_r1 = build_best_checkpoint_r2
        # R1's audited loop is parameterised by the relation function and the
        # lambda_r1 alias installed by parse_args.  The thin wrapper only
        # converts emitted field names and adds R2 valid-token statistics.
        r1.train_one_epoch_r1 = train_one_epoch_r2
        r1.smoke_test_r1 = smoke_test_r2
        r1._postprocess_metrics_r1 = _postprocess_metrics_r2
        r1.audit_shapes_r1 = audit_shapes_r2
        r1._patched_torch_save_atomic_r1 = _patched_torch_save_atomic_r2
        r1._patched_evaluate_r1 = _patched_evaluate_r2
        r1._r1_print = _r2_print
        r1._r1_tqdm = _r2_tqdm
        r1._ensure_locked_k_shared_initialization = _ensure_locked_k_shared_initialization
        r1._aggregate_gradient_record = _aggregate_gradient_record_r2
        r1._update_relation_stop_gate = _update_relation_stop_gate_r2
        r1._restore_relation_gate_state = _restore_relation_gate_state_r2
        r1._existing_first_batch_equivalence = _existing_first_batch_equivalence_r2
        r1.run_training(args)
    finally:
        r0._validate_k1_reference = saved_r0_validate_k1_reference
        for name, value in saved.items():
            setattr(r1, name, value)
        _RELATION_SPEC = None
        _REFERENCE_TESTS = None


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
