"""R0 server entry point: controlled reproduction of K1 feature KD.

R0 is the no-relation anchor for the Cityscapes R-group experiments.  Its
optimization trajectory must be the locked K1 trajectory:

    L = L_seg + warmup(step) * 1.0 * mean_l MSE(f_s^l, PCA_l(f_t^l))

The implementation deliberately reuses ``dino_k1_server`` instead of
forking its training loop.  This entry point only adds R-group isolation and
auditing:

* output goes to ``result/R_MobileNetV2_RASPP_server/R0/``;
* the student is loaded from the existing K-group shared initialization;
* no R-specific initialization may be created;
* relation losses are explicitly disabled and recorded as such;
* the first formal batch is compared rank-by-rank with K1 seed 42;
* the final dev mIoU is checked against the locked K1 reference.

Typical two-GPU server command::

    torchrun --standalone --nproc_per_node=2 dino_r0_server.py \
        --seed 42 --batch-size 2 --global-batch-size 8 \
        --num-workers 8 --multiprocessing-context spawn \
        --no-pin-memory --persistent-workers

Windows/local functional smoke (does not replace Linux two-GPU DDP smoke)::

    python -B dino_r0_server.py --device cuda --smoke-test \
        --batch-size 1 --global-batch-size 1 --num-workers 0 \
        --no-persistent-workers --no-pin-memory --no-amp
"""

from __future__ import annotations

import builtins
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import dino as common
import dino_a0_server as a0
import dino_k0_server as k0
import dino_k1_server as k1
import dino_s2_0 as base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "R_MobileNetV2_RASPP_server"
K_GROUP_OUTPUT_DIR = SCRIPT_DIR / "result" / "K_MobileNetV2_RASPP_server"
K1_REFERENCE_DIR = K_GROUP_OUTPUT_DIR / "K1" / "seed_42"

EXPERIMENT = "R0"
EXPERIMENT_GROUP = "R_MobileNetV2_RASPP_server"
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_r0_k1_reproduction"
ARTIFACT_FORMAT_VERSION = 1
FORMAL_SEEDS = (42,)

RELATION_EPSILON = 1e-6
K1_REFERENCE_MIOU = 0.522120045088882
K1_MIOU_SAMPLE_STD = 0.00219
FIRST_BATCH_ABS_TOLERANCE = 1e-6
FIRST_BATCH_REL_TOLERANCE = 1e-6


_ORIGINAL_K1_BUILD_CONFIG = k1.build_config
_ORIGINAL_K1_BUILD_BEST_CHECKPOINT = k1.build_best_checkpoint
_ORIGINAL_K1_TRAIN_ONE_EPOCH = k1.train_one_epoch_k1
_ORIGINAL_K1_SMOKE_TEST = k1.smoke_test_k1
_ORIGINAL_K1_POSTPROCESS = k1._postprocess_metrics
_ORIGINAL_K1_AUDIT_SHAPES = k1.audit_k1_shapes
_ORIGINAL_K_SHARED_INITIALIZATION = k1._ORIGINAL_ENSURE_SHARED_INITIALIZATION
_ORIGINAL_TQDM = k1.tqdm

_K1_REFERENCE: Optional[Dict[str, object]] = None
_REFERENCE_VALIDATION: Optional[Dict[str, object]] = None
_RELATION_SPEC: Optional[Dict[str, object]] = None
_FIRST_BATCH_EQUIVALENCE: Optional[Dict[str, object]] = None


def parse_args() -> Any:
    """Reuse K1's CLI while changing the default output and locking R0."""

    saved_default = k1.DEFAULT_OUTPUT_DIR
    saved_argparse = k1.argparse

    class R0ArgparseProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(saved_argparse, name)

        @staticmethod
        def ArgumentParser(*parser_args: Any, **parser_kwargs: Any):
            parser_kwargs["description"] = (
                "R0 MobileNetV2+R-ASPP controlled reproduction of K1: hard-label "
                "CE plus the locked A0 fixed StandardScaler+PCA feature target, "
                "with no relation loss."
            )
            return saved_argparse.ArgumentParser(*parser_args, **parser_kwargs)

    k1.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    k1.argparse = R0ArgparseProxy()
    try:
        args = k1.parse_args()
    finally:
        k1.DEFAULT_OUTPUT_DIR = saved_default
        k1.argparse = saved_argparse

    if args.seed != 42:
        raise SystemExit("R0 is pre-registered for --seed 42")
    if not args.smoke_test and args.max_steps != 80_000:
        raise SystemExit("Formal R0 is locked to exactly 80,000 optimizer steps")
    if not args.smoke_test and args.eval_every_steps != 5_000:
        raise SystemExit("Formal R0 is locked to --eval-every-steps 5000")
    if not args.smoke_test and args.gradient_log_steps != 500:
        raise SystemExit("Formal R0 is locked to --gradient-log-steps 500")
    if args.output_dir.resolve() == K_GROUP_OUTPUT_DIR.resolve():
        raise SystemExit(
            "R0 output must not point at the K-group directory; use the separate "
            "R_MobileNetV2_RASPP_server output root"
        )
    return args


def r0_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    """Use K0/K1's artifact names below the independent R0 directory."""

    original = k1._ORIGINAL_K0_PATHS(output_dir, seed)
    run_dir = output_dir.resolve() / EXPERIMENT / f"seed_{seed}"
    return {
        key: run_dir if key == "run_dir" else run_dir / value.name
        for key, value in original.items()
    }


def _canonical_sha256(value: Mapping[str, object]) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _relation_spec(
    args: Any, accumulation_steps: int, world_size: int
) -> Dict[str, object]:
    """Record the common R-group relation contract, disabled for R0."""

    physical_batch = int(args.batch_size) * int(world_size)
    effective_batch = physical_batch * int(accumulation_steps)
    return {
        "enabled": False,
        "active_relation_types": [],
        "relation_feature_source": {
            "teacher": "native OS=4/8/16 features (reserved; unused by R0)",
            "student": "native OS=4/8/16 features (reserved; unused by R0)",
            "a0_projected_features_used_for_relation": False,
        },
        "epsilon": RELATION_EPSILON,
        "physical_relation_batch_size": physical_batch,
        "effective_optimizer_batch_size": effective_batch,
        "mask_policy": "targets != 255; inactive because R0 has no relation loss",
        "r1": {
            "enabled": False,
            "representation": "masked GAP then BxB signed cosine matrix",
            "diagonal_policy": "keep",
            "reduction": "mean over all B^2 entries",
            "lambda": 0.0,
        },
        "r2": {
            "enabled": False,
            "pool_size": [8, 16],
            "representation": "per-image 128x128 signed token cosine matrix",
            "diagonal_policy": "keep for valid tokens",
            "reduction": "sum squared error divided by valid-pair count",
            "lambda": 0.0,
        },
        "relation_warmup_steps": 4_000,
        "relation_gradient_gate": None,
        "reason": "R0 is the CE+A0-feature-KD anchor and contains no relation term",
    }


def _reference_paths() -> Dict[str, Path]:
    return {
        "config": K1_REFERENCE_DIR / "config.json",
        "first_batch": K1_REFERENCE_DIR / "first_batch_audit.json",
        "metrics": K1_REFERENCE_DIR / "metrics.json",
    }


def _read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required K1 reference artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _same_value(actual: object, expected: object) -> bool:
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    return actual == expected


def _validate_formal_args_against_k1(
    args: Any, reference_config: Mapping[str, object]
) -> List[str]:
    checks = {
        "max_steps": "max_optimizer_steps",
        "batch_size": "batch_size_per_gpu",
        "global_batch_size": "global_batch_size",
        "eval_batch_size": "eval_batch_size",
        "num_workers": "num_workers_per_gpu",
        "multiprocessing_context": "multiprocessing_context",
        "pin_memory": "pin_memory",
        "persistent_workers": "persistent_workers",
        "prefetch_factor": "prefetch_factor",
        "lr": "learning_rate",
        "momentum": "momentum",
        "weight_decay": "weight_decay",
        "poly_power": "poly_power",
        "min_lr_ratio": "min_lr_ratio",
        "eval_every_steps": "eval_every_steps",
        "head_channels": "head_channels",
        "dropout": "dropout",
        "amp": "amp",
        "deterministic": "deterministic",
    }
    mismatches: List[str] = []
    for arg_name, config_name in checks.items():
        actual = getattr(args, arg_name)
        expected = reference_config.get(config_name)
        if not _same_value(actual, expected):
            mismatches.append(f"{arg_name}: actual={actual!r}, K1={expected!r}")

    compound_checks = {
        "crop_size": [args.crop_height, args.crop_width],
        "random_scale": [args.scale_min, args.scale_max],
    }
    for name, actual in compound_checks.items():
        expected = reference_config.get(name)
        if actual != expected:
            mismatches.append(f"{name}: actual={actual!r}, K1={expected!r}")
    if args.boundary_tolerance != 2:
        mismatches.append(
            f"boundary_tolerance: actual={args.boundary_tolerance!r}, K1=2"
        )
    return mismatches


def _validate_k1_reference(
    args: Any,
    allow_seed_specific_shared_init: bool = False,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Validate the locked K1 reference and the selected seed initialization.

    The original R0/R1 protocol compares the selected shared-init file with
    the seed-42 K1 reference hash.  R2 independent mode keeps all K1 protocol
    and code checks, but permits a pre-generated shared initialization for a
    different formal seed; that file is then verified by its own sidecar and
    embedded seed/model/state hashes.
    """
    paths = _reference_paths()
    reference = {
        "config": _read_json(paths["config"]),
        "first_batch": _read_json(paths["first_batch"]),
        "metrics": _read_json(paths["metrics"]),
    }
    config = reference["config"]
    metrics = reference["metrics"]
    assert isinstance(config, Mapping)
    assert isinstance(metrics, Mapping)

    failures: List[str] = []
    if config.get("experiment") != "K1" or metrics.get("experiment") != "K1":
        failures.append("reference artifacts are not K1")
    reference_miou = float(
        metrics.get("best_dev_metrics", {}).get("mIoU", float("nan"))  # type: ignore[union-attr]
    )
    if not math.isclose(reference_miou, K1_REFERENCE_MIOU, rel_tol=0.0, abs_tol=1e-12):
        failures.append(
            f"K1 reference mIoU changed: actual={reference_miou}, "
            f"expected={K1_REFERENCE_MIOU}"
        )

    hashes = metrics.get("hashes", {})
    if not isinstance(hashes, Mapping):
        failures.append("K1 metrics has no hash mapping")
        hashes = {}
    code_checks = {
        "training_script_sha256": common.sha256_file(Path(k1.__file__).resolve()),
        "k0_shared_training_runner_sha256": common.sha256_file(
            Path(k0.__file__).resolve()
        ),
        "shared_student_module_sha256": common.sha256_file(Path(base.__file__).resolve()),
        "common_module_sha256": common.sha256_file(Path(common.__file__).resolve()),
    }
    for key, actual in code_checks.items():
        expected = hashes.get(key)
        if actual != expected:
            failures.append(f"{key}: local={actual}, K1={expected}")

    shared_init = k0._shared_init_path(K_GROUP_OUTPUT_DIR, args.seed)
    shared_init_matches_reference = None
    shared_init_file_hash = None
    if not shared_init.is_file():
        failures.append(f"locked K shared init is missing: {shared_init}")
    else:
        shared_file_hash = common.verify_checkpoint_sidecar(shared_init)
        shared_init_file_hash = shared_file_hash
        shared_init_matches_reference = (
            shared_file_hash == hashes.get("student_init_file_sha256")
        )
        if allow_seed_specific_shared_init:
            try:
                payload = common.safe_torch_load(
                    shared_init, map_location="cpu", weights_only=True
                )
                k0._validate_shared_init_payload(payload, args, args.seed)
            except Exception as exc:
                failures.append(
                    "seed-specific K shared init failed payload validation: "
                    f"{type(exc).__name__}: {exc}"
                )
        elif not shared_init_matches_reference:
            failures.append(
                "K shared-init file hash differs from the K1 reference: "
                f"local={shared_file_hash}, K1={hashes.get('student_init_file_sha256')}"
            )

    if not args.smoke_test:
        failures.extend(_validate_formal_args_against_k1(args, config))

    validation = {
        "passed": not failures,
        "failures": failures,
        "reference_directory": str(K1_REFERENCE_DIR.resolve()),
        "reference_metrics_sha256": common.sha256_file(paths["metrics"]),
        "reference_first_batch_sha256": common.sha256_file(paths["first_batch"]),
        "reference_mIoU": reference_miou,
        "mIoU_tolerance": K1_MIOU_SAMPLE_STD,
        "run_seed": int(args.seed),
        "shared_init_path": str(shared_init.resolve()),
        "shared_init_file_sha256": shared_init_file_hash,
        "reference_shared_init_file_sha256": hashes.get("student_init_file_sha256"),
        "shared_init_matches_k1_reference": shared_init_matches_reference,
        "seed_specific_shared_init_allowed": allow_seed_specific_shared_init,
        "local_k1_runner_sha256": code_checks["training_script_sha256"],
        "local_k0_runner_sha256": code_checks["k0_shared_training_runner_sha256"],
    }
    if failures:
        raise RuntimeError(
            "R0 cannot start because the locked K1 reference is incompatible:\n- "
            + "\n- ".join(failures)
        )
    return reference, validation


def _ensure_locked_k_shared_initialization(
    model: base.MobileNetV2RASPPStudent,
    args: Any,
    _r_output_dir: Path,
    seed: int,
    rank: int,
    world_size: int,
) -> Tuple[str, str, Path]:
    """Load the existing K initialization and refuse to create an R copy."""

    path = k0._shared_init_path(K_GROUP_OUTPUT_DIR, seed)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(
            "R0 requires the existing K-group shared initialization and will not "
            f"generate a replacement: {path}"
        )
    return _ORIGINAL_K_SHARED_INITIALIZATION(
        model, args, K_GROUP_OUTPUT_DIR, seed, rank, world_size
    )


def build_config_r0(
    args: Any,
    accumulation_steps: int,
    world_size: int,
    device: torch.device,
    shared_init_state_sha256: str,
    shared_init_file_sha256: str,
) -> Dict[str, object]:
    global _RELATION_SPEC

    config = _ORIGINAL_K1_BUILD_CONFIG(
        args,
        accumulation_steps,
        world_size,
        device,
        shared_init_state_sha256,
        shared_init_file_sha256,
    )
    _RELATION_SPEC = _relation_spec(args, accumulation_steps, world_size)
    relation_hash = _canonical_sha256(_RELATION_SPEC)

    if not args.smoke_test:
        if world_size != 2:
            raise RuntimeError(f"Formal R0 requires world_size=2, got {world_size}")
        if _RELATION_SPEC["physical_relation_batch_size"] != 4:
            raise RuntimeError("Formal R0 requires physical_relation_batch_size=4")
        if _RELATION_SPEC["effective_optimizer_batch_size"] != 8:
            raise RuntimeError("Formal R0 requires effective_optimizer_batch_size=8")

    config["experiment"] = EXPERIMENT
    config["experiment_group"] = EXPERIMENT_GROUP
    config["artifact_type"] = ARTIFACT_TYPE
    config["server_entry_point"] = str(Path(__file__).resolve())
    config["formal_seeds"] = list(FORMAL_SEEDS)
    config["shared_initialization"] = {
        "source_group": "K_MobileNetV2_RASPP_server",
        "path": str(k0._shared_init_path(K_GROUP_OUTPUT_DIR, args.seed).resolve()),
        "state_sha256": shared_init_state_sha256,
        "file_sha256": shared_init_file_sha256,
        "r_specific_initialization_created": False,
    }
    config["relation"] = copy.deepcopy(_RELATION_SPEC)
    config["relation_spec_sha256"] = relation_hash
    config["r0_k1_equivalence"] = {
        "reference_directory": str(K1_REFERENCE_DIR.resolve()),
        "reference_mIoU": K1_REFERENCE_MIOU,
        "final_mIoU_abs_tolerance": K1_MIOU_SAMPLE_STD,
        "first_batch_abs_tolerance": FIRST_BATCH_ABS_TOLERANCE,
        "first_batch_rel_tolerance": FIRST_BATCH_REL_TOLERANCE,
        "required_before_relational_candidates": True,
    }
    return config


def audit_shapes_r0(
    model: base.MobileNetV2RASPPStudent,
    device: torch.device,
    height: int,
    width: int,
    amp_enabled: bool,
) -> Dict[str, object]:
    audit = _ORIGINAL_K1_AUDIT_SHAPES(
        model, device, height, width, amp_enabled
    )
    audit["experiment"] = EXPERIMENT
    audit["relation"] = {
        "enabled": False,
        "native_teacher_student_taps_reserved": list(a0.A0_LAYER_ORDER),
        "a0_projection_used_only_by_pointwise_feature_anchor": True,
    }
    return audit


def build_best_checkpoint_r0(*args: Any, **kwargs: Any) -> Dict[str, object]:
    payload = _ORIGINAL_K1_BUILD_BEST_CHECKPOINT(*args, **kwargs)
    payload["experiment"] = EXPERIMENT
    payload["experiment_group"] = EXPERIMENT_GROUP
    payload["artifact_type"] = ARTIFACT_TYPE
    payload["relation"] = copy.deepcopy(_RELATION_SPEC)
    payload["relation_spec_sha256"] = (
        None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC)
    )
    payload["r0_k1_first_batch_equivalence"] = copy.deepcopy(
        _FIRST_BATCH_EQUIVALENCE
    )
    return payload


def _patched_torch_save_atomic_r0(payload: object, path: Path) -> None:
    if isinstance(payload, Mapping) and payload.get("artifact_type") == ARTIFACT_TYPE:
        payload = dict(payload)
        payload["experiment"] = EXPERIMENT
        payload["experiment_group"] = EXPERIMENT_GROUP
        payload["relation"] = copy.deepcopy(_RELATION_SPEC)
        payload["relation_spec_sha256"] = (
            None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC)
        )
        payload["r0_k1_first_batch_equivalence"] = copy.deepcopy(
            _FIRST_BATCH_EQUIVALENCE
        )
        payload["hashes"] = {
            **dict(payload.get("hashes", {})),
            **k1._resource_hashes(),
            "relation_spec_sha256": payload["relation_spec_sha256"],
            "k1_reference_metrics_sha256": (
                None
                if _REFERENCE_VALIDATION is None
                else _REFERENCE_VALIDATION.get("reference_metrics_sha256")
            ),
            "r0_training_script_sha256": common.sha256_file(
                Path(__file__).resolve()
            ),
        }
        payload["pca_parameters_sha256_record"] = copy.deepcopy(
            k1._PCA_PARAMETER_RECORD
        )
    k1._ORIGINAL_TORCH_SAVE_ATOMIC(payload, path)


def _patched_evaluate_r0(*args: Any, **kwargs: Any):
    split_name = kwargs.get("split_name")
    if isinstance(split_name, str):
        kwargs["split_name"] = split_name.replace("K0", EXPERIMENT).replace(
            "K1", EXPERIMENT
        )
    return k1._ORIGINAL_EVALUATE(*args, **kwargs)


def _r0_print(*values: object, **kwargs: object) -> None:
    adjusted = tuple(
        value.replace("K0", EXPERIMENT).replace("K1", EXPERIMENT)
        if isinstance(value, str)
        else value
        for value in values
    )
    builtins.print(*adjusted, **kwargs)


def _r0_tqdm(*args: Any, **kwargs: Any):
    description = kwargs.get("desc")
    if isinstance(description, str):
        kwargs["desc"] = description.replace("K1", EXPERIMENT)
    return _ORIGINAL_TQDM(*args, **kwargs)


def _float_match(actual: object, expected: object) -> bool:
    try:
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=FIRST_BATCH_REL_TOLERANCE,
            abs_tol=FIRST_BATCH_ABS_TOLERANCE,
        )
    except (TypeError, ValueError):
        return False


def _reference_rank_row(rank: int) -> Mapping[str, object]:
    if _K1_REFERENCE is None:
        raise RuntimeError("K1 reference was not initialized")
    audit = _K1_REFERENCE["first_batch"]
    assert isinstance(audit, Mapping)
    rows = audit.get("per_rank")
    if not isinstance(rows, Sequence):
        raise RuntimeError("K1 first-batch reference has no per-rank rows")
    for row in rows:
        if isinstance(row, Mapping) and int(row.get("rank", -1)) == rank:
            return row
    raise RuntimeError(f"K1 first-batch reference has no rank {rank}")


def _compare_first_batch_to_k1(
    row: Mapping[str, object], rank: int
) -> Dict[str, object]:
    reference = _reference_rank_row(rank)
    exact_fields = (
        "paths",
        "image_tensor_shape",
        "target_tensor_shape",
        "image_tensor_sha256",
        "target_tensor_sha256",
        "valid_pixels",
        "student_feature_shapes",
        "teacher_feature_shapes",
        "projected_teacher_shapes",
        "teacher_checkpoint_sha256",
        "k0_shared_training_runner_sha256",
        "pca_parameter_record_sha256",
        "pca_parameter_sha256",
        "projection_parameter_sha256",
        "pca_sampling_manifest_sha256",
    )
    exact_mismatches = {
        field: {"actual": row.get(field), "expected": reference.get(field)}
        for field in exact_fields
        if row.get(field) != reference.get(field)
    }

    scalar_fields = ("feature_loss", "ce_loss", "warmup_weight")
    scalar_mismatches = {
        field: {"actual": row.get(field), "expected": reference.get(field)}
        for field in scalar_fields
        if not _float_match(row.get(field), reference.get(field))
    }
    layer_mismatches: Dict[str, object] = {}
    actual_layers = row.get("feature_loss_by_layer", {})
    reference_layers = reference.get("feature_loss_by_layer", {})
    if isinstance(actual_layers, Mapping) and isinstance(reference_layers, Mapping):
        for layer in a0.A0_LAYER_ORDER:
            if not _float_match(actual_layers.get(layer), reference_layers.get(layer)):
                layer_mismatches[layer] = {
                    "actual": actual_layers.get(layer),
                    "expected": reference_layers.get(layer),
                }
    else:
        layer_mismatches["schema"] = {
            "actual": actual_layers,
            "expected": reference_layers,
        }

    passed = not exact_mismatches and not scalar_mismatches and not layer_mismatches
    return {
        "rank": rank,
        "passed": passed,
        "reference": str(_reference_paths()["first_batch"].resolve()),
        "absolute_tolerance": FIRST_BATCH_ABS_TOLERANCE,
        "relative_tolerance": FIRST_BATCH_REL_TOLERANCE,
        "exact_mismatches": exact_mismatches,
        "scalar_mismatches": scalar_mismatches,
        "feature_layer_mismatches": layer_mismatches,
    }


def _relation_gradient_defaults(physical_batch_size: int) -> Dict[str, object]:
    return {
        "relation_enabled": False,
        "relation_r1_loss_raw": None,
        "relation_r2_loss_raw": None,
        "relation_loss_weighted": 0.0,
        "grad_l2_relation_r1_os4": None,
        "grad_l2_relation_r1_os8": None,
        "grad_l2_relation_r1_os16": None,
        "grad_l2_relation_r2_os4": None,
        "grad_l2_relation_r2_os8": None,
        "grad_l2_relation_r2_os16": None,
        "grad_l2_relation_effective_os16": 0.0,
        "relation_valid_token_count": None,
        "relation_valid_pair_count": None,
        "relation_physical_batch_size": physical_batch_size,
        "relation_finite": True,
    }


def train_one_epoch_r0(
    model: torch.nn.Module,
    loader: DataLoader,
    sampler: Optional[DistributedSampler],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    accumulation_steps: int,
    epoch: int,
    starting_optimizer_step: int,
    remaining_optimizer_steps: int,
    rank: int,
    world_size: int,
) -> Tuple[
    Dict[str, object],
    int,
    List[Dict[str, object]],
    Optional[Dict[str, object]],
]:
    global _FIRST_BATCH_EQUIVALENCE

    metrics, optimizer_steps, gradient_records, first_batch = (
        _ORIGINAL_K1_TRAIN_ONE_EPOCH(
            model=model,
            loader=loader,
            sampler=sampler,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            accumulation_steps=accumulation_steps,
            epoch=epoch,
            starting_optimizer_step=starting_optimizer_step,
            remaining_optimizer_steps=remaining_optimizer_steps,
            rank=rank,
            world_size=world_size,
        )
    )

    physical_batch_size = int(k1._ACTIVE_ARGS.batch_size) * world_size
    defaults = _relation_gradient_defaults(physical_batch_size)
    for record in gradient_records:
        record.update(defaults)
        record["relation_spec_sha256"] = (
            None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC)
        )
    metrics.update(
        {
            "relation_enabled": False,
            "relation_r1_loss": None,
            "relation_r2_loss": None,
            "relation_loss_weighted": 0.0,
            "relation_valid_token_count": None,
            "relation_valid_pair_count": None,
            "relation_physical_batch_size": physical_batch_size,
        }
    )

    if first_batch is not None:
        local_equivalence = _compare_first_batch_to_k1(first_batch, rank)
        gathered: List[Optional[Dict[str, object]]] = [None for _ in range(world_size)]
        if world_size > 1:
            dist.all_gather_object(gathered, local_equivalence)
        else:
            gathered[0] = local_equivalence
        global_pass = all(
            row is not None and bool(row.get("passed")) for row in gathered
        )
        _FIRST_BATCH_EQUIVALENCE = {
            "passed": global_pass,
            "world_size": world_size,
            "comparison": "R0 first formal batch versus locked K1 seed=42",
            "per_rank": gathered,
        }
        first_batch["experiment"] = EXPERIMENT
        first_batch["relation"] = {
            "enabled": False,
            "physical_batch_size": physical_batch_size,
            "r1_loss": None,
            "r2_loss": None,
        }
        first_batch["relation_spec_sha256"] = (
            None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC)
        )
        first_batch["r0_k1_equivalence"] = local_equivalence
        if not global_pass:
            raise RuntimeError(
                "R0 first-batch equivalence with K1 failed: "
                + json.dumps(gathered, ensure_ascii=False, sort_keys=True)
            )
    elif starting_optimizer_step == 0:
        raise RuntimeError("R0 did not receive a first-batch audit at step 0")

    return metrics, optimizer_steps, gradient_records, first_batch


def smoke_test_r0(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    rank: int,
) -> None:
    _ORIGINAL_K1_SMOKE_TEST(model, loader, device, amp_enabled, rank)
    if rank == 0:
        _r0_print(
            "[OK] R0 protocol smoke: K1 CE+fixed-A0-feature objective reused; "
            "R1/R2 relation losses are disabled; K shared init was loaded."
        )


def _existing_first_batch_equivalence(args: Any) -> Optional[Dict[str, object]]:
    path = r0_paths(args.output_dir, args.seed)["first_batch_audit"]
    if not path.is_file():
        return None
    audit = _read_json(path)
    rows = audit.get("per_rank")
    if not isinstance(rows, Sequence):
        return None
    per_rank = []
    for row in rows:
        if isinstance(row, Mapping):
            value = row.get("r0_k1_equivalence")
            if isinstance(value, Mapping):
                per_rank.append(dict(value))
    if not per_rank:
        return None
    return {
        "passed": all(bool(row.get("passed")) for row in per_rank),
        "world_size": len(per_rank),
        "comparison": "restored from existing R0 first_batch_audit.json",
        "per_rank": per_rank,
    }


def _postprocess_metrics_r0(args: Any) -> None:
    _ORIGINAL_K1_POSTPROCESS(args)
    if int(os.environ.get("RANK", "0")) != 0:
        return
    metrics_path = r0_paths(args.output_dir, args.seed)["metrics"]
    if not metrics_path.is_file():
        raise FileNotFoundError(f"R0 metrics were not created: {metrics_path}")
    if _K1_REFERENCE is None or _REFERENCE_VALIDATION is None:
        raise RuntimeError("R0 K1 reference validation state is unavailable")

    results = _read_json(metrics_path)
    reference_metrics = _K1_REFERENCE["metrics"]
    assert isinstance(reference_metrics, Mapping)
    reference_best = reference_metrics.get("best_dev_metrics", {})
    result_best = results.get("best_dev_metrics", {})
    if not isinstance(reference_best, Mapping) or not isinstance(result_best, Mapping):
        raise RuntimeError("R0/K1 metrics lack best_dev_metrics")

    actual_miou = float(result_best["mIoU"])
    reference_miou = float(reference_best["mIoU"])
    miou_delta = actual_miou - reference_miou
    miou_passed = abs(miou_delta) <= K1_MIOU_SAMPLE_STD
    first_equivalence = _FIRST_BATCH_EQUIVALENCE or _existing_first_batch_equivalence(
        args
    )
    first_passed = bool(first_equivalence and first_equivalence.get("passed"))
    overall_passed = bool(
        _REFERENCE_VALIDATION.get("passed") and first_passed and miou_passed
    )

    relation_spec = _RELATION_SPEC
    if relation_spec is None:
        config = results.get("config", {})
        if isinstance(config, Mapping) and isinstance(config.get("relation"), Mapping):
            relation_spec = dict(config["relation"])  # type: ignore[arg-type]
    relation_hash = (
        None if relation_spec is None else _canonical_sha256(relation_spec)
    )

    results["experiment"] = EXPERIMENT
    results["experiment_group"] = EXPERIMENT_GROUP
    results["artifact_type"] = ARTIFACT_TYPE
    results["protocol"] = (
        "R0 controlled reproduction of K1: existing K shared scratch "
        "MobileNetV2+R-ASPP initialization, hard-label CE plus locked A0 fixed "
        "StandardScaler+PCA OS=4/8/16 feature MSE, 4000-step auxiliary warm-up, "
        "no logits KD, no R1/R2 relation term, fixed 80k budget, dev_local "
        "selection, and no test_local evaluation."
    )
    results["relation"] = copy.deepcopy(relation_spec)
    results["relation_spec_sha256"] = relation_hash
    results["physical_relation_batch_size"] = (
        None
        if relation_spec is None
        else relation_spec.get("physical_relation_batch_size")
    )
    results["effective_optimizer_batch_size"] = (
        None
        if relation_spec is None
        else relation_spec.get("effective_optimizer_batch_size")
    )
    results["r0_k1_equivalence"] = {
        "passed": overall_passed,
        "reference_validation": copy.deepcopy(_REFERENCE_VALIDATION),
        "first_batch": copy.deepcopy(first_equivalence),
        "final_dev": {
            "passed": miou_passed,
            "R0_mIoU": actual_miou,
            "K1_mIoU": reference_miou,
            "delta_R0_minus_K1": miou_delta,
            "absolute_tolerance": K1_MIOU_SAMPLE_STD,
            "R0_selected_model_state_sha256": results.get("hashes", {}).get(
                "selected_model_state_sha256"
            )
            if isinstance(results.get("hashes"), Mapping)
            else None,
            "K1_selected_model_state_sha256": reference_metrics.get(
                "hashes", {}
            ).get("selected_model_state_sha256")
            if isinstance(reference_metrics.get("hashes"), Mapping)
            else None,
        },
        "interpretation": (
            "R1/R2 may proceed only when this R0 equivalence acceptance passes."
        ),
    }
    loss = results.get("loss")
    if isinstance(loss, dict):
        loss.update(
            {
                "relation_kd": False,
                "relation_r1": False,
                "relation_r2": False,
                "lambda_r1": 0.0,
                "lambda_r2": 0.0,
            }
        )
    results["hashes"] = {
        **dict(results.get("hashes", {})),
        "relation_spec_sha256": relation_hash,
        "k1_reference_metrics_sha256": _REFERENCE_VALIDATION.get(
            "reference_metrics_sha256"
        ),
        "k1_reference_first_batch_sha256": _REFERENCE_VALIDATION.get(
            "reference_first_batch_sha256"
        ),
        "r0_training_script_sha256": common.sha256_file(Path(__file__).resolve()),
    }
    results["test_local_evaluated"] = False
    common.write_json_atomic(metrics_path, results)

    if not overall_passed:
        raise RuntimeError(
            "R0 completed but failed the K1 equivalence acceptance; inspect "
            f"{metrics_path} before implementing or running R1/R2"
        )


def run_training(args: Any) -> None:
    """Temporarily route K1's audited runner through the R0 contract."""

    global _K1_REFERENCE
    global _REFERENCE_VALIDATION
    global _RELATION_SPEC
    global _FIRST_BATCH_EQUIVALENCE

    _K1_REFERENCE, _REFERENCE_VALIDATION = _validate_k1_reference(args)
    _RELATION_SPEC = None
    _FIRST_BATCH_EQUIVALENCE = (
        _existing_first_batch_equivalence(args) if args.resume else None
    )

    saved: Dict[str, object] = {
        "__file__": k1.__file__,
        "EXPERIMENT": k1.EXPERIMENT,
        "ARTIFACT_TYPE": k1.ARTIFACT_TYPE,
        "ARTIFACT_FORMAT_VERSION": k1.ARTIFACT_FORMAT_VERSION,
        "k1_paths": k1.k1_paths,
        "build_config": k1.build_config,
        "build_best_checkpoint": k1.build_best_checkpoint,
        "train_one_epoch_k1": k1.train_one_epoch_k1,
        "smoke_test_k1": k1.smoke_test_k1,
        "_postprocess_metrics": k1._postprocess_metrics,
        "audit_k1_shapes": k1.audit_k1_shapes,
        "_patched_torch_save_atomic": k1._patched_torch_save_atomic,
        "_patched_evaluate": k1._patched_evaluate,
        "_k1_print": k1._k1_print,
        "tqdm": k1.tqdm,
        "_ORIGINAL_ENSURE_SHARED_INITIALIZATION": (
            k1._ORIGINAL_ENSURE_SHARED_INITIALIZATION
        ),
    }
    had_module_print = "print" in k1.__dict__
    saved_module_print = k1.__dict__.get("print")

    k1.__file__ = str(Path(__file__).resolve())
    k1.EXPERIMENT = EXPERIMENT
    k1.ARTIFACT_TYPE = ARTIFACT_TYPE
    k1.ARTIFACT_FORMAT_VERSION = ARTIFACT_FORMAT_VERSION
    k1.k1_paths = r0_paths
    k1.build_config = build_config_r0
    k1.build_best_checkpoint = build_best_checkpoint_r0
    k1.train_one_epoch_k1 = train_one_epoch_r0
    k1.smoke_test_k1 = smoke_test_r0
    k1._postprocess_metrics = _postprocess_metrics_r0
    k1.audit_k1_shapes = audit_shapes_r0
    k1._patched_torch_save_atomic = _patched_torch_save_atomic_r0
    k1._patched_evaluate = _patched_evaluate_r0
    k1._k1_print = _r0_print
    k1.tqdm = _r0_tqdm
    k1._ORIGINAL_ENSURE_SHARED_INITIALIZATION = (
        _ensure_locked_k_shared_initialization
    )
    k1.print = _r0_print
    try:
        k1.run_training(args)
    finally:
        for name, value in saved.items():
            setattr(k1, name, value)
        if had_module_print:
            k1.print = saved_module_print
        else:
            k1.__dict__.pop("print", None)
        _RELATION_SPEC = None


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
