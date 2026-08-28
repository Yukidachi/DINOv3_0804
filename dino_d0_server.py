"""D0 server entry point: the controlled K1 feature-KD anchor.

D0 is the first experiment in the D group.  Its objective is deliberately
identical to K1::

    L_D0 = L_seg + min(1, step / 4000) * L_feat

where ``L_feat`` is the equal-weight OS=4/8/16 A0 fixed
StandardScaler+PCA feature MSE.  The purpose of this entry point is protocol
isolation and an auditable K1-equivalence check, not a new loss.  It therefore
reuses the already audited R0/K1 runner while changing only the D-group
artifact contract and output directory.

The student initialization is *always* loaded from the existing K-group
shared initialization::

    result/K_MobileNetV2_RASPP_server/shared_init/seed_<seed>/student_init.pth

No D-specific initialization or trained student checkpoint is created.

Typical two-GPU server command::

    torchrun --standalone --nproc_per_node=2 dino_d0_server.py \
        --seed 42 --batch-size 2 --global-batch-size 8 \
        --num-workers 8 --multiprocessing-context spawn \
        --no-pin-memory --persistent-workers

Windows/local functional smoke (does not replace Linux two-GPU DDP smoke)::

    python -B dino_d0_server.py --device cuda --smoke-test \
        --batch-size 1 --global-batch-size 1 --num-workers 0 \
        --no-persistent-workers --no-pin-memory --no-amp
"""

from __future__ import annotations

import argparse
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

import dino as common
import dino_k0_server as k0
import dino_r0_server as r0


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "D_MobileNetV2_RASPP_server"
K_GROUP_OUTPUT_DIR = SCRIPT_DIR / "result" / "K_MobileNetV2_RASPP_server"
K1_REFERENCE_DIR = K_GROUP_OUTPUT_DIR / "K1" / "seed_42"
DEFAULT_TEACHER_CHECKPOINT = r0.k1.DEFAULT_TEACHER_CHECKPOINT
DEFAULT_PCA_DIR = r0.k1.DEFAULT_PCA_DIR
EXPECTED_COMBINED_MANIFEST_SHA256 = k0.EXPECTED_COMBINED_MANIFEST_SHA256

EXPERIMENT = "D0"
EXPERIMENT_GROUP = "D_MobileNetV2_RASPP_server"
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_d0_k1_anchor"
ARTIFACT_FORMAT_VERSION = 1
FORMAL_SEEDS = (42,)
MODEL_NAME = r0.k1.base.MODEL_NAME
LAMBDA_FEAT = 1.0
FEATURE_WARMUP_RATIO = 0.05

DISTRIBUTION_SPEC_VERSION = 1
DISTRIBUTION_TOKEN_CAP = 256
DISTRIBUTION_WARMUP_STEPS = 4_000
FIRST_BATCH_ABS_TOLERANCE = 1e-6
FIRST_BATCH_REL_TOLERANCE = 1e-6
K1_REFERENCE_MIOU = r0.K1_REFERENCE_MIOU
K1_MIOU_SAMPLE_STD = r0.K1_MIOU_SAMPLE_STD

# Keep references to the audited R0 functions before the temporary D0
# routing below replaces their module-level names.  Calling the live names
# from a replacement would recurse once ``run_training`` installs the hooks.
_ORIGINAL_R0_TRAIN_ONE_EPOCH = r0.train_one_epoch_r0
_ORIGINAL_R0_POSTPROCESS = r0._postprocess_metrics_r0
_ORIGINAL_K0_PATHS = r0.k1._ORIGINAL_K0_PATHS


def _canonical_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    """Reuse the audited R0/K1 CLI with D0 defaults and locks."""

    saved_default = r0.DEFAULT_OUTPUT_DIR
    saved_argparse = r0.k1.argparse

    class D0ArgparseProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(saved_argparse, name)

        @staticmethod
        def ArgumentParser(*parser_args: Any, **parser_kwargs: Any):
            parser_kwargs["description"] = (
                "D0 MobileNetV2+R-ASPP controlled reproduction of K1: "
                "hard-label CE plus the locked A0 fixed StandardScaler+PCA "
                "feature target, with distribution terms disabled."
            )
            return saved_argparse.ArgumentParser(*parser_args, **parser_kwargs)

    r0.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    r0.k1.argparse = D0ArgparseProxy()
    try:
        args = r0.parse_args()
    finally:
        r0.DEFAULT_OUTPUT_DIR = saved_default
        r0.k1.argparse = saved_argparse

    if args.seed != 42:
        raise SystemExit("D0 is pre-registered for --seed 42")
    if not args.smoke_test and args.max_steps != 80_000:
        raise SystemExit("Formal D0 is locked to exactly 80,000 optimizer steps")
    if not args.smoke_test and args.eval_every_steps != 5_000:
        raise SystemExit("Formal D0 is locked to --eval-every-steps 5000")
    if not args.smoke_test and args.gradient_log_steps != 500:
        raise SystemExit("Formal D0 is locked to --gradient-log-steps 500")
    if args.output_dir.resolve() == K_GROUP_OUTPUT_DIR.resolve():
        raise SystemExit(
            "D0 output must be separate from the K-group directory; use "
            "result/D_MobileNetV2_RASPP_server"
        )
    return args


def d0_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    """Place D0 artifacts below ``D_MobileNetV2_RASPP_server/D0``."""

    original = _ORIGINAL_K0_PATHS(output_dir, seed)
    run_dir = output_dir.resolve() / EXPERIMENT / f"seed_{seed}"
    return {
        key: run_dir if key == "run_dir" else run_dir / value.name
        for key, value in original.items()
    }


def distribution_spec(
    args: argparse.Namespace, accumulation_steps: int, world_size: int
) -> Dict[str, object]:
    """Record the D-group distribution contract, disabled for D0.

    Keeping the common token/mask/batch schema in D0 artifacts makes the
    anchor directly comparable with D1-D4 without accidentally enabling a
    distribution term.
    """

    physical_batch = int(args.batch_size) * int(world_size)
    effective_batch = physical_batch * int(accumulation_steps)
    return {
        "spec_version": DISTRIBUTION_SPEC_VERSION,
        "enabled": False,
        "type": "none",
        "active_terms": [],
        "teacher_source": "A0 projected OS=4/8/16 features (reserved; unused by D0 distribution)",
        "student_source": "native OS=4/8/16 student taps (reserved; unused by D0 distribution)",
        "mask_policy": "targets != 255; distribution term disabled",
        "physical_distribution_batch_size": physical_batch,
        "effective_optimizer_batch_size": effective_batch,
        "token_cap_per_layer": DISTRIBUTION_TOKEN_CAP,
        "token_sampling": "deterministic without replacement; disabled for D0",
        "statistics": "teacher-batch statistics; disabled for D0",
        "layers": ["os4", "os8", "os16"],
        "layer_reduction": "equal mean over OS=4/8/16; disabled for D0",
        "warmup_steps": DISTRIBUTION_WARMUP_STEPS,
        "lambda_coral": 0.0,
        "lambda_swd": 0.0,
        "lambda_adv": 0.0,
        "num_slices": None,
        "direction_seed": None,
        "discriminator": None,
    }


def _distribution_from_config(
    args: argparse.Namespace, accumulation_steps: int, world_size: int
) -> Tuple[Dict[str, object], str]:
    spec = distribution_spec(args, accumulation_steps, world_size)
    return spec, _canonical_sha256(spec)


def build_config_d0(
    args: argparse.Namespace,
    accumulation_steps: int,
    world_size: int,
    device: torch.device,
    shared_init_state_sha256: str,
    shared_init_file_sha256: str,
) -> Dict[str, object]:
    config = r0._ORIGINAL_K1_BUILD_CONFIG(
        args,
        accumulation_steps,
        world_size,
        device,
        shared_init_state_sha256,
        shared_init_file_sha256,
    )
    spec, spec_hash = _distribution_from_config(args, accumulation_steps, world_size)
    r0._RELATION_SPEC = copy.deepcopy(spec)

    if not args.smoke_test:
        if world_size != 2:
            raise RuntimeError(f"Formal D0 requires world_size=2, got {world_size}")
        if spec["physical_distribution_batch_size"] != 4:
            raise RuntimeError("Formal D0 requires physical_distribution_batch_size=4")
        if spec["effective_optimizer_batch_size"] != 8:
            raise RuntimeError("Formal D0 requires effective_optimizer_batch_size=8")

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
                "d_specific_initialization_created": False,
            },
            "distribution": copy.deepcopy(spec),
            "distribution_spec_sha256": spec_hash,
            "d0_k1_equivalence": {
                "reference_directory": str(K1_REFERENCE_DIR.resolve()),
                "first_batch_abs_tolerance": FIRST_BATCH_ABS_TOLERANCE,
                "first_batch_rel_tolerance": FIRST_BATCH_REL_TOLERANCE,
                "final_mIoU_abs_tolerance": r0.K1_MIOU_SAMPLE_STD,
                "required_before_d1_d2": True,
            },
            "test_local_evaluated": False,
        }
    )
    loss = config.get("loss")
    if isinstance(loss, dict):
        loss.update(
            {
                "distribution_kd": False,
                "distribution_type": "none",
                "lambda_coral": 0.0,
                "lambda_swd": 0.0,
                "lambda_adv": 0.0,
            }
        )
    config.pop("relation", None)
    config.pop("relation_spec_sha256", None)
    return config


def audit_shapes_d0(
    model: torch.nn.Module,
    device: torch.device,
    height: int,
    width: int,
    amp_enabled: bool,
) -> Dict[str, object]:
    audit = r0._ORIGINAL_K1_AUDIT_SHAPES(model, device, height, width, amp_enabled)
    audit.update(
        {
            "experiment": EXPERIMENT,
            "distribution": {
                "enabled": False,
                "source": "A0 projected teacher/student OS=4/8/16 features (reserved)",
                "mask_policy": "targets != 255",
                "token_cap_per_layer": DISTRIBUTION_TOKEN_CAP,
            },
        }
    )
    return audit


def build_best_checkpoint_d0(*args: Any, **kwargs: Any) -> Dict[str, object]:
    payload = r0._ORIGINAL_K1_BUILD_BEST_CHECKPOINT(*args, **kwargs)
    config = payload.get("config", {})
    if isinstance(config, Mapping):
        spec = config.get("distribution")
        spec_hash = config.get("distribution_spec_sha256")
    else:
        spec, spec_hash = None, None
    payload.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "distribution": copy.deepcopy(spec),
            "distribution_spec_sha256": spec_hash,
            "d0_k1_equivalence": copy.deepcopy(r0._FIRST_BATCH_EQUIVALENCE),
            "d0_k1_reference_validation": copy.deepcopy(r0._REFERENCE_VALIDATION),
        }
    )
    payload["hashes"] = {
        **dict(payload.get("hashes", {})),
        "training_script_sha256": common.sha256_file(Path(__file__).resolve()),
        "d0_training_script_sha256": common.sha256_file(Path(__file__).resolve()),
    }
    payload.pop("relation", None)
    payload.pop("relation_spec_sha256", None)
    return payload


def _patched_torch_save_atomic_d0(payload: object, path: Path) -> None:
    if isinstance(payload, Mapping) and payload.get("artifact_type") == ARTIFACT_TYPE:
        payload = dict(payload)
        config = payload.get("config", {})
        spec = config.get("distribution") if isinstance(config, Mapping) else None
        spec_hash = (
            config.get("distribution_spec_sha256")
            if isinstance(config, Mapping)
            else None
        )
        payload.update(
            {
                "experiment": EXPERIMENT,
                "experiment_group": EXPERIMENT_GROUP,
                "distribution": copy.deepcopy(spec),
                "distribution_spec_sha256": spec_hash,
                "d0_k1_equivalence": copy.deepcopy(r0._FIRST_BATCH_EQUIVALENCE),
                "d0_k1_reference_validation": copy.deepcopy(
                    r0._REFERENCE_VALIDATION
                ),
                "hashes": {
                    **dict(payload.get("hashes", {})),
                    **r0.k1._resource_hashes(),
                    "training_script_sha256": common.sha256_file(
                        Path(__file__).resolve()
                    ),
                    "d0_training_script_sha256": common.sha256_file(Path(__file__).resolve()),
                },
            }
        )
        payload.pop("relation", None)
        payload.pop("relation_spec_sha256", None)
        payload["pca_parameters_sha256_record"] = copy.deepcopy(
            r0.k1._PCA_PARAMETER_RECORD
        )
    # The original atomic writer is stored on dino_k1_server, not on the
    # R0 wrapper module.  Using the wrong private attribute only becomes
    # visible when the first epoch attempts to persist last_checkpoint.pth.
    r0.k1._ORIGINAL_TORCH_SAVE_ATOMIC(payload, path)


def _patched_evaluate_d0(*args: Any, **kwargs: Any):
    split_name = kwargs.get("split_name")
    if isinstance(split_name, str):
        kwargs["split_name"] = split_name.replace("K0", EXPERIMENT).replace(
            "K1", EXPERIMENT
        )
    return r0.k1._ORIGINAL_EVALUATE(*args, **kwargs)


def train_one_epoch_d0(*args: Any, **kwargs: Any):
    # R0's wrapper performs the locked first-batch comparison.  Reuse it, then
    # strip R-group relation fields so D0 cannot be mistaken for an R run.
    metrics, optimizer_steps, gradient_records, first_batch = _ORIGINAL_R0_TRAIN_ONE_EPOCH(
        *args, **kwargs
    )
    for key in list(metrics):
        if key.startswith("relation_"):
            metrics.pop(key, None)
    metrics.update(
        {
            "distribution_enabled": False,
            "distribution_type": "none",
            "distribution_loss": None,
            "distribution_loss_weighted": 0.0,
            "distribution_spec_sha256": _canonical_sha256(r0._RELATION_SPEC or {}),
            "distribution_physical_batch_size": r0._RELATION_SPEC.get(
                "physical_distribution_batch_size"
            )
            if isinstance(r0._RELATION_SPEC, Mapping)
            else None,
            "distribution_global_token_count": None,
        }
    )
    for record in gradient_records:
        for key in list(record):
            if key.startswith("relation_"):
                record.pop(key, None)
        record.update(
            {
                "distribution_enabled": False,
                "distribution_type": "none",
                "distribution_lambda": 0.0,
                "distribution_raw_loss": None,
                "distribution_weighted_loss": 0.0,
                "distribution_global_token_count_os4": None,
                "distribution_global_token_count_os8": None,
                "distribution_global_token_count_os16": None,
                "distribution_spec_sha256": _canonical_sha256(
                    r0._RELATION_SPEC or {}
                ),
            }
        )
    if first_batch is not None:
        first_batch.pop("relation", None)
        first_batch.pop("relation_spec_sha256", None)
        equivalence = first_batch.pop("r0_k1_equivalence", None)
        if equivalence is not None:
            first_batch["d0_k1_equivalence"] = equivalence
        first_batch["distribution"] = {
            "enabled": False,
            "type": "none",
            "physical_batch_size": metrics.get("distribution_physical_batch_size"),
        }
        first_batch["distribution_spec_sha256"] = _canonical_sha256(
            r0._RELATION_SPEC or {}
        )
    if isinstance(r0._FIRST_BATCH_EQUIVALENCE, dict):
        r0._FIRST_BATCH_EQUIVALENCE["comparison"] = (
            "D0 first formal batch versus locked K1 seed=42"
        )
    return metrics, optimizer_steps, gradient_records, first_batch


def smoke_test_d0(*args: Any, **kwargs: Any) -> None:
    r0._ORIGINAL_K1_SMOKE_TEST(*args, **kwargs)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        builtins.print(
            "[OK] D0 protocol smoke: K1 CE+fixed-A0-feature objective reused; "
            "distribution terms are disabled; K shared init was loaded."
        )


def _d0_print(*values: object, **kwargs: object) -> None:
    adjusted = tuple(
        value.replace("K0", EXPERIMENT)
        .replace("K1", EXPERIMENT)
        .replace("R0", EXPERIMENT)
        if isinstance(value, str)
        else value
        for value in values
    )
    builtins.print(*adjusted, **kwargs)


def _d0_tqdm(*args: Any, **kwargs: Any):
    description = kwargs.get("desc")
    if isinstance(description, str):
        kwargs["desc"] = description.replace("K1", EXPERIMENT)
    return r0._ORIGINAL_TQDM(*args, **kwargs)


def _postprocess_metrics_d0(args: argparse.Namespace) -> None:
    _ORIGINAL_R0_POSTPROCESS(args)
    if int(os.environ.get("RANK", "0")) != 0:
        return

    paths = d0_paths(args.output_dir, args.seed)
    metrics_path = paths["metrics"]
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    spec = r0._RELATION_SPEC or distribution_spec(args, 2, 2)
    spec_hash = _canonical_sha256(spec)
    equivalence = results.pop("r0_k1_equivalence", None)
    physical_batch = results.pop("physical_relation_batch_size", None)
    effective_batch = results.pop("effective_optimizer_batch_size", None)
    if physical_batch is None:
        physical_batch = spec.get("physical_distribution_batch_size")
    if effective_batch is None:
        effective_batch = spec.get("effective_optimizer_batch_size")
    results.pop("relation", None)
    results.pop("relation_spec_sha256", None)
    results["experiment"] = EXPERIMENT
    results["experiment_group"] = EXPERIMENT_GROUP
    results["artifact_type"] = ARTIFACT_TYPE
    results["protocol"] = (
        "D0 controlled K1-anchor reproduction: existing K shared scratch "
        "MobileNetV2+R-ASPP initialization, hard-label CE plus locked A0 fixed "
        "StandardScaler+PCA OS=4/8/16 feature MSE, 4000-step auxiliary "
        "warm-up, no logits KD, no distribution term, fixed 80k budget, "
        "dev_local selection, and no test_local evaluation."
    )
    results["distribution"] = copy.deepcopy(spec)
    results["distribution_spec_sha256"] = spec_hash
    results["physical_distribution_batch_size"] = physical_batch
    results["effective_optimizer_batch_size"] = effective_batch
    results["distribution_global_token_count"] = None
    results["d0_k1_equivalence"] = equivalence
    results["hashes"] = {
        **dict(results.get("hashes", {})),
        "training_script_sha256": common.sha256_file(Path(__file__).resolve()),
        "distribution_spec_sha256": spec_hash,
        "d0_training_script_sha256": common.sha256_file(Path(__file__).resolve()),
    }
    results["test_local_evaluated"] = False
    loss = results.get("loss")
    if isinstance(loss, dict):
        for key in ("relation_kd", "relation_r1", "relation_r2", "lambda_r1", "lambda_r2"):
            loss.pop(key, None)
        loss.update(
            {
                "distribution_kd": False,
                "distribution_type": "none",
                "lambda_coral": 0.0,
                "lambda_swd": 0.0,
                "lambda_adv": 0.0,
            }
        )
    common.write_json_atomic(metrics_path, results)

    config_path = paths["config"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["experiment"] = EXPERIMENT
    config["experiment_group"] = EXPERIMENT_GROUP
    config["artifact_type"] = ARTIFACT_TYPE
    config["distribution"] = copy.deepcopy(spec)
    config["distribution_spec_sha256"] = spec_hash
    config["d0_k1_equivalence"] = copy.deepcopy(equivalence)
    config.pop("relation", None)
    config.pop("relation_spec_sha256", None)
    common.write_json_atomic(config_path, config)
    # Keep the embedded metrics config byte-for-byte aligned with the final
    # standalone config artifact after adding the completed equivalence audit.
    results["config"] = copy.deepcopy(config)
    common.write_json_atomic(metrics_path, results)


def run_training(args: argparse.Namespace) -> None:
    """Route the audited R0/K1 runner through the D0 artifact contract."""

    saved = {
        "EXPERIMENT": r0.EXPERIMENT,
        "EXPERIMENT_GROUP": r0.EXPERIMENT_GROUP,
        "ARTIFACT_TYPE": r0.ARTIFACT_TYPE,
        "r0_paths": r0.r0_paths,
        "build_config_r0": r0.build_config_r0,
        "build_best_checkpoint_r0": r0.build_best_checkpoint_r0,
        "train_one_epoch_r0": r0.train_one_epoch_r0,
        "smoke_test_r0": r0.smoke_test_r0,
        "audit_shapes_r0": r0.audit_shapes_r0,
        "_patched_torch_save_atomic_r0": r0._patched_torch_save_atomic_r0,
        "_patched_evaluate_r0": r0._patched_evaluate_r0,
        "_postprocess_metrics_r0": r0._postprocess_metrics_r0,
        "_r0_print": r0._r0_print,
        "_r0_tqdm": r0._r0_tqdm,
    }
    r0.EXPERIMENT = EXPERIMENT
    r0.EXPERIMENT_GROUP = EXPERIMENT_GROUP
    r0.ARTIFACT_TYPE = ARTIFACT_TYPE
    r0.r0_paths = d0_paths
    r0.build_config_r0 = build_config_d0
    r0.build_best_checkpoint_r0 = build_best_checkpoint_d0
    r0.train_one_epoch_r0 = train_one_epoch_d0
    r0.smoke_test_r0 = smoke_test_d0
    r0.audit_shapes_r0 = audit_shapes_d0
    r0._patched_torch_save_atomic_r0 = _patched_torch_save_atomic_d0
    r0._patched_evaluate_r0 = _patched_evaluate_d0
    r0._postprocess_metrics_r0 = _postprocess_metrics_d0
    r0._r0_print = _d0_print
    r0._r0_tqdm = _d0_tqdm
    try:
        r0.run_training(args)
    finally:
        for name, value in saved.items():
            setattr(r0, name, value)


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
