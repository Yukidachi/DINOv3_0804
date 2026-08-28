"""H0 server entry point: the ReLU6 matched anchor for the H group.

H0 deliberately keeps the original torchvision MobileNetV2 activations at
every registered site and reuses the locked R5 objective and training runner:

    L = L_CE + warmup * (1.0 * L_feature + 0.3 * L_R2 + 0.5 * L_logit)

The model is built from the same K-group shared scratch initialization as the
other H runs.  No trained R5/K3/R2 student checkpoint is used as an
initialization.  This file is a thin, auditable adapter around
``dino_r5_server`` so that the feature, relation, logit, DDP, resume and
checkpoint implementations remain identical across R5 and H0.

Typical two-GPU launch::

    torchrun --standalone --nproc_per_node=2 dino_h0_server.py \
        --seed 42 --batch-size 2 --global-batch-size 8 \
        --num-workers 8 --multiprocessing-context spawn \
        --no-pin-memory --persistent-workers

Local functional smoke::

    conda run -n pytorch_e python -B dino_h0_server.py --device cuda \
        --smoke-test --batch-size 1 --global-batch-size 1 \
        --num-workers 0 --no-persistent-workers --no-pin-memory --no-amp
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
import dino_k3_server as k3
import dino_r1_server as r1
import dino_r2_server as r2
import dino_r5_server as r5
import dino_s2_0 as base
import dino_s2_0_server as server_base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "H_MobileNetV2_RASPP_server"
K_GROUP_OUTPUT_DIR = SCRIPT_DIR / "result" / "K_MobileNetV2_RASPP_server"
R_GROUP_OUTPUT_DIR = SCRIPT_DIR / "result" / "R_MobileNetV2_RASPP_server"

EXPERIMENT = "H0"
EXPERIMENT_GROUP = "H_MobileNetV2_RASPP_server"
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_h0_relu6_r5_anchor"
ARTIFACT_FORMAT_VERSION = 1

FORMAL_SEEDS = (42, 3407, 260805)
SCREENING_SEED = 42
LAMBDA_FEAT = 1.0
LAMBDA_R2 = 0.3
LAMBDA_LOGIT = 0.5
TEMPERATURE = 4.0
AUXILIARY_WARMUP_RATIO = 0.05
GRADIENT_LOG_STEPS = 500
FIXED_GRADIENT_AUDIT_STEPS = (1, 4_000, 20_000, 40_000, 60_000, 80_000)

# Keep immutable references because ``run_training`` temporarily installs the
# H0 wrappers under the R5 function names used by the shared runner.
_ORIGINAL_R5_BUILD_CONFIG = r5.build_config_r5
_ORIGINAL_R5_AUDIT_SHAPES = r5.audit_shapes_r5
_ORIGINAL_R5_BUILD_BEST_CHECKPOINT = r5.build_best_checkpoint_r5
_ORIGINAL_R5_TRAIN_ONE_EPOCH = r5.train_one_epoch_r5


def h0_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    """Return the standard K0 artifact names under ``H0/seed_<seed>``."""

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


def _h0_activation_spec() -> Dict[str, object]:
    """Static H0 contract, including the exact eligible MobileNetV2 paths."""

    eligible = []
    for block_index in range(1, 18):
        # Block 1 has expansion ratio 1 and therefore only has conv.0.2.
        eligible.append(
            {
                "path": f"backbone.{block_index}.conv.0.2",
                "block_index": block_index,
                "path_type": "depthwise" if block_index == 1 else "expansion",
                "exists_when_built": True,
                "activation": "ReLU6",
            }
        )
        if block_index != 1:
            eligible.append(
                {
                    "path": f"backbone.{block_index}.conv.1.2",
                    "block_index": block_index,
                    "path_type": "depthwise",
                    "exists_when_built": True,
                    "activation": "ReLU6",
                }
            )
    # The path list above intentionally records the logical contract.  The
    # actual audit below records only modules that exist in this torchvision
    # version (and checks that block 1's non-existent expansion path is absent).
    return {
        "name": "ReLU6",
        "implementation": "torch.nn.ReLU6(inplace=True)",
        "formula": "relu6(x)=min(max(x, 0), 6)",
        "placement_scope": "original MobileNetV2 ReLU6; no replacement",
        "eligible_blocks": list(range(1, 18)),
        "eligible_paths": eligible,
        "replaced_module_paths": [],
        "replacement_count": 0,
        "stem_path": "backbone.0.2",
        "final_path": "backbone.18.2",
        "head_path": "head.project.2",
        "linear_bottleneck_activation": False,
    }


def _activation_audit(model: nn.Module) -> Dict[str, object]:
    """Verify H0 leaves every activation boundary at the registered type."""

    probe = torch.tensor([-6.0, -3.0, 0.0, 3.0, 6.0, 9.0])
    reference = torch.minimum(
        torch.maximum(probe, torch.zeros_like(probe)),
        torch.full_like(probe, 6.0),
    )
    observed = nn.ReLU6()(probe)
    reference_error = float((observed - reference).abs().max().item())
    if reference_error > 1e-7 or not bool(torch.isfinite(observed).all().item()):
        raise RuntimeError(f"H0 ReLU6 reference test failed: max_abs_error={reference_error}")

    student = model.module if hasattr(model, "module") else model
    if not hasattr(student, "backbone") or not hasattr(student, "head"):
        raise RuntimeError("H0 activation audit received an unexpected student")
    backbone = student.backbone
    if len(backbone) != 19:
        raise RuntimeError(f"H0 expected 19 MobileNetV2 backbone modules, got {len(backbone)}")

    actual = []
    missing_logical_paths = []
    for index in range(1, 18):
        block = backbone[index]
        conv = getattr(block, "conv", None)
        if conv is None:
            raise RuntimeError(f"H0 block {index} has no conv sequential")
        candidates = (
            (("depthwise", 0),)
            if index == 1
            else (("expansion", 0), ("depthwise", 1))
        )
        for path_type, sub_index in candidates:
            path = f"backbone.{index}.conv.{sub_index}.2"
            try:
                activation = conv[sub_index][2]
            except (IndexError, KeyError, TypeError):
                missing_logical_paths.append(path)
                continue
            if not isinstance(activation, nn.ReLU6):
                raise RuntimeError(
                    f"H0 activation mismatch at {path}: {_module_type(activation)}"
                )
            actual.append(
                {
                    "path": path,
                    "block_index": index,
                    "path_type": path_type,
                    "module_type": _module_type(activation),
                }
            )

        # The projection/linear bottleneck must not contain an activation.
        projection = conv[2] if len(conv) > 2 else None
        if projection is None:
            raise RuntimeError(f"H0 block {index} has no linear bottleneck projection")
        if any(
            isinstance(module, (nn.ReLU6, nn.Hardswish, nn.SiLU, nn.GELU))
            for module in projection.modules()
        ):
            raise RuntimeError(f"H0 block {index} linear bottleneck contains an activation")

    fixed = {
        "stem": backbone[0][2],
        "final": backbone[18][2],
        "raspp_head": student.head.project[2],
    }
    fixed_types = {name: _module_type(module) for name, module in fixed.items()}
    if not all(isinstance(module, nn.ReLU6) for module in fixed.values()):
        raise RuntimeError(f"H0 stem/final/R-ASPP activation contract failed: {fixed_types}")

    expected_count = 33  # block 1 has depthwise only; blocks 2..17 have two sites.
    if len(actual) != expected_count:
        raise RuntimeError(
            f"H0 expected {expected_count} eligible ReLU6 modules, observed {len(actual)}; "
            f"missing={missing_logical_paths}"
        )
    audit = {
        "experiment": EXPERIMENT,
        "activation_name": "ReLU6",
        "activation_formula": "min(max(x, 0), 6)",
        "eligible_module_count": expected_count,
        "eligible_module_paths": actual,
        "missing_logical_paths": missing_logical_paths,
        "fixed_paths": {
            "backbone.0.2": fixed_types["stem"],
            "backbone.18.2": fixed_types["final"],
            "head.project.2": fixed_types["raspp_head"],
        },
        "linear_bottleneck_activation": False,
        "replacement_count": 0,
        "reference_test": {
            "max_abs_error": reference_error,
            "tolerance": 1e-7,
            "passed": True,
        },
    }
    audit["activation_spec_sha256"] = _canonical_hash(_h0_activation_spec())
    audit["activation_audit_sha256"] = _canonical_hash(audit)
    return audit


def parse_args() -> argparse.Namespace:
    """Use K3's locked CLI, adding only the fixed R2 weight for H0."""

    saved_default = k3.DEFAULT_OUTPUT_DIR
    saved_argparse = k3.argparse

    class H0ArgparseProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(saved_argparse, name)

        @staticmethod
        def ArgumentParser(*parser_args: Any, **parser_kwargs: Any):
            parser_kwargs["description"] = (
                "H0 MobileNetV2+R-ASPP: ReLU6 matched anchor with the locked "
                "R5 feature+R2+pixel-logit KD protocol."
            )
            parser = saved_argparse.ArgumentParser(*parser_args, **parser_kwargs)
            parser.add_argument(
                "--lambda-r2",
                type=float,
                default=LAMBDA_R2,
                help="Fixed H0/R5 spatial-relation weight (0.3).",
            )
            return parser

    k3.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    k3.argparse = H0ArgparseProxy()
    try:
        args = k3.parse_args()
    finally:
        k3.DEFAULT_OUTPUT_DIR = saved_default
        k3.argparse = saved_argparse

    if args.seed not in FORMAL_SEEDS:
        raise SystemExit(f"H0 seed must be one of {FORMAL_SEEDS}")
    if not math.isclose(args.lambda_r2, LAMBDA_R2, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("Formal H0 is locked to --lambda-r2 0.3")
    if args.output_dir.resolve() == K_GROUP_OUTPUT_DIR.resolve():
        raise SystemExit("H0 output must use the separate H-group output directory")
    if not args.smoke_test:
        if args.max_steps != 80_000:
            raise SystemExit("Formal H0 is locked to exactly 80,000 optimizer steps")
        if args.eval_every_steps != 5_000:
            raise SystemExit("Formal H0 is locked to --eval-every-steps 5000")
        if args.gradient_log_steps != GRADIENT_LOG_STEPS:
            raise SystemExit("Formal H0 is locked to --gradient-log-steps 500")
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


def _validate_launch_gate_h0(args: argparse.Namespace) -> Dict[str, object]:
    """Load matched first-batch references without R5's expansion gate.

    H0 is the anchor used to compare H1-H3.  It must not inherit R5's
    requirement that R5 itself beat K3/R2 before confirmation seeds are run.
    Existing K3/R2 first-batch audits are required for formal runs so a
    mismatched loader cannot silently become the H0 anchor.
    """

    if args.smoke_test:
        r5._REFERENCE_FIRST_BATCH = {}
        return {"status": "skipped_for_smoke_test"}

    k3_first = r5._ORIGINAL_K3_PATHS(K_GROUP_OUTPUT_DIR, args.seed)["first_batch_audit"]
    r2_first = r2.r2_paths(R_GROUP_OUTPUT_DIR, args.seed, LAMBDA_R2)["first_batch_audit"]
    k3_rows = _reference_rows(k3_first)
    r2_rows = _reference_rows(r2_first)
    if k3_rows is None or r2_rows is None:
        raise FileNotFoundError(
            "H0 requires matched K3 and R2 first-batch audits for the same seed: "
            f"K3={k3_first}, R2={r2_first}"
        )
    r5._REFERENCE_FIRST_BATCH = {"K3": k3_rows, "R2": r2_rows}
    status = "matched_k3_r2_first_batch_loaded"
    return {
        "status": status,
        "comparison_seed": args.seed,
        "K3_first_batch_audit": str(k3_first.resolve()),
        "R2_first_batch_audit": str(r2_first.resolve()),
        "K3_first_batch_present": k3_rows is not None,
        "R2_first_batch_present": r2_rows is not None,
        "r5_expansion_gate": "not used by H0",
    }


def _resource_hashes_h0() -> Dict[str, object]:
    return {
        **r5.k3._resource_hashes(),
        "r2_relation_source_sha256": common.sha256_file(Path(r2.__file__).resolve()),
        "h0_training_script_sha256": common.sha256_file(Path(__file__).resolve()),
        "relation_spec_sha256": (
            None
            if r5._RELATION_SPEC is None
            else r2._canonical_sha256(r5._RELATION_SPEC)
        ),
        "activation_spec_sha256": _canonical_hash(_h0_activation_spec()),
    }


def build_config_h0(*args: Any, **kwargs: Any) -> Dict[str, object]:
    config = _ORIGINAL_R5_BUILD_CONFIG(*args, **kwargs)
    activation = _h0_activation_spec()
    config.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "server_entry_point": str(Path(__file__).resolve()),
            "activation": activation,
            "activation_name": "ReLU6",
            "activation_formula": "min(max(x, 0), 6)",
            "activation_spec_sha256": _canonical_hash(activation),
            "activation_replacement": {
                "name": "ReLU6",
                "replacement_count": 0,
                "replacement_paths": [],
                "scope": "none; original MobileNetV2 activations",
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


def audit_shapes_h0(model: base.MobileNetV2RASPPStudent, *args: Any, **kwargs: Any) -> Dict[str, object]:
    audit = _ORIGINAL_R5_AUDIT_SHAPES(model, *args, **kwargs)
    activation = _activation_audit(model)
    audit["experiment"] = EXPERIMENT
    audit["activation"] = activation
    audit["activation_spec_sha256"] = activation["activation_spec_sha256"]
    audit["r5_compatible_loss"] = True
    return audit


def build_best_checkpoint_h0(*args: Any, **kwargs: Any) -> Dict[str, object]:
    payload = _ORIGINAL_R5_BUILD_BEST_CHECKPOINT(*args, **kwargs)
    activation = _h0_activation_spec()
    payload.update(
        {
            "artifact_type": ARTIFACT_TYPE,
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "activation": activation,
            "activation_name": "ReLU6",
            "activation_formula": "min(max(x, 0), 6)",
            "activation_spec_sha256": _canonical_hash(activation),
            "initialization": "K-group shared scratch initialization; weights=None",
        }
    )
    payload["hashes"] = {**dict(payload.get("hashes", {})), **_resource_hashes_h0()}
    return payload


def train_one_epoch_h0(*args: Any, **kwargs: Any):
    metrics, steps, gradients, first_batch = _ORIGINAL_R5_TRAIN_ONE_EPOCH(
        *args, **kwargs
    )
    if isinstance(metrics, Mapping):
        metrics = dict(metrics)
        metrics["experiment"] = EXPERIMENT
    adjusted_gradients = []
    for record in gradients:
        adjusted = dict(record)
        adjusted["experiment"] = EXPERIMENT
        adjusted_gradients.append(adjusted)
    if isinstance(first_batch, Mapping):
        first_batch = dict(first_batch)
        first_batch["experiment"] = EXPERIMENT
        first_batch["activation_spec_sha256"] = _canonical_hash(_h0_activation_spec())
    return metrics, steps, adjusted_gradients, first_batch


def _patched_torch_save_atomic_h0(payload: object, path: Path) -> None:
    if isinstance(payload, Mapping) and payload.get("artifact_type") == ARTIFACT_TYPE:
        payload = dict(payload)
        activation = _h0_activation_spec()
        payload.update(
            {
                "experiment": EXPERIMENT,
                "experiment_group": EXPERIMENT_GROUP,
                "activation": activation,
                "activation_spec_sha256": _canonical_hash(activation),
                "hashes": {
                    **dict(payload.get("hashes", {})),
                    **_resource_hashes_h0(),
                },
            }
        )
    r5._ORIGINAL_TORCH_SAVE_ATOMIC(payload, path)


def _patched_evaluate_h0(*args: Any, **kwargs: Any):
    split_name = kwargs.get("split_name")
    if isinstance(split_name, str):
        kwargs["split_name"] = split_name.replace("K0", EXPERIMENT).replace(
            "K3", EXPERIMENT
        )
    return r5._ORIGINAL_EVALUATE(*args, **kwargs)


def _h0_print(*values: object, **kwargs: object) -> None:
    adjusted = tuple(
        value.replace("R5", EXPERIMENT).replace("K3", EXPERIMENT)
        if isinstance(value, str)
        else value
        for value in values
    )
    print(*adjusted, **kwargs)


def _postprocess_metrics_h0(args: argparse.Namespace) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    r5._ORIGINAL_K3_POSTPROCESS(args)
    metrics_path = h0_paths(args.output_dir, args.seed)["metrics"]
    if not metrics_path.is_file():
        return
    results = _read_json(metrics_path)
    activation = _h0_activation_spec()
    results.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "protocol": (
                "H0 matched ReLU6 anchor: K-group shared scratch MobileNetV2+R-ASPP, "
                "hard-label CE plus locked A0 feature MSE, native masked 8x16 R2 "
                "relation MSE, and frozen T1 full-resolution masked pixel KL "
                "(T=4); weights 1.0/0.3/0.5, shared 4000-step warm-up, 80k "
                "optimizer steps, dev_local selection, and no test_local evaluation."
            ),
            "activation": activation,
            "activation_name": "ReLU6",
            "activation_formula": "min(max(x, 0), 6)",
            "activation_spec_sha256": _canonical_hash(activation),
            "activation_replacement": {
                "name": "ReLU6",
                "replacement_count": 0,
                "replacement_paths": [],
                "scope": "none; original MobileNetV2 activations",
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
            "hashes": {**dict(results.get("hashes", {})), **_resource_hashes_h0()},
        }
    )
    activation_path = h0_paths(args.output_dir, args.seed)["activation_replacement"]
    common.write_json_atomic(
        activation_path,
        {
            **activation,
            "experiment": EXPERIMENT,
            "artifact_type": "activation_replacement_audit",
            "activation_spec_sha256": _canonical_hash(activation),
        },
    )
    efficiency_path = h0_paths(args.output_dir, args.seed)["efficiency"]
    if efficiency_path.is_file():
        efficiency = _read_json(efficiency_path)
        efficiency["activation"] = {
            "name": "ReLU6",
            "replacement_count": 0,
            "operator_fusion_candidate": "baseline ReLU6",
            "activation_spec_sha256": _canonical_hash(activation),
        }
        common.write_json_atomic(efficiency_path, efficiency)
    common.write_json_atomic(metrics_path, results)


def run_training(args: argparse.Namespace) -> None:
    """Install the H0 metadata/hooks and execute the shared R5 runner."""

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
    saved = {name: getattr(r5, name, None) for name in names}
    try:
        r5.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
        r5.EXPERIMENT = EXPERIMENT
        r5.EXPERIMENT_GROUP = EXPERIMENT_GROUP
        r5.ARTIFACT_TYPE = ARTIFACT_TYPE
        r5.ARTIFACT_FORMAT_VERSION = ARTIFACT_FORMAT_VERSION
        r5.r5_paths = h0_paths
        r5._validate_launch_gate = _validate_launch_gate_h0
        r5._resource_hashes = _resource_hashes_h0
        r5.build_config_r5 = build_config_h0
        r5.audit_shapes_r5 = audit_shapes_h0
        r5.build_best_checkpoint_r5 = build_best_checkpoint_h0
        r5.train_one_epoch_r5 = train_one_epoch_h0
        r5._patched_torch_save_atomic_r5 = _patched_torch_save_atomic_h0
        r5._patched_evaluate_r5 = _patched_evaluate_h0
        r5._r5_print = _h0_print
        r5._postprocess_metrics_r5 = _postprocess_metrics_h0
        r5.print = _h0_print
        r5.run_training(args)
    finally:
        for name, value in saved.items():
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
