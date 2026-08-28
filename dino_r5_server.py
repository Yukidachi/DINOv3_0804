"""R5 server entry point: feature, spatial-relation, and pixel-logit KD.

R5 is the gated cross-loss candidate registered after the R2 screening runs::

    L = L_CE + warmup(step) * (
        1.0 * L_feature + 0.3 * L_R2 + 0.5 * L_logit
    )

The feature target is K3's fixed A0 StandardScaler+PCA projection, R2 is the
native-feature masked 8x16 token-cosine objective, and the logit target is
K3/K2's full-resolution masked pixel KL at T=4 (including T**2).  One frozen
teacher backbone forward supplies all three targets.

The first formal run is seed 42.  Confirmation seeds require an explicit JSON
gate produced after the seed-42 mIoU, paired-bootstrap, and gradient reviews.

Typical two-GPU screening command::

    torchrun --standalone --nproc_per_node=2 dino_r5_server.py \
        --seed 42 --batch-size 2 --global-batch-size 8 \
        --num-workers 8 --multiprocessing-context spawn \
        --no-pin-memory --persistent-workers

Windows/local functional smoke (not a replacement for Linux two-GPU smoke)::

    python -B dino_r5_server.py --device cuda --smoke-test \
        --batch-size 1 --global-batch-size 1 --num-workers 0 \
        --no-persistent-workers --no-pin-memory --no-amp
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import dino as common
import dino_a0_server as a0
import dino_k0_server as k0
import dino_k1_server as k1
import dino_k2_server as k2
import dino_k3_server as k3
import dino_r1_server as r1
import dino_r2_server as r2
import dino_s2_0 as base
import dino_s2_0_server as server_base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "R_MobileNetV2_RASPP_server"
K_GROUP_OUTPUT_DIR = SCRIPT_DIR / "result" / "K_MobileNetV2_RASPP_server"
EXPERIMENT = "R5"
EXPERIMENT_GROUP = "R_MobileNetV2_RASPP_server"
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_r5_feature_relation_logit_kd"
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
AUXILIARY_CE_STOP_RATIO = 2.0
AUXILIARY_CE_STOP_CONSECUTIVE = 3
MIOU_EXPANSION_MARGIN = 0.00425
FIRST_BATCH_ABS_TOLERANCE = 1e-6

_REFERENCE_GATE: Optional[Dict[str, object]] = None
_REFERENCE_FIRST_BATCH: Dict[str, object] = {}
_RELATION_SPEC: Optional[Dict[str, object]] = None
_REFERENCE_TESTS: Optional[Dict[str, object]] = None
_AUXILIARY_GATE_CONSECUTIVE_EXCESS = 0

_ORIGINAL_K3_FILE = k3.__file__
_ORIGINAL_K3_EXPERIMENT = k3.EXPERIMENT
_ORIGINAL_K3_ARTIFACT_TYPE = k3.ARTIFACT_TYPE
_ORIGINAL_K3_ARTIFACT_FORMAT_VERSION = k3.ARTIFACT_FORMAT_VERSION
_ORIGINAL_K3_PATHS = k3.k3_paths
_ORIGINAL_K3_ENSURE_RESOURCES = k3.ensure_k3_resources
_ORIGINAL_K3_BUILD_CONFIG = k3.build_config
_ORIGINAL_K3_BUILD_BEST_CHECKPOINT = k3.build_best_checkpoint
_ORIGINAL_K3_TRAIN_ONE_EPOCH = k3.train_one_epoch_k3
_ORIGINAL_K3_SMOKE_TEST = k3.smoke_test_k3
_ORIGINAL_K3_AUDIT_SHAPES = k3.audit_k3_shapes
_ORIGINAL_K3_POSTPROCESS = k3._postprocess_metrics
_ORIGINAL_K3_PATCHED_SAVE = k3._patched_torch_save_atomic
_ORIGINAL_K3_PATCHED_EVALUATE = k3._patched_evaluate
_ORIGINAL_K3_PRINT = k3._k3_print
_ORIGINAL_TORCH_SAVE_ATOMIC = k3._ORIGINAL_TORCH_SAVE_ATOMIC
_ORIGINAL_EVALUATE = k3._ORIGINAL_EVALUATE


def parse_args() -> argparse.Namespace:
    """Reuse K3's locked CLI and add R5's fixed relation/gate options."""

    saved_default = k3.DEFAULT_OUTPUT_DIR
    saved_argparse = k3.argparse

    class R5ArgparseProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(saved_argparse, name)

        @staticmethod
        def ArgumentParser(*parser_args: Any, **parser_kwargs: Any):
            parser_kwargs["description"] = (
                "R5 MobileNetV2+R-ASPP: hard-label CE plus locked A0 feature "
                "KD, R2 spatial relation KD, and frozen T1 pixel-logit KD."
            )
            parser = saved_argparse.ArgumentParser(*parser_args, **parser_kwargs)
            parser.add_argument(
                "--lambda-r2",
                type=float,
                default=LAMBDA_R2,
                help="Fixed R2 spatial-relation weight (formal R5: 0.3).",
            )
            parser.add_argument(
                "--expansion-gate",
                type=Path,
                default=None,
                help=(
                    "Optional seed-42 R5 expansion-gate JSON. It is validated "
                    "when supplied for a non-seed-42 run."
                ),
            )
            parser.add_argument(
                "--enforce-expansion-gate",
                action=argparse.BooleanOptionalAction,
                default=False,
                help=(
                    "Require --expansion-gate for seed 3407/260805. By default "
                    "R5 formal seeds run independently after matched K3/R2 "
                    "reference checks."
                ),
            )
            return parser

    k3.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    k3.argparse = R5ArgparseProxy()
    try:
        args = k3.parse_args()
    finally:
        k3.DEFAULT_OUTPUT_DIR = saved_default
        k3.argparse = saved_argparse

    if args.seed not in FORMAL_SEEDS:
        raise SystemExit(f"R5 seed must be one of {FORMAL_SEEDS}")
    if not math.isclose(args.lambda_r2, LAMBDA_R2, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit("Formal R5 is locked to --lambda-r2 0.3")
    if args.output_dir.resolve() == K_GROUP_OUTPUT_DIR.resolve():
        raise SystemExit("R5 output must use the separate R-group output directory")
    if not args.smoke_test:
        if args.max_steps != 80_000:
            raise SystemExit("Formal R5 is locked to exactly 80,000 optimizer steps")
        if args.eval_every_steps != 5_000:
            raise SystemExit("Formal R5 is locked to --eval-every-steps 5000")
        if args.gradient_log_steps != GRADIENT_LOG_STEPS:
            raise SystemExit("Formal R5 is locked to --gradient-log-steps 500")
        if (
            args.seed != SCREENING_SEED
            and args.enforce_expansion_gate
            and args.expansion_gate is None
        ):
            raise SystemExit(
                "R5 expansion-gate enforcement requires --expansion-gate for "
                "seed 3407/260805"
            )
    return args


def r5_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    original = k3._ORIGINAL_K0_PATHS(output_dir, seed)
    run_dir = output_dir.resolve() / EXPERIMENT / f"seed_{seed}"
    return {
        key: run_dir if key == "run_dir" else run_dir / value.name
        for key, value in original.items()
    }


def _read_json(path: Path) -> Dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite: {result}")
    return result


def _best_miou(metrics: Mapping[str, object], label: str) -> float:
    best = metrics.get("best_dev_metrics")
    if not isinstance(best, Mapping):
        raise RuntimeError(f"{label} has no best_dev_metrics object")
    return _number(best.get("mIoU"), f"{label} best dev mIoU")


def _rank_rows(path: Path) -> Dict[int, Dict[str, object]]:
    audit = _read_json(path)
    rows = audit.get("per_rank")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise RuntimeError(f"First-batch audit has no per_rank rows: {path}")
    result: Dict[int, Dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError(f"Malformed first-batch row: {path}")
        rank = int(row.get("rank", -1))
        result[rank] = dict(row)
    return result


def _validate_r2_confirmation(args: argparse.Namespace) -> Dict[str, object]:
    """Require the completed, positive matched-seed R2 lambda=0.3 screen."""

    rows: List[Dict[str, object]] = []
    failures: List[str] = []
    for seed in FORMAL_SEEDS:
        r2_path = r2.r2_paths(args.output_dir, seed, LAMBDA_R2)["metrics"]
        k1_path = k1.k1_paths(K_GROUP_OUTPUT_DIR, seed)["metrics"]
        if not r2_path.is_file():
            failures.append(f"missing R2 seed {seed} metrics: {r2_path}")
            continue
        if not k1_path.is_file():
            failures.append(f"missing matched K1 seed {seed} metrics: {k1_path}")
            continue
        r2_metrics = _read_json(r2_path)
        k1_metrics = _read_json(k1_path)
        if r2_metrics.get("experiment") != "R2":
            failures.append(f"seed {seed} R2 artifact has wrong experiment")
        if r2_metrics.get("test_local_evaluated") is not False:
            failures.append(f"seed {seed} R2 does not lock test_local")
        loss = r2_metrics.get("loss")
        if not isinstance(loss, Mapping) or not math.isclose(
            _number(loss.get("lambda_r2"), f"R2 seed {seed} lambda"),
            LAMBDA_R2,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            failures.append(f"seed {seed} R2 is not the lambda=0.3 candidate")
        reference = r2_metrics.get("relation_reference_tests")
        if not isinstance(reference, Mapping) or not bool(reference.get("passed")):
            failures.append(f"seed {seed} R2 relation reference tests did not pass")
        gradient = r2_metrics.get("gradient_gate")
        if not isinstance(gradient, Mapping) or not bool(
            gradient.get("passed_target_at_any_record")
        ):
            failures.append(f"seed {seed} R2 gradient gate did not pass")
        r2_miou = _best_miou(r2_metrics, f"R2 seed {seed}")
        k1_miou = _best_miou(k1_metrics, f"K1 seed {seed}")
        delta = r2_miou - k1_miou
        if delta <= 0.0:
            failures.append(
                f"R2 seed {seed} is not positive versus matched K1: {delta:+.6f}"
            )
        rows.append(
            {
                "seed": seed,
                "r2_metrics": str(r2_path.resolve()),
                "r2_metrics_sha256": common.sha256_file(r2_path),
                "k1_metrics": str(k1_path.resolve()),
                "k1_metrics_sha256": common.sha256_file(k1_path),
                "r2_mIoU": r2_miou,
                "k1_mIoU": k1_miou,
                "delta_R2_minus_K1": delta,
            }
        )
    if failures:
        raise RuntimeError("R5 R2-confirmation gate failed:\n- " + "\n- ".join(failures))
    return {
        "passed": True,
        "requirement": "R2 lambda=0.3 is positive versus matched K1 for all formal seeds",
        "rows": rows,
    }


def _validate_expansion_gate(
    args: argparse.Namespace, k3_miou: float, r2_miou: float
) -> Dict[str, object]:
    assert args.expansion_gate is not None
    path = args.expansion_gate.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"R5 expansion gate not found: {path}")
    gate = _read_json(path)
    screening_metrics_path = r5_paths(args.output_dir, SCREENING_SEED)["metrics"]
    if not screening_metrics_path.is_file():
        raise FileNotFoundError(
            f"R5 seed-42 metrics required by expansion gate: {screening_metrics_path}"
        )
    screening_metrics = _read_json(screening_metrics_path)
    actual_hash = common.sha256_file(screening_metrics_path)
    screening_miou = _best_miou(screening_metrics, "R5 seed 42")
    expected_gain = screening_miou - max(k3_miou, r2_miou)
    failures: List[str] = []
    if gate.get("experiment") != EXPERIMENT or int(gate.get("screening_seed", -1)) != 42:
        failures.append("gate is not an R5 seed-42 expansion decision")
    if gate.get("passed") is not True:
        failures.append("gate does not explicitly record passed=true")
    if gate.get("r5_metrics_sha256") != actual_hash:
        failures.append("gate R5 metrics hash differs from the current seed-42 metrics")
    recorded_gain = _number(
        gate.get("miou_gain_over_best_reference"), "expansion-gate mIoU gain"
    )
    if not math.isclose(recorded_gain, expected_gain, rel_tol=0.0, abs_tol=1e-12):
        failures.append("gate mIoU gain does not reproduce current metrics")
    if recorded_gain <= MIOU_EXPANSION_MARGIN:
        failures.append(
            f"mIoU gain {recorded_gain:.6f} does not exceed {MIOU_EXPANSION_MARGIN:.5f}"
        )
    ci_lower = _number(
        gate.get("paired_bootstrap_ci_lower"), "expansion-gate paired CI lower"
    )
    if ci_lower <= 0.0:
        failures.append("paired-bootstrap 95% CI lower bound is not above zero")
    if gate.get("gradient_gate_passed") is not True:
        failures.append("gradient gate is not recorded as passed")
    if failures:
        raise RuntimeError("R5 expansion gate failed:\n- " + "\n- ".join(failures))
    return {
        "passed": True,
        "path": str(path),
        "sha256": common.sha256_file(path),
        "screening_metrics": str(screening_metrics_path.resolve()),
        "screening_metrics_sha256": actual_hash,
        "miou_gain_over_best_reference": recorded_gain,
        "paired_bootstrap_ci_lower": ci_lower,
        "gradient_gate_passed": True,
    }


def _validate_independent_seed_references(
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Validate only the matched K3/R2 artifacts for the current R5 seed.

    R5 seed 3407/260805 is a formally pre-registered extension of the same
    objective, not a conditional continuation that must wait for a seed-42
    expansion decision.  The matched references are still required because
    they provide the same-seed first-batch audit and the controlled R5-K3/R2
    comparisons.
    """

    global _REFERENCE_FIRST_BATCH
    seed = int(args.seed)
    k3_path = _ORIGINAL_K3_PATHS(K_GROUP_OUTPUT_DIR, seed)["metrics"]
    r2_path = r2.r2_paths(args.output_dir, seed, LAMBDA_R2)["metrics"]
    k3_first = _ORIGINAL_K3_PATHS(K_GROUP_OUTPUT_DIR, seed)["first_batch_audit"]
    r2_first = r2.r2_paths(args.output_dir, seed, LAMBDA_R2)["first_batch_audit"]
    missing = [
        str(path)
        for path in (k3_path, r2_path, k3_first, r2_first)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "R5 independent seed requires matched K3/R2 artifacts:\n- "
            + "\n- ".join(missing)
        )

    k3_metrics = _read_json(k3_path)
    r2_metrics = _read_json(r2_path)
    if k3_metrics.get("experiment") != "K3":
        raise RuntimeError(f"Matched K3 seed {seed} artifact has the wrong experiment")
    if r2_metrics.get("experiment") != "R2":
        raise RuntimeError(f"Matched R2 seed {seed} artifact has the wrong experiment")
    if k3_metrics.get("test_local_evaluated") is not False:
        raise RuntimeError(f"Matched K3 seed {seed} evaluated test_local")
    if r2_metrics.get("test_local_evaluated") is not False:
        raise RuntimeError(f"Matched R2 seed {seed} evaluated test_local")
    k3_loss = k3_metrics.get("loss")
    r2_loss = r2_metrics.get("loss")
    if not isinstance(k3_loss, Mapping):
        raise RuntimeError(f"Matched K3 seed {seed} has no loss contract")
    if not isinstance(r2_loss, Mapping):
        raise RuntimeError(f"Matched R2 seed {seed} has no loss contract")
    for key, expected in (
        ("lambda_feat", LAMBDA_FEAT),
        ("lambda_logit", LAMBDA_LOGIT),
        ("temperature", TEMPERATURE),
    ):
        if not math.isclose(
            _number(k3_loss.get(key), f"K3 seed {seed} {key}"),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Matched K3 seed {seed} changed {key}")
    if not math.isclose(
        _number(r2_loss.get("lambda_r2"), f"R2 seed {seed} lambda_r2"),
        LAMBDA_R2,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"Matched R2 seed {seed} is not lambda_r2=0.3")

    _REFERENCE_FIRST_BATCH = {
        "K3": _rank_rows(k3_first),
        "R2": _rank_rows(r2_first),
    }
    k3_miou = _best_miou(k3_metrics, f"K3 seed {seed}")
    r2_miou = _best_miou(r2_metrics, f"R2 seed {seed}")

    expansion = None
    if args.expansion_gate is not None:
        # An expansion-gate file, when explicitly supplied, always refers to
        # the original seed-42 decision and is validated against seed-42 K3/R2
        # reference mIoUs, even during an independent seed run.
        k3_seed42_path = _ORIGINAL_K3_PATHS(K_GROUP_OUTPUT_DIR, SCREENING_SEED)[
            "metrics"
        ]
        r2_seed42_path = r2.r2_paths(
            args.output_dir, SCREENING_SEED, LAMBDA_R2
        )["metrics"]
        if not k3_seed42_path.is_file() or not r2_seed42_path.is_file():
            raise FileNotFoundError(
                "An explicit R5 expansion gate requires seed-42 K3/R2 metrics"
            )
        expansion = _validate_expansion_gate(
            args,
            _best_miou(_read_json(k3_seed42_path), "K3 seed 42"),
            _best_miou(_read_json(r2_seed42_path), "R2 seed 42"),
        )
    return {
        "passed": True,
        "launch_mode": "independent_formal_seed",
        "comparison_seed": seed,
        "screening_seed": SCREENING_SEED,
        "references": {
            "K3": {
                "metrics": str(k3_path.resolve()),
                "metrics_sha256": common.sha256_file(k3_path),
                "first_batch_audit": str(k3_first.resolve()),
                "first_batch_audit_sha256": common.sha256_file(k3_first),
                "mIoU": k3_miou,
            },
            "R2": {
                "metrics": str(r2_path.resolve()),
                "metrics_sha256": common.sha256_file(r2_path),
                "first_batch_audit": str(r2_first.resolve()),
                "first_batch_audit_sha256": common.sha256_file(r2_first),
                "mIoU": r2_miou,
                "lambda_r2": LAMBDA_R2,
            },
        },
        "best_reference": "K3" if k3_miou >= r2_miou else "R2",
        "best_reference_mIoU": max(k3_miou, r2_miou),
        "required_seed42_gain": MIOU_EXPANSION_MARGIN,
        "r2_three_seed_confirmation": {
            "required_for_launch": False,
            "status": "not_used_in_independent_seed_mode",
        },
        "confirmation_expansion_gate": expansion,
    }


def _validate_launch_gate(args: argparse.Namespace) -> Dict[str, object]:
    global _REFERENCE_FIRST_BATCH
    if args.smoke_test:
        _REFERENCE_FIRST_BATCH = {}
        return {"status": "skipped_for_smoke_test"}

    if args.seed != SCREENING_SEED:
        return _validate_independent_seed_references(args)

    r2_confirmation = _validate_r2_confirmation(args)
    k3_path = _ORIGINAL_K3_PATHS(K_GROUP_OUTPUT_DIR, SCREENING_SEED)["metrics"]
    r2_path = r2.r2_paths(args.output_dir, SCREENING_SEED, LAMBDA_R2)["metrics"]
    if not k3_path.is_file() or not r2_path.is_file():
        raise FileNotFoundError(
            "R5 requires completed seed-42 K3 and R2 metrics: "
            f"K3={k3_path}, R2={r2_path}"
        )
    k3_metrics = _read_json(k3_path)
    r2_metrics = _read_json(r2_path)
    if k3_metrics.get("experiment") != "K3":
        raise RuntimeError("R5 K3 reference has the wrong experiment identifier")
    if k3_metrics.get("test_local_evaluated") is not False:
        raise RuntimeError("R5 K3 reference does not lock test_local")
    k3_loss = k3_metrics.get("loss")
    if not isinstance(k3_loss, Mapping):
        raise RuntimeError("R5 K3 reference has no loss contract")
    expected = {
        "lambda_feat": LAMBDA_FEAT,
        "lambda_logit": LAMBDA_LOGIT,
        "temperature": TEMPERATURE,
    }
    for key, value in expected.items():
        if not math.isclose(
            _number(k3_loss.get(key), f"K3 {key}"), value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"R5 K3 reference changed {key}")

    comparison_seed = int(args.seed)
    k3_first = _ORIGINAL_K3_PATHS(K_GROUP_OUTPUT_DIR, comparison_seed)[
        "first_batch_audit"
    ]
    r2_first = r2.r2_paths(args.output_dir, comparison_seed, LAMBDA_R2)[
        "first_batch_audit"
    ]
    if not k3_first.is_file() or not r2_first.is_file():
        raise FileNotFoundError(
            f"R5 requires K3/R2 first-batch audits: K3={k3_first}, R2={r2_first}"
        )
    _REFERENCE_FIRST_BATCH = {
        "K3": _rank_rows(k3_first),
        "R2": _rank_rows(r2_first),
    }
    k3_miou = _best_miou(k3_metrics, "K3 seed 42")
    r2_miou = _best_miou(r2_metrics, "R2 seed 42")
    expansion = None
    if args.seed != SCREENING_SEED:
        expansion = _validate_expansion_gate(args, k3_miou, r2_miou)
    return {
        "passed": True,
        "screening_seed": SCREENING_SEED,
        "first_batch_comparison_seed": comparison_seed,
        "r2_three_seed_confirmation": r2_confirmation,
        "references": {
            "K3": {
                "metrics": str(k3_path.resolve()),
                "metrics_sha256": common.sha256_file(k3_path),
                "first_batch_audit": str(k3_first.resolve()),
                "first_batch_audit_sha256": common.sha256_file(k3_first),
                "mIoU": k3_miou,
            },
            "R2": {
                "metrics": str(r2_path.resolve()),
                "metrics_sha256": common.sha256_file(r2_path),
                "first_batch_audit": str(r2_first.resolve()),
                "first_batch_audit_sha256": common.sha256_file(r2_first),
                "mIoU": r2_miou,
                "lambda_r2": LAMBDA_R2,
            },
        },
        "best_reference": "K3" if k3_miou >= r2_miou else "R2",
        "best_reference_mIoU": max(k3_miou, r2_miou),
        "required_seed42_gain": MIOU_EXPANSION_MARGIN,
        "confirmation_expansion_gate": expansion,
    }


def _resource_hashes() -> Dict[str, object]:
    return {
        **k3._resource_hashes(),
        "r2_relation_source_sha256": common.sha256_file(Path(r2.__file__).resolve()),
        "r5_training_script_sha256": common.sha256_file(Path(__file__).resolve()),
        "relation_spec_sha256": (
            None if _RELATION_SPEC is None else r2._canonical_sha256(_RELATION_SPEC)
        ),
    }


def ensure_r5_resources(
    model: base.MobileNetV2RASPPStudent,
    args: argparse.Namespace,
    _output_dir: Path,
    seed: int,
    rank: int,
    world_size: int,
) -> Tuple[str, str, Path]:
    """Load the K-group shared scratch state; never create an R-specific init."""

    init_path = k0._shared_init_path(K_GROUP_OUTPUT_DIR, seed)
    if not init_path.is_file() or not init_path.with_suffix(
        init_path.suffix + ".sha256"
    ).is_file():
        raise FileNotFoundError(
            "R5 requires the existing K-group shared initialization: "
            f"{init_path}"
        )
    result = k1.ensure_k1_resources(
        model, args, K_GROUP_OUTPUT_DIR, seed, rank, world_size
    )
    teacher, projection = k1._require_resources()
    if k1._TEACHER_CHECKPOINT_SHA256 != k1.EXPECTED_TEACHER_CHECKPOINT_SHA256:
        raise RuntimeError("R5 teacher checkpoint differs from the locked T1")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("R5 teacher is not fully frozen")
    if list(projection.parameters()):
        raise RuntimeError("R5 fixed A0 projection unexpectedly has parameters")
    return result


def _relation_spec_r5(
    args: argparse.Namespace, accumulation_steps: int, world_size: int
) -> Dict[str, object]:
    spec = copy.deepcopy(r2._relation_spec_r2(args, accumulation_steps, world_size))
    spec["relation_warmup_shared_with_feature_kd"] = True
    spec["relation_warmup_shared_with_logit_kd"] = True
    spec["r5_logit_kd_enabled"] = True
    return spec


def build_config_r5(
    args: argparse.Namespace,
    accumulation_steps: int,
    world_size: int,
    device: torch.device,
    shared_init_state_sha256: str,
    shared_init_file_sha256: str,
) -> Dict[str, object]:
    global _RELATION_SPEC, _REFERENCE_TESTS
    config = _ORIGINAL_K3_BUILD_CONFIG(
        args,
        accumulation_steps,
        world_size,
        device,
        shared_init_state_sha256,
        shared_init_file_sha256,
    )
    _RELATION_SPEC = _relation_spec_r5(args, accumulation_steps, world_size)
    _REFERENCE_TESTS = r2.run_r2_reference_tests(device, world_size)
    if not args.smoke_test:
        physical_batch = int(args.batch_size) * int(world_size)
        if world_size != 2:
            raise RuntimeError(f"Formal R5 requires world_size=2, got {world_size}")
        if physical_batch != 4 or physical_batch * accumulation_steps != 8:
            raise RuntimeError(
                "Formal R5 requires physical relation batch 4 and optimizer batch 8"
            )
    config.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "server_entry_point": str(Path(__file__).resolve()),
            "formal_seeds": list(FORMAL_SEEDS),
            "screening_seed": SCREENING_SEED,
            "screening_or_confirmation": (
                "screening"
                if args.seed == SCREENING_SEED
                else "independent_formal_seed"
            ),
            "shared_initialization": {
                "source_group": "K_MobileNetV2_RASPP_server",
                "path": str(k0._shared_init_path(K_GROUP_OUTPUT_DIR, args.seed).resolve()),
                "state_sha256": shared_init_state_sha256,
                "file_sha256": shared_init_file_sha256,
                "r_specific_initialization_created": False,
            },
            "relation": copy.deepcopy(_RELATION_SPEC),
            "relation_spec_sha256": r2._canonical_sha256(_RELATION_SPEC),
            "relation_reference_tests": copy.deepcopy(_REFERENCE_TESTS),
            "launch_gate": copy.deepcopy(_REFERENCE_GATE),
            "gradient_audit": {
                "tap": "student native os16",
                "interval_optimizer_steps": args.gradient_log_steps,
                "fixed_steps": list(FIXED_GRADIENT_AUDIT_STEPS),
                "raw_components": ["CE", "feature", "R2", "logit"],
                "effective_auxiliary_components": ["feature", "R2", "logit"],
                "auxiliary_pairwise_cosines": [
                    "feature_R2",
                    "feature_logit",
                    "R2_logit",
                ],
                "stop_if_total_auxiliary_to_ce_exceeds": AUXILIARY_CE_STOP_RATIO,
                "consecutive_records_before_stop": AUXILIARY_CE_STOP_CONSECUTIVE,
            },
        }
    )
    loss = dict(config.get("loss", {}))
    loss.update(
        {
            "hard_label_ce": True,
            "feature_kd": True,
            "relation_kd": True,
            "relation_r1": False,
            "relation_r2": True,
            "logit_kd": True,
            "lambda_feat": args.lambda_feat,
            "lambda_r2": args.lambda_r2,
            "lambda_logit": args.lambda_logit,
            "temperature": args.temperature,
            "total": (
                "CE + warmup * (lambda_feat * feature + lambda_r2 * R2 + "
                "lambda_logit * logits_KL_T2)"
            ),
        }
    )
    config["loss"] = loss
    return config


def audit_shapes_r5(
    model: base.MobileNetV2RASPPStudent,
    device: torch.device,
    height: int,
    width: int,
    amp_enabled: bool,
) -> Dict[str, object]:
    audit = _ORIGINAL_K3_AUDIT_SHAPES(model, device, height, width, amp_enabled)
    audit["experiment"] = EXPERIMENT
    audit["relation"] = {
        "enabled": True,
        "type": "R2_within_image_spatial",
        "native_teacher_student_taps": list(a0.A0_LAYER_ORDER),
        "pool_size": list(r2.POOL_SIZE),
        "token_count": r2.NUM_TOKENS,
        "matrix_shape": [r2.NUM_TOKENS, r2.NUM_TOKENS],
        "matrix_dtype": "float32",
        "a0_projection_used_only_by_pointwise_feature_anchor": True,
    }
    audit["teacher_forward_shared_by_feature_relation_logit"] = True
    return audit


def build_best_checkpoint_r5(*args: Any, **kwargs: Any) -> Dict[str, object]:
    payload = _ORIGINAL_K3_BUILD_BEST_CHECKPOINT(*args, **kwargs)
    payload.update(
        {
            "artifact_type": ARTIFACT_TYPE,
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "relation": copy.deepcopy(_RELATION_SPEC),
            "relation_spec_sha256": (
                None
                if _RELATION_SPEC is None
                else r2._canonical_sha256(_RELATION_SPEC)
            ),
            "relation_reference_tests": copy.deepcopy(_REFERENCE_TESTS),
            "launch_gate": copy.deepcopy(_REFERENCE_GATE),
            "loss_schema": {
                "hard_label_ce": True,
                "feature_kd": True,
                "relation_r2": True,
                "logit_kd": True,
                "lambda_feat": LAMBDA_FEAT,
                "lambda_r2": LAMBDA_R2,
                "lambda_logit": LAMBDA_LOGIT,
                "temperature": TEMPERATURE,
            },
        }
    )
    payload["hashes"] = {**dict(payload.get("hashes", {})), **_resource_hashes()}
    return payload


def _patched_torch_save_atomic_r5(payload: object, path: Path) -> None:
    if isinstance(payload, Mapping) and payload.get("artifact_type") == ARTIFACT_TYPE:
        payload = dict(payload)
        payload.update(
            {
                "experiment": EXPERIMENT,
                "experiment_group": EXPERIMENT_GROUP,
                "relation": copy.deepcopy(_RELATION_SPEC),
                "relation_spec_sha256": (
                    None
                    if _RELATION_SPEC is None
                    else r2._canonical_sha256(_RELATION_SPEC)
                ),
                "relation_reference_tests": copy.deepcopy(_REFERENCE_TESTS),
                "launch_gate": copy.deepcopy(_REFERENCE_GATE),
                "hashes": {
                    **dict(payload.get("hashes", {})),
                    **_resource_hashes(),
                },
            }
        )
    _ORIGINAL_TORCH_SAVE_ATOMIC(payload, path)


def _patched_evaluate_r5(*args: Any, **kwargs: Any):
    split_name = kwargs.get("split_name")
    if isinstance(split_name, str):
        kwargs["split_name"] = split_name.replace("K0", EXPERIMENT).replace(
            "K3", EXPERIMENT
        )
    return _ORIGINAL_EVALUATE(*args, **kwargs)


def _r5_print(*values: object, **kwargs: object) -> None:
    adjusted = tuple(
        value.replace("K0", EXPERIMENT).replace("K3", EXPERIMENT)
        if isinstance(value, str)
        else value
        for value in values
    )
    builtins.print(*adjusted, **kwargs)


def _gradient_l2(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm(2).item())


def _gradient_cosine(first: torch.Tensor, second: torch.Tensor) -> Optional[float]:
    a = first.detach().float().reshape(-1)
    b = second.detach().float().reshape(-1)
    denominator = float(a.norm().item() * b.norm().item())
    if denominator <= 1e-12:
        return None
    return float(torch.dot(a, b).item() / denominator)


def _mean_std(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not finite:
        return {"mean": None, "sample_std": None}
    mean = sum(finite) / len(finite)
    sample_std = (
        0.0
        if len(finite) < 2
        else math.sqrt(sum((value - mean) ** 2 for value in finite) / (len(finite) - 1))
    )
    return {"mean": mean, "sample_std": sample_std}


def _aggregate_gradient_record(
    local_record: Dict[str, object], world_size: int
) -> Dict[str, object]:
    rows: List[Optional[Dict[str, object]]] = [None for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather_object(rows, local_record)
    else:
        rows[0] = local_record
    valid_rows = [row for row in rows if row is not None]
    if len(valid_rows) != world_size:
        raise RuntimeError("R5 failed to gather every rank's gradient audit")
    summary = dict(local_record)
    summary["rank_aggregation"] = "mean across ranks; sample_std and per_rank retained"
    summary["per_rank"] = valid_rows
    numeric_fields = (
        "grad_l2_ce_os16",
        "grad_l2_feature_raw_os16",
        "grad_l2_r2_raw_os16",
        "grad_l2_logit_raw_os16",
        "grad_l2_feature_effective_os16",
        "grad_l2_r2_effective_os16",
        "grad_l2_logit_effective_os16",
        "grad_l2_total_auxiliary_effective_os16",
        "grad_l2_total_effective_os16",
        "grad_l2_total_student",
        "total_auxiliary_to_ce_effective_ratio_os16",
        "cos_feature_r2_os16",
        "cos_feature_logit_os16",
        "cos_r2_logit_os16",
        "cos_ce_feature_os16",
        "cos_ce_r2_os16",
        "cos_ce_logit_os16",
    )
    for field in numeric_fields:
        stats = _mean_std([row.get(field) for row in valid_rows])  # type: ignore[arg-type]
        summary[field] = stats["mean"]
        summary[f"{field}_sample_std"] = stats["sample_std"]
    return summary


def _update_auxiliary_stop_gate(record: Mapping[str, object]) -> None:
    global _AUXILIARY_GATE_CONSECUTIVE_EXCESS
    ratio = record.get("total_auxiliary_to_ce_effective_ratio_os16")
    if ratio is not None and float(ratio) > AUXILIARY_CE_STOP_RATIO:
        _AUXILIARY_GATE_CONSECUTIVE_EXCESS += 1
    else:
        _AUXILIARY_GATE_CONSECUTIVE_EXCESS = 0
    if _AUXILIARY_GATE_CONSECUTIVE_EXCESS >= AUXILIARY_CE_STOP_CONSECUTIVE:
        raise RuntimeError(
            "R5 total effective auxiliary gradient exceeded 2x CE for three "
            "consecutive records; stop and inspect the combined objective"
        )


def _restore_auxiliary_gate_state(args: argparse.Namespace) -> None:
    global _AUXILIARY_GATE_CONSECUTIVE_EXCESS
    _AUXILIARY_GATE_CONSECUTIVE_EXCESS = 0
    if not args.resume:
        return
    path = r5_paths(args.output_dir, args.seed)["gradient_norms"]
    if not path.is_file():
        return
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    for line in lines[-AUXILIARY_CE_STOP_CONSECUTIVE:]:
        record = json.loads(line)
        ratio = record.get("total_auxiliary_to_ce_effective_ratio_os16")
        if ratio is not None and float(ratio) > AUXILIARY_CE_STOP_RATIO:
            _AUXILIARY_GATE_CONSECUTIVE_EXCESS += 1
        else:
            _AUXILIARY_GATE_CONSECUTIVE_EXCESS = 0


def _compare_first_batch(
    audit: Mapping[str, object], rank: int
) -> Dict[str, object]:
    if not _REFERENCE_FIRST_BATCH:
        return {"passed": True, "status": "skipped_for_smoke_test"}
    exact_fields = ("paths", "image_tensor_sha256", "target_tensor_sha256")
    scalar_contract = {
        "K3": ("ce_loss", "feature_loss", "logit_loss"),
        "R2": ("ce_loss", "feature_loss", "relation_r2_loss"),
    }
    mismatches: List[Dict[str, object]] = []
    for experiment, fields in scalar_contract.items():
        rows = _REFERENCE_FIRST_BATCH.get(experiment)
        if not isinstance(rows, Mapping) or rank not in rows:
            mismatches.append({"reference": experiment, "field": "rank", "rank": rank})
            continue
        reference = rows[rank]
        if not isinstance(reference, Mapping):
            mismatches.append({"reference": experiment, "field": "row"})
            continue
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
        for field in fields:
            actual = _number(audit.get(field), f"R5 first batch {field}")
            expected = _number(reference.get(field), f"{experiment} first batch {field}")
            if not math.isclose(
                actual, expected, rel_tol=0.0, abs_tol=FIRST_BATCH_ABS_TOLERANCE
            ):
                mismatches.append(
                    {
                        "reference": experiment,
                        "field": field,
                        "actual": actual,
                        "expected": expected,
                        "absolute_error": abs(actual - expected),
                    }
                )
    return {
        "passed": not mismatches,
        "comparison": (
            "R5 CE/feature/logit versus K3 and CE/feature/R2 versus R2 on the "
            "same matched-seed first micro-batch"
        ),
        "absolute_tolerance": FIRST_BATCH_ABS_TOLERANCE,
        "mismatches": mismatches,
    }


def _reduce_training_statistics(
    feature_layer_sums: Mapping[str, float],
    relation_layer_sums: Mapping[str, float],
    feature_sum: float,
    relation_sum: float,
    logit_sum: float,
    total_sum: float,
    valid_token_sum: float,
    valid_pair_sum: float,
    physical_batch_sum: float,
    batch_count: int,
    device: torch.device,
    world_size: int,
) -> Tuple[Dict[str, float], Dict[str, float], List[float], int]:
    values = [feature_layer_sums[layer] for layer in a0.A0_LAYER_ORDER]
    values.extend(relation_layer_sums[layer] for layer in a0.A0_LAYER_ORDER)
    values.extend(
        [
            feature_sum,
            relation_sum,
            logit_sum,
            total_sum,
            valid_token_sum,
            valid_pair_sum,
            physical_batch_sum,
            float(batch_count),
        ]
    )
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    count = int(tensor[-1].item())
    denominator = max(count, 1)
    feature_means = {
        layer: float(tensor[index].item() / denominator)
        for index, layer in enumerate(a0.A0_LAYER_ORDER)
    }
    offset = len(a0.A0_LAYER_ORDER)
    relation_means = {
        layer: float(tensor[offset + index].item() / denominator)
        for index, layer in enumerate(a0.A0_LAYER_ORDER)
    }
    tail = [float(value.item() / denominator) for value in tensor[-8:-1]]
    return feature_means, relation_means, tail, count


def train_one_epoch_r5(
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
    teacher, projection = k1._require_resources()
    args = k3._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("R5 active arguments were not set")
    warmup_steps = k3._warmup_steps(args)
    if sampler is not None:
        sampler.set_epoch(epoch)
    model.train()
    teacher.eval()
    projection.eval()
    optimizer.zero_grad(set_to_none=True)

    confusion = torch.zeros(common.NUM_CLASSES, common.NUM_CLASSES, dtype=torch.int64)
    ce_loss_sum = 0.0
    valid_pixels = 0
    feature_sum = 0.0
    relation_sum = 0.0
    logit_sum = 0.0
    total_sum = 0.0
    valid_token_sum = 0.0
    valid_pair_sum = 0.0
    physical_batch_sum = 0.0
    batch_count = 0
    optimizer_steps = 0
    last_warmup_weight = 0.0
    feature_layer_sums = {layer: 0.0 for layer in a0.A0_LAYER_ORDER}
    relation_layer_sums = {layer: 0.0 for layer in a0.A0_LAYER_ORDER}
    gradient_records: List[Dict[str, object]] = []
    first_batch_audit: Optional[Dict[str, object]] = None

    possible_steps = math.ceil(len(loader) / accumulation_steps)
    target_steps = min(possible_steps, remaining_optimizer_steps)
    max_batches = min(len(loader), target_steps * accumulation_steps)
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch} [R5 CE+feature+R2+logit]",
        disable=rank != 0,
    )

    for batch_index, (images, targets, paths) in enumerate(progress):
        if batch_index >= max_batches:
            break
        group_position = batch_index % accumulation_steps
        if group_position == 0:
            group_size = min(accumulation_steps, max_batches - batch_index)
        sync_gradients = group_position + 1 == group_size
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        next_optimizer_step = starting_optimizer_step + optimizer_steps + 1
        warmup_weight = min(1.0, next_optimizer_step / warmup_steps)

        sync_context = contextlib.nullcontext()
        if isinstance(model, DDP) and not sync_gradients:
            sync_context = model.no_sync()
        with sync_context:
            with common.autocast_context(device, amp_enabled):
                student_output = model(images)
                if not isinstance(student_output, Mapping):
                    raise RuntimeError("R5 training forward did not return features")
                logits = student_output["logits"]
                student_features = student_output["features"]
                with torch.no_grad():
                    teacher_features, teacher_logits = k3._teacher_features_and_logits(
                        teacher, images
                    )
                feature_layer_losses, projected_shapes = k3._feature_kd_losses(
                    student_features, teacher_features, projection
                )

            logits_float = logits.float()
            batch_ce_sum = F.cross_entropy(
                logits_float,
                targets,
                ignore_index=common.IGNORE_INDEX,
                reduction="sum",
            )
            batch_valid = int((targets != common.IGNORE_INDEX).sum().item())
            if batch_valid == 0:
                raise RuntimeError("R5 training batch contains no valid Cityscapes pixels")
            loss_ce = batch_ce_sum / batch_valid
            loss_feature = sum(feature_layer_losses.values()) / len(a0.A0_LAYER_ORDER)
            loss_r2, relation_layer_losses, relation_audit = r2.r2_relation_losses(
                student_features, teacher_features, targets, world_size
            )
            loss_logit = k2._masked_pixel_kl(
                teacher_logits, logits_float, targets, args.temperature
            )
            total_loss = loss_ce + warmup_weight * (
                args.lambda_feat * loss_feature
                + args.lambda_r2 * loss_r2
                + args.lambda_logit * loss_logit
            )
            finite_values = [
                loss_ce,
                loss_feature,
                loss_r2,
                loss_logit,
                total_loss,
                *feature_layer_losses.values(),
                *relation_layer_losses.values(),
            ]
            if not all(
                bool(torch.isfinite(value).all().item()) for value in finite_values
            ):
                raise RuntimeError("R5 produced a non-finite CE/feature/R2/KL loss")

            log_gradients = sync_gradients and (
                next_optimizer_step == 1
                or next_optimizer_step % args.gradient_log_steps == 0
            )
            local_gradient_record: Optional[Dict[str, object]] = None
            if log_gradients:
                tap = student_features["os16"]
                gradients = {
                    "ce": torch.autograd.grad(
                        loss_ce, tap, retain_graph=True, allow_unused=False
                    )[0].detach().float(),
                    "feature": torch.autograd.grad(
                        loss_feature, tap, retain_graph=True, allow_unused=False
                    )[0].detach().float(),
                    "r2": torch.autograd.grad(
                        loss_r2, tap, retain_graph=True, allow_unused=False
                    )[0].detach().float(),
                    "logit": torch.autograd.grad(
                        loss_logit, tap, retain_graph=True, allow_unused=False
                    )[0].detach().float(),
                }
                if not all(
                    bool(torch.isfinite(gradient).all().item())
                    for gradient in gradients.values()
                ):
                    raise RuntimeError(
                        "R5 gradient audit found a non-finite CE/feature/R2/logit gradient"
                    )
                effective_feature = warmup_weight * args.lambda_feat * gradients["feature"]
                effective_r2 = warmup_weight * args.lambda_r2 * gradients["r2"]
                effective_logit = warmup_weight * args.lambda_logit * gradients["logit"]
                effective_auxiliary = effective_feature + effective_r2 + effective_logit
                effective_total = gradients["ce"] + effective_auxiliary
                ce_norm = _gradient_l2(gradients["ce"])
                auxiliary_norm = _gradient_l2(effective_auxiliary)
                local_gradient_record = {
                    "experiment": EXPERIMENT,
                    "optimizer_step": next_optimizer_step,
                    "fixed_audit_step": next_optimizer_step in FIXED_GRADIENT_AUDIT_STEPS,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "warmup_weight": warmup_weight,
                    "lambda_feat": args.lambda_feat,
                    "lambda_r2": args.lambda_r2,
                    "lambda_logit": args.lambda_logit,
                    "temperature": args.temperature,
                    "tap_shape": list(tap.shape),
                    "gradient_component_scope": "student native os16 tap",
                    "gradient_definition": "raw component gradients before lambda/warm-up; effective fields after both",
                    "loss_ce": float(loss_ce.detach().item()),
                    "loss_feature": float(loss_feature.detach().item()),
                    "loss_r2": float(loss_r2.detach().item()),
                    "loss_logit": float(loss_logit.detach().item()),
                    "grad_l2_ce_os16": ce_norm,
                    "grad_l2_feature_raw_os16": _gradient_l2(gradients["feature"]),
                    "grad_l2_r2_raw_os16": _gradient_l2(gradients["r2"]),
                    "grad_l2_logit_raw_os16": _gradient_l2(gradients["logit"]),
                    "grad_l2_feature_effective_os16": _gradient_l2(effective_feature),
                    "grad_l2_r2_effective_os16": _gradient_l2(effective_r2),
                    "grad_l2_logit_effective_os16": _gradient_l2(effective_logit),
                    "grad_l2_total_auxiliary_effective_os16": auxiliary_norm,
                    "grad_l2_total_effective_os16": _gradient_l2(effective_total),
                    "total_auxiliary_to_ce_effective_ratio_os16": auxiliary_norm
                    / max(ce_norm, 1e-12),
                    "cos_feature_r2_os16": _gradient_cosine(
                        effective_feature, effective_r2
                    ),
                    "cos_feature_logit_os16": _gradient_cosine(
                        effective_feature, effective_logit
                    ),
                    "cos_r2_logit_os16": _gradient_cosine(
                        effective_r2, effective_logit
                    ),
                    "cos_ce_feature_os16": _gradient_cosine(
                        gradients["ce"], effective_feature
                    ),
                    "cos_ce_r2_os16": _gradient_cosine(
                        gradients["ce"], effective_r2
                    ),
                    "cos_ce_logit_os16": _gradient_cosine(
                        gradients["ce"], effective_logit
                    ),
                    "relation_valid_token_count": relation_audit["valid_token_count"],
                    "relation_valid_pair_count": relation_audit["valid_pair_count"],
                    "relation_spec_sha256": (
                        None
                        if _RELATION_SPEC is None
                        else r2._canonical_sha256(_RELATION_SPEC)
                    ),
                }
            scaler.scale(total_loss / group_size).backward()

        if sync_gradients:
            scaler.unscale_(optimizer)
            optimizer_steps += 1
            if local_gradient_record is not None:
                local_gradient_record["grad_l2_total_student"] = k0._gradient_l2_named(
                    model
                )
                aggregated = _aggregate_gradient_record(
                    local_gradient_record, world_size
                )
                _update_auxiliary_stop_gate(aggregated)
                if rank == 0:
                    gradient_records.append(aggregated)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if first_batch_audit is None and starting_optimizer_step == 0 and batch_index == 0:
            first_batch_audit = {
                "rank": rank,
                "epoch": epoch,
                "micro_batch_index": 0,
                "paths": list(paths),
                "image_tensor_shape": list(images.shape),
                "target_tensor_shape": list(targets.shape),
                "image_tensor_sha256": k0._tensor_sha256(images),
                "target_tensor_sha256": k0._tensor_sha256(targets),
                "valid_pixels": batch_valid,
                "student_logit_shape": list(logits_float.shape),
                "teacher_logit_shape": list(teacher_logits.shape),
                "student_feature_shapes": {
                    layer: list(student_features[layer].shape)
                    for layer in a0.A0_LAYER_ORDER
                },
                "teacher_feature_shapes": {
                    layer: list(teacher_features[layer].shape)
                    for layer in a0.A0_LAYER_ORDER
                },
                "projected_teacher_shapes": projected_shapes,
                "feature_loss_by_layer": {
                    layer: float(feature_layer_losses[layer].detach().item())
                    for layer in a0.A0_LAYER_ORDER
                },
                "relation_r2_loss_by_layer": {
                    layer: float(relation_layer_losses[layer].detach().item())
                    for layer in a0.A0_LAYER_ORDER
                },
                "ce_loss": float(loss_ce.detach().item()),
                "feature_loss": float(loss_feature.detach().item()),
                "relation_r2_loss": float(loss_r2.detach().item()),
                "logit_loss": float(loss_logit.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                "relation": relation_audit,
                "temperature": args.temperature,
                "lambda_feat": args.lambda_feat,
                "lambda_r2": args.lambda_r2,
                "lambda_logit": args.lambda_logit,
                "warmup_weight": warmup_weight,
                "teacher_backbone_forward_count": 1,
                **_resource_hashes(),
            }
            equivalence = _compare_first_batch(first_batch_audit, rank)
            first_batch_audit["r5_controlled_first_batch_equivalence"] = equivalence
            if not bool(equivalence.get("passed")):
                raise RuntimeError(
                    "R5 first-batch base fields differ from K3/R2 references: "
                    + json.dumps(equivalence, sort_keys=True)
                )

        predictions = logits_float.detach().argmax(dim=1)
        confusion += common.confusion_counts(predictions, targets)
        ce_loss_sum += float(batch_ce_sum.detach().item())
        valid_pixels += batch_valid
        feature_value = float(loss_feature.detach().item())
        relation_value = float(loss_r2.detach().item())
        logit_value = float(loss_logit.detach().item())
        feature_sum += feature_value
        relation_sum += relation_value
        logit_sum += logit_value
        total_sum += float(total_loss.detach().item())
        valid_token_sum += float(relation_audit["valid_token_count"])
        valid_pair_sum += float(relation_audit["valid_pair_count"])
        physical_batch_sum += float(relation_audit["physical_batch_size"])
        for layer in a0.A0_LAYER_ORDER:
            feature_layer_sums[layer] += float(
                feature_layer_losses[layer].detach().item()
            )
            relation_layer_sums[layer] += float(
                relation_layer_losses[layer].detach().item()
            )
        batch_count += 1
        last_warmup_weight = warmup_weight
        if rank == 0:
            running = common.metrics_from_confusion(
                confusion, ce_loss_sum, valid_pixels
            )
            progress.set_postfix(
                {
                    "CE": f"{running['loss']:.4f}",
                    "feat": f"{feature_value:.4f}",
                    "R2": f"{relation_value:.4f}",
                    "KL": f"{logit_value:.4f}",
                    "mIoU": f"{running['mIoU']:.4f}",
                    "warm": f"{warmup_weight:.3f}",
                    "steps": optimizer_steps,
                }
            )

    if optimizer_steps != target_steps:
        raise RuntimeError(
            f"R5 optimizer-step accounting failed: actual={optimizer_steps}, "
            f"expected={target_steps}"
        )
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("R5 training found a gradient on the frozen teacher")
    if list(projection.parameters()):
        raise RuntimeError("R5 projection became trainable during training")
    if batch_count == 0:
        raise RuntimeError("R5 epoch processed no micro-batches")

    metrics = server_base._reduce_train_metrics(
        confusion, ce_loss_sum, valid_pixels, device, world_size
    )
    feature_means, relation_means, tail, global_batches = _reduce_training_statistics(
        feature_layer_sums,
        relation_layer_sums,
        feature_sum,
        relation_sum,
        logit_sum,
        total_sum,
        valid_token_sum,
        valid_pair_sum,
        physical_batch_sum,
        batch_count,
        device,
        world_size,
    )
    (
        feature_mean,
        relation_mean,
        logit_mean,
        total_mean,
        valid_token_mean,
        valid_pair_mean,
        physical_batch_mean,
    ) = tail
    metrics.update(
        {
            "loss_schema": (
                "hard_label_CE_plus_A0_feature_MSE_plus_R2_relation_MSE_plus_"
                "full_resolution_masked_pixel_KL"
            ),
            "ce_loss": metrics["loss"],
            "feature_loss": feature_mean,
            "feature_loss_by_layer": feature_means,
            "relation_enabled": True,
            "relation_r1_loss": None,
            "relation_r2_loss": relation_mean,
            "relation_r2_loss_by_layer": relation_means,
            "logit_loss": logit_mean,
            "relation_valid_token_count": valid_token_mean,
            "relation_valid_pair_count": valid_pair_mean,
            "relation_physical_batch_size_mean": physical_batch_mean,
            "total_loss_micro_batch_mean": total_mean,
            "warmup_weight": last_warmup_weight,
            "micro_batches_global": global_batches,
        }
    )
    return metrics, optimizer_steps, gradient_records, first_batch_audit


def smoke_test_r5(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    rank: int,
) -> None:
    teacher, projection = k1._require_resources()
    args = k3._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("R5 active arguments were not set")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    model.train()
    teacher.eval()
    images, targets, paths = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    model.zero_grad(set_to_none=True)
    with common.autocast_context(device, amp_enabled):
        output = model(images)
        if not isinstance(output, Mapping):
            raise RuntimeError("R5 smoke forward did not expose features")
        with torch.no_grad():
            teacher_features, teacher_logits = k3._teacher_features_and_logits(
                teacher, images
            )
        feature_layers, _ = k3._feature_kd_losses(
            output["features"], teacher_features, projection
        )
    logits = output["logits"].float()
    valid = int((targets != common.IGNORE_INDEX).sum().item())
    if valid == 0:
        raise RuntimeError("R5 smoke batch contains no valid pixels")
    ce = F.cross_entropy(
        logits, targets, ignore_index=common.IGNORE_INDEX, reduction="sum"
    ) / valid
    feature = sum(feature_layers.values()) / len(a0.A0_LAYER_ORDER)
    relation, _, relation_audit = r2.r2_relation_losses(
        output["features"], teacher_features, targets, world_size
    )
    logit = k2._masked_pixel_kl(teacher_logits, logits, targets, args.temperature)
    warmup = 1.0 / k3._warmup_steps(args)
    total = ce + warmup * (
        args.lambda_feat * feature
        + args.lambda_r2 * relation
        + args.lambda_logit * logit
    )
    tap = output["features"]["os16"]
    gradients = [
        torch.autograd.grad(loss, tap, retain_graph=True)[0].detach().float()
        for loss in (ce, feature, relation, logit)
    ]
    effective_auxiliary = warmup * (
        args.lambda_feat * gradients[1]
        + args.lambda_r2 * gradients[2]
        + args.lambda_logit * gradients[3]
    )
    auxiliary_ratio = _gradient_l2(effective_auxiliary) / max(
        _gradient_l2(gradients[0]), 1e-12
    )
    total.backward()
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (ce, feature, relation, logit, total)
    ):
        raise RuntimeError("R5 smoke test produced a non-finite loss")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("R5 smoke test found a teacher gradient")
    if list(projection.parameters()):
        raise RuntimeError("R5 smoke test found trainable projection parameters")
    backbone_gradients = sum(
        parameter.grad is not None
        for parameter in k0.unwrap_model(model).backbone.parameters()
    )
    head_gradients = sum(
        parameter.grad is not None
        for parameter in k0.unwrap_model(model).head.parameters()
    )
    if backbone_gradients == 0 or head_gradients == 0:
        raise RuntimeError("R5 smoke test did not produce end-to-end student gradients")
    if rank == 0:
        print(
            f"[OK] R5 server DDP smoke: sample={paths[0]}, logits={tuple(logits.shape)}, "
            f"CE={ce.item():.6f}, feature={feature.item():.6f}, "
            f"R2={relation.item():.6f}, KL={logit.item():.6f}, "
            f"total={total.item():.6f}, warmup={warmup:.6f}, "
            f"aux/CE_grad_os16={auxiliary_ratio:.6f}, "
            f"valid_tokens={relation_audit['valid_token_count']}, "
            f"valid_pairs={relation_audit['valid_pair_count']}, "
            f"backbone_grad_tensors={backbone_gradients}, "
            f"head_grad_tensors={head_gradients}"
        )


def _gradient_gate_summary(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {
            "passed": False,
            "records": 0,
            "reason": f"missing gradient audit file: {path}",
        }
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ratios = [
        float(record["total_auxiliary_to_ce_effective_ratio_os16"])
        for record in records
        if record.get("total_auxiliary_to_ce_effective_ratio_os16") is not None
    ]
    consecutive = 0
    maximum_consecutive = 0
    for ratio in ratios:
        consecutive = consecutive + 1 if ratio > AUXILIARY_CE_STOP_RATIO else 0
        maximum_consecutive = max(maximum_consecutive, consecutive)
    observed = [
        int(record["optimizer_step"])
        for record in records
        if record.get("fixed_audit_step")
    ]
    fixed_audit_complete = all(
        step in observed for step in FIXED_GRADIENT_AUDIT_STEPS
    )
    finite_ratios = bool(ratios) and all(math.isfinite(ratio) for ratio in ratios)
    return {
        "passed": bool(records)
        and finite_ratios
        and fixed_audit_complete
        and maximum_consecutive < AUXILIARY_CE_STOP_CONSECUTIVE,
        "records": len(records),
        "fixed_audit_steps_expected": list(FIXED_GRADIENT_AUDIT_STEPS),
        "fixed_audit_steps_observed": observed,
        "fixed_audit_complete": fixed_audit_complete,
        "stop_ratio": AUXILIARY_CE_STOP_RATIO,
        "stop_consecutive_records": AUXILIARY_CE_STOP_CONSECUTIVE,
        "total_auxiliary_to_ce_effective_ratio_min": min(ratios) if ratios else None,
        "total_auxiliary_to_ce_effective_ratio_max": max(ratios) if ratios else None,
        "maximum_consecutive_above_stop_ratio": maximum_consecutive,
        "three_consecutive_auxiliary_gt_2x_ce": maximum_consecutive >= 3,
    }


def _postprocess_metrics_r5(args: argparse.Namespace) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    _ORIGINAL_K3_POSTPROCESS(args)
    metrics_path = r5_paths(args.output_dir, args.seed)["metrics"]
    if not metrics_path.is_file():
        return
    results = _read_json(metrics_path)
    gradient_path = r5_paths(args.output_dir, args.seed)["gradient_norms"]
    gradient_gate = _gradient_gate_summary(gradient_path)
    relation = _RELATION_SPEC
    if relation is None:
        config = results.get("config")
        if isinstance(config, Mapping) and isinstance(config.get("relation"), Mapping):
            relation = dict(config["relation"])
    results.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "protocol": (
                "R5 gated feature+relation+logits run: K-group shared scratch "
                "MobileNetV2+R-ASPP initialization, hard-label CE, locked A0 "
                "feature MSE, native masked 8x16 R2 relation MSE, and frozen T1 "
                "full-resolution masked pixel KL (T=4); fixed weights 1.0/0.3/0.5, "
                "shared 4000-step warm-up, one teacher backbone forward per batch, "
                "80k optimizer steps, dev_local selection, and no test_local evaluation."
            ),
            "loss": {
                "hard_label_ce": True,
                "feature_kd": True,
                "feature_mechanism": "A0 fixed StandardScaler+PCA",
                "feature_layers": list(a0.A0_LAYER_ORDER),
                "lambda_feat": args.lambda_feat,
                "relation_kd": True,
                "relation_r1": False,
                "relation_r2": True,
                "lambda_r2": args.lambda_r2,
                "relation_pool_size": list(r2.POOL_SIZE),
                "logit_kd": True,
                "logit_mechanism": "full-resolution masked pixel KL",
                "lambda_logit": args.lambda_logit,
                "temperature": args.temperature,
                "temperature_squared_factor": True,
                "warmup_steps": k3._warmup_steps(args),
                "warmup_ratio": args.feature_warmup_ratio,
                "shared_auxiliary_warmup": True,
                "total": "CE + warmup * (1.0*feature + 0.3*R2 + 0.5*logit_KL_T2)",
            },
            "relation": copy.deepcopy(relation),
            "relation_spec_sha256": (
                None if relation is None else r2._canonical_sha256(relation)
            ),
            "relation_reference_tests": copy.deepcopy(_REFERENCE_TESTS),
            "launch_gate": copy.deepcopy(_REFERENCE_GATE),
            "gradient_gate": gradient_gate,
            "test_local_evaluated": False,
            "hashes": {**dict(results.get("hashes", {})), **_resource_hashes()},
        }
    )

    if args.seed == SCREENING_SEED and isinstance(_REFERENCE_GATE, Mapping):
        best_reference = _number(
            _REFERENCE_GATE.get("best_reference_mIoU"), "R5 best reference mIoU"
        )
        gain = _best_miou(results, "R5 seed 42") - best_reference
        template_path = r5_paths(args.output_dir, args.seed)["run_dir"] / "expansion_gate.json"
        screening_gate = {
            "mIoU_gain_threshold": MIOU_EXPANSION_MARGIN,
            "miou_gain_over_best_reference": gain,
            "miou_gate_passed": gain > MIOU_EXPANSION_MARGIN,
            "paired_bootstrap_ci_lower_required": 0.0,
            "paired_bootstrap_ci_lower": None,
            "paired_bootstrap_gate_passed": None,
            "gradient_gate_passed": bool(gradient_gate.get("passed")),
            "eligible_for_confirmation_seeds": False,
            "reason": "paired-bootstrap result must be entered in expansion_gate.json",
            "expansion_gate_path": str(template_path.resolve()),
        }
        results["screening_expansion_gate"] = screening_gate
        common.write_json_atomic(metrics_path, results)
        metrics_hash = common.sha256_file(metrics_path)
        template = {
            "experiment": EXPERIMENT,
            "screening_seed": SCREENING_SEED,
            "passed": False,
            "r5_metrics": str(metrics_path.resolve()),
            "r5_metrics_sha256": metrics_hash,
            "best_reference": _REFERENCE_GATE.get("best_reference"),
            "best_reference_mIoU": best_reference,
            "miou_gain_threshold": MIOU_EXPANSION_MARGIN,
            "miou_gain_over_best_reference": gain,
            "paired_bootstrap_repetitions": 100_000,
            "paired_bootstrap_ci_lower": None,
            "gradient_gate_passed": bool(gradient_gate.get("passed")),
            "instructions": (
                "After the paired bootstrap, set paired_bootstrap_ci_lower and "
                "set passed=true only if the mIoU gain exceeds the threshold, "
                "the CI lower bound is >0, and gradient_gate_passed is true."
            ),
        }
        common.write_json_atomic(template_path, template)
        return

    common.write_json_atomic(metrics_path, results)


def run_training(args: argparse.Namespace) -> None:
    global _REFERENCE_GATE, _RELATION_SPEC, _REFERENCE_TESTS
    _REFERENCE_GATE = _validate_launch_gate(args)
    _RELATION_SPEC = None
    _REFERENCE_TESTS = None
    _restore_auxiliary_gate_state(args)

    # R2/R1 relation losses require a crop to retain at least one valid
    # location after nearest-neighbor resizing at OS=4/8/16.  The ordinary K3
    # loader only rejects an all-ignore crop at image resolution, so using it
    # here can eventually produce an image with zero valid relation tokens.
    # Reuse the registered relation-valid resampling loader used by R2.
    saved_server_build_train_loader = server_base.build_train_loader

    saved = {
        "__file__": k3.__file__,
        "EXPERIMENT": k3.EXPERIMENT,
        "ARTIFACT_TYPE": k3.ARTIFACT_TYPE,
        "ARTIFACT_FORMAT_VERSION": k3.ARTIFACT_FORMAT_VERSION,
        "k3_paths": k3.k3_paths,
        "ensure_k3_resources": k3.ensure_k3_resources,
        "build_config": k3.build_config,
        "build_best_checkpoint": k3.build_best_checkpoint,
        "train_one_epoch_k3": k3.train_one_epoch_k3,
        "smoke_test_k3": k3.smoke_test_k3,
        "audit_k3_shapes": k3.audit_k3_shapes,
        "_postprocess_metrics": k3._postprocess_metrics,
        "_patched_torch_save_atomic": k3._patched_torch_save_atomic,
        "_patched_evaluate": k3._patched_evaluate,
        "_k3_print": k3._k3_print,
    }
    try:
        server_base.build_train_loader = r1.build_train_loader_r1
        k3.__file__ = str(Path(__file__).resolve())
        k3.EXPERIMENT = EXPERIMENT
        k3.ARTIFACT_TYPE = ARTIFACT_TYPE
        k3.ARTIFACT_FORMAT_VERSION = ARTIFACT_FORMAT_VERSION
        k3.k3_paths = r5_paths
        k3.ensure_k3_resources = ensure_r5_resources
        k3.build_config = build_config_r5
        k3.build_best_checkpoint = build_best_checkpoint_r5
        k3.train_one_epoch_k3 = train_one_epoch_r5
        k3.smoke_test_k3 = smoke_test_r5
        k3.audit_k3_shapes = audit_shapes_r5
        k3._postprocess_metrics = _postprocess_metrics_r5
        k3._patched_torch_save_atomic = _patched_torch_save_atomic_r5
        k3._patched_evaluate = _patched_evaluate_r5
        k3._k3_print = _r5_print
        k3.run_training(args)
    finally:
        server_base.build_train_loader = saved_server_build_train_loader
        for name, value in saved.items():
            setattr(k3, name, value)
        _RELATION_SPEC = None
        _REFERENCE_TESTS = None


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
