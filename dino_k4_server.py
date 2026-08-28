"""K4 server entry point: A0 probe initialization plus pixel-logit KD.

K4 is the staged-initialization response-distillation experiment registered in
``plan_markdown/Cityscapes知识蒸馏实验详单.md`` and refined by the completed K
group summary.  Each run starts from the same-seed locked A0 best probe
checkpoint (both MobileNetV2 backbone and R-ASPP head), unfreezes the complete
student, and follows the A0-FT/S2-0 80k-step fine-tune protocol.  Its only
additional training term is the K2 response loss::

    L = L_seg + warmup(step) * 0.5 * L_logit

``L_logit`` is the full-resolution masked pixel KL with ``T=4``, class sum,
valid-pixel mean, and ``T**2`` scaling.  K4 does not use online feature KD,
PCA projection, or a student adapter; A0/PCA appears only in the initialization
provenance.

Typical two-GPU server command::

    torchrun --standalone --nproc_per_node=2 dino_k4_server.py \
        --seed 42 --batch-size 2 --global-batch-size 8 \
        --num-workers 8 --multiprocessing-context spawn \
        --no-pin-memory --persistent-workers

Windows/local functional smoke (does not replace Linux two-GPU DDP smoke)::

    python -B dino_k4_server.py --device cuda --smoke-test \
        --batch-size 1 --global-batch-size 1 --num-workers 0 \
        --no-persistent-workers --no-pin-memory --no-amp
"""

from __future__ import annotations

import argparse
import builtins
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dino as common
import dino_k0_server as k0
import dino_k2_server as k2
import dino_s2_0 as base
import dino_s2_0_server as server_base
from dino_t1 import load_teacher_for_distillation


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "K_MobileNetV2_RASPP_server"
DEFAULT_A0_OUTPUT_DIR = (
    SCRIPT_DIR
    / "result"
    / "A_MobileNetV2_RASPP_server"
)


def default_probe_checkpoint(seed: int) -> Path:
    return (
        DEFAULT_A0_OUTPUT_DIR
        / "A0"
        / f"seed_{seed}"
        / "a0_probe_mobilenetv2_raspp_best.pth"
    )


DEFAULT_PROBE_CHECKPOINT = default_probe_checkpoint(42)
DEFAULT_TEACHER_CHECKPOINT = k2.DEFAULT_TEACHER_CHECKPOINT

EXPECTED_COMBINED_MANIFEST_SHA256 = k0.EXPECTED_COMBINED_MANIFEST_SHA256
EXPECTED_PROBE_CHECKPOINT_SHA256_BY_SEED = {
    42: "9abb4fcb422cfb7f31811caa433aa62de66806ae055d107b9adc3b08aad0e95c",
    3407: "a53b370edb5740230713e7689fd258196d6deb88bcf10c2d1280c7172ca0a321",
    260805: "0620ec82a8d48264c5e2f4b70fe3a3d2cedca4c23e64c900f23e39e7c90056ff",
}
EXPECTED_TEACHER_CHECKPOINT_SHA256 = k2.EXPECTED_TEACHER_CHECKPOINT_SHA256

EXPERIMENT = "K4"
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_k4_a0_init_logit_kd"
ARTIFACT_FORMAT_VERSION = 1
FORMAL_SEEDS = (42, 3407, 260805)
SCREENING_SEED = 42
MAX_STEPS = 80_000
TEMPERATURE = 4.0
LAMBDA_LOGIT = 0.5
LOGIT_WARMUP_STEPS = 4_000
LOGIT_WARMUP_RATIO = 0.05
GRADIENT_LOG_STEPS = 500
INTERACTION_THRESHOLD = 0.00425
SOURCE_EXPERIMENT = "A0"
SOURCE_ARTIFACT_TYPE = "a0_probe_mobilenetv2_raspp_fixed_pca"


_ACTIVE_ARGS: Optional[argparse.Namespace] = None
_PROBE_CHECKPOINT_SHA256: Optional[str] = None
_PROBE_MODEL_STATE_SHA256: Optional[str] = None
_PROBE_BACKBONE_STATE_SHA256: Optional[str] = None
_PROBE_ARTIFACT_TYPE: Optional[str] = None

_ORIGINAL_K2_WARMUP_STEPS = k2._warmup_steps
_ORIGINAL_K2_RESOURCE_HASHES = k2._resource_hashes
_ORIGINAL_K2_TQDM = k2.tqdm
_ORIGINAL_SET_GLOBAL_SEED = common.set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "K4 MobileNetV2+R-ASPP: end-to-end 80k-step fine-tuning from the "
            "same-seed locked A0 best probe checkpoint, with hard-label CE and "
            "frozen T1 full-resolution pixel-logit KD."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=common.DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--probe-checkpoint",
        type=Path,
        default=None,
        help=(
            "Same-seed A0 best probe checkpoint. Defaults to "
            "result/A_MobileNetV2_RASPP_server/A0/seed_<seed>/"
            "a0_probe_mobilenetv2_raspp_best.pth."
        ),
    )
    parser.add_argument(
        "--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT
    )
    parser.add_argument("--teacher-repo-dir", type=Path, default=common.DEFAULT_REPO_DIR)
    parser.add_argument(
        "--teacher-weights-path", type=Path, default=common.DEFAULT_WEIGHTS_PATH
    )
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--lambda-logit", type=float, default=LAMBDA_LOGIT)
    parser.add_argument(
        "--logit-warmup-ratio", type=float, default=LOGIT_WARMUP_RATIO
    )
    parser.add_argument("--gradient-log-steps", type=int, default=GRADIENT_LOG_STEPS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-GPU batch size.")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=8,
        help="Global batch size; must be divisible by batch-size * world-size.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--multiprocessing-context",
        choices=("auto", "fork", "spawn", "forkserver"),
        default="spawn",
    )
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--eval-every-steps", type=int, default=5_000)
    parser.add_argument("--head-channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--crop-height", type=int, default=512)
    parser.add_argument("--crop-width", type=int, default=1024)
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--benchmark-height", type=int, default=1024)
    parser.add_argument("--benchmark-width", type=int, default=2048)
    parser.add_argument("--benchmark-warmup", type=int, default=10)
    parser.add_argument("--benchmark-runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SCREENING_SEED)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, or cuda; torchrun assigns one CUDA device per rank.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--benchmark", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    positive = (
        "max_steps",
        "batch_size",
        "global_batch_size",
        "eval_batch_size",
        "accumulation_steps",
        "eval_every_steps",
        "head_channels",
        "crop_height",
        "crop_width",
        "benchmark_height",
        "benchmark_width",
        "benchmark_warmup",
        "benchmark_runs",
        "prefetch_factor",
        "gradient_log_steps",
    )
    for field in positive:
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.lr <= 0 or not 0 <= args.momentum < 1 or args.weight_decay < 0:
        parser.error("Invalid optimizer settings")
    if args.poly_power <= 0 or not 0 < args.min_lr_ratio <= 1:
        parser.error("Invalid polynomial scheduler settings")
    if not math.isclose(args.temperature, TEMPERATURE):
        parser.error("Formal K4 is locked to --temperature 4")
    if not math.isclose(args.lambda_logit, LAMBDA_LOGIT):
        parser.error("Formal K4 is locked to --lambda-logit 0.5")
    if not math.isclose(args.logit_warmup_ratio, LOGIT_WARMUP_RATIO):
        parser.error("Formal K4 is locked to --logit-warmup-ratio 0.05")
    if not args.smoke_test and args.max_steps != MAX_STEPS:
        parser.error("Formal K4 is locked to --max-steps 80000")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if not 0 < args.scale_min <= args.scale_max:
        parser.error("Require 0 < --scale-min <= --scale-max")
    if args.boundary_tolerance < 0:
        parser.error("--boundary-tolerance cannot be negative")
    if args.crop_height % common.OUTPUT_STRIDE or args.crop_width % common.OUTPUT_STRIDE:
        parser.error(f"Crop dimensions must be divisible by {common.OUTPUT_STRIDE}")
    if args.benchmark_height % common.OUTPUT_STRIDE or args.benchmark_width % common.OUTPUT_STRIDE:
        parser.error(f"Benchmark dimensions must be divisible by {common.OUTPUT_STRIDE}")
    if args.resume and args.smoke_test:
        parser.error("--resume cannot be combined with --smoke-test")
    if args.seed not in FORMAL_SEEDS:
        parser.error(f"--seed must be one of the registered K4 seeds: {FORMAL_SEEDS}")
    if args.probe_checkpoint is None:
        args.probe_checkpoint = default_probe_checkpoint(args.seed)
    return args


def _warmup_steps(_args: argparse.Namespace) -> int:
    return LOGIT_WARMUP_STEPS


def _a0_ft_compatible_set_global_seed(seed: int, deterministic: bool) -> None:
    """Match A0-FT's rank-aware RNG stream before model/DataLoader creation."""

    rank = int(os.environ.get("RANK", "0"))
    _ORIGINAL_SET_GLOBAL_SEED(seed + rank, deterministic)


def _k4_tqdm(*args: Any, **kwargs: Any):
    description = kwargs.get("desc")
    if isinstance(description, str):
        kwargs["desc"] = description.replace("K2", EXPERIMENT)
    return _ORIGINAL_K2_TQDM(*args, **kwargs)


def k4_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    original = k2._ORIGINAL_K0_PATHS(output_dir, seed)
    run_dir = output_dir.resolve() / EXPERIMENT / f"seed_{seed}"
    return {
        key: run_dir if key == "run_dir" else run_dir / value.name
        for key, value in original.items()
    }


def _probe_metadata() -> Dict[str, object]:
    args = _ACTIVE_ARGS
    return {
        "source_experiment": SOURCE_EXPERIMENT,
        "source_probe_seed": args.seed if args is not None else None,
        "source_probe_checkpoint": (
            str(args.probe_checkpoint.resolve()) if args is not None else None
        ),
        "source_probe_checkpoint_sha256": _PROBE_CHECKPOINT_SHA256,
        "source_probe_model_state_sha256": _PROBE_MODEL_STATE_SHA256,
        "source_probe_backbone_state_sha256": _PROBE_BACKBONE_STATE_SHA256,
        "source_probe_artifact_type": _PROBE_ARTIFACT_TYPE,
    }


def _resource_hashes() -> Dict[str, object]:
    return {
        **_ORIGINAL_K2_RESOURCE_HASHES(),
        **_probe_metadata(),
        "online_pca_parameter_sha256": None,
    }


def _load_locked_a0_probe(
    model: base.MobileNetV2RASPPStudent,
    args: argparse.Namespace,
    rank: int,
) -> Tuple[str, str, Path]:
    global _PROBE_CHECKPOINT_SHA256
    global _PROBE_MODEL_STATE_SHA256
    global _PROBE_BACKBONE_STATE_SHA256
    global _PROBE_ARTIFACT_TYPE

    checkpoint = args.probe_checkpoint.resolve()
    checkpoint_hash = common.verify_checkpoint_sidecar(checkpoint)
    expected_checkpoint_hash = EXPECTED_PROBE_CHECKPOINT_SHA256_BY_SEED[args.seed]
    if checkpoint_hash != expected_checkpoint_hash:
        raise RuntimeError(
            f"K4 seed={args.seed} must start from the same-seed locked A0 best "
            f"probe checkpoint: actual={checkpoint_hash}, "
            f"expected={expected_checkpoint_hash}"
        )
    payload = common.safe_torch_load(
        checkpoint, map_location="cpu", weights_only=True
    )
    if payload.get("artifact_type") != SOURCE_ARTIFACT_TYPE:
        raise RuntimeError(
            f"K4 source is not the locked A0 probe artifact: {payload.get('artifact_type')!r}"
        )
    source_config = payload.get("config")
    if not isinstance(source_config, Mapping):
        raise RuntimeError("K4 A0 source checkpoint is missing its configuration")
    expected_source = {
        "experiment": SOURCE_EXPERIMENT,
        "seed": args.seed,
        "head_channels": args.head_channels,
        "dropout": args.dropout,
        "num_classes": common.NUM_CLASSES,
        "output_stride": common.OUTPUT_STRIDE,
        "test_local_evaluated": False,
    }
    for key, expected in expected_source.items():
        if source_config.get(key) != expected:
            raise RuntimeError(
                f"K4 A0 source config mismatch for {key}: "
                f"actual={source_config.get(key)!r}, expected={expected!r}"
            )
    if payload.get("feature_taps") != base.FEATURE_TAPS:
        raise RuntimeError("K4 A0 source feature taps differ from the locked student")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("K4 A0 source checkpoint is missing model_state_dict")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(True)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("K4 must unfreeze the complete A0-initialized student")
    model_state_hash = common.state_dict_sha256(model.state_dict())
    if model_state_hash != payload.get("model_state_sha256"):
        raise RuntimeError("K4 A0 source model-state SHA-256 verification failed")
    backbone_state = {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith("backbone.")
    }
    backbone_hash = common.state_dict_sha256(backbone_state)
    expected_backbone_hash = payload.get("backbone_state_sha256")
    if expected_backbone_hash and backbone_hash != expected_backbone_hash:
        raise RuntimeError("K4 A0 source backbone SHA-256 verification failed")

    _PROBE_CHECKPOINT_SHA256 = checkpoint_hash
    _PROBE_MODEL_STATE_SHA256 = model_state_hash
    _PROBE_BACKBONE_STATE_SHA256 = backbone_hash
    _PROBE_ARTIFACT_TYPE = str(payload["artifact_type"])
    if rank == 0:
        builtins.print(
            f"[OK] K4 loaded locked A0 probe: file_sha256={checkpoint_hash}, "
            f"model_state_sha256={model_state_hash}"
        )
    return model_state_hash, checkpoint_hash, checkpoint


def ensure_k4_resources(
    model: base.MobileNetV2RASPPStudent,
    args: argparse.Namespace,
    output_dir: Path,
    seed: int,
    rank: int,
    world_size: int,
) -> Tuple[str, str, Path]:
    del output_dir, seed, world_size
    init_result = _load_locked_a0_probe(model, args, rank)

    if k2._TEACHER is not None:
        return init_result
    device = next(model.parameters()).device
    rng_state = k0._capture_rank_rng_state(device)
    try:
        teacher_checkpoint = args.teacher_checkpoint.resolve()
        teacher_hash = common.verify_checkpoint_sidecar(teacher_checkpoint)
        if teacher_hash != EXPECTED_TEACHER_CHECKPOINT_SHA256:
            raise RuntimeError(
                "K4 must use the locked T1 seed=3407 teacher checkpoint: "
                f"actual={teacher_hash}, expected={EXPECTED_TEACHER_CHECKPOINT_SHA256}"
            )
        teacher, _teacher_payload = load_teacher_for_distillation(
            teacher_checkpoint,
            repo_dir=args.teacher_repo_dir,
            weights_path=args.teacher_weights_path,
            device=device,
            verify_checkpoint_file=True,
        )
        teacher.freeze_for_distillation().eval()
        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise RuntimeError("The K4 teacher is not fully frozen")
    finally:
        k0._restore_rank_rng_state(rng_state, device)
    k2._TEACHER = teacher
    k2._TEACHER_CHECKPOINT_SHA256 = teacher_hash
    return init_result


def audit_k4_shapes(
    model: base.MobileNetV2RASPPStudent,
    device: torch.device,
    height: int,
    width: int,
    amp_enabled: bool,
) -> Dict[str, object]:
    audit = k2.audit_k2_shapes(model, device, height, width, amp_enabled)
    audit["experiment"] = EXPERIMENT
    audit["student_initialization"] = "same-seed locked A0 best probe backbone+head"
    audit.update(_probe_metadata())
    return audit


def build_config(
    args: argparse.Namespace,
    accumulation_steps: int,
    world_size: int,
    device: torch.device,
    source_state_sha256: str,
    source_checkpoint_sha256: str,
) -> Dict[str, object]:
    config = k2.build_config(
        args,
        accumulation_steps,
        world_size,
        device,
        source_state_sha256,
        source_checkpoint_sha256,
    )
    config["experiment"] = EXPERIMENT
    config["server_entry_point"] = str(Path(__file__).resolve())
    config["formal_seeds"] = list(FORMAL_SEEDS)
    config["screening_seed"] = SCREENING_SEED
    config["rank_seed_policy"] = "A0-FT compatible: global seed = seed + rank"
    config["initialization"] = "same-seed locked A0 best probe checkpoint (backbone+head)"
    config["source_probe"] = _probe_metadata()
    config["knowledge_distillation"] = True
    config["registered_analysis"] = {
        "requires_all_three_seeds": True,
        "delta_s": "K4_s - A0-FT_s",
        "interaction_s": "(K4_s - A0-FT_s) - (K2_s - K0_s)",
        "measurable_interaction_rule": (
            "all three interaction_s values have the same sign and "
            f"abs(mean(interaction_s)) > {INTERACTION_THRESHOLD}"
        ),
        "reference_k_group_seed_std": INTERACTION_THRESHOLD,
    }
    config["pca"] = {
        "enabled_in_current_training": False,
        "feature_kd": False,
        "initialization_provenance": "A0 fixed StandardScaler+PCA pretraining",
        "online_projection": None,
        "trainable": False,
    }
    config.pop("shared_init_state_sha256", None)
    config.pop("shared_init_file_sha256", None)
    return config


def build_best_checkpoint(
    model: base.MobileNetV2RASPPStudent,
    epoch: int,
    optimizer_step: int,
    dev_metrics: Mapping[str, object],
    config: Mapping[str, object],
    hashes: Mapping[str, object],
    dataset_lock: Mapping[str, object],
    shape_audit: Mapping[str, object],
) -> Dict[str, object]:
    payload = k2.build_best_checkpoint(
        model,
        epoch,
        optimizer_step,
        dev_metrics,
        config,
        {**hashes, **_resource_hashes()},
        dataset_lock,
        shape_audit,
    )
    payload["artifact_type"] = ARTIFACT_TYPE
    payload["experiment"] = EXPERIMENT
    payload["initialization"] = "same-seed locked A0 best probe checkpoint (backbone+head)"
    payload["source_probe"] = _probe_metadata()
    payload["hashes"] = {**dict(payload.get("hashes", {})), **_resource_hashes()}
    return payload


def train_one_epoch_k4(*args: Any, **kwargs: Any):
    metrics, steps, gradients, first_batch = k2.train_one_epoch_k2(*args, **kwargs)
    metrics["experiment"] = EXPERIMENT
    if first_batch is not None:
        first_batch["experiment"] = EXPERIMENT
        first_batch["initialization"] = "same-seed A0 best probe backbone+head"
        first_batch.update(_probe_metadata())
    return metrics, steps, gradients, first_batch


def smoke_test_k4(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    rank: int,
) -> None:
    teacher = k2._require_teacher()
    args = _ACTIVE_ARGS
    if args is None:
        raise RuntimeError("K4 active arguments were not set")
    model.train()
    teacher.eval()
    images, targets, paths = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    model.zero_grad(set_to_none=True)
    with common.autocast_context(device, amp_enabled):
        student_output = model(images)
        if not isinstance(student_output, Mapping):
            raise RuntimeError("K4 smoke forward did not expose the OS=16 tap")
        logits = student_output["logits"]
        with torch.no_grad():
            teacher_logits = teacher(images)
    logits_float = logits.float()
    valid_pixels = int((targets != common.IGNORE_INDEX).sum().item())
    if valid_pixels == 0:
        raise RuntimeError("K4 smoke batch contains no valid pixels")
    loss_seg = F.cross_entropy(
        logits_float,
        targets,
        ignore_index=common.IGNORE_INDEX,
        reduction="sum",
    ) / valid_pixels
    loss_logit = k2._masked_pixel_kl(
        teacher_logits, logits_float, targets, args.temperature
    )
    warmup_weight = 1.0 / LOGIT_WARMUP_STEPS
    total_loss = loss_seg + warmup_weight * args.lambda_logit * loss_logit
    total_loss.backward()
    if not all(torch.isfinite(value) for value in (loss_seg, loss_logit, total_loss)):
        raise RuntimeError("K4 smoke test produced a non-finite loss")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("K4 smoke test found a teacher gradient")
    student = k0.unwrap_model(model)
    backbone_gradients = sum(
        parameter.grad is not None for parameter in student.backbone.parameters()
    )
    head_gradients = sum(
        parameter.grad is not None for parameter in student.head.parameters()
    )
    if backbone_gradients == 0 or head_gradients == 0:
        raise RuntimeError("K4 smoke test did not produce end-to-end student gradients")
    if rank == 0:
        builtins.print(
            f"[OK] K4 server smoke test: sample={paths[0]}, "
            f"student_logits={tuple(logits_float.shape)}, "
            f"teacher_logits={tuple(teacher_logits.shape)}, "
            f"CE={loss_seg.item():.6f}, KL={loss_logit.item():.6f}, "
            f"total={total_loss.item():.6f}, warmup={warmup_weight:.6f}, "
            f"backbone_grad_tensors={backbone_gradients}, "
            f"head_grad_tensors={head_gradients}"
        )


def _patched_torch_save_atomic(payload: object, path: Path) -> None:
    if isinstance(payload, Mapping) and payload.get("artifact_type") == ARTIFACT_TYPE:
        payload = dict(payload)
        payload["experiment"] = EXPERIMENT
        payload["source_probe"] = _probe_metadata()
        payload["hashes"] = {
            **dict(payload.get("hashes", {})),
            **_resource_hashes(),
        }
    k2._ORIGINAL_TORCH_SAVE_ATOMIC(payload, path)


def _patched_evaluate(*args: Any, **kwargs: Any):
    split_name = kwargs.get("split_name")
    if isinstance(split_name, str):
        kwargs["split_name"] = split_name.replace("K0", EXPERIMENT)
    return k2._ORIGINAL_EVALUATE(*args, **kwargs)


def _k4_print(*values: object, **kwargs: object) -> None:
    adjusted = tuple(
        value.replace("K0", EXPERIMENT).replace("K2", EXPERIMENT)
        if isinstance(value, str)
        else value
        for value in values
    )
    builtins.print(*adjusted, **kwargs)


def _postprocess_metrics(args: argparse.Namespace) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    metrics_path = k4_paths(args.output_dir, args.seed)["metrics"]
    if not metrics_path.is_file():
        return
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    results["experiment"] = EXPERIMENT
    results["protocol"] = (
        "K4 staged-initialization logits-KD run: MobileNetV2+R-ASPP starts "
        "from the same-seed locked A0 best probe backbone+head, is fully "
        "unfrozen, and follows the A0-FT/S2-0 80k SGD+poly protocol with "
        "hard-label CE plus frozen T1 full-resolution masked pixel KL "
        "(T=4, lambda=0.5, first 4000 optimizer steps linear warm-up). No "
        "online feature KD/PCA/adapter and no test_local evaluation."
    )
    results["initialization"] = _probe_metadata()
    results["model"]["initialization"] = (
        "same-seed locked A0 best probe checkpoint (backbone+head)"
    )
    results["loss"] = {
        "hard_label_ce": True,
        "feature_kd": False,
        "logit_kd": True,
        "logit_mechanism": "full-resolution masked pixel KL",
        "logit_reduction": "mean over valid pixels after sum over 19 classes",
        "lambda_feat": None,
        "lambda_logit": args.lambda_logit,
        "temperature": args.temperature,
        "warmup_steps": LOGIT_WARMUP_STEPS,
        "warmup_ratio": LOGIT_WARMUP_RATIO,
        "ignore_index_masked": True,
        "temperature_squared_factor": True,
    }
    results["teacher"] = {
        "checkpoint": str(args.teacher_checkpoint.resolve()),
        "checkpoint_sha256": k2._TEACHER_CHECKPOINT_SHA256,
        "features_used": [],
        "logits_used": True,
        "logits_resolution": "full input resolution",
        "frozen": True,
    }
    results["pca"] = {
        "enabled_in_current_training": False,
        "feature_kd": False,
        "initialization_provenance": "A0 fixed StandardScaler+PCA pretraining",
        "online_projection": None,
    }
    results["registered_analysis"] = results["config"]["registered_analysis"]
    results["hashes"] = {
        **dict(results.get("hashes", {})),
        **_resource_hashes(),
    }
    results["test_local_evaluated"] = False
    common.write_json_atomic(metrics_path, results)


def _install_k4_hooks() -> None:
    k2._warmup_steps = _warmup_steps
    k2._resource_hashes = _resource_hashes
    k2.tqdm = _k4_tqdm
    k2._install_k2_hooks()
    k0.__file__ = str(Path(__file__).resolve())
    k0.EXPERIMENT = EXPERIMENT
    k0.ARTIFACT_TYPE = ARTIFACT_TYPE
    k0.ARTIFACT_FORMAT_VERSION = ARTIFACT_FORMAT_VERSION
    k0.k0_paths = k4_paths
    k0.ensure_shared_initialization = ensure_k4_resources
    k0.build_config = build_config
    k0.build_best_checkpoint = build_best_checkpoint
    k0.train_one_epoch_k0 = train_one_epoch_k4
    k0._smoke_test_k0 = smoke_test_k4
    k0.print = _k4_print
    base.audit_model_shapes = audit_k4_shapes
    common.set_global_seed = _a0_ft_compatible_set_global_seed
    common.torch_save_atomic = _patched_torch_save_atomic
    common.evaluate = _patched_evaluate


def _remove_k4_hooks() -> None:
    k2._remove_k2_hooks()
    k2._warmup_steps = _ORIGINAL_K2_WARMUP_STEPS
    k2._resource_hashes = _ORIGINAL_K2_RESOURCE_HASHES
    k2.tqdm = _ORIGINAL_K2_TQDM
    common.set_global_seed = _ORIGINAL_SET_GLOBAL_SEED


def run_training(args: argparse.Namespace) -> None:
    global _ACTIVE_ARGS
    global _PROBE_CHECKPOINT_SHA256
    global _PROBE_MODEL_STATE_SHA256
    global _PROBE_BACKBONE_STATE_SHA256
    global _PROBE_ARTIFACT_TYPE

    _ACTIVE_ARGS = args
    k2._ACTIVE_ARGS = args
    _install_k4_hooks()
    try:
        k0.run_training(args)
        if not args.smoke_test:
            _postprocess_metrics(args)
    finally:
        device = (
            torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
            if torch.cuda.is_available() and args.device != "cpu"
            else torch.device("cpu")
        )
        server_base._synchronize_cuda(device)
        k2._TEACHER = None
        k2._TEACHER_CHECKPOINT_SHA256 = None
        k2._ACTIVE_ARGS = None
        _ACTIVE_ARGS = None
        _PROBE_CHECKPOINT_SHA256 = None
        _PROBE_MODEL_STATE_SHA256 = None
        _PROBE_BACKBONE_STATE_SHA256 = None
        _PROBE_ARTIFACT_TYPE = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _remove_k4_hooks()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
