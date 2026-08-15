"""K2 server training entry point: hard-label CE plus pixel-logit KD.

K2 is the logits-only member of the controlled K0--K3 experiment.  It uses
the same shared scratch MobileNetV2 + R-ASPP initialization, data protocol,
optimizer, DDP runner, evaluation, checkpoint schema, and orderly shutdown
as :mod:`dino_k0_server`, and adds only the locked T1 teacher response loss::

    L = L_seg + warmup(step) * 0.5 * L_logit

where ``L_logit`` is a full-resolution, non-ignore-pixel KL divergence with
temperature ``T=4`` and the usual ``T**2`` scaling.  The teacher is the
validated and frozen T1 DINOv3 ConvNeXt-T + R-ASPP checkpoint; no PCA,
feature loss, student adapter, or teacher parameters are used.

Typical two-GPU server command::

    torchrun --standalone --nproc_per_node=2 dino_k2_server.py \
        --seed 42 --batch-size 2 --global-batch-size 8 \
        --num-workers 8 --multiprocessing-context spawn \
        --no-pin-memory --persistent-workers

Windows/local functional smoke (does not replace Linux two-GPU DDP smoke)::

    python -B dino_k2_server.py --device cuda --smoke-test \
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
import dino_k0_server as k0
import dino_s2_0 as base
import dino_s2_0_server as server_base
from dino_t1 import load_teacher_for_distillation


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "K_MobileNetV2_RASPP_server"
DEFAULT_TEACHER_CHECKPOINT = (
    SCRIPT_DIR
    / "result"
    / "T1_DINOv3_RASPP"
    / "seed_3407"
    / "t1_dinov3_raspp_teacher.pth"
)

EXPECTED_COMBINED_MANIFEST_SHA256 = k0.EXPECTED_COMBINED_MANIFEST_SHA256
EXPECTED_TEACHER_CHECKPOINT_SHA256 = (
    "73cb1d3161c746d1b4ea30918ec6a1f0de5e3a4952c000cf85ddf95f3ccaddeb"
)

EXPERIMENT = "K2"
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_k2_logit_kd"
ARTIFACT_FORMAT_VERSION = 1
FORMAL_SEEDS = k0.FORMAL_SEEDS
TEMPERATURE = 4.0
LAMBDA_LOGIT = 0.5
LOGIT_WARMUP_RATIO = 0.05
GRADIENT_LOG_STEPS = 500


_TEACHER: Optional[torch.nn.Module] = None
_TEACHER_CHECKPOINT_SHA256: Optional[str] = None
_ACTIVE_ARGS: Optional[argparse.Namespace] = None


# Keep references to every K0 symbol that is temporarily overridden.  K2 is
# normally a fresh process, but restoring the module is useful for imports,
# tests, and callers that invoke more than one entry point in one interpreter.
_ORIGINAL_K0_FILE = k0.__file__
_ORIGINAL_K0_EXPERIMENT = k0.EXPERIMENT
_ORIGINAL_K0_ARTIFACT_TYPE = k0.ARTIFACT_TYPE
_ORIGINAL_K0_ARTIFACT_FORMAT_VERSION = k0.ARTIFACT_FORMAT_VERSION
_ORIGINAL_K0_PATHS = k0.k0_paths
_ORIGINAL_K0_PRINT = getattr(k0, "print", None)
_ORIGINAL_ENSURE_SHARED_INITIALIZATION = k0.ensure_shared_initialization
_ORIGINAL_BUILD_CONFIG = k0.build_config
_ORIGINAL_BUILD_BEST_CHECKPOINT = k0.build_best_checkpoint
_ORIGINAL_TRAIN_ONE_EPOCH = k0.train_one_epoch_k0
_ORIGINAL_SMOKE_TEST = k0._smoke_test_k0
_ORIGINAL_TORCH_SAVE_ATOMIC = common.torch_save_atomic
_ORIGINAL_EVALUATE = common.evaluate
_ORIGINAL_BUILD_MODEL = base.build_model
_ORIGINAL_AUDIT_MODEL_SHAPES = base.audit_model_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "K2 MobileNetV2+R-ASPP training with hard-label CE and frozen "
            "T1 full-resolution pixel-logit KD."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=common.DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    parser.add_argument("--max-steps", type=int, default=80_000)
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
    parser.add_argument("--seed", type=int, default=42)
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
        parser.error("Formal K2 is locked to --temperature 4")
    if not math.isclose(args.lambda_logit, LAMBDA_LOGIT):
        parser.error("Formal K2 is locked to --lambda-logit 0.5")
    if not 0 < args.logit_warmup_ratio <= 1:
        parser.error("--logit-warmup-ratio must be in (0, 1]")
    if not math.isclose(args.logit_warmup_ratio, LOGIT_WARMUP_RATIO):
        parser.error("Formal K2 is locked to --logit-warmup-ratio 0.05")
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
    return args


def _warmup_steps(args: argparse.Namespace) -> int:
    return max(1, int(round(args.max_steps * args.logit_warmup_ratio)))


class K2MobileNetV2RASPPStudent(base.MobileNetV2RASPPStudent):
    """Expose the OS=16 tap during training for component gradient checks."""

    def forward(self, images: torch.Tensor) -> Any:
        input_size = images.shape[-2:]
        features = self.extract_features(images)
        logits = self.head(features["raspp_input"])
        logits = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )
        if not self.training:
            return logits
        return {"logits": logits, "os16": features["os16"]}


def build_k2_model(head_channels: int, dropout: float) -> K2MobileNetV2RASPPStudent:
    # The module/state-dict layout deliberately matches the K0 student so the
    # shared per-seed initialization remains byte-for-byte compatible.
    model = K2MobileNetV2RASPPStudent(
        backbone=base.build_backbone(),
        head=common.RASPPHead(
            in_channels=base.FEATURE_TAPS["raspp_input"]["channels"],
            num_classes=common.NUM_CLASSES,
            inter_channels=head_channels,
            dropout=dropout,
        ),
    )
    model.requires_grad_(True)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("K2 must train the complete student end to end")
    return model


def k2_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    original = _ORIGINAL_K0_PATHS(output_dir, seed)
    run_dir = output_dir.resolve() / "K2" / f"seed_{seed}"
    return {
        key: run_dir if key == "run_dir" else run_dir / value.name
        for key, value in original.items()
    }


def _require_teacher() -> torch.nn.Module:
    if _TEACHER is None:
        raise RuntimeError("K2 teacher was not initialized")
    return _TEACHER


def _resource_hashes() -> Dict[str, object]:
    return {
        "teacher_checkpoint_sha256": _TEACHER_CHECKPOINT_SHA256,
        "k0_shared_training_runner_sha256": common.sha256_file(
            Path(_ORIGINAL_K0_FILE).resolve()
        ),
        "pca_parameter_sha256": None,
        "pca_parameter_record_sha256": None,
    }


def ensure_k2_resources(
    model: base.MobileNetV2RASPPStudent,
    args: argparse.Namespace,
    output_dir: Path,
    seed: int,
    rank: int,
    world_size: int,
) -> Tuple[str, str, Path]:
    global _TEACHER
    global _TEACHER_CHECKPOINT_SHA256

    init_result = _ORIGINAL_ENSURE_SHARED_INITIALIZATION(
        model, args, output_dir, seed, rank, world_size
    )
    if _TEACHER is not None:
        return init_result

    device = next(model.parameters()).device
    rng_state = k0._capture_rank_rng_state(device)
    try:
        teacher_checkpoint = args.teacher_checkpoint.resolve()
        teacher_hash = common.verify_checkpoint_sidecar(teacher_checkpoint)
        if teacher_hash != EXPECTED_TEACHER_CHECKPOINT_SHA256:
            raise RuntimeError(
                "K2 must use the locked T1 seed=3407 teacher checkpoint: "
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
            raise RuntimeError("The K2 teacher is not fully frozen")
    finally:
        # Teacher construction can initialize a backbone.  It must not alter
        # the student's subsequent augmentation/DataLoader RNG stream.
        k0._restore_rank_rng_state(rng_state, device)

    _TEACHER = teacher
    _TEACHER_CHECKPOINT_SHA256 = teacher_hash
    return init_result


def audit_k2_shapes(
    model: base.MobileNetV2RASPPStudent,
    device: torch.device,
    height: int,
    width: int,
    amp_enabled: bool,
) -> Dict[str, object]:
    audit = _ORIGINAL_AUDIT_MODEL_SHAPES(
        model, device, height, width, amp_enabled
    )
    teacher = _require_teacher()
    teacher.eval()
    sample = torch.zeros(1, 3, height, width, device=device)
    with torch.no_grad(), common.autocast_context(device, amp_enabled):
        teacher_logits = teacher(sample)
    expected = (1, common.NUM_CLASSES, height, width)
    if tuple(teacher_logits.shape) != expected:
        raise RuntimeError(
            f"K2 teacher logit shape mismatch: actual={tuple(teacher_logits.shape)}, "
            f"expected={expected}"
        )
    audit.update(
        {
            "experiment": EXPERIMENT,
            "teacher_logit_shape": list(teacher_logits.shape),
            "teacher_logits_resolution": "full input resolution",
            "teacher_logits_align_corners": False,
            "teacher_checkpoint_sha256": _TEACHER_CHECKPOINT_SHA256,
        }
    )
    return audit


def build_config(
    args: argparse.Namespace,
    accumulation_steps: int,
    world_size: int,
    device: torch.device,
    shared_init_state_sha256: str,
    shared_init_file_sha256: str,
) -> Dict[str, object]:
    config = _ORIGINAL_BUILD_CONFIG(
        args,
        accumulation_steps,
        world_size,
        device,
        shared_init_state_sha256,
        shared_init_file_sha256,
    )
    config["experiment"] = EXPERIMENT
    config["server_entry_point"] = str(Path(__file__).resolve())
    config["loss"] = {
        "hard_label_ce": True,
        "feature_kd": False,
        "logit_kd": True,
        "logit_mechanism": "full-resolution masked pixel KL",
        "logit_reduction": "mean over valid pixels after sum over 19 classes",
        "lambda_feat": None,
        "lambda_logit": args.lambda_logit,
        "temperature": args.temperature,
        "auxiliary_warmup_steps": _warmup_steps(args),
        "auxiliary_warmup_ratio": args.logit_warmup_ratio,
        "warmup_step_unit": "optimizer_step",
        "ignore_index_masked": True,
        "teacher_logits_detached": True,
        "temperature_squared_factor": True,
    }
    config["teacher"] = {
        "enabled": True,
        "type": "T1 DINOv3 ConvNeXt-T + R-ASPP, full-resolution logits",
        "checkpoint": str(args.teacher_checkpoint.resolve()),
        "checkpoint_sha256": _TEACHER_CHECKPOINT_SHA256,
        "frozen": True,
        "wrapped_in_ddp": False,
        "features_used_for_loss": [],
        "logits_used": True,
        "logits_resolution": "full input resolution",
    }
    config["pca"] = {
        "enabled": False,
        "directory": None,
        "parameter_record_sha256": None,
        "trainable": False,
    }
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
    payload = _ORIGINAL_BUILD_BEST_CHECKPOINT(
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
    payload["loss_schema"] = {
        "hard_label_ce": True,
        "feature_kd": False,
        "logit_kd": True,
        "logit_mechanism": "full-resolution masked pixel KL",
        "temperature": TEMPERATURE,
        "lambda_logit": LAMBDA_LOGIT,
    }
    return payload


def _patched_torch_save_atomic(payload: object, path: Path) -> None:
    if isinstance(payload, Mapping) and payload.get("artifact_type") == ARTIFACT_TYPE:
        payload = dict(payload)
        payload["hashes"] = {
            **dict(payload.get("hashes", {})),
            **_resource_hashes(),
        }
    _ORIGINAL_TORCH_SAVE_ATOMIC(payload, path)


def _patched_evaluate(*args: Any, **kwargs: Any):
    split_name = kwargs.get("split_name")
    if isinstance(split_name, str):
        kwargs["split_name"] = split_name.replace("K0", "K2")
    return _ORIGINAL_EVALUATE(*args, **kwargs)


def _gradient_l2(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm(2).item())


def _masked_pixel_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute the registered full-resolution KL on non-ignore pixels only."""

    if teacher_logits.shape != student_logits.shape:
        raise RuntimeError(
            "Teacher/student logits must have identical shapes for K2 KL: "
            f"teacher={tuple(teacher_logits.shape)}, student={tuple(student_logits.shape)}"
        )
    if teacher_logits.ndim != 4 or targets.shape != teacher_logits.shape[:1] + teacher_logits.shape[2:]:
        raise RuntimeError(
            "K2 KL expects logits [B,C,H,W] and targets [B,H,W], got "
            f"logits={tuple(teacher_logits.shape)}, targets={tuple(targets.shape)}"
        )
    valid = targets != common.IGNORE_INDEX
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        raise RuntimeError("K2 KL batch contains no valid Cityscapes pixels")

    teacher_float = teacher_logits.float()
    student_float = student_logits.float()
    teacher_prob = F.softmax(teacher_float / temperature, dim=1)
    student_log_prob = F.log_softmax(student_float / temperature, dim=1)
    kl_per_pixel = F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="none",
    ).sum(dim=1)
    # Sum over classes is intentional; taking another mean would shrink the
    # registered loss by 19.  The mask excludes void pixels from the mean.
    return (temperature * temperature) * kl_per_pixel[valid].mean()


def _reduce_logit_statistics(
    logit_sum: float,
    total_sum: float,
    batch_count: int,
    device: torch.device,
    world_size: int,
) -> Tuple[float, float, int]:
    values = torch.tensor(
        [logit_sum, total_sum, float(batch_count)],
        device=device,
        dtype=torch.float64,
    )
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    count = int(values[2].item())
    denominator = max(count, 1)
    return (
        float(values[0].item() / denominator),
        float(values[1].item() / denominator),
        count,
    )


def train_one_epoch_k2(
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
    teacher = _require_teacher()
    args = _ACTIVE_ARGS
    if args is None:
        raise RuntimeError("K2 active arguments were not set")
    warmup_steps = _warmup_steps(args)
    if sampler is not None:
        sampler.set_epoch(epoch)
    model.train()
    teacher.eval()
    optimizer.zero_grad(set_to_none=True)

    confusion = torch.zeros(common.NUM_CLASSES, common.NUM_CLASSES, dtype=torch.int64)
    ce_loss_sum = 0.0
    valid_pixels = 0
    logit_sum = 0.0
    total_sum = 0.0
    batch_count = 0
    optimizer_steps = 0
    last_warmup_weight = 0.0
    gradient_records: List[Dict[str, object]] = []
    first_batch_audit: Optional[Dict[str, object]] = None

    possible_steps = math.ceil(len(loader) / accumulation_steps)
    target_steps = min(possible_steps, remaining_optimizer_steps)
    max_batches = min(len(loader), target_steps * accumulation_steps)
    progress = tqdm(loader, desc=f"Epoch {epoch} [K2 CE+logit]", disable=rank != 0)

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
                    raise RuntimeError("K2 training forward did not expose the OS=16 tap")
                logits = student_output["logits"]
                student_os16 = student_output["os16"]
                with torch.no_grad():
                    teacher_logits = teacher(images)

            logits_float = logits.float()
            batch_ce_sum = F.cross_entropy(
                logits_float,
                targets,
                ignore_index=common.IGNORE_INDEX,
                reduction="sum",
            )
            batch_valid = int((targets != common.IGNORE_INDEX).sum().item())
            if batch_valid == 0:
                raise RuntimeError("Training batch contains no valid Cityscapes pixels")
            loss_seg = batch_ce_sum / batch_valid
            loss_logit = _masked_pixel_kl(
                teacher_logits,
                logits_float,
                targets,
                args.temperature,
            )
            total_loss = loss_seg + warmup_weight * args.lambda_logit * loss_logit
            if not all(
                torch.isfinite(value)
                for value in (loss_seg, loss_logit, total_loss)
            ):
                raise RuntimeError("K2 produced a non-finite CE/KL loss")

            log_gradients = sync_gradients and (
                next_optimizer_step == 1
                or next_optimizer_step % args.gradient_log_steps == 0
            )
            grad_record: Optional[Dict[str, object]] = None
            if log_gradients:
                grad_seg = torch.autograd.grad(
                    loss_seg, student_os16, retain_graph=True, allow_unused=False
                )[0]
                grad_logit = torch.autograd.grad(
                    loss_logit, student_os16, retain_graph=True, allow_unused=False
                )[0]
                grad_total_os16 = (
                    grad_seg + warmup_weight * args.lambda_logit * grad_logit
                )
                grad_record = {
                    "optimizer_step": next_optimizer_step,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "warmup_weight": warmup_weight,
                    "lambda_logit": args.lambda_logit,
                    "temperature": args.temperature,
                    "grad_l2_ce": _gradient_l2(grad_seg),
                    "grad_l2_feature": None,
                    "grad_l2_logit": _gradient_l2(grad_logit),
                    "grad_l2_seg_os16": _gradient_l2(grad_seg),
                    "grad_l2_feat_os16": None,
                    "grad_l2_logit_os16": _gradient_l2(grad_logit),
                    "grad_l2_total_os16": _gradient_l2(grad_total_os16),
                    "feature_kd_enabled": False,
                    "logit_kd_enabled": True,
                    "gradient_component_scope": "student os16 tap",
                }
            scaler.scale(total_loss / group_size).backward()

        if sync_gradients:
            scaler.unscale_(optimizer)
            optimizer_steps += 1
            if grad_record is not None:
                grad_record["grad_l2_total_student"] = k0._gradient_l2_named(model)
                gradient_records.append(grad_record)
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
                "ce_loss": float(loss_seg.detach().item()),
                "logit_loss": float(loss_logit.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                "temperature": args.temperature,
                "lambda_logit": args.lambda_logit,
                "warmup_weight": warmup_weight,
                **_resource_hashes(),
            }

        predictions = logits_float.detach().argmax(dim=1)
        confusion += common.confusion_counts(predictions, targets)
        ce_loss_sum += float(batch_ce_sum.detach().item())
        valid_pixels += batch_valid
        logit_value = float(loss_logit.detach().item())
        logit_sum += logit_value
        total_sum += float(total_loss.detach().item())
        batch_count += 1
        last_warmup_weight = warmup_weight
        if rank == 0:
            running = common.metrics_from_confusion(
                confusion, ce_loss_sum, valid_pixels
            )
            progress.set_postfix(
                {
                    "CE": f"{running['loss']:.4f}",
                    "KL": f"{logit_value:.4f}",
                    "mIoU": f"{running['mIoU']:.4f}",
                    "warm": f"{warmup_weight:.3f}",
                    "steps": optimizer_steps,
                }
            )

    if optimizer_steps != target_steps:
        raise RuntimeError(
            f"K2 optimizer-step accounting failed: actual={optimizer_steps}, "
            f"expected={target_steps}"
        )
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("K2 training found a gradient on the frozen teacher")
    metrics = server_base._reduce_train_metrics(
        confusion, ce_loss_sum, valid_pixels, device, world_size
    )
    logit_mean, total_mean, global_batches = _reduce_logit_statistics(
        logit_sum, total_sum, batch_count, device, world_size
    )
    metrics["loss_schema"] = "hard_label_CE_plus_full_resolution_masked_pixel_KL"
    metrics["ce_loss"] = metrics["loss"]
    metrics["feature_loss"] = None
    metrics["logit_loss"] = logit_mean
    metrics["total_loss_micro_batch_mean"] = total_mean
    metrics["warmup_weight"] = last_warmup_weight
    metrics["micro_batches_global"] = global_batches
    return metrics, optimizer_steps, gradient_records, first_batch_audit


def smoke_test_k2(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    rank: int,
) -> None:
    teacher = _require_teacher()
    args = _ACTIVE_ARGS
    if args is None:
        raise RuntimeError("K2 active arguments were not set")
    model.train()
    teacher.eval()
    images, targets, paths = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    model.zero_grad(set_to_none=True)
    with common.autocast_context(device, amp_enabled):
        student_output = model(images)
        if not isinstance(student_output, Mapping):
            raise RuntimeError("K2 smoke training forward did not expose the OS=16 tap")
        logits = student_output["logits"]
        with torch.no_grad():
            teacher_logits = teacher(images)
    logits_float = logits.float()
    valid_pixels = int((targets != common.IGNORE_INDEX).sum().item())
    if valid_pixels == 0:
        raise RuntimeError("K2 smoke batch contains no valid pixels")
    loss_seg = F.cross_entropy(
        logits_float,
        targets,
        ignore_index=common.IGNORE_INDEX,
        reduction="sum",
    ) / valid_pixels
    loss_logit = _masked_pixel_kl(
        teacher_logits, logits_float, targets, args.temperature
    )
    warmup_weight = 1.0 / _warmup_steps(args)
    total_loss = loss_seg + warmup_weight * args.lambda_logit * loss_logit
    total_loss.backward()
    if not all(
        torch.isfinite(value) for value in (loss_seg, loss_logit, total_loss)
    ):
        raise RuntimeError("K2 smoke test produced a non-finite loss")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("K2 smoke test found a teacher gradient")
    backbone_gradients = sum(
        parameter.grad is not None
        for parameter in k0.unwrap_model(model).backbone.parameters()
    )
    head_gradients = sum(
        parameter.grad is not None
        for parameter in k0.unwrap_model(model).head.parameters()
    )
    if backbone_gradients == 0 or head_gradients == 0:
        raise RuntimeError("K2 smoke test did not produce end-to-end student gradients")
    if rank == 0:
        print(
            f"[OK] K2 server DDP smoke test: sample={paths[0]}, "
            f"student_logits={tuple(logits_float.shape)}, "
            f"teacher_logits={tuple(teacher_logits.shape)}, "
            f"CE={loss_seg.item():.6f}, KL={loss_logit.item():.6f}, "
            f"total={total_loss.item():.6f}, warmup={warmup_weight:.6f}, "
            f"backbone_grad_tensors={backbone_gradients}, "
            f"head_grad_tensors={head_gradients}"
        )


def _postprocess_metrics(args: argparse.Namespace) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    metrics_path = k2_paths(args.output_dir, args.seed)["metrics"]
    if not metrics_path.is_file():
        return
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    results["experiment"] = EXPERIMENT
    results["protocol"] = (
        "K2 controlled logits-KD run: shared scratch MobileNetV2+R-ASPP "
        "initialization, hard-label CE plus frozen T1 full-resolution masked "
        "pixel-logit KL (T=4, lambda=0.5), 5% optimizer-step warm-up, no "
        "feature KD/PCA, fixed 80k budget, dev_local selection, and no "
        "test_local evaluation."
    )
    results["loss"] = {
        "hard_label_ce": True,
        "feature_kd": False,
        "logit_kd": True,
        "logit_mechanism": "full-resolution masked pixel KL",
        "logit_reduction": "mean over valid pixels after sum over classes",
        "lambda_feat": None,
        "lambda_logit": args.lambda_logit,
        "temperature": args.temperature,
        "warmup_steps": _warmup_steps(args),
        "warmup_ratio": args.logit_warmup_ratio,
        "ignore_index_masked": True,
        "temperature_squared_factor": True,
    }
    results["teacher"] = {
        "checkpoint": str(args.teacher_checkpoint.resolve()),
        "checkpoint_sha256": _TEACHER_CHECKPOINT_SHA256,
        "features_used": [],
        "logits_used": True,
        "logits_resolution": "full input resolution",
        "frozen": True,
    }
    results["pca"] = {
        "enabled": False,
        "directory": None,
        "parameter_record_sha256": None,
        "projection_parameter_sha256": None,
    }
    results["hashes"] = {
        **dict(results.get("hashes", {})),
        **_resource_hashes(),
    }
    results["test_local_evaluated"] = False
    common.write_json_atomic(metrics_path, results)


def _k2_print(*values: object, **kwargs: object) -> None:
    adjusted = tuple(
        value.replace("K0", "K2") if isinstance(value, str) else value
        for value in values
    )
    builtins.print(*adjusted, **kwargs)


def _install_k2_hooks() -> None:
    k0.__file__ = str(Path(__file__).resolve())
    k0.EXPERIMENT = EXPERIMENT
    k0.ARTIFACT_TYPE = ARTIFACT_TYPE
    k0.ARTIFACT_FORMAT_VERSION = ARTIFACT_FORMAT_VERSION
    k0.k0_paths = k2_paths
    k0.ensure_shared_initialization = ensure_k2_resources
    k0.build_config = build_config
    k0.build_best_checkpoint = build_best_checkpoint
    k0.train_one_epoch_k0 = train_one_epoch_k2
    k0._smoke_test_k0 = smoke_test_k2
    k0.print = _k2_print
    base.build_model = build_k2_model
    base.audit_model_shapes = audit_k2_shapes
    common.torch_save_atomic = _patched_torch_save_atomic
    common.evaluate = _patched_evaluate


def _remove_k2_hooks() -> None:
    k0.__file__ = _ORIGINAL_K0_FILE
    k0.EXPERIMENT = _ORIGINAL_K0_EXPERIMENT
    k0.ARTIFACT_TYPE = _ORIGINAL_K0_ARTIFACT_TYPE
    k0.ARTIFACT_FORMAT_VERSION = _ORIGINAL_K0_ARTIFACT_FORMAT_VERSION
    k0.k0_paths = _ORIGINAL_K0_PATHS
    k0.ensure_shared_initialization = _ORIGINAL_ENSURE_SHARED_INITIALIZATION
    k0.build_config = _ORIGINAL_BUILD_CONFIG
    k0.build_best_checkpoint = _ORIGINAL_BUILD_BEST_CHECKPOINT
    k0.train_one_epoch_k0 = _ORIGINAL_TRAIN_ONE_EPOCH
    k0._smoke_test_k0 = _ORIGINAL_SMOKE_TEST
    if _ORIGINAL_K0_PRINT is None:
        if hasattr(k0, "print"):
            delattr(k0, "print")
    else:
        k0.print = _ORIGINAL_K0_PRINT
    base.build_model = _ORIGINAL_BUILD_MODEL
    base.audit_model_shapes = _ORIGINAL_AUDIT_MODEL_SHAPES
    common.torch_save_atomic = _ORIGINAL_TORCH_SAVE_ATOMIC
    common.evaluate = _ORIGINAL_EVALUATE


def run_training(args: argparse.Namespace) -> None:
    global _ACTIVE_ARGS
    global _TEACHER
    global _TEACHER_CHECKPOINT_SHA256
    _ACTIVE_ARGS = args
    _install_k2_hooks()
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
        _TEACHER = None
        _TEACHER_CHECKPOINT_SHA256 = None
        _ACTIVE_ARGS = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _remove_k2_hooks()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
