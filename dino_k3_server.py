"""K3 server entry point: hard-label CE plus feature and pixel-logit KD.

K3 is the combined member of the controlled K0--K3 experiment.  It starts
from the same per-seed scratch MobileNetV2 + R-ASPP state as K0--K2 and uses
the locked K1 and K2 auxiliary objectives without changing their weights::

    L = L_seg + warmup(step) * (
        1.0 * L_feat + 0.5 * L_logit
    )

``L_feat`` is the equal mean of the fixed A0 StandardScaler+PCA feature MSE
at OS=4/8/16.  ``L_logit`` is K2's full-resolution masked pixel KL with
temperature T=4 and T**2 scaling.  A single frozen-teacher feature forward
is shared by both losses; the teacher R-ASPP head consumes the returned
OS=16 feature, so K3 does not run the teacher backbone twice per batch.

Typical two-GPU server command::

    torchrun --standalone --nproc_per_node=2 dino_k3_server.py \
        --seed 42 --batch-size 2 --global-batch-size 8 \
        --num-workers 8 --multiprocessing-context spawn \
        --no-pin-memory --persistent-workers

Windows/local functional smoke (does not replace Linux two-GPU DDP smoke)::

    python -B dino_k3_server.py --device cuda --smoke-test \
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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import dino as common
import dino_a0_server as a0
import dino_k0_server as k0
import dino_k1_server as k1
import dino_k2_server as k2
import dino_s2_0 as base
import dino_s2_0_server as server_base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "K_MobileNetV2_RASPP_server"
DEFAULT_TEACHER_CHECKPOINT = k1.DEFAULT_TEACHER_CHECKPOINT
DEFAULT_PCA_DIR = k1.DEFAULT_PCA_DIR

EXPECTED_COMBINED_MANIFEST_SHA256 = k0.EXPECTED_COMBINED_MANIFEST_SHA256
EXPECTED_TEACHER_CHECKPOINT_SHA256 = k1.EXPECTED_TEACHER_CHECKPOINT_SHA256
EXPERIMENT = "K3"
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_k3_feature_logit_kd"
ARTIFACT_FORMAT_VERSION = 1
FORMAL_SEEDS = k0.FORMAL_SEEDS
LAMBDA_FEAT = k1.LAMBDA_FEAT
LAMBDA_LOGIT = k2.LAMBDA_LOGIT
TEMPERATURE = k2.TEMPERATURE
AUXILIARY_WARMUP_RATIO = 0.05
GRADIENT_LOG_STEPS = 500


_ACTIVE_ARGS: Optional[argparse.Namespace] = None

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
            "K3 MobileNetV2+R-ASPP training with hard-label CE, the locked "
            "A0 feature target, and frozen T1 full-resolution pixel-logit KD."
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
    parser.add_argument("--pca-dir", type=Path, default=DEFAULT_PCA_DIR)
    parser.add_argument("--lambda-feat", type=float, default=LAMBDA_FEAT)
    parser.add_argument("--lambda-logit", type=float, default=LAMBDA_LOGIT)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument(
        "--feature-warmup-ratio", type=float, default=AUXILIARY_WARMUP_RATIO
    )
    parser.add_argument(
        "--logit-warmup-ratio", type=float, default=AUXILIARY_WARMUP_RATIO
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
    if not math.isclose(args.lambda_feat, LAMBDA_FEAT):
        parser.error("Formal K3 is locked to --lambda-feat 1.0")
    if not math.isclose(args.lambda_logit, LAMBDA_LOGIT):
        parser.error("Formal K3 is locked to --lambda-logit 0.5")
    if not math.isclose(args.temperature, TEMPERATURE):
        parser.error("Formal K3 is locked to --temperature 4")
    for name in ("feature_warmup_ratio", "logit_warmup_ratio"):
        value = getattr(args, name)
        if not 0 < value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1]")
        if not math.isclose(value, AUXILIARY_WARMUP_RATIO):
            parser.error(
                f"Formal K3 is locked to --{name.replace('_', '-')} 0.05"
            )
    if not math.isclose(args.feature_warmup_ratio, args.logit_warmup_ratio):
        parser.error("K3 feature/logit warm-up ratios must be identical")
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
    return max(1, int(round(args.max_steps * args.feature_warmup_ratio)))


def build_k3_model(
    head_channels: int, dropout: float
) -> k1.K1MobileNetV2RASPPStudent:
    # Reuse K1's tapped student.  Its module/state-dict layout is identical to
    # K0/K2, while its training output exposes all three registered features.
    return k1.build_k1_model(head_channels, dropout)


def k3_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    original = _ORIGINAL_K0_PATHS(output_dir, seed)
    run_dir = output_dir.resolve() / "K3" / f"seed_{seed}"
    return {
        key: run_dir if key == "run_dir" else run_dir / value.name
        for key, value in original.items()
    }


def _require_resources() -> Tuple[torch.nn.Module, nn.ModuleDict]:
    return k1._require_resources()


def _resource_hashes() -> Dict[str, object]:
    return {
        **k1._resource_hashes(),
        "k1_feature_contract_source_sha256": common.sha256_file(
            Path(k1.__file__).resolve()
        ),
        "k2_masked_kl_source_sha256": common.sha256_file(
            Path(k2.__file__).resolve()
        ),
    }


def ensure_k3_resources(
    model: base.MobileNetV2RASPPStudent,
    args: argparse.Namespace,
    output_dir: Path,
    seed: int,
    rank: int,
    world_size: int,
) -> Tuple[str, str, Path]:
    # K1 owns the locked teacher/PCA loader and restores RNG state after
    # resource construction, preserving K0--K3 batch and augmentation order.
    result = k1.ensure_k1_resources(
        model, args, output_dir, seed, rank, world_size
    )
    teacher, projection = _require_resources()
    if k1._TEACHER_CHECKPOINT_SHA256 != EXPECTED_TEACHER_CHECKPOINT_SHA256:
        raise RuntimeError("K3 teacher checkpoint hash differs from the locked T1")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("The K3 teacher is not fully frozen")
    if list(projection.parameters()):
        raise RuntimeError("The K3 fixed A0 projection unexpectedly has parameters")
    return result


def _teacher_features_and_logits(
    teacher: torch.nn.Module, images: torch.Tensor
) -> Tuple[Mapping[str, torch.Tensor], torch.Tensor]:
    features = teacher.extract_features(images)
    low_resolution_logits = teacher.teacher_head(features["os16"])
    logits = F.interpolate(
        low_resolution_logits,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return features, logits


def _feature_kd_losses(
    student_features: Mapping[str, torch.Tensor],
    teacher_features: Mapping[str, torch.Tensor],
    projection: nn.ModuleDict,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[int]]]:
    layer_losses: Dict[str, torch.Tensor] = {}
    projected_shapes: Dict[str, List[int]] = {}
    for layer in a0.A0_LAYER_ORDER:
        projected_teacher = projection[layer](teacher_features[layer].detach())
        projected_shapes[layer] = list(projected_teacher.shape)
        layer_losses[layer] = F.mse_loss(
            student_features[layer].float(), projected_teacher.float()
        )
    return layer_losses, projected_shapes


def audit_k3_shapes(
    model: base.MobileNetV2RASPPStudent,
    device: torch.device,
    height: int,
    width: int,
    amp_enabled: bool,
) -> Dict[str, object]:
    audit = _ORIGINAL_AUDIT_MODEL_SHAPES(
        model, device, height, width, amp_enabled
    )
    teacher, projection = _require_resources()
    teacher.eval()
    sample = torch.zeros(1, 3, height, width, device=device)
    with torch.no_grad(), common.autocast_context(device, amp_enabled):
        teacher_features, teacher_logits = _teacher_features_and_logits(
            teacher, sample
        )

    teacher_shapes: Dict[str, List[int]] = {}
    projected_shapes: Dict[str, List[int]] = {}
    for layer in a0.A0_LAYER_ORDER:
        expected_teacher = [
            1,
            a0.TEACHER_CHANNELS[layer],
            height // int(layer[2:]),
            width // int(layer[2:]),
        ]
        actual_teacher = list(teacher_features[layer].shape)
        if actual_teacher != expected_teacher:
            raise RuntimeError(
                f"K3 teacher {layer} shape mismatch: "
                f"actual={actual_teacher}, expected={expected_teacher}"
            )
        with torch.no_grad():
            projected = projection[layer](teacher_features[layer].float())
        expected_projected = [
            1,
            a0.STUDENT_CHANNELS[layer],
            height // int(layer[2:]),
            width // int(layer[2:]),
        ]
        if list(projected.shape) != expected_projected:
            raise RuntimeError(
                f"K3 projected {layer} shape mismatch: "
                f"actual={list(projected.shape)}, expected={expected_projected}"
            )
        teacher_shapes[layer] = actual_teacher
        projected_shapes[layer] = list(projected.shape)

    expected_logits = (1, common.NUM_CLASSES, height, width)
    if tuple(teacher_logits.shape) != expected_logits:
        raise RuntimeError(
            "K3 shared teacher logit shape mismatch: "
            f"actual={tuple(teacher_logits.shape)}, expected={expected_logits}"
        )

    # Prove in FP32 that the shared feature->head route is the teacher's
    # normal forward route.  A smaller audit tensor limits one-time peak RAM.
    equivalence_height = min(height, 64)
    equivalence_width = min(width, 128)
    equivalence_sample = torch.zeros(
        1, 3, equivalence_height, equivalence_width, device=device
    )
    with torch.no_grad():
        shared_features, shared_logits = _teacher_features_and_logits(
            teacher, equivalence_sample
        )
        reference_logits = teacher(equivalence_sample)
    max_abs_error = float(
        (shared_logits.float() - reference_logits.float()).abs().max().item()
    )
    equivalent = torch.allclose(
        shared_logits.float(), reference_logits.float(), rtol=1e-5, atol=1e-6
    )
    if not equivalent:
        raise RuntimeError(
            "K3 shared teacher feature->head path differs from teacher(images): "
            f"max_abs_error={max_abs_error:.8g}"
        )
    del shared_features

    audit.update(
        {
            "experiment": EXPERIMENT,
            "teacher_feature_shapes": teacher_shapes,
            "projected_teacher_feature_shapes": projected_shapes,
            "teacher_logit_shape": list(teacher_logits.shape),
            "teacher_logits_resolution": "full input resolution",
            "teacher_logits_align_corners": False,
            "teacher_forward_calls_per_training_batch": 1,
            "teacher_shared_feature_head_path": True,
            "teacher_shared_path_equivalence": {
                "passed": bool(equivalent),
                "dtype": "float32",
                "input_shape": list(equivalence_sample.shape),
                "rtol": 1e-5,
                "atol": 1e-6,
                "max_abs_error": max_abs_error,
            },
            "projection": "fixed A0 StandardScaler+PCA, teacher to student channels",
            "teacher_checkpoint_sha256": k1._TEACHER_CHECKPOINT_SHA256,
            "projection_parameter_sha256": copy.deepcopy(k1._PROJECTION_HASHES),
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
        "feature_kd": True,
        "feature_mechanism": "A0 fixed StandardScaler+PCA teacher-to-student",
        "feature_layers": list(a0.A0_LAYER_ORDER),
        "feature_reduction": "mean per BCHW layer, then equal mean over 3 layers",
        "logit_kd": True,
        "logit_mechanism": "full-resolution masked pixel KL",
        "logit_reduction": "mean over valid pixels after sum over 19 classes",
        "lambda_feat": args.lambda_feat,
        "lambda_logit": args.lambda_logit,
        "temperature": args.temperature,
        "auxiliary_warmup_steps": _warmup_steps(args),
        "auxiliary_warmup_ratio": args.feature_warmup_ratio,
        "warmup_step_unit": "optimizer_step",
        "shared_auxiliary_warmup": True,
        "ignore_index_masked": True,
        "teacher_logits_detached": True,
        "temperature_squared_factor": True,
    }
    config["teacher"] = {
        "enabled": True,
        "type": "T1 DINOv3 ConvNeXt-T + R-ASPP, shared features and logits",
        "checkpoint": str(args.teacher_checkpoint.resolve()),
        "checkpoint_sha256": k1._TEACHER_CHECKPOINT_SHA256,
        "frozen": True,
        "wrapped_in_ddp": False,
        "features_used_for_loss": list(a0.A0_LAYER_ORDER),
        "logits_used": True,
        "logits_resolution": "full input resolution",
        "single_backbone_forward_per_batch": True,
    }
    config["pca"] = {
        "enabled": True,
        "mechanism": "A0 fixed StandardScaler+PCA",
        "directory": str(args.pca_dir.resolve()),
        "parameter_record_sha256": k1._PCA_PARAMETER_RECORD_SHA256,
        "parameter_sha256": copy.deepcopy(k1._PROJECTION_HASHES),
        "sampling_manifest_sha256": (
            None
            if k1._PCA_PARAMETER_RECORD is None
            else k1._PCA_PARAMETER_RECORD.get("sampling_manifest_sha256")
        ),
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
        "feature_kd": True,
        "feature_mechanism": "A0 fixed StandardScaler+PCA",
        "logit_kd": True,
        "logit_mechanism": "full-resolution masked pixel KL",
        "lambda_feat": LAMBDA_FEAT,
        "lambda_logit": LAMBDA_LOGIT,
        "temperature": TEMPERATURE,
    }
    payload["pca_parameters_sha256_record"] = copy.deepcopy(
        k1._PCA_PARAMETER_RECORD
    )
    return payload


def _patched_torch_save_atomic(payload: object, path: Path) -> None:
    if isinstance(payload, Mapping) and payload.get("artifact_type") == ARTIFACT_TYPE:
        payload = dict(payload)
        payload["hashes"] = {
            **dict(payload.get("hashes", {})),
            **_resource_hashes(),
        }
        payload["pca_parameters_sha256_record"] = copy.deepcopy(
            k1._PCA_PARAMETER_RECORD
        )
    _ORIGINAL_TORCH_SAVE_ATOMIC(payload, path)


def _patched_evaluate(*args: Any, **kwargs: Any):
    split_name = kwargs.get("split_name")
    if isinstance(split_name, str):
        kwargs["split_name"] = split_name.replace("K0", "K3")
    return _ORIGINAL_EVALUATE(*args, **kwargs)


def _gradient_l2(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm(2).item())


def _reduce_auxiliary_statistics(
    layer_sums: Mapping[str, float],
    feature_sum: float,
    logit_sum: float,
    total_sum: float,
    batch_count: int,
    device: torch.device,
    world_size: int,
) -> Tuple[Dict[str, float], float, float, float, int]:
    values = [layer_sums[layer] for layer in a0.A0_LAYER_ORDER]
    values.extend(
        [feature_sum, logit_sum, total_sum, float(batch_count)]
    )
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    count = int(tensor[-1].item())
    denominator = max(count, 1)
    layer_means = {
        layer: float(tensor[index].item() / denominator)
        for index, layer in enumerate(a0.A0_LAYER_ORDER)
    }
    return (
        layer_means,
        float(tensor[-4].item() / denominator),
        float(tensor[-3].item() / denominator),
        float(tensor[-2].item() / denominator),
        count,
    )


def train_one_epoch_k3(
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
    teacher, projection = _require_resources()
    args = _ACTIVE_ARGS
    if args is None:
        raise RuntimeError("K3 active arguments were not set")
    warmup_steps = _warmup_steps(args)
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
    logit_sum = 0.0
    total_sum = 0.0
    layer_sums = {layer: 0.0 for layer in a0.A0_LAYER_ORDER}
    batch_count = 0
    optimizer_steps = 0
    last_warmup_weight = 0.0
    gradient_records: List[Dict[str, object]] = []
    first_batch_audit: Optional[Dict[str, object]] = None

    possible_steps = math.ceil(len(loader) / accumulation_steps)
    target_steps = min(possible_steps, remaining_optimizer_steps)
    max_batches = min(len(loader), target_steps * accumulation_steps)
    progress = tqdm(
        loader, desc=f"Epoch {epoch} [K3 CE+feature+logit]", disable=rank != 0
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
                    raise RuntimeError("K3 training forward did not return features")
                logits = student_output["logits"]
                student_features = student_output["features"]
                with torch.no_grad():
                    teacher_features, teacher_logits = _teacher_features_and_logits(
                        teacher, images
                    )
                layer_losses, projected_shapes = _feature_kd_losses(
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
                raise RuntimeError("Training batch contains no valid Cityscapes pixels")
            loss_seg = batch_ce_sum / batch_valid
            loss_feat = sum(layer_losses.values()) / len(a0.A0_LAYER_ORDER)
            loss_logit = k2._masked_pixel_kl(
                teacher_logits,
                logits_float,
                targets,
                args.temperature,
            )
            total_loss = loss_seg + warmup_weight * (
                args.lambda_feat * loss_feat + args.lambda_logit * loss_logit
            )
            if not all(
                torch.isfinite(value)
                for value in [
                    loss_seg,
                    loss_feat,
                    loss_logit,
                    total_loss,
                    *layer_losses.values(),
                ]
            ):
                raise RuntimeError("K3 produced a non-finite CE/feature/KL loss")

            log_gradients = sync_gradients and (
                next_optimizer_step == 1
                or next_optimizer_step % args.gradient_log_steps == 0
            )
            grad_record: Optional[Dict[str, object]] = None
            if log_gradients:
                student_os16 = student_features["os16"]
                grad_seg = torch.autograd.grad(
                    loss_seg, student_os16, retain_graph=True, allow_unused=False
                )[0]
                grad_feat = torch.autograd.grad(
                    loss_feat, student_os16, retain_graph=True, allow_unused=False
                )[0]
                grad_logit = torch.autograd.grad(
                    loss_logit, student_os16, retain_graph=True, allow_unused=False
                )[0]
                grad_total_os16 = grad_seg + warmup_weight * (
                    args.lambda_feat * grad_feat + args.lambda_logit * grad_logit
                )
                grad_record = {
                    "optimizer_step": next_optimizer_step,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "warmup_weight": warmup_weight,
                    "lambda_feat": args.lambda_feat,
                    "lambda_logit": args.lambda_logit,
                    "temperature": args.temperature,
                    "grad_l2_ce": _gradient_l2(grad_seg),
                    "grad_l2_feature": _gradient_l2(grad_feat),
                    "grad_l2_logit": _gradient_l2(grad_logit),
                    "grad_l2_seg_os16": _gradient_l2(grad_seg),
                    "grad_l2_feat_os16": _gradient_l2(grad_feat),
                    "grad_l2_logit_os16": _gradient_l2(grad_logit),
                    "grad_l2_total_os16": _gradient_l2(grad_total_os16),
                    "feature_kd_enabled": True,
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
                    layer: float(layer_losses[layer].detach().item())
                    for layer in a0.A0_LAYER_ORDER
                },
                "ce_loss": float(loss_seg.detach().item()),
                "feature_loss": float(loss_feat.detach().item()),
                "logit_loss": float(loss_logit.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                "temperature": args.temperature,
                "lambda_feat": args.lambda_feat,
                "lambda_logit": args.lambda_logit,
                "warmup_weight": warmup_weight,
                "teacher_backbone_forward_count": 1,
                **_resource_hashes(),
            }

        predictions = logits_float.detach().argmax(dim=1)
        confusion += common.confusion_counts(predictions, targets)
        ce_loss_sum += float(batch_ce_sum.detach().item())
        valid_pixels += batch_valid
        feature_value = float(loss_feat.detach().item())
        logit_value = float(loss_logit.detach().item())
        feature_sum += feature_value
        logit_sum += logit_value
        total_sum += float(total_loss.detach().item())
        for layer in a0.A0_LAYER_ORDER:
            layer_sums[layer] += float(layer_losses[layer].detach().item())
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
                    "KL": f"{logit_value:.4f}",
                    "mIoU": f"{running['mIoU']:.4f}",
                    "warm": f"{warmup_weight:.3f}",
                    "steps": optimizer_steps,
                }
            )

    if optimizer_steps != target_steps:
        raise RuntimeError(
            f"K3 optimizer-step accounting failed: actual={optimizer_steps}, "
            f"expected={target_steps}"
        )
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("K3 training found a gradient on the frozen teacher")
    if list(projection.parameters()):
        raise RuntimeError("K3 projection became trainable during training")
    metrics = server_base._reduce_train_metrics(
        confusion, ce_loss_sum, valid_pixels, device, world_size
    )
    (
        layer_means,
        feature_mean,
        logit_mean,
        total_mean,
        global_batches,
    ) = _reduce_auxiliary_statistics(
        layer_sums,
        feature_sum,
        logit_sum,
        total_sum,
        batch_count,
        device,
        world_size,
    )
    metrics["loss_schema"] = (
        "hard_label_CE_plus_A0_fixed_PCA_feature_MSE_plus_"
        "full_resolution_masked_pixel_KL"
    )
    metrics["ce_loss"] = metrics["loss"]
    metrics["feature_loss"] = feature_mean
    metrics["feature_loss_by_layer"] = layer_means
    metrics["logit_loss"] = logit_mean
    metrics["total_loss_micro_batch_mean"] = total_mean
    metrics["warmup_weight"] = last_warmup_weight
    metrics["micro_batches_global"] = global_batches
    return metrics, optimizer_steps, gradient_records, first_batch_audit


def smoke_test_k3(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    rank: int,
) -> None:
    teacher, projection = _require_resources()
    args = _ACTIVE_ARGS
    if args is None:
        raise RuntimeError("K3 active arguments were not set")
    model.train()
    teacher.eval()
    images, targets, paths = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    model.zero_grad(set_to_none=True)
    with common.autocast_context(device, amp_enabled):
        student_output = model(images)
        if not isinstance(student_output, Mapping):
            raise RuntimeError("K3 smoke training forward did not expose features")
        with torch.no_grad():
            teacher_features, teacher_logits = _teacher_features_and_logits(
                teacher, images
            )
        student_features = student_output["features"]
        layer_losses, _projected_shapes = _feature_kd_losses(
            student_features, teacher_features, projection
        )

    logits = student_output["logits"].float()
    valid_pixels = int((targets != common.IGNORE_INDEX).sum().item())
    if valid_pixels == 0:
        raise RuntimeError("K3 smoke batch contains no valid pixels")
    loss_seg = F.cross_entropy(
        logits,
        targets,
        ignore_index=common.IGNORE_INDEX,
        reduction="sum",
    ) / valid_pixels
    loss_feat = sum(layer_losses.values()) / len(a0.A0_LAYER_ORDER)
    loss_logit = k2._masked_pixel_kl(
        teacher_logits, logits, targets, args.temperature
    )
    warmup_weight = 1.0 / _warmup_steps(args)
    total_loss = loss_seg + warmup_weight * (
        args.lambda_feat * loss_feat + args.lambda_logit * loss_logit
    )
    total_loss.backward()
    if not all(
        torch.isfinite(value)
        for value in [
            loss_seg,
            loss_feat,
            loss_logit,
            total_loss,
            *layer_losses.values(),
        ]
    ):
        raise RuntimeError("K3 smoke test produced a non-finite loss")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("K3 smoke test found a teacher gradient")
    if list(projection.parameters()):
        raise RuntimeError("K3 smoke test found trainable projection parameters")
    backbone_gradients = sum(
        parameter.grad is not None
        for parameter in k0.unwrap_model(model).backbone.parameters()
    )
    head_gradients = sum(
        parameter.grad is not None
        for parameter in k0.unwrap_model(model).head.parameters()
    )
    if backbone_gradients == 0 or head_gradients == 0:
        raise RuntimeError("K3 smoke test did not produce end-to-end student gradients")
    if rank == 0:
        print(
            f"[OK] K3 server DDP smoke test: sample={paths[0]}, "
            f"student_logits={tuple(logits.shape)}, "
            f"teacher_logits={tuple(teacher_logits.shape)}, "
            f"CE={loss_seg.item():.6f}, feature={loss_feat.item():.6f}, "
            f"KL={loss_logit.item():.6f}, total={total_loss.item():.6f}, "
            f"warmup={warmup_weight:.6f}, "
            f"backbone_grad_tensors={backbone_gradients}, "
            f"head_grad_tensors={head_gradients}"
        )


def _postprocess_metrics(args: argparse.Namespace) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    metrics_path = k3_paths(args.output_dir, args.seed)["metrics"]
    if not metrics_path.is_file():
        return
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    results["experiment"] = EXPERIMENT
    results["protocol"] = (
        "K3 controlled feature+logits KD run: shared scratch "
        "MobileNetV2+R-ASPP initialization, hard-label CE plus locked A0 "
        "fixed StandardScaler+PCA OS=4/8/16 feature MSE and frozen T1 "
        "full-resolution masked pixel-logit KL (T=4, lambda_feat=1.0, "
        "lambda_logit=0.5), shared 5% optimizer-step warm-up, one teacher "
        "backbone forward per batch, fixed 80k budget, dev_local selection, "
        "and no test_local evaluation."
    )
    results["loss"] = {
        "hard_label_ce": True,
        "feature_kd": True,
        "feature_mechanism": "A0 fixed StandardScaler+PCA",
        "feature_layers": list(a0.A0_LAYER_ORDER),
        "lambda_feat": args.lambda_feat,
        "logit_kd": True,
        "logit_mechanism": "full-resolution masked pixel KL",
        "logit_reduction": "mean over valid pixels after sum over classes",
        "lambda_logit": args.lambda_logit,
        "temperature": args.temperature,
        "warmup_steps": _warmup_steps(args),
        "warmup_ratio": args.feature_warmup_ratio,
        "ignore_index_masked": True,
        "temperature_squared_factor": True,
    }
    results["teacher"] = {
        "checkpoint": str(args.teacher_checkpoint.resolve()),
        "checkpoint_sha256": k1._TEACHER_CHECKPOINT_SHA256,
        "features_used": list(a0.A0_LAYER_ORDER),
        "logits_used": True,
        "logits_resolution": "full input resolution",
        "single_backbone_forward_per_batch": True,
        "frozen": True,
    }
    results["pca"] = {
        "enabled": True,
        "directory": str(args.pca_dir.resolve()),
        "parameter_record_sha256": k1._PCA_PARAMETER_RECORD_SHA256,
        "projection_parameter_sha256": copy.deepcopy(k1._PROJECTION_HASHES),
        "sampling_manifest_sha256": (
            None
            if k1._PCA_PARAMETER_RECORD is None
            else k1._PCA_PARAMETER_RECORD.get("sampling_manifest_sha256")
        ),
    }
    results["hashes"] = {
        **dict(results.get("hashes", {})),
        **_resource_hashes(),
    }
    results["test_local_evaluated"] = False
    common.write_json_atomic(metrics_path, results)


def _k3_print(*values: object, **kwargs: object) -> None:
    adjusted = tuple(
        value.replace("K0", "K3") if isinstance(value, str) else value
        for value in values
    )
    builtins.print(*adjusted, **kwargs)


def _install_k3_hooks() -> None:
    k0.__file__ = str(Path(__file__).resolve())
    k0.EXPERIMENT = EXPERIMENT
    k0.ARTIFACT_TYPE = ARTIFACT_TYPE
    k0.ARTIFACT_FORMAT_VERSION = ARTIFACT_FORMAT_VERSION
    k0.k0_paths = k3_paths
    k0.ensure_shared_initialization = ensure_k3_resources
    k0.build_config = build_config
    k0.build_best_checkpoint = build_best_checkpoint
    k0.train_one_epoch_k0 = train_one_epoch_k3
    k0._smoke_test_k0 = smoke_test_k3
    k0.print = _k3_print
    base.build_model = build_k3_model
    base.audit_model_shapes = audit_k3_shapes
    common.torch_save_atomic = _patched_torch_save_atomic
    common.evaluate = _patched_evaluate


def _remove_k3_hooks() -> None:
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


def _release_resources() -> None:
    k1._TEACHER = None
    k1._PROJECTION = None
    k1._PCA_PARAMETER_RECORD = None
    k1._PROJECTION_HASHES = {}
    k1._TEACHER_CHECKPOINT_SHA256 = None
    k1._PCA_PARAMETER_RECORD_SHA256 = None


def run_training(args: argparse.Namespace) -> None:
    global _ACTIVE_ARGS
    _ACTIVE_ARGS = args
    _install_k3_hooks()
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
        _release_resources()
        _ACTIVE_ARGS = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _remove_k3_hooks()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
