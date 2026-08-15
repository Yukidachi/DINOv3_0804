"""K0 server training entry point.

K0 is the CE-only member of the K0--K3 controlled 2x2 experiment.  It uses
the same scratch MobileNetV2 + R-ASPP model, Cityscapes split, optimizer,
augmentation, fixed optimizer-step budget, and DDP lifecycle as the later K
experiments.  The only enabled training loss is hard-label pixel cross
entropy; no teacher or PCA artifact is loaded.

The entry point is intentionally separate from ``dino_s2_0_server.py``.  K0
must produce the K-group artifact schema and the shared per-seed student
initialization which K1--K3 can consume later.

Typical server command (global batch 8 on two GPUs)::

    torchrun --standalone --nproc_per_node=2 dino_k0_server.py \
        --seed 42 --batch-size 2 --global-batch-size 8 \
        --num-workers 8 --multiprocessing-context spawn \
        --no-pin-memory --persistent-workers

The ``--batch-size`` value is per GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import dino as common
import dino_s2_0 as base
import dino_s2_0_server as server_base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "K_MobileNetV2_RASPP_server"
EXPECTED_COMBINED_MANIFEST_SHA256 = (
    "033161572be28a6de295e0c5dfb62d83cd4d0a18b6039321347c58ab28b9d3c2"
)

EXPERIMENT = "K0"
MODEL_NAME = base.MODEL_NAME
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_k0"
ARTIFACT_FORMAT_VERSION = 1
FORMAL_SEEDS = (42, 3407, 260805)
SHARED_INIT_ARTIFACT_TYPE = "k_shared_mobilenetv2_raspp_student_init"
SHARED_INIT_FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K0 CE-only MobileNetV2+R-ASPP training for a Linux server."
    )
    parser.add_argument("--dataset-root", type=Path, default=common.DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-GPU batch size.")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=8,
        help="Global batch size. Must be divisible by batch-size * world-size.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--multiprocessing-context",
        choices=("auto", "fork", "spawn", "forkserver"),
        default="spawn",
    )
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=False)
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


def setup_distributed(args: argparse.Namespace) -> Tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if args.device == "cpu":
        if world_size > 1:
            raise RuntimeError("CPU DDP is not supported by this server entry point")
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu for local diagnostics")
        # Bind the rank before initializing NCCL and before creating any model.
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> base.MobileNetV2RASPPStudent:
    return model.module if isinstance(model, DDP) else model  # type: ignore[return-value]


def _gradient_l2_named(
    model: torch.nn.Module,
    name_prefix: Optional[str] = None,
) -> float:
    module = unwrap_model(model)
    values = []
    for name, parameter in module.named_parameters():
        if name_prefix is not None and not name.startswith(name_prefix):
            continue
        if parameter.grad is not None:
            values.append(parameter.grad.detach().float().norm(2))
    if not values:
        return 0.0
    return float(torch.stack(values).norm(2).item())


def _model_spec(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "model_name": MODEL_NAME,
        "initialization": "weights=None",
        "head_type": "R-ASPP",
        "head_channels": args.head_channels,
        "dropout": args.dropout,
        "num_classes": common.NUM_CLASSES,
        "ignore_index": common.IGNORE_INDEX,
        "output_stride": common.OUTPUT_STRIDE,
        "feature_taps": copy.deepcopy(base.FEATURE_TAPS),
    }


def _shared_init_path(output_dir: Path, seed: int) -> Path:
    return output_dir.resolve() / "shared_init" / f"seed_{seed}" / "student_init.pth"


def _validate_shared_init_payload(
    payload: Mapping[str, object],
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, torch.Tensor]:
    if payload.get("artifact_type") != SHARED_INIT_ARTIFACT_TYPE:
        raise RuntimeError("Shared K-group initialization has an incompatible artifact type")
    if payload.get("format_version") != SHARED_INIT_FORMAT_VERSION:
        raise RuntimeError("Unsupported K-group shared initialization format")
    if int(payload.get("seed", -1)) != seed:
        raise RuntimeError("Shared initialization seed does not match this run")
    if payload.get("model_spec") != _model_spec(args):
        raise RuntimeError(
            "Shared initialization model spec differs; use the original head/dropout settings"
        )
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Shared initialization does not contain a model state dict")
    state = dict(state)
    state_hash = common.state_dict_sha256(state)
    if state_hash != payload.get("state_dict_sha256"):
        raise RuntimeError("Shared initialization state SHA-256 verification failed")
    return state  # type: ignore[return-value]


def ensure_shared_initialization(
    model: base.MobileNetV2RASPPStudent,
    args: argparse.Namespace,
    output_dir: Path,
    seed: int,
    rank: int,
    world_size: int,
) -> Tuple[str, str, Path]:
    """Create once, then reload, the common scratch state for a K seed."""

    path = _shared_init_path(output_dir, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0 and not path.exists():
        state = common.cpu_state_dict(model)
        payload = {
            "format_version": SHARED_INIT_FORMAT_VERSION,
            "artifact_type": SHARED_INIT_ARTIFACT_TYPE,
            "experiment": "K0-K3 shared initialization",
            "seed": seed,
            "model_spec": _model_spec(args),
            "model_state_dict": state,
            "state_dict_sha256": common.state_dict_sha256(state),
            "created_by": str(Path(__file__).resolve()),
        }
        common.write_checkpoint_with_sidecar(payload, path)
    barrier(world_size)
    if not path.is_file():
        raise FileNotFoundError(f"Shared K-group initialization was not created: {path}")
    file_hash = common.verify_checkpoint_sidecar(path)
    payload = common.safe_torch_load(path, map_location="cpu", weights_only=True)
    state = _validate_shared_init_payload(payload, args, seed)
    model.load_state_dict(state, strict=True)
    state_hash = common.state_dict_sha256(model.state_dict())
    if state_hash != payload["state_dict_sha256"]:
        raise RuntimeError("Reloaded shared initialization does not match its recorded hash")
    return str(state_hash), file_hash, path


def k0_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    run_dir = output_dir.resolve() / "K0" / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "best": run_dir / "best_checkpoint.pth",
        "last": run_dir / "last_checkpoint.pth",
        "config": run_dir / "config.json",
        "feature_taps": run_dir / "feature_taps.json",
        "first_batch_audit": run_dir / "first_batch_audit.json",
        "history": run_dir / "training_history.json",
        "gradient_norms": run_dir / "gradient_norms.jsonl",
        "dev_metrics": run_dir / "dev_metrics.json",
        "dev_confusion": run_dir / "dev_confusion_matrix.json",
        "per_image": run_dir / "dev_per_image_confusion.jsonl",
        "efficiency": run_dir / "efficiency.json",
        "software": run_dir / "software.json",
        "metrics": run_dir / "metrics.json",
    }


def build_config(
    args: argparse.Namespace,
    accumulation_steps: int,
    world_size: int,
    device: torch.device,
    shared_init_state_sha256: str,
    shared_init_file_sha256: str,
) -> Dict[str, object]:
    amp_enabled = bool(args.amp and device.type == "cuda")
    return {
        "experiment": EXPERIMENT,
        "experiment_group": "K_MobileNetV2_RASPP_server",
        "server_entry_point": str(Path(__file__).resolve()),
        "seed": args.seed,
        "formal_seeds": list(FORMAL_SEEDS),
        "world_size": world_size,
        "batch_size_per_gpu": args.batch_size,
        "global_batch_size": args.batch_size * accumulation_steps * world_size,
        "accumulation_steps_per_gpu": accumulation_steps,
        "max_optimizer_steps": args.max_steps,
        "eval_batch_size": args.eval_batch_size,
        "num_workers_per_gpu": args.num_workers,
        "multiprocessing_context": args.multiprocessing_context,
        "pin_memory": bool(args.pin_memory),
        "persistent_workers": bool(args.persistent_workers and args.num_workers > 0),
        "prefetch_factor": args.prefetch_factor,
        "optimizer": "SGD",
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "scheduler": "polynomial",
        "poly_power": args.poly_power,
        "min_lr_ratio": args.min_lr_ratio,
        "eval_every_steps": args.eval_every_steps,
        "eval_policy": "first epoch boundary at or after each target step",
        "amp": amp_enabled,
        "deterministic": args.deterministic,
        "crop_size": [args.crop_height, args.crop_width],
        "random_scale": [args.scale_min, args.scale_max],
        "horizontal_flip_probability": 0.5,
        "eval_resolution": [1024, 2048],
        "head_channels": args.head_channels,
        "dropout": args.dropout,
        "num_classes": common.NUM_CLASSES,
        "ignore_index": common.IGNORE_INDEX,
        "output_stride": common.OUTPUT_STRIDE,
        "initialization": "shared scratch state; weights=None",
        "backbone_frozen": False,
        "loss": {
            "hard_label_ce": True,
            "feature_kd": False,
            "logit_kd": False,
            "lambda_feat": None,
            "lambda_logit": None,
            "temperature": None,
            "auxiliary_warmup_steps": None,
        },
        "teacher": {
            "enabled": False,
            "checkpoint": None,
            "checkpoint_sha256": None,
        },
        "pca": {
            "enabled": False,
            "directory": None,
            "parameter_sha256": None,
        },
        "shared_init_state_sha256": shared_init_state_sha256,
        "shared_init_file_sha256": shared_init_file_sha256,
        "test_local_evaluated": False,
        "benchmark": bool(args.benchmark),
        "benchmark_resolution": [args.benchmark_height, args.benchmark_width],
        "distributed_backend": "nccl" if world_size > 1 else None,
    }


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
    state = common.cpu_state_dict(model)
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "experiment": EXPERIMENT,
        "experiment_group": "K_MobileNetV2_RASPP_server",
        "model_name": MODEL_NAME,
        "initialization": "shared scratch state; weights=None",
        "num_classes": common.NUM_CLASSES,
        "class_names": list(common.CITYSCAPES_CLASSES),
        "output_stride": common.OUTPUT_STRIDE,
        "head_type": "R-ASPP",
        "feature_taps": copy.deepcopy(base.FEATURE_TAPS),
        "model_state_dict": state,
        "model_state_sha256": common.state_dict_sha256(state),
        "best_epoch": epoch,
        "best_optimizer_step": optimizer_step,
        "best_dev_metrics": copy.deepcopy(dev_metrics),
        "config": copy.deepcopy(config),
        "hashes": copy.deepcopy(hashes),
        "dataset_lock": copy.deepcopy(dataset_lock),
        "shape_audit": copy.deepcopy(shape_audit),
        "loss_schema": {
            "hard_label_ce": True,
            "feature_kd": False,
            "logit_kd": False,
        },
        "test_local_evaluated": False,
    }


def load_k0_model(
    checkpoint_path: Path,
    config: Mapping[str, object],
    device: torch.device,
) -> Tuple[base.MobileNetV2RASPPStudent, Dict[str, object]]:
    checkpoint_hash = common.verify_checkpoint_sidecar(checkpoint_path)
    payload = common.safe_torch_load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise RuntimeError(f"Not a K0 checkpoint: {payload.get('artifact_type')!r}")
    if payload.get("feature_taps") != base.FEATURE_TAPS:
        raise RuntimeError("K0 checkpoint feature taps differ from the locked contract")
    if payload.get("config") != config:
        raise RuntimeError("K0 checkpoint configuration differs from the current run")
    model = base.build_model(int(config["head_channels"]), float(config["dropout"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    actual_hash = common.state_dict_sha256(model.state_dict())
    if actual_hash != payload.get("model_state_sha256"):
        raise RuntimeError("K0 checkpoint model-state SHA-256 verification failed")
    model = model.to(device).eval()
    payload["checkpoint_sha256"] = checkpoint_hash
    return model, payload


def _make_gradient_record(
    model: torch.nn.Module,
    optimizer_step: int,
    learning_rate: float,
) -> Dict[str, object]:
    total = _gradient_l2_named(model)
    ce_os16 = _gradient_l2_named(model, "backbone.17")
    return {
        "optimizer_step": optimizer_step,
        "learning_rate": learning_rate,
        "grad_l2_ce": total,
        "grad_l2_feature": None,
        "grad_l2_logit": None,
        "grad_l2_total_student": total,
        "grad_l2_seg_os16": ce_os16,
        "grad_l2_feat_os16": None,
        "grad_l2_logit_os16": None,
        "feature_kd_enabled": False,
        "logit_kd_enabled": False,
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    metadata = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
    ).encode("utf-8")
    return common.sha256_bytes(metadata + b"\0" + tensor.numpy().tobytes())


def _capture_rank_rng_state(device: torch.device) -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_local": (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        ),
    }


def _restore_rank_rng_state(state: Mapping[str, object], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda_local")
    if cuda_state is not None and device.type == "cuda":
        torch.cuda.set_rng_state(cuda_state, device)


def train_one_epoch_k0(
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
    if sampler is not None:
        sampler.set_epoch(epoch)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    confusion = torch.zeros(common.NUM_CLASSES, common.NUM_CLASSES, dtype=torch.int64)
    loss_sum = 0.0
    valid_pixels = 0
    optimizer_steps = 0
    gradient_records: List[Dict[str, object]] = []
    first_batch_audit: Optional[Dict[str, object]] = None

    possible_steps = math.ceil(len(loader) / accumulation_steps)
    target_steps = min(possible_steps, remaining_optimizer_steps)
    max_batches = min(len(loader), target_steps * accumulation_steps)
    progress = tqdm(loader, desc=f"Epoch {epoch} [K0 CE]", disable=rank != 0)

    for batch_index, (images, targets, paths) in enumerate(progress):
        if batch_index >= max_batches:
            break
        if starting_optimizer_step == 0 and batch_index == 0:
            first_batch_audit = {
                "rank": rank,
                "epoch": epoch,
                "micro_batch_index": 0,
                "paths": list(paths),
                "image_tensor_shape": list(images.shape),
                "target_tensor_shape": list(targets.shape),
                "image_tensor_sha256": _tensor_sha256(images),
                "target_tensor_sha256": _tensor_sha256(targets),
                "valid_pixels": int((targets != common.IGNORE_INDEX).sum().item()),
            }
        group_position = batch_index % accumulation_steps
        if group_position == 0:
            group_size = min(accumulation_steps, max_batches - batch_index)
        sync_gradients = group_position + 1 == group_size
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        sync_context = contextlib.nullcontext()
        if isinstance(model, DDP) and not sync_gradients:
            sync_context = model.no_sync()
        with sync_context:
            with common.autocast_context(device, amp_enabled):
                logits = model(images)
            logits_float = logits.float()
            batch_loss_sum = F.cross_entropy(
                logits_float,
                targets,
                ignore_index=common.IGNORE_INDEX,
                reduction="sum",
            )
            batch_valid = int((targets != common.IGNORE_INDEX).sum().item())
            if batch_valid == 0:
                raise RuntimeError("Training batch contains no valid Cityscapes pixels")
            batch_loss = batch_loss_sum / batch_valid
            if not torch.isfinite(batch_loss):
                raise RuntimeError("K0 produced a non-finite CE loss")
            scaler.scale(batch_loss / group_size).backward()

        if sync_gradients:
            scaler.unscale_(optimizer)
            optimizer_steps += 1
            global_step = starting_optimizer_step + optimizer_steps
            if global_step == 1 or global_step % 500 == 0:
                gradient_records.append(
                    _make_gradient_record(
                        model,
                        global_step,
                        float(optimizer.param_groups[0]["lr"]),
                    )
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        predictions = logits_float.detach().argmax(dim=1)
        confusion += common.confusion_counts(predictions, targets)
        loss_sum += float(batch_loss_sum.detach().item())
        valid_pixels += batch_valid
        if rank == 0:
            running = common.metrics_from_confusion(confusion, loss_sum, valid_pixels)
            progress.set_postfix(
                {
                    "loss": f"{running['loss']:.4f}",
                    "mIoU": f"{running['mIoU']:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "steps": optimizer_steps,
                }
            )

    if optimizer_steps != target_steps:
        raise RuntimeError(
            f"K0 optimizer-step accounting failed: actual={optimizer_steps}, expected={target_steps}"
        )
    metrics = server_base._reduce_train_metrics(
        confusion, loss_sum, valid_pixels, device, world_size
    )
    metrics["loss_schema"] = "hard_label_pixel_cross_entropy_only"
    metrics["feature_loss"] = None
    metrics["logit_loss"] = None
    metrics["warmup_weight"] = None
    return metrics, optimizer_steps, gradient_records, first_batch_audit


def _smoke_test_k0(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    rank: int,
) -> None:
    model.train()
    images, targets, paths = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    model.zero_grad(set_to_none=True)
    with common.autocast_context(device, amp_enabled):
        logits = model(images)
    logits_float = logits.float()
    loss = F.cross_entropy(
        logits_float,
        targets,
        ignore_index=common.IGNORE_INDEX,
        reduction="mean",
    )
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite K0 smoke-test loss: {loss.item()}")
    backbone_gradients = sum(
        parameter.grad is not None for parameter in unwrap_model(model).backbone.parameters()
    )
    head_gradients = sum(
        parameter.grad is not None for parameter in unwrap_model(model).head.parameters()
    )
    if backbone_gradients == 0 or head_gradients == 0:
        raise RuntimeError("K0 smoke test did not produce end-to-end gradients")
    if rank == 0:
        print(
            f"[OK] K0 server DDP smoke test: sample={paths[0]}, "
            f"logits={tuple(logits.shape)}, loss={loss.item():.6f}, "
            f"backbone_grad_tensors={backbone_gradients}, head_grad_tensors={head_gradients}"
        )


def _write_gradient_records(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def _software_info() -> Dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(base.torchvision.__version__),
        "numpy": np.__version__,
        "pillow": __import__("PIL").__version__,
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def run_training(args: argparse.Namespace) -> None:
    rank, local_rank, world_size, device = setup_distributed(args)
    main_process = rank == 0
    train_loader: Optional[DataLoader] = None
    dev_loader: Optional[DataLoader] = None
    model: Optional[torch.nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    scaler: Optional[torch.amp.GradScaler] = None
    selected_model: Optional[torch.nn.Module] = None
    successful_exit = False

    try:
        accumulation_steps = server_base.effective_accumulation_steps(args, world_size)
        actual_global_batch = args.batch_size * accumulation_steps * world_size
        if actual_global_batch != args.global_batch_size:
            raise RuntimeError(
                f"K0 global batch accounting failed: {actual_global_batch} != "
                f"requested {args.global_batch_size}"
            )

        # All ranks use the same seed for model construction.  DistributedSampler
        # and each rank's DataLoader generator still receive rank-aware state in
        # the shared server loader helper.
        common.set_global_seed(args.seed, args.deterministic)
        dataset_root = args.dataset_root.resolve()
        dataset_lock, entries_by_split = common.validate_dataset_lock(dataset_root)
        if dataset_lock["combined_manifest_sha256"] != EXPECTED_COMBINED_MANIFEST_SHA256:
            raise RuntimeError(
                "Cityscapes split lock differs from the K-group frozen protocol: "
                f"actual={dataset_lock['combined_manifest_sha256']}"
            )

        model = base.build_model(args.head_channels, args.dropout).to(device)
        initial_model_hash, initial_file_hash, init_path = ensure_shared_initialization(
            model,
            args,
            args.output_dir,
            args.seed,
            rank,
            world_size,
        )
        amp_enabled = bool(args.amp and device.type == "cuda")
        shape_audit = base.audit_model_shapes(
            model,
            device,
            args.crop_height,
            args.crop_width,
            amp_enabled,
        )
        parameters = base._parameter_report(model)

        train_loader, train_sampler, train_generator = server_base.build_train_loader(
            args, dataset_root, entries_by_split, device, rank, world_size
        )
        dev_loader = (
            server_base.build_dev_loader(args, dataset_root, entries_by_split, device)
            if main_process
            else None
        )

        if world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=True,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

        if main_process:
            print(
                f"[INFO] K0 server DDP: world_size={world_size}, device={device}, "
                f"AMP={amp_enabled}, workers/rank={args.num_workers}, "
                f"context={args.multiprocessing_context}, pin_memory={args.pin_memory}"
            )
            print(
                f"[OK] params={parameters['trainable_parameters']:,}; "
                f"local batch={args.batch_size}; global batch={actual_global_batch}; "
                f"shared init state sha256={initial_model_hash}"
            )

        if args.smoke_test:
            _smoke_test_k0(model, train_loader, device, amp_enabled, rank)
            successful_exit = True
            return

        paths = k0_paths(args.output_dir, args.seed)
        paths["run_dir"].mkdir(parents=True, exist_ok=True)
        barrier(world_size)
        artifact_paths = [
            paths["best"],
            paths["last"],
            paths["config"],
            paths["history"],
            paths["metrics"],
        ]
        if not args.resume and any(path.exists() for path in artifact_paths):
            raise FileExistsError(
                f"K0 run artifacts already exist in {paths['run_dir']}; "
                "use --resume or another --output-dir"
            )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        def lr_factor(step: int) -> float:
            progress = min(step, args.max_steps) / max(args.max_steps, 1)
            return max((1.0 - progress) ** args.poly_power, args.min_lr_ratio)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        config = build_config(
            args,
            accumulation_steps,
            world_size,
            device,
            initial_model_hash,
            initial_file_hash,
        )
        hashes: Dict[str, object] = {
            "student_init_state_sha256": initial_model_hash,
            "student_init_file_sha256": initial_file_hash,
            "student_init_path": str(init_path),
            "training_script_sha256": common.sha256_file(Path(__file__).resolve()),
            "shared_student_module_sha256": common.sha256_file(Path(base.__file__).resolve()),
            "common_module_sha256": common.sha256_file(Path(common.__file__).resolve()),
            "dataset_combined_manifest_sha256": dataset_lock["combined_manifest_sha256"],
            "teacher_checkpoint_sha256": None,
            "pca_parameter_sha256": None,
        }

        if main_process:
            common.write_json_atomic(paths["config"], config)
            common.write_json_atomic(paths["feature_taps"], shape_audit)
            common.write_json_atomic(paths["software"], _software_info())
            common.write_json_atomic(
                paths["efficiency"],
                {
                    "enabled": bool(args.benchmark),
                    "training_time_teacher_or_projection_included": False,
                    "result": None,
                },
            )
        barrier(world_size)

        history: List[Dict[str, object]] = []
        best_key: Optional[Tuple[float, float, float, float]] = None
        best_epoch: Optional[int] = None
        best_optimizer_step: Optional[int] = None
        best_dev_metrics: Optional[Dict[str, object]] = None
        epoch = 0
        cumulative_optimizer_steps = 0
        next_eval_target = args.eval_every_steps

        if args.resume:
            if not paths["last"].is_file():
                raise FileNotFoundError(f"K0 resume checkpoint not found: {paths['last']}")
            resume_payload = common.safe_torch_load(
                paths["last"], map_location="cpu", weights_only=False
            )
            if resume_payload.get("config") != config:
                raise RuntimeError("Resume configuration differs from current K0 arguments")
            resume_hashes = resume_payload.get("hashes", {})
            if resume_hashes.get("student_init_state_sha256") != initial_model_hash:
                raise RuntimeError("Resume checkpoint uses a different shared student init")
            model_to_load = unwrap_model(model)
            model_to_load.load_state_dict(resume_payload["model_state_dict"], strict=True)
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
            scaler.load_state_dict(resume_payload["scaler_state_dict"])
            history = list(resume_payload["history"])
            best_key = resume_payload["best_key"]
            best_epoch = resume_payload["best_epoch"]
            best_optimizer_step = resume_payload["best_optimizer_step"]
            best_dev_metrics = resume_payload["best_dev_metrics"]
            epoch = int(resume_payload["epoch"])
            cumulative_optimizer_steps = int(resume_payload["optimizer_steps"])
            generator_states = resume_payload.get("train_generator_states_by_rank")
            if generator_states is not None:
                if len(generator_states) != world_size:
                    raise RuntimeError(
                        "Resume DataLoader generator-state count differs from world size"
                    )
                train_generator.set_state(generator_states[rank])
            rng_states = resume_payload.get("rng_states_by_rank")
            if rng_states is not None:
                if len(rng_states) != world_size:
                    raise RuntimeError("Resume RNG-state count differs from world size")
                _restore_rank_rng_state(rng_states[rank], device)
            next_eval_target = int(
                resume_payload.get(
                    "next_eval_target",
                    ((cumulative_optimizer_steps // args.eval_every_steps) + 1)
                    * args.eval_every_steps,
                )
            )
            if main_process:
                print(
                    f"[OK] Resuming K0 after epoch {epoch}, "
                    f"step {cumulative_optimizer_steps:,}"
                )

        steps_per_full_epoch = math.ceil(len(train_loader) / accumulation_steps)
        estimated_epochs = math.ceil(args.max_steps / steps_per_full_epoch)
        if main_process:
            print(
                f"[INFO] steps/full epoch={steps_per_full_epoch}; "
                f"about {estimated_epochs} epochs; fixed budget={args.max_steps:,}"
            )

        training_started = time.time()
        while cumulative_optimizer_steps < args.max_steps:
            epoch += 1
            remaining_steps = args.max_steps - cumulative_optimizer_steps
            train_metrics, optimizer_steps, gradient_records, first_batch_audit = (
                train_one_epoch_k0(
                model=model,
                loader=train_loader,
                sampler=train_sampler,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                amp_enabled=amp_enabled,
                accumulation_steps=accumulation_steps,
                epoch=epoch,
                starting_optimizer_step=cumulative_optimizer_steps,
                remaining_optimizer_steps=remaining_steps,
                rank=rank,
                world_size=world_size,
                )
            )
            cumulative_optimizer_steps += optimizer_steps
            if main_process:
                _write_gradient_records(paths["gradient_norms"], gradient_records)
            if first_batch_audit is not None:
                first_batch_rows: List[Optional[Dict[str, object]]] = [
                    None for _ in range(world_size)
                ]
                if world_size > 1:
                    dist.all_gather_object(first_batch_rows, first_batch_audit)
                else:
                    first_batch_rows[0] = first_batch_audit
                if main_process:
                    common.write_json_atomic(
                        paths["first_batch_audit"],
                        {
                            "experiment": EXPERIMENT,
                            "seed": args.seed,
                            "student_init_state_sha256": initial_model_hash,
                            "world_size": world_size,
                            "per_rank": first_batch_rows,
                        },
                    )

            should_evaluate = (
                cumulative_optimizer_steps >= next_eval_target
                or cumulative_optimizer_steps == args.max_steps
            )
            dev_metrics: Optional[Dict[str, object]] = None
            eval_target: Optional[int] = None
            if should_evaluate:
                eval_target = next_eval_target
                barrier(world_size)
                if main_process:
                    assert dev_loader is not None
                    dev_metrics, _ = common.evaluate(
                        model=unwrap_model(model),
                        loader=dev_loader,
                        device=device,
                        amp_enabled=amp_enabled,
                        split_name="dev_local K0",
                        boundary_tolerance=args.boundary_tolerance,
                        collect_per_image=False,
                    )
                    candidate_key = (
                        float(dev_metrics["mIoU"]),
                        float(dev_metrics["mAcc"]),
                        float(dev_metrics["pixel_accuracy"]),
                        -float(dev_metrics["loss"]),
                    )
                    if best_key is None or candidate_key > best_key:
                        best_key = candidate_key
                        best_epoch = epoch
                        best_optimizer_step = cumulative_optimizer_steps
                        best_dev_metrics = copy.deepcopy(dev_metrics)
                        best_payload = build_best_checkpoint(
                            model=unwrap_model(model),
                            epoch=epoch,
                            optimizer_step=cumulative_optimizer_steps,
                            dev_metrics=dev_metrics,
                            config=config,
                            hashes=hashes,
                            dataset_lock=dataset_lock,
                            shape_audit=shape_audit,
                        )
                        checkpoint_hash = common.write_checkpoint_with_sidecar(
                            best_payload, paths["best"]
                        )
                        print(
                            f"[OK] K0 best updated: step={cumulative_optimizer_steps:,}, "
                            f"dev_mIoU={dev_metrics['mIoU']:.6f}, sha256={checkpoint_hash}"
                        )
                barrier(world_size)
                while next_eval_target <= cumulative_optimizer_steps:
                    next_eval_target += args.eval_every_steps

            if main_process:
                generator_states: List[Optional[torch.Tensor]] = [
                    None for _ in range(world_size)
                ]
                rng_states: List[Optional[Dict[str, object]]] = [
                    None for _ in range(world_size)
                ]
            else:
                generator_states = [None for _ in range(world_size)]
                rng_states = [None for _ in range(world_size)]
            if world_size > 1:
                dist.all_gather_object(generator_states, train_generator.get_state())
                dist.all_gather_object(rng_states, _capture_rank_rng_state(device))
            else:
                generator_states[0] = train_generator.get_state()
                rng_states[0] = _capture_rank_rng_state(device)

            if main_process:
                epoch_record: Dict[str, object] = {
                    "epoch": epoch,
                    "optimizer_steps": cumulative_optimizer_steps,
                    "optimizer_steps_this_epoch": optimizer_steps,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train": train_metrics,
                    "dev": dev_metrics,
                    "eval_target_step": eval_target,
                }
                history.append(epoch_record)
                common.write_json_atomic(paths["history"], history)
                last_payload = {
                    "format_version": ARTIFACT_FORMAT_VERSION,
                    "artifact_type": ARTIFACT_TYPE,
                    "epoch": epoch,
                    "optimizer_steps": cumulative_optimizer_steps,
                    "next_eval_target": next_eval_target,
                    "model_state_dict": common.cpu_state_dict(unwrap_model(model)),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "train_generator_states_by_rank": generator_states,
                    "rng_states_by_rank": rng_states,
                    "history": history,
                    "best_key": best_key,
                    "best_epoch": best_epoch,
                    "best_optimizer_step": best_optimizer_step,
                    "best_dev_metrics": best_dev_metrics,
                    "config": config,
                    "hashes": hashes,
                    "dataset_lock": dataset_lock,
                    "test_local_evaluated": False,
                }
                common.torch_save_atomic(last_payload, paths["last"])
                message = (
                    f"Epoch {epoch}: step={cumulative_optimizer_steps:,}/{args.max_steps:,}, "
                    f"train_mIoU={train_metrics['mIoU']:.4f}, "
                    f"train_loss={train_metrics['loss']:.4f}"
                )
                if dev_metrics is not None:
                    message += f", dev_mIoU={dev_metrics['mIoU']:.4f}"
                print(message)
            barrier(world_size)

        barrier(world_size)
        if main_process:
            if best_epoch is None or best_optimizer_step is None or best_dev_metrics is None:
                raise RuntimeError("K0 run ended without a selected dev checkpoint")
            selected_model, selected_payload = load_k0_model(
                paths["best"], config, device
            )
            assert dev_loader is not None
            selected_dev_metrics, per_image_rows = common.evaluate(
                model=selected_model,
                loader=dev_loader,
                device=device,
                amp_enabled=amp_enabled,
                split_name="selected dev_local K0",
                boundary_tolerance=args.boundary_tolerance,
                collect_per_image=True,
            )
            if not common.metrics_reproduce(selected_dev_metrics, best_dev_metrics):
                raise RuntimeError(
                    "Reloaded K0 best checkpoint did not reproduce its saved dev metrics"
                )
            common.write_json_atomic(paths["dev_metrics"], selected_dev_metrics)
            common.write_json_atomic(
                paths["dev_confusion"], selected_dev_metrics["confusion_matrix"]
            )
            common.write_jsonl_atomic(paths["per_image"], per_image_rows)

            efficiency_result = None
            if args.benchmark:
                efficiency_result = base.benchmark_model(
                    selected_model,
                    device,
                    args.benchmark_height,
                    args.benchmark_width,
                    args.benchmark_warmup,
                    args.benchmark_runs,
                )
            common.write_json_atomic(
                paths["efficiency"],
                {
                    "enabled": bool(args.benchmark),
                    "training_time_teacher_or_projection_included": False,
                    "model": "MobileNetV2+R-ASPP student only",
                    "result": efficiency_result,
                },
            )
            results = {
                "experiment": EXPERIMENT,
                "protocol": (
                    "K0 CE-only controlled rerun: scratch MobileNetV2+R-ASPP, "
                    "fixed 80k optimizer-step budget, dev_local checkpoint selection, "
                    "and no teacher/PCA/test_local evaluation."
                ),
                "best_epoch": best_epoch,
                "best_optimizer_step": best_optimizer_step,
                "best_dev_metrics": selected_dev_metrics,
                "class_names": common.CITYSCAPES_CLASSES,
                "config": config,
                "shape_audit": shape_audit,
                "dataset_lock": dataset_lock,
                "model": {
                    "model_name": MODEL_NAME,
                    "initialization": "shared scratch state; weights=None",
                    "head": "R-ASPP",
                    "feature_taps": base.FEATURE_TAPS,
                    **parameters,
                },
                "loss": {
                    "hard_label_ce": True,
                    "feature_kd": False,
                    "logit_kd": False,
                    "lambda_feat": None,
                    "lambda_logit": None,
                    "temperature": None,
                    "warmup": None,
                },
                "efficiency": efficiency_result,
                "hashes": {
                    **hashes,
                    "selected_model_state_sha256": selected_payload["model_state_sha256"],
                    "checkpoint_sha256": selected_payload["checkpoint_sha256"],
                },
                "training": {
                    "elapsed_seconds": time.time() - training_started,
                    "optimizer_steps": cumulative_optimizer_steps,
                    "epochs_completed": epoch,
                    "steps_per_full_epoch": steps_per_full_epoch,
                },
                "software": _software_info(),
                "artifacts": {key: str(value) for key, value in paths.items()},
                "test_local_evaluated": False,
            }
            common.write_json_atomic(paths["metrics"], results)
            print(
                f"[DONE] K0 server DDP: GPUs={world_size}, "
                f"steps={cumulative_optimizer_steps:,}, "
                f"best dev mIoU={selected_dev_metrics['mIoU']:.6f}"
            )
        barrier(world_size)
        successful_exit = True
    finally:
        server_base._shutdown_loader(train_loader)
        server_base._shutdown_loader(dev_loader)
        server_base._synchronize_cuda(device)
        if successful_exit and world_size > 1 and dist.is_initialized():
            barrier(world_size)

        selected_model = None
        scaler = None
        scheduler = None
        optimizer = None
        model = None
        server_base._synchronize_cuda(device)
        if successful_exit and world_size > 1 and dist.is_initialized():
            barrier(world_size)
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
