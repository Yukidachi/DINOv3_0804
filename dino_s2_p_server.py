"""S2-P server training entry point.

S2-P changes only the student initialization relative to S2-0: the
MobileNetV2 backbone is first pretrained on a local ImageNet-1K dataset, then
the complete MobileNetV2+R-ASPP model is trained end to end on Cityscapes.
This entry point also exposes an explicit ImageNette-10 diagnostic route; it
must not be reported as strict ImageNet-1K S2-P.

It is designed for the two-GPU Linux server diagnosed in
``profile_dino_s2_0.py``:

* launch with ``torchrun --nproc_per_node=2``;
* use DDP and ``no_sync`` during gradient accumulation;
* use DataLoader ``spawn`` workers;
* default to ``pin_memory=False`` because the server's pinned-memory path
  serialized the workers and reduced throughput by several times;
* write S2-P server-suffixed artifacts in separate output directories.

ImageNet pretraining stage:

    torchrun --standalone --nproc_per_node=2 dino_s2_p_server.py \
        --stage imagenet-pretrain --imagenet-root /data/imagenet \
        --imagenet-batch-size 128 --imagenet-global-batch-size 256 \
        --num-workers 8 --no-pin-memory --persistent-workers

Cityscapes S2-P stage:

    torchrun --standalone --nproc_per_node=2 dino_s2_p_server.py \
        --stage cityscapes --imagenet-checkpoint \
        result/ImageNet_MobileNetV2_server/seed_42/imagenet_mobilenetv2_best.pth \
        --seed 42 --batch-size 2 --global-batch-size 8 \
        --num-workers 8 --no-pin-memory --persistent-workers

The ``--batch-size`` value is per GPU.  Without ``--global-batch-size``, the
effective global batch is ``batch_size * accumulation_steps * world_size``.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torchvision import datasets as torchvision_datasets
from torchvision import transforms as torchvision_transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import dino as common
import dino_s2_0 as base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "S2_P_MobileNetV2_RASPP_server"
DEFAULT_IMAGENET_OUTPUT_DIR = SCRIPT_DIR / "result" / "ImageNet_MobileNetV2_server"
DEFAULT_S2P_IMAGENETTE_OUTPUT_DIR = SCRIPT_DIR / "result" / "S2_P_ImageNette_MobileNetV2_RASPP_server"
S2P_EXPERIMENT = "S2-P"
S2P_ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_s2_p_server"
S2P_ARTIFACT_FORMAT_VERSION = 1
IMAGENET_EXPERIMENT = "ImageNet-MobileNetV2-pretrain"
IMAGENET_ARTIFACT_TYPE = "mobilenetv2_imagenet1k_pretrain_server"
IMAGENET_ARTIFACT_FORMAT_VERSION = 1
IMAGENETTE_EXPERIMENT = "ImageNette-10-MobileNetV2-pretrain"
IMAGENETTE_ARTIFACT_TYPE = "mobilenetv2_imagenette10_pretrain_server"
S2P_IMAGENETTE_EXPERIMENT = "S2-P-ImageNette"
S2P_IMAGENETTE_ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_s2_p_imagenette_server"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S2-P MobileNetV2+R-ASPP server pipeline: local ImageNet-1K or "
            "ImageNette-10 pretraining followed by Cityscapes fine-tuning."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("imagenet-pretrain", "imagenette-pretrain", "cityscapes"),
        default="cityscapes",
        help=(
            "Run ImageNet-1K pretraining, ImageNette-10 pretraining, "
            "or the Cityscapes transfer stage."
        ),
    )
    parser.add_argument(
        "--imagenet-root",
        type=Path,
        default=None,
        help=(
            "ImageNet/ImageNette root containing train/ and optionally val/ "
            "class directories."
        ),
    )
    parser.add_argument(
        "--imagenet-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint produced by an ImageNet or ImageNette pretraining stage.",
    )
    parser.add_argument("--imagenet-output-dir", type=Path, default=DEFAULT_IMAGENET_OUTPUT_DIR)
    parser.add_argument(
        "--imagenette-output-dir",
        type=Path,
        default=SCRIPT_DIR / "result" / "ImageNette_MobileNetV2_server",
    )
    parser.add_argument("--imagenet-epochs", type=int, default=90)
    parser.add_argument("--imagenet-batch-size", type=int, default=128, help="Per-GPU ImageNet batch size.")
    parser.add_argument(
        "--imagenet-global-batch-size",
        type=int,
        default=None,
        help="Optional global ImageNet batch size; derives accumulation steps.",
    )
    parser.add_argument("--imagenet-accumulation-steps", type=int, default=1)
    parser.add_argument("--imagenet-eval-batch-size", type=int, default=256)
    parser.add_argument("--imagenet-image-size", type=int, default=224)
    parser.add_argument("--imagenet-lr", type=float, default=0.1)
    parser.add_argument("--imagenet-momentum", type=float, default=0.9)
    parser.add_argument("--imagenet-weight-decay", type=float, default=4e-5)
    parser.add_argument("--imagenet-label-smoothing", type=float, default=0.1)
    parser.add_argument("--dataset-root", type=Path, default=common.DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-GPU batch size.")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Optional global batch size. Derives accumulation steps across all GPUs.",
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
        help="Keep spawn workers alive across epochs to avoid repeated startup cost.",
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
    parser.add_argument("--device", default="auto", help="Use auto or cuda; torchrun assigns one GPU per rank.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    positive = (
        "max_steps",
        "batch_size",
        "eval_batch_size",
        "accumulation_steps",
        "eval_every_steps",
        "head_channels",
        "crop_height",
        "crop_width",
        "benchmark_height",
        "benchmark_width",
        "benchmark_runs",
        "prefetch_factor",
        "imagenet_epochs",
        "imagenet_batch_size",
        "imagenet_accumulation_steps",
        "imagenet_eval_batch_size",
        "imagenet_image_size",
    )
    for field in positive:
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be at least 1")
    if args.global_batch_size is not None and args.global_batch_size < 1:
        parser.error("--global-batch-size must be at least 1")
    if args.imagenet_global_batch_size is not None and args.imagenet_global_batch_size < 1:
        parser.error("--imagenet-global-batch-size must be at least 1")
    if args.stage == "cityscapes" and args.imagenet_checkpoint is None:
        parser.error("--stage cityscapes requires --imagenet-checkpoint")
    if args.stage in ("imagenet-pretrain", "imagenette-pretrain") and args.imagenet_root is None:
        parser.error(f"--stage {args.stage} requires --imagenet-root")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.lr <= 0 or not 0 <= args.momentum < 1 or args.weight_decay < 0:
        parser.error("Invalid optimizer settings")
    if args.imagenet_lr <= 0 or not 0 <= args.imagenet_momentum < 1:
        parser.error("Invalid ImageNet optimizer settings")
    if args.imagenet_weight_decay < 0:
        parser.error("--imagenet-weight-decay cannot be negative")
    if not 0 <= args.imagenet_label_smoothing < 1:
        parser.error("--imagenet-label-smoothing must be in [0, 1)")
    if args.imagenet_image_size < 32:
        parser.error("--imagenet-image-size must be at least 32")
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
    return args


def setup_distributed(args: argparse.Namespace) -> Tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP server training requires CUDA")
        dist.init_process_group(backend="nccl", init_method="env://")
    if args.device == "cpu":
        if world_size > 1:
            raise RuntimeError("CPU DDP is not supported by this server entry point")
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use dino_s2_0.py for CPU diagnostics")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    return rank, local_rank, world_size, device


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _shutdown_loader(loader: Optional[DataLoader]) -> None:
    """Stop a DataLoader's persistent workers before tearing down CUDA/DDP."""

    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception as error:  # pragma: no cover - PyTorch-version dependent
            print(f"[WARN] DataLoader worker cleanup failed: {error}", file=sys.stderr)


def _synchronize_cuda(device: torch.device) -> None:
    """Finish queued CUDA work so NCCL/model destructors run synchronously."""

    if device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
    except Exception as error:  # pragma: no cover - only exercised on a failed CUDA context
        print(f"[WARN] CUDA synchronization during shutdown failed: {error}", file=sys.stderr)


def effective_accumulation_steps(args: argparse.Namespace, world_size: int) -> int:
    if args.global_batch_size is None:
        return args.accumulation_steps
    denominator = args.batch_size * world_size
    if args.global_batch_size % denominator:
        raise ValueError(
            "global batch size must be divisible by batch_size * world_size: "
            f"{args.global_batch_size} % ({args.batch_size} * {world_size}) != 0"
        )
    return args.global_batch_size // denominator


def _loader_kwargs(args: argparse.Namespace, device: torch.device, workers: int) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(args.pin_memory and device.type == "cuda"),
        "persistent_workers": bool(args.persistent_workers and workers > 0),
        "worker_init_fn": common.seed_worker,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        if args.multiprocessing_context != "auto":
            kwargs["multiprocessing_context"] = args.multiprocessing_context
    return kwargs


def build_train_loader(
    args: argparse.Namespace,
    dataset_root: Path,
    entries_by_split: Mapping[str, Sequence[Tuple[str, str]]],
    device: torch.device,
    rank: int,
    world_size: int,
):
    dataset = common.CityscapesManifestDataset(
        dataset_root=dataset_root,
        entries=entries_by_split["train_local"],
        transform=common.CityscapesTrainTransform(
            crop_size=(args.crop_height, args.crop_width),
            scale_range=(args.scale_min, args.scale_max),
        ),
        reject_all_ignore=True,
    )
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
    generator = torch.Generator()
    generator.manual_seed(args.seed + rank)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
        generator=generator,
        **_loader_kwargs(args, device, args.num_workers),
    )
    return loader, sampler, generator


def build_dev_loader(
    args: argparse.Namespace,
    dataset_root: Path,
    entries_by_split: Mapping[str, Sequence[Tuple[str, str]]],
    device: torch.device,
):
    dataset = common.CityscapesManifestDataset(
        dataset_root=dataset_root,
        entries=entries_by_split["dev_local"],
        transform=common.CityscapesEvalTransform(),
        reject_all_ignore=False,
    )
    return DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(args, device, args.num_workers),
    )


def server_paths(output_dir: Path, seed: int, artifact_prefix: str = "s2_p") -> Dict[str, Path]:
    run_dir = output_dir.resolve() / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "best": run_dir / f"{artifact_prefix}_server_mobilenetv2_raspp.pth",
        "last": run_dir / f"{artifact_prefix}_server_last_checkpoint.pth",
        "history": run_dir / f"{artifact_prefix}_server_training_history.json",
        "metrics": run_dir / f"{artifact_prefix}_server_metrics.json",
        "per_image": run_dir / f"{artifact_prefix}_server_dev_per_image_confusion.jsonl",
    }


def imagenet_paths(output_dir: Path, seed: int, artifact_prefix: str = "imagenet") -> Dict[str, Path]:
    run_dir = output_dir.resolve() / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "best": run_dir / f"{artifact_prefix}_mobilenetv2_best.pth",
        "last": run_dir / f"{artifact_prefix}_mobilenetv2_last_checkpoint.pth",
        "history": run_dir / f"{artifact_prefix}_mobilenetv2_training_history.json",
        "metrics": run_dir / f"{artifact_prefix}_mobilenetv2_metrics.json",
    }


def _resolve_imagenet_split(root: Path, split: str) -> Path:
    root = root.resolve()
    candidate = root / split
    if candidate.is_dir():
        return candidate
    if root.is_dir():
        return root
    raise FileNotFoundError(
        f"ImageNet {split} directory not found. Expected {root / split} "
        f"or a class-directory root at {root}."
    )


def _has_class_directories(path: Path) -> bool:
    return any(child.is_dir() for child in path.iterdir())


def _imagenet_train_transform(image_size: int):
    return torchvision_transforms.Compose(
        [
            torchvision_transforms.RandomResizedCrop(
                image_size,
                scale=(0.08, 1.0),
                interpolation=torchvision_transforms.InterpolationMode.BILINEAR,
            ),
            torchvision_transforms.RandomHorizontalFlip(),
            torchvision_transforms.ToTensor(),
            torchvision_transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _imagenet_eval_transform(image_size: int):
    resize_size = int(round(image_size / 0.875))
    return torchvision_transforms.Compose(
        [
            torchvision_transforms.Resize(
                resize_size,
                interpolation=torchvision_transforms.InterpolationMode.BILINEAR,
            ),
            torchvision_transforms.CenterCrop(image_size),
            torchvision_transforms.ToTensor(),
            torchvision_transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def effective_imagenet_accumulation_steps(args: argparse.Namespace, world_size: int) -> int:
    denominator = args.imagenet_batch_size * world_size
    if args.imagenet_global_batch_size is None:
        return args.imagenet_accumulation_steps
    if args.imagenet_global_batch_size % denominator:
        raise ValueError(
            "ImageNet global batch size must be divisible by imagenet-batch-size * world_size: "
            f"{args.imagenet_global_batch_size} % ({args.imagenet_batch_size} * {world_size}) != 0"
        )
    return args.imagenet_global_batch_size // denominator


def build_imagenet_loaders(
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    world_size: int,
    expected_num_classes: int = 1000,
    dataset_label: str = "ImageNet-1K",
):
    assert args.imagenet_root is not None
    train_dir = _resolve_imagenet_split(args.imagenet_root, "train")
    train_dataset = torchvision_datasets.ImageFolder(
        str(train_dir), transform=_imagenet_train_transform(args.imagenet_image_size)
    )
    if len(train_dataset.classes) != expected_num_classes:
        raise RuntimeError(
            f"{dataset_label} pretraining requires exactly {expected_num_classes} class folders; "
            f"found {len(train_dataset.classes)} under {train_dir}"
        )
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
    else:
        train_sampler = None
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed + rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.imagenet_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        generator=train_generator,
        **_loader_kwargs(args, device, args.num_workers),
    )
    if len(train_loader) == 0:
        raise RuntimeError("ImageNet train DataLoader is empty")

    val_loader = None
    val_dir = args.imagenet_root.resolve() / "val"
    if val_dir.is_dir() and rank == 0 and _has_class_directories(val_dir):
        val_dataset = torchvision_datasets.ImageFolder(
            str(val_dir), transform=_imagenet_eval_transform(args.imagenet_image_size)
        )
        if val_dataset.classes != train_dataset.classes:
            raise RuntimeError("ImageNet train/val class folders do not match")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.imagenet_eval_batch_size,
            shuffle=False,
            drop_last=False,
            **_loader_kwargs(args, device, args.num_workers),
        )
    elif val_dir.is_dir() and rank == 0:
        print(
            f"[WARN] ImageNet val directory {val_dir} has no class subdirectories; "
            "val top-1 selection is disabled and train top-1 will select the checkpoint."
        )
    return train_loader, train_sampler, train_generator, val_loader, train_dataset.classes


def build_imagenet_classifier(num_classes: int = 1000) -> torch.nn.Module:
    return torchvision.models.mobilenet_v2(weights=None, num_classes=num_classes)


def _reduce_imagenet_metrics(
    loss_sum: float,
    correct: int,
    samples: int,
    device: torch.device,
    world_size: int,
) -> Dict[str, float]:
    totals = torch.tensor(
        [loss_sum, float(correct), float(samples)], device=device, dtype=torch.float64
    )
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    total_loss, total_correct, total_samples = (float(value.item()) for value in totals)
    return {
        "loss": total_loss / max(total_samples, 1.0),
        "top1": total_correct / max(total_samples, 1.0),
        "samples": int(total_samples),
    }


def train_one_epoch_imagenet(
    model: torch.nn.Module,
    loader: DataLoader,
    sampler: Optional[DistributedSampler],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    accumulation_steps: int,
    label_smoothing: float,
    epoch: int,
    rank: int,
    world_size: int,
) -> Dict[str, float]:
    if sampler is not None:
        sampler.set_epoch(epoch)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    correct = 0
    samples = 0
    first_step_gradient_l2: Optional[float] = None
    max_batches = len(loader)
    progress = tqdm(loader, desc=f"Epoch {epoch} [ImageNet pretrain]", disable=rank != 0)
    optimizer_steps = 0

    for batch_index, (images, targets) in enumerate(progress):
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
            batch_loss = F.cross_entropy(
                logits.float(), targets, label_smoothing=label_smoothing
            )
            scaler.scale(batch_loss / group_size).backward()
        if sync_gradients:
            scaler.unscale_(optimizer)
            if first_step_gradient_l2 is None:
                first_step_gradient_l2 = base._gradient_l2_norm(model.parameters())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1
        with torch.no_grad():
            predictions = logits.float().argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            samples += int(targets.numel())
            loss_sum += float(batch_loss.detach().item()) * int(targets.shape[0])
        if rank == 0:
            progress.set_postfix(
                {
                    "loss": f"{loss_sum / max(samples, 1):.4f}",
                    "top1": f"{correct / max(samples, 1):.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )
    metrics = _reduce_imagenet_metrics(loss_sum, correct, samples, device, world_size)
    metrics["optimizer_steps"] = optimizer_steps
    metrics["ce_gradient_l2_first_optimizer_step"] = float(first_step_gradient_l2 or 0.0)
    return metrics


def evaluate_imagenet(
    model: torch.nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    amp_enabled: bool,
) -> Optional[Dict[str, float]]:
    if loader is None:
        return None
    model.eval()
    loss_sum = 0.0
    correct = 0
    samples = 0
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="ImageNet val", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with common.autocast_context(device, amp_enabled):
                logits = model(images)
            loss = F.cross_entropy(logits.float(), targets, reduction="sum")
            predictions = logits.float().argmax(dim=1)
            loss_sum += float(loss.item())
            correct += int((predictions == targets).sum().item())
            samples += int(targets.numel())
    return {
        "loss": loss_sum / max(samples, 1),
        "top1": correct / max(samples, 1),
        "samples": samples,
    }


def build_imagenet_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    metrics: Mapping[str, object],
    config: Mapping[str, object],
    hashes: Mapping[str, object],
    class_names: Sequence[str],
    num_classes: int = 1000,
    experiment: str = IMAGENET_EXPERIMENT,
    artifact_type: str = IMAGENET_ARTIFACT_TYPE,
    dataset_name: str = "ImageNet-1K",
) -> Dict[str, object]:
    model_state = common.cpu_state_dict(model)
    return {
        "format_version": IMAGENET_ARTIFACT_FORMAT_VERSION,
        "artifact_type": artifact_type,
        "experiment": experiment,
        "model_name": "mobilenet_v2",
        "initialization": "weights=None",
        "pretraining_dataset": dataset_name,
        "num_classes": num_classes,
        "class_names": list(class_names),
        "model_state_dict": model_state,
        "model_state_sha256": common.state_dict_sha256(model_state),
        "best_epoch": epoch,
        "best_metrics": copy.deepcopy(metrics),
        "config": copy.deepcopy(config),
        "hashes": copy.deepcopy(hashes),
    }


def load_imagenet_backbone(model: torch.nn.Module, checkpoint_path: Path) -> Dict[str, object]:
    checkpoint_path = checkpoint_path.resolve()
    common.verify_checkpoint_sidecar(checkpoint_path)
    payload = common.safe_torch_load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("artifact_type") not in {
        IMAGENET_ARTIFACT_TYPE,
        IMAGENETTE_ARTIFACT_TYPE,
    }:
        raise RuntimeError(
            "--imagenet-checkpoint is not an S2-P local ImageNet artifact: "
            f"{payload.get('artifact_type')!r}"
        )
    if payload.get("num_classes") not in {10, 1000}:
        raise RuntimeError(
            "Pretraining checkpoint must contain either 1000 ImageNet classes or "
            f"10 ImageNette classes, got {payload.get('num_classes')!r}"
        )
    state_dict = payload["model_state_dict"]
    if common.state_dict_sha256(state_dict) != payload.get("model_state_sha256"):
        raise RuntimeError("Pretraining checkpoint model state failed SHA-256 verification")
    feature_state = {
        key[len("features."):]: value
        for key, value in state_dict.items()
        if key.startswith("features.")
    }
    if not feature_state:
        raise RuntimeError("ImageNet checkpoint contains no MobileNetV2 features.* weights")
    model.backbone.load_state_dict(feature_state, strict=True)
    return payload


def build_s2p_best_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    optimizer_step: int,
    dev_metrics: Mapping[str, object],
    config: Mapping[str, object],
    hashes: Mapping[str, object],
    dataset_lock: Mapping[str, object],
    shape_audit: Mapping[str, object],
    experiment: str = S2P_EXPERIMENT,
    artifact_type: str = S2P_ARTIFACT_TYPE,
    initialization: str = "locally trained ImageNet-1K MobileNetV2",
    pretrain_artifact: Optional[str] = None,
) -> Dict[str, object]:
    model_state = common.cpu_state_dict(model)
    return {
        "format_version": S2P_ARTIFACT_FORMAT_VERSION,
        "artifact_type": artifact_type,
        "experiment": experiment,
        "model_name": "mobilenet_v2",
        "initialization": initialization,
        "pretraining_artifact": pretrain_artifact,
        "pretrained": True,
        "num_classes": common.NUM_CLASSES,
        "class_names": list(common.CITYSCAPES_CLASSES),
        "output_stride": common.OUTPUT_STRIDE,
        "head_type": "R-ASPP",
        "feature_taps": copy.deepcopy(base.FEATURE_TAPS),
        "model_state_dict": model_state,
        "model_state_sha256": common.state_dict_sha256(model_state),
        "best_epoch": epoch,
        "best_optimizer_step": optimizer_step,
        "best_dev_metrics": copy.deepcopy(dev_metrics),
        "config": copy.deepcopy(config),
        "hashes": copy.deepcopy(hashes),
        "dataset_lock": copy.deepcopy(dataset_lock),
        "shape_audit": copy.deepcopy(shape_audit),
    }


def load_s2p_model(
    checkpoint_path: Path,
    device: object = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    checkpoint_path = checkpoint_path.resolve()
    common.verify_checkpoint_sidecar(checkpoint_path)
    payload = common.safe_torch_load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("artifact_type") not in {
        S2P_ARTIFACT_TYPE,
        S2P_IMAGENETTE_ARTIFACT_TYPE,
    }:
        raise RuntimeError(f"Not an S2-P artifact: {payload.get('artifact_type')!r}")
    config = payload["config"]
    model = base.build_model(config["head_channels"], config["dropout"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if common.state_dict_sha256(model.state_dict()) != payload["model_state_sha256"]:
        raise RuntimeError("S2-P model state failed SHA-256 verification")
    return model.to(torch.device(device)).eval(), payload


def _reduce_train_metrics(
    confusion: torch.Tensor,
    loss_sum: float,
    valid_pixels: int,
    device: torch.device,
    world_size: int,
) -> Dict[str, object]:
    if world_size == 1:
        return common.metrics_from_confusion(confusion, loss_sum, valid_pixels)
    confusion_device = confusion.to(device=device, dtype=torch.int64)
    totals = torch.tensor([loss_sum, float(valid_pixels)], device=device, dtype=torch.float64)
    dist.all_reduce(confusion_device, op=dist.ReduceOp.SUM)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return common.metrics_from_confusion(
        confusion_device.cpu(),
        float(totals[0].item()),
        int(totals[1].item()),
    )


def train_one_epoch_server(
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
    remaining_optimizer_steps: int,
    rank: int,
    world_size: int,
) -> Tuple[Dict[str, object], int]:
    if sampler is not None:
        sampler.set_epoch(epoch)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    confusion = torch.zeros(common.NUM_CLASSES, common.NUM_CLASSES, dtype=torch.int64)
    loss_sum = 0.0
    valid_pixels = 0
    optimizer_steps = 0
    first_step_gradient_l2: Optional[float] = None
    possible_steps = math.ceil(len(loader) / accumulation_steps)
    target_steps = min(possible_steps, remaining_optimizer_steps)
    max_batches = min(len(loader), target_steps * accumulation_steps)
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch} [S2-P Cityscapes server DDP]",
        disable=rank != 0,
    )

    for batch_index, (images, targets, _) in enumerate(progress):
        if batch_index >= max_batches:
            break
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
            scaler.scale(batch_loss / group_size).backward()

        if sync_gradients:
            scaler.unscale_(optimizer)
            if first_step_gradient_l2 is None:
                first_step_gradient_l2 = base._gradient_l2_norm(model.parameters())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1

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
            f"Optimizer-step accounting failed: actual={optimizer_steps}, expected={target_steps}"
        )
    metrics = _reduce_train_metrics(confusion, loss_sum, valid_pixels, device, world_size)
    metrics["ce_gradient_l2_first_optimizer_step"] = first_step_gradient_l2
    return metrics, optimizer_steps


def _smoke_test(
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
    loss = F.cross_entropy(logits.float(), targets, ignore_index=common.IGNORE_INDEX)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke-test loss: {loss.item()}")
    if rank == 0:
        print(f"[OK] server DDP smoke test: sample={paths[0]}, logits={tuple(logits.shape)}, loss={loss.item():.6f}")


def _imagenet_smoke_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    label_smoothing: float,
    rank: int,
    dataset_label: str = "ImageNet-1K",
) -> None:
    model.train()
    images, targets = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    model.zero_grad(set_to_none=True)
    with common.autocast_context(device, amp_enabled):
        logits = model(images)
        loss = F.cross_entropy(logits.float(), targets, label_smoothing=label_smoothing)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite ImageNet smoke-test loss: {loss.item()}")
    if rank == 0:
        print(
            f"[OK] {dataset_label} pretrain smoke test: batch={images.shape[0]}, "
            f"logits={tuple(logits.shape)}, loss={loss.item():.6f}"
        )


def run_imagenet_pretraining(args: argparse.Namespace) -> None:
    rank, local_rank, world_size, device = setup_distributed(args)
    main_process = rank == 0
    imagenette_stage = args.stage == "imagenette-pretrain"
    pretrain_num_classes = 10 if imagenette_stage else 1000
    pretrain_dataset_name = "ImageNette-10" if imagenette_stage else "ImageNet-1K"
    pretrain_experiment = IMAGENETTE_EXPERIMENT if imagenette_stage else IMAGENET_EXPERIMENT
    pretrain_artifact_type = (
        IMAGENETTE_ARTIFACT_TYPE if imagenette_stage else IMAGENET_ARTIFACT_TYPE
    )
    pretrain_output_dir = (
        args.imagenette_output_dir if imagenette_stage else args.imagenet_output_dir
    )
    pretrain_artifact_prefix = "imagenette" if imagenette_stage else "imagenet"
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    model: Optional[torch.nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    scaler: Optional[torch.amp.GradScaler] = None
    successful_exit = False
    try:
        common.set_global_seed(args.seed + rank, args.deterministic)
        accumulation_steps = effective_imagenet_accumulation_steps(args, world_size)
        (
            train_loader,
            train_sampler,
            train_generator,
            val_loader,
            class_names,
        ) = build_imagenet_loaders(
            args,
            device,
            rank,
            world_size,
            expected_num_classes=pretrain_num_classes,
            dataset_label=pretrain_dataset_name,
        )
        model = build_imagenet_classifier(pretrain_num_classes).to(device)
        initial_model_hash = common.state_dict_sha256(model.state_dict())
        if world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=True,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        amp_enabled = bool(args.amp and device.type == "cuda")
        steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
        total_optimizer_steps = steps_per_epoch * args.imagenet_epochs
        if main_process:
            print(
                f"[INFO] {pretrain_dataset_name} pretrain: world_size={world_size}, device={device}, "
                f"AMP={amp_enabled}, classes={len(class_names)}, "
                f"workers/rank={args.num_workers}, context={args.multiprocessing_context}, "
                f"pin_memory={args.pin_memory}"
            )
            print(
                f"[OK] {pretrain_dataset_name} samples={len(train_loader.dataset):,}; "
                f"local batch={args.imagenet_batch_size}; global batch="
                f"{args.imagenet_batch_size * accumulation_steps * world_size}; "
                f"steps/epoch={steps_per_epoch}; total steps={total_optimizer_steps:,}"
            )
        if args.smoke_test:
            _imagenet_smoke_test(
                model,
                train_loader,
                device,
                amp_enabled,
                args.imagenet_label_smoothing,
                rank,
                pretrain_dataset_name,
            )
            successful_exit = True
            return

        paths = imagenet_paths(pretrain_output_dir, args.seed, pretrain_artifact_prefix)
        paths["run_dir"].mkdir(parents=True, exist_ok=True)
        artifact_paths = [paths["best"], paths["last"], paths["history"], paths["metrics"]]
        if not args.resume and any(path.exists() for path in artifact_paths):
            raise FileExistsError(
                f"{pretrain_dataset_name} pretraining artifacts already exist in {paths['run_dir']}; "
                "use --resume or another pretraining output directory"
            )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.imagenet_lr,
            momentum=args.imagenet_momentum,
            weight_decay=args.imagenet_weight_decay,
        )

        def lr_factor(step: int) -> float:
            progress = min(step, total_optimizer_steps) / max(total_optimizer_steps, 1)
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        config: Dict[str, object] = {
            "experiment": pretrain_experiment,
            "server_entry_point": str(Path(__file__).resolve()),
            "seed": args.seed,
            "world_size": world_size,
            "dataset_root": str(args.imagenet_root.resolve()),
            "train_directory": str(_resolve_imagenet_split(args.imagenet_root, "train")),
            "val_directory": str(args.imagenet_root.resolve() / "val")
            if (args.imagenet_root.resolve() / "val").is_dir()
            else None,
            "pretraining_dataset": pretrain_dataset_name,
            "pretraining_artifact_type": pretrain_artifact_type,
            "num_classes": pretrain_num_classes,
            "class_count": len(class_names),
            "epochs": args.imagenet_epochs,
            "batch_size_per_gpu": args.imagenet_batch_size,
            "global_batch_size": args.imagenet_batch_size * accumulation_steps * world_size,
            "accumulation_steps_per_gpu": accumulation_steps,
            "eval_batch_size": args.imagenet_eval_batch_size,
            "num_workers_per_gpu": args.num_workers,
            "multiprocessing_context": args.multiprocessing_context,
            "pin_memory": bool(args.pin_memory),
            "persistent_workers": bool(args.persistent_workers),
            "prefetch_factor": args.prefetch_factor,
            "image_size": args.imagenet_image_size,
            "optimizer": "SGD",
            "learning_rate": args.imagenet_lr,
            "momentum": args.imagenet_momentum,
            "weight_decay": args.imagenet_weight_decay,
            "label_smoothing": args.imagenet_label_smoothing,
            "scheduler": "cosine_with_1_percent_floor",
            "amp": amp_enabled,
            "deterministic": args.deterministic,
        }
        hashes = {
            "initial_model_state_sha256": initial_model_hash,
            "training_script_sha256": common.sha256_file(Path(__file__).resolve()),
        }
        history: List[Dict[str, object]] = []
        best_score = float("-inf")
        best_epoch: Optional[int] = None
        best_metrics: Optional[Dict[str, float]] = None
        start_epoch = 1
        if args.resume:
            if not paths["last"].is_file():
                raise FileNotFoundError(f"ImageNet resume checkpoint not found: {paths['last']}")
            resume_payload = common.safe_torch_load(
                paths["last"], map_location="cpu", weights_only=False
            )
            if resume_payload.get("config") != config:
                raise RuntimeError(
                    f"{pretrain_dataset_name} resume configuration differs from current arguments"
                )
            model_to_load = model.module if isinstance(model, DDP) else model
            model_to_load.load_state_dict(resume_payload["model_state_dict"], strict=True)
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
            scaler.load_state_dict(resume_payload["scaler_state_dict"])
            history = resume_payload["history"]
            best_score = float(resume_payload["best_score"])
            best_epoch = resume_payload["best_epoch"]
            best_metrics = resume_payload["best_metrics"]
            start_epoch = int(resume_payload["epoch"]) + 1
            if main_process:
                print(f"[OK] Resuming {pretrain_dataset_name} pretraining at epoch {start_epoch}")

        training_started = time.time()
        for epoch in range(start_epoch, args.imagenet_epochs + 1):
            train_metrics = train_one_epoch_imagenet(
                model=model,
                loader=train_loader,
                sampler=train_sampler,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                amp_enabled=amp_enabled,
                accumulation_steps=accumulation_steps,
                label_smoothing=args.imagenet_label_smoothing,
                epoch=epoch,
                rank=rank,
                world_size=world_size,
            )
            barrier(world_size)
            val_metrics = evaluate_imagenet(
                model.module if isinstance(model, DDP) else model,
                val_loader,
                device,
                amp_enabled,
            ) if main_process else None
            candidate_metrics = val_metrics or train_metrics
            candidate_score = float(candidate_metrics["top1"])
            if main_process and candidate_score > best_score:
                best_score = candidate_score
                best_epoch = epoch
                best_metrics = copy.deepcopy(candidate_metrics)
                best_payload = build_imagenet_checkpoint(
                    model.module if isinstance(model, DDP) else model,
                    epoch,
                    candidate_metrics,
                    config,
                    hashes,
                    class_names,
                    num_classes=pretrain_num_classes,
                    experiment=pretrain_experiment,
                    artifact_type=pretrain_artifact_type,
                    dataset_name=pretrain_dataset_name,
                )
                checkpoint_hash = common.write_checkpoint_with_sidecar(best_payload, paths["best"])
                print(
                    f"[OK] {pretrain_dataset_name} best updated: epoch={epoch}, "
                    f"top1={candidate_score:.6f}, sha256={checkpoint_hash}"
                )
            if main_process:
                epoch_record = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                history.append(epoch_record)
                common.write_json_atomic(paths["history"], history)
                last_payload = {
                    "epoch": epoch,
                    "model_state_dict": common.cpu_state_dict(
                        model.module if isinstance(model, DDP) else model
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "train_generator_state": train_generator.get_state(),
                    "history": history,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                    "config": config,
                    "hashes": hashes,
                }
                common.torch_save_atomic(last_payload, paths["last"])
                message = (
                    f"Epoch {epoch}/{args.imagenet_epochs}: "
                    f"train_top1={train_metrics['top1']:.4f}, "
                    f"train_loss={train_metrics['loss']:.4f}"
                )
                if val_metrics is not None:
                    message += f", val_top1={val_metrics['top1']:.4f}"
                print(message)
            barrier(world_size)

        barrier(world_size)
        if main_process:
            if best_epoch is None or best_metrics is None:
                raise RuntimeError("ImageNet pretraining ended without a selected checkpoint")
            checkpoint_hash = common.verify_checkpoint_sidecar(paths["best"])
            results = {
                "experiment": pretrain_experiment,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "config": config,
                "hashes": {**hashes, "checkpoint_sha256": checkpoint_hash},
                "training": {
                    "elapsed_seconds": time.time() - training_started,
                    "epochs_completed": args.imagenet_epochs,
                    "optimizer_steps": total_optimizer_steps,
                },
                "software": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "torchvision": str(torchvision.__version__),
                    "numpy": np.__version__,
                    "pillow": __import__("PIL").__version__,
                    "platform": platform.platform(),
                },
                "artifacts": {key: str(value) for key, value in paths.items() if key != "run_dir"},
            }
            common.write_json_atomic(paths["metrics"], results)
            print(
                f"[DONE] {pretrain_dataset_name} MobileNetV2 pretraining: "
                f"epochs={args.imagenet_epochs}, "
                f"best top1={best_metrics['top1']:.6f}, checkpoint={paths['best']}"
            )
        barrier(world_size)
        successful_exit = True
    finally:
        _shutdown_loader(train_loader)
        _shutdown_loader(val_loader)
        _synchronize_cuda(device)
        if successful_exit and world_size > 1 and dist.is_initialized():
            barrier(world_size)
        scaler = None
        scheduler = None
        optimizer = None
        model = None
        _synchronize_cuda(device)
        if successful_exit and world_size > 1 and dist.is_initialized():
            barrier(world_size)
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def run_cityscapes_training(args: argparse.Namespace) -> None:
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
        accumulation_steps = effective_accumulation_steps(args, world_size)
        common.set_global_seed(args.seed + rank, args.deterministic)
        dataset_root = args.dataset_root.resolve()
        dataset_lock, entries_by_split = common.validate_dataset_lock(dataset_root)
        train_loader, train_sampler, train_generator = build_train_loader(
            args, dataset_root, entries_by_split, device, rank, world_size
        )
        dev_loader = build_dev_loader(args, dataset_root, entries_by_split, device) if main_process else None

        model = base.build_model(args.head_channels, args.dropout)
        assert args.imagenet_checkpoint is not None
        imagenet_payload = load_imagenet_backbone(model, args.imagenet_checkpoint)
        model = model.to(device)
        shape_audit = base.audit_model_shapes(
            model, device, args.crop_height, args.crop_width, bool(args.amp and device.type == "cuda")
        )
        initial_model_hash = common.state_dict_sha256(model.state_dict())
        imagenet_checkpoint_hash = common.verify_checkpoint_sidecar(args.imagenet_checkpoint.resolve())
        imagenet_backbone_hash = common.state_dict_sha256(model.backbone.state_dict())
        imagenette_transfer = (
            imagenet_payload.get("artifact_type") == IMAGENETTE_ARTIFACT_TYPE
        )
        city_experiment = S2P_IMAGENETTE_EXPERIMENT if imagenette_transfer else S2P_EXPERIMENT
        city_artifact_type = (
            S2P_IMAGENETTE_ARTIFACT_TYPE if imagenette_transfer else S2P_ARTIFACT_TYPE
        )
        city_initialization = (
            "locally trained ImageNette-10 MobileNetV2"
            if imagenette_transfer
            else "locally trained ImageNet-1K MobileNetV2"
        )
        city_artifact_prefix = "s2_p_imagenette" if imagenette_transfer else "s2_p"
        city_output_dir = args.output_dir
        if imagenette_transfer and args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
            city_output_dir = DEFAULT_S2P_IMAGENETTE_OUTPUT_DIR
        if world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=True,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        amp_enabled = bool(args.amp and device.type == "cuda")
        parameters = base._parameter_report(model.module if isinstance(model, DDP) else model)
        steps_per_full_epoch = math.ceil(len(train_loader) / accumulation_steps)
        estimated_epochs = math.ceil(args.max_steps / steps_per_full_epoch)
        if main_process:
            print(
                f"[INFO] {city_experiment} server DDP: world_size={world_size}, device={device}, "
                f"AMP={amp_enabled}, workers/rank={args.num_workers}, "
                f"context={args.multiprocessing_context}, pin_memory={args.pin_memory}"
            )
            print(
                f"[OK] params={parameters['trainable_parameters']:,}; "
                f"local batch={args.batch_size}; global batch="
                f"{args.batch_size * accumulation_steps * world_size}; "
                f"steps/full epoch={steps_per_full_epoch}; about {estimated_epochs} epochs"
            )
        if args.smoke_test:
            _smoke_test(model, train_loader, device, amp_enabled, rank)
            # Mark this before returning so the common finally block can perform
            # an ordered DDP/CUDA/DataLoader teardown on every rank.
            successful_exit = True
            return

        paths = server_paths(city_output_dir, args.seed, city_artifact_prefix)
        paths["run_dir"].mkdir(parents=True, exist_ok=True)
        artifact_paths = [paths["best"], paths["last"], paths["history"], paths["metrics"], paths["per_image"]]
        if not args.resume and any(path.exists() for path in artifact_paths):
            raise FileExistsError(
                f"{city_experiment} server artifacts already exist in {paths['run_dir']}; "
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
        config: Dict[str, object] = {
            "experiment": city_experiment,
            "server_entry_point": str(Path(__file__).resolve()),
            "seed": args.seed,
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": args.batch_size * accumulation_steps * world_size,
            "accumulation_steps_per_gpu": accumulation_steps,
            "max_optimizer_steps": args.max_steps,
            "eval_batch_size": args.eval_batch_size,
            "num_workers_per_gpu": args.num_workers,
            "multiprocessing_context": args.multiprocessing_context,
            "pin_memory": bool(args.pin_memory),
            "persistent_workers": bool(args.persistent_workers),
            "prefetch_factor": args.prefetch_factor,
            "optimizer": "SGD",
            "learning_rate": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "scheduler": "polynomial",
            "poly_power": args.poly_power,
            "min_lr_ratio": args.min_lr_ratio,
            "eval_every_steps": args.eval_every_steps,
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
            "initialization": city_initialization,
            "pretraining_dataset": imagenet_payload.get("pretraining_dataset"),
            "pretraining_artifact_type": imagenet_payload.get("artifact_type"),
            "imagenet_checkpoint": str(args.imagenet_checkpoint.resolve()),
            "imagenet_checkpoint_artifact": imagenet_payload.get("artifact_type"),
            "imagenet_checkpoint_sha256": imagenet_checkpoint_hash,
            "imagenet_backbone_state_sha256": imagenet_backbone_hash,
            "backbone_frozen": False,
            "loss": "pixel_cross_entropy_only",
            "knowledge_distillation": False,
            "test_local_evaluated": False,
            "distributed_backend": "nccl" if world_size > 1 else None,
        }
        hashes = {
            "initial_model_state_sha256": initial_model_hash,
            "imagenet_checkpoint_sha256": imagenet_checkpoint_hash,
            "imagenet_backbone_state_sha256": imagenet_backbone_hash,
            "training_script_sha256": common.sha256_file(Path(__file__).resolve()),
        }

        history: List[Dict[str, object]] = []
        best_key: Optional[Tuple[float, float, float, float]] = None
        best_epoch: Optional[int] = None
        best_optimizer_step: Optional[int] = None
        best_dev_metrics: Optional[Dict[str, object]] = None
        epoch = 0
        cumulative_optimizer_steps = 0
        if args.resume:
            resume_payload = common.safe_torch_load(paths["last"], map_location="cpu", weights_only=False)
            if resume_payload.get("config") != config:
                raise RuntimeError(
                    f"{city_experiment} server resume configuration differs from current arguments"
                )
            model_to_load = model.module if isinstance(model, DDP) else model
            model_to_load.load_state_dict(resume_payload["model_state_dict"], strict=True)
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
            scaler.load_state_dict(resume_payload["scaler_state_dict"])
            history = resume_payload["history"]
            best_key = resume_payload["best_key"]
            best_epoch = resume_payload["best_epoch"]
            best_optimizer_step = resume_payload["best_optimizer_step"]
            best_dev_metrics = resume_payload["best_dev_metrics"]
            epoch = int(resume_payload["epoch"])
            cumulative_optimizer_steps = int(resume_payload["optimizer_steps"])
            if main_process:
                print(f"[OK] Resuming server run after epoch {epoch}, step {cumulative_optimizer_steps:,}")

        training_started = time.time()
        while cumulative_optimizer_steps < args.max_steps:
            epoch += 1
            remaining_steps = args.max_steps - cumulative_optimizer_steps
            train_metrics, optimizer_steps = train_one_epoch_server(
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
                remaining_optimizer_steps=remaining_steps,
                rank=rank,
                world_size=world_size,
            )
            cumulative_optimizer_steps += optimizer_steps
            should_evaluate = (
                cumulative_optimizer_steps % args.eval_every_steps == 0
                or cumulative_optimizer_steps == args.max_steps
            )
            dev_metrics: Optional[Dict[str, object]] = None
            if should_evaluate:
                barrier(world_size)
                if main_process:
                    assert dev_loader is not None
                    dev_metrics, _ = common.evaluate(
                        model=model.module if isinstance(model, DDP) else model,
                        loader=dev_loader,
                        device=device,
                        amp_enabled=amp_enabled,
                        split_name="dev_local server",
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
                        best_payload = build_s2p_best_checkpoint(
                            model=model.module if isinstance(model, DDP) else model,
                            epoch=epoch,
                            optimizer_step=cumulative_optimizer_steps,
                            dev_metrics=dev_metrics,
                            config=config,
                            hashes=hashes,
                            dataset_lock=dataset_lock,
                            shape_audit=shape_audit,
                            experiment=city_experiment,
                            artifact_type=city_artifact_type,
                            initialization=city_initialization,
                            pretrain_artifact=imagenet_payload.get("artifact_type"),
                        )
                        checkpoint_hash = common.write_checkpoint_with_sidecar(best_payload, paths["best"])
                        print(
                            f"[OK] {city_experiment} server best updated: step={cumulative_optimizer_steps:,}, "
                            f"dev_mIoU={dev_metrics['mIoU']:.6f}, sha256={checkpoint_hash}"
                        )
                barrier(world_size)

            if main_process:
                epoch_record = {
                    "epoch": epoch,
                    "optimizer_steps": cumulative_optimizer_steps,
                    "optimizer_steps_this_epoch": optimizer_steps,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train": train_metrics,
                    "dev": dev_metrics,
                }
                history.append(epoch_record)
                common.write_json_atomic(paths["history"], history)
                last_payload = {
                    "epoch": epoch,
                    "optimizer_steps": cumulative_optimizer_steps,
                    "model_state_dict": common.cpu_state_dict(model.module if isinstance(model, DDP) else model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "train_generator_state": train_generator.get_state(),
                    "history": history,
                    "best_key": best_key,
                    "best_epoch": best_epoch,
                    "best_optimizer_step": best_optimizer_step,
                    "best_dev_metrics": best_dev_metrics,
                    "config": config,
                    "hashes": hashes,
                    "dataset_lock": dataset_lock,
                }
                common.torch_save_atomic(last_payload, paths["last"])
                message = (
                    f"Epoch {epoch}: step={cumulative_optimizer_steps:,}/{args.max_steps:,}, "
                    f"train_mIoU={train_metrics['mIoU']:.4f}, train_loss={train_metrics['loss']:.4f}"
                )
                if dev_metrics is not None:
                    message += f", dev_mIoU={dev_metrics['mIoU']:.4f}"
                print(message)
            barrier(world_size)

        barrier(world_size)
        if main_process:
            if best_epoch is None or best_optimizer_step is None or best_dev_metrics is None:
                raise RuntimeError("Server run ended without a selected dev checkpoint")
            selected_model, selected_payload = load_s2p_model(paths["best"], device=device)
            selected_dev_metrics, per_image_rows = common.evaluate(
                model=selected_model,
                loader=dev_loader,
                device=device,
                amp_enabled=amp_enabled,
                split_name="selected dev_local server",
                boundary_tolerance=args.boundary_tolerance,
                collect_per_image=True,
            )
            if not common.metrics_reproduce(selected_dev_metrics, best_dev_metrics):
                raise RuntimeError(
                    f"Reloaded {city_experiment} checkpoint did not reproduce best dev metrics: "
                    f"saved={best_dev_metrics['mIoU']}, reloaded={selected_dev_metrics['mIoU']}"
                )
            common.write_jsonl_atomic(paths["per_image"], per_image_rows)
            checkpoint_hash = common.verify_checkpoint_sidecar(paths["best"])
            efficiency = None
            if args.benchmark:
                efficiency = base.benchmark_model(
                    selected_model,
                    device,
                    args.benchmark_height,
                    args.benchmark_width,
                    args.benchmark_warmup,
                    args.benchmark_runs,
                )
            results = {
                "experiment": f"{city_experiment}-server-DDP",
                "protocol": (
                    f"MobileNetV2 initialized from a locally trained "
                    f"{imagenet_payload.get('pretraining_dataset', 'ImageNet')} "
                    "checkpoint and trained end to end with pixel CE for a fixed "
                    "80k-step Cityscapes budget; best checkpoint is selected by "
                    "dev_local mIoU; test_local is not evaluated."
                ),
                "best_epoch": best_epoch,
                "best_optimizer_step": best_optimizer_step,
                "best_dev_metrics": selected_dev_metrics,
                "config": config,
                "shape_audit": shape_audit,
                "dataset_lock": dataset_lock,
                "model": {
                    "model_name": base.MODEL_NAME,
                    "initialization": city_initialization,
                    "head": "R-ASPP",
                    "feature_taps": base.FEATURE_TAPS,
                    **parameters,
                },
                "efficiency": efficiency,
                "hashes": {
                    **hashes,
                    "selected_model_state_sha256": selected_payload["model_state_sha256"],
                    "checkpoint_sha256": checkpoint_hash,
                },
                "training": {
                    "elapsed_seconds": time.time() - training_started,
                    "optimizer_steps": cumulative_optimizer_steps,
                    "epochs_completed": epoch,
                    "steps_per_full_epoch": steps_per_full_epoch,
                },
                "software": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "torchvision": str(base.torchvision.__version__),
                    "numpy": np.__version__,
                    "pillow": __import__("PIL").__version__,
                    "platform": platform.platform(),
                },
                "artifacts": {key: str(value) for key, value in paths.items() if key != "run_dir"},
            }
            common.write_json_atomic(paths["metrics"], results)
            print(
                f"[DONE] {city_experiment} server DDP: GPUs={world_size}, "
                f"steps={cumulative_optimizer_steps:,}, "
                f"best dev mIoU={selected_dev_metrics['mIoU']:.6f}"
            )
        barrier(world_size)
        successful_exit = True
    finally:
        # DataLoader workers and DDP reducers can outlive the Python frame.  On
        # Linux/NCCL, destroying the process group while either still references
        # CUDA state can segfault during interpreter shutdown.  On a successful
        # run all ranks have reached the final barrier, so an additional pair of
        # barriers makes the cleanup order deterministic.  Exception paths avoid
        # collectives so a failed rank cannot strand the others in a barrier.
        _shutdown_loader(train_loader)
        _shutdown_loader(dev_loader)
        _synchronize_cuda(device)
        if successful_exit and world_size > 1 and dist.is_initialized():
            barrier(world_size)

        selected_model = None
        scaler = None
        scheduler = None
        optimizer = None
        model = None
        _synchronize_cuda(device)
        if successful_exit and world_size > 1 and dist.is_initialized():
            barrier(world_size)
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def run_training(args: argparse.Namespace) -> None:
    if args.stage in ("imagenet-pretrain", "imagenette-pretrain"):
        run_imagenet_pretraining(args)
    elif args.stage == "cityscapes":
        run_cityscapes_training(args)
    else:  # pragma: no cover - argparse restricts this value
        raise ValueError(f"Unsupported S2-P stage: {args.stage}")


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
