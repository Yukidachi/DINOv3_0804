"""S2-F server training entry point.

This file is deliberately separate from the ordinary training entries.  It is
designed for the Linux server diagnosed in ``profile_dino_s2_0.py``:

* launch with ``torchrun --nproc_per_node=2``;
* use DDP and ``no_sync`` during gradient accumulation;
* use DataLoader ``spawn`` workers;
* default to ``pin_memory=False`` because the server's pinned-memory path
  serialized the workers and reduced throughput by several times;
* write S2-F server-suffixed artifacts in a separate output directory.

S2-F is the frozen probe lower bound from the Cityscapes experiment plan:
MobileNetV2 is constructed with ``weights=None``, its complete backbone is
frozen in both parameter and BatchNorm running-statistic state, and only the
R-ASPP head is optimized with pixel cross entropy.

Example, preserving the original global batch size of 8:

    torchrun --standalone --nproc_per_node=2 dino_s2_f_server.py \
        --seed 42 --batch-size 2 --global-batch-size 8 \
        --num-workers 8 --persistent-workers

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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import dino as common
import dino_s2_0 as base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "S2_F_MobileNetV2_RASPP_server"

EXPERIMENT = "S2-F"
MODEL_NAME = base.MODEL_NAME
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_s2_f_server"
ARTIFACT_FORMAT_VERSION = 1


def build_model(head_channels: int, dropout: float):
    """Build the scratch MobileNetV2 probe and freeze its complete backbone."""

    model = base.build_model(head_channels, dropout)
    model.backbone.requires_grad_(False)
    model.head.requires_grad_(True)
    model.backbone.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("S2-F must optimize only the R-ASPP head")
    return model


def set_probe_train_mode(model: torch.nn.Module) -> None:
    """Train the head while keeping frozen backbone modules, including BatchNorm, eval."""

    model.train()
    module = model.module if isinstance(model, DDP) else model
    module.backbone.eval()
    module.head.train()


def parameter_report(model: torch.nn.Module) -> Dict[str, int]:
    backbone_parameters = sum(parameter.numel() for parameter in model.backbone.parameters())
    head_parameters = sum(parameter.numel() for parameter in model.head.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if total_parameters != backbone_parameters + head_parameters:
        raise RuntimeError("Parameter accounting failed")
    if trainable_parameters != head_parameters:
        raise RuntimeError("S2-F trainable parameter accounting failed")
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("S2-F backbone is not fully frozen")
    return {
        "total_parameters": total_parameters,
        "backbone_parameters": backbone_parameters,
        "head_parameters": head_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": backbone_parameters,
    }


def build_best_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    optimizer_step: int,
    dev_metrics: Mapping[str, object],
    config: Mapping[str, object],
    hashes: Mapping[str, object],
    dataset_lock: Mapping[str, object],
    shape_audit: Mapping[str, object],
) -> Dict[str, object]:
    model_state = common.cpu_state_dict(model)
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "experiment": EXPERIMENT,
        "model_name": MODEL_NAME,
        "initialization": "weights=None",
        "backbone_frozen": True,
        "trainable_scope": "R-ASPP head only",
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


def load_s2_f_model(
    checkpoint_path: Path,
    device: object = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    checkpoint_path = Path(checkpoint_path).resolve()
    common.verify_checkpoint_sidecar(checkpoint_path)
    payload = common.safe_torch_load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise RuntimeError(f"Not an S2-F artifact: {payload.get('artifact_type')!r}")
    if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise RuntimeError("Unsupported S2-F artifact format")
    if payload.get("initialization") != "weights=None" or not payload.get("backbone_frozen"):
        raise RuntimeError("S2-F artifact does not declare frozen scratch initialization")
    if payload.get("feature_taps") != base.FEATURE_TAPS:
        raise RuntimeError("S2-F artifact feature taps differ from the locked contract")
    config = payload["config"]
    model = build_model(config["head_channels"], config["dropout"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    actual_hash = common.state_dict_sha256(model.state_dict())
    if actual_hash != payload["model_state_sha256"]:
        raise RuntimeError("S2-F model state failed SHA-256 verification")
    model = model.to(torch.device(device)).eval()
    return model, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S2-F random MobileNetV2+R-ASPP frozen-backbone probe training "
            "for a multi-GPU Linux server."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=common.DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=40_000,
        help="Fixed optimizer-step budget for the frozen R-ASPP probe.",
    )
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
    )
    for field in positive:
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be at least 1")
    if args.global_batch_size is not None and args.global_batch_size < 1:
        parser.error("--global-batch-size must be at least 1")
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
            raise RuntimeError("CUDA is unavailable; use a single-process entry for CPU diagnostics")
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


def server_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    run_dir = output_dir.resolve() / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "best": run_dir / "s2_f_server_mobilenetv2_raspp.pth",
        "last": run_dir / "s2_f_server_last_checkpoint.pth",
        "history": run_dir / "s2_f_server_training_history.json",
        "metrics": run_dir / "s2_f_server_metrics.json",
        "per_image": run_dir / "s2_f_server_dev_per_image_confusion.jsonl",
    }


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
    set_probe_train_mode(model)
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
        desc=f"Epoch {epoch} [S2-F server DDP]",
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
    set_probe_train_mode(model)
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
    module = model.module if isinstance(model, DDP) else model
    if any(parameter.grad is not None for parameter in module.backbone.parameters()):
        raise RuntimeError("S2-F smoke test found a gradient in the frozen backbone")
    if rank == 0:
        print(f"[OK] S2-F server DDP smoke test: sample={paths[0]}, logits={tuple(logits.shape)}, loss={loss.item():.6f}")


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
        accumulation_steps = effective_accumulation_steps(args, world_size)
        common.set_global_seed(args.seed + rank, args.deterministic)
        dataset_root = args.dataset_root.resolve()
        dataset_lock, entries_by_split = common.validate_dataset_lock(dataset_root)
        train_loader, train_sampler, train_generator = build_train_loader(
            args, dataset_root, entries_by_split, device, rank, world_size
        )
        dev_loader = build_dev_loader(args, dataset_root, entries_by_split, device) if main_process else None

        model = build_model(args.head_channels, args.dropout).to(device)
        shape_audit = base.audit_model_shapes(
            model, device, args.crop_height, args.crop_width, bool(args.amp and device.type == "cuda")
        )
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
        model_core = model.module if isinstance(model, DDP) else model
        initial_backbone_hash = common.state_dict_sha256(model_core.backbone.state_dict())
        parameters = parameter_report(model_core)
        steps_per_full_epoch = math.ceil(len(train_loader) / accumulation_steps)
        estimated_epochs = math.ceil(args.max_steps / steps_per_full_epoch)
        if main_process:
            print(
                f"[INFO] S2-F server DDP: world_size={world_size}, device={device}, "
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

        paths = server_paths(args.output_dir, args.seed)
        paths["run_dir"].mkdir(parents=True, exist_ok=True)
        artifact_paths = [paths["best"], paths["last"], paths["history"], paths["metrics"], paths["per_image"]]
        if not args.resume and any(path.exists() for path in artifact_paths):
            raise FileExistsError(
                f"S2-F server artifacts already exist in {paths['run_dir']}; use --resume or another --output-dir"
            )

        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
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
            "experiment": EXPERIMENT,
            "server_entry_point": str(Path(__file__).resolve()),
            "seed": args.seed,
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": args.batch_size * accumulation_steps * world_size,
            "accumulation_steps_per_gpu": accumulation_steps,
            "max_optimizer_steps": args.max_steps,
            "budget_role": "A-group frozen backbone probe",
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
            "initialization": "weights=None",
            "backbone_frozen": True,
            "trainable_scope": "R-ASPP head only",
            "loss": "pixel_cross_entropy_only",
            "knowledge_distillation": False,
            "test_local_evaluated": False,
            "distributed_backend": "nccl" if world_size > 1 else None,
        }
        hashes = {
            "initial_model_state_sha256": initial_model_hash,
            "initial_backbone_state_sha256": initial_backbone_hash,
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
                raise RuntimeError("S2-F server resume configuration differs from current arguments")
            model_to_load = model.module if isinstance(model, DDP) else model
            model_to_load.load_state_dict(resume_payload["model_state_dict"], strict=True)
            if common.state_dict_sha256(model_to_load.backbone.state_dict()) != initial_backbone_hash:
                raise RuntimeError("Resume checkpoint changed the frozen S2-F backbone")
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
            if common.state_dict_sha256(model_core.backbone.state_dict()) != initial_backbone_hash:
                raise RuntimeError("S2-F frozen backbone changed during training")
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
                        best_payload = build_best_checkpoint(
                            model=model.module if isinstance(model, DDP) else model,
                            epoch=epoch,
                            optimizer_step=cumulative_optimizer_steps,
                            dev_metrics=dev_metrics,
                            config=config,
                            hashes=hashes,
                            dataset_lock=dataset_lock,
                            shape_audit=shape_audit,
                        )
                        checkpoint_hash = common.write_checkpoint_with_sidecar(best_payload, paths["best"])
                        print(
                            f"[OK] S2-F server best updated: step={cumulative_optimizer_steps:,}, "
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
            selected_model, selected_payload = load_s2_f_model(paths["best"], device=device)
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
                    "Reloaded S2-F checkpoint did not reproduce best dev metrics: "
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
                "experiment": "S2-F-server-DDP",
                "protocol": (
                    "Random MobileNetV2 weights=None with a fully frozen backbone; "
                    "only the 19-class R-ASPP head is trained with pixel CE for a "
                    "fixed 40k-step probe budget. Best checkpoint is selected by "
                    "dev_local mIoU; test_local is not evaluated."
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
                    "initialization": "weights=None",
                    "head": "R-ASPP",
                    "backbone_frozen": True,
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
                f"[DONE] S2-F server DDP: GPUs={world_size}, steps={cumulative_optimizer_steps:,}, "
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


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
