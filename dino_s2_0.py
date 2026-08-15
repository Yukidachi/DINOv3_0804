"""S2-0 baseline: scratch MobileNetV2 + R-ASPP on locked Cityscapes splits.

This entry point follows ``plan_markdown/Cityscapes知识蒸馏实验详单.md``:

* MobileNetV2 is constructed with ``weights=None``;
* the backbone is converted to output stride 16;
* the complete backbone and 19-class R-ASPP head are trained end to end;
* the only training loss is pixel cross entropy with ``ignore_index=255``;
* training stops at a fixed optimizer-step budget (80k by default);
* checkpoints are selected only by ``dev_local`` mIoU;
* ``test_local`` is validated by the split lock but is never evaluated here.

Typical commands (run in the ``pytorch`` conda environment):

    python -B dino_s2_0.py --smoke-test --seed 42
    python -B dino_s2_0.py --seed 42
    python -B dino_s2_0.py --seed 3407
    python -B dino_s2_0.py --seed 260805
    python -B dino_s2_0.py --seed 42 --resume
    python -B dino_s2_0.py --seed 42 --verify-only
"""

from __future__ import annotations

import argparse
import copy
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
from tqdm import tqdm

import dino as common


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "S2_0_MobileNetV2_RASPP"

EXPERIMENT = "S2-0"
MODEL_NAME = "mobilenet_v2"
NUM_CLASSES = common.NUM_CLASSES
IGNORE_INDEX = common.IGNORE_INDEX
OUTPUT_STRIDE = common.OUTPUT_STRIDE
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_s2_0"
ARTIFACT_FORMAT_VERSION = 1
FORMAL_SEEDS = (42, 3407, 260805)

FEATURE_TAPS = {
    "os4": {"module": "backbone.3", "index": 3, "channels": 24, "stride": 4},
    "os8": {"module": "backbone.6", "index": 6, "channels": 32, "stride": 8},
    "os16": {"module": "backbone.17", "index": 17, "channels": 320, "stride": 16},
    "raspp_input": {
        "module": "backbone.18",
        "index": 18,
        "channels": 1280,
        "stride": 16,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run S2-0: scratch MobileNetV2 (OS=16) + R-ASPP, end-to-end CE only."
        )
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=common.DEFAULT_DATASET_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
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
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision for training and evaluation.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request deterministic kernels where PyTorch provides them.",
    )
    parser.add_argument(
        "--benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure full-resolution FP32 batch=1 MACs, memory, and latency.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the seed run's last epoch-boundary checkpoint.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate data, OS/taps, forward, CE, and end-to-end gradients.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the selected S2-0 checkpoint, hashes, and shape contract.",
    )
    args = parser.parse_args()

    positive_int_fields = (
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
    )
    for field in positive_int_fields:
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if not 0 <= args.momentum < 1:
        parser.error("--momentum must be in [0, 1)")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if args.poly_power <= 0:
        parser.error("--poly-power must be positive")
    if not 0 < args.min_lr_ratio <= 1:
        parser.error("--min-lr-ratio must be in (0, 1]")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if not 0 < args.scale_min <= args.scale_max:
        parser.error("Require 0 < --scale-min <= --scale-max")
    if args.crop_height % OUTPUT_STRIDE or args.crop_width % OUTPUT_STRIDE:
        parser.error(f"Crop dimensions must be divisible by {OUTPUT_STRIDE}")
    if args.benchmark_height % OUTPUT_STRIDE or args.benchmark_width % OUTPUT_STRIDE:
        parser.error(f"Benchmark dimensions must be divisible by {OUTPUT_STRIDE}")
    if args.boundary_tolerance < 0:
        parser.error("--boundary-tolerance cannot be negative")
    if args.benchmark_warmup < 0:
        parser.error("--benchmark-warmup cannot be negative")
    if args.resume and (args.smoke_test or args.verify_only):
        parser.error("--resume cannot be combined with --smoke-test or --verify-only")
    return args


def _depthwise_3x3(block: nn.Module) -> nn.Conv2d:
    candidates = [
        module
        for module in block.modules()
        if isinstance(module, nn.Conv2d)
        and module.kernel_size == (3, 3)
        and module.groups == module.in_channels
        and module.out_channels == module.in_channels
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one MobileNetV2 depthwise 3x3 convolution, got {len(candidates)}"
        )
    return candidates[0]


def convert_backbone_to_output_stride16(backbone: nn.Sequential) -> None:
    """Replace the final stride-2 stage with stride 1 and dilation 2."""

    if getattr(backbone, "_cityscapes_output_stride", None) == OUTPUT_STRIDE:
        return
    if len(backbone) != 19:
        raise RuntimeError(f"Unexpected MobileNetV2 feature count: {len(backbone)}")
    for index in range(14, 18):
        depthwise = _depthwise_3x3(backbone[index])
        expected_stride = (2, 2) if index == 14 else (1, 1)
        if depthwise.stride != expected_stride:
            raise RuntimeError(
                f"Unexpected features.{index} depthwise stride: {depthwise.stride}"
            )
        if index == 14:
            depthwise.stride = (1, 1)
            if hasattr(backbone[index], "stride"):
                backbone[index].stride = 1
        depthwise.dilation = (2, 2)
        depthwise.padding = (2, 2)
    backbone._cityscapes_output_stride = OUTPUT_STRIDE


def build_backbone() -> nn.Sequential:
    # Explicit ``weights=None`` is the defining S2-0 initialization constraint.
    classification_model = mobilenet_v2(weights=None)
    backbone = classification_model.features
    convert_backbone_to_output_stride16(backbone)
    return backbone


class MobileNetV2RASPPStudent(nn.Module):
    def __init__(self, backbone: nn.Sequential, head: common.RASPPHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        tensor = images
        for index, block in enumerate(self.backbone):
            tensor = block(tensor)
            if index == FEATURE_TAPS["os4"]["index"]:
                outputs["os4"] = tensor
            elif index == FEATURE_TAPS["os8"]["index"]:
                outputs["os8"] = tensor
            elif index == FEATURE_TAPS["os16"]["index"]:
                outputs["os16"] = tensor
            elif index == FEATURE_TAPS["raspp_input"]["index"]:
                outputs["raspp_input"] = tensor
        if set(outputs) != set(FEATURE_TAPS):
            raise RuntimeError(f"Missing MobileNetV2 features: {set(FEATURE_TAPS) - set(outputs)}")
        return outputs

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        input_size = images.shape[-2:]
        features = self.extract_features(images)
        logits = self.head(features["raspp_input"])
        return F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )


def build_model(head_channels: int, dropout: float) -> MobileNetV2RASPPStudent:
    model = MobileNetV2RASPPStudent(
        backbone=build_backbone(),
        head=common.RASPPHead(
            in_channels=FEATURE_TAPS["raspp_input"]["channels"],
            num_classes=NUM_CLASSES,
            inter_channels=head_channels,
            dropout=dropout,
        ),
    )
    model.requires_grad_(True)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("S2-0 must train every MobileNetV2 and R-ASPP parameter")
    return model


def audit_model_shapes(
    model: MobileNetV2RASPPStudent,
    device: torch.device,
    height: int,
    width: int,
    amp_enabled: bool,
) -> Dict[str, object]:
    model.eval()
    sample = torch.zeros(1, 3, height, width, device=device)
    with torch.inference_mode(), common.autocast_context(device, amp_enabled):
        features = model.extract_features(sample)
        logits = model(sample)
    feature_shapes: Dict[str, List[int]] = {}
    for name, contract in FEATURE_TAPS.items():
        expected = (
            1,
            contract["channels"],
            height // contract["stride"],
            width // contract["stride"],
        )
        actual = tuple(features[name].shape)
        if actual != expected:
            raise RuntimeError(
                f"Shape audit failed for {name}: actual={actual}, expected={expected}"
            )
        feature_shapes[name] = list(actual)
    if logits.shape != (1, NUM_CLASSES, height, width):
        raise RuntimeError(f"Logit shape audit failed: {tuple(logits.shape)}")
    depthwise_contract = {}
    for index in range(14, 18):
        depthwise = _depthwise_3x3(model.backbone[index])
        depthwise_contract[f"backbone.{index}"] = {
            "stride": list(depthwise.stride),
            "dilation": list(depthwise.dilation),
            "padding": list(depthwise.padding),
        }
    return {
        "input_shape": [1, 3, height, width],
        "feature_taps": copy.deepcopy(FEATURE_TAPS),
        "feature_shapes": feature_shapes,
        "logit_shape": list(logits.shape),
        "output_stride": OUTPUT_STRIDE,
        "conversion": (
            "features.14 depthwise stride 2->1; features.14-17 depthwise "
            "convolutions use dilation=2 and padding=2"
        ),
        "depthwise_contract": depthwise_contract,
        "align_corners": False,
    }


def _gradient_l2_norm(parameters: Iterable[nn.Parameter]) -> float:
    norms = []
    for parameter in parameters:
        if parameter.grad is not None:
            norms.append(parameter.grad.detach().float().norm(2))
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm(2).item())


def run_smoke_test(
    model: MobileNetV2RASPPStudent,
    train_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> None:
    model.train()
    images, targets, paths = next(iter(train_loader))
    images = images[:1].to(device)
    targets = targets[:1].to(device)
    model.zero_grad(set_to_none=True)
    with common.autocast_context(device, amp_enabled):
        logits = model(images)
    loss = F.cross_entropy(
        logits.float(), targets, ignore_index=IGNORE_INDEX, reduction="mean"
    )
    loss.backward()
    backbone_gradients = sum(
        parameter.grad is not None for parameter in model.backbone.parameters()
    )
    head_gradients = sum(parameter.grad is not None for parameter in model.head.parameters())
    if backbone_gradients == 0 or head_gradients == 0:
        raise RuntimeError("S2-0 smoke test did not produce end-to-end gradients")
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke-test loss: {loss.item()}")
    print("[OK] S2-0 smoke test passed")
    print(f"   - sample: {paths[0]}")
    print(f"   - logits: {tuple(logits.shape)}")
    print(f"   - valid pixels: {int((targets != IGNORE_INDEX).sum().item())}")
    print(f"   - CE loss: {loss.item():.6f}")
    print(f"   - backbone gradient tensors: {backbone_gradients}")
    print(f"   - head gradient tensors: {head_gradients}")
    print(f"   - backbone gradient L2: {_gradient_l2_norm(model.backbone.parameters()):.6f}")
    print(f"   - head gradient L2: {_gradient_l2_norm(model.head.parameters()):.6f}")


def train_one_epoch(
    model: MobileNetV2RASPPStudent,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    accumulation_steps: int,
    epoch: int,
    remaining_optimizer_steps: int,
) -> Tuple[Dict[str, object], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    loss_sum = 0.0
    valid_pixels = 0
    optimizer_steps = 0
    first_step_gradient_l2: Optional[float] = None
    possible_steps = math.ceil(len(loader) / accumulation_steps)
    target_steps = min(possible_steps, remaining_optimizer_steps)
    max_batches = min(len(loader), target_steps * accumulation_steps)
    progress = tqdm(loader, desc=f"Epoch {epoch} [S2-0 CE]")

    for batch_index, (images, targets, _) in enumerate(progress):
        if batch_index >= max_batches:
            break
        group_position = batch_index % accumulation_steps
        if group_position == 0:
            group_size = min(accumulation_steps, max_batches - batch_index)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with common.autocast_context(device, amp_enabled):
            logits = model(images)
        logits_float = logits.float()
        batch_loss_sum = F.cross_entropy(
            logits_float,
            targets,
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        batch_valid = int((targets != IGNORE_INDEX).sum().item())
        if batch_valid == 0:
            raise RuntimeError("Training batch contains no valid Cityscapes pixels")
        batch_loss = batch_loss_sum / batch_valid
        scaler.scale(batch_loss / group_size).backward()

        if group_position + 1 == group_size:
            scaler.unscale_(optimizer)
            if first_step_gradient_l2 is None:
                first_step_gradient_l2 = _gradient_l2_norm(model.parameters())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1

        predictions = logits_float.detach().argmax(dim=1)
        confusion += common.confusion_counts(predictions, targets)
        loss_sum += float(batch_loss_sum.detach().item())
        valid_pixels += batch_valid
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
    metrics = common.metrics_from_confusion(confusion, loss_sum, valid_pixels)
    metrics["ce_gradient_l2_first_optimizer_step"] = first_step_gradient_l2
    return metrics, optimizer_steps


def capture_rng_state() -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def build_best_checkpoint(
    model: MobileNetV2RASPPStudent,
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
        "num_classes": NUM_CLASSES,
        "class_names": list(common.CITYSCAPES_CLASSES),
        "output_stride": OUTPUT_STRIDE,
        "head_type": "R-ASPP",
        "feature_taps": copy.deepcopy(FEATURE_TAPS),
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


def load_s2_0_model(
    checkpoint_path: Path,
    device: object = "cpu",
) -> Tuple[MobileNetV2RASPPStudent, Dict[str, object]]:
    checkpoint_path = Path(checkpoint_path).resolve()
    common.verify_checkpoint_sidecar(checkpoint_path)
    payload = common.safe_torch_load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise RuntimeError(f"Not an S2-0 artifact: {payload.get('artifact_type')!r}")
    if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise RuntimeError("Unsupported S2-0 artifact format")
    if payload.get("initialization") != "weights=None":
        raise RuntimeError("S2-0 artifact does not declare scratch initialization")
    if payload.get("feature_taps") != FEATURE_TAPS:
        raise RuntimeError("S2-0 artifact feature taps differ from the locked contract")
    config = payload["config"]
    model = build_model(config["head_channels"], config["dropout"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    actual_hash = common.state_dict_sha256(model.state_dict())
    if actual_hash != payload["model_state_sha256"]:
        raise RuntimeError("S2-0 model state failed SHA-256 verification")
    model = model.to(torch.device(device)).eval()
    return model, payload


def count_macs(model: nn.Module, sample: torch.Tensor) -> int:
    """Count Conv2d/Linear multiply-accumulates for one forward pass."""

    total = 0
    handles = []

    def conv_hook(module: nn.Conv2d, _inputs, output: torch.Tensor) -> None:
        nonlocal total
        kernel_ops = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * module.in_channels
            // module.groups
        )
        total += int(output.numel() * kernel_ops)

    def linear_hook(module: nn.Linear, _inputs, output: torch.Tensor) -> None:
        nonlocal total
        total += int(output.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    try:
        with torch.inference_mode():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()
    return total


def benchmark_model(
    model: MobileNetV2RASPPStudent,
    device: torch.device,
    height: int,
    width: int,
    warmup: int,
    runs: int,
) -> Dict[str, object]:
    model.eval().float()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    sample = torch.zeros(1, 3, height, width, device=device, dtype=torch.float32)
    macs = count_macs(model, sample)

    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings_ms: List[float] = []
    with torch.inference_mode():
        if device.type == "cuda":
            for _ in range(runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(sample)
                end.record()
                end.synchronize()
                timings_ms.append(float(start.elapsed_time(end)))
        else:
            for _ in range(runs):
                start_time = time.perf_counter()
                model(sample)
                timings_ms.append((time.perf_counter() - start_time) * 1000.0)

    values = np.asarray(timings_ms, dtype=np.float64)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return {
        "input_shape": [1, 3, height, width],
        "precision": "FP32",
        "batch_size": 1,
        "warmup_runs": warmup,
        "measured_runs": runs,
        "macs": macs,
        "gmacs": macs / 1e9,
        "mac_count_scope": (
            "Conv2d and Linear MACs; interpolation, pooling, normalization, "
            "activation, sigmoid, and elementwise multiply are excluded"
        ),
        "latency_ms_mean": float(values.mean()),
        "latency_ms_std": float(values.std(ddof=0)),
        "latency_ms_median": float(np.median(values)),
        "latency_ms_p90": float(np.percentile(values, 90)),
        "latency_ms_min": float(values.min()),
        "latency_ms_max": float(values.max()),
        "peak_cuda_memory_bytes": peak_memory,
        "peak_cuda_memory_mib": (
            None if peak_memory is None else peak_memory / (1024.0 * 1024.0)
        ),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
        ),
    }


def _parameter_report(model: MobileNetV2RASPPStudent) -> Dict[str, int]:
    backbone_parameters = sum(parameter.numel() for parameter in model.backbone.parameters())
    head_parameters = sum(parameter.numel() for parameter in model.head.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if total_parameters != backbone_parameters + head_parameters:
        raise RuntimeError("Parameter accounting failed")
    if trainable_parameters != total_parameters:
        raise RuntimeError("S2-0 is not configured for end-to-end training")
    return {
        "total_parameters": total_parameters,
        "backbone_parameters": backbone_parameters,
        "head_parameters": head_parameters,
        "trainable_parameters": trainable_parameters,
    }


def _run_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    run_dir = output_dir.resolve() / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "best": run_dir / "s2_0_mobilenetv2_raspp.pth",
        "last": run_dir / "s2_0_last_checkpoint.pth",
        "history": run_dir / "training_history.json",
        "metrics": run_dir / "s2_0_metrics.json",
        "per_image": run_dir / "dev_per_image_confusion.jsonl",
    }


def run_training(args: argparse.Namespace) -> None:
    invocation = {
        "executable": sys.executable,
        "argv": list(sys.argv),
    }
    common.set_global_seed(args.seed, args.deterministic)
    device = common.resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    dataset_root = args.dataset_root.resolve()
    dataset_lock, entries_by_split = common.validate_dataset_lock(dataset_root)
    print(
        "[OK] Locked Cityscapes splits: "
        f"train={len(entries_by_split['train_local'])}, "
        f"dev={len(entries_by_split['dev_local'])}, "
        f"test={len(entries_by_split['test_local'])} (not evaluated)"
    )
    if args.seed not in FORMAL_SEEDS:
        print(
            f"[WARN] Seed {args.seed} is not one of the formal S2-0 seeds {FORMAL_SEEDS}; "
            "treat this run as diagnostic."
        )

    model = build_model(args.head_channels, args.dropout).to(device)
    initial_model_hash = common.state_dict_sha256(model.state_dict())
    shape_audit = audit_model_shapes(
        model, device, args.crop_height, args.crop_width, amp_enabled
    )
    train_loader, dev_loader, train_generator = common.make_data_loaders(
        dataset_root, entries_by_split, args, device
    )
    parameters = _parameter_report(model)
    steps_per_full_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    estimated_epochs = math.ceil(args.max_steps / steps_per_full_epoch)
    print(f"[INFO] Device={device}; AMP={amp_enabled}")
    print(
        f"[OK] End-to-end trainable params={parameters['trainable_parameters']:,}; "
        f"backbone={parameters['backbone_parameters']:,}; head={parameters['head_parameters']:,}"
    )
    print(f"[OK] Feature shapes: {shape_audit['feature_shapes']}")
    print(
        f"[OK] Fixed budget={args.max_steps:,} optimizer steps; "
        f"{steps_per_full_epoch} steps/full epoch; about {estimated_epochs} epochs"
    )

    if args.smoke_test:
        run_smoke_test(model, train_loader, device, amp_enabled)
        return

    paths = _run_paths(args.output_dir, args.seed)
    run_dir = paths["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        selected_model, payload = load_s2_0_model(paths["best"], device=device)
        if (
            payload["dataset_lock"]["combined_manifest_sha256"]
            != dataset_lock["combined_manifest_sha256"]
        ):
            raise RuntimeError("Current locked split differs from the S2-0 artifact")
        verified_shape = audit_model_shapes(
            selected_model,
            device,
            args.crop_height,
            args.crop_width,
            amp_enabled,
        )
        print("[OK] S2-0 artifact verified")
        print(f"   - checkpoint: {paths['best']}")
        print(f"   - checkpoint SHA-256: {common.verify_checkpoint_sidecar(paths['best'])}")
        print(f"   - model state SHA-256: {payload['model_state_sha256']}")
        print(f"   - best optimizer step: {payload['best_optimizer_step']}")
        print(f"   - best dev mIoU: {payload['best_dev_metrics']['mIoU']:.6f}")
        print(f"   - shapes: {verified_shape['feature_shapes']}")
        return

    artifact_paths = [paths["best"], paths["last"], paths["history"], paths["metrics"], paths["per_image"]]
    if not args.resume and any(path.exists() for path in artifact_paths):
        raise FileExistsError(
            f"Run artifacts already exist in {run_dir}. Use --resume or another --output-dir."
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
    config = {
        "experiment": EXPERIMENT,
        "seed": args.seed,
        "formal_seeds": list(FORMAL_SEEDS),
        "max_optimizer_steps": args.max_steps,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "accumulation_steps": args.accumulation_steps,
        "global_batch_size": args.batch_size * args.accumulation_steps,
        "num_workers": args.num_workers,
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
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "output_stride": OUTPUT_STRIDE,
        "initialization": "weights=None",
        "backbone_frozen": False,
        "loss": "pixel_cross_entropy_only",
        "knowledge_distillation": False,
        "test_local_evaluated": False,
        "benchmark": args.benchmark,
        "benchmark_resolution": [args.benchmark_height, args.benchmark_width],
        "benchmark_warmup": args.benchmark_warmup,
        "benchmark_runs": args.benchmark_runs,
    }
    hashes = {
        "initial_model_state_sha256": initial_model_hash,
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
        if not paths["last"].is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {paths['last']}")
        resume_payload = common.safe_torch_load(
            paths["last"], map_location="cpu", weights_only=False
        )
        if resume_payload.get("config") != config:
            raise RuntimeError("Resume configuration differs from current S2-0 arguments")
        if (
            resume_payload["dataset_lock"]["combined_manifest_sha256"]
            != dataset_lock["combined_manifest_sha256"]
        ):
            raise RuntimeError("Resume dataset lock differs from the current split")
        resume_hashes = resume_payload.get("hashes", {})
        if resume_hashes.get("initial_model_state_sha256") != initial_model_hash:
            raise RuntimeError("Scratch initialization differs from the resumed S2-0 run")
        if resume_hashes.get("training_script_sha256") != hashes["training_script_sha256"]:
            print("[WARN] Training script SHA-256 differs from the resume checkpoint")
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        scaler.load_state_dict(resume_payload["scaler_state_dict"])
        train_generator.set_state(resume_payload["train_generator_state"])
        restore_rng_state(resume_payload["rng_state"])
        history = resume_payload["history"]
        best_key = resume_payload["best_key"]
        best_epoch = resume_payload["best_epoch"]
        best_optimizer_step = resume_payload["best_optimizer_step"]
        best_dev_metrics = resume_payload["best_dev_metrics"]
        epoch = int(resume_payload["epoch"])
        cumulative_optimizer_steps = int(resume_payload["optimizer_steps"])
        if cumulative_optimizer_steps >= args.max_steps:
            print("[OK] Resume checkpoint already reached the fixed training budget")
        else:
            print(
                f"[OK] Resuming S2-0 after epoch {epoch} at optimizer step "
                f"{cumulative_optimizer_steps:,}"
            )

    training_started = time.time()
    while cumulative_optimizer_steps < args.max_steps:
        epoch += 1
        remaining_steps = args.max_steps - cumulative_optimizer_steps
        train_metrics, optimizer_steps = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            accumulation_steps=args.accumulation_steps,
            epoch=epoch,
            remaining_optimizer_steps=remaining_steps,
        )
        previous_steps = cumulative_optimizer_steps
        cumulative_optimizer_steps += optimizer_steps
        crossed_eval_boundary = (
            previous_steps // args.eval_every_steps
            < cumulative_optimizer_steps // args.eval_every_steps
        )
        should_evaluate = crossed_eval_boundary or cumulative_optimizer_steps == args.max_steps
        dev_metrics = None
        if should_evaluate:
            dev_metrics, _ = common.evaluate(
                model=model,
                loader=dev_loader,
                device=device,
                amp_enabled=amp_enabled,
                split_name="dev_local",
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
                    model=model,
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
                    f"[OK] Best S2-0 updated: step={cumulative_optimizer_steps:,}, "
                    f"dev_mIoU={dev_metrics['mIoU']:.6f}, sha256={checkpoint_hash}"
                )

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
            "model_state_dict": common.cpu_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_generator_state": train_generator.get_state(),
            "rng_state": capture_rng_state(),
            "history": history,
            "best_key": best_key,
            "best_epoch": best_epoch,
            "best_optimizer_step": best_optimizer_step,
            "best_dev_metrics": best_dev_metrics,
            "config": config,
            "hashes": hashes,
            "dataset_lock": dataset_lock,
            "latest_invocation": invocation,
        }
        common.torch_save_atomic(last_payload, paths["last"])
        message = (
            f"Epoch {epoch}: step={cumulative_optimizer_steps:,}/{args.max_steps:,}, "
            f"train_mIoU={train_metrics['mIoU']:.4f}, "
            f"train_loss={train_metrics['loss']:.4f}"
        )
        if dev_metrics is not None:
            message += (
                f", dev_mIoU={dev_metrics['mIoU']:.4f}, "
                f"dev_bF1={dev_metrics['boundary_f1']:.4f}"
            )
        print(message)

    if best_epoch is None or best_optimizer_step is None or best_dev_metrics is None:
        raise RuntimeError("S2-0 ended without a selected dev checkpoint")
    if cumulative_optimizer_steps != args.max_steps:
        raise RuntimeError(
            f"S2-0 budget mismatch: actual={cumulative_optimizer_steps}, expected={args.max_steps}"
        )

    training_elapsed_seconds = time.time() - training_started
    del model, optimizer, scheduler, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    selected_model, selected_payload = load_s2_0_model(paths["best"], device=device)
    selected_dev_metrics, per_image_rows = common.evaluate(
        model=selected_model,
        loader=dev_loader,
        device=device,
        amp_enabled=amp_enabled,
        split_name="selected dev_local",
        boundary_tolerance=args.boundary_tolerance,
        collect_per_image=True,
    )
    if not common.metrics_reproduce(selected_dev_metrics, best_dev_metrics):
        raise RuntimeError(
            "Reloaded S2-0 checkpoint did not reproduce best dev metrics: "
            f"saved={best_dev_metrics['mIoU']}, reloaded={selected_dev_metrics['mIoU']}"
        )
    common.write_jsonl_atomic(paths["per_image"], per_image_rows)
    checkpoint_hash = common.verify_checkpoint_sidecar(paths["best"])
    efficiency = None
    if args.benchmark:
        print("[INFO] Benchmarking selected S2-0 model at batch=1 FP32")
        efficiency = benchmark_model(
            selected_model,
            device,
            args.benchmark_height,
            args.benchmark_width,
            args.benchmark_warmup,
            args.benchmark_runs,
        )

    results = {
        "experiment": EXPERIMENT,
        "invocation": invocation,
        "protocol": (
            "MobileNetV2 weights=None and R-ASPP are trained end to end with CE only "
            "for a fixed optimizer-step budget; best checkpoint is selected by "
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
            "feature_taps": FEATURE_TAPS,
            **parameters,
        },
        "efficiency": efficiency,
        "hashes": {
            **hashes,
            "selected_model_state_sha256": selected_payload["model_state_sha256"],
            "checkpoint_sha256": checkpoint_hash,
        },
        "training": {
            "elapsed_seconds": training_elapsed_seconds,
            "optimizer_steps": cumulative_optimizer_steps,
            "epochs_completed": epoch,
            "steps_per_full_epoch": steps_per_full_epoch,
        },
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "numpy": np.__version__,
            "pillow": __import__("PIL").__version__,
            "platform": platform.platform(),
        },
        "artifacts": {
            "checkpoint": str(paths["best"]),
            "checkpoint_sha256": str(common._checkpoint_sidecar_path(paths["best"])),
            "last_checkpoint": str(paths["last"]),
            "history": str(paths["history"]),
            "dev_per_image_confusion": str(paths["per_image"]),
        },
    }
    common.write_json_atomic(paths["metrics"], results)

    print("\n[DONE] S2-0 MobileNetV2+R-ASPP scratch baseline selected")
    print(f"   - seed: {args.seed}")
    print(f"   - optimizer steps: {cumulative_optimizer_steps:,}")
    print(f"   - best optimizer step: {best_optimizer_step:,}")
    print(f"   - dev mIoU: {selected_dev_metrics['mIoU']:.6f}")
    print(f"   - dev mAcc: {selected_dev_metrics['mAcc']:.6f}")
    print(f"   - dev pixel accuracy: {selected_dev_metrics['pixel_accuracy']:.6f}")
    print(f"   - dev boundary F1: {selected_dev_metrics['boundary_f1']:.6f}")
    print("   - test_local evaluated: False")
    print(f"   - model state SHA-256: {selected_payload['model_state_sha256']}")
    print(f"   - checkpoint SHA-256: {checkpoint_hash}")
    if efficiency is not None:
        print(f"   - parameters: {parameters['total_parameters']:,}")
        print(f"   - MACs: {efficiency['gmacs']:.3f} GMAC")
        print(f"   - FP32 latency: {efficiency['latency_ms_mean']:.3f} ms mean")
        print(f"   - peak CUDA memory: {efficiency['peak_cuda_memory_mib']} MiB")
    print(f"   - metrics: {paths['metrics']}")


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
