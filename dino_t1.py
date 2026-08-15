"""T1 baseline: warm up R-ASPP, then fine-tune the last DINOv3 stage.

This program implements experiment T1 from:

* ``知识蒸馏实验分析与后续实验方向.md``
* ``plan_markdown/Cityscapes知识蒸馏实验详单.md``

The Cityscapes split, transforms, labels, metrics, OS=16 conversion, and
R-ASPP implementation are shared with ``dino.py`` so T0 and T1 differ only in
the intended optimization scope.  T1 trains the head alone during warm-up,
then trains the head plus the final ConvNeXt downsample, final stage, and final
normalization.  ``test_local`` is validated but never evaluated here.

Typical commands:

    python -B dino_t1.py --smoke-test --seed 42
    python -B dino_t1.py --seed 42
    python -B dino_t1.py --seed 3407
    python -B dino_t1.py --seed 260805
    python -B dino_t1.py --seed 42 --resume
    python -B dino_t1.py --seed 42 --verify-only

Future K2/K3 code can import ``load_teacher_for_distillation`` from this file.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Set before CUDA/CUBLAS initialization.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import dino as t0


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "T1_DINOv3_RASPP"

MODEL_NAME = t0.MODEL_NAME
NUM_CLASSES = t0.NUM_CLASSES
IGNORE_INDEX = t0.IGNORE_INDEX
OUTPUT_STRIDE = t0.OUTPUT_STRIDE
CITYSCAPES_CLASSES = t0.CITYSCAPES_CLASSES

ARTIFACT_TYPE = "dinov3_cityscapes19_last_stage_raspp_t1"
ARTIFACT_FORMAT_VERSION = 1
PHASE_HEAD_WARMUP = "head_warmup"
PHASE_LAST_STAGE = "last_stage_finetune"
PHASE_FROZEN = "frozen_for_distillation"

LAST_STAGE_STATE_PREFIXES = (
    "downsample_layers.3.",
    "stages.3.",
    "norm.",
    "norms.3.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure T1: warm up DINOv3+R-ASPP head, then fine-tune only the "
            "last ConvNeXt stage on Cityscapes."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=t0.DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-dir", type=Path, default=t0.DEFAULT_REPO_DIR)
    parser.add_argument("--weights-path", type=Path, default=t0.DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--head-channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--crop-height", type=int, default=512)
    parser.add_argument("--crop-width", type=int, default=1024)
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request deterministic kernels where PyTorch provides them.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the seed run's T1 last checkpoint.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Check warm-up/fine-tune gradients and shapes without updating weights.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the selected T1 artifact and its hashes without training.",
    )
    args = parser.parse_args()

    positive_int_fields = (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "accumulation_steps",
        "head_channels",
        "crop_height",
        "crop_width",
        "eval_every",
    )
    for field in positive_int_fields:
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be at least 1")
    if not 1 <= args.warmup_epochs < args.epochs:
        parser.error("--warmup-epochs must satisfy 1 <= warmup < epochs")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.head_lr <= 0 or args.backbone_lr <= 0:
        parser.error("--head-lr and --backbone-lr must be positive")
    if args.backbone_lr > args.head_lr:
        parser.error("--backbone-lr must not exceed --head-lr")
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
        parser.error(f"Crop dimensions must be divisible by output stride {OUTPUT_STRIDE}")
    if args.boundary_tolerance < 0:
        parser.error("--boundary-tolerance cannot be negative")
    if args.resume and (args.smoke_test or args.verify_only):
        parser.error("--resume cannot be combined with --smoke-test or --verify-only")
    return args


def _unique_parameters(modules: Sequence[nn.Module]) -> List[nn.Parameter]:
    parameters: List[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


def _last_stage_modules(backbone: nn.Module) -> List[nn.Module]:
    if not hasattr(backbone, "downsample_layers") or not hasattr(backbone, "stages"):
        raise RuntimeError("T1 requires the expected DINOv3 ConvNeXt backbone")
    if len(backbone.downsample_layers) != 4 or len(backbone.stages) != 4:
        raise RuntimeError("Unexpected ConvNeXt stage count")
    if not hasattr(backbone, "norm"):
        raise RuntimeError("DINOv3 ConvNeXt final normalization is missing")
    return [backbone.downsample_layers[3], backbone.stages[3], backbone.norm]


def _is_last_stage_state(name: str) -> bool:
    return name.startswith(LAST_STAGE_STATE_PREFIXES)


def _state_subset(
    state_dict: Mapping[str, torch.Tensor], keep_last_stage: bool
) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state_dict.items()
        if _is_last_stage_state(name) == keep_last_stage
    }


def last_stage_state_dict(backbone: nn.Module) -> Dict[str, torch.Tensor]:
    return _state_subset(backbone.state_dict(), keep_last_stage=True)


def frozen_prefix_state_dict(backbone: nn.Module) -> Dict[str, torch.Tensor]:
    return _state_subset(backbone.state_dict(), keep_last_stage=False)


def frozen_prefix_state_from_full(
    state_dict: Mapping[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    return _state_subset(state_dict, keep_last_stage=False)


def load_last_stage_state(
    backbone: nn.Module, saved_state: Mapping[str, torch.Tensor]
) -> None:
    current_state = backbone.state_dict()
    expected_names = {name for name in current_state if _is_last_stage_state(name)}
    actual_names = set(saved_state)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise RuntimeError(
            "T1 last-stage checkpoint keys differ from the model: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    current_state.update(saved_state)
    backbone.load_state_dict(current_state, strict=True)


def phase_for_epoch(epoch: int, warmup_epochs: int) -> str:
    return PHASE_HEAD_WARMUP if epoch <= warmup_epochs else PHASE_LAST_STAGE


class DINOv3RASPPTeacherT1(nn.Module):
    """DINOv3+R-ASPP with an explicit head-only/last-stage phase switch."""

    def __init__(self, backbone: nn.Module, head: t0.RASPPHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self._last_stage_modules = _last_stage_modules(backbone)
        self._last_stage_parameters = _unique_parameters(self._last_stage_modules)
        self._phase = PHASE_HEAD_WARMUP
        self.configure_phase(PHASE_HEAD_WARMUP)

    @property
    def teacher_head(self) -> t0.RASPPHead:
        return self.head

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def last_stage_parameters(self) -> List[nn.Parameter]:
        return list(self._last_stage_parameters)

    def configure_phase(self, phase: str) -> "DINOv3RASPPTeacherT1":
        if phase not in (PHASE_HEAD_WARMUP, PHASE_LAST_STAGE, PHASE_FROZEN):
            raise ValueError(f"Unknown T1 phase: {phase}")
        self.backbone.requires_grad_(False)
        self.head.requires_grad_(phase != PHASE_FROZEN)
        if phase == PHASE_LAST_STAGE:
            for parameter in self._last_stage_parameters:
                parameter.requires_grad_(True)
        self._phase = phase
        if self.training:
            self.train(True)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            return self
        if self._phase == PHASE_FROZEN:
            super().train(False)
            return self
        self.backbone.eval()
        self.head.train(True)
        if self._phase == PHASE_LAST_STAGE:
            for module in self._last_stage_modules:
                module.train(True)
        return self

    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self._phase in (PHASE_HEAD_WARMUP, PHASE_FROZEN):
            with torch.no_grad():
                outputs = self.backbone.get_intermediate_layers(
                    images,
                    n=[0, 1, 2, 3],
                    reshape=True,
                    return_class_token=False,
                    norm=True,
                )
        else:
            outputs = self.backbone.get_intermediate_layers(
                images,
                n=[0, 1, 2, 3],
                reshape=True,
                return_class_token=False,
                norm=True,
            )
        if len(outputs) != 4:
            raise RuntimeError(f"Expected four ConvNeXt stage outputs, got {len(outputs)}")
        return {
            "os4": outputs[0],
            "os8": outputs[1],
            "os16_mid": outputs[2],
            "os16": outputs[3],
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        input_size = images.shape[-2:]
        high_feature = self.extract_features(images)["os16"]
        logits = self.head(high_feature)
        return F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

    def freeze_for_distillation(self) -> "DINOv3RASPPTeacherT1":
        self.configure_phase(PHASE_FROZEN)
        self.eval()
        return self


def build_model(
    backbone: nn.Module, head_channels: int, dropout: float
) -> DINOv3RASPPTeacherT1:
    head = t0.RASPPHead(
        in_channels=768,
        num_classes=NUM_CLASSES,
        inter_channels=head_channels,
        dropout=dropout,
    )
    model = DINOv3RASPPTeacherT1(backbone=backbone, head=head)
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("T1 must begin with the complete backbone frozen")
    if not all(parameter.requires_grad for parameter in model.head.parameters()):
        raise RuntimeError("Every R-ASPP parameter must train during T1 warm-up")
    return model


def _parameter_names(
    module: nn.Module, selected_parameters: Iterable[nn.Parameter]
) -> List[str]:
    selected_ids = {id(parameter) for parameter in selected_parameters}
    return [
        name
        for name, parameter in module.named_parameters()
        if id(parameter) in selected_ids
    ]


def _gradient_l2_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared_norm: Optional[torch.Tensor] = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().pow(2).sum()
        squared_norm = value if squared_norm is None else squared_norm + value
    if squared_norm is None:
        return 0.0
    return float(squared_norm.sqrt().item())


def _group_learning_rates(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    return {
        str(group["group_name"]): float(group["lr"])
        for group in optimizer.param_groups
    }


def train_one_epoch(
    model: DINOv3RASPPTeacherT1,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    accumulation_steps: int,
    epoch: int,
    epochs: int,
) -> Tuple[Dict[str, object], int, Dict[str, float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    loss_sum = 0.0
    valid_pixels = 0
    optimizer_steps = 0
    final_gradient_norms = {"head": 0.0, "last_stage": 0.0}
    group_size = accumulation_steps
    phase_label = "warm-up" if model.phase == PHASE_HEAD_WARMUP else "last-stage"
    progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [T1 {phase_label}]")

    for batch_index, (images, targets, _) in enumerate(progress):
        group_position = batch_index % accumulation_steps
        if group_position == 0:
            group_size = min(accumulation_steps, len(loader) - batch_index)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with t0.autocast_context(device, amp_enabled):
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

        is_group_end = group_position + 1 == group_size
        if is_group_end:
            is_final_update = batch_index + 1 == len(loader)
            if is_final_update:
                scaler.unscale_(optimizer)
                final_gradient_norms = {
                    "head": _gradient_l2_norm(model.head.parameters()),
                    "last_stage": _gradient_l2_norm(model.last_stage_parameters),
                }
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1

        predictions = logits_float.detach().argmax(dim=1)
        confusion += t0.confusion_counts(predictions, targets)
        loss_sum += float(batch_loss_sum.detach().item())
        valid_pixels += batch_valid
        running = t0.metrics_from_confusion(confusion, loss_sum, valid_pixels)
        learning_rates = _group_learning_rates(optimizer)
        progress.set_postfix(
            {
                "loss": f"{running['loss']:.4f}",
                "mIoU": f"{running['mIoU']:.4f}",
                "head_lr": f"{learning_rates['head']:.2e}",
                "stage_lr": f"{learning_rates['last_stage']:.2e}",
            }
        )

    return (
        t0.metrics_from_confusion(confusion, loss_sum, valid_pixels),
        optimizer_steps,
        final_gradient_norms,
    )


def build_best_checkpoint(
    model: DINOv3RASPPTeacherT1,
    epoch: int,
    dev_metrics: Mapping[str, object],
    config: Mapping[str, object],
    hashes: Mapping[str, object],
    dataset_lock: Mapping[str, object],
    shape_audit: Mapping[str, object],
) -> Dict[str, object]:
    backbone_state = t0.cpu_state_dict(model.backbone)
    head_state = t0.cpu_state_dict(model.head)
    stage_state = _state_subset(backbone_state, keep_last_stage=True)
    prefix_state = frozen_prefix_state_from_full(backbone_state)
    prefix_hash = t0.state_dict_sha256(prefix_state)
    if prefix_hash != hashes["initial_frozen_prefix_state_sha256"]:
        raise RuntimeError("Frozen DINOv3 prefix changed while building T1 artifact")
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "frozen_for_distillation": True,
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "class_names": list(CITYSCAPES_CLASSES),
        "output_stride": OUTPUT_STRIDE,
        "head_type": "R-ASPP",
        "fine_tuned_scope": list(LAST_STAGE_STATE_PREFIXES),
        "training_phase": PHASE_LAST_STAGE,
        "backbone_state_dict": backbone_state,
        "backbone_state_sha256": t0.state_dict_sha256(backbone_state),
        "last_stage_state_sha256": t0.state_dict_sha256(stage_state),
        "head_state_dict": head_state,
        "head_state_sha256": t0.state_dict_sha256(head_state),
        "best_epoch": epoch,
        "best_dev_metrics": copy.deepcopy(dev_metrics),
        "config": copy.deepcopy(config),
        "hashes": copy.deepcopy(hashes),
        "dataset_lock": copy.deepcopy(dataset_lock),
        "shape_audit": copy.deepcopy(shape_audit),
    }


def load_teacher_for_distillation(
    checkpoint_path: Path,
    repo_dir: Path = t0.DEFAULT_REPO_DIR,
    weights_path: Path = t0.DEFAULT_WEIGHTS_PATH,
    device: object = None,
    verify_checkpoint_file: bool = True,
) -> Tuple[DINOv3RASPPTeacherT1, Dict[str, object]]:
    checkpoint_path = Path(checkpoint_path).resolve()
    if verify_checkpoint_file:
        t0.verify_checkpoint_sidecar(checkpoint_path)
    payload = t0.safe_torch_load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    required = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "frozen_for_distillation": True,
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "output_stride": OUTPUT_STRIDE,
        "head_type": "R-ASPP",
        "training_phase": PHASE_LAST_STAGE,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"Incompatible T1 artifact field {key}: "
                f"actual={payload.get(key)!r}, expected={expected!r}"
            )
    if payload.get("class_names") != CITYSCAPES_CLASSES:
        raise RuntimeError("T1 artifact Cityscapes class mapping is incompatible")
    if payload.get("fine_tuned_scope") != list(LAST_STAGE_STATE_PREFIXES):
        raise RuntimeError("T1 artifact fine-tuned scope is incompatible")

    weights_path = Path(weights_path).resolve()
    weights_hash = t0.sha256_file(weights_path)
    if weights_hash != payload["hashes"]["backbone_weights_sha256"]:
        raise RuntimeError("Current DINOv3 weights differ from the T1 artifact")
    backbone = t0.load_backbone(repo_dir, weights_path)
    initial_hash = t0.state_dict_sha256(backbone.state_dict())
    if initial_hash != payload["hashes"]["initial_converted_backbone_state_sha256"]:
        raise RuntimeError("Initial OS=16 DINOv3 backbone differs from the T1 artifact")

    backbone_state = payload["backbone_state_dict"]
    actual_backbone_hash = t0.state_dict_sha256(backbone_state)
    if actual_backbone_hash != payload.get("backbone_state_sha256"):
        raise RuntimeError("T1 fine-tuned backbone state hash mismatch")
    prefix_hash = t0.state_dict_sha256(
        frozen_prefix_state_from_full(backbone_state)
    )
    if prefix_hash != payload["hashes"]["initial_frozen_prefix_state_sha256"]:
        raise RuntimeError("T1 artifact modified parameters outside the final stage")
    stage_hash = t0.state_dict_sha256(
        _state_subset(backbone_state, keep_last_stage=True)
    )
    if stage_hash != payload.get("last_stage_state_sha256"):
        raise RuntimeError("T1 final-stage state hash mismatch")
    backbone.load_state_dict(backbone_state, strict=True)

    head_state = payload["head_state_dict"]
    actual_head_hash = t0.state_dict_sha256(head_state)
    if actual_head_hash != payload.get("head_state_sha256"):
        raise RuntimeError("T1 R-ASPP state hash mismatch")
    config = payload["config"]
    head = t0.RASPPHead(
        in_channels=768,
        num_classes=NUM_CLASSES,
        inter_channels=int(config["head_channels"]),
        dropout=float(config["dropout"]),
    )
    head.load_state_dict(head_state, strict=True)
    teacher = DINOv3RASPPTeacherT1(backbone, head).freeze_for_distillation()

    if device is None:
        target_device = t0.resolve_device("auto")
    elif isinstance(device, str):
        target_device = t0.resolve_device(device)
    else:
        target_device = torch.device(device)
    teacher = teacher.to(target_device).freeze_for_distillation()
    return teacher, payload


def run_smoke_test(
    model: DINOv3RASPPTeacherT1,
    train_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> None:
    images, targets, _ = next(iter(train_loader))
    images = images.to(device)
    targets = targets.to(device)
    stage_ids = {id(parameter) for parameter in model.last_stage_parameters}
    frozen_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if id(parameter) not in stage_ids
    ]
    initial_prefix_hash = t0.state_dict_sha256(
        frozen_prefix_state_dict(model.backbone)
    )
    initial_stage_hash = t0.state_dict_sha256(last_stage_state_dict(model.backbone))
    initial_head_hash = t0.state_dict_sha256(model.head.state_dict())
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.head.parameters()), "lr": 1e-3},
            {"params": model.last_stage_parameters, "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )

    model.configure_phase(PHASE_HEAD_WARMUP).train()
    optimizer.zero_grad(set_to_none=True)
    with t0.autocast_context(device, amp_enabled):
        warmup_logits = model(images)
        warmup_loss = F.cross_entropy(
            warmup_logits.float(), targets, ignore_index=IGNORE_INDEX
        )
    warmup_loss.backward()
    if not any(parameter.grad is not None for parameter in model.head.parameters()):
        raise RuntimeError("T1 smoke test: R-ASPP received no warm-up gradient")
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise RuntimeError("T1 smoke test: backbone received a warm-up gradient")
    optimizer.step()
    if t0.state_dict_sha256(last_stage_state_dict(model.backbone)) != initial_stage_hash:
        raise RuntimeError("T1 smoke test: final stage changed during head warm-up")
    if t0.state_dict_sha256(frozen_prefix_state_dict(model.backbone)) != initial_prefix_hash:
        raise RuntimeError("T1 smoke test: frozen prefix changed during head warm-up")
    if t0.state_dict_sha256(model.head.state_dict()) == initial_head_hash:
        raise RuntimeError("T1 smoke test: R-ASPP did not update during head warm-up")

    model.configure_phase(PHASE_LAST_STAGE).train()
    optimizer.zero_grad(set_to_none=True)
    with t0.autocast_context(device, amp_enabled):
        fine_tune_logits = model(images)
        fine_tune_loss = F.cross_entropy(
            fine_tune_logits.float(), targets, ignore_index=IGNORE_INDEX
        )
    fine_tune_loss.backward()
    if not any(parameter.grad is not None for parameter in model.head.parameters()):
        raise RuntimeError("T1 smoke test: R-ASPP received no fine-tune gradient")
    if not any(parameter.grad is not None for parameter in model.last_stage_parameters):
        raise RuntimeError("T1 smoke test: final stage received no gradient")
    if any(parameter.grad is not None for parameter in frozen_parameters):
        raise RuntimeError("T1 smoke test: frozen ConvNeXt prefix received a gradient")
    optimizer.step()
    if t0.state_dict_sha256(last_stage_state_dict(model.backbone)) == initial_stage_hash:
        raise RuntimeError("T1 smoke test: final stage did not update after unfreezing")
    if t0.state_dict_sha256(frozen_prefix_state_dict(model.backbone)) != initial_prefix_hash:
        raise RuntimeError("T1 smoke test: frozen prefix changed during fine-tuning")
    optimizer.zero_grad(set_to_none=True)
    smoke_payload = build_best_checkpoint(
        model=model,
        epoch=2,
        dev_metrics={"mIoU": 0.0},
        config={"head_channels": model.head.project[0].out_channels, "dropout": 0.1},
        hashes={"initial_frozen_prefix_state_sha256": initial_prefix_hash},
        dataset_lock={},
        shape_audit={},
    )
    artifact_buffer = io.BytesIO()
    torch.save(smoke_payload, artifact_buffer)
    artifact_buffer.seek(0)
    restored_payload = torch.load(
        artifact_buffer, map_location="cpu", weights_only=True
    )
    if (
        t0.state_dict_sha256(restored_payload["backbone_state_dict"])
        != restored_payload["backbone_state_sha256"]
    ):
        raise RuntimeError("T1 smoke test: backbone artifact round-trip failed")
    if (
        t0.state_dict_sha256(restored_payload["head_state_dict"])
        != restored_payload["head_state_sha256"]
    ):
        raise RuntimeError("T1 smoke test: R-ASPP artifact round-trip failed")
    model.configure_phase(PHASE_HEAD_WARMUP).eval()
    print("[OK] T1 smoke test passed")
    print(f"   - warm-up loss: {float(warmup_loss.item()):.6f}")
    print(f"   - fine-tune loss: {float(fine_tune_loss.item()):.6f}")
    print("   - warm-up gradients: R-ASPP only")
    print("   - fine-tune gradients: R-ASPP + final ConvNeXt stage only")
    print("   - optimizer updates respect both phase boundaries")
    print("   - full T1 teacher artifact round-trip passed")


def run_training(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    weights_path = args.weights_path.resolve()
    device = t0.resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    t0.set_global_seed(args.seed, args.deterministic)

    dataset_lock, entries_by_split = t0.validate_dataset_lock(dataset_root)
    print(
        "[OK] Locked Cityscapes splits: "
        f"train={len(entries_by_split['train_local'])}, "
        f"dev={len(entries_by_split['dev_local'])}, "
        f"test={len(entries_by_split['test_local'])} (not evaluated)"
    )

    weights_hash = t0.sha256_file(weights_path)
    backbone = t0.load_backbone(args.repo_dir, weights_path)
    initial_backbone_hash = t0.state_dict_sha256(backbone.state_dict())
    initial_prefix_hash = t0.state_dict_sha256(frozen_prefix_state_dict(backbone))
    model = build_model(backbone, args.head_channels, args.dropout).to(device)
    initial_head_hash = t0.state_dict_sha256(model.head.state_dict())
    shape_audit = t0.audit_model_shapes(
        model,
        device,
        args.crop_height,
        args.crop_width,
        amp_enabled,
    )
    train_loader, dev_loader, train_generator = t0.make_data_loaders(
        dataset_root, entries_by_split, args, device
    )

    stage_parameters = model.last_stage_parameters
    stage_parameter_ids = {id(parameter) for parameter in stage_parameters}
    frozen_prefix_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if id(parameter) not in stage_parameter_ids
    ]
    stage_parameter_names = _parameter_names(model.backbone, stage_parameters)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    backbone_parameters = sum(parameter.numel() for parameter in model.backbone.parameters())
    head_parameters = sum(parameter.numel() for parameter in model.head.parameters())
    stage_parameter_count = sum(parameter.numel() for parameter in stage_parameters)
    frozen_prefix_parameter_count = sum(
        parameter.numel() for parameter in frozen_prefix_parameters
    )
    if backbone_parameters != stage_parameter_count + frozen_prefix_parameter_count:
        raise RuntimeError("T1 parameter partition does not cover the backbone exactly")

    print(f"[INFO] Device={device}; AMP={amp_enabled}")
    print(
        f"[OK] T1 params: frozen prefix={frozen_prefix_parameter_count:,}; "
        f"fine-tuned final stage={stage_parameter_count:,}; "
        f"trainable R-ASPP={head_parameters:,}"
    )
    print(
        f"[OK] Phase schedule: epochs 1-{args.warmup_epochs} head-only; "
        f"epochs {args.warmup_epochs + 1}-{args.epochs} head+last-stage"
    )
    print(f"[OK] Feature shapes: {shape_audit['feature_shapes']}")

    if args.smoke_test:
        run_smoke_test(model, train_loader, device, amp_enabled)
        return

    run_dir = args.output_dir.resolve() / f"seed_{args.seed}"
    best_checkpoint_path = run_dir / "t1_dinov3_raspp_teacher.pth"
    last_checkpoint_path = run_dir / "t1_last_checkpoint.pth"
    history_path = run_dir / "training_history.json"
    results_path = run_dir / "t1_metrics.json"
    per_image_path = run_dir / "dev_per_image_confusion.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        teacher, payload = load_teacher_for_distillation(
            best_checkpoint_path,
            repo_dir=args.repo_dir,
            weights_path=weights_path,
            device=device,
        )
        if (
            payload["dataset_lock"]["combined_manifest_sha256"]
            != dataset_lock["combined_manifest_sha256"]
        ):
            raise RuntimeError("Current locked split differs from the T1 artifact")
        verified_shape = t0.audit_model_shapes(
            teacher,
            device,
            args.crop_height,
            args.crop_width,
            amp_enabled,
        )
        del teacher
        print("[OK] T1 artifact verified")
        print(f"   - checkpoint: {best_checkpoint_path}")
        print(f"   - checkpoint SHA-256: {t0.verify_checkpoint_sidecar(best_checkpoint_path)}")
        print(f"   - backbone SHA-256: {payload['backbone_state_sha256']}")
        print(f"   - R-ASPP SHA-256: {payload['head_state_sha256']}")
        print(f"   - best epoch: {payload['best_epoch']}")
        print(f"   - best dev mIoU: {payload['best_dev_metrics']['mIoU']:.6f}")
        print(f"   - shapes: {verified_shape['feature_shapes']}")
        return

    if not args.resume and any(
        path.exists()
        for path in (
            best_checkpoint_path,
            last_checkpoint_path,
            history_path,
            results_path,
            per_image_path,
        )
    ):
        raise FileExistsError(
            f"T1 run artifacts already exist in {run_dir}. "
            "Use --resume or a different --output-dir."
        )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.head.parameters()),
                "lr": args.head_lr,
                "weight_decay": args.weight_decay,
                "group_name": "head",
            },
            {
                "params": stage_parameters,
                "lr": args.backbone_lr,
                "weight_decay": args.weight_decay,
                "group_name": "last_stage",
            },
        ]
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    total_optimizer_steps = steps_per_epoch * args.epochs

    def lr_factor(step: int) -> float:
        progress = min(step, total_optimizer_steps) / max(total_optimizer_steps, 1)
        return max((1.0 - progress) ** args.poly_power, args.min_lr_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    config = {
        "experiment": "T1",
        "seed": args.seed,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "selection_starts_epoch": args.warmup_epochs + 1,
        "phase_schedule": {
            PHASE_HEAD_WARMUP: [1, args.warmup_epochs],
            PHASE_LAST_STAGE: [args.warmup_epochs + 1, args.epochs],
        },
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "accumulation_steps": args.accumulation_steps,
        "global_batch_size": args.batch_size * args.accumulation_steps,
        "num_workers": args.num_workers,
        "optimizer": "AdamW",
        "head_learning_rate": args.head_lr,
        "backbone_learning_rate": args.backbone_lr,
        "weight_decay": args.weight_decay,
        "scheduler": "polynomial",
        "poly_power": args.poly_power,
        "min_lr_ratio": args.min_lr_ratio,
        "eval_every": args.eval_every,
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
        "fine_tuned_scope": list(LAST_STAGE_STATE_PREFIXES),
        "fine_tuned_parameter_names": stage_parameter_names,
        "initialization": "DINOv3 pretrained backbone + seeded random R-ASPP",
        "test_local_evaluated": False,
    }
    hashes = {
        "backbone_weights_sha256": weights_hash,
        "initial_converted_backbone_state_sha256": initial_backbone_hash,
        "initial_frozen_prefix_state_sha256": initial_prefix_hash,
        "initial_head_state_sha256": initial_head_hash,
        "training_script_sha256": t0.sha256_file(Path(__file__).resolve()),
        "shared_dino_module_sha256": t0.sha256_file(Path(t0.__file__).resolve()),
    }

    history: List[Dict[str, object]] = []
    best_key: Optional[Tuple[float, float, float, float]] = None
    best_epoch: Optional[int] = None
    best_dev_metrics: Optional[Dict[str, object]] = None
    start_epoch = 1
    cumulative_optimizer_steps = 0

    if args.resume:
        if not last_checkpoint_path.is_file():
            raise FileNotFoundError(f"T1 resume checkpoint not found: {last_checkpoint_path}")
        resume_payload = t0.safe_torch_load(
            last_checkpoint_path, map_location="cpu", weights_only=False
        )
        if resume_payload.get("config") != config:
            raise RuntimeError("Resume configuration differs from the current T1 arguments")
        resume_hashes = resume_payload.get("hashes", {})
        strict_hash_names = (
            "backbone_weights_sha256",
            "initial_converted_backbone_state_sha256",
            "initial_frozen_prefix_state_sha256",
            "initial_head_state_sha256",
            "shared_dino_module_sha256",
        )
        for hash_name in strict_hash_names:
            if resume_hashes.get(hash_name) != hashes[hash_name]:
                raise RuntimeError(f"Resume {hash_name} differs from the current T1 run")
        if resume_hashes.get("training_script_sha256") != hashes["training_script_sha256"]:
            print(
                "[WARN] T1 training script SHA-256 differs from the resume checkpoint; "
                "all model/data invariants still match."
            )
        if (
            resume_payload["dataset_lock"]["combined_manifest_sha256"]
            != dataset_lock["combined_manifest_sha256"]
        ):
            raise RuntimeError("Resume dataset lock differs from the current split")
        load_last_stage_state(
            model.backbone, resume_payload["last_stage_state_dict"]
        )
        model.head.load_state_dict(resume_payload["head_state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        scaler.load_state_dict(resume_payload["scaler_state_dict"])
        train_generator.set_state(resume_payload["train_generator_state"])
        history = resume_payload["history"]
        best_key = resume_payload["best_key"]
        best_epoch = resume_payload["best_epoch"]
        best_dev_metrics = resume_payload["best_dev_metrics"]
        cumulative_optimizer_steps = int(resume_payload["optimizer_steps"])
        start_epoch = int(resume_payload["epoch"]) + 1
        expected_saved_phase = phase_for_epoch(
            int(resume_payload["epoch"]), args.warmup_epochs
        )
        if resume_payload.get("phase") != expected_saved_phase:
            raise RuntimeError("Resume checkpoint phase is inconsistent with its epoch")
        if t0.state_dict_sha256(frozen_prefix_state_dict(model.backbone)) != initial_prefix_hash:
            raise RuntimeError("Resume checkpoint changed the frozen backbone prefix")
        print(f"[OK] Resuming T1 at epoch {start_epoch}")

    training_started = time.time()
    previous_phase: Optional[str] = None
    for epoch in range(start_epoch, args.epochs + 1):
        phase = phase_for_epoch(epoch, args.warmup_epochs)
        model.configure_phase(phase)
        if phase != previous_phase:
            if phase == PHASE_HEAD_WARMUP:
                print("[INFO] T1 phase: R-ASPP head warm-up; complete backbone frozen")
            else:
                print(
                    "[INFO] T1 phase: R-ASPP + final ConvNeXt downsample/stage/norm"
                )
            previous_phase = phase

        train_metrics, optimizer_steps, gradient_norms = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            accumulation_steps=args.accumulation_steps,
            epoch=epoch,
            epochs=args.epochs,
        )
        cumulative_optimizer_steps += optimizer_steps
        should_evaluate = epoch % args.eval_every == 0 or epoch == args.epochs
        dev_metrics = None
        eligible_for_selection = phase == PHASE_LAST_STAGE
        if should_evaluate:
            dev_metrics, _ = t0.evaluate(
                model=model,
                loader=dev_loader,
                device=device,
                amp_enabled=amp_enabled,
                split_name="dev_local",
                boundary_tolerance=args.boundary_tolerance,
                collect_per_image=False,
            )
            if eligible_for_selection:
                candidate_key = (
                    float(dev_metrics["mIoU"]),
                    float(dev_metrics["mAcc"]),
                    float(dev_metrics["pixel_accuracy"]),
                    -float(dev_metrics["loss"]),
                )
                if best_key is None or candidate_key > best_key:
                    best_key = candidate_key
                    best_epoch = epoch
                    best_dev_metrics = copy.deepcopy(dev_metrics)
                    best_payload = build_best_checkpoint(
                        model=model,
                        epoch=epoch,
                        dev_metrics=dev_metrics,
                        config=config,
                        hashes=hashes,
                        dataset_lock=dataset_lock,
                        shape_audit=shape_audit,
                    )
                    checkpoint_hash = t0.write_checkpoint_with_sidecar(
                        best_payload, best_checkpoint_path
                    )
                    print(
                        f"[OK] Best T1 updated: epoch={epoch}, "
                        f"dev_mIoU={dev_metrics['mIoU']:.6f}, "
                        f"sha256={checkpoint_hash}"
                    )
            else:
                print(
                    "[INFO] Warm-up dev metrics are diagnostic only and cannot "
                    "select the T1 teacher."
                )

        learning_rates = _group_learning_rates(optimizer)
        epoch_record = {
            "epoch": epoch,
            "phase": phase,
            "eligible_for_selection": eligible_for_selection,
            "optimizer_steps": cumulative_optimizer_steps,
            "learning_rates": learning_rates,
            "gradient_l2_norm": gradient_norms,
            "train": train_metrics,
            "dev": dev_metrics,
        }
        history.append(epoch_record)
        t0.write_json_atomic(history_path, history)
        current_stage_state = last_stage_state_dict(model.backbone)
        last_payload = {
            "epoch": epoch,
            "phase": phase,
            "last_stage_state_dict": current_stage_state,
            "last_stage_state_sha256": t0.state_dict_sha256(current_stage_state),
            "head_state_dict": t0.cpu_state_dict(model.head),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_generator_state": train_generator.get_state(),
            "history": history,
            "best_key": best_key,
            "best_epoch": best_epoch,
            "best_dev_metrics": best_dev_metrics,
            "optimizer_steps": cumulative_optimizer_steps,
            "config": config,
            "hashes": hashes,
            "dataset_lock": dataset_lock,
        }
        t0.torch_save_atomic(last_payload, last_checkpoint_path)
        message = (
            f"Epoch {epoch}: phase={phase}, "
            f"train_mIoU={train_metrics['mIoU']:.4f}, "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"head_lr={learning_rates['head']:.2e}, "
            f"stage_lr={learning_rates['last_stage']:.2e}"
        )
        if dev_metrics is not None:
            message += (
                f", dev_mIoU={dev_metrics['mIoU']:.4f}, "
                f"dev_bF1={dev_metrics['boundary_f1']:.4f}"
            )
        print(message)

    if best_epoch is None or best_dev_metrics is None:
        raise RuntimeError("T1 ended without a post-warm-up dev checkpoint")
    if t0.state_dict_sha256(frozen_prefix_state_dict(model.backbone)) != initial_prefix_hash:
        raise RuntimeError("Frozen DINOv3 prefix changed during T1 training")

    selected_payload = t0.safe_torch_load(
        best_checkpoint_path, map_location="cpu", weights_only=True
    )
    if (
        t0.state_dict_sha256(selected_payload["backbone_state_dict"])
        != selected_payload["backbone_state_sha256"]
    ):
        raise RuntimeError("Selected T1 backbone failed SHA-256 verification")
    if (
        t0.state_dict_sha256(selected_payload["head_state_dict"])
        != selected_payload["head_state_sha256"]
    ):
        raise RuntimeError("Selected T1 R-ASPP failed SHA-256 verification")
    model.backbone.load_state_dict(
        selected_payload["backbone_state_dict"], strict=True
    )
    model.head.load_state_dict(selected_payload["head_state_dict"], strict=True)
    model.freeze_for_distillation()
    selected_dev_metrics, per_image_rows = t0.evaluate(
        model=model,
        loader=dev_loader,
        device=device,
        amp_enabled=amp_enabled,
        split_name="selected dev_local",
        boundary_tolerance=args.boundary_tolerance,
        collect_per_image=True,
    )
    if not t0.metrics_reproduce(selected_dev_metrics, best_dev_metrics):
        raise RuntimeError(
            "Reloaded T1 checkpoint did not reproduce best dev metrics: "
            f"saved={best_dev_metrics['mIoU']}, "
            f"reloaded={selected_dev_metrics['mIoU']}"
        )
    t0.write_jsonl_atomic(per_image_path, per_image_rows)
    checkpoint_hash = t0.verify_checkpoint_sidecar(best_checkpoint_path)

    results = {
        "experiment": "T1",
        "protocol": (
            "DINOv3 ConvNeXt-T starts frozen while R-ASPP warms up; then only "
            "the final downsample, final stage, final norm, and R-ASPP train. "
            "Best checkpoint is selected by post-warm-up dev_local mIoU; "
            "test_local is not evaluated."
        ),
        "best_epoch": best_epoch,
        "best_dev_metrics": selected_dev_metrics,
        "class_names": CITYSCAPES_CLASSES,
        "config": config,
        "shape_audit": shape_audit,
        "dataset_lock": dataset_lock,
        "model": {
            "model_name": MODEL_NAME,
            "head": "R-ASPP",
            "total_parameters": total_parameters,
            "backbone_parameters": backbone_parameters,
            "head_parameters": head_parameters,
            "fine_tuned_last_stage_parameters": stage_parameter_count,
            "frozen_prefix_parameters": frozen_prefix_parameter_count,
            "inference_parameters": total_parameters,
            "frozen_for_distillation": True,
        },
        "hashes": {
            **hashes,
            "selected_backbone_state_sha256": selected_payload[
                "backbone_state_sha256"
            ],
            "selected_last_stage_state_sha256": selected_payload[
                "last_stage_state_sha256"
            ],
            "selected_head_state_sha256": selected_payload["head_state_sha256"],
            "checkpoint_sha256": checkpoint_hash,
        },
        "training": {
            "elapsed_seconds": time.time() - training_started,
            "optimizer_steps": cumulative_optimizer_steps,
        },
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pillow": __import__("PIL").__version__,
            "platform": platform.platform(),
        },
        "artifacts": {
            "checkpoint": str(best_checkpoint_path),
            "checkpoint_sha256": str(t0._checkpoint_sidecar_path(best_checkpoint_path)),
            "last_checkpoint": str(last_checkpoint_path),
            "history": str(history_path),
            "dev_per_image_confusion": str(per_image_path),
        },
    }
    t0.write_json_atomic(results_path, results)

    print("\n[DONE] T1 DINOv3+R-ASPP teacher selected and frozen")
    print(f"   - seed: {args.seed}")
    print(f"   - warm-up epochs: {args.warmup_epochs}")
    print(f"   - best epoch: {best_epoch}")
    print(f"   - dev mIoU: {selected_dev_metrics['mIoU']:.6f}")
    print(f"   - dev mAcc: {selected_dev_metrics['mAcc']:.6f}")
    print(f"   - dev pixel accuracy: {selected_dev_metrics['pixel_accuracy']:.6f}")
    print(f"   - dev boundary F1@{args.boundary_tolerance}px: {selected_dev_metrics['boundary_f1']:.6f}")
    print("   - test_local evaluated: False")
    print(f"   - backbone SHA-256: {selected_payload['backbone_state_sha256']}")
    print(f"   - R-ASPP SHA-256: {selected_payload['head_state_sha256']}")
    print(f"   - checkpoint SHA-256: {checkpoint_hash}")
    print(f"   - metrics: {results_path}")


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
