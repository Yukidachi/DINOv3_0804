"""A2 server entry point: trainable PCA-initialized teacher adapters.

This entry point keeps A0/A1's locked data, PCA artifacts, student
initialization, feature taps, probe, DDP and ordered-shutdown protocol.  The
single scientific change is the teacher-side projection used during the
label-free feature-pretraining stage:

    A1 fixed PCA-Conv -> trainable PCA-Conv, initialized from A1

The adapter uses ``0.1 * student_lr`` and the registered anchor penalty from
``plan_markdown/A实验的具体实施方案.md`` section 6.  The adapter is reset to
its PCA initialization and frozen before the probe stage; it is a training-
only target-side module and is not part of the deployable student.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import dino as t0
import dino_a0_server as a0
import dino_a1_server as a1
import dino_s2_0 as base
import dino_s2_0_server as s2_0_server


EXPERIMENT = "A2"
ARTIFACT_TYPE_PRETRAIN = "a2_pretrain_mobilenetv2_backbone_trainable_pca_conv"
ARTIFACT_TYPE_PROBE = "a2_probe_mobilenetv2_raspp_trainable_pca_conv"
ARTIFACT_FORMAT_VERSION = 1

ADAPTER_LR_RATIO = 0.1
ADAPTER_ANCHOR_LAMBDA = 0.01
ADAPTER_ANCHOR_EPS = 1e-12

DEFAULT_OUTPUT_DIR = a0.DEFAULT_OUTPUT_DIR
DEFAULT_PCA_DIR = a0.DEFAULT_PCA_DIR
DEFAULT_TEACHER_CHECKPOINT = a0.DEFAULT_TEACHER_CHECKPOINT

_EQUIVALENCE_REPORT: Optional[Dict[str, Dict[str, object]]] = None
_A0_BUILD_CONFIG = a0.build_config
_A0_BUILD_PROBE_BEST_CHECKPOINT = a0.build_probe_best_checkpoint
_A0_RUN_PROBE_STAGE = a0.run_probe_stage


class TrainablePCAConv(nn.Module):
    """A1's fused PCA-Conv with a fixed copy for the A2 anchor penalty."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.shape[0]:
            raise RuntimeError("A2 adapter parameters have inconsistent shapes")
        weight = weight.detach().float().contiguous()
        bias = bias.detach().float().contiguous()
        self.weight = nn.Parameter(weight[:, :, None, None].clone())
        self.bias = nn.Parameter(bias.clone())
        self.register_buffer("weight_initial", self.weight.detach().clone())
        self.register_buffer("bias_initial", self.bias.detach().clone())
        self.c_in = int(weight.shape[1])
        self.d_out = int(weight.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.c_in:
            raise RuntimeError(
                f"A2 adapter expects [B,{self.c_in},H,W], got {tuple(x.shape)}"
            )
        return F.conv2d(x, self.weight, self.bias)

    def parameter_sha256(self) -> str:
        return a0.numpy_arrays_sha256(
            self.weight.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy(),
        )

    def initial_parameter_sha256(self) -> str:
        return a0.numpy_arrays_sha256(
            self.weight_initial.detach().cpu().numpy(),
            self.bias_initial.detach().cpu().numpy(),
        )

    @torch.no_grad()
    def reset_to_initial_and_freeze(self) -> None:
        self.weight.copy_(self.weight_initial)
        self.bias.copy_(self.bias_initial)
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)


def build_projection_bundle(
    scalers: Mapping[str, Mapping[str, np.ndarray]],
    pcas: Mapping[str, Mapping[str, np.ndarray]],
) -> nn.ModuleDict:
    """Build A2 adapters from the exact A1 fused coefficients."""

    projections: Dict[str, TrainablePCAConv] = {}
    for layer in a0.A0_LAYER_ORDER:
        reference = a0.FixedPCAProjection(
            scaler_mean=scalers[layer]["mean_"],
            scaler_scale=scalers[layer]["scale_"],
            pca_mean=pcas[layer]["mean_"],
            components=pcas[layer]["components_"],
        )
        weight, bias = reference.fused_conv_parameters()
        projection = TrainablePCAConv(weight, bias)
        if (
            projection.c_in != a0.TEACHER_CHANNELS[layer]
            or projection.d_out != a0.STUDENT_CHANNELS[layer]
        ):
            raise RuntimeError(
                f"A2 projection contract mismatch for {layer}: "
                f"got d_out={projection.d_out}, c_in={projection.c_in}, "
                f"expected d_out={a0.STUDENT_CHANNELS[layer]}, "
                f"c_in={a0.TEACHER_CHANNELS[layer]}"
            )
        projections[layer] = projection
    return nn.ModuleDict(projections)


def _compare_projection(
    reference: nn.Module,
    candidate: nn.Module,
    sample: torch.Tensor,
) -> Dict[str, object]:
    return a1._compare_projection(reference, candidate, sample)


def build_equivalence_report(
    args,
    scalers: Mapping[str, Mapping[str, np.ndarray]],
    pcas: Mapping[str, Mapping[str, np.ndarray]],
    teacher: torch.nn.Module,
    dataset_root: Path,
    entries: Sequence[Tuple[str, str]],
    device: torch.device,
) -> Dict[str, Dict[str, object]]:
    """Run A1's random/real feature gate on A2's initial adapter."""

    report = a1.build_equivalence_report(
        args,
        scalers,
        pcas,
        teacher,
        dataset_root,
        entries,
        device,
    )
    for layer in a0.A0_LAYER_ORDER:
        report[layer]["candidate"] = (
            "A2 trainable Conv2d(C_t,C_s,1), initialized from A1 fused PCA-Conv"
        )
        report[layer]["trainable_after_preflight"] = True
        report[layer]["adapter_initial_sha256"] = report[layer]["weight_sha256"]
    return report


def check_projection_conv_equivalence(
    reference: a0.FixedPCAProjection,
    sample: torch.Tensor,
) -> Dict[str, object]:
    """Return the preflight report, or run the local mathematical check."""

    if _EQUIVALENCE_REPORT is not None:
        layer = next(
            (
                candidate
                for candidate in a0.A0_LAYER_ORDER
                if a0.TEACHER_CHANNELS[candidate] == reference.c_in
                and a0.STUDENT_CHANNELS[candidate] == reference.d_out
            ),
            None,
        )
        if layer is not None:
            return _EQUIVALENCE_REPORT[layer]
    weight, bias = reference.fused_conv_parameters()
    candidate = TrainablePCAConv(weight, bias)
    result = _compare_projection(reference, candidate, sample)
    result["candidate"] = "A2 trainable PCA-Conv at its initial state"
    result["weight_sha256"] = candidate.parameter_sha256()
    return result


def a2_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    run_dir = output_dir.resolve() / "A2" / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "config": run_dir / "config.json",
        "feature_taps": run_dir / "feature_taps.json",
        "pretrain_last": run_dir / "a2_pretrain_last.pth",
        "pretrain_history": run_dir / "a2_pretrain_history.json",
        "pretrain_gradients": run_dir / "a2_pretrain_gradient_norms.jsonl",
        "pretrain_snapshots": run_dir / "pretrain_snapshots",
        "probe_last": run_dir / "a2_probe_last.pth",
        "probe_history": run_dir / "a2_probe_history.json",
        "best_probe": run_dir / "a2_probe_mobilenetv2_raspp_best.pth",
        "dev_metrics": run_dir / "a2_dev_metrics.json",
        "efficiency": run_dir / "efficiency.json",
        "per_image": run_dir / "a2_dev_per_image_confusion.jsonl",
        "projection_equivalence": run_dir / "projection_equivalence.json",
    }


class A2PretrainStudent(a0.PretrainStudent):
    """Student backbone plus registered A2 adapters for DDP parameter tracking."""

    def __init__(self, projection: nn.ModuleDict) -> None:
        super().__init__(base.build_backbone())
        self.projection = projection

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = super().forward(images)
        # The teacher-side adapter is evaluated outside the student forward,
        # but its parameters still belong to this DDP module.  This zero-valued
        # edge makes DDP's unused-parameter discovery see every adapter while
        # preserving the exact student feature values and the real adapter
        # gradients produced by the feature loss.
        dependency = None
        for parameter in self.projection.parameters():
            term = parameter.sum() * 0.0
            dependency = term if dependency is None else dependency + term
        if dependency is None:
            raise RuntimeError("A2 pretrain student has no adapter parameters")
        outputs["os16"] = outputs["os16"] + dependency.to(outputs["os16"].dtype)
        return outputs


def _adapter_anchor_loss(
    projection: nn.ModuleDict,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    anchor = None
    per_layer: Dict[str, float] = {}
    for layer in a0.A0_LAYER_ORDER:
        adapter = projection[layer]
        weight_delta = adapter.weight - adapter.weight_initial
        bias_delta = adapter.bias - adapter.bias_initial
        weight_term = weight_delta.float().pow(2).sum() / (
            adapter.weight_initial.float().pow(2).sum() + ADAPTER_ANCHOR_EPS
        )
        bias_term = bias_delta.float().pow(2).sum() / (
            adapter.bias_initial.float().pow(2).sum() + ADAPTER_ANCHOR_EPS
        )
        layer_anchor = weight_term + bias_term
        anchor = layer_anchor if anchor is None else anchor + layer_anchor
        per_layer[layer] = layer_anchor.detach()
    if anchor is None:
        raise RuntimeError("A2 anchor loss received an empty projection bundle")
    return anchor, per_layer


def _adapter_diagnostics(
    projection: nn.ModuleDict,
    projected_rms: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    with torch.no_grad():
        for layer in a0.A0_LAYER_ORDER:
            adapter = projection[layer]
            weight = adapter.weight.detach().float().flatten(1)
            initial_weight = adapter.weight_initial.detach().float().flatten(1)
            weight_norm = weight.norm().clamp_min(ADAPTER_ANCHOR_EPS)
            initial_norm = initial_weight.norm().clamp_min(ADAPTER_ANCHOR_EPS)
            singular_values = torch.linalg.svdvals(weight)
            row: Dict[str, object] = {
                "weight_delta_relative_fro": float(
                    (weight - initial_weight).norm().item() / initial_norm.item()
                ),
                "bias_delta_l2": float(
                    (adapter.bias.detach().float() - adapter.bias_initial.detach().float())
                    .norm()
                    .item()
                ),
                "weight_norm_ratio": float((weight_norm / initial_norm).item()),
                "minimum_singular_value": float(singular_values[-1].item()),
                "weight_sha256": adapter.parameter_sha256(),
                "initial_weight_sha256": adapter.initial_parameter_sha256(),
            }
            if projected_rms is not None and layer in projected_rms:
                row["projected_teacher_rms"] = float(
                    projected_rms[layer].detach().float().pow(2).mean().sqrt().item()
                )
            result[layer] = row
    return result


def _a2_pretrain_one_epoch_server(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    projection: nn.ModuleDict,
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
    lambda_feat: float,
    warmup_steps: int,
    current_optimizer_step: int,
    gradient_log_steps: int,
    rank: int,
    world_size: int,
) -> Tuple[Dict[str, object], int, List[Dict[str, object]]]:
    if sampler is not None:
        sampler.set_epoch(epoch)
    model.train()
    teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    anchor_sum = 0.0
    batch_count = 0
    layer_loss_sums = {layer: 0.0 for layer in a0.A0_LAYER_ORDER}
    optimizer_steps = 0
    first_step_gradient_l2: Optional[float] = None
    gradient_samples: List[Dict[str, object]] = []
    possible_steps = math.ceil(len(loader) / accumulation_steps)
    target_steps = min(possible_steps, remaining_optimizer_steps)
    max_batches = min(len(loader), target_steps * accumulation_steps)
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch} [A2 pretrain]",
        disable=rank != 0,
    )
    last_adapter_diagnostics: Dict[str, object] = {}
    last_feat_weight = 0.0

    for batch_index, (images, _targets, _paths) in enumerate(progress):
        if batch_index >= max_batches:
            break
        group_position = batch_index % accumulation_steps
        if group_position == 0:
            group_size = min(accumulation_steps, max_batches - batch_index)
        sync_gradients = group_position + 1 == group_size
        images = images.to(device, non_blocking=True)

        with t0.autocast_context(device, amp_enabled):
            with torch.no_grad():
                teacher_features = teacher.extract_features(images)
            student_features = model(images)
            layer_losses: Dict[str, torch.Tensor] = {}
            projected_teacher_features: Dict[str, torch.Tensor] = {}
            for layer in a0.A0_LAYER_ORDER:
                projected_teacher = projection[layer](teacher_features[layer])
                projected_teacher_features[layer] = projected_teacher
                layer_losses[layer] = F.mse_loss(
                    student_features[layer].float(), projected_teacher.float()
                )
            feature_loss = sum(layer_losses.values()) / len(a0.A0_LAYER_ORDER)
            anchor_loss, _anchor_by_layer = _adapter_anchor_loss(projection)

        next_optimizer_step = current_optimizer_step + optimizer_steps + 1
        feat_weight = lambda_feat * min(
            1.0, next_optimizer_step / max(int(warmup_steps), 1)
        )
        batch_loss = feat_weight * feature_loss + ADAPTER_ANCHOR_LAMBDA * anchor_loss
        last_feat_weight = feat_weight

        log_gradients = sync_gradients and gradient_log_steps > 0 and (
            next_optimizer_step % gradient_log_steps == 0
            or next_optimizer_step == current_optimizer_step + target_steps
        )
        per_layer_grads: Dict[str, float] = {}
        if log_gradients:
            for layer in a0.A0_LAYER_ORDER:
                grads = torch.autograd.grad(
                    layer_losses[layer],
                    student_features[layer],
                    retain_graph=True,
                    allow_unused=False,
                )
                per_layer_grads[layer] = float(grads[0].detach().float().norm(2).item())

        sync_context = contextlib.nullcontext()
        if isinstance(model, DDP) and not sync_gradients:
            sync_context = model.no_sync()
        with sync_context:
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
            current_optimizer_step += 1
            if log_gradients:
                last_adapter_diagnostics = _adapter_diagnostics(
                    projection, projected_teacher_features
                )
                sample: Dict[str, object] = {
                    "optimizer_step": next_optimizer_step,
                    "feat_weight": feat_weight,
                    "anchor_loss": float(anchor_loss.detach().item()),
                    "anchor_lambda": ADAPTER_ANCHOR_LAMBDA,
                    "adapter": last_adapter_diagnostics,
                }
                for layer in a0.A0_LAYER_ORDER:
                    sample[f"gradient_l2_{layer}"] = per_layer_grads[layer]
                    sample[f"student_grad_l2_{layer}"] = per_layer_grads[layer]
                    sample[f"adapter_weight_delta_{layer}"] = last_adapter_diagnostics[
                        layer
                    ]["weight_delta_relative_fro"]
                    sample[f"adapter_bias_delta_{layer}"] = last_adapter_diagnostics[
                        layer
                    ]["bias_delta_l2"]
                    sample[f"student_feature_mean_{layer}"] = float(
                        student_features[layer].detach().float().mean().item()
                    )
                    sample[f"student_feature_std_{layer}"] = float(
                        student_features[layer].detach().float().std().item()
                    )
                gradient_samples.append(sample)

        loss_sum += float(batch_loss.detach().item())
        anchor_sum += float(anchor_loss.detach().item())
        batch_count += 1
        for layer in a0.A0_LAYER_ORDER:
            layer_loss_sums[layer] += float(layer_losses[layer].detach().item())
        if rank == 0:
            progress.set_postfix(
                {
                    "feat": f"{loss_sum / max(batch_count, 1):.5f}",
                    "anchor": f"{anchor_sum / max(batch_count, 1):.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "steps": optimizer_steps,
                }
            )

    if optimizer_steps != target_steps:
        raise RuntimeError(
            "A2 pretrain optimizer-step accounting failed: "
            f"actual={optimizer_steps}, expected={target_steps}"
        )
    loss_sum = a0._reduce_scalar_sum(loss_sum, device, world_size)
    anchor_sum = a0._reduce_scalar_sum(anchor_sum, device, world_size)
    batch_count = int(a0._reduce_scalar_sum(float(batch_count), device, world_size))
    layer_loss_sums = {
        layer: a0._reduce_scalar_sum(value, device, world_size)
        for layer, value in layer_loss_sums.items()
    }
    metrics: Dict[str, object] = {
        "loss_total": loss_sum / max(batch_count, 1),
        "loss_os4": layer_loss_sums["os4"] / max(batch_count, 1),
        "loss_os8": layer_loss_sums["os8"] / max(batch_count, 1),
        "loss_os16": layer_loss_sums["os16"] / max(batch_count, 1),
        "loss_total_unweighted": sum(layer_loss_sums.values())
        / (len(a0.A0_LAYER_ORDER) * max(batch_count, 1)),
        "anchor_loss": anchor_sum / max(batch_count, 1),
        "anchor_lambda": ADAPTER_ANCHOR_LAMBDA,
        "feat_weight_final": last_feat_weight,
        "feature_gradient_l2_first_optimizer_step": first_step_gradient_l2,
        "adapter": last_adapter_diagnostics,
    }
    return metrics, optimizer_steps, gradient_samples


def _a2_pretrain_smoke_test(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    projection: nn.ModuleDict,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    rank: int,
) -> None:
    model.train()
    teacher.eval()
    projection.train()
    images, _targets, paths = next(iter(loader))
    images = images.to(device, non_blocking=True)
    with t0.autocast_context(device, amp_enabled):
        with torch.no_grad():
            teacher_features = teacher.extract_features(images)
        student_features = model(images)
        losses = {
            layer: F.mse_loss(
                student_features[layer].float(),
                projection[layer](teacher_features[layer]).float(),
            )
            for layer in a0.A0_LAYER_ORDER
        }
        anchor_loss, _ = _adapter_anchor_loss(projection)
        total = sum(losses.values()) / len(a0.A0_LAYER_ORDER) + (
            ADAPTER_ANCHOR_LAMBDA * anchor_loss
        )
    model.zero_grad(set_to_none=True)
    projection.zero_grad(set_to_none=True)
    total.backward()
    if not torch.isfinite(total):
        raise RuntimeError(f"Non-finite A2 pretrain smoke loss: {total.item()}")
    module = model.module if isinstance(model, DDP) else model
    backbone_gradients = sum(
        parameter.grad is not None for parameter in module.backbone.parameters()
    )
    adapter_gradients = sum(
        parameter.grad is not None for parameter in projection.parameters()
    )
    if backbone_gradients == 0:
        raise RuntimeError("A2 pretrain smoke test produced no backbone gradients")
    if adapter_gradients == 0:
        raise RuntimeError("A2 pretrain smoke test produced no adapter gradients")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("A2 pretrain smoke test found a teacher gradient")
    if rank == 0:
        print(
            f"[OK] A2 pretrain smoke test: sample={paths[0]}, "
            f"teacher os16={tuple(teacher_features['os16'].shape)}, "
            f"student os16={tuple(student_features['os16'].shape)}, "
            f"feature loss={total.item():.6f}, "
            f"backbone_grad_tensors={backbone_gradients}, "
            f"adapter_grad_tensors={adapter_gradients}"
        )
    model.zero_grad(set_to_none=True)
    projection.zero_grad(set_to_none=True)


def build_pretrain_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    optimizer_step: int,
    initial_backbone_state_sha256: str,
    config: Mapping[str, object],
    hashes: Mapping[str, object],
    dataset_lock: Mapping[str, object],
) -> Dict[str, object]:
    model_core = model.module if isinstance(model, DDP) else model
    state = t0.cpu_state_dict(model_core)
    projection = model_core.projection
    adapter_hashes = {
        layer: projection[layer].parameter_sha256() for layer in a0.A0_LAYER_ORDER
    }
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_type": ARTIFACT_TYPE_PRETRAIN,
        "experiment": EXPERIMENT,
        "stage": "pretrain",
        "initialization": "weights=None + A1 PCA-Conv initialization",
        "loss": "3-layer dense feature MSE + anchored trainable PCA-Conv",
        "adapter_trainable": True,
        "adapter_lr_ratio": ADAPTER_LR_RATIO,
        "adapter_anchor_lambda": ADAPTER_ANCHOR_LAMBDA,
        "model_state_dict": state,
        "model_state_sha256": t0.state_dict_sha256(state),
        "initial_backbone_state_sha256": initial_backbone_state_sha256,
        "adapter_parameter_sha256": adapter_hashes,
        "best_epoch": epoch,
        "best_optimizer_step": optimizer_step,
        "config": copy.deepcopy(config),
        "hashes": copy.deepcopy(hashes),
        "dataset_lock": copy.deepcopy(dataset_lock),
    }


def load_pretrain_backbone_state(pretrain_checkpoint: Path) -> Dict[str, torch.Tensor]:
    checkpoint = Path(pretrain_checkpoint).resolve()
    sidecar = checkpoint.with_name(f"{checkpoint.name}.sha256")
    if sidecar.is_file():
        t0.verify_checkpoint_sidecar(checkpoint)
    payload = t0.safe_torch_load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("artifact_type") != ARTIFACT_TYPE_PRETRAIN:
        raise RuntimeError(
            f"Not an A2 pretrain artifact: {payload.get('artifact_type')!r}"
        )
    state = payload["model_state_dict"]
    expected_hash = payload.get("model_state_sha256")
    if expected_hash and t0.state_dict_sha256(state) != expected_hash:
        raise RuntimeError("A2 pretrain model state failed SHA-256 verification")
    backbone_state = {
        key[len("backbone.") :]: value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise RuntimeError("A2 pretrain checkpoint contains no backbone state")
    return backbone_state


def run_pretrain_stage(
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
    amp_enabled: bool,
    teacher: torch.nn.Module,
    projection: nn.ModuleDict,
    train_loader: DataLoader,
    train_sampler: Optional[DistributedSampler],
    train_generator: torch.Generator,
    accumulation_steps: int,
    dataset_lock: Mapping[str, object],
    paths: Mapping[str, Path],
    config: Mapping[str, object],
    hashes: Mapping[str, object],
    resume: bool,
) -> Dict[str, object]:
    """A2 pretraining with separate backbone and 0.1x adapter LR groups."""

    main_process = rank == 0
    model = A2PretrainStudent(projection).to(device)
    initial_backbone_hash = t0.state_dict_sha256(model.backbone.state_dict())
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    module = model.module if isinstance(model, DDP) else model
    backbone_parameters = list(module.backbone.parameters())
    adapter_parameters = list(module.projection.parameters())
    optimizer = torch.optim.SGD(
        [
            {"params": backbone_parameters, "lr": args.lr},
            {"params": adapter_parameters, "lr": args.lr * ADAPTER_LR_RATIO},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=a0._poly_lr_factor(args, args.pretrain_max_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    warmup_steps = int(args.pretrain_max_steps * args.feature_warmup_ratio)

    history: List[Dict[str, object]] = []
    gradient_rows: List[Dict[str, object]] = []
    epoch = 0
    cumulative_optimizer_steps = 0
    if resume and paths["pretrain_last"].is_file():
        resume_payload = t0.safe_torch_load(
            paths["pretrain_last"], map_location="cpu", weights_only=False
        )
        if resume_payload.get("config") != config:
            raise RuntimeError("A2 pretrain resume configuration differs from current arguments")
        if resume_payload.get("artifact_type") != ARTIFACT_TYPE_PRETRAIN:
            raise RuntimeError("Resume file is not an A2 pretrain checkpoint")
        saved_initial_hash = resume_payload.get("initial_backbone_state_sha256")
        if saved_initial_hash != initial_backbone_hash:
            raise RuntimeError(
                "Resume checkpoint was created from a different scratch backbone initialization"
            )
        module.load_state_dict(resume_payload["model_state_dict"], strict=True)
        if t0.state_dict_sha256(module.state_dict()) != resume_payload.get("model_state_sha256"):
            raise RuntimeError("A2 pretrain resume model state hash verification failed")
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        scaler.load_state_dict(resume_payload["scaler_state_dict"])
        history = resume_payload["history"]
        gradient_rows = resume_payload["gradient_rows"]
        train_generator.set_state(resume_payload["train_generator_state"])
        epoch = int(resume_payload["epoch"])
        cumulative_optimizer_steps = int(resume_payload["optimizer_steps"])
        if main_process:
            print(
                f"[OK] Resuming A2 pretrain after epoch {epoch}, "
                f"step {cumulative_optimizer_steps:,}"
            )

    if not resume and any(
        path.is_file() for path in (paths["pretrain_last"], paths["pretrain_history"])
    ):
        raise FileExistsError(
            f"A2 pretrain artifacts already exist in {paths['run_dir']}; use --resume"
        )

    paths["pretrain_snapshots"].mkdir(parents=True, exist_ok=True)
    next_snapshot_step = (
        ((cumulative_optimizer_steps // args.pretrain_snapshot_steps) + 1)
        * args.pretrain_snapshot_steps
        if args.pretrain_snapshot_steps > 0
        else math.inf
    )
    training_started = time.time()
    while cumulative_optimizer_steps < args.pretrain_max_steps:
        epoch += 1
        remaining_steps = args.pretrain_max_steps - cumulative_optimizer_steps
        train_metrics, optimizer_steps, grad_samples = _a2_pretrain_one_epoch_server(
            model=model,
            teacher=teacher,
            projection=module.projection,
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
            lambda_feat=args.lambda_feat,
            warmup_steps=warmup_steps,
            current_optimizer_step=cumulative_optimizer_steps,
            gradient_log_steps=args.gradient_log_steps,
            rank=rank,
            world_size=world_size,
        )
        cumulative_optimizer_steps += optimizer_steps
        gradient_rows.extend(grad_samples)
        should_snapshot = (
            args.pretrain_snapshot_steps > 0
            and cumulative_optimizer_steps >= next_snapshot_step
        ) or cumulative_optimizer_steps == args.pretrain_max_steps
        if should_snapshot and args.pretrain_snapshot_steps > 0:
            while cumulative_optimizer_steps >= next_snapshot_step:
                next_snapshot_step += args.pretrain_snapshot_steps
        if main_process:
            history.append(
                {
                    "epoch": epoch,
                    "optimizer_steps": cumulative_optimizer_steps,
                    "optimizer_steps_this_epoch": optimizer_steps,
                    "learning_rate_backbone": optimizer.param_groups[0]["lr"],
                    "learning_rate_adapter": optimizer.param_groups[1]["lr"],
                    "train": train_metrics,
                }
            )
            t0.write_json_atomic(paths["pretrain_history"], history)
            t0.write_jsonl_atomic(paths["pretrain_gradients"], gradient_rows)
            last_payload = {
                "format_version": ARTIFACT_FORMAT_VERSION,
                "artifact_type": ARTIFACT_TYPE_PRETRAIN,
                "experiment": EXPERIMENT,
                "stage": "pretrain",
                "model_state_dict": t0.cpu_state_dict(module),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_generator_state": train_generator.get_state(),
                "history": history,
                "gradient_rows": gradient_rows,
                "epoch": epoch,
                "optimizer_steps": cumulative_optimizer_steps,
                "initial_backbone_state_sha256": initial_backbone_hash,
                "config": config,
                "hashes": hashes,
                "dataset_lock": dataset_lock,
                "adapter_lr_ratio": ADAPTER_LR_RATIO,
                "adapter_anchor_lambda": ADAPTER_ANCHOR_LAMBDA,
                "adapter_parameter_sha256": {
                    layer: module.projection[layer].parameter_sha256()
                    for layer in a0.A0_LAYER_ORDER
                },
            }
            last_payload["model_state_sha256"] = t0.state_dict_sha256(
                last_payload["model_state_dict"]
            )
            t0.torch_save_atomic(last_payload, paths["pretrain_last"])
            if should_snapshot:
                snapshot_payload = build_pretrain_checkpoint(
                    model=module,
                    epoch=epoch,
                    optimizer_step=cumulative_optimizer_steps,
                    initial_backbone_state_sha256=initial_backbone_hash,
                    config=config,
                    hashes=hashes,
                    dataset_lock=dataset_lock,
                )
                snapshot_path = (
                    paths["pretrain_snapshots"]
                    / f"a2_pretrain_snapshot_step_{cumulative_optimizer_steps:05d}.pth"
                )
                snapshot_hash = t0.write_checkpoint_with_sidecar(snapshot_payload, snapshot_path)
                print(
                    f"[OK] A2 pretrain snapshot: step={cumulative_optimizer_steps:,}, "
                    f"feature_loss={train_metrics['loss_total']:.5f}, sha256={snapshot_hash}"
                )
            print(
                f"Epoch {epoch}: step={cumulative_optimizer_steps:,}/"
                f"{args.pretrain_max_steps:,}, feature_loss={train_metrics['loss_total']:.5f}, "
                f"anchor={train_metrics['anchor_loss']:.5f}, "
                f"lr_backbone={optimizer.param_groups[0]['lr']:.2e}, "
                f"lr_adapter={optimizer.param_groups[1]['lr']:.2e}"
            )
        s2_0_server.barrier(world_size)

    s2_0_server.barrier(world_size)
    final_backbone_hash = t0.state_dict_sha256(module.backbone.state_dict())
    final_adapter_hashes = {
        layer: module.projection[layer].parameter_sha256()
        for layer in a0.A0_LAYER_ORDER
    }
    info = {
        "pretrain_checkpoint": paths["pretrain_last"],
        "pretrain_optimizer_steps": cumulative_optimizer_steps,
        "pretrain_epochs": epoch,
        "initial_backbone_state_sha256": initial_backbone_hash,
        "final_backbone_state_sha256": final_backbone_hash,
        "final_adapter_parameter_sha256": final_adapter_hashes,
        "elapsed_seconds": time.time() - training_started,
    }
    if main_process:
        print(
            f"[DONE] A2 pretrain: steps={cumulative_optimizer_steps:,}, "
            f"epochs={epoch}, backbone_sha256={final_backbone_hash}"
        )
    return info


def build_probe_best_checkpoint(*args, **kwargs):
    payload = _A0_BUILD_PROBE_BEST_CHECKPOINT(*args, **kwargs)
    payload.update(
        {
            "experiment": EXPERIMENT,
            "artifact_type": ARTIFACT_TYPE_PROBE,
            "initialization": "weights=None + A2 trainable PCA-Conv feature pretrain",
            "projection": (
                "A2 trainable PCA-Conv was used only during feature pretraining; "
                "adapter reset/removed before probe"
            ),
            "trainable_adapter_removed_before_probe": True,
            "adapter_lr_ratio": ADAPTER_LR_RATIO,
            "adapter_anchor_lambda": ADAPTER_ANCHOR_LAMBDA,
        }
    )
    return payload


def build_config(args: argparse.Namespace, *positional_args, **kwargs):
    config = _A0_BUILD_CONFIG(args, *positional_args, **kwargs)
    config.update(
        {
            "experiment": EXPERIMENT,
            "projection_implementation": (
                "trainable 1x1 Conv initialized from shared A1 fused StandardScaler+PCA"
            ),
            "projection_reference": "A1 fixed PCA-Conv / A0 explicit StandardScaler+PCA",
            "projection_trainable_during_pretrain": True,
            "projection_trainable_during_probe": False,
            "adapter_lr_ratio": ADAPTER_LR_RATIO,
            "adapter_lr": float(args.lr * ADAPTER_LR_RATIO),
            "adapter_anchor_lambda": ADAPTER_ANCHOR_LAMBDA,
            "adapter_anchor_epsilon": ADAPTER_ANCHOR_EPS,
            "adapter_anchor_formula": (
                "0.01 * sum_l(||W_l-W0_l||_F^2/(||W0_l||_F^2+eps) + "
                "||b_l-b0_l||_2^2/(||b0_l||_2^2+eps))"
            ),
            "pca_refit": False,
            "pca_resampling": False,
            "probe_adapter_policy": "reset_to_A1_initialization_and_remove_from_probe",
        }
    )
    return config


def _patch_a0_hooks() -> None:
    a0.__dict__["__file__"] = str(Path(__file__).resolve())
    a0.EXPERIMENT = EXPERIMENT
    a0.ARTIFACT_TYPE_PRETRAIN = ARTIFACT_TYPE_PRETRAIN
    a0.ARTIFACT_TYPE_PROBE = ARTIFACT_TYPE_PROBE
    a0.a0_paths = a2_paths
    a0.build_projection_bundle = build_projection_bundle
    a0.check_projection_conv_equivalence = check_projection_conv_equivalence
    a0.build_config = build_config
    a0.build_pretrain_checkpoint = build_pretrain_checkpoint
    a0.build_probe_best_checkpoint = build_probe_best_checkpoint
    a0.load_pretrain_backbone_state = load_pretrain_backbone_state
    a0.run_pretrain_stage = run_pretrain_stage
    a0.run_probe_stage = _run_probe_stage_a2
    a0._pretrain_smoke_test = _a2_pretrain_smoke_test


def _run_probe_stage_a2(
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
    amp_enabled: bool,
    teacher: torch.nn.Module,
    projection: nn.ModuleDict,
    train_loader: DataLoader,
    train_sampler: Optional[DistributedSampler],
    train_generator: torch.Generator,
    dev_loader: Optional[DataLoader],
    accumulation_steps: int,
    dataset_lock: Mapping[str, object],
    paths: Mapping[str, Path],
    config: Mapping[str, object],
    hashes: Mapping[str, object],
    projection_equivalence: Mapping[str, object],
    pca_parameter_record: Mapping[str, object],
    resume: bool,
) -> Dict[str, object]:
    # A2's adapter is target-side and training-only.  Resetting to A1 before
    # probe makes the removal explicit and ensures diagnostics use the locked
    # PCA target rather than a moving target that is absent at deployment.
    for layer in a0.A0_LAYER_ORDER:
        projection[layer].reset_to_initial_and_freeze()
    return _A0_RUN_PROBE_STAGE(
        args,
        rank,
        local_rank,
        world_size,
        device,
        amp_enabled,
        teacher,
        projection,
        train_loader,
        train_sampler,
        train_generator,
        dev_loader,
        accumulation_steps,
        dataset_lock,
        paths,
        config,
        hashes,
        projection_equivalence,
        pca_parameter_record,
        resume,
    )


def _rewrite_final_metrics(args: argparse.Namespace) -> None:
    metrics_path = a2_paths(args.output_dir, args.seed)["dev_metrics"]
    if not metrics_path.is_file():
        return
    with metrics_path.open("r", encoding="utf-8") as file_obj:
        results = json.load(file_obj)
    results["experiment"] = EXPERIMENT
    results["protocol"] = (
        "Scratch MobileNetV2 backbone trained label-free for 40k steps with "
        "three trainable teacher-side PCA-Conv adapters initialized from A1, "
        "adapter LR=0.1x backbone LR, and the fixed 0.01 normalized PCA anchor. "
        "The adapters were reset to their A1 initialization and removed before "
        "the unified 40k-step frozen-backbone 19-class R-ASPP probe. Best probe "
        "checkpoint is selected by dev_local mIoU; test_local is not evaluated."
    )
    results["model"] = {
        **results.get("model", {}),
        "initialization": "weights=None + A2 trainable PCA-Conv feature pretrain",
        "projection": "training-only trainable PCA-Conv; removed before probe",
        "adapter_removed_before_probe": True,
    }
    pretrain_last = a2_paths(args.output_dir, args.seed)["pretrain_last"]
    if pretrain_last.is_file():
        pretrain_payload = t0.safe_torch_load(
            pretrain_last, map_location="cpu", weights_only=False
        )
        adapter_hashes = pretrain_payload.get("adapter_parameter_sha256")
        if adapter_hashes:
            results.setdefault("hashes", {})[
                "pretrain_final_adapter_parameter_sha256"
            ] = adapter_hashes
            results.setdefault("training", {})[
                "pretrain_final_adapter_parameter_sha256"
            ] = adapter_hashes
    t0.write_json_atomic(metrics_path, results)


def _write_config_before_training(
    args: argparse.Namespace,
    device: torch.device,
    pca_record: Mapping[str, object],
    projection: nn.ModuleDict,
) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    accumulation_steps = a0.s2_0_server.effective_accumulation_steps(args, world_size)
    amp_enabled = bool(args.amp and device.type == "cuda")
    teacher_hash = t0.verify_checkpoint_sidecar(args.teacher_checkpoint)
    sampling_hash = str(pca_record.get("sampling_manifest_sha256", ""))
    projection_hashes = {
        layer: projection[layer].parameter_sha256() for layer in a0.A0_LAYER_ORDER
    }
    config = build_config(
        args,
        accumulation_steps,
        world_size,
        amp_enabled,
        teacher_hash,
        sampling_hash,
        pca_record,
        projection_hashes,
    )
    paths = a2_paths(args.output_dir, args.seed)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    t0.write_json_atomic(paths["config"], config)


def parse_args() -> argparse.Namespace:
    args = a0.parse_args()
    if args.stage == "pca":
        raise RuntimeError(
            "A2 does not refit PCA. Run dino_a0_server.py --stage pca once, "
            "then point A2 --pca-dir at the shared pca_shared directory."
        )
    return args


def main() -> None:
    global _EQUIVALENCE_REPORT
    _patch_a0_hooks()
    args = parse_args()

    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

    dataset_root = args.dataset_root.resolve()
    _dataset_lock, entries_by_split = t0.validate_dataset_lock(dataset_root)
    scalers, pcas, pca_record = a0.load_pca_parameters(args.pca_dir.resolve())
    teacher, _teacher_payload = a0.load_teacher_for_distillation(
        args.teacher_checkpoint,
        repo_dir=args.teacher_repo_dir,
        weights_path=args.teacher_weights_path,
        device=device,
        verify_checkpoint_file=True,
    )
    teacher.eval()
    _EQUIVALENCE_REPORT = build_equivalence_report(
        args,
        scalers,
        pcas,
        teacher,
        dataset_root,
        entries_by_split["train_local"],
        device,
    )
    print(
        "[OK] A2 initial PCA-Conv equivalence checks:",
        {layer: _EQUIVALENCE_REPORT[layer]["passed"] for layer in a0.A0_LAYER_ORDER},
    )
    del teacher
    _write_config_before_training(
        args,
        device,
        pca_record,
        build_projection_bundle(scalers, pcas),
    )
    a0.run_training(args)
    _rewrite_final_metrics(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
