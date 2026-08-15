"""A5 server entry point: trainable student-side coordinate adapters.

A5 keeps the locked A0 protocol unchanged on the teacher side:

    frozen T1 DINOv3 teacher
    shared per-layer StandardScaler + PCA target, T -> C_s
    scratch MobileNetV2 backbone
    40k label-free feature-pretraining steps + 40k frozen-backbone probe

The only scientific change is on the student side.  After the OS=4/8/16
feature taps, A5 inserts one trainable ``Conv2d(C_s, C_s, 1)`` per layer.
Each adapter is identity-initialized and uses ``0.1 * student_lr``.  The
adapters are present only in feature pretraining.  The probe loader extracts
only ``backbone.*`` from the pretraining artifact, records the adapter-removal
hash audit, and constructs the same adapter-free R-ASPP student as A0.

The long-running DDP/data/checkpoint implementation is delegated to
``dino_a0_server`` after its experiment-specific hooks are patched.  This
keeps A5 aligned with the server ``spawn``/no-pinned-memory/ordered-shutdown
protocol documented in ``server_training_issues_and_solutions.md``.
"""

from __future__ import annotations

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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import dino as t0
import dino_a0_server as a0
import dino_s2_0 as base


EXPERIMENT = "A5"
ARTIFACT_TYPE_PRETRAIN = "a5_pretrain_mobilenetv2_backbone_student_coordinate_adapter"
ARTIFACT_TYPE_PROBE = "a5_probe_mobilenetv2_raspp_student_coordinate_adapter_removed"
ARTIFACT_FORMAT_VERSION = 1

ADAPTER_LR_RATIO = 0.1
ADAPTER_EQUIVALENCE_TOLERANCE = 1e-7

DEFAULT_OUTPUT_DIR = a0.DEFAULT_OUTPUT_DIR
DEFAULT_PCA_DIR = a0.DEFAULT_PCA_DIR
DEFAULT_TEACHER_CHECKPOINT = a0.DEFAULT_TEACHER_CHECKPOINT

_A0_BUILD_CONFIG = a0.build_config
_A0_CHECK_PROJECTION = a0.check_projection_conv_equivalence
_A0_BUILD_PROBE_BEST_CHECKPOINT = a0.build_probe_best_checkpoint


class StudentCoordinateAdapter(nn.Conv2d):
    """Identity-initialized trainable ``C_s -> C_s`` 1x1 adapter."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels, channels, kernel_size=1, bias=True)
        with torch.no_grad():
            self.weight.zero_()
            indices = torch.arange(channels)
            self.weight[indices, indices, 0, 0] = 1.0
            self.bias.zero_()
        self.register_buffer("weight_initial", self.weight.detach().clone())
        self.register_buffer("bias_initial", self.bias.detach().clone())
        self.channels = int(channels)

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


def build_student_adapter_bundle() -> nn.ModuleDict:
    adapters = {
        layer: StudentCoordinateAdapter(a0.STUDENT_CHANNELS[layer])
        for layer in a0.A0_LAYER_ORDER
    }
    return nn.ModuleDict(adapters)


class A5PretrainStudent(a0.PretrainStudent):
    """Scratch student backbone followed by the three A5 trainable adapters."""

    def __init__(self) -> None:
        super().__init__(base.build_backbone())
        self.student_adapters = build_student_adapter_bundle()

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_features = super().forward(images)
        return {
            layer: self.student_adapters[layer](backbone_features[layer])
            for layer in a0.A0_LAYER_ORDER
        }


def build_pretrain_student() -> A5PretrainStudent:
    return A5PretrainStudent()


def build_projection_bundle(
    scalers: Mapping[str, Mapping[str, np.ndarray]],
    pcas: Mapping[str, Mapping[str, np.ndarray]],
) -> nn.ModuleDict:
    """Build the fixed A0 StandardScaler+PCA teacher target."""

    projections: Dict[str, a0.FixedPCAProjection] = {}
    for layer in a0.A0_LAYER_ORDER:
        projection = a0.FixedPCAProjection(
            scaler_mean=scalers[layer]["mean_"],
            scaler_scale=scalers[layer]["scale_"],
            pca_mean=pcas[layer]["mean_"],
            components=pcas[layer]["components_"],
        )
        if (
            projection.c_in != a0.TEACHER_CHANNELS[layer]
            or projection.d_out != a0.STUDENT_CHANNELS[layer]
        ):
            raise RuntimeError(
                f"A5 projection contract mismatch for {layer}: "
                f"got d_out={projection.d_out}, c_in={projection.c_in}; "
                f"expected d_out={a0.STUDENT_CHANNELS[layer]}, "
                f"c_in={a0.TEACHER_CHANNELS[layer]}"
            )
        if any(parameter.requires_grad for parameter in projection.parameters()):
            raise RuntimeError("A5 fixed PCA target unexpectedly has trainable parameters")
        projections[layer] = projection
    return nn.ModuleDict(projections)


def _adapter_diagnostics(adapters: nn.ModuleDict) -> Dict[str, Dict[str, object]]:
    """Return auditable student-adapter movement and rank diagnostics."""

    result: Dict[str, Dict[str, object]] = {}
    with torch.no_grad():
        for layer in a0.A0_LAYER_ORDER:
            adapter = adapters[layer]
            weight = adapter.weight.detach().float().flatten(1)
            initial_weight = adapter.weight_initial.detach().float().flatten(1)
            initial_norm = initial_weight.norm().clamp_min(1e-12)
            singular_values = torch.linalg.svdvals(weight)
            result[layer] = {
                "weight_delta_relative_fro": float(
                    (weight - initial_weight).norm().item() / initial_norm.item()
                ),
                "bias_delta_l2": float(
                    (
                        adapter.bias.detach().float()
                        - adapter.bias_initial.detach().float()
                    ).norm().item()
                ),
                "weight_norm_ratio": float(
                    (weight.norm().clamp_min(1e-12) / initial_norm).item()
                ),
                "minimum_singular_value": float(singular_values[-1].item()),
                "weight_sha256": adapter.parameter_sha256(),
                "initial_parameter_sha256": adapter.initial_parameter_sha256(),
            }
    return result


def _adapter_initial_hashes(adapters: nn.ModuleDict) -> Dict[str, str]:
    return {
        layer: adapters[layer].initial_parameter_sha256()
        for layer in a0.A0_LAYER_ORDER
    }


def _adapter_current_hashes(adapters: nn.ModuleDict) -> Dict[str, str]:
    return {
        layer: adapters[layer].parameter_sha256()
        for layer in a0.A0_LAYER_ORDER
    }


def build_student_adapter_equivalence_report() -> Dict[str, Dict[str, object]]:
    """Audit that the freshly constructed A5 adapters are exact identities."""

    adapters = build_student_adapter_bundle()
    report: Dict[str, Dict[str, object]] = {}
    for index, layer in enumerate(a0.A0_LAYER_ORDER):
        channels = a0.STUDENT_CHANNELS[layer]
        height, width = a0.PCA_VIEW_SHAPES[layer]
        generator = torch.Generator(device="cpu").manual_seed(5_001 + index)
        sample = torch.randn(
            2, channels, height, width, generator=generator, dtype=torch.float32
        )
        with torch.inference_mode():
            output = adapters[layer](sample)
        difference = (output - sample).abs()
        max_abs_error = float(difference.max().item())
        report[layer] = {
            "input_shape": list(sample.shape),
            "output_shape": list(output.shape),
            "max_abs_error": max_abs_error,
            "mean_abs_error": float(difference.mean().item()),
            "passed": bool(max_abs_error <= ADAPTER_EQUIVALENCE_TOLERANCE),
            "initialization": "identity weight + zero bias",
            "channels": channels,
            "initial_parameter_sha256": adapters[layer].initial_parameter_sha256(),
        }
        if not report[layer]["passed"]:
            raise RuntimeError(
                f"A5 student adapter identity check failed for {layer}: "
                f"max_abs_error={max_abs_error}"
            )
    return report


def check_projection_conv_equivalence(
    reference: a0.FixedPCAProjection,
    sample: torch.Tensor,
) -> Dict[str, object]:
    """Keep A0's fixed-PCA/1x1-Conv mathematical audit in the A5 artifacts."""

    return _A0_CHECK_PROJECTION(reference, sample)


def _a5_pretrain_one_epoch_server(
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
    """One A5 feature-pretraining epoch with student-side adapters."""

    if sampler is not None:
        sampler.set_epoch(epoch)
    model.train()
    teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
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
        desc=f"Epoch {epoch} [A5 pretrain]",
        disable=rank != 0,
    )
    final_adapter_diagnostics: Dict[str, Dict[str, object]] = {}
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
            for layer in a0.A0_LAYER_ORDER:
                projected_teacher = projection[layer](teacher_features[layer])
                layer_losses[layer] = F.mse_loss(
                    student_features[layer].float(), projected_teacher.float()
                )
            feature_loss = sum(layer_losses.values()) / len(a0.A0_LAYER_ORDER)

        next_optimizer_step = current_optimizer_step + optimizer_steps + 1
        feat_weight = lambda_feat * min(
            1.0, next_optimizer_step / max(int(warmup_steps), 1)
        )
        batch_loss = feat_weight * feature_loss
        last_feat_weight = feat_weight
        log_gradients = sync_gradients and gradient_log_steps > 0 and (
            next_optimizer_step % gradient_log_steps == 0
            or next_optimizer_step == current_optimizer_step + 1
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
                per_layer_grads[layer] = float(
                    grads[0].detach().float().norm(2).item()
                )

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
                module = model.module if isinstance(model, DDP) else model
                final_adapter_diagnostics = _adapter_diagnostics(
                    module.student_adapters
                )
                sample: Dict[str, object] = {
                    "optimizer_step": next_optimizer_step,
                    "feat_weight": feat_weight,
                    "adapter": copy.deepcopy(final_adapter_diagnostics),
                }
                for layer in a0.A0_LAYER_ORDER:
                    sample[f"gradient_l2_{layer}"] = per_layer_grads[layer]
                    sample[f"student_grad_l2_{layer}"] = per_layer_grads[layer]
                    sample[f"student_feature_mean_{layer}"] = float(
                        student_features[layer].detach().float().mean().item()
                    )
                    sample[f"student_feature_std_{layer}"] = float(
                        student_features[layer].detach().float().std().item()
                    )
                    sample[f"adapter_weight_delta_{layer}"] = (
                        final_adapter_diagnostics[layer]["weight_delta_relative_fro"]
                    )
                    sample[f"adapter_bias_delta_{layer}"] = (
                        final_adapter_diagnostics[layer]["bias_delta_l2"]
                    )
                gradient_samples.append(sample)

        loss_sum += float(batch_loss.detach().item())
        batch_count += 1
        for layer in a0.A0_LAYER_ORDER:
            layer_loss_sums[layer] += float(layer_losses[layer].detach().item())
        if rank == 0:
            progress.set_postfix(
                {
                    "feat": f"{loss_sum / max(batch_count, 1):.5f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "lr_adapt": f"{optimizer.param_groups[1]['lr']:.2e}",
                    "steps": optimizer_steps,
                }
            )

    if optimizer_steps != target_steps:
        raise RuntimeError(
            "A5 pretrain optimizer-step accounting failed: "
            f"actual={optimizer_steps}, expected={target_steps}"
        )
    loss_sum = a0._reduce_scalar_sum(loss_sum, device, world_size)
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
        "feat_weight_final": last_feat_weight,
        "feature_gradient_l2_first_optimizer_step": first_step_gradient_l2,
        "adapter": final_adapter_diagnostics,
        "adapter_lr_ratio": ADAPTER_LR_RATIO,
    }
    return metrics, optimizer_steps, gradient_samples


def _a5_pretrain_smoke_test(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    projection: nn.ModuleDict,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    rank: int,
) -> None:
    """Verify finite loss, teacher freeze, backbone gradients and adapter gradients."""

    model.train()
    teacher.eval()
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
        total = sum(losses.values()) / len(a0.A0_LAYER_ORDER)
    model.zero_grad(set_to_none=True)
    total.backward()
    if not torch.isfinite(total):
        raise RuntimeError(f"Non-finite A5 pretrain smoke loss: {total.item()}")
    module = model.module if isinstance(model, DDP) else model
    backbone_gradients = sum(
        parameter.grad is not None for parameter in module.backbone.parameters()
    )
    adapter_gradients = sum(
        parameter.grad is not None for parameter in module.student_adapters.parameters()
    )
    if backbone_gradients == 0:
        raise RuntimeError("A5 pretrain smoke test produced no backbone gradients")
    if adapter_gradients == 0:
        raise RuntimeError("A5 pretrain smoke test produced no student-adapter gradients")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("A5 pretrain smoke test found a teacher gradient")
    if rank == 0:
        print(
            f"[OK] A5 pretrain smoke test: sample={paths[0]}, "
            f"student os16={tuple(student_features['os16'].shape)}, "
            f"feature loss={total.item():.6f}, "
            f"backbone_grad_tensors={backbone_gradients}, "
            f"adapter_grad_tensors={adapter_gradients}"
        )
    model.zero_grad(set_to_none=True)


def a5_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    run_dir = Path(output_dir).resolve() / "A5" / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "config": run_dir / "config.json",
        "feature_taps": run_dir / "feature_taps.json",
        "pretrain_last": run_dir / "a5_pretrain_last.pth",
        "pretrain_history": run_dir / "a5_pretrain_history.json",
        "pretrain_gradients": run_dir / "a5_pretrain_gradient_norms.jsonl",
        "pretrain_snapshots": run_dir / "pretrain_snapshots",
        "probe_last": run_dir / "a5_probe_last.pth",
        "probe_history": run_dir / "a5_probe_history.json",
        "best_probe": run_dir / "a5_probe_mobilenetv2_raspp_best.pth",
        "dev_metrics": run_dir / "a5_dev_metrics.json",
        "efficiency": run_dir / "efficiency.json",
        "per_image": run_dir / "a5_dev_per_image_confusion.jsonl",
        "projection_equivalence": run_dir / "projection_equivalence.json",
        "student_adapter_equivalence": run_dir / "student_adapter_equivalence.json",
        "adapter_removal": run_dir / "student_adapter_removal.json",
    }


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
    adapter_hashes = _adapter_current_hashes(model_core.student_adapters)
    initial_adapter_hashes = _adapter_initial_hashes(model_core.student_adapters)
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_type": ARTIFACT_TYPE_PRETRAIN,
        "experiment": EXPERIMENT,
        "stage": "pretrain",
        "initialization": "weights=None + identity student coordinate adapters",
        "loss": "3-layer dense feature MSE with student-side coordinate adapters",
        "student_adapter_trainable": True,
        "student_adapter_location": "after student OS=4/8/16 feature taps",
        "student_adapter_initialization": "identity weight + zero bias",
        "student_adapter_lr_ratio": ADAPTER_LR_RATIO,
        "model_state_dict": state,
        "model_state_sha256": t0.state_dict_sha256(state),
        "initial_backbone_state_sha256": initial_backbone_state_sha256,
        "student_adapter_initial_parameter_sha256": initial_adapter_hashes,
        "student_adapter_parameter_sha256": adapter_hashes,
        "best_epoch": epoch,
        "best_optimizer_step": optimizer_step,
        "config": copy.deepcopy(config),
        "hashes": copy.deepcopy(hashes),
        "dataset_lock": copy.deepcopy(dataset_lock),
    }


def load_pretrain_backbone_state(pretrain_checkpoint: Path) -> Dict[str, torch.Tensor]:
    """Load only backbone tensors and audit that A5 adapters are discarded."""

    checkpoint = Path(pretrain_checkpoint).resolve()
    sidecar = checkpoint.with_name(f"{checkpoint.name}.sha256")
    if sidecar.is_file():
        t0.verify_checkpoint_sidecar(checkpoint)
    payload = t0.safe_torch_load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("artifact_type") != ARTIFACT_TYPE_PRETRAIN:
        raise RuntimeError(
            f"Not an A5 pretrain artifact: {payload.get('artifact_type')!r}"
        )
    state = payload["model_state_dict"]
    expected_hash = payload.get("model_state_sha256")
    if expected_hash and t0.state_dict_sha256(state) != expected_hash:
        raise RuntimeError("A5 pretrain model state failed SHA-256 verification")
    adapter_keys = [key for key in state if key.startswith("student_adapters.")]
    if not adapter_keys:
        raise RuntimeError("A5 pretrain checkpoint contains no student adapter state")
    backbone_state = {
        key[len("backbone.") :]: value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise RuntimeError("A5 pretrain checkpoint contains no backbone state")
    backbone_hash_before = t0.state_dict_sha256(backbone_state)
    adapter_hashes = payload.get("student_adapter_parameter_sha256")
    if adapter_hashes:
        for layer in a0.A0_LAYER_ORDER:
            weight_key = f"student_adapters.{layer}.weight"
            bias_key = f"student_adapters.{layer}.bias"
            actual = a0.numpy_arrays_sha256(
                state[weight_key].detach().cpu().numpy(),
                state[bias_key].detach().cpu().numpy(),
            )
            if actual != adapter_hashes.get(layer):
                raise RuntimeError(
                    f"A5 adapter hash verification failed for {layer}: "
                    f"payload={adapter_hashes.get(layer)}, actual={actual}"
                )

    # Removing the adapter means retaining exactly the backbone-only state.
    # Its hash must be unchanged by this projection of the checkpoint.
    backbone_state_after = dict(backbone_state)
    backbone_hash_after = t0.state_dict_sha256(backbone_state_after)
    if backbone_hash_before != backbone_hash_after:
        raise RuntimeError("A5 adapter removal changed the backbone state hash")
    record = {
        "experiment": EXPERIMENT,
        "pretrain_checkpoint": str(checkpoint),
        "pretrain_model_state_sha256": expected_hash,
        "adapter_state_key_count": len(adapter_keys),
        "adapter_state_keys": sorted(adapter_keys),
        "student_adapter_parameter_sha256": copy.deepcopy(adapter_hashes),
        "backbone_hash_before_adapter_removal": backbone_hash_before,
        "backbone_hash_after_adapter_removal": backbone_hash_after,
        "adapter_removed_before_probe": True,
        "probe_load_policy": "load only keys prefixed with backbone.; discard student_adapters.*",
    }
    if int(os.environ.get("RANK", "0")) == 0:
        t0.write_json_atomic(checkpoint.parent / "student_adapter_removal.json", record)
    return backbone_state_after


def run_pretrain_stage(
    args: object,
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
    """A5 pretraining with backbone LR=1.0 and adapter LR=0.1."""

    main_process = rank == 0
    model = build_pretrain_student().to(device)
    initial_backbone_hash = t0.state_dict_sha256(model.backbone.state_dict())
    initial_adapter_hashes = _adapter_initial_hashes(model.student_adapters)
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
    optimizer = torch.optim.SGD(
        [
            {"params": list(module.backbone.parameters()), "lr": args.lr},
            {
                "params": list(module.student_adapters.parameters()),
                "lr": args.lr * ADAPTER_LR_RATIO,
            },
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
            raise RuntimeError("A5 pretrain resume configuration differs from current arguments")
        if resume_payload.get("artifact_type") != ARTIFACT_TYPE_PRETRAIN:
            raise RuntimeError("Resume file is not an A5 pretrain checkpoint")
        if resume_payload.get("initial_backbone_state_sha256") != initial_backbone_hash:
            raise RuntimeError(
                "Resume checkpoint was created from a different scratch backbone initialization"
            )
        if resume_payload.get("student_adapter_initial_parameter_sha256") != initial_adapter_hashes:
            raise RuntimeError(
                "Resume checkpoint was created from a different identity adapter initialization"
            )
        module.load_state_dict(resume_payload["model_state_dict"], strict=True)
        if t0.state_dict_sha256(module.state_dict()) != resume_payload.get("model_state_sha256"):
            raise RuntimeError("A5 pretrain resume model state hash verification failed")
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
                f"[OK] Resuming A5 pretrain after epoch {epoch}, "
                f"step {cumulative_optimizer_steps:,}"
            )

    if not resume and any(
        path.is_file() for path in (paths["pretrain_last"], paths["pretrain_history"])
    ):
        raise FileExistsError(
            f"A5 pretrain artifacts already exist in {paths['run_dir']}; use --resume"
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
        train_metrics, optimizer_steps, grad_samples = _a5_pretrain_one_epoch_server(
            model=model,
            teacher=teacher,
            projection=projection,
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
        current_adapter_diagnostics = _adapter_diagnostics(module.student_adapters)
        train_metrics["adapter"] = current_adapter_diagnostics
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
                "initialization": "weights=None + identity student coordinate adapters",
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
                "student_adapter_initial_parameter_sha256": initial_adapter_hashes,
                "student_adapter_parameter_sha256": _adapter_current_hashes(
                    module.student_adapters
                ),
                "student_adapter_lr_ratio": ADAPTER_LR_RATIO,
                "config": config,
                "hashes": hashes,
                "dataset_lock": dataset_lock,
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
                    / f"a5_pretrain_snapshot_step_{cumulative_optimizer_steps:05d}.pth"
                )
                snapshot_hash = t0.write_checkpoint_with_sidecar(
                    snapshot_payload, snapshot_path
                )
                print(
                    f"[OK] A5 pretrain snapshot: step={cumulative_optimizer_steps:,}, "
                    f"feature_loss={train_metrics['loss_total']:.5f}, "
                    f"sha256={snapshot_hash}"
                )
            print(
                f"Epoch {epoch}: step={cumulative_optimizer_steps:,}/"
                f"{args.pretrain_max_steps:,}, "
                f"feature_loss={train_metrics['loss_total']:.5f}, "
                f"lr_backbone={optimizer.param_groups[0]['lr']:.2e}, "
                f"lr_adapter={optimizer.param_groups[1]['lr']:.2e}"
            )
        a0.s2_0_server.barrier(world_size)

    a0.s2_0_server.barrier(world_size)
    final_backbone_hash = t0.state_dict_sha256(module.backbone.state_dict())
    final_adapter_hashes = _adapter_current_hashes(module.student_adapters)
    info = {
        "pretrain_checkpoint": paths["pretrain_last"],
        "pretrain_optimizer_steps": cumulative_optimizer_steps,
        "pretrain_epochs": epoch,
        "initial_backbone_state_sha256": initial_backbone_hash,
        "final_backbone_state_sha256": final_backbone_hash,
        "initial_student_adapter_parameter_sha256": initial_adapter_hashes,
        "final_student_adapter_parameter_sha256": final_adapter_hashes,
        "elapsed_seconds": time.time() - training_started,
    }
    if main_process:
        print(
            f"[DONE] A5 pretrain: steps={cumulative_optimizer_steps:,}, "
            f"epochs={epoch}, backbone_sha256={final_backbone_hash}"
        )
    return info


def build_probe_best_checkpoint(*args, **kwargs):
    payload = _A0_BUILD_PROBE_BEST_CHECKPOINT(*args, **kwargs)
    payload.update(
        {
            "experiment": EXPERIMENT,
            "artifact_type": ARTIFACT_TYPE_PROBE,
            "initialization": "weights=None + A5 student-coordinate-adapter feature pretrain",
            "projection": "fixed StandardScaler+PCA teacher target",
            "student_adapter_policy": (
                "student-side Conv2d(C_s,C_s,1) adapters used only in pretrain; "
                "discarded before probe"
            ),
            "student_adapter_removed_before_probe": True,
            "student_adapter_lr_ratio": ADAPTER_LR_RATIO,
        }
    )
    return payload


def build_config(args, *positional_args, **kwargs):
    config = _A0_BUILD_CONFIG(args, *positional_args, **kwargs)
    adapter_probe = build_student_adapter_bundle()
    config.update(
        {
            "experiment": EXPERIMENT,
            "projection_implementation": "fixed explicit StandardScaler+PCA teacher target (same as A0)",
            "projection_reference": "A0 explicit StandardScaler+PCA",
            "projection_trainable": False,
            "student_adapter_implementation": "three Conv2d(C_s,C_s,1) after OS=4/8/16 student taps",
            "student_adapter_initialization": "identity weight + zero bias",
            "student_adapter_trainable_during_pretrain": True,
            "student_adapter_trainable_during_probe": False,
            "student_adapter_lr_ratio": ADAPTER_LR_RATIO,
            "student_adapter_lr": float(args.lr * ADAPTER_LR_RATIO),
            "student_adapter_initial_parameter_sha256": _adapter_initial_hashes(
                adapter_probe
            ),
            "probe_adapter_policy": (
                "load only backbone.* from A5 pretrain checkpoint; discard "
                "student_adapters.* before adapter-free R-ASPP probe"
            ),
            "pca_refit": False,
            "pca_resampling": False,
        }
    )
    return config


def _patch_a0_hooks() -> None:
    a0.__dict__["__file__"] = str(Path(__file__).resolve())
    a0.EXPERIMENT = EXPERIMENT
    a0.ARTIFACT_TYPE_PRETRAIN = ARTIFACT_TYPE_PRETRAIN
    a0.ARTIFACT_TYPE_PROBE = ARTIFACT_TYPE_PROBE
    a0.a0_paths = a5_paths
    a0.build_projection_bundle = build_projection_bundle
    a0.check_projection_conv_equivalence = check_projection_conv_equivalence
    a0.build_config = build_config
    a0.build_pretrain_student = build_pretrain_student
    a0.build_pretrain_checkpoint = build_pretrain_checkpoint
    a0.build_probe_best_checkpoint = build_probe_best_checkpoint
    a0.load_pretrain_backbone_state = load_pretrain_backbone_state
    a0.run_pretrain_stage = run_pretrain_stage
    a0._pretrain_smoke_test = _a5_pretrain_smoke_test


def _find_adapter_removal_record(args: object) -> Optional[Dict[str, object]]:
    candidates = [a5_paths(args.output_dir, args.seed)["adapter_removal"]]
    if args.pretrain_checkpoint:
        candidates.append(
            Path(args.pretrain_checkpoint).resolve().parent
            / "student_adapter_removal.json"
        )
    for path in candidates:
        if path.is_file():
            with path.open("r", encoding="utf-8") as file_obj:
                return json.load(file_obj)
    return None


def _rewrite_final_metrics(args: object) -> None:
    metrics_path = a5_paths(args.output_dir, args.seed)["dev_metrics"]
    if not metrics_path.is_file():
        return
    with metrics_path.open("r", encoding="utf-8") as file_obj:
        results = json.load(file_obj)
    results["experiment"] = EXPERIMENT
    results["protocol"] = (
        "Scratch MobileNetV2 backbone trained label-free for 40k steps with "
        "the fixed A0 StandardScaler+PCA teacher target and three identity-"
        "initialized trainable student-side Conv2d(C_s,C_s,1) adapters at "
        "0.1x backbone LR. The adapters are discarded before the common "
        "40k-step frozen-backbone 19-class R-ASPP probe; best checkpoint is "
        "selected by dev_local mIoU and test_local is not evaluated."
    )
    results["model"] = {
        **results.get("model", {}),
        "initialization": "weights=None + A5 student-coordinate-adapter feature pretrain",
        "projection": "fixed StandardScaler+PCA teacher target",
        "student_adapter": "training-only identity-initialized Conv2d(C_s,C_s,1)",
        "student_adapter_removed_before_probe": True,
    }
    pretrain_last = a5_paths(args.output_dir, args.seed)["pretrain_last"]
    if pretrain_last.is_file():
        pretrain_payload = t0.safe_torch_load(
            pretrain_last, map_location="cpu", weights_only=False
        )
        hashes = results.setdefault("hashes", {})
        training = results.setdefault("training", {})
        for key in (
            "initial_backbone_state_sha256",
            "student_adapter_initial_parameter_sha256",
            "student_adapter_parameter_sha256",
        ):
            if key in pretrain_payload:
                hashes[f"pretrain_{key}"] = pretrain_payload[key]
                training[f"pretrain_{key}"] = pretrain_payload[key]
        results["student_adapter"] = {
            "initial_parameter_sha256": pretrain_payload.get(
                "student_adapter_initial_parameter_sha256"
            ),
            "final_parameter_sha256": pretrain_payload.get(
                "student_adapter_parameter_sha256"
            ),
            "lr_ratio": ADAPTER_LR_RATIO,
            "removed_before_probe": True,
        }
    removal_record = _find_adapter_removal_record(args)
    if removal_record is not None:
        results["student_adapter_removal"] = removal_record
        results.setdefault("hashes", {}).update(
            {
                "backbone_hash_before_adapter_removal": removal_record.get(
                    "backbone_hash_before_adapter_removal"
                ),
                "backbone_hash_after_adapter_removal": removal_record.get(
                    "backbone_hash_after_adapter_removal"
                ),
            }
        )
    t0.write_json_atomic(metrics_path, results)


def _write_config_before_training(
    args: object,
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
    paths = a5_paths(args.output_dir, args.seed)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    t0.write_json_atomic(paths["config"], config)


def parse_args() -> object:
    args = a0.parse_args()
    if args.stage == "pca":
        raise RuntimeError(
            "A5 does not refit PCA. Run dino_a0_server.py --stage pca once, "
            "then point A5 --pca-dir at the shared pca_shared directory."
        )
    return args


def main() -> None:
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
    adapter_report = build_student_adapter_equivalence_report()
    paths = a5_paths(args.output_dir, args.seed)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0:
        t0.write_json_atomic(paths["student_adapter_equivalence"], adapter_report)
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
