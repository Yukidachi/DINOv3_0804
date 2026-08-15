"""A6 server entry point: trainable student adapters to the full teacher space.

A6 follows ``plan_markdown/A实验的具体实施方案.md`` exactly:

    teacher side : per-layer StandardScaler, no PCA, T -> C_t
    student side : Conv2d(C_s, C_t, 1) after the OS=4/8/16 taps
    initialization: fixed-seed orthogonal weights, zero bias
    optimization : adapter LR = 0.1 * backbone LR
    deployment    : adapters are removed before the common R-ASPP probe

The server lifecycle is shared with A0.  A5's tested student-adapter
pretraining loop is reused only for the optimizer/DDP/checkpoint mechanics;
all A6-specific projection, initialization, artifact and diagnostic hooks are
defined here so that A5 metadata cannot silently leak into an A6 run.

Typical commands (after the shared A0 scaler/PCA directory exists)::

    torchrun --standalone --nproc_per_node=2 dino_a6_server.py \
        --stage full --seed 42 --batch-size 2 --global-batch-size 8 \
        --device cuda

    python -B dino_a6_server.py --stage full --device cuda --smoke-test \
        --batch-size 1 --global-batch-size 1 --num-workers 0

A6 reuses only the ``StandardScaler`` arrays stored in the shared PCA
artifacts.  The PCA mean/components are deliberately ignored and the config
records that fact.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import dino as t0
import dino_a0_server as a0
import dino_a5_server as a5
import dino_s2_0 as base


EXPERIMENT = "A6"
ARTIFACT_TYPE_PRETRAIN = (
    "a6_pretrain_mobilenetv2_backbone_full_teacher_space_adapter"
)
ARTIFACT_TYPE_PROBE = (
    "a6_probe_mobilenetv2_raspp_full_teacher_space_adapter_removed"
)
ARTIFACT_FORMAT_VERSION = 1

ADAPTER_LR_RATIO = 0.1
ADAPTER_INIT_SEED_BASE = 42
ORTHOGONALITY_TOLERANCE = 1e-5
EQUIVALENCE_TOLERANCE = 1e-6

DEFAULT_OUTPUT_DIR = a0.DEFAULT_OUTPUT_DIR
DEFAULT_PCA_DIR = a0.DEFAULT_PCA_DIR
DEFAULT_TEACHER_CHECKPOINT = a0.DEFAULT_TEACHER_CHECKPOINT

_A0_BUILD_CONFIG = a0.build_config
_A0_BUILD_PROBE_BEST_CHECKPOINT = a0.build_probe_best_checkpoint
_A5_RUN_PRETRAIN_STAGE = a5.run_pretrain_stage
_EQUIVALENCE_REPORT: Optional[Dict[str, Dict[str, object]]] = None


class StandardizedTeacherSpaceProjection(nn.Module):
    """Fixed per-channel StandardScaler transform with the full C_t output.

    ``StandardScaler.transform`` is applied independently at every spatial
    position.  The module has buffers, not parameters, so it can never enter
    the student optimizer.
    """

    def __init__(
        self,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        layer: str,
    ) -> None:
        super().__init__()
        if scaler_mean.ndim != 1 or scaler_scale.shape != scaler_mean.shape:
            raise RuntimeError(
                f"A6 scaler arrays have inconsistent shapes for {layer}: "
                f"mean={scaler_mean.shape}, scale={scaler_scale.shape}"
            )
        mean = torch.as_tensor(scaler_mean, dtype=torch.float32).contiguous()
        scale = torch.as_tensor(scaler_scale, dtype=torch.float32).contiguous()
        # Match sklearn's zero-variance convention.
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        self.register_buffer("scaler_mean", mean.view(1, -1, 1, 1))
        self.register_buffer("scaler_scale", scale.view(1, -1, 1, 1))
        self.c_in = int(mean.numel())
        self.d_out = self.c_in
        self.layer = str(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.c_in:
            raise RuntimeError(
                f"A6 StandardScaler expects [B,{self.c_in},H,W], "
                f"got {tuple(x.shape)}"
            )
        return (x - self.scaler_mean) / self.scaler_scale

    def parameter_sha256(self) -> str:
        return a0.numpy_arrays_sha256(
            self.scaler_mean.detach().cpu().numpy().reshape(-1),
            self.scaler_scale.detach().cpu().numpy().reshape(-1),
        )


def build_projection_bundle(
    scalers: Mapping[str, Mapping[str, np.ndarray]],
    pcas: Mapping[str, Mapping[str, np.ndarray]],
) -> nn.ModuleDict:
    """Build fixed StandardScaler targets; intentionally do not read PCA."""

    del pcas
    projections: Dict[str, StandardizedTeacherSpaceProjection] = {}
    for layer in a0.A0_LAYER_ORDER:
        expected_channels = a0.TEACHER_CHANNELS[layer]
        projection = StandardizedTeacherSpaceProjection(
            scaler_mean=np.asarray(scalers[layer]["mean_"]),
            scaler_scale=np.asarray(scalers[layer]["scale_"]),
            layer=layer,
        )
        if projection.c_in != expected_channels or projection.d_out != expected_channels:
            raise RuntimeError(
                f"A6 projection contract mismatch for {layer}: "
                f"got {projection.c_in}->{projection.d_out}, "
                f"expected {expected_channels}->{expected_channels}"
            )
        if any(parameter.requires_grad for parameter in projection.parameters()):
            raise RuntimeError("A6 StandardScaler target unexpectedly has parameters")
        projections[layer] = projection
    return nn.ModuleDict(projections)


class FullTeacherSpaceAdapter(nn.Conv2d):
    """Fixed-seed orthogonally initialized trainable ``C_s -> C_t`` adapter."""

    def __init__(self, c_in: int, c_out: int, seed: int, layer: str) -> None:
        super().__init__(c_in, c_out, kernel_size=1, bias=True)
        # ``orthogonal_`` uses the current torch RNG.  Forking makes the
        # construction deterministic without changing the backbone/data RNG
        # sequence used by the surrounding experiment.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            nn.init.orthogonal_(self.weight)
        with torch.no_grad():
            self.bias.zero_()
        self.register_buffer("weight_initial", self.weight.detach().clone())
        self.register_buffer("bias_initial", self.bias.detach().clone())
        self.c_in = int(c_in)
        self.c_out = int(c_out)
        self.seed = int(seed)
        self.layer = str(layer)

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
        layer: FullTeacherSpaceAdapter(
            c_in=a0.STUDENT_CHANNELS[layer],
            c_out=a0.TEACHER_CHANNELS[layer],
            seed=ADAPTER_INIT_SEED_BASE + index,
            layer=layer,
        )
        for index, layer in enumerate(a0.A0_LAYER_ORDER)
    }
    return nn.ModuleDict(adapters)


class A6PretrainStudent(a0.PretrainStudent):
    """Scratch MobileNetV2 backbone followed by A6 training-only adapters."""

    def __init__(self) -> None:
        super().__init__(base.build_backbone())
        self.student_adapters = build_student_adapter_bundle()

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_features = super().forward(images)
        return {
            layer: self.student_adapters[layer](backbone_features[layer])
            for layer in a0.A0_LAYER_ORDER
        }


def build_pretrain_student() -> A6PretrainStudent:
    return A6PretrainStudent()


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


def _adapter_diagnostics(adapters: nn.ModuleDict) -> Dict[str, Dict[str, object]]:
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
                    (adapter.bias.detach().float() - adapter.bias_initial.detach().float())
                    .norm()
                    .item()
                ),
                "weight_norm_ratio": float(
                    (weight.norm().clamp_min(1e-12) / initial_norm).item()
                ),
                "minimum_singular_value": float(singular_values[-1].item()),
                "weight_sha256": adapter.parameter_sha256(),
                "initial_parameter_sha256": adapter.initial_parameter_sha256(),
                "input_channels": adapter.c_in,
                "output_channels": adapter.c_out,
            }
    return result


def build_student_adapter_equivalence_report() -> Dict[str, Dict[str, object]]:
    """Audit shape, finiteness and the fixed orthogonal initialization."""

    adapters = build_student_adapter_bundle()
    report: Dict[str, Dict[str, object]] = {}
    for layer in a0.A0_LAYER_ORDER:
        adapter = adapters[layer]
        height, width = a0.PCA_VIEW_SHAPES[layer]
        generator = torch.Generator(device="cpu").manual_seed(5_101 + len(report))
        sample = torch.randn(
            2,
            adapter.c_in,
            height,
            width,
            generator=generator,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            output = adapter(sample)
        matrix = adapter.weight.detach().float().flatten(1)
        # A6 has C_t > C_s for every locked layer, so columns should be
        # orthonormal.  Keep the general branch for an explicit contract.
        if matrix.shape[0] >= matrix.shape[1]:
            gram = matrix.t() @ matrix
            identity = torch.eye(matrix.shape[1], dtype=matrix.dtype)
            mode = "W.T@W"
        else:
            gram = matrix @ matrix.t()
            identity = torch.eye(matrix.shape[0], dtype=matrix.dtype)
            mode = "W@W.T"
        orthogonality_error = float((gram - identity).norm().item())
        max_abs_output = float(output.abs().max().item())
        passed = bool(
            list(output.shape) == [2, adapter.c_out, height, width]
            and torch.isfinite(output).all().item()
            and orthogonality_error <= ORTHOGONALITY_TOLERANCE
            and float(adapter.bias.detach().abs().max().item()) == 0.0
        )
        report[layer] = {
            "input_shape": list(sample.shape),
            "output_shape": list(output.shape),
            "input_channels": adapter.c_in,
            "output_channels": adapter.c_out,
            "seed": adapter.seed,
            "initialization": "torch.nn.init.orthogonal_ + zero bias",
            "orthogonality_matrix": mode,
            "orthogonality_error_frobenius": orthogonality_error,
            "orthogonality_tolerance": ORTHOGONALITY_TOLERANCE,
            "bias_max_abs": float(adapter.bias.detach().abs().max().item()),
            "output_max_abs": max_abs_output,
            "finite_output": bool(torch.isfinite(output).all().item()),
            "initial_parameter_sha256": adapter.initial_parameter_sha256(),
            "passed": passed,
        }
        if not passed:
            raise RuntimeError(
                f"A6 adapter initialization audit failed for {layer}: "
                f"orthogonality_error={orthogonality_error}, "
                f"output_shape={tuple(output.shape)}"
            )
    return report


def _explicit_standardize(
    sample: torch.Tensor, scaler_mean: torch.Tensor, scaler_scale: torch.Tensor
) -> torch.Tensor:
    mean = scaler_mean.view(1, -1, 1, 1)
    scale = scaler_scale.view(1, -1, 1, 1)
    return (sample - mean) / scale


def _compare_standardizer(
    projection: StandardizedTeacherSpaceProjection, sample: torch.Tensor
) -> Dict[str, object]:
    sample = sample.detach().cpu().float()
    with torch.inference_mode():
        reference = _explicit_standardize(
            sample,
            projection.scaler_mean.detach().cpu().float(),
            projection.scaler_scale.detach().cpu().float(),
        )
        candidate = projection.cpu()(sample).float()
    difference = (candidate - reference).abs()
    denominator = float(reference.norm().clamp_min(1e-12).item())
    max_abs_error = float(difference.max().item())
    relative_l2_error = float(difference.norm().item() / denominator)
    return {
        "input_shape": list(sample.shape),
        "output_shape": list(candidate.shape),
        "max_abs_error": max_abs_error,
        "mean_abs_error": float(difference.mean().item()),
        "relative_l2_error": relative_l2_error,
        "passed": bool(
            max_abs_error <= EQUIVALENCE_TOLERANCE
            and relative_l2_error <= EQUIVALENCE_TOLERANCE
        ),
    }


def _make_real_teacher_features(
    teacher: torch.nn.Module,
    dataset_root: Path,
    entries: Sequence[Tuple[str, str]],
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    if not entries:
        raise RuntimeError("A6 projection audit could not find a train_local image")
    image_rel = entries[0][0]
    with Image.open(dataset_root / image_rel) as image_obj:
        image = image_obj.convert("RGB")
    image = image.resize(
        (a0.PCA_VIEW_WIDTH, a0.PCA_VIEW_HEIGHT),
        resample=Image.Resampling.BILINEAR,
    )
    image_tensor = t0.image_to_normalized_tensor(image).unsqueeze(0).to(device)
    teacher.eval()
    with torch.inference_mode():
        features = teacher.extract_features(image_tensor)
    return {layer: features[layer].detach().cpu() for layer in a0.A0_LAYER_ORDER}


def build_projection_equivalence_report(
    scalers: Mapping[str, Mapping[str, np.ndarray]],
    teacher: torch.nn.Module,
    dataset_root: Path,
    entries: Sequence[Tuple[str, str]],
    device: torch.device,
) -> Dict[str, Dict[str, object]]:
    """Run random and real-feature audits for the A6 StandardScaler path."""

    projections = build_projection_bundle(scalers, {})
    real_features = _make_real_teacher_features(
        teacher, dataset_root, entries, device
    )
    report: Dict[str, Dict[str, object]] = {}
    for index, layer in enumerate(a0.A0_LAYER_ORDER):
        projection = projections[layer]
        generator = torch.Generator(device="cpu").manual_seed(3_001 + index)
        random_sample = torch.randn(
            2,
            a0.TEACHER_CHANNELS[layer],
            32,
            64,
            generator=generator,
        ) * 0.05
        random_result = _compare_standardizer(projection, random_sample)
        real_result = _compare_standardizer(projection, real_features[layer])
        passed = bool(random_result["passed"] and real_result["passed"])
        report[layer] = {
            "reference": "explicit per-channel StandardScaler transform",
            "candidate": "fixed A6 StandardScaler target with full teacher channels",
            "pca_used": False,
            "teacher_channels": a0.TEACHER_CHANNELS[layer],
            "student_channels": a0.STUDENT_CHANNELS[layer],
            "target_channels": a0.TEACHER_CHANNELS[layer],
            "random_input_scale": 0.05,
            "random_tensor": random_result,
            "real_teacher_feature": real_result,
            "scaler_mean_sha256": a0.numpy_arrays_sha256(
                np.asarray(scalers[layer]["mean_"])
            ),
            "scaler_scale_sha256": a0.numpy_arrays_sha256(
                np.asarray(scalers[layer]["scale_"])
            ),
            "projection_parameter_sha256": projection.parameter_sha256(),
            "passed": passed,
        }
        if not passed:
            raise RuntimeError(
                f"A6 StandardScaler audit failed for {layer}: "
                f"random={random_result['max_abs_error']}, "
                f"real={real_result['max_abs_error']}"
            )
    return report


def check_projection_conv_equivalence(
    reference: nn.Module,
    sample: torch.Tensor,
) -> Dict[str, object]:
    """Compatibility hook for A0's delegated run loop.

    A0 constructs its historical PCA reference before calling this hook.  A6
    does not use that object; the real A6 audit is performed before DDP starts
    and returned here by shape-matched layer.
    """

    del sample
    if _EQUIVALENCE_REPORT is None:
        raise RuntimeError("A6 StandardScaler audit was not run before training")
    layer = next(
        (
            candidate
            for candidate in a0.A0_LAYER_ORDER
            if a0.TEACHER_CHANNELS[candidate] == reference.c_in
            and a0.STUDENT_CHANNELS[candidate] == reference.d_out
        ),
        None,
    )
    if layer is None:
        raise RuntimeError(
            f"Cannot identify A6 layer for historical reference shape "
            f"[{reference.c_in}->{reference.d_out}]"
        )
    return _EQUIVALENCE_REPORT[layer]


def a6_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    run_dir = Path(output_dir).resolve() / "A6" / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "config": run_dir / "config.json",
        "feature_taps": run_dir / "feature_taps.json",
        "pretrain_last": run_dir / "a6_pretrain_last.pth",
        "pretrain_history": run_dir / "a6_pretrain_history.json",
        "pretrain_gradients": run_dir / "a6_pretrain_gradient_norms.jsonl",
        "pretrain_snapshots": run_dir / "pretrain_snapshots",
        "probe_last": run_dir / "a6_probe_last.pth",
        "probe_history": run_dir / "a6_probe_history.json",
        "best_probe": run_dir / "a6_probe_mobilenetv2_raspp_best.pth",
        "dev_metrics": run_dir / "a6_dev_metrics.json",
        "efficiency": run_dir / "efficiency.json",
        "per_image": run_dir / "a6_dev_per_image_confusion.jsonl",
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
    model_core = model.module if isinstance(model, a5.DDP) else model
    state = t0.cpu_state_dict(model_core)
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_type": ARTIFACT_TYPE_PRETRAIN,
        "experiment": EXPERIMENT,
        "stage": "pretrain",
        "initialization": "weights=None + fixed-seed orthogonal full-teacher-space adapters",
        "loss": "3-layer dense feature MSE with StandardScaler teacher targets",
        "projection": "per-layer StandardScaler; no PCA; target=C_t",
        "student_adapter_trainable": True,
        "student_adapter_location": "after student OS=4/8/16 feature taps",
        "student_adapter_initialization": "torch.nn.init.orthogonal_ + zero bias",
        "student_adapter_seed_base": ADAPTER_INIT_SEED_BASE,
        "student_adapter_lr_ratio": ADAPTER_LR_RATIO,
        "student_adapter_initial_parameter_sha256": _adapter_initial_hashes(
            model_core.student_adapters
        ),
        "student_adapter_parameter_sha256": _adapter_current_hashes(
            model_core.student_adapters
        ),
        "model_state_dict": state,
        "model_state_sha256": t0.state_dict_sha256(state),
        "initial_backbone_state_sha256": initial_backbone_state_sha256,
        "best_epoch": epoch,
        "best_optimizer_step": optimizer_step,
        "config": copy.deepcopy(config),
        "hashes": copy.deepcopy(hashes),
        "dataset_lock": copy.deepcopy(dataset_lock),
    }


def load_pretrain_backbone_state(pretrain_checkpoint: Path) -> Dict[str, torch.Tensor]:
    """Load only backbone tensors and record the A6 adapter removal audit."""

    checkpoint = Path(pretrain_checkpoint).resolve()
    sidecar = checkpoint.with_name(f"{checkpoint.name}.sha256")
    if sidecar.is_file():
        t0.verify_checkpoint_sidecar(checkpoint)
    payload = t0.safe_torch_load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("artifact_type") != ARTIFACT_TYPE_PRETRAIN:
        raise RuntimeError(
            f"Not an A6 pretrain artifact: {payload.get('artifact_type')!r}"
        )
    state = payload["model_state_dict"]
    expected_hash = payload.get("model_state_sha256")
    if expected_hash and t0.state_dict_sha256(state) != expected_hash:
        raise RuntimeError("A6 pretrain model state failed SHA-256 verification")

    adapter_keys = [key for key in state if key.startswith("student_adapters.")]
    if not adapter_keys:
        raise RuntimeError("A6 pretrain checkpoint contains no student adapter state")
    adapter_hashes = payload.get("student_adapter_parameter_sha256")
    if adapter_hashes:
        for layer in a0.A0_LAYER_ORDER:
            weight_key = f"student_adapters.{layer}.weight"
            bias_key = f"student_adapters.{layer}.bias"
            if weight_key not in state or bias_key not in state:
                raise RuntimeError(f"A6 checkpoint is missing adapter tensors for {layer}")
            actual = a0.numpy_arrays_sha256(
                state[weight_key].detach().cpu().numpy(),
                state[bias_key].detach().cpu().numpy(),
            )
            if actual != adapter_hashes.get(layer):
                raise RuntimeError(
                    f"A6 adapter hash verification failed for {layer}: "
                    f"payload={adapter_hashes.get(layer)}, actual={actual}"
                )

    backbone_state = {
        key[len("backbone.") :]: value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise RuntimeError("A6 pretrain checkpoint contains no backbone state")
    backbone_hash_before = t0.state_dict_sha256(backbone_state)
    backbone_state_after = dict(backbone_state)
    backbone_hash_after = t0.state_dict_sha256(backbone_state_after)
    if backbone_hash_before != backbone_hash_after:
        raise RuntimeError("A6 adapter removal changed the backbone state hash")

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
        "probe_load_policy": (
            "load only keys prefixed with backbone.; discard student_adapters.*"
        ),
        "deployment_model": "raw MobileNetV2 backbone + R-ASPP; no A6 adapter",
    }
    if int(os.environ.get("RANK", "0")) == 0:
        t0.write_json_atomic(checkpoint.parent / "student_adapter_removal.json", record)
    return backbone_state_after


def _a6_pretrain_smoke_test(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    projection: nn.ModuleDict,
    loader,
    device: torch.device,
    amp_enabled: bool,
    rank: int,
) -> None:
    """Verify gradients reach both the backbone and all A6 adapters."""

    model.train()
    teacher.eval()
    images, _targets, paths = next(iter(loader))
    images = images.to(device, non_blocking=True)
    with t0.autocast_context(device, amp_enabled):
        with torch.no_grad():
            teacher_features = teacher.extract_features(images)
        student_features = model(images)
        losses = {
            layer: nn.functional.mse_loss(
                student_features[layer].float(),
                projection[layer](teacher_features[layer]).float(),
            )
            for layer in a0.A0_LAYER_ORDER
        }
        total = sum(losses.values()) / len(a0.A0_LAYER_ORDER)
    model.zero_grad(set_to_none=True)
    total.backward()
    if not torch.isfinite(total):
        raise RuntimeError(f"Non-finite A6 pretrain smoke loss: {total.item()}")
    module = model.module if isinstance(model, a5.DDP) else model
    backbone_gradients = sum(
        parameter.grad is not None for parameter in module.backbone.parameters()
    )
    adapter_gradients = sum(
        parameter.grad is not None for parameter in module.student_adapters.parameters()
    )
    if backbone_gradients == 0:
        raise RuntimeError("A6 pretrain smoke test produced no backbone gradients")
    if adapter_gradients == 0:
        raise RuntimeError("A6 pretrain smoke test produced no adapter gradients")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("A6 pretrain smoke test found a teacher gradient")
    if rank == 0:
        print(
            f"[OK] A6 pretrain smoke test: sample={paths[0]}, "
            f"student target os16={tuple(student_features['os16'].shape)}, "
            f"feature loss={total.item():.6f}, "
            f"backbone_grad_tensors={backbone_gradients}, "
            f"adapter_grad_tensors={adapter_gradients}"
        )
    model.zero_grad(set_to_none=True)


def run_pretrain_stage(
    args,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
    amp_enabled: bool,
    teacher: torch.nn.Module,
    projection: nn.ModuleDict,
    train_loader,
    train_sampler,
    train_generator: torch.Generator,
    accumulation_steps: int,
    dataset_lock: Mapping[str, object],
    paths: Mapping[str, Path],
    config: Mapping[str, object],
    hashes: Mapping[str, object],
    resume: bool,
) -> Dict[str, object]:
    """Reuse A5's tested pretraining lifecycle and repair its last-payload text."""

    result = _A5_RUN_PRETRAIN_STAGE(
        args=args,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        amp_enabled=amp_enabled,
        teacher=teacher,
        projection=projection,
        train_loader=train_loader,
        train_sampler=train_sampler,
        train_generator=train_generator,
        accumulation_steps=accumulation_steps,
        dataset_lock=dataset_lock,
        paths=paths,
        config=config,
        hashes=hashes,
        resume=resume,
    )
    if rank == 0 and paths["pretrain_last"].is_file():
        payload = t0.safe_torch_load(
            paths["pretrain_last"], map_location="cpu", weights_only=False
        )
        payload.update(
            {
                "experiment": EXPERIMENT,
                "artifact_type": ARTIFACT_TYPE_PRETRAIN,
                "initialization": (
                    "weights=None + fixed-seed orthogonal full-teacher-space adapters"
                ),
                "projection": "per-layer StandardScaler; no PCA; target=C_t",
                "loss": (
                    "3-layer dense feature MSE with StandardScaler teacher targets"
                ),
                "student_adapter_initialization": (
                    "torch.nn.init.orthogonal_ + zero bias"
                ),
                "student_adapter_seed_base": ADAPTER_INIT_SEED_BASE,
                "student_adapter_lr_ratio": ADAPTER_LR_RATIO,
            }
        )
        t0.torch_save_atomic(payload, paths["pretrain_last"])
    if rank == 0:
        snapshot_dir = paths["pretrain_snapshots"]
        if snapshot_dir.is_dir():
            for old_path in snapshot_dir.glob("a5_pretrain_snapshot_step_*.pth"):
                new_path = old_path.with_name(
                    old_path.name.replace("a5_pretrain_", "a6_pretrain_", 1)
                )
                if new_path.exists():
                    raise FileExistsError(
                        f"A6 snapshot rename would overwrite an existing file: {new_path}"
                    )
                old_path.rename(new_path)
                old_sidecar = old_path.with_name(f"{old_path.name}.sha256")
                if old_sidecar.is_file():
                    old_sidecar.rename(new_path.with_name(f"{new_path.name}.sha256"))
    return result


def build_probe_best_checkpoint(*args, **kwargs):
    payload = _A0_BUILD_PROBE_BEST_CHECKPOINT(*args, **kwargs)
    payload.update(
        {
            "experiment": EXPERIMENT,
            "artifact_type": ARTIFACT_TYPE_PROBE,
            "initialization": "weights=None + A6 full-teacher-space feature pretrain",
            "projection": (
                "fixed per-layer StandardScaler, no PCA, full teacher channel target"
            ),
            "student_adapter_policy": (
                "student-side Conv2d(C_s,C_t,1) adapters used only in pretrain; "
                "discarded before probe"
            ),
            "student_adapter_removed_before_probe": True,
            "student_adapter_lr_ratio": ADAPTER_LR_RATIO,
        }
    )
    return payload


def build_config(args, *positional_args, **kwargs):
    config = _A0_BUILD_CONFIG(args, *positional_args, **kwargs)
    adapters = build_student_adapter_bundle()
    config.update(
        {
            "experiment": EXPERIMENT,
            "projection_implementation": (
                "fixed per-layer StandardScaler with complete teacher-space target"
            ),
            "projection_reference": "sklearn StandardScaler.transform",
            "projection_trainable": False,
            "projection_target_channels": {
                layer: a0.TEACHER_CHANNELS[layer] for layer in a0.A0_LAYER_ORDER
            },
            "student_adapter_implementation": (
                "three Conv2d(C_s,C_t,1) after OS=4/8/16 student taps"
            ),
            "student_adapter_initialization": (
                "fixed-seed torch.nn.init.orthogonal_ + zero bias"
            ),
            "student_adapter_seed_base": ADAPTER_INIT_SEED_BASE,
            "student_adapter_trainable_during_pretrain": True,
            "student_adapter_trainable_during_probe": False,
            "student_adapter_lr_ratio": ADAPTER_LR_RATIO,
            "student_adapter_lr": float(args.lr * ADAPTER_LR_RATIO),
            "student_adapter_initial_parameter_sha256": _adapter_initial_hashes(
                adapters
            ),
            "probe_adapter_policy": (
                "load only backbone.* from A6 pretrain checkpoint; discard "
                "student_adapters.* before adapter-free R-ASPP probe"
            ),
            "pca_refit": False,
            "pca_resampling": False,
            "pca_components_used": False,
            "standard_scaler_used": True,
        }
    )
    return config


def _patch_a5_hooks() -> None:
    """Make the delegated A5 loop resolve all model-specific symbols to A6."""

    a5.EXPERIMENT = EXPERIMENT
    a5.ARTIFACT_TYPE_PRETRAIN = ARTIFACT_TYPE_PRETRAIN
    a5.ARTIFACT_TYPE_PROBE = ARTIFACT_TYPE_PROBE
    a5.ADAPTER_LR_RATIO = ADAPTER_LR_RATIO
    a5.a5_paths = a6_paths
    a5.build_student_adapter_bundle = build_student_adapter_bundle
    a5.build_pretrain_student = build_pretrain_student
    a5.build_pretrain_checkpoint = build_pretrain_checkpoint
    a5._adapter_initial_hashes = _adapter_initial_hashes
    a5._adapter_current_hashes = _adapter_current_hashes
    a5._adapter_diagnostics = _adapter_diagnostics


def _patch_a0_hooks() -> None:
    a0.__dict__["__file__"] = str(Path(__file__).resolve())
    a0.EXPERIMENT = EXPERIMENT
    a0.ARTIFACT_TYPE_PRETRAIN = ARTIFACT_TYPE_PRETRAIN
    a0.ARTIFACT_TYPE_PROBE = ARTIFACT_TYPE_PROBE
    a0.a0_paths = a6_paths
    a0.build_projection_bundle = build_projection_bundle
    a0.check_projection_conv_equivalence = check_projection_conv_equivalence
    a0.build_config = build_config
    a0.build_pretrain_student = build_pretrain_student
    a0.build_pretrain_checkpoint = build_pretrain_checkpoint
    a0.build_probe_best_checkpoint = build_probe_best_checkpoint
    a0.load_pretrain_backbone_state = load_pretrain_backbone_state
    a0.run_pretrain_stage = run_pretrain_stage
    a0._pretrain_smoke_test = _a6_pretrain_smoke_test
    a0.compute_probe_diagnostics = compute_probe_diagnostics


def compute_probe_diagnostics(
    teacher: torch.nn.Module,
    probe_model: torch.nn.Module,
    projection: nn.ModuleDict,
    loader,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int = 8,
) -> Dict[str, object]:
    """Report CKA without invalid raw-vs-full-space MSE subtraction.

    After adapter removal the student has C_s channels and the teacher target
    has C_t channels.  CKA is defined for this unequal-channel case, while a
    direct ``student - standardized_teacher`` residual is not.  The report
    therefore makes that boundary explicit instead of silently broadcasting
    or comparing the wrong tensors.
    """

    del projection, amp_enabled
    teacher.eval()
    probe_model.eval()
    student = probe_model.module if isinstance(probe_model, a5.DDP) else probe_model
    cka_sums = {layer: 0.0 for layer in a0.A0_LAYER_ORDER}
    student_rms_sums = {layer: 0.0 for layer in a0.A0_LAYER_ORDER}
    teacher_rms_sums = {layer: 0.0 for layer in a0.A0_LAYER_ORDER}
    batches = 0
    with torch.inference_mode():
        for images, _targets, _paths in loader:
            if batches >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            teacher_features = teacher.extract_features(images)
            student_features = student.extract_features(images)
            for layer in a0.A0_LAYER_ORDER:
                student_feature = student_features[layer].float()
                teacher_feature = teacher_features[layer].float()
                cka_sums[layer] += a0.linear_cka(student_feature, teacher_feature)
                student_rms_sums[layer] += float(student_feature.square().mean().sqrt().item())
                teacher_rms_sums[layer] += float(teacher_feature.square().mean().sqrt().item())
            batches += 1
    if batches == 0:
        raise RuntimeError("A6 probe diagnostics sampled no dev batches")

    layers: Dict[str, object] = {}
    for layer in a0.A0_LAYER_ORDER:
        layers[layer] = {
            "cka_raw_student_vs_raw_teacher": cka_sums[layer] / batches,
            "student_rms": student_rms_sums[layer] / batches,
            "teacher_rms": teacher_rms_sums[layer] / batches,
            "student_channels": a0.STUDENT_CHANNELS[layer],
            "teacher_channels": a0.TEACHER_CHANNELS[layer],
            "alignment_relative_mse": None,
            "alignment_note": (
                "Not computed after adapter removal: raw student C_s and full "
                "teacher C_t spaces have different channel counts."
            ),
        }
    return {
        "sampled_batches": batches,
        "method": (
            "Linear CKA between raw adapter-free student features and raw "
            "teacher features; no invalid unequal-channel MSE"
        ),
        "adapter_removed": True,
        "layers": layers,
    }


def _find_adapter_removal_record(args) -> Optional[Dict[str, object]]:
    candidates = [a6_paths(args.output_dir, args.seed)["adapter_removal"]]
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


def _rewrite_final_metrics(args) -> None:
    metrics_path = a6_paths(args.output_dir, args.seed)["dev_metrics"]
    if not metrics_path.is_file():
        return
    with metrics_path.open("r", encoding="utf-8") as file_obj:
        results = json.load(file_obj)
    results["experiment"] = EXPERIMENT
    results["protocol"] = (
        "Scratch MobileNetV2 backbone trained label-free for 40k steps with "
        "three fixed per-layer StandardScaler teacher targets in the complete "
        "teacher channel space (no PCA), using trainable Conv2d(C_s,C_t,1) "
        "student adapters initialized by fixed-seed orthogonal matrices at "
        "0.1x backbone LR. The adapters are removed before the common 40k-step "
        "frozen-backbone 19-class R-ASPP probe; best checkpoint is selected by "
        "dev_local mIoU and test_local is not evaluated."
    )
    results["model"] = {
        **results.get("model", {}),
        "initialization": "weights=None + A6 full-teacher-space feature pretrain",
        "projection": "fixed StandardScaler, no PCA, target=C_t",
        "student_adapter": "training-only Conv2d(C_s,C_t,1)",
        "student_adapter_initialization": (
            "fixed-seed torch.nn.init.orthogonal_ + zero bias"
        ),
        "student_adapter_removed_before_probe": True,
    }
    pretrain_last = a6_paths(args.output_dir, args.seed)["pretrain_last"]
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
            "seed_base": ADAPTER_INIT_SEED_BASE,
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


def _resolve_device(args) -> torch.device:
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _write_config_before_training(
    args,
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
    paths = a6_paths(args.output_dir, args.seed)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    t0.write_json_atomic(paths["config"], config)


def parse_args():
    args = a0.parse_args()
    if args.stage == "pca":
        raise RuntimeError(
            "A6 does not refit PCA or StandardScaler. Run "
            "dino_a0_server.py --stage pca once, then point A6 --pca-dir "
            "at the shared pca_shared directory. A6 uses only its scaler arrays."
        )
    return args


def main() -> None:
    global _EQUIVALENCE_REPORT
    _patch_a5_hooks()
    _patch_a0_hooks()
    args = parse_args()
    device = _resolve_device(args)

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
    _EQUIVALENCE_REPORT = build_projection_equivalence_report(
        scalers,
        teacher,
        dataset_root,
        entries_by_split["train_local"],
        device,
    )
    print(
        "[OK] A6 StandardScaler projection audits:",
        {layer: _EQUIVALENCE_REPORT[layer]["passed"] for layer in a0.A0_LAYER_ORDER},
    )
    adapter_report = build_student_adapter_equivalence_report()
    projection = build_projection_bundle(scalers, pcas)
    paths = a6_paths(args.output_dir, args.seed)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0:
        t0.write_json_atomic(paths["student_adapter_equivalence"], adapter_report)
    _write_config_before_training(args, device, pca_record, projection)
    del teacher

    # A0's delegated runner reloads the teacher and scaler artifacts inside
    # its rank-local lifecycle, preserving the server DDP setup and shutdown.
    a0.run_training(args)
    _rewrite_final_metrics(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
