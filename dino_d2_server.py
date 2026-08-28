"""D2 server entry point: A0 feature KD plus masked sliced Wasserstein loss.

D2 keeps the controlled K1/D0 protocol unchanged (T1 teacher, fixed A0
projection, K-group shared scratch initialization, Cityscapes local split,
and the audited server DDP lifecycle) and adds only the registered SWD term::

    L_D2 = L_seg + warmup(step) * (lambda_feat * L_feat
                                   + lambda_swd * L_swd)

The formal run uses 64 fixed Gaussian directions per feature layer,
``lambda_swd=0.1``, a 4,000 optimizer-step warm-up, and at most 256 valid
feature tokens per layer per physical global micro-batch.  ``test_local`` is
deliberately never evaluated by this entry point.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import dino as common
import dino_a0_server as a0
import dino_k0_server as k0
import dino_k1_server as k1
import dino_s2_0_server as server_base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "D_MobileNetV2_RASPP_server"
K_GROUP_OUTPUT_DIR = SCRIPT_DIR / "result" / "K_MobileNetV2_RASPP_server"

EXPERIMENT = "D2"
EXPERIMENT_GROUP = "D_MobileNetV2_RASPP_server"
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_d2_swd"
ARTIFACT_FORMAT_VERSION = 1
# D2 seed=42 passed the implementation/screening gate.  The two additional
# seeds use the already audited K-group shared initializations; no D-specific
# initialization is generated here.
FORMAL_SEEDS = (42, 3407, 260805)
LAMBDA_FEAT = 1.0
LAMBDA_SWD = 0.1
FEATURE_WARMUP_STEPS = 4_000
DISTRIBUTION_TOKEN_CAP = 256
DISTRIBUTION_EPSILON = 1e-6
NUM_SWD_SLICES = 64
SWD_DIRECTION_SEED_BASE = 20_260_821
DISTRIBUTION_SPEC_VERSION = 1

_ORIGINAL_K0_PATHS = k0.k0_paths
_ORIGINAL_K1_BUILD_CONFIG = k1.build_config
_ORIGINAL_K1_BUILD_BEST_CHECKPOINT = k1.build_best_checkpoint
_ORIGINAL_K1_POSTPROCESS = k1._postprocess_metrics
_ORIGINAL_K1_TORCH_SAVE_ATOMIC = common.torch_save_atomic
_ORIGINAL_EVALUATE = k1._ORIGINAL_EVALUATE
_ORIGINAL_K1_REMOVE_HOOKS = k1._remove_k1_hooks
_ORIGINAL_K1_INSTALL_HOOKS = k1._install_k1_hooks
_ORIGINAL_K1_ENSURE_SHARED = k1._ORIGINAL_ENSURE_SHARED_INITIALIZATION

_ACTIVE_ARGS: Optional[argparse.Namespace] = None
_SWD_DIRECTIONS: Optional["FixedSWDDirections"] = None


def _canonical_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().float().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


class FixedSWDDirections(nn.Module):
    """Deterministic, unit-normalized random directions for all D2 layers."""

    def __init__(self, num_slices: int = NUM_SWD_SLICES) -> None:
        super().__init__()
        if num_slices < 1:
            raise ValueError("SWD requires at least one slice")
        self.num_slices = int(num_slices)
        for layer_index, layer in enumerate(a0.A0_LAYER_ORDER):
            channels = int(a0.STUDENT_CHANNELS[layer])
            generator = torch.Generator(device="cpu")
            generator.manual_seed(SWD_DIRECTION_SEED_BASE + layer_index)
            directions = torch.randn(
                channels,
                self.num_slices,
                generator=generator,
                dtype=torch.float32,
            )
            directions = F.normalize(directions, p=2, dim=0)
            if not torch.isfinite(directions).all():
                raise RuntimeError(f"Non-finite SWD directions for {layer}")
            self.register_buffer(f"{layer}_directions", directions, persistent=True)

        if list(self.parameters()):
            raise RuntimeError("SWD directions must not contain trainable parameters")
        self.requires_grad_(False)
        # Direction values never change.  Cache hashes while buffers are still
        # on CPU so training-time audits do not introduce CUDA synchronizations.
        self._metadata = self._build_metadata()

    def __getitem__(self, layer: str) -> torch.Tensor:
        if layer not in a0.A0_LAYER_ORDER:
            raise KeyError(f"Unknown SWD layer: {layer}")
        return getattr(self, f"{layer}_directions")

    def _build_metadata(self) -> Dict[str, object]:
        layers: Dict[str, object] = {}
        hashes: Dict[str, object] = {}
        for layer_index, layer in enumerate(a0.A0_LAYER_ORDER):
            directions = self[layer]
            direction_hash = _tensor_sha256(directions)
            hashes[layer] = direction_hash
            layers[layer] = {
                "channels": int(directions.shape[0]),
                "num_slices": int(directions.shape[1]),
                "seed": SWD_DIRECTION_SEED_BASE + layer_index,
                "shape": list(directions.shape),
                "sha256": direction_hash,
                "max_unit_norm_error": float(
                    (directions.detach().float().norm(dim=0) - 1.0)
                    .abs()
                    .max()
                    .cpu()
                    .item()
                ),
            }
        return {
            "generator": "torch.randn CPU float32 with an explicit per-layer generator",
            "normalization": "L2 per direction over channel dimension",
            "seed_formula": "20260821 + layer_index",
            "num_slices": self.num_slices,
            "layers": layers,
            "combined_sha256": _canonical_sha256(hashes),
        }

    def metadata(self) -> Dict[str, object]:
        return copy.deepcopy(self._metadata)


def _require_swd_directions() -> FixedSWDDirections:
    if _SWD_DIRECTIONS is None:
        raise RuntimeError("D2 SWD directions were not initialized")
    return _SWD_DIRECTIONS


def _swd_resource_hashes() -> Dict[str, object]:
    metadata = _require_swd_directions().metadata()
    layers = metadata["layers"]
    assert isinstance(layers, Mapping)
    return {
        "swd_direction_sha256": {
            layer: layers[layer]["sha256"]  # type: ignore[index]
            for layer in a0.A0_LAYER_ORDER
        },
        "swd_direction_combined_sha256": metadata["combined_sha256"],
    }


class _D2ArgumentParser(argparse.ArgumentParser):
    """Change K1's historical accumulation default to the D-group value."""

    def add_argument(self, *option_strings: str, **kwargs: Any):  # type: ignore[override]
        if "--accumulation-steps" in option_strings:
            kwargs["default"] = 2
        return super().add_argument(*option_strings, **kwargs)


def parse_args() -> argparse.Namespace:
    saved_default = k1.DEFAULT_OUTPUT_DIR
    saved_argparse = k1.argparse

    class D2ArgparseProxy:
        BooleanOptionalAction = argparse.BooleanOptionalAction

        def __getattr__(self, name: str) -> Any:
            return getattr(saved_argparse, name)

        @staticmethod
        def ArgumentParser(*parser_args: Any, **parser_kwargs: Any):
            parser_kwargs["description"] = (
                "D2 MobileNetV2+R-ASPP: hard-label CE, fixed A0 feature KD, "
                "and masked sliced Wasserstein feature-distribution alignment."
            )
            parser = _D2ArgumentParser(*parser_args, **parser_kwargs)
            parser.add_argument(
                "--lambda-swd",
                type=float,
                default=LAMBDA_SWD,
                help="Registered D2 SWD weight (formal default: 0.1).",
            )
            return parser

    k1.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    k1.argparse = D2ArgparseProxy()
    try:
        args = k1.parse_args()
    finally:
        k1.DEFAULT_OUTPUT_DIR = saved_default
        k1.argparse = saved_argparse

    if args.seed not in FORMAL_SEEDS:
        allowed = ", ".join(str(seed) for seed in FORMAL_SEEDS)
        raise SystemExit(f"D2 formal run requires --seed to be one of: {allowed}")
    if not args.smoke_test:
        locked = {
            "max_steps": 80_000,
            "eval_every_steps": 5_000,
            "gradient_log_steps": 500,
            "accumulation_steps": 2,
            "batch_size": 2,
            "global_batch_size": 8,
            "lambda_feat": LAMBDA_FEAT,
            "lambda_swd": LAMBDA_SWD,
            "feature_warmup_ratio": FEATURE_WARMUP_STEPS / 80_000,
        }
        for name, expected in locked.items():
            if getattr(args, name) != expected:
                raise SystemExit(
                    f"Formal D2 locks --{name.replace('_', '-')} to {expected}"
                )
    if not 0 < args.lambda_swd:
        raise SystemExit("--lambda-swd must be positive")
    if args.output_dir.resolve() == K_GROUP_OUTPUT_DIR.resolve():
        raise SystemExit("D2 output must be separate from the K-group directory")
    return args


def d2_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    original = _ORIGINAL_K0_PATHS(output_dir, seed)
    lambda_value = (
        float(_ACTIVE_ARGS.lambda_swd) if _ACTIVE_ARGS is not None else LAMBDA_SWD
    )
    run_dir = (
        output_dir.resolve()
        / EXPERIMENT
        / f"seed_{seed}_lambda_{lambda_value:g}_slices_{NUM_SWD_SLICES}"
    )
    return {
        key: run_dir if key == "run_dir" else run_dir / value.name
        for key, value in original.items()
    }


def _distribution_spec(
    args: argparse.Namespace, accumulation_steps: int, world_size: int
) -> Dict[str, object]:
    physical_batch = int(args.batch_size) * int(world_size)
    direction_metadata = _require_swd_directions().metadata()
    return {
        "spec_version": DISTRIBUTION_SPEC_VERSION,
        "enabled": True,
        "type": "SWD",
        "active_terms": ["swd"],
        "teacher_source": "A0 fixed projected teacher features, detached",
        "student_source": "native student OS=4/8/16 taps",
        "mask_policy": "resize targets != 255 with nearest-neighbor; ignore excluded",
        "token_cap_per_layer": DISTRIBUTION_TOKEN_CAP,
        "token_cap_scope": "physical global micro-batch",
        "local_token_cap": max(1, DISTRIBUTION_TOKEN_CAP // max(world_size, 1)),
        "token_sampling": "deterministic without replacement",
        "sampling_seed_formula": "seed*1000003 + optimizer_step*9176 + layer_index",
        "statistics": "teacher-batch mean/std normalization; global token matrix",
        "normalization_formula": "(x - teacher_mean) / (teacher_std + 1e-6)",
        "normalization_epsilon": DISTRIBUTION_EPSILON,
        "layers": list(a0.A0_LAYER_ORDER),
        "layer_reduction": "equal mean over OS=4/8/16",
        "swd_reduction": "FP32 sort per slice; mean absolute paired quantile difference over K*N",
        "ddp_gradient_policy": "global-token loss scaled by world_size for DDP averaging",
        "warmup_steps": FEATURE_WARMUP_STEPS,
        "lambda_feat": float(args.lambda_feat),
        "lambda_swd": float(args.lambda_swd),
        "physical_distribution_batch_size": physical_batch,
        "effective_optimizer_batch_size": physical_batch * int(accumulation_steps),
        "directions": direction_metadata,
        "coral": None,
        "discriminator": None,
    }


distribution_spec = _distribution_spec


def _token_hash(tokens: torch.Tensor) -> str:
    return _tensor_sha256(tokens)


def _sample_layer_tokens(
    feature: torch.Tensor,
    targets: torch.Tensor,
    layer: str,
    seed: int,
    optimizer_step: int,
    world_size: int,
) -> Tuple[torch.Tensor, int, str]:
    """Return valid BCHW positions as an N x C matrix with fixed sampling."""

    _, channels, height, width = feature.shape
    valid = F.interpolate(
        (targets != common.IGNORE_INDEX).float().unsqueeze(1),
        size=(height, width),
        mode="nearest",
    ).squeeze(1).bool()
    tokens = feature.permute(0, 2, 3, 1)[valid]
    local_cap = max(1, DISTRIBUTION_TOKEN_CAP // max(world_size, 1))
    if tokens.shape[0] > local_cap:
        layer_index = list(a0.A0_LAYER_ORDER).index(layer)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(seed) * 1_000_003 + int(optimizer_step) * 9_176 + layer_index
        )
        indices = torch.randperm(tokens.shape[0], generator=generator)[:local_cap]
        tokens = tokens[indices.to(tokens.device)]
    if tokens.shape[0] == 0:
        tokens = feature.new_empty((0, channels))
    return tokens, int(tokens.shape[0]), _token_hash(tokens)


def _gather_tokens_with_local_grad(
    tokens: torch.Tensor, world_size: int
) -> torch.Tensor:
    """Gather variable-length token matrices while retaining this rank's grad."""

    if world_size <= 1:
        return tokens
    device = tokens.device
    local_size = torch.tensor([tokens.shape[0]], dtype=torch.long, device=device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    counts = [int(value.item()) for value in sizes]
    max_size = max(counts, default=0)
    channels = tokens.shape[1]
    if tokens.shape[0] < max_size:
        padded = torch.cat(
            [tokens, tokens.new_zeros((max_size - tokens.shape[0], channels))], dim=0
        )
    else:
        padded = tokens
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded.detach())
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
    gathered[rank] = tokens
    return torch.cat(
        [chunk[:count] for chunk, count in zip(gathered, counts)], dim=0
    )


def sliced_wasserstein_loss(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    directions: torch.Tensor,
    epsilon: float = DISTRIBUTION_EPSILON,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute teacher-normalized empirical SWD using fixed directions."""

    if student_tokens.ndim != 2 or teacher_tokens.ndim != 2:
        raise ValueError("SWD expects [N, C] token matrices")
    if student_tokens.shape != teacher_tokens.shape:
        raise ValueError("Student and teacher SWD token shapes must match")
    token_count, channels = student_tokens.shape
    if token_count == 0:
        raise RuntimeError("SWD received no valid feature tokens")
    if directions.ndim != 2 or directions.shape[0] != channels:
        raise ValueError(
            "SWD directions must have shape [C, K], got "
            f"{tuple(directions.shape)} for C={channels}"
        )
    if directions.shape[1] != NUM_SWD_SLICES:
        raise ValueError(
            f"D2 requires exactly {NUM_SWD_SLICES} SWD slices, "
            f"got {directions.shape[1]}"
        )

    teacher = teacher_tokens.detach().float()
    student = student_tokens.float()
    fixed_directions = directions.detach().to(device=student.device, dtype=torch.float32)
    teacher_mean = teacher.mean(dim=0)
    teacher_std = teacher.std(dim=0, unbiased=False)
    teacher_normalized = (teacher - teacher_mean) / (teacher_std + epsilon)
    student_normalized = (student - teacher_mean) / (teacher_std + epsilon)
    projected_teacher = teacher_normalized.matmul(fixed_directions)
    projected_student = student_normalized.matmul(fixed_directions)
    sorted_teacher = torch.sort(projected_teacher, dim=0).values
    sorted_student = torch.sort(projected_student, dim=0).values
    slice_losses = (sorted_student - sorted_teacher).abs().mean(dim=0)
    loss = slice_losses.mean()
    if not torch.isfinite(loss) or not torch.isfinite(slice_losses).all():
        raise RuntimeError("SWD produced a non-finite value")
    return loss, slice_losses


compute_swd_loss = sliced_wasserstein_loss


def _swd_layer_loss(
    student_feature: torch.Tensor,
    projected_teacher: torch.Tensor,
    targets: torch.Tensor,
    layer: str,
    args: argparse.Namespace,
    optimizer_step: int,
    world_size: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    student_tokens, student_count, student_hash = _sample_layer_tokens(
        student_feature, targets, layer, args.seed, optimizer_step, world_size
    )
    teacher_tokens, teacher_count, teacher_hash = _sample_layer_tokens(
        projected_teacher.detach(),
        targets,
        layer,
        args.seed,
        optimizer_step,
        world_size,
    )
    if student_count != teacher_count:
        raise RuntimeError("Student/teacher distribution token counts diverged")
    student_global = _gather_tokens_with_local_grad(student_tokens, world_size)
    teacher_global = _gather_tokens_with_local_grad(
        teacher_tokens.detach(), world_size
    ).detach()
    directions = _require_swd_directions()[layer]
    loss, slice_losses = sliced_wasserstein_loss(
        student_global, teacher_global, directions
    )
    direction_metadata = _require_swd_directions().metadata()["layers"]
    assert isinstance(direction_metadata, Mapping)
    audit = {
        "local_token_count": student_count,
        "global_token_count": int(student_global.shape[0]),
        "student_token_hash": student_hash,
        "teacher_token_hash": teacher_hash,
        "global_student_token_hash": _token_hash(student_global),
        "global_teacher_token_hash": _token_hash(teacher_global),
        "num_slices": NUM_SWD_SLICES,
        "direction_seed": direction_metadata[layer]["seed"],  # type: ignore[index]
        "direction_sha256": direction_metadata[layer]["sha256"],  # type: ignore[index]
        "slice_l1_mean": float(slice_losses.detach().mean().item()),
        "slice_l1_min": float(slice_losses.detach().min().item()),
        "slice_l1_max": float(slice_losses.detach().max().item()),
    }
    return loss, audit


def ensure_d2_resources(
    model: torch.nn.Module,
    args: argparse.Namespace,
    _output_dir: Path,
    seed: int,
    rank: int,
    world_size: int,
) -> Tuple[str, str, Path]:
    """Load K shared init/K1 resources and create audited fixed directions."""

    global _SWD_DIRECTIONS
    saved = k1._ORIGINAL_ENSURE_SHARED_INITIALIZATION
    k1._ORIGINAL_ENSURE_SHARED_INITIALIZATION = (
        lambda m, a, _ignored, s, r, w: _ORIGINAL_K1_ENSURE_SHARED(
            m, a, K_GROUP_OUTPUT_DIR, s, r, w
        )
    )
    try:
        init_result = k1.ensure_k1_resources(
            model, args, K_GROUP_OUTPUT_DIR, seed, rank, world_size
        )
    finally:
        k1._ORIGINAL_ENSURE_SHARED_INITIALIZATION = saved

    device = next(model.parameters()).device
    if _SWD_DIRECTIONS is None:
        _SWD_DIRECTIONS = FixedSWDDirections().to(device)
    metadata = _SWD_DIRECTIONS.metadata()
    combined_hash = metadata["combined_sha256"]
    if world_size > 1:
        gathered_hashes: List[Optional[str]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_hashes, combined_hash)
        if len(set(gathered_hashes)) != 1:
            raise RuntimeError(
                f"SWD direction hashes differ across ranks: {gathered_hashes}"
            )
    return init_result


def build_config_d2(
    args: argparse.Namespace,
    accumulation_steps: int,
    world_size: int,
    device: torch.device,
    shared_init_state_sha256: str,
    shared_init_file_sha256: str,
) -> Dict[str, object]:
    if not args.smoke_test:
        physical_batch = int(args.batch_size) * int(world_size)
        effective_batch = physical_batch * int(accumulation_steps)
        if world_size != 2:
            raise RuntimeError(f"Formal D2 requires world_size=2, got {world_size}")
        if physical_batch != 4:
            raise RuntimeError(
                "Formal D2 requires physical_distribution_batch_size=4, "
                f"got {physical_batch}"
            )
        if accumulation_steps != 2 or effective_batch != 8:
            raise RuntimeError(
                "Formal D2 requires accumulation_steps=2 and effective global "
                f"batch=8, got accumulation={accumulation_steps}, "
                f"global_batch={effective_batch}"
            )
    config = _ORIGINAL_K1_BUILD_CONFIG(
        args,
        accumulation_steps,
        world_size,
        device,
        shared_init_state_sha256,
        shared_init_file_sha256,
    )
    spec = _distribution_spec(args, accumulation_steps, world_size)
    config.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "server_entry_point": str(Path(__file__).resolve()),
            "formal_seeds": list(FORMAL_SEEDS),
            "shared_initialization": {
                "source_group": "K_MobileNetV2_RASPP_server",
                "path": str(
                    k0._shared_init_path(K_GROUP_OUTPUT_DIR, args.seed).resolve()
                ),
                "state_sha256": shared_init_state_sha256,
                "file_sha256": shared_init_file_sha256,
                "d_specific_initialization_created": False,
            },
            "distribution": copy.deepcopy(spec),
            "distribution_spec_sha256": _canonical_sha256(spec),
            "test_local_evaluated": False,
        }
    )
    config["loss"] = {
        "hard_label_ce": True,
        "feature_kd": True,
        "feature_mechanism": "A0 fixed StandardScaler+PCA teacher-to-student",
        "feature_layers": list(a0.A0_LAYER_ORDER),
        "feature_reduction": "mean per BCHW layer, then equal mean over 3 layers",
        "distribution_kd": True,
        "distribution_type": "SWD",
        "distribution_reduction": (
            "64 fixed slices; mean L1 paired quantile distance; "
            "equal mean over OS=4/8/16; teacher-batch normalized"
        ),
        "lambda_feat": float(args.lambda_feat),
        "lambda_swd": float(args.lambda_swd),
        "auxiliary_warmup_steps": FEATURE_WARMUP_STEPS,
        "warmup_step_unit": "optimizer_step",
        "logit_kd": False,
    }
    return config


def _direction_state_dict_cpu() -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in _require_swd_directions().state_dict().items()
    }


def build_best_checkpoint_d2(*args: Any, **kwargs: Any) -> Dict[str, object]:
    payload = _ORIGINAL_K1_BUILD_BEST_CHECKPOINT(*args, **kwargs)
    config = payload.get("config", {})
    spec = config.get("distribution") if isinstance(config, Mapping) else None
    payload.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "distribution": copy.deepcopy(spec),
            "distribution_spec_sha256": (
                _canonical_sha256(spec) if isinstance(spec, Mapping) else None
            ),
            "swd_direction_state_dict": _direction_state_dict_cpu(),
            "loss_schema": "CE_plus_A0_fixed_feature_MSE_plus_SWD",
        }
    )
    return payload


def audit_shapes_d2(
    model: torch.nn.Module,
    device: torch.device,
    height: int,
    width: int,
    amp_enabled: bool,
) -> Dict[str, object]:
    audit = k1.audit_k1_shapes(model, device, height, width, amp_enabled)
    audit["experiment"] = EXPERIMENT
    audit["distribution"] = {
        "type": "SWD",
        "layers": list(a0.A0_LAYER_ORDER),
        "mask_policy": "targets != 255, nearest-neighbor resize",
        "token_cap_per_layer": DISTRIBUTION_TOKEN_CAP,
        "normalization": "teacher-batch mean/std",
        "sorting_dtype": "float32",
        "directions": _require_swd_directions().metadata(),
    }
    return audit


def _patched_torch_save_atomic_d2(payload: object, path: Path) -> None:
    if isinstance(payload, Mapping) and payload.get("artifact_type") == ARTIFACT_TYPE:
        value = dict(payload)
        value["hashes"] = {
            **dict(value.get("hashes", {})),
            **k1._resource_hashes(),
            **_swd_resource_hashes(),
        }
        value["experiment"] = EXPERIMENT
        value["experiment_group"] = EXPERIMENT_GROUP
        value["swd_direction_state_dict"] = _direction_state_dict_cpu()
        try:
            _ORIGINAL_K1_TORCH_SAVE_ATOMIC(value, path)
        finally:
            common.torch_save_atomic = _patched_torch_save_atomic_d2
    else:
        _ORIGINAL_K1_TORCH_SAVE_ATOMIC(payload, path)


def _evaluate_d2(*args: Any, **kwargs: Any):
    split_name = kwargs.get("split_name")
    if isinstance(split_name, str):
        kwargs["split_name"] = split_name.replace("K0", EXPERIMENT).replace(
            "K1", EXPERIMENT
        )
    return _ORIGINAL_EVALUATE(*args, **kwargs)


def _gradient_l2(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm(2).item())


def _gradient_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.detach().float().reshape(-1)
    right_flat = right.detach().float().reshape(-1)
    denominator = left_flat.norm() * right_flat.norm()
    if float(denominator.item()) == 0.0:
        return 0.0
    return float(torch.dot(left_flat, right_flat).div(denominator).item())


def _reduce_d2_statistics(
    values: Sequence[float], count: int, device: torch.device, world_size: int
) -> Tuple[List[float], int]:
    tensor = torch.tensor([*values, float(count)], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_count = int(tensor[-1].item())
    denominator = max(global_count, 1)
    return [float(value / denominator) for value in tensor[:-1].tolist()], global_count


def _reduce_global_token_sums(
    token_sums: Mapping[str, int], device: torch.device, world_size: int
) -> Dict[str, int]:
    values = torch.tensor(
        [int(token_sums[layer]) for layer in a0.A0_LAYER_ORDER],
        dtype=torch.long,
        device=device,
    )
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return {
        layer: int(values[index].item())
        for index, layer in enumerate(a0.A0_LAYER_ORDER)
    }


def train_one_epoch_d2(
    model: torch.nn.Module,
    loader: Any,
    sampler: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
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
    teacher, projection = k1._require_resources()
    directions = _require_swd_directions()
    args = k1._ACTIVE_ARGS
    if sampler is not None:
        sampler.set_epoch(epoch)
    model.train()
    teacher.eval()
    projection.eval()
    directions.eval()
    optimizer.zero_grad(set_to_none=True)
    confusion = torch.zeros(common.NUM_CLASSES, common.NUM_CLASSES, dtype=torch.int64)
    ce_sum = valid_pixels = batch_count = optimizer_steps = 0
    feature_sum = swd_sum = total_sum = 0.0
    feature_layer_sums = {layer: 0.0 for layer in a0.A0_LAYER_ORDER}
    swd_layer_sums = {layer: 0.0 for layer in a0.A0_LAYER_ORDER}
    token_sums = {layer: 0 for layer in a0.A0_LAYER_ORDER}
    gradient_records: List[Dict[str, object]] = []
    first_batch: Optional[Dict[str, object]] = None
    last_warmup = 0.0
    possible_steps = math.ceil(len(loader) / accumulation_steps)
    target_steps = min(possible_steps, remaining_optimizer_steps)
    max_batches = min(len(loader), target_steps * accumulation_steps)
    progress = k1.tqdm(
        loader,
        desc=f"Epoch {epoch} [D2 CE+feature+SWD]",
        disable=rank != 0,
    )

    for batch_index, (images, targets, paths) in enumerate(progress):
        if batch_index >= max_batches:
            break
        group_position = batch_index % accumulation_steps
        group_size = (
            min(accumulation_steps, max_batches - batch_index)
            if group_position == 0
            else group_size
        )
        sync_gradients = group_position + 1 == group_size
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        next_step = starting_optimizer_step + optimizer_steps + 1
        warmup = min(1.0, next_step / FEATURE_WARMUP_STEPS)
        sync_context = contextlib.nullcontext()
        if (
            isinstance(model, torch.nn.parallel.DistributedDataParallel)
            and not sync_gradients
        ):
            sync_context = model.no_sync()

        with sync_context:
            with common.autocast_context(device, amp_enabled):
                student_output = model(images)
                if not isinstance(student_output, Mapping):
                    raise RuntimeError("D2 training forward did not expose features")
                logits = student_output["logits"]
                student_features = student_output["features"]
                with torch.no_grad():
                    teacher_features = teacher.extract_features(images)
                layer_losses: Dict[str, torch.Tensor] = {}
                swd_layer_losses: Dict[str, torch.Tensor] = {}
                swd_audit: Dict[str, Dict[str, object]] = {}
                for layer in a0.A0_LAYER_ORDER:
                    projected = projection[layer](teacher_features[layer].detach())
                    layer_losses[layer] = F.mse_loss(
                        student_features[layer].float(), projected.float()
                    )
                    swd_layer_losses[layer], swd_audit[layer] = _swd_layer_loss(
                        student_features[layer],
                        projected,
                        targets,
                        layer,
                        args,
                        next_step,
                        world_size,
                    )

            logits_float = logits.float()
            ce_batch_sum = F.cross_entropy(
                logits_float,
                targets,
                ignore_index=common.IGNORE_INDEX,
                reduction="sum",
            )
            batch_valid = int((targets != common.IGNORE_INDEX).sum().item())
            if batch_valid == 0:
                raise RuntimeError("D2 training batch contains no valid Cityscapes pixels")
            loss_seg = ce_batch_sum / batch_valid
            loss_feat = sum(layer_losses.values()) / len(a0.A0_LAYER_ORDER)
            loss_swd = sum(swd_layer_losses.values()) / len(a0.A0_LAYER_ORDER)
            total_loss = loss_seg + warmup * (
                args.lambda_feat * loss_feat + args.lambda_swd * loss_swd
            )
            # Every rank sees the same global-token SWD value.  Only its local
            # gathered slot carries autograd, so compensate for DDP's gradient
            # averaging in the backward objective only.
            backward_loss = loss_seg + warmup * (
                args.lambda_feat * loss_feat
                + args.lambda_swd * world_size * loss_swd
            )
            losses = [
                loss_seg,
                loss_feat,
                loss_swd,
                total_loss,
                backward_loss,
                *layer_losses.values(),
                *swd_layer_losses.values(),
            ]
            if not all(torch.isfinite(value) for value in losses):
                raise RuntimeError("D2 produced a non-finite loss")

            log_gradients = sync_gradients and (
                next_step == 1 or next_step % args.gradient_log_steps == 0
            )
            grad_record: Optional[Dict[str, object]] = None
            if log_gradients:
                os16 = student_features["os16"]
                grad_seg = torch.autograd.grad(loss_seg, os16, retain_graph=True)[0]
                grad_feat = torch.autograd.grad(loss_feat, os16, retain_graph=True)[0]
                grad_swd = torch.autograd.grad(
                    loss_swd, os16, retain_graph=True, allow_unused=True
                )[0]
                if grad_swd is None:
                    grad_swd = torch.zeros_like(os16)
                grad_total = grad_seg + warmup * (
                    args.lambda_feat * grad_feat
                    + args.lambda_swd * world_size * grad_swd
                )
                raw_swd_norm = _gradient_l2(grad_swd)
                effective_swd_norm = warmup * args.lambda_swd * raw_swd_norm
                effective_feat_norm = (
                    warmup * args.lambda_feat * _gradient_l2(grad_feat)
                )
                grad_record = {
                    "optimizer_step": next_step,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "warmup_weight": warmup,
                    "lambda_feat": args.lambda_feat,
                    "lambda_swd": args.lambda_swd,
                    "distribution_type": "SWD",
                    "distribution_lambda": args.lambda_swd,
                    "distribution_raw_loss": float(loss_swd.detach().item()),
                    "distribution_weighted_loss": float(
                        (warmup * args.lambda_swd * loss_swd).detach().item()
                    ),
                    "grad_l2_seg_os16": _gradient_l2(grad_seg),
                    "grad_l2_feat_os16": _gradient_l2(grad_feat),
                    "grad_l2_swd_os16": raw_swd_norm,
                    "grad_l2_swd_effective_os16": effective_swd_norm,
                    "distribution_feature_gradient_ratio_os16": (
                        effective_swd_norm / (effective_feat_norm + 1e-12)
                    ),
                    "distribution_ce_gradient_ratio_os16": (
                        effective_swd_norm / (_gradient_l2(grad_seg) + 1e-12)
                    ),
                    "gradient_cosine_swd_feature_os16": _gradient_cosine(
                        grad_swd, grad_feat
                    ),
                    "gradient_cosine_swd_ce_os16": _gradient_cosine(
                        grad_swd, grad_seg
                    ),
                    "grad_l2_total_os16": _gradient_l2(grad_total),
                    "grad_l2_total_student": None,
                    "distribution_global_token_count_os4": swd_audit["os4"][
                        "global_token_count"
                    ],
                    "distribution_global_token_count_os8": swd_audit["os8"][
                        "global_token_count"
                    ],
                    "distribution_global_token_count_os16": swd_audit["os16"][
                        "global_token_count"
                    ],
                    "swd_num_slices": NUM_SWD_SLICES,
                    "swd_direction_combined_sha256": directions.metadata()[
                        "combined_sha256"
                    ],
                    "feature_kd_enabled": True,
                    "logit_kd_enabled": False,
                }
            scaler.scale(backward_loss / group_size).backward()

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

        if first_batch is None and starting_optimizer_step == 0 and batch_index == 0:
            first_batch = {
                "experiment": EXPERIMENT,
                "rank": rank,
                "epoch": epoch,
                "micro_batch_index": 0,
                "paths": list(paths),
                "image_tensor_shape": list(images.shape),
                "target_tensor_shape": list(targets.shape),
                "image_tensor_sha256": k0._tensor_sha256(images),
                "target_tensor_sha256": k0._tensor_sha256(targets),
                "valid_pixels": batch_valid,
                "student_feature_shapes": {
                    layer: list(student_features[layer].shape)
                    for layer in a0.A0_LAYER_ORDER
                },
                "teacher_feature_shapes": {
                    layer: list(teacher_features[layer].shape)
                    for layer in a0.A0_LAYER_ORDER
                },
                "projected_teacher_shapes": {
                    layer: list(
                        projection[layer](teacher_features[layer].detach()).shape
                    )
                    for layer in a0.A0_LAYER_ORDER
                },
                "feature_loss_by_layer": {
                    layer: float(layer_losses[layer].detach().item())
                    for layer in a0.A0_LAYER_ORDER
                },
                "swd_loss_by_layer": {
                    layer: float(swd_layer_losses[layer].detach().item())
                    for layer in a0.A0_LAYER_ORDER
                },
                "swd_token_audit": swd_audit,
                "swd_directions": directions.metadata(),
                "feature_loss": float(loss_feat.detach().item()),
                "swd_loss": float(loss_swd.detach().item()),
                "ce_loss": float(loss_seg.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                "warmup_weight": warmup,
                "distribution_spec_sha256": _canonical_sha256(
                    _distribution_spec(args, accumulation_steps, world_size)
                ),
                **k1._resource_hashes(),
                **_swd_resource_hashes(),
            }

        predictions = logits_float.detach().argmax(dim=1)
        confusion += common.confusion_counts(predictions, targets)
        ce_sum += float(ce_batch_sum.detach().item())
        valid_pixels += batch_valid
        feature_value = float(loss_feat.detach().item())
        swd_value = float(loss_swd.detach().item())
        feature_sum += feature_value
        swd_sum += swd_value
        total_sum += float(total_loss.detach().item())
        for layer in a0.A0_LAYER_ORDER:
            feature_layer_sums[layer] += float(layer_losses[layer].detach().item())
            swd_layer_sums[layer] += float(swd_layer_losses[layer].detach().item())
            token_sums[layer] += int(swd_audit[layer]["global_token_count"])
        batch_count += 1
        last_warmup = warmup
        if rank == 0:
            running = common.metrics_from_confusion(confusion, ce_sum, valid_pixels)
            progress.set_postfix(
                {
                    "CE": f"{running['loss']:.4f}",
                    "feat": f"{feature_value:.4f}",
                    "swd": f"{swd_value:.4f}",
                    "mIoU": f"{running['mIoU']:.4f}",
                    "steps": optimizer_steps,
                }
            )

    if optimizer_steps != target_steps:
        raise RuntimeError(
            f"D2 optimizer-step accounting failed: actual={optimizer_steps}, "
            f"expected={target_steps}"
        )
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("D2 teacher unexpectedly received gradients")
    if list(projection.parameters()) or list(directions.parameters()):
        raise RuntimeError("D2 projection/directions unexpectedly became trainable")

    metrics = server_base._reduce_train_metrics(
        confusion, ce_sum, valid_pixels, device, world_size
    )
    global_token_sums = _reduce_global_token_sums(token_sums, device, world_size)
    stat_values = [
        *feature_layer_sums.values(),
        feature_sum,
        swd_sum,
        total_sum,
        *swd_layer_sums.values(),
    ]
    reduced, global_batches = _reduce_d2_statistics(
        stat_values, batch_count, device, world_size
    )
    metrics.update(
        {
            "loss_schema": "hard_label_CE_plus_A0_fixed_feature_MSE_plus_SWD",
            "ce_loss": metrics["loss"],
            "feature_loss": reduced[3],
            "feature_loss_by_layer": dict(
                zip(a0.A0_LAYER_ORDER, reduced[:3])
            ),
            "swd_loss": reduced[4],
            "swd_loss_by_layer": dict(zip(a0.A0_LAYER_ORDER, reduced[6:9])),
            "distribution_loss": reduced[4],
            "distribution_loss_weighted": (
                last_warmup * args.lambda_swd * reduced[4]
            ),
            "total_loss_micro_batch_mean": reduced[5],
            "warmup_weight": last_warmup,
            "micro_batches_global": global_batches,
            "distribution_type": "SWD",
            "distribution_lambda": args.lambda_swd,
            "physical_distribution_batch_size": int(args.batch_size) * world_size,
            "distribution_token_cap_per_layer": DISTRIBUTION_TOKEN_CAP,
            "swd_num_slices": NUM_SWD_SLICES,
            "swd_direction_combined_sha256": directions.metadata()[
                "combined_sha256"
            ],
            "distribution_global_token_count_os4": global_token_sums["os4"],
            "distribution_global_token_count_os8": global_token_sums["os8"],
            "distribution_global_token_count_os16": global_token_sums["os16"],
        }
    )
    return metrics, optimizer_steps, gradient_records, first_batch


def smoke_test_d2(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    amp_enabled: bool,
    rank: int,
) -> None:
    teacher, projection = k1._require_resources()
    directions = _require_swd_directions()
    args = k1._ACTIVE_ARGS
    model.train()
    teacher.eval()
    projection.eval()
    directions.eval()
    images, targets, paths = next(iter(loader))
    images, targets = images.to(device), targets.to(device)
    model.zero_grad(set_to_none=True)
    with common.autocast_context(device, amp_enabled):
        output = model(images)
        if not isinstance(output, Mapping):
            raise RuntimeError("D2 smoke forward did not expose features")
        with torch.no_grad():
            teacher_features = teacher.extract_features(images)
        feature_losses: Dict[str, torch.Tensor] = {}
        swd_losses: Dict[str, torch.Tensor] = {}
        for layer in a0.A0_LAYER_ORDER:
            projected = projection[layer](teacher_features[layer].detach())
            feature_losses[layer] = F.mse_loss(
                output["features"][layer].float(), projected.float()
            )
            swd_losses[layer], _ = _swd_layer_loss(
                output["features"][layer],
                projected,
                targets,
                layer,
                args,
                1,
                int(os.environ.get("WORLD_SIZE", "1")),
            )
    valid = int((targets != common.IGNORE_INDEX).sum().item())
    if valid == 0:
        raise RuntimeError("D2 smoke batch contains no valid pixels")
    ce = F.cross_entropy(
        output["logits"].float(),
        targets,
        ignore_index=common.IGNORE_INDEX,
        reduction="sum",
    ) / valid
    feat = sum(feature_losses.values()) / len(a0.A0_LAYER_ORDER)
    swd = sum(swd_losses.values()) / len(a0.A0_LAYER_ORDER)
    warmup = 1 / FEATURE_WARMUP_STEPS
    total = ce + warmup * (args.lambda_feat * feat + args.lambda_swd * swd)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backward_loss = ce + warmup * (
        args.lambda_feat * feat + args.lambda_swd * world_size * swd
    )
    backward_loss.backward()
    if not all(torch.isfinite(value) for value in (ce, feat, swd, total)):
        raise RuntimeError("D2 smoke produced a non-finite loss")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("D2 smoke produced teacher gradients")
    if list(directions.parameters()) or any(
        buffer.requires_grad for buffer in directions.buffers()
    ):
        raise RuntimeError("D2 smoke found trainable SWD directions")
    if rank == 0:
        builtins.print(
            f"[OK] D2 server smoke: sample={paths[0]}, "
            f"logits={tuple(output['logits'].shape)}, CE={ce.item():.6f}, "
            f"feature={feat.item():.6f}, SWD={swd.item():.6f}, "
            f"total={total.item():.6f}, slices={NUM_SWD_SLICES}"
        )


def _postprocess_metrics_d2(args: argparse.Namespace) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    path = d2_paths(args.output_dir, args.seed)["metrics"]
    if not path.is_file():
        return
    results = json.loads(path.read_text(encoding="utf-8"))
    config_value = results.get("config", {})
    spec = (
        config_value.get("distribution", {})
        if isinstance(config_value, Mapping)
        else {}
    )
    results.update(
        {
            "experiment": EXPERIMENT,
            "experiment_group": EXPERIMENT_GROUP,
            "artifact_type": ARTIFACT_TYPE,
            "protocol": (
                "D2 CE + fixed A0 feature KD + masked teacher-normalized "
                "64-slice SWD; K shared initialization; 80k steps; "
                "dev_local only."
            ),
            "distribution": copy.deepcopy(spec),
            "distribution_spec_sha256": (
                _canonical_sha256(spec) if isinstance(spec, Mapping) else None
            ),
            "test_local_evaluated": False,
        }
    )
    results["loss"] = {
        "hard_label_ce": True,
        "feature_kd": True,
        "distribution_kd": True,
        "distribution_type": "SWD",
        "lambda_feat": args.lambda_feat,
        "lambda_swd": args.lambda_swd,
        "warmup_steps": FEATURE_WARMUP_STEPS,
        "warmup_step_unit": "optimizer_step",
        "logit_kd": False,
    }
    results["teacher"] = {
        "checkpoint": str(args.teacher_checkpoint.resolve()),
        "checkpoint_sha256": k1._TEACHER_CHECKPOINT_SHA256,
        "features_used": list(a0.A0_LAYER_ORDER),
        "logits_used": False,
        "frozen": True,
    }
    results["pca"] = {
        "directory": str(args.pca_dir.resolve()),
        "parameter_record_sha256": k1._PCA_PARAMETER_RECORD_SHA256,
        "projection_parameter_sha256": copy.deepcopy(k1._PROJECTION_HASHES),
        "sampling_manifest_sha256": (
            None
            if k1._PCA_PARAMETER_RECORD is None
            else k1._PCA_PARAMETER_RECORD.get("sampling_manifest_sha256")
        ),
    }
    results["swd_directions"] = _require_swd_directions().metadata()
    results["hashes"] = {
        **dict(results.get("hashes", {})),
        **k1._resource_hashes(),
        **_swd_resource_hashes(),
        "d2_training_script_sha256": common.sha256_file(Path(__file__).resolve()),
    }
    common.write_json_atomic(path, results)
    config_path = d2_paths(args.output_dir, args.seed)["config"]
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(
            {
                "experiment": EXPERIMENT,
                "experiment_group": EXPERIMENT_GROUP,
                "artifact_type": ARTIFACT_TYPE,
                "test_local_evaluated": False,
            }
        )
        common.write_json_atomic(config_path, config)


def run_training(args: argparse.Namespace) -> None:
    global _ACTIVE_ARGS
    global _SWD_DIRECTIONS
    _ACTIVE_ARGS = args
    k1._ACTIVE_ARGS = args
    _ORIGINAL_K1_INSTALL_HOOKS()
    k0.__file__ = str(Path(__file__).resolve())
    k0.EXPERIMENT = EXPERIMENT
    k0.ARTIFACT_TYPE = ARTIFACT_TYPE
    k0.ARTIFACT_FORMAT_VERSION = ARTIFACT_FORMAT_VERSION
    k0.k0_paths = d2_paths
    k0.ensure_shared_initialization = ensure_d2_resources
    k0.build_config = build_config_d2
    k0.build_best_checkpoint = build_best_checkpoint_d2
    k0.train_one_epoch_k0 = train_one_epoch_d2
    k0._smoke_test_k0 = smoke_test_d2
    k1.base.audit_model_shapes = audit_shapes_d2
    common.torch_save_atomic = _patched_torch_save_atomic_d2
    common.evaluate = _evaluate_d2
    k1._postprocess_metrics = _postprocess_metrics_d2
    try:
        k0.run_training(args)
        if not args.smoke_test:
            _postprocess_metrics_d2(args)
    finally:
        k1._postprocess_metrics = _ORIGINAL_K1_POSTPROCESS
        _ORIGINAL_K1_REMOVE_HOOKS()
        k1._TEACHER = None
        k1._PROJECTION = None
        _SWD_DIRECTIONS = None
        _ACTIVE_ARGS = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
