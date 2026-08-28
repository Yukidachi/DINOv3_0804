"""R1 server entry point: K1 feature KD plus cross-image relation KD.

R1 is the first relational candidate in the Cityscapes R-group.  It keeps the
R0/K1 initialization, teacher, A0 pointwise feature target, data stream,
optimizer, scheduler, and evaluation protocol, and adds only the registered
cross-image relation objective::

    L = L_seg + warmup(step) * (L_feat + lambda_r1 * L_R1)

For every native OS=4/8/16 teacher/student tap, R1 performs a label-masked
global average, synchronizes the current physical micro-batch across ranks,
forms a signed cosine BxB matrix, and averages the matrix MSE over all B^2
entries (including the diagonal).  Student synchronization is differentiable;
teacher targets are detached.

Typical two-GPU server command::

    torchrun --standalone --nproc_per_node=2 dino_r1_server.py \
        --seed 42 --batch-size 2 --global-batch-size 8 \
        --num-workers 8 --multiprocessing-context spawn \
        --no-pin-memory --persistent-workers

Two-GPU DDP smoke (required before a formal run)::

    torchrun --standalone --nproc_per_node=2 dino_r1_server.py \
        --device cuda --smoke-test --batch-size 2 --global-batch-size 4 \
        --num-workers 0 --no-persistent-workers --no-pin-memory --no-amp

Windows/local functional smoke (does not replace the DDP smoke)::

    python -B dino_r1_server.py --device cuda --smoke-test \
        --batch-size 4 --global-batch-size 4 --num-workers 0 \
        --no-persistent-workers --no-pin-memory --no-amp
"""

from __future__ import annotations

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
import torch.distributed.nn.functional as dist_nn_functional
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import dino as common
import dino_a0_server as a0
import dino_k0_server as k0
import dino_k1_server as k1
import dino_r0_server as r0
import dino_s2_0 as base
import dino_s2_0_server as server_base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "R_MobileNetV2_RASPP_server"
K_GROUP_OUTPUT_DIR = SCRIPT_DIR / "result" / "K_MobileNetV2_RASPP_server"
K1_REFERENCE_DIR = K_GROUP_OUTPUT_DIR / "K1" / "seed_42"

EXPERIMENT = "R1"
EXPERIMENT_GROUP = "R_MobileNetV2_RASPP_server"
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_r1_cross_image_relation_kd"
ARTIFACT_FORMAT_VERSION = 1
FORMAL_SEEDS = (42,)

RELATION_EPSILON = 1e-6
LAMBDA_R1 = 0.03
ALLOWED_LAMBDA_R1 = (0.015, 0.03, 0.06)
GRADIENT_GATE_MIN = 0.05
GRADIENT_GATE_MAX = 0.20
GRADIENT_CE_STOP_RATIO = 2.0
GRADIENT_CE_STOP_CONSECUTIVE = 3
FIXED_GRADIENT_AUDIT_STEPS = (1, 4_000, 20_000, 40_000, 60_000, 80_000)
FIRST_BATCH_ABS_TOLERANCE = r0.FIRST_BATCH_ABS_TOLERANCE
FIRST_BATCH_REL_TOLERANCE = r0.FIRST_BATCH_REL_TOLERANCE


_ORIGINAL_K1_BUILD_CONFIG = k1.build_config
_ORIGINAL_K1_BUILD_BEST_CHECKPOINT = k1.build_best_checkpoint
_ORIGINAL_K1_SMOKE_TEST = k1.smoke_test_k1
_ORIGINAL_K1_POSTPROCESS = k1._postprocess_metrics
_ORIGINAL_K1_AUDIT_SHAPES = k1.audit_k1_shapes
_ORIGINAL_K_SHARED_INITIALIZATION = k1._ORIGINAL_ENSURE_SHARED_INITIALIZATION
_ORIGINAL_TQDM = k1.tqdm

_K1_REFERENCE: Optional[Dict[str, object]] = None
_K1_REFERENCE_VALIDATION: Optional[Dict[str, object]] = None
_R0_GATE: Optional[Dict[str, object]] = None
_RELATION_SPEC: Optional[Dict[str, object]] = None
_REFERENCE_TESTS: Optional[Dict[str, object]] = None
_FIRST_BATCH_BASE_EQUIVALENCE: Optional[Dict[str, object]] = None
_RELATION_GATE_CONSECUTIVE_EXCESS = 0


def parse_args() -> Any:
    """Reuse K1's CLI while adding and validating the R1 relation weight."""

    saved_default = k1.DEFAULT_OUTPUT_DIR
    saved_argparse = k1.argparse

    class R1ArgparseProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(saved_argparse, name)

        @staticmethod
        def ArgumentParser(*parser_args: Any, **parser_kwargs: Any):
            parser_kwargs["description"] = (
                "R1 MobileNetV2+R-ASPP: hard-label CE plus the locked A0 "
                "feature target and masked-GAP cross-image relation KD."
            )
            parser = saved_argparse.ArgumentParser(*parser_args, **parser_kwargs)
            parser.add_argument(
                "--lambda-r1",
                type=float,
                default=LAMBDA_R1,
                help="Fixed R1 relation weight; registered calibration values only.",
            )
            return parser

    k1.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    k1.argparse = R1ArgparseProxy()
    try:
        args = k1.parse_args()
    finally:
        k1.DEFAULT_OUTPUT_DIR = saved_default
        k1.argparse = saved_argparse

    if args.seed != 42:
        raise SystemExit("R1 first-round screening is pre-registered for --seed 42")
    if not any(
        math.isclose(args.lambda_r1, value, rel_tol=0.0, abs_tol=1e-12)
        for value in ALLOWED_LAMBDA_R1
    ):
        raise SystemExit(
            "--lambda-r1 must be one of the registered values 0.015, 0.03, 0.06"
        )
    if not args.smoke_test and args.max_steps != 80_000:
        raise SystemExit("Formal R1 is locked to exactly 80,000 optimizer steps")
    if not args.smoke_test and args.eval_every_steps != 5_000:
        raise SystemExit("Formal R1 is locked to --eval-every-steps 5000")
    if not args.smoke_test and args.gradient_log_steps != 500:
        raise SystemExit("Formal R1 is locked to --gradient-log-steps 500")
    if args.output_dir.resolve() == K_GROUP_OUTPUT_DIR.resolve():
        raise SystemExit(
            "R1 output must not point at the K-group directory; use the separate "
            "R_MobileNetV2_RASPP_server output root"
        )
    return args


def r1_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    """Use K0/K1 artifact names below the independent R1 directory."""

    original = k1._ORIGINAL_K0_PATHS(output_dir, seed)
    run_dir = output_dir.resolve() / EXPERIMENT / f"seed_{seed}"
    return {
        key: run_dir if key == "run_dir" else run_dir / value.name
        for key, value in original.items()
    }


def _canonical_sha256(value: Mapping[str, object]) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _r0_metrics_path(args: Any) -> Path:
    return args.output_dir.resolve() / "R0" / f"seed_{args.seed}" / "metrics.json"


def _validate_r0_gate(args: Any) -> Dict[str, object]:
    """Require the completed R0/K1 equivalence gate before a formal R1 run."""

    metrics_path = _r0_metrics_path(args)
    if not metrics_path.is_file():
        if args.smoke_test:
            return {
                "required_for_formal_run": True,
                "checked": False,
                "passed": None,
                "reason": "R0 metrics are absent; protocol smoke is allowed",
                "metrics_path": str(metrics_path),
            }
        raise FileNotFoundError(
            "Formal R1 is gated on a completed, accepted R0 seed=42 run: "
            f"{metrics_path}"
        )

    metrics = _read_json(metrics_path)
    failures: List[str] = []
    if metrics.get("experiment") != "R0":
        failures.append("the gate artifact is not an R0 result")
    equivalence = metrics.get("r0_k1_equivalence")
    if not isinstance(equivalence, Mapping) or not bool(equivalence.get("passed")):
        failures.append("R0/K1 equivalence did not pass")
    if metrics.get("test_local_evaluated") is not False:
        failures.append("R0 does not record test_local_evaluated=false")

    hashes = metrics.get("hashes", {})
    if not isinstance(hashes, Mapping):
        failures.append("R0 metrics has no hash mapping")
        hashes = {}
    local_r0_hash = common.sha256_file(Path(r0.__file__).resolve())
    recorded_r0_hash = hashes.get("r0_training_script_sha256")
    if recorded_r0_hash != local_r0_hash:
        failures.append(
            "the current dino_r0_server.py differs from the script accepted by R0: "
            f"local={local_r0_hash}, recorded={recorded_r0_hash}"
        )

    result = {
        "required_for_formal_run": True,
        "checked": True,
        "passed": not failures,
        "failures": failures,
        "metrics_path": str(metrics_path),
        "metrics_sha256": common.sha256_file(metrics_path),
        "r0_training_script_sha256": local_r0_hash,
        "r0_best_dev_mIoU": (
            metrics.get("best_dev_metrics", {}).get("mIoU")
            if isinstance(metrics.get("best_dev_metrics"), Mapping)
            else None
        ),
        "r0_selected_model_state_sha256": (
            hashes.get("selected_model_state_sha256")
        ),
    }
    if failures and not args.smoke_test:
        raise RuntimeError("R1 R0-gate validation failed:\n- " + "\n- ".join(failures))
    return result


def _relation_spec(args: Any, accumulation_steps: int, world_size: int) -> Dict[str, object]:
    nominal_physical_batch = int(args.batch_size) * int(world_size)
    effective_batch = nominal_physical_batch * int(accumulation_steps)
    return {
        "enabled": True,
        "active_relation_types": ["R1_cross_image"],
        "relation_feature_source": {
            "teacher": "native OS=4/8/16 features",
            "student": "native OS=4/8/16 features",
            "a0_projected_features_used_for_relation": False,
        },
        "epsilon": RELATION_EPSILON,
        "nominal_physical_relation_batch_size": nominal_physical_batch,
        "physical_relation_batch_size": nominal_physical_batch,
        "effective_optimizer_batch_size": effective_batch,
        "accumulated_batches_used_for_relation": False,
        "tail_batch_policy": (
            "preserve the locked K1 drop_last=False stream; if the final synchronized "
            "micro-batch is partial, use its actual B and divide by actual B^2; never "
            "cache, pad, or combine samples across optimizer steps"
        ),
        "mask_policy": "nearest-resized targets != 255; every image must retain a valid location",
        "layer_aggregation": "equal mean over native OS=4/8/16 relation losses",
        "matrix_dtype": "float32",
        "normalization": "row / (L2 norm + epsilon)",
        "r1": {
            "enabled": True,
            "representation": "masked GAP then BxB signed cosine matrix",
            "gather": "differentiable student all_gather; detached teacher all_gather",
            "diagonal_policy": "keep",
            "reduction": "sum squared error divided by actual B^2",
            "lambda": float(args.lambda_r1),
        },
        "r2": {
            "enabled": False,
            "pool_size": [8, 16],
            "representation": "per-image 128x128 signed token cosine matrix",
            "diagonal_policy": "keep for valid tokens",
            "reduction": "sum squared error divided by valid-pair count",
            "lambda": 0.0,
        },
        "relation_warmup_steps": 4_000,
        "relation_warmup_shared_with_feature_kd": True,
        "relation_gradient_gate": {
            "target_relation_to_feature_effective_ratio": [
                GRADIENT_GATE_MIN,
                GRADIENT_GATE_MAX,
            ],
            "stop_if_relation_to_ce_exceeds": GRADIENT_CE_STOP_RATIO,
            "consecutive_records_before_stop": GRADIENT_CE_STOP_CONSECUTIVE,
            "lambda_is_fixed_during_formal_training": True,
        },
    }


def _ensure_locked_k_shared_initialization(
    model: base.MobileNetV2RASPPStudent,
    args: Any,
    _r_output_dir: Path,
    seed: int,
    rank: int,
    world_size: int,
) -> Tuple[str, str, Path]:
    path = k0._shared_init_path(K_GROUP_OUTPUT_DIR, seed)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(
            "R1 requires the existing K-group shared initialization and will not "
            f"generate an R-specific replacement: {path}"
        )
    return _ORIGINAL_K_SHARED_INITIALIZATION(
        model, args, K_GROUP_OUTPUT_DIR, seed, rank, world_size
    )


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"R1 {name} contains a non-finite value")


def _resize_valid_mask(targets: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    if targets.ndim != 3:
        raise RuntimeError(f"R1 targets must be [B,H,W], got {tuple(targets.shape)}")
    mask = (targets != common.IGNORE_INDEX).unsqueeze(1).to(dtype=torch.float32)
    return F.interpolate(mask, size=tuple(size), mode="nearest")


def _has_valid_relation_location(target: torch.Tensor) -> bool:
    """Check the post-augmentation target at every native relation stride.

    ``CityscapesManifestDataset`` already rejects crops with no valid pixel at
    the image resolution.  A very small valid island can nevertheless vanish
    when the mask is resized to OS=16 with nearest-neighbor interpolation.
    R1 needs one valid location per image at every tap, so those rare crops are
    resampled before they enter a batch.
    """

    if target.ndim != 2:
        raise RuntimeError(
            f"R1 transformed target must be [H,W], got {tuple(target.shape)}"
        )
    height, width = target.shape
    if height % 16 or width % 16:
        raise RuntimeError(
            "R1 relation target dimensions must be divisible by 16, "
            f"got {(int(height), int(width))}"
        )
    valid = (target != common.IGNORE_INDEX).to(dtype=torch.float32)
    for stride in (4, 8, 16):
        resized = F.interpolate(
            valid[None, None],
            size=(height // stride, width // stride),
            mode="nearest",
        )
        if not bool((resized > 0).any().item()):
            return False
    return True


class R1TrainDataset(torch.utils.data.Dataset):
    """R1 train view with a per-tap valid-location rejection gate.

    The wrapped dataset preserves the existing transform, random stream,
    manifest order, and ``reject_all_ignore`` behavior.  The extra gate only
    resamples a crop whose valid pixels would disappear at a relation tap.
    """

    def __init__(
        self,
        dataset_root: Path,
        entries: Sequence[Tuple[str, str]],
        transform: Any,
        max_attempts: int = 64,
    ) -> None:
        self._base = common.CityscapesManifestDataset(
            dataset_root=dataset_root,
            entries=entries,
            transform=transform,
            reject_all_ignore=True,
        )
        self.max_attempts = int(max_attempts)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int):
        image_rel = self._base.entries[index][0]
        last_error: Optional[Exception] = None
        for _ in range(self.max_attempts):
            try:
                image_tensor, target, path = self._base[index]
            except RuntimeError as error:
                last_error = error
                continue
            if _has_valid_relation_location(target):
                return image_tensor, target, path
        detail = "" if last_error is None else f"; last_error={last_error}"
        raise RuntimeError(
            "R1 could not draw a crop with valid locations at OS=4/8/16 "
            f"for {image_rel} after {self.max_attempts} attempts{detail}"
        )


def build_train_loader_r1(
    args: Any,
    dataset_root: Path,
    entries_by_split: Mapping[str, Sequence[Tuple[str, str]]],
    device: torch.device,
    rank: int,
    world_size: int,
):
    """Build K1's loader with only the R1 relation-validity resampling gate."""

    dataset = R1TrainDataset(
        dataset_root=dataset_root,
        entries=entries_by_split["train_local"],
        transform=common.CityscapesTrainTransform(
            crop_size=(args.crop_height, args.crop_width),
            scale_range=(args.scale_min, args.scale_max),
        ),
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
        **server_base._loader_kwargs(args, device, args.num_workers),
    )
    return loader, sampler, generator


def masked_global_average(
    features: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 masked GAP vectors and per-image valid feature counts."""

    if features.ndim != 4:
        raise RuntimeError(f"R1 features must be [B,C,H,W], got {tuple(features.shape)}")
    _assert_finite("native feature", features)
    mask = _resize_valid_mask(targets, features.shape[-2:])
    counts = mask.sum(dim=(2, 3))
    if bool((counts <= 0).any().item()):
        raise RuntimeError("R1 masked GAP found an image with no valid feature location")
    gap = (features.float() * mask).sum(dim=(2, 3)) / counts
    _assert_finite("masked GAP", gap)
    return gap, counts.squeeze(1)


def _row_normalize(vectors: torch.Tensor, epsilon: float) -> torch.Tensor:
    _assert_finite("relation vector", vectors)
    norms = vectors.float().norm(2, dim=1, keepdim=True)
    normalized = vectors.float() / (norms + epsilon)
    _assert_finite("normalized relation vector", normalized)
    return normalized


def cosine_relation_matrix(vectors: torch.Tensor, epsilon: float = RELATION_EPSILON) -> torch.Tensor:
    normalized = _row_normalize(vectors, epsilon)
    matrix = normalized @ normalized.transpose(0, 1)
    _assert_finite("cosine relation matrix", matrix)
    return matrix


def relation_matrix_mse(student_matrix: torch.Tensor, teacher_matrix: torch.Tensor) -> torch.Tensor:
    if student_matrix.shape != teacher_matrix.shape or student_matrix.ndim != 2:
        raise RuntimeError(
            "R1 relation matrices must have the same [B,B] shape: "
            f"student={tuple(student_matrix.shape)}, teacher={tuple(teacher_matrix.shape)}"
        )
    if student_matrix.shape[0] != student_matrix.shape[1]:
        raise RuntimeError("R1 relation matrix must be square")
    batch = int(student_matrix.shape[0])
    if batch < 1:
        raise RuntimeError("R1 relation matrix cannot be empty")
    loss = (student_matrix - teacher_matrix.detach()).square().sum() / (batch * batch)
    _assert_finite("relation loss", loss)
    return loss


def _synchronized_local_batch_size(
    local_batch: int, device: torch.device, world_size: int
) -> Tuple[int, List[int]]:
    if world_size <= 1:
        return local_batch, [local_batch]
    local = torch.tensor([local_batch], device=device, dtype=torch.int64)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    sizes = [int(value.item()) for value in gathered]
    if len(set(sizes)) != 1:
        raise RuntimeError(f"R1 requires equal per-rank micro-batches, got {sizes}")
    return sum(sizes), sizes


def _gather_student_vectors(vectors: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1:
        return vectors
    gathered = dist_nn_functional.all_gather(vectors.contiguous())
    return torch.cat(tuple(gathered), dim=0)


def _gather_teacher_vectors(vectors: torch.Tensor, world_size: int) -> torch.Tensor:
    detached = vectors.detach().contiguous()
    if world_size <= 1:
        return detached
    gathered = [torch.empty_like(detached) for _ in range(world_size)]
    dist.all_gather(gathered, detached)
    return torch.cat(gathered, dim=0).detach()


def _relation_from_gap(
    student_gap: torch.Tensor,
    teacher_gap: torch.Tensor,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_student = _gather_student_vectors(student_gap.float(), world_size)
    global_teacher = _gather_teacher_vectors(teacher_gap.float(), world_size)
    student_matrix = cosine_relation_matrix(global_student)
    teacher_matrix = cosine_relation_matrix(global_teacher).detach()
    loss = relation_matrix_mse(student_matrix, teacher_matrix)
    return loss, student_matrix, teacher_matrix


def r1_relation_losses(
    student_features: Mapping[str, torch.Tensor],
    teacher_features: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    world_size: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, object]]:
    physical_batch, per_rank_sizes = _synchronized_local_batch_size(
        int(targets.shape[0]), targets.device, world_size
    )
    layer_losses: Dict[str, torch.Tensor] = {}
    layer_audit: Dict[str, object] = {}
    for layer in a0.A0_LAYER_ORDER:
        student_gap, student_counts = masked_global_average(student_features[layer], targets)
        with torch.no_grad():
            teacher_gap, teacher_counts = masked_global_average(
                teacher_features[layer].detach(), targets
            )
        loss, student_matrix, teacher_matrix = _relation_from_gap(
            student_gap, teacher_gap, world_size
        )
        if list(student_matrix.shape) != [physical_batch, physical_batch]:
            raise RuntimeError(
                f"R1 {layer} matrix shape mismatch: {list(student_matrix.shape)} "
                f"!= {[physical_batch, physical_batch]}"
            )
        layer_losses[layer] = loss
        layer_audit[layer] = {
            "student_gap_shape": list(student_gap.shape),
            "teacher_gap_shape": list(teacher_gap.shape),
            "student_valid_locations_local": [
                int(value) for value in student_counts.detach().cpu().tolist()
            ],
            "teacher_valid_locations_local": [
                int(value) for value in teacher_counts.detach().cpu().tolist()
            ],
            "matrix_shape": list(student_matrix.shape),
            "student_matrix": student_matrix.detach().cpu().tolist(),
            "teacher_matrix": teacher_matrix.detach().cpu().tolist(),
            "loss": float(loss.detach().item()),
        }
    total = sum(layer_losses.values()) / len(a0.A0_LAYER_ORDER)
    _assert_finite("three-layer relation loss", total)
    return total, layer_losses, {
        "physical_batch_size": physical_batch,
        "per_rank_batch_sizes": per_rank_sizes,
        "valid_token_count": physical_batch,
        "valid_pair_count": physical_batch * physical_batch,
        "layers": layer_audit,
    }


def _gradient_l2(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm(2).item())


def _gradient_cosine(first: torch.Tensor, second: torch.Tensor) -> Optional[float]:
    first_flat = first.detach().float().reshape(-1)
    second_flat = second.detach().float().reshape(-1)
    denominator = first_flat.norm(2) * second_flat.norm(2)
    if not bool(torch.isfinite(denominator).item()) or float(denominator.item()) <= 0.0:
        return None
    value = torch.dot(first_flat, second_flat) / denominator
    if not bool(torch.isfinite(value).item()):
        return None
    return float(value.item())


def _run_local_reference_tests(device: torch.device) -> Dict[str, object]:
    targets = torch.tensor(
        [
            [[0, 0], [255, 255]],
            [[0, 255], [0, 255]],
            [[0, 0], [0, 0]],
            [[255, 0], [255, 0]],
        ],
        device=device,
        dtype=torch.long,
    )
    base_values = torch.arange(1, 4 * 3 * 2 * 2 + 1, device=device, dtype=torch.float32)
    features = base_values.reshape(4, 3, 2, 2) / 17.0
    gap, _counts = masked_global_average(features, targets)

    ignore_changed = features.clone()
    ignore_mask = (targets == common.IGNORE_INDEX).unsqueeze(1).expand_as(ignore_changed)
    ignore_changed[ignore_mask] += 1000.0
    gap_ignore_changed, _ = masked_global_average(ignore_changed, targets)
    if not torch.allclose(gap, gap_ignore_changed, atol=0.0, rtol=0.0):
        raise RuntimeError("R1 reference test: ignore pixels changed masked GAP")

    valid_changed = features.clone()
    valid_changed[0, :, 0, 0] += 1.0
    gap_valid_changed, _ = masked_global_average(valid_changed, targets)
    if torch.allclose(gap, gap_valid_changed):
        raise RuntimeError("R1 reference test: valid-pixel change did not change GAP")

    matrix = cosine_relation_matrix(gap)
    scaled_matrix = cosine_relation_matrix(gap * 7.0)
    if not torch.allclose(matrix, scaled_matrix, atol=2e-6, rtol=2e-6):
        raise RuntimeError("R1 reference test: positive scaling changed cosine matrix")
    zero_loss = relation_matrix_mse(matrix, matrix.clone())
    if float(zero_loss.item()) > 1e-12:
        raise RuntimeError("R1 reference test: identical matrices did not give zero loss")

    signed = cosine_relation_matrix(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]], device=device)
    )
    if float(signed[0, 1].item()) >= 0.0:
        raise RuntimeError("R1 reference test: signed negative cosine was not retained")

    student = gap.clone().requires_grad_(True)
    teacher = torch.arange(1, 4 * 5 + 1, device=device, dtype=torch.float32).reshape(4, 5)
    student_matrix = cosine_relation_matrix(student)
    teacher_matrix = cosine_relation_matrix(teacher).detach()
    loss = relation_matrix_mse(student_matrix, teacher_matrix)
    loss.backward()
    if student.grad is None or _gradient_l2(student.grad) <= 0.0:
        raise RuntimeError("R1 reference test: relation gradient did not reach student")
    if teacher.grad is not None:
        raise RuntimeError("R1 reference test: teacher unexpectedly received a gradient")

    permutation = torch.tensor([2, 0, 3, 1], device=device)
    permuted_loss = relation_matrix_mse(
        cosine_relation_matrix(student.detach()[permutation]),
        cosine_relation_matrix(teacher[permutation]),
    )
    if not torch.allclose(loss.detach(), permuted_loss, atol=1e-7, rtol=1e-7):
        raise RuntimeError("R1 reference test: joint sample permutation changed loss")
    teacher_only_permuted = relation_matrix_mse(
        cosine_relation_matrix(student.detach()),
        cosine_relation_matrix(teacher[permutation]),
    )
    if torch.allclose(loss.detach(), teacher_only_permuted, atol=1e-7, rtol=1e-7):
        raise RuntimeError("R1 reference test: teacher-only permutation did not change loss")

    zero_matrix = cosine_relation_matrix(torch.zeros(2, 3, device=device))
    if not bool(torch.isfinite(zero_matrix).all().item()):
        raise RuntimeError("R1 reference test: zero-norm handling is non-finite")
    try:
        cosine_relation_matrix(torch.tensor([[float("nan"), 0.0]], device=device))
    except RuntimeError:
        nonfinite_rejected = True
    else:
        nonfinite_rejected = False
    if not nonfinite_rejected:
        raise RuntimeError("R1 reference test: non-finite input was not rejected")

    row_norms = _row_normalize(gap, RELATION_EPSILON).norm(2, dim=1)
    return {
        "passed": True,
        "matrix_shape": list(matrix.shape),
        "denominator": int(matrix.numel()),
        "row_norm_min": float(row_norms.min().item()),
        "row_norm_max": float(row_norms.max().item()),
        "ignore_invariance": True,
        "valid_pixel_sensitivity": True,
        "positive_scale_invariance": True,
        "signed_cosine_retained": True,
        "diagonal_kept": True,
        "different_teacher_student_channels": True,
        "teacher_detached": True,
        "zero_norm_finite": True,
        "nonfinite_rejected": True,
    }


def _run_distributed_reference_test(
    device: torch.device, world_size: int
) -> Dict[str, object]:
    global_batch = 4
    if global_batch % world_size != 0:
        raise RuntimeError(
            f"R1 reference global batch {global_batch} is not divisible by world_size={world_size}"
        )
    rank = dist.get_rank() if world_size > 1 else 0
    local_batch = global_batch // world_size
    global_student = torch.sin(
        torch.arange(1, global_batch * 3 + 1, device=device, dtype=torch.float32)
    ).reshape(global_batch, 3)
    global_teacher = torch.cos(
        torch.arange(1, global_batch * 5 + 1, device=device, dtype=torch.float32) / 3.0
    ).reshape(global_batch, 5)
    start = rank * local_batch
    end = start + local_batch
    local_student = global_student[start:end].clone().requires_grad_(True)
    local_teacher = global_teacher[start:end].clone()
    distributed_loss, distributed_student_matrix, distributed_teacher_matrix = (
        _relation_from_gap(local_student, local_teacher, world_size)
    )
    distributed_gradient = torch.autograd.grad(distributed_loss, local_student)[0]

    reference_student = global_student.clone().requires_grad_(True)
    reference_student_matrix = cosine_relation_matrix(reference_student)
    reference_teacher_matrix = cosine_relation_matrix(global_teacher)
    reference_loss = relation_matrix_mse(
        reference_student_matrix, reference_teacher_matrix
    )
    reference_gradient = torch.autograd.grad(reference_loss, reference_student)[0]

    if not torch.allclose(
        distributed_student_matrix,
        reference_student_matrix.detach(),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise RuntimeError("R1 DDP reference student matrix mismatch")
    if not torch.allclose(
        distributed_teacher_matrix,
        reference_teacher_matrix.detach(),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise RuntimeError("R1 DDP reference teacher matrix mismatch")
    if not torch.allclose(distributed_loss, reference_loss.detach(), atol=1e-7, rtol=1e-7):
        raise RuntimeError("R1 DDP reference loss mismatch")

    # Every rank evaluates the same global relation loss.  Autograd all_gather
    # therefore sums one identical loss derivative per rank; DDP's subsequent
    # parameter-gradient averaging cancels this factor exactly.
    expected_local_gradient = reference_gradient[start:end].detach() * world_size
    if not torch.allclose(
        distributed_gradient,
        expected_local_gradient,
        atol=2e-6,
        rtol=2e-6,
    ):
        maximum_error = float(
            (distributed_gradient - expected_local_gradient).abs().max().item()
        )
        raise RuntimeError(
            "R1 DDP differentiable-gather gradient mismatch; "
            f"rank={rank}, max_abs_error={maximum_error}"
        )

    # Also verify the exact shared-parameter gradient after the averaging that
    # DDP applies to replicated student parameters.
    global_inputs = torch.cos(
        torch.arange(1, global_batch * 3 + 1, device=device, dtype=torch.float32) / 5.0
    ).reshape(global_batch, 3)
    initial_weight = torch.tensor(
        [[0.8, -0.2, 0.1], [0.3, 0.7, -0.4], [-0.1, 0.2, 0.9]],
        device=device,
        dtype=torch.float32,
    )
    distributed_weight = initial_weight.clone().requires_grad_(True)
    distributed_vectors = global_inputs[start:end] @ distributed_weight
    distributed_parameter_loss, _student_matrix, _teacher_matrix = _relation_from_gap(
        distributed_vectors, local_teacher, world_size
    )
    distributed_parameter_gradient = torch.autograd.grad(
        distributed_parameter_loss, distributed_weight
    )[0]
    if world_size > 1:
        dist.all_reduce(distributed_parameter_gradient, op=dist.ReduceOp.SUM)
        distributed_parameter_gradient /= world_size

    reference_weight = initial_weight.clone().requires_grad_(True)
    reference_vectors = global_inputs @ reference_weight
    reference_parameter_loss = relation_matrix_mse(
        cosine_relation_matrix(reference_vectors), reference_teacher_matrix
    )
    reference_parameter_gradient = torch.autograd.grad(
        reference_parameter_loss, reference_weight
    )[0]
    if not torch.allclose(
        distributed_parameter_gradient,
        reference_parameter_gradient,
        atol=2e-6,
        rtol=2e-6,
    ):
        maximum_error = float(
            (distributed_parameter_gradient - reference_parameter_gradient)
            .abs()
            .max()
            .item()
        )
        raise RuntimeError(
            "R1 DDP-averaged shared-parameter gradient mismatch; "
            f"rank={rank}, max_abs_error={maximum_error}"
        )
    local_gradient_norm = torch.tensor(
        [_gradient_l2(distributed_gradient)], device=device, dtype=torch.float64
    )
    if world_size > 1:
        gathered_gradient_norms = [
            torch.empty_like(local_gradient_norm) for _ in range(world_size)
        ]
        dist.all_gather(gathered_gradient_norms, local_gradient_norm)
        gradient_norms_by_rank = [
            float(value.item()) for value in gathered_gradient_norms
        ]
    else:
        gradient_norms_by_rank = [float(local_gradient_norm.item())]
    return {
        "passed": True,
        "world_size": world_size,
        "global_batch": global_batch,
        "local_batch": local_batch,
        "matrix_shape": list(distributed_student_matrix.shape),
        "loss": float(distributed_loss.detach().item()),
        "gradient_contract": (
            "autograd-all_gather local activation gradient equals world_size times "
            "the single-process slice; DDP parameter averaging restores the exact "
            "single-process global gradient"
        ),
        "student_gradient_l2_by_rank": gradient_norms_by_rank,
        "shared_parameter_gradient_matches_single_process": True,
    }


def run_relation_reference_tests(
    device: torch.device, world_size: int
) -> Dict[str, object]:
    local = _run_local_reference_tests(device)
    distributed = _run_distributed_reference_test(device, world_size)
    return {
        "passed": True,
        "local": local,
        "distributed": distributed,
        "formal_definition": {
            "diagonal": "kept",
            "reduction": "all B^2 entries",
            "epsilon": RELATION_EPSILON,
            "student_gather_autograd": True,
            "teacher_target_detached": True,
        },
    }


def build_config_r1(
    args: Any,
    accumulation_steps: int,
    world_size: int,
    device: torch.device,
    shared_init_state_sha256: str,
    shared_init_file_sha256: str,
) -> Dict[str, object]:
    global _RELATION_SPEC
    global _REFERENCE_TESTS

    config = _ORIGINAL_K1_BUILD_CONFIG(
        args,
        accumulation_steps,
        world_size,
        device,
        shared_init_state_sha256,
        shared_init_file_sha256,
    )
    _RELATION_SPEC = _relation_spec(args, accumulation_steps, world_size)
    _REFERENCE_TESTS = run_relation_reference_tests(device, world_size)
    relation_hash = _canonical_sha256(_RELATION_SPEC)

    if not args.smoke_test:
        if world_size != 2:
            raise RuntimeError(f"Formal R1 requires world_size=2, got {world_size}")
        if _RELATION_SPEC["physical_relation_batch_size"] != 4:
            raise RuntimeError("Formal R1 requires nominal physical_relation_batch_size=4")
        if _RELATION_SPEC["effective_optimizer_batch_size"] != 8:
            raise RuntimeError("Formal R1 requires effective_optimizer_batch_size=8")

    config["experiment"] = EXPERIMENT
    config["experiment_group"] = EXPERIMENT_GROUP
    config["artifact_type"] = ARTIFACT_TYPE
    config["server_entry_point"] = str(Path(__file__).resolve())
    config["formal_seeds"] = list(FORMAL_SEEDS)
    config["shared_initialization"] = {
        "source_group": "K_MobileNetV2_RASPP_server",
        "path": str(k0._shared_init_path(K_GROUP_OUTPUT_DIR, args.seed).resolve()),
        "state_sha256": shared_init_state_sha256,
        "file_sha256": shared_init_file_sha256,
        "r_specific_initialization_created": False,
    }
    config["relation"] = copy.deepcopy(_RELATION_SPEC)
    config["relation_spec_sha256"] = relation_hash
    config["relation_reference_tests"] = copy.deepcopy(_REFERENCE_TESTS)
    config["r0_gate"] = copy.deepcopy(_R0_GATE)
    config["k1_reference_validation"] = copy.deepcopy(_K1_REFERENCE_VALIDATION)
    loss = dict(config.get("loss", {}))
    loss.update(
        {
            "relation_kd": True,
            "relation_r1": True,
            "relation_r2": False,
            "lambda_r1": float(args.lambda_r1),
            "lambda_r2": 0.0,
            "total": "CE + warmup * (lambda_feat * feature + lambda_r1 * R1)",
        }
    )
    config["loss"] = loss
    return config


def audit_shapes_r1(
    model: base.MobileNetV2RASPPStudent,
    device: torch.device,
    height: int,
    width: int,
    amp_enabled: bool,
) -> Dict[str, object]:
    audit = _ORIGINAL_K1_AUDIT_SHAPES(model, device, height, width, amp_enabled)
    audit["experiment"] = EXPERIMENT
    audit["relation"] = {
        "enabled": True,
        "type": "R1_cross_image",
        "native_teacher_student_taps": list(a0.A0_LAYER_ORDER),
        "a0_projection_used_only_by_pointwise_feature_anchor": True,
        "masked_gap_output": "[local_batch, native_channels]",
        "formal_global_matrix_shape": [4, 4],
        "matrix_dtype": "float32",
    }
    return audit


def build_best_checkpoint_r1(*args: Any, **kwargs: Any) -> Dict[str, object]:
    payload = _ORIGINAL_K1_BUILD_BEST_CHECKPOINT(*args, **kwargs)
    payload["experiment"] = EXPERIMENT
    payload["experiment_group"] = EXPERIMENT_GROUP
    payload["artifact_type"] = ARTIFACT_TYPE
    payload["relation"] = copy.deepcopy(_RELATION_SPEC)
    payload["relation_spec_sha256"] = (
        None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC)
    )
    payload["r0_gate"] = copy.deepcopy(_R0_GATE)
    payload["k1_reference_validation"] = copy.deepcopy(_K1_REFERENCE_VALIDATION)
    payload["r1_first_batch_base_equivalence"] = copy.deepcopy(
        _FIRST_BATCH_BASE_EQUIVALENCE
    )
    return payload


def _patched_torch_save_atomic_r1(payload: object, path: Path) -> None:
    if isinstance(payload, Mapping) and payload.get("artifact_type") == ARTIFACT_TYPE:
        payload = dict(payload)
        payload["experiment"] = EXPERIMENT
        payload["experiment_group"] = EXPERIMENT_GROUP
        payload["relation"] = copy.deepcopy(_RELATION_SPEC)
        payload["relation_spec_sha256"] = (
            None if _RELATION_SPEC is None else _canonical_sha256(_RELATION_SPEC)
        )
        payload["r0_gate"] = copy.deepcopy(_R0_GATE)
        payload["k1_reference_validation"] = copy.deepcopy(
            _K1_REFERENCE_VALIDATION
        )
        payload["r1_first_batch_base_equivalence"] = copy.deepcopy(
            _FIRST_BATCH_BASE_EQUIVALENCE
        )
        payload["hashes"] = {
            **dict(payload.get("hashes", {})),
            **k1._resource_hashes(),
            "relation_spec_sha256": payload["relation_spec_sha256"],
            "r0_gate_metrics_sha256": (
                None if _R0_GATE is None else _R0_GATE.get("metrics_sha256")
            ),
            "k1_reference_metrics_sha256": (
                None
                if _K1_REFERENCE_VALIDATION is None
                else _K1_REFERENCE_VALIDATION.get("reference_metrics_sha256")
            ),
            "r1_training_script_sha256": common.sha256_file(Path(__file__).resolve()),
        }
        payload["pca_parameters_sha256_record"] = copy.deepcopy(
            k1._PCA_PARAMETER_RECORD
        )
    k1._ORIGINAL_TORCH_SAVE_ATOMIC(payload, path)


def _patched_evaluate_r1(*args: Any, **kwargs: Any):
    split_name = kwargs.get("split_name")
    if isinstance(split_name, str):
        kwargs["split_name"] = split_name.replace("K0", EXPERIMENT).replace(
            "K1", EXPERIMENT
        )
    return k1._ORIGINAL_EVALUATE(*args, **kwargs)


def _r1_print(*values: object, **kwargs: object) -> None:
    adjusted = tuple(
        value.replace("K0", EXPERIMENT).replace("K1", EXPERIMENT)
        if isinstance(value, str)
        else value
        for value in values
    )
    builtins.print(*adjusted, **kwargs)


def _r1_tqdm(*args: Any, **kwargs: Any):
    description = kwargs.get("desc")
    if isinstance(description, str):
        kwargs["desc"] = description.replace("K1", EXPERIMENT)
    return _ORIGINAL_TQDM(*args, **kwargs)


def _float_match(actual: object, expected: object) -> bool:
    try:
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=FIRST_BATCH_REL_TOLERANCE,
            abs_tol=FIRST_BATCH_ABS_TOLERANCE,
        )
    except (TypeError, ValueError):
        return False


def _reference_rank_row(rank: int) -> Mapping[str, object]:
    if _K1_REFERENCE is None:
        raise RuntimeError("K1 reference was not initialized")
    audit = _K1_REFERENCE["first_batch"]
    assert isinstance(audit, Mapping)
    rows = audit.get("per_rank")
    if not isinstance(rows, Sequence):
        raise RuntimeError("K1 first-batch reference has no per-rank rows")
    for row in rows:
        if isinstance(row, Mapping) and int(row.get("rank", -1)) == rank:
            return row
    raise RuntimeError(f"K1 first-batch reference has no rank {rank}")


def _compare_first_batch_base_to_k1(
    row: Mapping[str, object], rank: int
) -> Dict[str, object]:
    reference = _reference_rank_row(rank)
    exact_fields = (
        "paths",
        "image_tensor_shape",
        "target_tensor_shape",
        "image_tensor_sha256",
        "target_tensor_sha256",
        "valid_pixels",
        "student_feature_shapes",
        "teacher_feature_shapes",
        "projected_teacher_shapes",
        "teacher_checkpoint_sha256",
        "k0_shared_training_runner_sha256",
        "pca_parameter_record_sha256",
        "pca_parameter_sha256",
        "projection_parameter_sha256",
        "pca_sampling_manifest_sha256",
    )
    exact_mismatches = {
        field: {"actual": row.get(field), "expected": reference.get(field)}
        for field in exact_fields
        if row.get(field) != reference.get(field)
    }
    scalar_fields = ("feature_loss", "ce_loss", "warmup_weight")
    scalar_mismatches = {
        field: {"actual": row.get(field), "expected": reference.get(field)}
        for field in scalar_fields
        if not _float_match(row.get(field), reference.get(field))
    }
    layer_mismatches: Dict[str, object] = {}
    actual_layers = row.get("feature_loss_by_layer", {})
    reference_layers = reference.get("feature_loss_by_layer", {})
    if isinstance(actual_layers, Mapping) and isinstance(reference_layers, Mapping):
        for layer in a0.A0_LAYER_ORDER:
            if not _float_match(actual_layers.get(layer), reference_layers.get(layer)):
                layer_mismatches[layer] = {
                    "actual": actual_layers.get(layer),
                    "expected": reference_layers.get(layer),
                }
    else:
        layer_mismatches["schema"] = {
            "actual": actual_layers,
            "expected": reference_layers,
        }
    return {
        "rank": rank,
        "passed": not exact_mismatches and not scalar_mismatches and not layer_mismatches,
        "comparison": "R1 CE+feature base fields versus locked K1 seed=42",
        "reference": str((K1_REFERENCE_DIR / "first_batch_audit.json").resolve()),
        "absolute_tolerance": FIRST_BATCH_ABS_TOLERANCE,
        "relative_tolerance": FIRST_BATCH_REL_TOLERANCE,
        "exact_mismatches": exact_mismatches,
        "scalar_mismatches": scalar_mismatches,
        "feature_layer_mismatches": layer_mismatches,
    }


def _mean_std(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not finite:
        return {"mean": None, "sample_std": None}
    mean = sum(finite) / len(finite)
    sample_std = (
        0.0
        if len(finite) < 2
        else math.sqrt(sum((value - mean) ** 2 for value in finite) / (len(finite) - 1))
    )
    return {"mean": mean, "sample_std": sample_std}


def _aggregate_gradient_record(
    local_record: Dict[str, object], world_size: int
) -> Dict[str, object]:
    rows: List[Optional[Dict[str, object]]] = [None for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather_object(rows, local_record)
    else:
        rows[0] = local_record
    valid_rows = [row for row in rows if row is not None]
    if len(valid_rows) != world_size:
        raise RuntimeError("R1 failed to gather every rank's gradient audit")
    summary = dict(local_record)
    summary["rank_aggregation"] = "mean across ranks; sample_std and per_rank retained"
    summary["per_rank"] = valid_rows
    numeric_fields = (
        "grad_l2_ce",
        "grad_l2_feature",
        "grad_l2_relation_r1_os4",
        "grad_l2_relation_r1_os8",
        "grad_l2_relation_r1_os16",
        "grad_l2_relation_effective_os16",
        "grad_l2_total_os16",
        "grad_l2_total_student",
        "relation_to_feature_effective_ratio_os16",
        "relation_to_ce_effective_ratio_os16",
        "cos_ce_feature_os16",
        "cos_ce_relation_os16",
        "cos_feature_relation_os16",
    )
    for field in numeric_fields:
        values = [row.get(field) for row in valid_rows]
        stats = _mean_std(values)  # type: ignore[arg-type]
        summary[field] = stats["mean"]
        summary[f"{field}_sample_std"] = stats["sample_std"]
    return summary


def _update_relation_stop_gate(record: Mapping[str, object]) -> None:
    global _RELATION_GATE_CONSECUTIVE_EXCESS
    ratio = record.get("relation_to_ce_effective_ratio_os16")
    if ratio is not None and float(ratio) > GRADIENT_CE_STOP_RATIO:
        _RELATION_GATE_CONSECUTIVE_EXCESS += 1
    else:
        _RELATION_GATE_CONSECUTIVE_EXCESS = 0
    if _RELATION_GATE_CONSECUTIVE_EXCESS >= GRADIENT_CE_STOP_CONSECUTIVE:
        raise RuntimeError(
            "R1 effective relation gradient exceeded 2x CE for three consecutive "
            "gradient records; stop and inspect mask/reduction/lambda"
        )


def _reduce_r1_statistics(
    layer_sums: Mapping[str, float],
    feature_sum: float,
    relation_sum: float,
    total_sum: float,
    physical_batch_sum: float,
    batch_count: int,
    min_physical_batch: int,
    max_physical_batch: int,
    device: torch.device,
    world_size: int,
) -> Tuple[Dict[str, float], float, float, float, float, int, int, int]:
    values = [layer_sums[layer] for layer in a0.A0_LAYER_ORDER]
    values.extend(
        [feature_sum, relation_sum, total_sum, physical_batch_sum, float(batch_count)]
    )
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_count = int(tensor[-1].item())
    denominator = max(global_count, 1)
    layer_means = {
        layer: float(tensor[index].item() / denominator)
        for index, layer in enumerate(a0.A0_LAYER_ORDER)
    }
    extrema = torch.tensor(
        [min_physical_batch, max_physical_batch], device=device, dtype=torch.int64
    )
    if world_size > 1:
        minimum = extrema[:1].clone()
        maximum = extrema[1:].clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        global_min = int(minimum.item())
        global_max = int(maximum.item())
    else:
        global_min = min_physical_batch
        global_max = max_physical_batch
    return (
        layer_means,
        float(tensor[-5].item() / denominator),
        float(tensor[-4].item() / denominator),
        float(tensor[-3].item() / denominator),
        float(tensor[-2].item() / denominator),
        global_count,
        global_min,
        global_max,
    )


def train_one_epoch_r1(
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
    global _FIRST_BATCH_BASE_EQUIVALENCE

    teacher, projection = k1._require_resources()
    args = k1._ACTIVE_ARGS
    warmup_steps = k1._warmup_steps(args)
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
    relation_sum = 0.0
    total_sum = 0.0
    physical_batch_sum = 0.0
    min_physical_batch = 1 << 30
    max_physical_batch = 0
    layer_sums = {layer: 0.0 for layer in a0.A0_LAYER_ORDER}
    batch_count = 0
    optimizer_steps = 0
    last_warmup_weight = 0.0
    gradient_records: List[Dict[str, object]] = []
    first_batch_audit: Optional[Dict[str, object]] = None

    possible_steps = math.ceil(len(loader) / accumulation_steps)
    target_steps = min(possible_steps, remaining_optimizer_steps)
    max_batches = min(len(loader), target_steps * accumulation_steps)
    progress = k1.tqdm(
        loader, desc=f"Epoch {epoch} [R1 CE+feature+relation]", disable=rank != 0
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
                    raise RuntimeError("R1 training forward did not return features")
                logits = student_output["logits"]
                student_features = student_output["features"]
                with torch.no_grad():
                    teacher_features = teacher.extract_features(images)
                feature_layer_losses: Dict[str, torch.Tensor] = {}
                projected_shapes: Dict[str, List[int]] = {}
                for layer in a0.A0_LAYER_ORDER:
                    projected_teacher = projection[layer](
                        teacher_features[layer].detach()
                    )
                    projected_shapes[layer] = list(projected_teacher.shape)
                    feature_layer_losses[layer] = F.mse_loss(
                        student_features[layer].float(), projected_teacher.float()
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
                raise RuntimeError("R1 training batch contains no valid Cityscapes pixels")
            loss_seg = batch_ce_sum / batch_valid
            loss_feat = sum(feature_layer_losses.values()) / len(a0.A0_LAYER_ORDER)
            loss_r1, relation_layer_losses, relation_audit = r1_relation_losses(
                student_features, teacher_features, targets, world_size
            )
            total_loss = loss_seg + warmup_weight * (
                args.lambda_feat * loss_feat + args.lambda_r1 * loss_r1
            )
            finite_values = [
                loss_seg,
                loss_feat,
                loss_r1,
                total_loss,
                *feature_layer_losses.values(),
                *relation_layer_losses.values(),
            ]
            if not all(bool(torch.isfinite(value).all().item()) for value in finite_values):
                raise RuntimeError("R1 produced a non-finite CE/feature/relation loss")

            log_gradients = sync_gradients and (
                next_optimizer_step == 1
                or next_optimizer_step % args.gradient_log_steps == 0
            )
            local_grad_record: Optional[Dict[str, object]] = None
            if log_gradients:
                layer_gradient_records: Dict[str, Dict[str, object]] = {}
                gradients_by_layer: Dict[str, Dict[str, torch.Tensor]] = {}
                for layer in a0.A0_LAYER_ORDER:
                    tap = student_features[layer]
                    grad_ce = torch.autograd.grad(
                        loss_seg, tap, retain_graph=True, allow_unused=False
                    )[0].detach().float()
                    grad_feat = torch.autograd.grad(
                        loss_feat, tap, retain_graph=True, allow_unused=False
                    )[0].detach().float()
                    grad_relation = torch.autograd.grad(
                        loss_r1, tap, retain_graph=True, allow_unused=False
                    )[0].detach().float()
                    gradients_by_layer[layer] = {
                        "ce": grad_ce,
                        "feature": grad_feat,
                        "relation": grad_relation,
                    }
                    grad_total = grad_ce + warmup_weight * (
                        args.lambda_feat * grad_feat + args.lambda_r1 * grad_relation
                    )
                    layer_gradient_records[layer] = {
                        "tap_shape": list(tap.shape),
                        "grad_l2_ce": _gradient_l2(grad_ce),
                        "grad_l2_feature": _gradient_l2(grad_feat),
                        "grad_l2_relation_r1": _gradient_l2(grad_relation),
                        "grad_l2_feature_effective": _gradient_l2(
                            warmup_weight * args.lambda_feat * grad_feat
                        ),
                        "grad_l2_relation_effective": _gradient_l2(
                            warmup_weight * args.lambda_r1 * grad_relation
                        ),
                        "grad_l2_total_effective": _gradient_l2(grad_total),
                        "cos_ce_feature": _gradient_cosine(grad_ce, grad_feat),
                        "cos_ce_relation": _gradient_cosine(grad_ce, grad_relation),
                        "cos_feature_relation": _gradient_cosine(
                            grad_feat, grad_relation
                        ),
                    }
                os16_gradients = gradients_by_layer["os16"]
                effective_relation_os16 = warmup_weight * args.lambda_r1 * os16_gradients[
                    "relation"
                ]
                effective_feature_os16 = warmup_weight * args.lambda_feat * os16_gradients[
                    "feature"
                ]
                relation_norm = _gradient_l2(effective_relation_os16)
                feature_norm = _gradient_l2(effective_feature_os16)
                ce_norm = _gradient_l2(os16_gradients["ce"])
                local_grad_record = {
                    "optimizer_step": next_optimizer_step,
                    "fixed_audit_step": next_optimizer_step in FIXED_GRADIENT_AUDIT_STEPS,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "warmup_weight": warmup_weight,
                    "lambda_feat": args.lambda_feat,
                    "lambda_r1": args.lambda_r1,
                    "relation_r1_loss_raw": float(loss_r1.detach().item()),
                    "relation_loss_weighted": float(
                        (warmup_weight * args.lambda_r1 * loss_r1).detach().item()
                    ),
                    "grad_l2_ce": ce_norm,
                    "grad_l2_feature": _gradient_l2(os16_gradients["feature"]),
                    "grad_l2_logit": None,
                    "grad_l2_relation_r1_os4": _gradient_l2(
                        gradients_by_layer["os4"]["relation"]
                    ),
                    "grad_l2_relation_r1_os8": _gradient_l2(
                        gradients_by_layer["os8"]["relation"]
                    ),
                    "grad_l2_relation_r1_os16": _gradient_l2(
                        gradients_by_layer["os16"]["relation"]
                    ),
                    "grad_l2_relation_r2_os4": None,
                    "grad_l2_relation_r2_os8": None,
                    "grad_l2_relation_r2_os16": None,
                    "grad_l2_relation_effective_os16": relation_norm,
                    "grad_l2_total_os16": layer_gradient_records["os16"][
                        "grad_l2_total_effective"
                    ],
                    "relation_to_feature_effective_ratio_os16": relation_norm
                    / max(feature_norm, 1e-12),
                    "relation_to_ce_effective_ratio_os16": relation_norm
                    / max(ce_norm, 1e-12),
                    "cos_ce_feature_os16": layer_gradient_records["os16"][
                        "cos_ce_feature"
                    ],
                    "cos_ce_relation_os16": layer_gradient_records["os16"][
                        "cos_ce_relation"
                    ],
                    "cos_feature_relation_os16": layer_gradient_records["os16"][
                        "cos_feature_relation"
                    ],
                    "relation_valid_token_count": relation_audit["valid_token_count"],
                    "relation_valid_pair_count": relation_audit["valid_pair_count"],
                    "relation_physical_batch_size": relation_audit[
                        "physical_batch_size"
                    ],
                    "relation_finite": True,
                    "layers": layer_gradient_records,
                    "gradient_component_scope": "native student OS=4/8/16 taps",
                    "relation_spec_sha256": (
                        None
                        if _RELATION_SPEC is None
                        else _canonical_sha256(_RELATION_SPEC)
                    ),
                }
            scaler.scale(total_loss / group_size).backward()

        if sync_gradients:
            scaler.unscale_(optimizer)
            optimizer_steps += 1
            if local_grad_record is not None:
                local_grad_record["grad_l2_total_student"] = k0._gradient_l2_named(model)
                aggregated_record = _aggregate_gradient_record(
                    local_grad_record, world_size
                )
                _update_relation_stop_gate(aggregated_record)
                if rank == 0:
                    gradient_records.append(aggregated_record)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if first_batch_audit is None and starting_optimizer_step == 0 and batch_index == 0:
            global_paths: List[Optional[List[str]]] = [None for _ in range(world_size)]
            if world_size > 1:
                dist.all_gather_object(global_paths, list(paths))
            else:
                global_paths[0] = list(paths)
            first_batch_audit = {
                "rank": rank,
                "epoch": epoch,
                "micro_batch_index": 0,
                "paths": list(paths),
                "relation_global_path_order_by_rank": global_paths,
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
                "projected_teacher_shapes": projected_shapes,
                "feature_loss_by_layer": {
                    layer: float(feature_layer_losses[layer].detach().item())
                    for layer in a0.A0_LAYER_ORDER
                },
                "feature_loss": float(loss_feat.detach().item()),
                "ce_loss": float(loss_seg.detach().item()),
                "relation_r1_loss_by_layer": {
                    layer: float(relation_layer_losses[layer].detach().item())
                    for layer in a0.A0_LAYER_ORDER
                },
                "relation_r1_loss": float(loss_r1.detach().item()),
                "relation_loss_weighted": float(
                    (warmup_weight * args.lambda_r1 * loss_r1).detach().item()
                ),
                "relation": relation_audit,
                "warmup_weight": warmup_weight,
                "lambda_r1": args.lambda_r1,
                "total_loss": float(total_loss.detach().item()),
                "relation_spec_sha256": (
                    None
                    if _RELATION_SPEC is None
                    else _canonical_sha256(_RELATION_SPEC)
                ),
                **k1._resource_hashes(),
            }
            local_equivalence = _compare_first_batch_base_to_k1(
                first_batch_audit, rank
            )
            gathered_equivalence: List[Optional[Dict[str, object]]] = [
                None for _ in range(world_size)
            ]
            if world_size > 1:
                dist.all_gather_object(gathered_equivalence, local_equivalence)
            else:
                gathered_equivalence[0] = local_equivalence
            global_pass = all(
                value is not None and bool(value.get("passed"))
                for value in gathered_equivalence
            )
            _FIRST_BATCH_BASE_EQUIVALENCE = {
                "passed": global_pass,
                "world_size": world_size,
                "comparison": "R1 CE+feature base versus locked K1 seed=42",
                "per_rank": gathered_equivalence,
            }
            first_batch_audit["r1_base_k1_equivalence"] = local_equivalence
            if not global_pass:
                raise RuntimeError(
                    "R1 changed a locked K1 first-batch base field: "
                    + json.dumps(gathered_equivalence, ensure_ascii=False, sort_keys=True)
                )

        predictions = logits_float.detach().argmax(dim=1)
        confusion += common.confusion_counts(predictions, targets)
        ce_loss_sum += float(batch_ce_sum.detach().item())
        valid_pixels += batch_valid
        feature_value = float(loss_feat.detach().item())
        relation_value = float(loss_r1.detach().item())
        physical_batch = int(relation_audit["physical_batch_size"])
        feature_sum += feature_value
        relation_sum += relation_value
        total_sum += float(total_loss.detach().item())
        physical_batch_sum += physical_batch
        min_physical_batch = min(min_physical_batch, physical_batch)
        max_physical_batch = max(max_physical_batch, physical_batch)
        for layer in a0.A0_LAYER_ORDER:
            layer_sums[layer] += float(feature_layer_losses[layer].detach().item())
        batch_count += 1
        last_warmup_weight = warmup_weight
        if rank == 0:
            running = common.metrics_from_confusion(confusion, ce_loss_sum, valid_pixels)
            progress.set_postfix(
                {
                    "CE": f"{running['loss']:.4f}",
                    "feat": f"{feature_value:.4f}",
                    "R1": f"{relation_value:.4f}",
                    "mIoU": f"{running['mIoU']:.4f}",
                    "warm": f"{warmup_weight:.3f}",
                    "steps": optimizer_steps,
                }
            )

    if optimizer_steps != target_steps:
        raise RuntimeError(
            f"R1 optimizer-step accounting failed: actual={optimizer_steps}, "
            f"expected={target_steps}"
        )
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("R1 training found a gradient on the frozen teacher")
    if list(projection.parameters()):
        raise RuntimeError("R1 projection became trainable during training")
    if batch_count == 0:
        raise RuntimeError("R1 epoch processed no micro-batches")

    metrics = server_base._reduce_train_metrics(
        confusion, ce_loss_sum, valid_pixels, device, world_size
    )
    (
        layer_means,
        feature_mean,
        relation_mean,
        total_mean,
        physical_batch_mean,
        global_batches,
        global_min_batch,
        global_max_batch,
    ) = _reduce_r1_statistics(
        layer_sums,
        feature_sum,
        relation_sum,
        total_sum,
        physical_batch_sum,
        batch_count,
        min_physical_batch,
        max_physical_batch,
        device,
        world_size,
    )
    metrics["loss_schema"] = "hard_label_CE_plus_A0_feature_MSE_plus_R1_relation_MSE"
    metrics["ce_loss"] = metrics["loss"]
    metrics["feature_loss"] = feature_mean
    metrics["feature_loss_by_layer"] = layer_means
    metrics["logit_loss"] = None
    metrics["relation_enabled"] = True
    metrics["relation_r1_loss"] = relation_mean
    metrics["relation_r2_loss"] = None
    metrics["relation_loss_weighted_at_last_warmup"] = (
        last_warmup_weight * args.lambda_r1 * relation_mean
    )
    metrics["relation_valid_token_count"] = physical_batch_mean
    metrics["relation_valid_pair_count_nominal"] = (
        int(args.batch_size) * world_size
    ) ** 2
    metrics["relation_physical_batch_size_mean"] = physical_batch_mean
    metrics["relation_physical_batch_size_min"] = global_min_batch
    metrics["relation_physical_batch_size_max"] = global_max_batch
    metrics["total_loss_micro_batch_mean"] = total_mean
    metrics["warmup_weight"] = last_warmup_weight
    metrics["micro_batches_global"] = global_batches
    return metrics, optimizer_steps, gradient_records, first_batch_audit


def smoke_test_r1(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    rank: int,
) -> None:
    global _REFERENCE_TESTS

    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    _REFERENCE_TESTS = run_relation_reference_tests(device, world_size)
    teacher, projection = k1._require_resources()
    args = k1._ACTIVE_ARGS
    model.train()
    teacher.eval()
    images, targets, paths = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    model.zero_grad(set_to_none=True)
    with common.autocast_context(device, amp_enabled):
        student_output = model(images)
        if not isinstance(student_output, Mapping):
            raise RuntimeError("R1 smoke forward did not expose features")
        with torch.no_grad():
            teacher_features = teacher.extract_features(images)
        student_features = student_output["features"]
        feature_layer_losses = {
            layer: F.mse_loss(
                student_features[layer].float(),
                projection[layer](teacher_features[layer].detach()).float(),
            )
            for layer in a0.A0_LAYER_ORDER
        }
    logits = student_output["logits"].float()
    valid_pixels = int((targets != common.IGNORE_INDEX).sum().item())
    if valid_pixels == 0:
        raise RuntimeError("R1 smoke batch contains no valid pixels")
    loss_seg = F.cross_entropy(
        logits,
        targets,
        ignore_index=common.IGNORE_INDEX,
        reduction="sum",
    ) / valid_pixels
    loss_feat = sum(feature_layer_losses.values()) / len(a0.A0_LAYER_ORDER)
    loss_r1, _relation_layers, relation_audit = r1_relation_losses(
        student_features, teacher_features, targets, world_size
    )
    warmup_weight = 1.0 / k1._warmup_steps(args)
    total_loss = loss_seg + warmup_weight * (
        args.lambda_feat * loss_feat + args.lambda_r1 * loss_r1
    )
    relation_gradient = torch.autograd.grad(
        loss_r1, student_features["os16"], retain_graph=True
    )[0]
    total_loss.backward()
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in [loss_seg, loss_feat, loss_r1, total_loss]
    ):
        raise RuntimeError("R1 smoke test produced a non-finite loss")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("R1 smoke test found a teacher gradient")
    backbone_gradients = sum(
        parameter.grad is not None
        for parameter in k0.unwrap_model(model).backbone.parameters()
    )
    head_gradients = sum(
        parameter.grad is not None for parameter in k0.unwrap_model(model).head.parameters()
    )
    if backbone_gradients == 0 or head_gradients == 0:
        raise RuntimeError("R1 smoke test did not produce end-to-end student gradients")
    if int(relation_audit["physical_batch_size"]) > 1 and _gradient_l2(relation_gradient) <= 0:
        raise RuntimeError("R1 smoke test did not produce a relation gradient")
    if rank == 0:
        _r1_print(
            f"[OK] R1 server DDP smoke: sample={paths[0]}, "
            f"logits={tuple(logits.shape)}, CE={loss_seg.item():.6f}, "
            f"feature={loss_feat.item():.6f}, R1={loss_r1.item():.6f}, "
            f"total={total_loss.item():.6f}, warmup={warmup_weight:.6f}, "
            f"relation_B={relation_audit['physical_batch_size']}, "
            f"relation_grad_os16={_gradient_l2(relation_gradient):.6e}, "
            f"backbone_grad_tensors={backbone_gradients}, "
            f"head_grad_tensors={head_gradients}"
        )


def _existing_first_batch_equivalence(args: Any) -> Optional[Dict[str, object]]:
    path = r1_paths(args.output_dir, args.seed)["first_batch_audit"]
    if not path.is_file():
        return None
    audit = _read_json(path)
    rows = audit.get("per_rank")
    if not isinstance(rows, Sequence):
        return None
    values = []
    for row in rows:
        if isinstance(row, Mapping) and isinstance(
            row.get("r1_base_k1_equivalence"), Mapping
        ):
            values.append(dict(row["r1_base_k1_equivalence"]))
    if not values:
        return None
    return {
        "passed": all(bool(value.get("passed")) for value in values),
        "world_size": len(values),
        "comparison": "restored from existing R1 first_batch_audit.json",
        "per_rank": values,
    }


def _read_gradient_gate_summary(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {
            "records": 0,
            "target_ratio_range": [GRADIENT_GATE_MIN, GRADIENT_GATE_MAX],
            "passed_target_at_any_record": False,
        }
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    ratios = [
        float(record["relation_to_feature_effective_ratio_os16"])
        for record in records
        if record.get("relation_to_feature_effective_ratio_os16") is not None
    ]
    ce_ratios = [
        float(record["relation_to_ce_effective_ratio_os16"])
        for record in records
        if record.get("relation_to_ce_effective_ratio_os16") is not None
    ]
    return {
        "records": len(records),
        "fixed_audit_steps_expected": list(FIXED_GRADIENT_AUDIT_STEPS),
        "fixed_audit_steps_observed": [
            int(record["optimizer_step"])
            for record in records
            if bool(record.get("fixed_audit_step"))
        ],
        "target_ratio_range": [GRADIENT_GATE_MIN, GRADIENT_GATE_MAX],
        "relation_to_feature_effective_ratio_min": min(ratios) if ratios else None,
        "relation_to_feature_effective_ratio_max": max(ratios) if ratios else None,
        "passed_target_at_any_record": any(
            GRADIENT_GATE_MIN <= ratio <= GRADIENT_GATE_MAX for ratio in ratios
        ),
        "relation_to_ce_effective_ratio_max": max(ce_ratios) if ce_ratios else None,
        "three_consecutive_relation_gt_2x_ce": False,
    }


def _postprocess_metrics_r1(args: Any) -> None:
    _ORIGINAL_K1_POSTPROCESS(args)
    if int(os.environ.get("RANK", "0")) != 0:
        return
    metrics_path = r1_paths(args.output_dir, args.seed)["metrics"]
    if not metrics_path.is_file():
        raise FileNotFoundError(f"R1 metrics were not created: {metrics_path}")
    results = _read_json(metrics_path)
    relation_spec = _RELATION_SPEC
    if relation_spec is None:
        config = results.get("config", {})
        if isinstance(config, Mapping) and isinstance(config.get("relation"), Mapping):
            relation_spec = dict(config["relation"])  # type: ignore[arg-type]
    relation_hash = (
        None if relation_spec is None else _canonical_sha256(relation_spec)
    )
    first_equivalence = _FIRST_BATCH_BASE_EQUIVALENCE or _existing_first_batch_equivalence(
        args
    )
    r0_metrics: Optional[Dict[str, object]] = None
    if _r0_metrics_path(args).is_file():
        r0_metrics = _read_json(_r0_metrics_path(args))
    r1_best = results.get("best_dev_metrics", {})
    r0_best = None if r0_metrics is None else r0_metrics.get("best_dev_metrics", {})
    r1_miou = (
        float(r1_best["mIoU"]) if isinstance(r1_best, Mapping) else None
    )
    r0_miou = (
        float(r0_best["mIoU"]) if isinstance(r0_best, Mapping) else None
    )

    results["experiment"] = EXPERIMENT
    results["experiment_group"] = EXPERIMENT_GROUP
    results["artifact_type"] = ARTIFACT_TYPE
    results["protocol"] = (
        "R1 controlled relation-KD run: the accepted R0/K1 shared scratch "
        "MobileNetV2+R-ASPP initialization, hard-label CE, locked A0 fixed "
        "StandardScaler+PCA feature MSE, and one additional native-feature "
        "masked-GAP synchronized cross-image signed-cosine matrix MSE; no logits "
        "KD or R2 term, 4000-step shared auxiliary warm-up, fixed 80k budget, "
        "dev_local selection, and no test_local evaluation."
    )
    results["relation"] = copy.deepcopy(relation_spec)
    results["relation_spec_sha256"] = relation_hash
    results["relation_reference_tests"] = copy.deepcopy(_REFERENCE_TESTS)
    results["r0_gate"] = copy.deepcopy(_R0_GATE)
    results["k1_reference_validation"] = copy.deepcopy(_K1_REFERENCE_VALIDATION)
    results["r1_first_batch_base_equivalence"] = copy.deepcopy(first_equivalence)
    results["r1_vs_r0"] = {
        "R1_mIoU": r1_miou,
        "R0_mIoU": r0_miou,
        "delta_R1_minus_R0": (
            None if r1_miou is None or r0_miou is None else r1_miou - r0_miou
        ),
        "causal_comparison_valid": bool(
            _R0_GATE
            and _R0_GATE.get("passed")
            and first_equivalence
            and first_equivalence.get("passed")
        ),
    }
    loss = results.get("loss")
    if isinstance(loss, dict):
        loss.update(
            {
                "relation_kd": True,
                "relation_r1": True,
                "relation_r2": False,
                "lambda_r1": float(args.lambda_r1),
                "lambda_r2": 0.0,
            }
        )
    results["gradient_gate"] = _read_gradient_gate_summary(
        r1_paths(args.output_dir, args.seed)["gradient_norms"]
    )
    results["physical_relation_batch_size"] = (
        None
        if relation_spec is None
        else relation_spec.get("physical_relation_batch_size")
    )
    results["effective_optimizer_batch_size"] = (
        None
        if relation_spec is None
        else relation_spec.get("effective_optimizer_batch_size")
    )
    results["hashes"] = {
        **dict(results.get("hashes", {})),
        "relation_spec_sha256": relation_hash,
        "r0_gate_metrics_sha256": (
            None if _R0_GATE is None else _R0_GATE.get("metrics_sha256")
        ),
        "k1_reference_metrics_sha256": (
            None
            if _K1_REFERENCE_VALIDATION is None
            else _K1_REFERENCE_VALIDATION.get("reference_metrics_sha256")
        ),
        "r1_training_script_sha256": common.sha256_file(Path(__file__).resolve()),
    }
    results["test_local_evaluated"] = False
    common.write_json_atomic(metrics_path, results)


def _restore_relation_gate_state(args: Any) -> None:
    global _RELATION_GATE_CONSECUTIVE_EXCESS
    _RELATION_GATE_CONSECUTIVE_EXCESS = 0
    if not args.resume:
        return
    path = r1_paths(args.output_dir, args.seed)["gradient_norms"]
    if not path.is_file():
        return
    recent = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                recent.append(value)
    for record in recent[-GRADIENT_CE_STOP_CONSECUTIVE:]:
        ratio = record.get("relation_to_ce_effective_ratio_os16")
        if ratio is not None and float(ratio) > GRADIENT_CE_STOP_RATIO:
            _RELATION_GATE_CONSECUTIVE_EXCESS += 1
        else:
            _RELATION_GATE_CONSECUTIVE_EXCESS = 0


def run_training(args: Any) -> None:
    """Temporarily route K1's audited runner through the R1 contract."""

    global _K1_REFERENCE
    global _K1_REFERENCE_VALIDATION
    global _R0_GATE
    global _RELATION_SPEC
    global _REFERENCE_TESTS
    global _FIRST_BATCH_BASE_EQUIVALENCE

    _K1_REFERENCE, _K1_REFERENCE_VALIDATION = r0._validate_k1_reference(args)
    _R0_GATE = _validate_r0_gate(args)
    _RELATION_SPEC = None
    _REFERENCE_TESTS = None
    _FIRST_BATCH_BASE_EQUIVALENCE = (
        _existing_first_batch_equivalence(args) if args.resume else None
    )
    _restore_relation_gate_state(args)

    saved: Dict[str, object] = {
        "__file__": k1.__file__,
        "EXPERIMENT": k1.EXPERIMENT,
        "ARTIFACT_TYPE": k1.ARTIFACT_TYPE,
        "ARTIFACT_FORMAT_VERSION": k1.ARTIFACT_FORMAT_VERSION,
        "k1_paths": k1.k1_paths,
        "build_config": k1.build_config,
        "build_best_checkpoint": k1.build_best_checkpoint,
        "train_one_epoch_k1": k1.train_one_epoch_k1,
        "smoke_test_k1": k1.smoke_test_k1,
        "_postprocess_metrics": k1._postprocess_metrics,
        "audit_k1_shapes": k1.audit_k1_shapes,
        "_patched_torch_save_atomic": k1._patched_torch_save_atomic,
        "_patched_evaluate": k1._patched_evaluate,
        "_k1_print": k1._k1_print,
        "tqdm": k1.tqdm,
        "_ORIGINAL_ENSURE_SHARED_INITIALIZATION": (
            k1._ORIGINAL_ENSURE_SHARED_INITIALIZATION
        ),
    }
    had_module_print = "print" in k1.__dict__
    saved_module_print = k1.__dict__.get("print")
    saved_server_build_train_loader = server_base.build_train_loader

    k1.__file__ = str(Path(__file__).resolve())
    k1.EXPERIMENT = EXPERIMENT
    k1.ARTIFACT_TYPE = ARTIFACT_TYPE
    k1.ARTIFACT_FORMAT_VERSION = ARTIFACT_FORMAT_VERSION
    k1.k1_paths = r1_paths
    k1.build_config = build_config_r1
    k1.build_best_checkpoint = build_best_checkpoint_r1
    k1.train_one_epoch_k1 = train_one_epoch_r1
    k1.smoke_test_k1 = smoke_test_r1
    k1._postprocess_metrics = _postprocess_metrics_r1
    k1.audit_k1_shapes = audit_shapes_r1
    k1._patched_torch_save_atomic = _patched_torch_save_atomic_r1
    k1._patched_evaluate = _patched_evaluate_r1
    k1._k1_print = _r1_print
    k1.tqdm = _r1_tqdm
    k1._ORIGINAL_ENSURE_SHARED_INITIALIZATION = (
        _ensure_locked_k_shared_initialization
    )
    server_base.build_train_loader = build_train_loader_r1
    k1.print = _r1_print
    try:
        k1.run_training(args)
    finally:
        for name, value in saved.items():
            setattr(k1, name, value)
        server_base.build_train_loader = saved_server_build_train_loader
        if had_module_print:
            k1.print = saved_module_print
        else:
            k1.__dict__.pop("print", None)
        _RELATION_SPEC = None


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
