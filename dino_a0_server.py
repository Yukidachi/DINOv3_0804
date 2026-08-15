"""A0 server training entry point: fixed StandardScaler+PCA feature distillation.

Experiment A0 from ``plan_markdown/A实验的具体实施方案.md``:

    teacher side : per-layer fixed ``StandardScaler + PCA``, T -> S
    student side : no adapter, scratch MobileNetV2 backbone only
    projection   : fixed (never appears in the optimizer)
    loss         : 3-layer dense feature MSE (OS=4/8/16), label-free

The pipeline is split into stages so each scientific gate from the plan can be
checked in order:

1. ``--stage pca``        build the deterministic 200k-token manifest and fit
                          ``StandardScaler``/``IncrementalPCA`` in two passes
                          (plan sections 3.3-3.6).  Only ``train_local`` is
                          read; the PCA view is the full ``1024x2048`` image
                          bilinearly resized to ``512x1024`` with no crop or
                          flip.
2. ``--stage pretrain``   40k optimizer steps of label-free dense feature MSE.
                          Only the MobileNetV2 backbone is optimized; the T1
                          teacher is frozen in ``eval()``/``inference``.  The
                          fixed PCA projection is the explicit matmul path that
                          A1 must reproduce with three fixed ``1x1`` convolutions.
3. ``--stage probe``      freeze the pretrained backbone (parameters and BatchNorm
                          running statistics), then train the standard 19-class
                          R-ASPP head for 40k optimizer steps with pixel CE.
                          Checkpoints are selected only by ``dev_local`` mIoU.
4. ``--stage full``       pretrain then probe in one launch.

A0 uses the exact same DDP/spawn/no-pinned-memory/ordered-teardown conventions
as ``dino_s2_0_server.py``/``dino_s2_f_server.py`` (see
``plan_markdown/server_training_issues_and_solutions.md``).  The locked T1
teacher is ``result/T1_DINOv3_RASPP/seed_3407/t1_dinov3_raspp_teacher.pth``
(``seed=3407``, dev mIoU ``0.778346``), and per the plan A0-A6 must not switch
teachers once training starts.

Server examples:

    # one-time PCA statistics (run once; can use nproc_per_node=1)
    torchrun --standalone --nproc_per_node=1 dino_a0_server.py \\
        --stage pca --device cuda

    # full A0: 40k feature pretrain + 40k frozen-head probe
    torchrun --standalone --nproc_per_node=2 dino_a0_server.py \\
        --stage full --seed 42 --batch-size 2 --global-batch-size 8 \\
        --num-workers 8 --multiprocessing-context spawn \\
        --no-pin-memory --persistent-workers

    # resume a stage from its per-stage last checkpoint
    torchrun --standalone --nproc_per_node=2 dino_a0_server.py \\
        --stage full --seed 42 --batch-size 2 --global-batch-size 8 \\
        --num-workers 8 --multiprocessing-context spawn \\
        --no-pin-memory --persistent-workers --resume

Windows single-process smoke test (does not replace the two-GPU DDP smoke):

    python -B dino_a0_server.py --stage full --device cuda --smoke-test \\
        --batch-size 1 --global-batch-size 1 --num-workers 0 \\
        --no-persistent-workers --no-pin-memory --no-amp
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

import dino as t0
import dino_s2_0 as base
import dino_s2_0_server as s2_0_server
from dino_t1 import load_teacher_for_distillation


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "A_MobileNetV2_RASPP_server"
DEFAULT_PCA_DIR = DEFAULT_OUTPUT_DIR / "pca_shared"
DEFAULT_TEACHER_CHECKPOINT = (
    SCRIPT_DIR / "result" / "T1_DINOv3_RASPP" / "seed_3407" / "t1_dinov3_raspp_teacher.pth"
)

EXPERIMENT = "A0"
MODEL_NAME = base.MODEL_NAME
NUM_CLASSES = t0.NUM_CLASSES
IGNORE_INDEX = t0.IGNORE_INDEX
OUTPUT_STRIDE = t0.OUTPUT_STRIDE
FORMAL_SEEDS = (42, 3407, 260805)

ARTIFACT_TYPE_PRETRAIN = "a0_pretrain_mobilenetv2_backbone"
ARTIFACT_TYPE_PROBE = "a0_probe_mobilenetv2_raspp_fixed_pca"
ARTIFACT_FORMAT_VERSION = 1

# A0 layer contract (locked in ``A实验的具体实施方案.md`` section 2.2).
A0_LAYER_ORDER = ("os4", "os8", "os16")
STUDENT_CHANNELS = {"os4": 24, "os8": 32, "os16": 320}
TEACHER_CHANNELS = {"os4": 96, "os8": 192, "os16": 768}
PCA_VIEW_HEIGHT = 512
PCA_VIEW_WIDTH = 1024
PCA_VIEW_SHAPES = {
    "os4": (128, 256),   # [B, 96, 128, 256]  -> [B, 24, 128, 256]
    "os8": (64, 128),    # [B, 192, 64, 128]  -> [B, 32, 64, 128]
    "os16": (32, 64),    # [B, 768, 32, 64]   -> [B, 320, 32, 64]
}
PCA_TOTAL_TOKENS = 200_000
PCA_MAX_TOKENS_PER_IMAGE = 128
PCA_SAMPLING_SEED = 42
PCA_GRID = (8, 8)
PCA_CHUNK_SIZE = 8192
PCA_BATCH_SIZE = 8192

PRETRAIN_MAX_STEPS = 40_000
PROBE_MAX_STEPS = 40_000
FEATURE_WARMUP_RATIO = 0.05
LAMBDA_FEAT = 1.0
GRADIENT_LOG_STEPS = 500
PRETRAIN_SNAPSHOT_STEPS = 5_000


def _stable_hash_int(*parts) -> int:
    """Stable SHA-256 based non-negative integer key (plan section 3.4)."""

    key = "/".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest(), "big")


def numpy_arrays_sha256(*arrays: np.ndarray) -> str:
    """Deterministic SHA-256 over fixed-field-order, dtype, shape and C-order bytes."""

    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A0: label-free dense feature distillation from a frozen T1 DINOv3 "
            "teacher through fixed per-layer StandardScaler+PCA to a scratch "
            "MobileNetV2 backbone, followed by a frozen-backbone R-ASPP probe."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=t0.DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stage",
        choices=("pca", "pretrain", "probe", "full"),
        default="full",
        help=(
            "pca: build the sampling manifest and fit scaler/PCA. "
            "pretrain: 40k label-free feature MSE. "
            "probe: 40k frozen-backbone R-ASPP head. "
            "full: pretrain then probe."
        ),
    )
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--teacher-repo-dir", type=Path, default=t0.DEFAULT_REPO_DIR)
    parser.add_argument("--teacher-weights-path", type=Path, default=t0.DEFAULT_WEIGHTS_PATH)
    parser.add_argument(
        "--pca-dir",
        type=Path,
        default=DEFAULT_PCA_DIR,
        help="Directory holding sampling_manifest.json, scaler_<layer>.npz, pca_<layer>.npz.",
    )
    parser.add_argument("--force-pca", action="store_true", help="Overwrite existing PCA artifacts.")
    parser.add_argument("--pretrain-max-steps", type=int, default=PRETRAIN_MAX_STEPS)
    parser.add_argument("--probe-max-steps", type=int, default=PROBE_MAX_STEPS)
    parser.add_argument("--lambda-feat", type=float, default=LAMBDA_FEAT)
    parser.add_argument("--feature-warmup-ratio", type=float, default=FEATURE_WARMUP_RATIO)
    parser.add_argument("--gradient-log-steps", type=int, default=GRADIENT_LOG_STEPS)
    parser.add_argument("--pretrain-snapshot-steps", type=int, default=PRETRAIN_SNAPSHOT_STEPS)
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=None,
        help=(
            "Backbone checkpoint the probe starts from. Defaults to the A0 "
            "pretrain last checkpoint. The plan selects the pretrain artifact by "
            "frozen-head probe mIoU; point this argument at any snapshot after "
            "scoring snapshots with a short probe."
        ),
    )
    parser.add_argument("--pca-total-tokens", type=int, default=PCA_TOTAL_TOKENS)
    parser.add_argument("--pca-selection-seed", type=int, default=PCA_SAMPLING_SEED)
    parser.add_argument("--pca-chunk-size", type=int, default=PCA_CHUNK_SIZE)
    parser.add_argument("--pca-batch-size", type=int, default=PCA_BATCH_SIZE)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-GPU batch size.")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=8,
        help="Global batch size; derives accumulation steps across all GPUs (default: 8).",
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
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    positive = (
        "pretrain_max_steps",
        "probe_max_steps",
        "gradient_log_steps",
        "pretrain_snapshot_steps",
        "pca_total_tokens",
        "pca_selection_seed",
        "pca_chunk_size",
        "pca_batch_size",
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
    if args.lambda_feat < 0 or not 0 <= args.feature_warmup_ratio <= 1:
        parser.error("--lambda-feat must be >= 0 and --feature-warmup-ratio must be in [0, 1]")
    if args.boundary_tolerance < 0:
        parser.error("--boundary-tolerance cannot be negative")
    if args.crop_height % OUTPUT_STRIDE or args.crop_width % OUTPUT_STRIDE:
        parser.error(f"Crop dimensions must be divisible by {OUTPUT_STRIDE}")
    if args.stage == "probe" and args.smoke_test:
        parser.error("--stage probe --smoke-test is not supported; run --stage full --smoke-test")
    return args


def a0_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    run_dir = output_dir.resolve() / "A0" / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "config": run_dir / "config.json",
        "feature_taps": run_dir / "feature_taps.json",
        "pretrain_last": run_dir / "a0_pretrain_last.pth",
        "pretrain_history": run_dir / "a0_pretrain_history.json",
        "pretrain_gradients": run_dir / "a0_pretrain_gradient_norms.jsonl",
        "pretrain_snapshots": run_dir / "pretrain_snapshots",
        "probe_last": run_dir / "a0_probe_last.pth",
        "probe_history": run_dir / "a0_probe_history.json",
        "best_probe": run_dir / "a0_probe_mobilenetv2_raspp_best.pth",
        "dev_metrics": run_dir / "a0_dev_metrics.json",
        "efficiency": run_dir / "efficiency.json",
        "per_image": run_dir / "a0_dev_per_image_confusion.jsonl",
        "projection_equivalence": run_dir / "projection_equivalence.json",
    }


def build_feature_taps_record() -> Dict[str, object]:
    """Teacher/student taps at a 512x1024 crop (plan section 2.2)."""

    taps: Dict[str, object] = {
        "crop_size": [512, 1024],
        "output_stride": OUTPUT_STRIDE,
        "teacher_model": "DINOv3 ConvNeXt-T (T1, frozen)",
        "student_model": MODEL_NAME,
        "layers": {},
    }
    for layer in A0_LAYER_ORDER:
        taps["layers"][layer] = {
            "teacher_module": {
                "os4": "backbone.get_intermediate_layers[0]",
                "os8": "backbone.get_intermediate_layers[1]",
                "os16": "backbone.get_intermediate_layers[3]",
            }[layer],
            "teacher_channels": TEACHER_CHANNELS[layer],
            "teacher_shape": [1, TEACHER_CHANNELS[layer], *PCA_VIEW_SHAPES[layer]],
            "student_module": base.FEATURE_TAPS[layer]["module"],
            "student_channels": STUDENT_CHANNELS[layer],
            "student_shape": [1, STUDENT_CHANNELS[layer], *PCA_VIEW_SHAPES[layer]],
            "pca_output_dim": STUDENT_CHANNELS[layer],
        }
    return taps


def _manifest_sha256(manifest: Mapping[str, object]) -> str:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_pca_sampling_manifest(
    entries: Sequence[Tuple[str, str]],
    total_tokens: int,
    selection_seed: int,
    layer_specs: Mapping[str, Mapping[str, int]],
    grid: Tuple[int, int] = PCA_GRID,
    max_tokens_per_image: int = PCA_MAX_TOKENS_PER_IMAGE,
) -> Dict[str, object]:
    """Build the deterministic image-balanced + spatial-grid token manifest.

    Plan section 3.3/3.4: ``base = floor(N / 2530)``, the smallest
    ``SHA256(seed/path/"extra_quota")`` images get one extra token, and within
    each image the positions cycle over an 8x8 spatial grid with stable per-slot
    hashes.  No label information is read.
    """

    n_images = len(entries)
    if n_images == 0:
        raise RuntimeError("Cannot build a PCA manifest from an empty image list")
    if total_tokens < n_images:
        raise RuntimeError("PCA total tokens must cover every training image at least once")
    base_q, remainder = divmod(total_tokens, n_images)
    if base_q > max_tokens_per_image:
        raise RuntimeError("PCA base quota exceeds the per-image token cap")
    extra_order = sorted(
        (
            (_stable_hash_int(selection_seed, image_rel, "extra_quota"), image_rel)
            for image_rel, _ in entries
        ),
        key=lambda item: item[0],
    )
    extra_set = {image_rel for _, image_rel in extra_order[:remainder]}
    quota = {
        image_rel: base_q + (1 if image_rel in extra_set else 0)
        for image_rel, _ in entries
    }
    quota_counts = sorted(
        (
            (value, sum(1 for q in quota.values() if q == value))
            for value in sorted(set(quota.values()))
        )
    )

    grid_rows, grid_cols = grid
    selections_by_layer: Dict[str, List[Dict[str, object]]] = {}
    for layer, spec in layer_specs.items():
        height, width = int(spec["height"]), int(spec["width"])
        if height % grid_rows or width % grid_cols:
            raise RuntimeError(
                f"Layer {layer} shape ({height},{width}) is not divisible by grid {grid}"
            )
        cell_h, cell_w = height // grid_rows, width // grid_cols
        # ``slot`` cycles through the 8x8 cells.  Each cell therefore receives
        # at most ceil(max_image_quota / 64) samples, not ``base_q`` samples.
        # The old check compared a cell's capacity with 79 and rejected the
        # valid OS=16 map (32x64 -> 4x8 cells, capacity 32 per cell).
        max_quota = base_q + (1 if remainder else 0)
        max_per_cell = math.ceil(max_quota / (grid_rows * grid_cols))
        if cell_h * cell_w < max_per_cell:
            raise RuntimeError(
                f"Layer {layer} grid cell capacity too small: "
                f"capacity={cell_h * cell_w}, required={max_per_cell}"
            )
        layer_selections: List[Dict[str, object]] = []
        for image_rel, _ in entries:
            count = quota[image_rel]
            seen: set[int] = set()
            for slot in range(count):
                retry = 0
                while True:
                    local_position = _stable_hash_int(
                        selection_seed, image_rel, layer, slot, retry
                    ) % (cell_h * cell_w)
                    cell_id = slot % (grid_rows * grid_cols)
                    row = (cell_id // grid_cols) * cell_h + local_position // cell_w
                    col = (cell_id % grid_cols) * cell_w + local_position % cell_w
                    flat_index = row * width + col
                    if flat_index not in seen:
                        break
                    retry += 1
                seen.add(flat_index)
                layer_selections.append(
                    {
                        "path": image_rel,
                        "token_slot": slot,
                        "row": row,
                        "col": col,
                        "flat_index": flat_index,
                    }
                )
        selections_by_layer[layer] = layer_selections

    manifest: Dict[str, object] = {
        "schema_version": 1,
        "total_tokens_per_layer": total_tokens,
        "selection_seed": selection_seed,
        "max_tokens_per_image": max_tokens_per_image,
        "grid": {"rows": grid_rows, "cols": grid_cols},
        "image_quota": {
            "base": base_q,
            "remainder": remainder,
            "counts": {str(value): count for value, count in quota_counts},
            "total": n_images,
        },
        "layers": {
            layer: {
                "height": int(spec["height"]),
                "width": int(spec["width"]),
                "teacher_channels": int(spec["channels"]),
                "d_out": int(spec["d_out"]),
            }
            for layer, spec in layer_specs.items()
        },
        "selections": selections_by_layer,
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    return manifest


class PCASamplingViewDataset(Dataset):
    """Deterministic PCA view: full 1024x2048 image resized to 512x1024.

    No crop, no flip, ImageNet mean/std normalization (plan section 3.2).
    Labels are not read, keeping the main experiment label-free.
    """

    def __init__(self, dataset_root: Path, entries: Sequence[Tuple[str, str]], view_size=(512, 1024)):
        self.dataset_root = Path(dataset_root).resolve()
        self.entries = list(entries)
        self.view_size = view_size

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        image_rel, _ = self.entries[index]
        with Image.open(self.dataset_root / image_rel) as image_obj:
            image = image_obj.convert("RGB")
        image = image.resize(
            (self.view_size[1], self.view_size[0]), resample=Image.Resampling.BILINEAR
        )
        return t0.image_to_normalized_tensor(image), image_rel


class _ChunkBuffer:
    """Accumulate per-image token rows and hand out fixed-size chunks."""

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self._parts: List[np.ndarray] = []
        self._count = 0

    def push(self, rows: np.ndarray) -> Optional[np.ndarray]:
        self._parts.append(rows)
        self._count += int(rows.shape[0])
        if self._count >= self.chunk_size:
            return self._pop_all()
        return None

    def flush(self) -> Optional[np.ndarray]:
        if not self._parts:
            return None
        return self._pop_all()

    def _pop_all(self) -> np.ndarray:
        output = np.concatenate(self._parts, axis=0)
        self._parts = []
        self._count = 0
        return output


def _group_selections_by_path(manifest: Mapping[str, object]):
    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        layer: defaultdict(list) for layer in A0_LAYER_ORDER
    }
    for layer in A0_LAYER_ORDER:
        for selection in manifest["selections"][layer]:
            grouped[layer][str(selection["path"])].append(selection)
    return grouped


def run_pca_stage(
    args: argparse.Namespace,
    device: torch.device,
    dataset_root: Path,
    dataset_lock: Mapping[str, object],
    entries_by_split: Mapping[str, Sequence[Tuple[str, str]]],
) -> None:
    """Two-pass StandardScaler / IncrementalPCA fit on the fixed token manifest.

    Plan section 3: only ``train_local`` is used, each layer gets exactly
    ``--pca-total-tokens`` tokens, and the deterministic PCA view is the full
    image resized to ``512x1024`` with no crop or flip.
    """

    pca_dir = Path(args.pca_dir).resolve()
    pca_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pca_dir / "sampling_manifest.json"
    parameter_hashes_path = pca_dir / "pca_parameters_sha256.json"
    layer_paths = {
        layer: (
            pca_dir / f"scaler_{layer}.npz",
            pca_dir / f"pca_{layer}.npz",
        )
        for layer in A0_LAYER_ORDER
    }
    expected_artifacts = [manifest_path, parameter_hashes_path] + [
        path for pair in layer_paths.values() for path in pair
    ]
    if any(path.exists() for path in expected_artifacts) and not args.force_pca:
        raise FileExistsError(
            f"PCA artifacts already exist in {pca_dir}; use --force-pca to overwrite"
        )

    print("[INFO] Loading frozen T1 teacher for PCA feature extraction")
    teacher, teacher_payload = load_teacher_for_distillation(
        args.teacher_checkpoint,
        repo_dir=args.teacher_repo_dir,
        weights_path=args.teacher_weights_path,
        device=device,
        verify_checkpoint_file=True,
    )
    teacher.eval()
    entries = list(entries_by_split["train_local"])
    if len(entries) != 2530:
        raise RuntimeError(f"Expected 2530 train_local images for PCA, found {len(entries)}")
    layer_specs = {
        layer: {
            "height": PCA_VIEW_SHAPES[layer][0],
            "width": PCA_VIEW_SHAPES[layer][1],
            "channels": TEACHER_CHANNELS[layer],
            "d_out": STUDENT_CHANNELS[layer],
        }
        for layer in A0_LAYER_ORDER
    }
    manifest = build_pca_sampling_manifest(
        entries,
        total_tokens=args.pca_total_tokens,
        selection_seed=args.pca_selection_seed,
        layer_specs=layer_specs,
    )
    manifest["dataset_combined_manifest_sha256"] = dataset_lock["combined_manifest_sha256"]
    manifest["teacher_checkpoint_sha256"] = t0.verify_checkpoint_sidecar(args.teacher_checkpoint)
    manifest["teacher_checkpoint_path"] = str(Path(args.teacher_checkpoint).resolve())
    manifest["pca_view"] = {
        "height": PCA_VIEW_HEIGHT,
        "width": PCA_VIEW_WIDTH,
        "resample": "bilinear",
        "normalization": "imagenet_mean_std",
        "crop": "none",
        "flip": "none",
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    grouped = _group_selections_by_path(manifest)

    dataset = PCASamplingViewDataset(dataset_root, entries, (PCA_VIEW_HEIGHT, PCA_VIEW_WIDTH))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    amp_enabled = bool(args.amp and device.type == "cuda")

    scalers: Dict[str, StandardScaler] = {
        layer: StandardScaler() for layer in A0_LAYER_ORDER
    }
    input_hashes: Dict[str, Any] = {layer: hashlib.sha256() for layer in A0_LAYER_ORDER}
    buffers: Dict[str, _ChunkBuffer] = {
        layer: _ChunkBuffer(args.pca_chunk_size) for layer in A0_LAYER_ORDER
    }

    def extract_layer_tokens(features: Dict[str, torch.Tensor], path: str) -> None:
        for layer in A0_LAYER_ORDER:
            selections = grouped[layer].get(path)
            if not selections:
                raise RuntimeError(f"PCA manifest has no selections for {path} in layer {layer}")
            feature_map = features[layer][0].float().cpu().numpy()  # [C, H, W]
            tokens = np.stack(
                [feature_map[:, sel["row"], sel["col"]] for sel in selections]
            )  # [q, C], selections are in token_slot order
            chunk = buffers[layer].push(tokens)
            if chunk is not None:
                scalers[layer].partial_fit(chunk.astype(np.float64))
                input_hashes[layer].update(
                    np.ascontiguousarray(chunk, dtype=np.float32).tobytes(order="C")
                )

    print("[INFO] PCA pass 1 of 2: StandardScaler partial fit over fixed tokens")
    with torch.inference_mode():
        for images, paths in tqdm(loader, desc="PCA pass 1/2", disable=False):
            images = images.to(device, non_blocking=True)
            features = teacher.extract_features(images)
            extract_layer_tokens(features, paths[0])
    for layer in A0_LAYER_ORDER:
        tail = buffers[layer].flush()
        if tail is not None:
            scalers[layer].partial_fit(tail.astype(np.float64))
            input_hashes[layer].update(
                np.ascontiguousarray(tail, dtype=np.float32).tobytes(order="C")
            )
    for layer in A0_LAYER_ORDER:
        if int(scalers[layer].n_samples_seen_) != args.pca_total_tokens:
            raise RuntimeError(
                f"PCA token accounting failed for {layer}: "
                f"{int(scalers[layer].n_samples_seen_)} != {args.pca_total_tokens}"
            )
    print("[OK] StandardScaler fitted for os4/os8/os16")

    pcas: Dict[str, IncrementalPCA] = {
        layer: IncrementalPCA(
            n_components=STUDENT_CHANNELS[layer],
            batch_size=args.pca_batch_size,
            whiten=False,
        )
        for layer in A0_LAYER_ORDER
    }
    buffers = {layer: _ChunkBuffer(args.pca_chunk_size) for layer in A0_LAYER_ORDER}

    def extract_scaled_tokens(features: Dict[str, torch.Tensor], path: str) -> None:
        for layer in A0_LAYER_ORDER:
            selections = grouped[layer].get(path)
            if not selections:
                raise RuntimeError(f"PCA manifest has no selections for {path} in layer {layer}")
            feature_map = features[layer][0].float().cpu().numpy()  # [C, H, W]
            tokens = np.stack(
                [feature_map[:, sel["row"], sel["col"]] for sel in selections]
            )
            chunk = buffers[layer].push(tokens)
            if chunk is not None:
                scaled = scalers[layer].transform(chunk.astype(np.float64))
                pcas[layer].partial_fit(scaled)

    print("[INFO] PCA pass 2 of 2: IncrementalPCA partial fit over standardized tokens")
    with torch.inference_mode():
        for images, paths in tqdm(loader, desc="PCA pass 2/2", disable=False):
            images = images.to(device, non_blocking=True)
            features = teacher.extract_features(images)
            extract_scaled_tokens(features, paths[0])
    for layer in A0_LAYER_ORDER:
        tail = buffers[layer].flush()
        if tail is not None:
            scaled = scalers[layer].transform(tail.astype(np.float64))
            pcas[layer].partial_fit(scaled)
    for layer in A0_LAYER_ORDER:
        if int(pcas[layer].n_samples_seen_) != args.pca_total_tokens:
            raise RuntimeError(
                f"PCA sample accounting failed for {layer}: "
                f"{int(pcas[layer].n_samples_seen_)} != {args.pca_total_tokens}"
            )
        expected_components = (STUDENT_CHANNELS[layer], TEACHER_CHANNELS[layer])
        if tuple(pcas[layer].components_.shape) != expected_components:
            raise RuntimeError(
                f"PCA component shape mismatch for {layer}: "
                f"{tuple(pcas[layer].components_.shape)} != {expected_components}"
            )

    layers_record: Dict[str, object] = {}
    for layer in A0_LAYER_ORDER:
        scaler = scalers[layer]
        pca = pcas[layer]
        scaler_arrays = {
            "mean_": np.ascontiguousarray(scaler.mean_),
            "var_": np.ascontiguousarray(scaler.var_),
            "scale_": np.ascontiguousarray(scaler.scale_),
            "n_samples_seen_": np.asarray([scaler.n_samples_seen_]),
        }
        pca_arrays = {
            "components_": np.ascontiguousarray(pca.components_),
            "mean_": np.ascontiguousarray(pca.mean_),
            "explained_variance_": np.ascontiguousarray(pca.explained_variance_),
            "explained_variance_ratio_": np.ascontiguousarray(pca.explained_variance_ratio_),
            "singular_values_": np.ascontiguousarray(pca.singular_values_),
            "n_samples_seen_": np.asarray([pca.n_samples_seen_]),
        }
        scaler_path, pca_path = layer_paths[layer]
        np.savez_compressed(scaler_path, **scaler_arrays)
        np.savez_compressed(pca_path, **pca_arrays)
        scaler_hash = numpy_arrays_sha256(*scaler_arrays.values())
        pca_hash = numpy_arrays_sha256(*pca_arrays.values())
        layers_record[layer] = {
            "feature_input_sha256": input_hashes[layer].hexdigest(),
            "scaler_sha256": scaler_hash,
            "pca_sha256": pca_hash,
            "d_out": int(STUDENT_CHANNELS[layer]),
            "n_tokens": args.pca_total_tokens,
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "explained_variance_ratio_head": [float(value) for value in pca.explained_variance_ratio_[: min(10, pca.explained_variance_ratio_.size)]],
            "scaler_path": str(scaler_path),
            "pca_path": str(pca_path),
        }

    parameter_record = {
        "schema_version": 1,
        "sampling_manifest_path": str(manifest_path),
        "sampling_manifest_sha256": manifest["manifest_sha256"],
        "dataset_combined_manifest_sha256": dataset_lock["combined_manifest_sha256"],
        "teacher_checkpoint_sha256": manifest["teacher_checkpoint_sha256"],
        "fit_algorithm": "StandardScaler.partial_fit + IncrementalPCA.partial_fit (two passes)",
        "standardizer_dtype": "float64 accumulation, float32 feature input",
        "whiten": False,
        "layers": layers_record,
    }
    t0.write_json_atomic(manifest_path, manifest)
    t0.write_json_atomic(parameter_hashes_path, parameter_record)
    print("[DONE] PCA stage complete")
    for layer in A0_LAYER_ORDER:
        record = layers_record[layer]
        print(
            f"   - {layer}: d={record['d_out']}, "
            f"explained_var_ratio={record['explained_variance_ratio_sum']:.4f}, "
            f"feature_input_sha256={record['feature_input_sha256'][:16]}"
        )
    print(f"   - manifest SHA-256: {manifest['manifest_sha256']}")


def load_pca_parameters(pca_dir: Path) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Dict[str, object]]:
    """Load scaler/PCA npz artifacts; also return the parameter-hash record."""

    pca_dir = Path(pca_dir).resolve()
    parameter_record: Dict[str, object] = {}
    hash_path = pca_dir / "pca_parameters_sha256.json"
    if hash_path.is_file():
        with hash_path.open("r", encoding="utf-8") as file_obj:
            parameter_record = json.load(file_obj)
    else:
        raise FileNotFoundError(f"Missing PCA parameter record: {hash_path}")
    manifest_path = pca_dir / "sampling_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing PCA sampling manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    if manifest.get("manifest_sha256") != _manifest_sha256(manifest):
        raise RuntimeError("PCA sampling manifest SHA-256 verification failed")
    if parameter_record.get("sampling_manifest_sha256") != manifest.get("manifest_sha256"):
        raise RuntimeError("PCA parameter record refers to a different sampling manifest")
    scalers: Dict[str, Dict[str, np.ndarray]] = {}
    pcas: Dict[str, Dict[str, np.ndarray]] = {}
    for layer in A0_LAYER_ORDER:
        scaler_path = pca_dir / f"scaler_{layer}.npz"
        pca_path = pca_dir / f"pca_{layer}.npz"
        if not scaler_path.is_file() or not pca_path.is_file():
            raise FileNotFoundError(
                f"Missing A0 PCA artifact for {layer} in {pca_dir}; run --stage pca first"
            )
        with np.load(scaler_path) as scaler_data:
            scalers[layer] = {key: scaler_data[key] for key in scaler_data.files}
        with np.load(pca_path) as pca_data:
            pcas[layer] = {key: pca_data[key] for key in pca_data.files}
        layer_record = parameter_record.get("layers", {}).get(layer, {})
        scaler_arrays = {
            key: np.ascontiguousarray(scalers[layer][key])
            for key in ("mean_", "var_", "scale_", "n_samples_seen_")
        }
        pca_arrays = {
            key: np.ascontiguousarray(pcas[layer][key])
            for key in (
                "components_",
                "mean_",
                "explained_variance_",
                "explained_variance_ratio_",
                "singular_values_",
                "n_samples_seen_",
            )
        }
        if numpy_arrays_sha256(*scaler_arrays.values()) != layer_record.get("scaler_sha256"):
            raise RuntimeError(f"Scaler hash verification failed for {layer}")
        if numpy_arrays_sha256(*pca_arrays.values()) != layer_record.get("pca_sha256"):
            raise RuntimeError(f"PCA hash verification failed for {layer}")
    return scalers, pcas, parameter_record


class FixedPCAProjection(nn.Module):
    """Explicit ``StandardScaler.transform + PCA.transform`` path (A0 reference).

    For one teacher feature position ``x in R^{C_t}``:

        y = ((x - mu_s) / sigma_s - mu_p) @ V^T

    with ``V = pca.components_`` of shape ``[d_l, C_t]``.  All buffers are
    fixed and never appear in the optimizer (plan section 4.1/4.2).
    """

    def __init__(
        self,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        pca_mean: np.ndarray,
        components: np.ndarray,
    ) -> None:
        super().__init__()
        if (
            components.ndim != 2
            or components.shape[1] != scaler_mean.shape[0]
            or scaler_mean.shape != scaler_scale.shape
            or scaler_mean.shape != pca_mean.shape
        ):
            raise RuntimeError("A0 projection parameter shapes are inconsistent")
        scale = torch.as_tensor(scaler_scale, dtype=torch.float32)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        self.register_buffer(
            "scaler_mean", torch.as_tensor(scaler_mean, dtype=torch.float32).view(1, 1, -1)
        )
        self.register_buffer("scaler_scale", scale.view(1, 1, -1))
        self.register_buffer(
            "pca_mean", torch.as_tensor(pca_mean, dtype=torch.float32).view(1, 1, -1)
        )
        self.register_buffer(
            "components", torch.as_tensor(components, dtype=torch.float32).contiguous()
        )
        self.d_out = int(components.shape[0])
        self.c_in = int(components.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != self.c_in:
            raise RuntimeError(
                f"A0 projection expects [B,{self.c_in},H,W], got {tuple(x.shape)}"
            )
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C_t]
        x = (x - self.scaler_mean) / self.scaler_scale
        x = x - self.pca_mean
        y = torch.matmul(x, self.components.t())  # [B, H, W, d_l]
        return y.permute(0, 3, 1, 2)

    def fused_conv_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """1x1 Conv weight/bias that reproduces this projection exactly.

        W = V / sigma_s        [d_l, C_t]
        b = -(mu_s/sigma_s + mu_p) @ V^T
        """

        weight = self.components / self.scaler_scale[0, 0, :]
        bias = -(
            (self.scaler_mean[0, 0, :] / self.scaler_scale[0, 0, :])
            + self.pca_mean[0, 0, :]
        ) @ self.components.t()
        return weight, bias

    def parameter_sha256(self) -> str:
        return numpy_arrays_sha256(
            self.scaler_mean[0, 0, :].detach().cpu().numpy(),
            self.scaler_scale[0, 0, :].detach().cpu().numpy(),
            self.pca_mean[0, 0, :].detach().cpu().numpy(),
            self.components.detach().cpu().numpy(),
        )


def build_projection_bundle(
    scalers: Mapping[str, Mapping[str, np.ndarray]],
    pcas: Mapping[str, Mapping[str, np.ndarray]],
) -> nn.ModuleDict:
    projections: Dict[str, FixedPCAProjection] = {}
    for layer in A0_LAYER_ORDER:
        scaler = scalers[layer]
        pca = pcas[layer]
        projection = FixedPCAProjection(
            scaler_mean=scaler["mean_"],
            scaler_scale=scaler["scale_"],
            pca_mean=pca["mean_"],
            components=pca["components_"],
        )
        if projection.d_out != STUDENT_CHANNELS[layer] or projection.c_in != TEACHER_CHANNELS[layer]:
            raise RuntimeError(
                f"A0 projection contract mismatch for {layer}: "
                f"got d_out={projection.d_out}, c_in={projection.c_in}, "
                f"expected d_out={STUDENT_CHANNELS[layer]}, c_in={TEACHER_CHANNELS[layer]}"
            )
        projections[layer] = projection
    return nn.ModuleDict(projections)


def check_projection_conv_equivalence(
    projection: FixedPCAProjection,
    sample: torch.Tensor,
) -> Dict[str, object]:
    """Check mathematical equivalence and report expected FP32 round-off.

    A direct FP32 matmul and ``conv2d`` accumulate in different orders.  On
    the 96/192/768-channel layers this can produce an absolute difference of
    a few ``1e-5`` even when the fused weights and bias are exactly correct.
    Therefore the acceptance gate is evaluated in FP64, while the FP32 error
    is retained as a deployment-relevant diagnostic.
    """

    sample32 = sample.detach().cpu().float()
    conv32 = nn.Conv2d(projection.c_in, projection.d_out, kernel_size=1, bias=True)
    weight32, bias32 = projection.fused_conv_parameters()
    conv32.weight.data.copy_(weight32[:, :, None, None])
    conv32.bias.data.copy_(bias32)
    with torch.inference_mode():
        y_projected32 = projection(sample32)
        y_convolved32 = conv32(sample32)
    difference32 = (y_convolved32 - y_projected32).abs()
    max_abs_error32 = float(difference32.max().item())
    mean_abs_error32 = float(difference32.mean().item())
    denominator32 = float(y_projected32.norm().clamp_min(1e-12).item())
    relative_l2_error32 = float(difference32.norm().item() / denominator32)

    projection64 = copy.deepcopy(projection).double()
    sample64 = sample32.double()
    conv64 = nn.Conv2d(projection.c_in, projection.d_out, kernel_size=1, bias=True).double()
    weight64, bias64 = projection64.fused_conv_parameters()
    conv64.weight.data.copy_(weight64[:, :, None, None])
    conv64.bias.data.copy_(bias64)
    with torch.inference_mode():
        y_projected64 = projection64(sample64)
        y_convolved64 = conv64(sample64)
    difference64 = (y_convolved64 - y_projected64).abs()
    max_abs_error64 = float(difference64.max().item())
    mean_abs_error64 = float(difference64.mean().item())
    denominator64 = float(y_projected64.norm().clamp_min(1e-12).item())
    relative_l2_error64 = float(difference64.norm().item() / denominator64)
    return {
        "input_shape": list(sample32.shape),
        "max_abs_error": max_abs_error64,
        "mean_abs_error": mean_abs_error64,
        "relative_l2_error": relative_l2_error64,
        "fp64_max_abs_error": max_abs_error64,
        "fp64_mean_abs_error": mean_abs_error64,
        "fp64_relative_l2_error": relative_l2_error64,
        "fp32_max_abs_error": max_abs_error32,
        "fp32_mean_abs_error": mean_abs_error32,
        "fp32_relative_l2_error": relative_l2_error32,
        "passed": bool(max_abs_error64 <= 1e-10 and relative_l2_error64 <= 1e-12),
        "note": (
            "FP64 is the mathematical equivalence gate; FP32 reports normal "
            "matmul-vs-conv accumulation round-off"
        ),
    }


class PretrainStudent(nn.Module):
    """Backbone-only feature extractor used by the A-pretrain stage.

    ``forward`` returns the ``os4/os8/os16`` dictionary so DDP gradient
    reduction works when this module is wrapped with ``DistributedDataParallel``.
    """

    def __init__(self, backbone: nn.Sequential) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        tensor = images
        for index, block in enumerate(self.backbone):
            tensor = block(tensor)
            if index == base.FEATURE_TAPS["os4"]["index"]:
                outputs["os4"] = tensor
            elif index == base.FEATURE_TAPS["os8"]["index"]:
                outputs["os8"] = tensor
            elif index == base.FEATURE_TAPS["os16"]["index"]:
                outputs["os16"] = tensor
                # A-pretrain supervises only OS=4/8/16.  MobileNetV2's
                # backbone.18 is the R-ASPP input expansion block and is not
                # part of the feature loss; executing it here would leave its
                # parameters unused and make DDP reduction fail on the next
                # iteration.  The full block remains in the probe model.
                break
        if set(outputs) != set(A0_LAYER_ORDER):
            raise RuntimeError(
                f"Missing pretrain student features: {set(A0_LAYER_ORDER) - set(outputs)}"
            )
        return outputs


def build_pretrain_student() -> PretrainStudent:
    """Scratch MobileNetV2 backbone (``weights=None``), the A0-A6 common init."""

    return PretrainStudent(base.build_backbone())


def build_probe_model(head_channels: int, dropout: float):
    """Full student used by the probe: load the pretrained backbone, freeze it."""

    model = base.build_model(head_channels, dropout)
    model.backbone.requires_grad_(False)
    model.head.requires_grad_(True)
    model.backbone.eval()
    if not model.head.parameters() or any(
        parameter.requires_grad for parameter in model.backbone.parameters()
    ):
        raise RuntimeError("A0 probe must optimize only the R-ASPP head")
    return model


def set_probe_train_mode(model: torch.nn.Module) -> None:
    """Train the head while keeping the frozen backbone (including BatchNorm) eval."""

    model.train()
    module = model.module if isinstance(model, DDP) else model
    module.backbone.eval()
    module.head.train()


def load_pretrain_backbone_state(pretrain_checkpoint: Path) -> Dict[str, torch.Tensor]:
    sidecar = Path(pretrain_checkpoint).with_name(
        f"{Path(pretrain_checkpoint).name}.sha256"
    )
    if sidecar.is_file():
        t0.verify_checkpoint_sidecar(Path(pretrain_checkpoint))
    payload = t0.safe_torch_load(Path(pretrain_checkpoint), map_location="cpu", weights_only=False)
    if payload.get("artifact_type") != ARTIFACT_TYPE_PRETRAIN:
        raise RuntimeError(
            f"Not an A0 pretrain artifact: {payload.get('artifact_type')!r}"
        )
    state = payload["model_state_dict"]
    expected_hash = payload.get("model_state_sha256")
    if expected_hash and t0.state_dict_sha256(state) != expected_hash:
        raise RuntimeError("A0 pretrain checkpoint model-state SHA-256 verification failed")
    expected = {key for key in state if key.startswith("backbone.")}
    unexpected = set(state) - expected
    if unexpected:
        raise RuntimeError(
            f"A0 pretrain artifact contains unexpected state keys: {sorted(unexpected)[:5]}"
        )
    if not expected:
        raise RuntimeError("A0 pretrain artifact contains no backbone parameters")
    # The pretrain module is ``PretrainStudent(backbone=...)`` and therefore
    # serializes keys as ``backbone.<module-key>``.  The probe loads directly
    # into ``model.backbone``, whose keys start at ``<module-key>``.
    return {
        key[len("backbone.") :]: value
        for key, value in state.items()
    }


def _reduce_scalar_sum(value: float, device: torch.device, world_size: int) -> float:
    if world_size == 1:
        return value
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def pretrain_one_epoch_server(
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
    """One A-pretrain epoch: label-free dense feature MSE, backbone only."""

    if sampler is not None:
        sampler.set_epoch(epoch)
    model.train()
    teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    batch_count = 0
    layer_loss_sums = {layer: 0.0 for layer in A0_LAYER_ORDER}
    optimizer_steps = 0
    first_step_gradient_l2: Optional[float] = None
    gradient_samples: List[Dict[str, object]] = []
    possible_steps = math.ceil(len(loader) / accumulation_steps)
    target_steps = min(possible_steps, remaining_optimizer_steps)
    max_batches = min(len(loader), target_steps * accumulation_steps)
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch} [A0 pretrain]",
        disable=rank != 0,
    )

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
            for layer in A0_LAYER_ORDER:
                projected_teacher = projection[layer](teacher_features[layer])
                layer_losses[layer] = F.mse_loss(
                    student_features[layer].float(), projected_teacher.float()
                )
        next_optimizer_step = current_optimizer_step + optimizer_steps + 1
        feat_weight = lambda_feat * min(
            1.0, next_optimizer_step / max(int(warmup_steps), 1)
        )
        batch_loss = (sum(layer_losses.values()) / len(A0_LAYER_ORDER)) * feat_weight

        log_gradients = sync_gradients and gradient_log_steps > 0 and (
            next_optimizer_step % gradient_log_steps == 0
        )
        if log_gradients:
            per_layer_grads: Dict[str, float] = {}
            for layer in A0_LAYER_ORDER:
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
                sample: Dict[str, object] = {
                    "optimizer_step": next_optimizer_step,
                    "feat_weight": feat_weight,
                }
                for layer in A0_LAYER_ORDER:
                    sample[f"gradient_l2_{layer}"] = per_layer_grads[layer]
                    sample[f"student_feature_mean_abs_{layer}"] = float(
                        student_features[layer].detach().float().abs().mean().item()
                    )
                gradient_samples.append(sample)

        loss_sum += float(batch_loss.detach().item())
        batch_count += 1
        for layer in A0_LAYER_ORDER:
            layer_loss_sums[layer] += float(layer_losses[layer].detach().item())
        if rank == 0:
            running_mean = loss_sum / max(batch_count, 1)
            progress.set_postfix(
                {
                    "feat": f"{running_mean:.5f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "steps": optimizer_steps,
                }
            )

    if optimizer_steps != target_steps:
        raise RuntimeError(
            f"Pretrain optimizer-step accounting failed: "
            f"actual={optimizer_steps}, expected={target_steps}"
        )
    loss_sum = _reduce_scalar_sum(loss_sum, device, world_size)
    batch_count = int(_reduce_scalar_sum(float(batch_count), device, world_size))
    layer_loss_sums = {
        layer: _reduce_scalar_sum(value, device, world_size)
        for layer, value in layer_loss_sums.items()
    }
    metrics: Dict[str, object] = {
        "loss_total": loss_sum / max(batch_count, 1),
        "loss_os4": layer_loss_sums["os4"] / max(batch_count, 1),
        "loss_os8": layer_loss_sums["os8"] / max(batch_count, 1),
        "loss_os16": layer_loss_sums["os16"] / max(batch_count, 1),
        "loss_total_unweighted": (
            sum(layer_loss_sums.values())
            / (len(A0_LAYER_ORDER) * max(batch_count, 1))
        ),
        "feat_weight_final": feat_weight,
        "feature_gradient_l2_first_optimizer_step": first_step_gradient_l2,
    }
    return metrics, optimizer_steps, gradient_samples


def probe_one_epoch_server(
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
    """One A-probe epoch: frozen backbone + head-only pixel CE (S2-F style)."""

    if sampler is not None:
        sampler.set_epoch(epoch)
    set_probe_train_mode(model)
    optimizer.zero_grad(set_to_none=True)
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    loss_sum = 0.0
    valid_pixels = 0
    optimizer_steps = 0
    first_step_gradient_l2: Optional[float] = None
    possible_steps = math.ceil(len(loader) / accumulation_steps)
    target_steps = min(possible_steps, remaining_optimizer_steps)
    max_batches = min(len(loader), target_steps * accumulation_steps)
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch} [A0 probe]",
        disable=rank != 0,
    )

    for batch_index, (images, targets, _paths) in enumerate(progress):
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
                raise RuntimeError("Probe batch contains no valid Cityscapes pixels")
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
        confusion += t0.confusion_counts(predictions, targets)
        loss_sum += float(batch_loss_sum.detach().item())
        valid_pixels += batch_valid
        if rank == 0:
            running = t0.metrics_from_confusion(confusion, loss_sum, valid_pixels)
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
            f"Probe optimizer-step accounting failed: "
            f"actual={optimizer_steps}, expected={target_steps}"
        )
    metrics = s2_0_server._reduce_train_metrics(confusion, loss_sum, valid_pixels, device, world_size)
    metrics["ce_gradient_l2_first_optimizer_step"] = first_step_gradient_l2
    return metrics, optimizer_steps


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear-kernel CKA between two feature tensors of equal N."""

    x = x.float().flatten(2).transpose(1, 2).reshape(-1, x.shape[1])
    y = y.float().flatten(2).transpose(1, 2).reshape(-1, y.shape[1])
    if x.shape[0] != y.shape[0]:
        raise RuntimeError(f"CKA spatial mismatch: {x.shape} vs {y.shape}")
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    cross = x.t() @ y
    hsic = cross.pow(2).sum()
    denominator = (x.t() @ x).norm() * (y.t() @ y).norm()
    if float(denominator) <= 1e-12:
        return 0.0
    return float((hsic / denominator).item())


def compute_probe_diagnostics(
    teacher: torch.nn.Module,
    probe_model: torch.nn.Module,
    projection: nn.ModuleDict,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int = 8,
) -> Dict[str, object]:
    """CKA and alignment residuals over the first ``max_batches`` of dev_local."""

    teacher.eval()
    probe_model.eval()
    # Keep the full student wrapper so the same feature-tap implementation as
    # training is used.  ``model.backbone(images)`` returns only a Tensor, not
    # the ``{"os4", "os8", "os16", ...}`` feature dictionary expected below.
    student = probe_model.module if isinstance(probe_model, DDP) else probe_model
    cka_sums = {layer: 0.0 for layer in A0_LAYER_ORDER}
    residual_sums = {layer: 0.0 for layer in A0_LAYER_ORDER}
    teacher_norm_sums = {layer: 0.0 for layer in A0_LAYER_ORDER}
    batches = 0
    with torch.inference_mode():
        for images, _targets, _paths in loader:
            if batches >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            teacher_features = teacher.extract_features(images)
            student_features = student.extract_features(images)
            for layer in A0_LAYER_ORDER:
                projected_teacher = projection[layer](teacher_features[layer]).float()
                student_feature = student_features[layer].float()
                cka_sums[layer] += linear_cka(student_feature, projected_teacher)
                residual_sums[layer] += float(
                    (student_feature - projected_teacher).pow(2).sum().item()
                )
                teacher_norm_sums[layer] += float(projected_teacher.pow(2).sum().item())
            batches += 1

    if batches == 0:
        raise RuntimeError("Probe diagnostics sampled no dev batches")
    diagnostics: Dict[str, object] = {
        "sampled_batches": batches,
        "method": (
            "Linear CKA between student and projected teacher features; "
            "relative MSE alignment residual ||s - p(t)||^2 / ||p(t)||^2"
        ),
        "layers": {},
    }
    for layer in A0_LAYER_ORDER:
        diagnostics["layers"][layer] = {
            "cka": cka_sums[layer] / batches,
            "alignment_relative_mse": residual_sums[layer] / max(teacher_norm_sums[layer], 1e-12),
            "student_channels": STUDENT_CHANNELS[layer],
            "teacher_channels": TEACHER_CHANNELS[layer],
        }
    return diagnostics


def _pretrain_smoke_test(
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
    images, _targets, paths = next(iter(loader))
    images = images.to(device, non_blocking=True)
    with t0.autocast_context(device, amp_enabled):
        with torch.no_grad():
            teacher_features = teacher.extract_features(images)
        student_features = model(images)
        losses: Dict[str, torch.Tensor] = {}
        for layer in A0_LAYER_ORDER:
            projected_teacher = projection[layer](teacher_features[layer])
            losses[layer] = F.mse_loss(
                student_features[layer].float(), projected_teacher.float()
            )
    total = sum(losses.values()) / len(A0_LAYER_ORDER)
    total.backward()
    if not torch.isfinite(total):
        raise RuntimeError(f"Non-finite A0 pretrain smoke loss: {total.item()}")
    module = model.module if isinstance(model, DDP) else model
    backbone_gradients = sum(
        parameter.grad is not None for parameter in module.backbone.parameters()
    )
    if backbone_gradients == 0:
        raise RuntimeError("A0 pretrain smoke test produced no backbone gradients")
    if rank == 0:
        print(
            f"[OK] A0 pretrain smoke test: sample={paths[0]}, "
            f"teacher os16={tuple(teacher_features['os16'].shape)}, "
            f"student os16={tuple(student_features['os16'].shape)}, "
            f"feature loss={total.item():.6f}, backbone_grad_tensors={backbone_gradients}"
        )


def _probe_smoke_test(
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
    with t0.autocast_context(device, amp_enabled):
        logits = model(images)
    loss = F.cross_entropy(logits.float(), targets, ignore_index=IGNORE_INDEX)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite A0 probe smoke loss: {loss.item()}")
    module = model.module if isinstance(model, DDP) else model
    if any(parameter.grad is not None for parameter in module.backbone.parameters()):
        raise RuntimeError("A0 probe smoke test found a gradient in the frozen backbone")
    if rank == 0:
        print(
            f"[OK] A0 probe smoke test: sample={paths[0]}, "
            f"logits={tuple(logits.shape)}, loss={loss.item():.6f}"
        )


def build_pretrain_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    optimizer_step: int,
    initial_backbone_state_sha256: str,
    config: Mapping[str, object],
    hashes: Mapping[str, object],
    dataset_lock: Mapping[str, object],
) -> Dict[str, object]:
    state = t0.cpu_state_dict(model)
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_type": ARTIFACT_TYPE_PRETRAIN,
        "experiment": EXPERIMENT,
        "stage": "pretrain",
        "initialization": "weights=None",
        "loss": "3-layer dense feature MSE (label-free)",
        "model_state_dict": state,
        "model_state_sha256": t0.state_dict_sha256(state),
        "initial_backbone_state_sha256": initial_backbone_state_sha256,
        "best_epoch": epoch,
        "best_optimizer_step": optimizer_step,
        "config": copy.deepcopy(config),
        "hashes": copy.deepcopy(hashes),
        "dataset_lock": copy.deepcopy(dataset_lock),
    }


def build_probe_best_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    optimizer_step: int,
    dev_metrics: Mapping[str, object],
    config: Mapping[str, object],
    hashes: Mapping[str, object],
    dataset_lock: Mapping[str, object],
    shape_audit: Mapping[str, object],
    projection_equivalence: Mapping[str, object],
    pca_parameter_record: Mapping[str, object],
) -> Dict[str, object]:
    model_state = t0.cpu_state_dict(model)
    backbone_state = {k: v for k, v in model_state.items() if k.startswith("backbone.")}
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_type": ARTIFACT_TYPE_PROBE,
        "experiment": EXPERIMENT,
        "model_name": MODEL_NAME,
        "initialization": "weights=None + A0 fixed-PCA feature pretrain",
        "projection": "standard_scaler+pca (fixed, explicit path)",
        "backbone_frozen": True,
        "trainable_scope": "R-ASPP head only (frozen-backbone probe)",
        "num_classes": NUM_CLASSES,
        "class_names": list(t0.CITYSCAPES_CLASSES),
        "output_stride": OUTPUT_STRIDE,
        "head_type": "R-ASPP",
        "feature_taps": copy.deepcopy(base.FEATURE_TAPS),
        "model_state_dict": model_state,
        "model_state_sha256": t0.state_dict_sha256(model_state),
        "backbone_state_sha256": t0.state_dict_sha256(backbone_state),
        "best_epoch": epoch,
        "best_optimizer_step": optimizer_step,
        "best_dev_metrics": copy.deepcopy(dev_metrics),
        "config": copy.deepcopy(config),
        "hashes": copy.deepcopy(hashes),
        "dataset_lock": copy.deepcopy(dataset_lock),
        "shape_audit": copy.deepcopy(shape_audit),
        "projection_equivalence": copy.deepcopy(projection_equivalence),
        "pca_parameters_sha256_record": copy.deepcopy(pca_parameter_record),
    }


def load_probe_model(
    checkpoint_path: Path,
    device: object = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    checkpoint_path = Path(checkpoint_path).resolve()
    t0.verify_checkpoint_sidecar(checkpoint_path)
    payload = t0.safe_torch_load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("artifact_type") != ARTIFACT_TYPE_PROBE:
        raise RuntimeError(f"Not an A0 probe artifact: {payload.get('artifact_type')!r}")
    if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise RuntimeError("Unsupported A0 probe artifact format")
    if payload.get("feature_taps") != base.FEATURE_TAPS:
        raise RuntimeError("A0 probe artifact feature taps differ from the locked contract")
    config = payload["config"]
    model = build_probe_model(config["head_channels"], config["dropout"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    actual_hash = t0.state_dict_sha256(model.state_dict())
    if actual_hash != payload["model_state_sha256"]:
        raise RuntimeError("A0 probe model state failed SHA-256 verification")
    model = model.to(torch.device(device)).eval()
    return model, payload


def _poly_lr_factor(args: argparse.Namespace, max_steps: int) -> Any:
    def lr_factor(step: int) -> float:
        progress = min(step, max_steps) / max(max_steps, 1)
        return max((1.0 - progress) ** args.poly_power, args.min_lr_ratio)
    return lr_factor


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
    """A-pretrain stage: 40k label-free dense feature MSE on the backbone."""

    main_process = rank == 0
    model = build_pretrain_student().to(device)
    initial_backbone_hash = t0.state_dict_sha256(model.backbone.state_dict())
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            # backbone.18 is intentionally outside the OS=4/8/16 feature
            # pretraining graph.  Let DDP account for that unused block;
            # the probe stage uses the complete backbone separately.
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_poly_lr_factor(args, args.pretrain_max_steps)
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
            raise RuntimeError("A0 pretrain resume configuration differs from current arguments")
        if resume_payload.get("artifact_type") != ARTIFACT_TYPE_PRETRAIN:
            raise RuntimeError("Resume file is not an A0 pretrain checkpoint")
        model_to_load = model.module if isinstance(model, DDP) else model
        saved_initial_hash = resume_payload.get("initial_backbone_state_sha256")
        if saved_initial_hash != initial_backbone_hash:
            raise RuntimeError(
                "Resume checkpoint was created from a different scratch backbone initialization"
            )
        model_to_load.load_state_dict(resume_payload["model_state_dict"], strict=True)
        saved_model_hash = resume_payload.get("model_state_sha256")
        if saved_model_hash and t0.state_dict_sha256(model_to_load.state_dict()) != saved_model_hash:
            raise RuntimeError("A0 pretrain resume model state hash verification failed")
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
                f"[OK] Resuming A0 pretrain after epoch {epoch}, "
                f"step {cumulative_optimizer_steps:,}"
            )

    if not resume and any(
        path.is_file() for path in (paths["pretrain_last"], paths["pretrain_history"])
    ):
        raise FileExistsError(
            f"A0 pretrain artifacts already exist in {paths['run_dir']}; use --resume"
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
        train_metrics, optimizer_steps, grad_samples = pretrain_one_epoch_server(
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
        gradient_rows.extend(grad_samples)
        # Epochs rarely end exactly on a 5k-step boundary (with the locked
        # dataset/global batch this is 317 steps/epoch).  Snapshot once the
        # boundary has been crossed, otherwise the old modulo test produced
        # only the final snapshot.
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
                    "learning_rate": optimizer.param_groups[0]["lr"],
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
                "initialization": "weights=None",
                "model_state_dict": t0.cpu_state_dict(
                    model.module if isinstance(model, DDP) else model
                ),
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
            }
            last_payload["model_state_sha256"] = t0.state_dict_sha256(
                last_payload["model_state_dict"]
            )
            t0.torch_save_atomic(last_payload, paths["pretrain_last"])
            if should_snapshot:
                snapshot_payload = build_pretrain_checkpoint(
                    model=model.module if isinstance(model, DDP) else model,
                    epoch=epoch,
                    optimizer_step=cumulative_optimizer_steps,
                    initial_backbone_state_sha256=initial_backbone_hash,
                    config=config,
                    hashes=hashes,
                    dataset_lock=dataset_lock,
                )
                snapshot_path = (
                    paths["pretrain_snapshots"]
                    / f"a0_pretrain_snapshot_step_{cumulative_optimizer_steps:05d}.pth"
                )
                snapshot_hash = t0.write_checkpoint_with_sidecar(snapshot_payload, snapshot_path)
                print(
                    f"[OK] A0 pretrain snapshot: step={cumulative_optimizer_steps:,}, "
                    f"feature_loss={train_metrics['loss_total']:.5f}, sha256={snapshot_hash}"
                )
            print(
                f"Epoch {epoch}: step={cumulative_optimizer_steps:,}/"
                f"{args.pretrain_max_steps:,}, "
                f"feature_loss={train_metrics['loss_total']:.5f}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
        s2_0_server.barrier(world_size)

    s2_0_server.barrier(world_size)
    model_core = model.module if isinstance(model, DDP) else model
    final_backbone_hash = t0.state_dict_sha256(model_core.backbone.state_dict())
    info = {
        "pretrain_checkpoint": paths["pretrain_last"],
        "pretrain_optimizer_steps": cumulative_optimizer_steps,
        "pretrain_epochs": epoch,
        "initial_backbone_state_sha256": initial_backbone_hash,
        "final_backbone_state_sha256": final_backbone_hash,
        "elapsed_seconds": time.time() - training_started,
    }
    if main_process:
        print(
            f"[DONE] A0 pretrain: steps={cumulative_optimizer_steps:,}, "
            f"epochs={epoch}, backbone_sha256={final_backbone_hash}"
        )
    return info


def run_probe_stage(
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
    """A-probe stage: frozen pretrained backbone + head-only pixel CE, 40k steps."""

    main_process = rank == 0
    pretrain_checkpoint = Path(args.pretrain_checkpoint or paths["pretrain_last"]).resolve()
    if not pretrain_checkpoint.is_file():
        raise FileNotFoundError(
            f"Pretrain checkpoint not found: {pretrain_checkpoint}. "
            "Run --stage pretrain or --stage full first, or pass --pretrain-checkpoint."
        )
    backbone_state = load_pretrain_backbone_state(pretrain_checkpoint)

    model = build_probe_model(args.head_channels, args.dropout).to(device)
    model.backbone.load_state_dict(backbone_state, strict=True)
    loaded_backbone_hash = t0.state_dict_sha256(model.backbone.state_dict())
    if main_process:
        print(
            f"[OK] A0 probe loads pretrained backbone from {pretrain_checkpoint}, "
            f"backbone_sha256={loaded_backbone_hash}"
        )
    shape_audit = base.audit_model_shapes(
        model, device, args.crop_height, args.crop_width, amp_enabled
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

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_poly_lr_factor(args, args.probe_max_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: List[Dict[str, object]] = []
    best_key: Optional[Tuple[float, float, float, float]] = None
    best_epoch: Optional[int] = None
    best_optimizer_step: Optional[int] = None
    best_dev_metrics: Optional[Dict[str, object]] = None
    epoch = 0
    cumulative_optimizer_steps = 0
    if resume and paths["probe_last"].is_file():
        resume_payload = t0.safe_torch_load(
            paths["probe_last"], map_location="cpu", weights_only=False
        )
        if resume_payload.get("config") != config:
            raise RuntimeError("A0 probe resume configuration differs from current arguments")
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(resume_payload["model_state_dict"], strict=True)
        if t0.state_dict_sha256(model_to_load.backbone.state_dict()) != loaded_backbone_hash:
            raise RuntimeError("Resume checkpoint changed the frozen A0 probe backbone")
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        scaler.load_state_dict(resume_payload["scaler_state_dict"])
        history = resume_payload["history"]
        best_key = resume_payload["best_key"]
        best_epoch = resume_payload["best_epoch"]
        best_optimizer_step = resume_payload["best_optimizer_step"]
        best_dev_metrics = resume_payload["best_dev_metrics"]
        train_generator.set_state(resume_payload["train_generator_state"])
        epoch = int(resume_payload["epoch"])
        cumulative_optimizer_steps = int(resume_payload["optimizer_steps"])
        if main_process:
            print(
                f"[OK] Resuming A0 probe after epoch {epoch}, "
                f"step {cumulative_optimizer_steps:,}"
            )

    if not resume and any(
        path.is_file() for path in (paths["probe_last"], paths["best_probe"], paths["dev_metrics"])
    ):
        raise FileExistsError(
            f"A0 probe artifacts already exist in {paths['run_dir']}; use --resume"
        )

    next_eval_step = (
        ((cumulative_optimizer_steps // args.eval_every_steps) + 1)
        * args.eval_every_steps
        if args.eval_every_steps > 0
        else math.inf
    )
    training_started = time.time()
    while cumulative_optimizer_steps < args.probe_max_steps:
        epoch += 1
        remaining_steps = args.probe_max_steps - cumulative_optimizer_steps
        train_metrics, optimizer_steps = probe_one_epoch_server(
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
        # Evaluate at the first epoch boundary at/after each requested
        # interval.  Exact modulo checks miss every 5k evaluation because an
        # epoch is 317 optimizer steps under the locked global batch of 8.
        should_evaluate = (
            cumulative_optimizer_steps >= next_eval_step
        ) or cumulative_optimizer_steps == args.probe_max_steps
        if should_evaluate and args.eval_every_steps > 0:
            while cumulative_optimizer_steps >= next_eval_step:
                next_eval_step += args.eval_every_steps
        dev_metrics: Optional[Dict[str, object]] = None
        if should_evaluate:
            s2_0_server.barrier(world_size)
            if main_process:
                assert dev_loader is not None
                dev_metrics, _ = t0.evaluate(
                    model=model.module if isinstance(model, DDP) else model,
                    loader=dev_loader,
                    device=device,
                    amp_enabled=amp_enabled,
                    split_name="dev_local server (A0 probe)",
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
                    best_payload = build_probe_best_checkpoint(
                        model=model.module if isinstance(model, DDP) else model,
                        epoch=epoch,
                        optimizer_step=cumulative_optimizer_steps,
                        dev_metrics=dev_metrics,
                        config=config,
                        hashes=hashes,
                        dataset_lock=dataset_lock,
                        shape_audit=shape_audit,
                        projection_equivalence=projection_equivalence,
                        pca_parameter_record=pca_parameter_record,
                    )
                    checkpoint_hash = t0.write_checkpoint_with_sidecar(
                        best_payload, paths["best_probe"]
                    )
                    print(
                        f"[OK] A0 probe best updated: step={cumulative_optimizer_steps:,}, "
                        f"dev_mIoU={dev_metrics['mIoU']:.6f}, sha256={checkpoint_hash}"
                    )
            s2_0_server.barrier(world_size)

        if main_process:
            history.append(
                {
                    "epoch": epoch,
                    "optimizer_steps": cumulative_optimizer_steps,
                    "optimizer_steps_this_epoch": optimizer_steps,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train": train_metrics,
                    "dev": dev_metrics,
                }
            )
            t0.write_json_atomic(paths["probe_history"], history)
            last_payload = {
                "format_version": ARTIFACT_FORMAT_VERSION,
                "artifact_type": ARTIFACT_TYPE_PROBE,
                "experiment": EXPERIMENT,
                "stage": "probe",
                "model_state_dict": t0.cpu_state_dict(
                    model.module if isinstance(model, DDP) else model
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_generator_state": train_generator.get_state(),
                "history": history,
                "best_key": best_key,
                "best_epoch": best_epoch,
                "best_optimizer_step": best_optimizer_step,
                "best_dev_metrics": best_dev_metrics,
                "epoch": epoch,
                "optimizer_steps": cumulative_optimizer_steps,
                "config": config,
                "hashes": hashes,
                "dataset_lock": dataset_lock,
            }
            t0.torch_save_atomic(last_payload, paths["probe_last"])
            message = (
                f"Epoch {epoch}: step={cumulative_optimizer_steps:,}/{args.probe_max_steps:,}, "
                f"train_mIoU={train_metrics['mIoU']:.4f}, "
                f"train_loss={train_metrics['loss']:.4f}"
            )
            if dev_metrics is not None:
                message += f", dev_mIoU={dev_metrics['mIoU']:.4f}"
            print(message)
        s2_0_server.barrier(world_size)

    s2_0_server.barrier(world_size)
    selected_dev_metrics: Optional[Dict[str, object]] = None
    diagnostics: Optional[Dict[str, object]] = None
    efficiency: Optional[Dict[str, object]] = None
    if main_process:
        if best_epoch is None or best_optimizer_step is None or best_dev_metrics is None:
            raise RuntimeError("A0 probe ended without a selected dev checkpoint")
        selected_model, selected_payload = load_probe_model(paths["best_probe"], device=device)
        selected_dev_metrics, per_image_rows = t0.evaluate(
            model=selected_model,
            loader=dev_loader,
            device=device,
            amp_enabled=amp_enabled,
            split_name="selected dev_local server (A0 probe)",
            boundary_tolerance=args.boundary_tolerance,
            collect_per_image=True,
        )
        if not t0.metrics_reproduce(selected_dev_metrics, best_dev_metrics):
            raise RuntimeError(
                "Reloaded A0 probe checkpoint did not reproduce best dev metrics: "
                f"saved={best_dev_metrics['mIoU']}, reloaded={selected_dev_metrics['mIoU']}"
            )
        t0.write_jsonl_atomic(paths["per_image"], per_image_rows)
        checkpoint_hash = t0.verify_checkpoint_sidecar(paths["best_probe"])
        if args.benchmark:
            efficiency = base.benchmark_model(
                selected_model,
                device,
                args.benchmark_height,
                args.benchmark_width,
                args.benchmark_warmup,
                args.benchmark_runs,
            )
        t0.write_json_atomic(paths["efficiency"], efficiency)
        assert dev_loader is not None
        diagnostics = compute_probe_diagnostics(
            teacher,
            selected_model,
            projection,
            dev_loader,
            device,
            amp_enabled,
            max_batches=8,
        )
        results = {
            "experiment": EXPERIMENT,
            "protocol": (
                "Scratch MobileNetV2 backbone trained label-free with 3-layer fixed "
                "StandardScaler+PCA feature MSE from the frozen T1 teacher (40k steps), "
                "then a frozen-backbone 19-class R-ASPP probe trained with pixel CE "
                "(40k steps). Best checkpoint is selected by dev_local mIoU; test_local "
                "is not evaluated. A0 has no train-time adapter, so the probe model is "
                "deployable as-is."
            ),
            "best_epoch": best_epoch,
            "best_optimizer_step": best_optimizer_step,
            "best_dev_metrics": selected_dev_metrics,
            "diagnostics": diagnostics,
            "class_names": t0.CITYSCAPES_CLASSES,
            "config": config,
            "shape_audit": shape_audit,
            "dataset_lock": dataset_lock,
            "model": {
                "model_name": MODEL_NAME,
                "initialization": "weights=None + A0 fixed-PCA feature pretrain",
                "head": "R-ASPP",
                "backbone_frozen": True,
                "feature_taps": base.FEATURE_TAPS,
            },
            "efficiency": efficiency,
            "hashes": {
                **hashes,
                "selected_model_state_sha256": selected_payload["model_state_sha256"],
                "checkpoint_sha256": checkpoint_hash,
                "pretrain_backbone_state_sha256": loaded_backbone_hash,
            },
            "training": {
                "elapsed_seconds": time.time() - training_started,
                "optimizer_steps": cumulative_optimizer_steps,
                "epochs_completed": epoch,
                "steps_per_full_epoch": math.ceil(len(train_loader) / accumulation_steps),
            },
            "software": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "torchvision": str(base.torchvision.__version__),
                "numpy": np.__version__,
                "pillow": __import__("PIL").__version__,
                "platform": platform.platform(),
            },
            "artifacts": {key: str(value) for key, value in paths.items()},
        }
        t0.write_json_atomic(paths["dev_metrics"], results)
        print(
            f"[DONE] A0 probe: steps={cumulative_optimizer_steps:,}, "
            f"best dev mIoU={selected_dev_metrics['mIoU']:.6f}"
        )
    s2_0_server.barrier(world_size)
    return {
        "best_dev_metrics": selected_dev_metrics,
        "best_optimizer_step": best_optimizer_step,
        "best_epoch": best_epoch,
        "checkpoint": paths["best_probe"],
        "pretrain_checkpoint": pretrain_checkpoint,
    }


def build_config(
    args: argparse.Namespace,
    accumulation_steps: int,
    world_size: int,
    amp_enabled: bool,
    teacher_checkpoint_sha256: str,
    sampling_manifest_sha256: str,
    pca_parameter_record: Mapping[str, object],
    projection_hashes: Mapping[str, str],
) -> Dict[str, object]:
    return {
        "experiment": EXPERIMENT,
        "server_entry_point": str(Path(__file__).resolve()),
        "stage": args.stage,
        "seed": args.seed,
        "world_size": world_size,
        "batch_size_per_gpu": args.batch_size,
        "global_batch_size": args.batch_size * accumulation_steps * world_size,
        "accumulation_steps_per_gpu": accumulation_steps,
        "pretrain_max_optimizer_steps": args.pretrain_max_steps,
        "probe_max_optimizer_steps": args.probe_max_steps,
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
        "lambda_feat": args.lambda_feat,
        "feature_warmup_ratio": args.feature_warmup_ratio,
        "feature_warmup_steps": int(args.pretrain_max_steps * args.feature_warmup_ratio),
        "gradient_log_steps": args.gradient_log_steps,
        "pretrain_snapshot_steps": args.pretrain_snapshot_steps,
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
        "loss": "3-layer feature MSE (pretrain) + frozen-backbone pixel CE probe",
        "knowledge_distillation": True,
        "distillation_type": "label-free intermediate feature distillation",
        "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        "pca_dir": str(Path(args.pca_dir).resolve()),
        "sampling_manifest_sha256": sampling_manifest_sha256,
        "pca_parameters_sha256_record": pca_parameter_record,
        "projection_parameter_sha256": dict(projection_hashes),
        "test_local_evaluated": False,
        "distributed_backend": "nccl" if world_size > 1 else None,
    }


def run_training(args: argparse.Namespace) -> None:
    rank, local_rank, world_size, device = s2_0_server.setup_distributed(args)
    main_process = rank == 0
    train_loader: Optional[DataLoader] = None
    dev_loader: Optional[DataLoader] = None
    successful_exit = False
    try:
        accumulation_steps = s2_0_server.effective_accumulation_steps(args, world_size)
        t0.set_global_seed(args.seed + rank, args.deterministic)
        dataset_root = args.dataset_root.resolve()
        dataset_lock, entries_by_split = t0.validate_dataset_lock(dataset_root)

        if args.stage == "pca":
            pca_error: Optional[str] = None
            if main_process:
                try:
                    run_pca_stage(args, device, dataset_root, dataset_lock, entries_by_split)
                except Exception as error:
                    pca_error = f"{type(error).__name__}: {error}"
            if world_size > 1:
                error_payload = [pca_error]
                dist.broadcast_object_list(error_payload, src=0)
                pca_error = error_payload[0]
            if pca_error:
                raise RuntimeError(f"A0 PCA stage failed on rank 0: {pca_error}")
            successful_exit = True
            return

        teacher, teacher_payload = load_teacher_for_distillation(
            args.teacher_checkpoint,
            repo_dir=args.teacher_repo_dir,
            weights_path=args.teacher_weights_path,
            device=device,
            verify_checkpoint_file=True,
        )
        teacher.eval()
        scalers, pcas, pca_parameter_record = load_pca_parameters(args.pca_dir)
        pca_teacher_hash = str(pca_parameter_record.get("teacher_checkpoint_sha256", ""))
        current_teacher_hash = t0.verify_checkpoint_sidecar(args.teacher_checkpoint)
        if pca_teacher_hash and pca_teacher_hash != current_teacher_hash:
            raise RuntimeError(
                "PCA artifacts were fitted with a different teacher checkpoint: "
                f"pca={pca_teacher_hash}, current={current_teacher_hash}"
            )
        pca_dataset_hash = str(
            pca_parameter_record.get("dataset_combined_manifest_sha256", "")
        )
        if pca_dataset_hash and pca_dataset_hash != dataset_lock["combined_manifest_sha256"]:
            raise RuntimeError("PCA artifacts were fitted with a different dataset manifest")
        projection = build_projection_bundle(scalers, pcas).to(device)
        projection_hashes = {
            layer: projection[layer].parameter_sha256() for layer in A0_LAYER_ORDER
        }
        sampling_manifest_sha256 = str(
            pca_parameter_record.get("sampling_manifest_sha256", "")
        )
        teacher_checkpoint_sha256 = t0.verify_checkpoint_sidecar(args.teacher_checkpoint)

        paths = a0_paths(args.output_dir, args.seed)
        paths["run_dir"].mkdir(parents=True, exist_ok=True)
        amp_enabled = bool(args.amp and device.type == "cuda")
        config = build_config(
            args,
            accumulation_steps,
            world_size,
            amp_enabled,
            teacher_checkpoint_sha256,
            sampling_manifest_sha256,
            pca_parameter_record,
            projection_hashes,
        )
        hashes = {
            "training_script_sha256": t0.sha256_file(Path(__file__).resolve()),
            "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
            "sampling_manifest_sha256": sampling_manifest_sha256,
            "projection_parameter_sha256": dict(projection_hashes),
        }

        feature_taps = build_feature_taps_record()
        t0.write_json_atomic(paths["feature_taps"], feature_taps)
        projection_equivalence: Dict[str, object] = {}
        if main_process:
            for layer in A0_LAYER_ORDER:
                cpu_projection = FixedPCAProjection(
                    scaler_mean=scalers[layer]["mean_"],
                    scaler_scale=scalers[layer]["scale_"],
                    pca_mean=pcas[layer]["mean_"],
                    components=pcas[layer]["components_"],
                )
                sample = torch.randn(2, TEACHER_CHANNELS[layer], 32, 64)
                projection_equivalence[layer] = check_projection_conv_equivalence(
                    cpu_projection, sample
                )
                if not projection_equivalence[layer]["passed"]:
                    raise RuntimeError(
                        f"A0 fixed PCA projection equivalence failed for {layer}: "
                        f"max_abs_error={projection_equivalence[layer]['max_abs_error']}"
                    )
            t0.write_json_atomic(paths["projection_equivalence"], projection_equivalence)
            print(
                "[OK] A0 projection equivalence checks:",
                {layer: projection_equivalence[layer]["passed"] for layer in A0_LAYER_ORDER},
            )
        s2_0_server.barrier(world_size)

        train_loader, train_sampler, train_generator = s2_0_server.build_train_loader(
            args, dataset_root, entries_by_split, device, rank, world_size
        )
        dev_loader = (
            s2_0_server.build_dev_loader(args, dataset_root, entries_by_split, device)
            if main_process
            else None
        )
        if main_process:
            steps_per_full_epoch = math.ceil(len(train_loader) / accumulation_steps)
            print(
                f"[INFO] A0 server DDP: world_size={world_size}, device={device}, "
                f"AMP={amp_enabled}, workers/rank={args.num_workers}, "
                f"context={args.multiprocessing_context}, pin_memory={args.pin_memory}"
            )
            print(
                f"[INFO] global batch={args.batch_size * accumulation_steps * world_size}; "
                f"pretrain steps={args.pretrain_max_steps:,} ({steps_per_full_epoch} steps/epoch); "
                f"probe steps={args.probe_max_steps:,}"
            )

        if args.smoke_test:
            smoke_model = build_pretrain_student().to(device)
            if world_size > 1:
                smoke_model = DDP(
                    smoke_model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    broadcast_buffers=True,
                    find_unused_parameters=True,
                    gradient_as_bucket_view=True,
                )
            _pretrain_smoke_test(
                smoke_model, teacher, projection, train_loader, device, amp_enabled, rank
            )
            smoke_model = None
            probe_smoke_model = build_probe_model(args.head_channels, args.dropout).to(device)
            if world_size > 1:
                probe_smoke_model = DDP(
                    probe_smoke_model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    broadcast_buffers=True,
                    find_unused_parameters=False,
                    gradient_as_bucket_view=True,
                )
            _probe_smoke_test(
                probe_smoke_model, train_loader, device, amp_enabled, rank
            )
            probe_smoke_model = None
            successful_exit = True
            return

        pretrain_info: Optional[Dict[str, object]] = None
        if args.stage in ("full", "pretrain"):
            pretrain_info = run_pretrain_stage(
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
                accumulation_steps,
                dataset_lock,
                paths,
                config,
                hashes,
                resume=args.resume,
            )

        probe_info: Optional[Dict[str, object]] = None
        if args.stage in ("full", "probe"):
            probe_info = run_probe_stage(
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
                resume=args.resume,
            )

        successful_exit = True
    finally:
        # Ordered teardown from ``server_training_issues_and_solutions.md``:
        # stop workers -> CUDA synchronize -> release DDP/optimizer -> barrier ->
        # destroy process group.  Exception paths avoid collectives.
        s2_0_server._shutdown_loader(train_loader)
        s2_0_server._shutdown_loader(dev_loader)
        s2_0_server._synchronize_cuda(device)
        if successful_exit and world_size > 1 and dist.is_initialized():
            s2_0_server.barrier(world_size)
        s2_0_server._synchronize_cuda(device)
        if successful_exit and world_size > 1 and dist.is_initialized():
            s2_0_server.barrier(world_size)
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
