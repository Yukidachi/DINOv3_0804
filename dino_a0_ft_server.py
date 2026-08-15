"""A0-FT server training entry point: end-to-end CE fine-tune from the A0 probe.

Experiment A0-FT from ``plan_markdown/Cityscapes知识蒸馏实验详单.md``:

    start point : best A0 frozen-backbone probe checkpoint
                  (``result/A_MobileNetV2_RASPP_server/A0/seed_<seed>/
                    a0_probe_mobilenetv2_raspp_best.pth``); both the
                  PCA-pretrained MobileNetV2 backbone and the R-ASPP head are
                  loaded from it.
    backbone    : UNFROZEN, jointly optimized with the head
    loss        : 19-class pixel cross-entropy only (no feature/logits KD)
    budget      : 80,000 optimizer steps, identical to the S2-0 scratch baseline
    optimizer   : SGD(lr=0.01, momentum=0.9, weight_decay=1e-4) + poly(0.9)

A0-FT answers the decisive A-group question: does the label-free feature
pretraining actually help *after* supervised end-to-end adaptation, i.e. can a
fine-tuned A0 backbone beat the S2-0 scratch baseline (dev mIoU ~0.4957)?  The
frozen-backbone probe (A0 dev mIoU ~0.3886) only measures linear decodability
of the frozen representation and is structurally capped below S2-0.

This entry point reuses the DDP / ``spawn`` workers / default-off pinned memory
/ gradient-accumulation / ordered-teardown conventions from
``dino_s2_0_server.py`` and ``dino_a0_server.py`` (see
``plan_markdown/server_training_issues_and_solutions.md``).  The training loop,
dev evaluation and checkpoint discipline mirror S2-0 so the only experimental
variable versus S2-0 is the initialization.

To adapt this file for A1-FT / A5-FT: change ``EXPERIMENT``,
``ARTIFACT_TYPE_FINETUNE``, ``RUN_SUBDIR`` and ``DEFAULT_SOURCE_EXPERIMENT``;
the source probe checkpoint is auto-resolved from the sibling A1/A5 run dir and
its artifact type is accepted through ``ACCEPTED_PROBE_ARTIFACT_TYPES``.

Server examples:

    # full A0-FT: 80k end-to-end CE fine-tune from the A0 probe best checkpoint
    torchrun --standalone --nproc_per_node=2 dino_a0_ft_server.py \\
        --seed 42 --batch-size 2 --global-batch-size 8 \\
        --num-workers 8 --multiprocessing-context spawn \\
        --no-pin-memory --persistent-workers

    # resume from the per-run last checkpoint
    torchrun --standalone --nproc_per_node=2 dino_a0_ft_server.py \\
        --seed 42 --batch-size 2 --global-batch-size 8 \\
        --num-workers 8 --multiprocessing-context spawn \\
        --no-pin-memory --persistent-workers --resume

Windows single-process smoke test (does not replace the two-GPU DDP smoke):

    python -B dino_a0_ft_server.py --device cuda --smoke-test \\
        --batch-size 1 --global-batch-size 1 --num-workers 0 \\
        --no-persistent-workers --no-pin-memory --no-amp
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import dino as t0
import dino_s2_0 as base
import dino_s2_0_server as s2_0_server
import dino_a0_server as a0


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "A_MobileNetV2_RASPP_server"

EXPERIMENT = "A0-FT"
RUN_SUBDIR = "A0-FT"
DEFAULT_SOURCE_EXPERIMENT = "A0"
MODEL_NAME = base.MODEL_NAME
NUM_CLASSES = t0.NUM_CLASSES
IGNORE_INDEX = t0.IGNORE_INDEX
OUTPUT_STRIDE = t0.OUTPUT_STRIDE

ARTIFACT_TYPE_FINETUNE = "a0_ft_mobilenetv2_raspp"
ARTIFACT_FORMAT_VERSION = 1
FINETUNE_MAX_STEPS = 80_000

# Probe artifacts written by the A0/A1/A5 server entries.  A0-FT only needs the
# full ``backbone.* + head.*`` state; the fixed-PCA vs student-adapter provenance
# does not change the fine-tune, so all A-group probe artifact types are accepted.
ACCEPTED_PROBE_ARTIFACT_TYPES = (
    "a0_probe_mobilenetv2_raspp_fixed_pca",
    "a1_probe_mobilenetv2_raspp",
    "a1_probe_mobilenetv2_raspp_fixed_conv",
    "a5_probe_mobilenetv2_raspp",
    "a5_probe_mobilenetv2_raspp_student_adapter",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"{EXPERIMENT}: end-to-end 80k-step pixel-CE fine-tune of the MobileNetV2 "
            f"backbone + R-ASPP head initialized from the best {DEFAULT_SOURCE_EXPERIMENT} "
            "frozen-backbone "
            "probe checkpoint. Same optimizer/schedule/augmentation as S2-0; the "
            "only variable versus S2-0 is the initialization."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=t0.DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-experiment",
        default=DEFAULT_SOURCE_EXPERIMENT,
        help=(
            "A-group experiment whose probe checkpoint seeds the fine-tune "
            f"(default: {DEFAULT_SOURCE_EXPERIMENT})."
        ),
    )
    parser.add_argument(
        "--probe-checkpoint",
        type=Path,
        default=None,
        help=(
            "Best probe checkpoint to fine-tune. Defaults to "
            "<output-dir>/<source-experiment>/seed_<seed>/"
            "<source>_probe_mobilenetv2_raspp_best.pth."
        ),
    )
    parser.add_argument("--finetune-max-steps", type=int, default=FINETUNE_MAX_STEPS)
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
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto or cuda; torchrun assigns one GPU per rank.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    positive = (
        "finetune_max_steps",
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
    if args.crop_height % OUTPUT_STRIDE or args.crop_width % OUTPUT_STRIDE:
        parser.error(f"Crop dimensions must be divisible by {OUTPUT_STRIDE}")
    return args


def ft_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    run_dir = output_dir.resolve() / RUN_SUBDIR / f"seed_{seed}"
    prefix = RUN_SUBDIR.lower().replace("-", "_")
    return {
        "run_dir": run_dir,
        "config": run_dir / "config.json",
        "feature_taps": run_dir / "feature_taps.json",
        "last": run_dir / f"{prefix}_last.pth",
        "history": run_dir / f"{prefix}_history.json",
        "best": run_dir / f"{prefix}_mobilenetv2_raspp_best.pth",
        "dev_metrics": run_dir / f"{prefix}_dev_metrics.json",
        "per_image": run_dir / f"{prefix}_dev_per_image_confusion.jsonl",
        "efficiency": run_dir / "efficiency.json",
    }


def default_probe_checkpoint(output_dir: Path, source_experiment: str, seed: int) -> Path:
    prefix = source_experiment.lower().replace("-", "_")
    return (
        output_dir.resolve()
        / source_experiment
        / f"seed_{seed}"
        / f"{prefix}_probe_mobilenetv2_raspp_best.pth"
    )


def load_probe_as_finetune_start(
    checkpoint_path: Path,
    head_channels: int,
    dropout: float,
    rank: int,
) -> tuple:
    """Load a frozen-backbone probe checkpoint and return an unfrozen model.

    Accepts any A-group probe artifact type so A0-FT, A1-FT and A5-FT can all
    use this function.  The backbone is unfrozen here so the entire model
    participates in the fine-tune optimizer.

    Returns (model, probe_backbone_sha256, source_artifact_type).
    """

    checkpoint_path = Path(checkpoint_path).resolve()
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    if sidecar.is_file():
        t0.verify_checkpoint_sidecar(checkpoint_path)
    payload = t0.safe_torch_load(checkpoint_path, map_location="cpu", weights_only=True)

    artifact_type = payload.get("artifact_type", "")
    if artifact_type not in ACCEPTED_PROBE_ARTIFACT_TYPES:
        # Also accept anything that starts with "a0_probe" / "a1_probe" /
        # "a5_probe" to be resilient to minor naming variations.
        accepted_prefix = any(
            artifact_type.startswith(prefix)
            for prefix in ("a0_probe", "a1_probe", "a5_probe", "a2_probe", "a3_probe",
                           "a4_probe", "a6_probe")
        )
        if not accepted_prefix:
            raise RuntimeError(
                f"{EXPERIMENT} requires an A-group probe checkpoint; got artifact_type={artifact_type!r}. "
                f"Accepted types: {ACCEPTED_PROBE_ARTIFACT_TYPES}"
            )

    model_state = payload.get("model_state_dict")
    if model_state is None:
        raise RuntimeError("Probe checkpoint is missing 'model_state_dict'")

    # The probe model has a frozen backbone; build the same architecture then
    # unfreeze the backbone before loading weights.
    model = base.build_model(head_channels, dropout)
    # Unfreeze backbone immediately so the state dict keys and requires_grad
    # are consistent with a fully trainable model.
    model.backbone.requires_grad_(True)
    model.head.requires_grad_(True)
    model.load_state_dict(model_state, strict=True)

    loaded_hash = t0.state_dict_sha256(model.state_dict())
    expected_hash = payload.get("model_state_sha256")
    if expected_hash and loaded_hash != expected_hash:
        raise RuntimeError(f"{EXPERIMENT} start: probe model state SHA-256 verification failed")

    backbone_hash = t0.state_dict_sha256(
        {k: v for k, v in model.state_dict().items() if k.startswith("backbone.")}
    )
    if rank == 0:
        print(
            f"[OK] {EXPERIMENT} loaded probe checkpoint: artifact_type={artifact_type!r}, "
            f"backbone_sha256={backbone_hash}"
        )
    return model, backbone_hash, artifact_type


def build_finetune_checkpoint(
    model: "base.MobileNetV2RASPPStudent",
    epoch: int,
    optimizer_step: int,
    dev_metrics: Mapping[str, object],
    config: Mapping[str, object],
    hashes: Mapping[str, object],
    dataset_lock: Mapping[str, object],
    shape_audit: Mapping[str, object],
) -> Dict[str, object]:
    """Best-checkpoint payload for the A0-FT run."""

    import copy as _copy

    model_state = t0.cpu_state_dict(model)
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_type": ARTIFACT_TYPE_FINETUNE,
        "experiment": EXPERIMENT,
        "model_name": MODEL_NAME,
        "initialization": f"A-group probe fine-tune ({config.get('source_experiment', '?')})",
        "num_classes": NUM_CLASSES,
        "class_names": list(t0.CITYSCAPES_CLASSES),
        "output_stride": OUTPUT_STRIDE,
        "head_type": "R-ASPP",
        "feature_taps": _copy.deepcopy(base.FEATURE_TAPS),
        "model_state_dict": model_state,
        "model_state_sha256": t0.state_dict_sha256(model_state),
        "best_epoch": epoch,
        "best_optimizer_step": optimizer_step,
        "best_dev_metrics": _copy.deepcopy(dev_metrics),
        "config": _copy.deepcopy(config),
        "hashes": _copy.deepcopy(hashes),
        "dataset_lock": _copy.deepcopy(dataset_lock),
        "shape_audit": _copy.deepcopy(shape_audit),
    }


def load_finetune_model(
    checkpoint_path: Path,
    device: object = "cpu",
) -> tuple:
    """Load a completed A0-FT best checkpoint for re-evaluation or inference."""

    import copy as _copy

    checkpoint_path = Path(checkpoint_path).resolve()
    t0.verify_checkpoint_sidecar(checkpoint_path)
    payload = t0.safe_torch_load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("artifact_type") != ARTIFACT_TYPE_FINETUNE:
        raise RuntimeError(
            f"Not a {EXPERIMENT} artifact: {payload.get('artifact_type')!r}"
        )
    if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise RuntimeError("Unsupported A0-FT artifact format")
    config = payload["config"]
    model = base.build_model(config["head_channels"], config["dropout"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    actual_hash = t0.state_dict_sha256(model.state_dict())
    if actual_hash != payload["model_state_sha256"]:
        raise RuntimeError(f"{EXPERIMENT} model state failed SHA-256 verification")
    model = model.to(torch.device(device)).eval()
    return model, payload


def _smoke_test(
    model: "torch.nn.Module",
    loader: "DataLoader",
    device: "torch.device",
    amp_enabled: bool,
    rank: int,
) -> None:
    """Single-batch forward+backward smoke, checking both backbone and head gradients."""

    model.train()
    images, targets, paths = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    model.zero_grad(set_to_none=True)
    with t0.autocast_context(device, amp_enabled):
        logits = model(images)
    loss = F.cross_entropy(logits.float(), targets, ignore_index=IGNORE_INDEX)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite {EXPERIMENT} smoke loss: {loss.item()}")
    inner = model.module if isinstance(model, DDP) else model
    backbone_grads = sum(
        1 for p in inner.backbone.parameters() if p.grad is not None
    )
    head_grads = sum(
        1 for p in inner.head.parameters() if p.grad is not None
    )
    if backbone_grads == 0:
        raise RuntimeError(f"{EXPERIMENT} smoke: backbone produced no gradients (still frozen?)")
    if head_grads == 0:
        raise RuntimeError(f"{EXPERIMENT} smoke: head produced no gradients")
    if rank == 0:
        print(
            f"[OK] {EXPERIMENT} smoke test: sample={paths[0]}, "
            f"logits={tuple(logits.shape)}, loss={loss.item():.6f}, "
            f"backbone_grad_tensors={backbone_grads}, head_grad_tensors={head_grads}"
        )


def _poly_lr_factor(max_steps: int, poly_power: float, min_lr_ratio: float):
    def lr_factor(step: int) -> float:
        progress = min(step, max_steps) / max(max_steps, 1)
        return max((1.0 - progress) ** poly_power, min_lr_ratio)
    return lr_factor


def run_training(args: argparse.Namespace) -> None:
    rank, local_rank, world_size, device = s2_0_server.setup_distributed(args)
    main_process = rank == 0
    train_loader: Optional[DataLoader] = None
    dev_loader: Optional[DataLoader] = None
    model: Optional["torch.nn.Module"] = None
    optimizer: Optional["torch.optim.Optimizer"] = None
    scheduler = None
    amp_scaler = None
    selected_model = None
    successful_exit = False

    try:
        import copy as _copy

        accumulation_steps = s2_0_server.effective_accumulation_steps(args, world_size)
        t0.set_global_seed(args.seed + rank, args.deterministic)
        dataset_root = args.dataset_root.resolve()
        dataset_lock, entries_by_split = t0.validate_dataset_lock(dataset_root)

        # ------------------------------------------------------------------
        # Resolve probe checkpoint path
        # ------------------------------------------------------------------
        probe_checkpoint_path = args.probe_checkpoint
        if probe_checkpoint_path is None:
            probe_checkpoint_path = default_probe_checkpoint(
                args.output_dir, args.source_experiment, args.seed
            )
        probe_checkpoint_path = Path(probe_checkpoint_path).resolve()
        if not probe_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Probe checkpoint not found: {probe_checkpoint_path}. "
                f"Run the {args.source_experiment} probe stage first, or pass "
                "--probe-checkpoint explicitly."
            )

        # ------------------------------------------------------------------
        # Build model from probe checkpoint (backbone unfrozen)
        # ------------------------------------------------------------------
        model, initial_backbone_hash, source_artifact_type = load_probe_as_finetune_start(
            probe_checkpoint_path,
            head_channels=args.head_channels,
            dropout=args.dropout,
            rank=rank,
        )
        model = model.to(device)
        initial_model_hash = t0.state_dict_sha256(model.state_dict())

        shape_audit = base.audit_model_shapes(
            model, device, args.crop_height, args.crop_width,
            bool(args.amp and device.type == "cuda"),
        )
        amp_enabled = bool(args.amp and device.type == "cuda")

        if world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=True,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

        # ------------------------------------------------------------------
        # DataLoaders
        # ------------------------------------------------------------------
        train_loader, train_sampler, train_generator = s2_0_server.build_train_loader(
            args, dataset_root, entries_by_split, device, rank, world_size
        )
        dev_loader = (
            s2_0_server.build_dev_loader(args, dataset_root, entries_by_split, device)
            if main_process else None
        )

        steps_per_full_epoch = math.ceil(len(train_loader) / accumulation_steps)
        if main_process:
            print(
                f"[INFO] {EXPERIMENT} server DDP: world_size={world_size}, device={device}, "
                f"AMP={amp_enabled}, workers/rank={args.num_workers}, "
                f"context={args.multiprocessing_context}, pin_memory={args.pin_memory}"
            )
            print(
                f"[INFO] global batch={args.batch_size * accumulation_steps * world_size}; "
                f"steps={args.finetune_max_steps:,} (~{steps_per_full_epoch} steps/epoch); "
                f"source={args.source_experiment!r} probe backbone"
            )

        # ------------------------------------------------------------------
        # Smoke test
        # ------------------------------------------------------------------
        if args.smoke_test:
            _smoke_test(model, train_loader, device, amp_enabled, rank)
            successful_exit = True
            return

        # ------------------------------------------------------------------
        # Paths and existing-artifact guard
        # ------------------------------------------------------------------
        paths = ft_paths(args.output_dir, args.seed)
        paths["run_dir"].mkdir(parents=True, exist_ok=True)
        t0.write_json_atomic(paths["feature_taps"], a0.build_feature_taps_record())
        artifact_files = [paths["best"], paths["last"], paths["history"], paths["dev_metrics"]]
        if not args.resume and any(p.exists() for p in artifact_files):
            raise FileExistsError(
                f"{EXPERIMENT} artifacts already exist in {paths['run_dir']}; use --resume"
            )

        # ------------------------------------------------------------------
        # Optimizer and scheduler — identical to S2-0
        # ------------------------------------------------------------------
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=_poly_lr_factor(
                args.finetune_max_steps, args.poly_power, args.min_lr_ratio
            ),
        )
        amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        # ------------------------------------------------------------------
        # Config record
        # ------------------------------------------------------------------
        config: Dict[str, object] = {
            "experiment": EXPERIMENT,
            "server_entry_point": str(Path(__file__).resolve()),
            "source_experiment": args.source_experiment,
            "source_probe_checkpoint": str(probe_checkpoint_path),
            "source_probe_artifact_type": source_artifact_type,
            "seed": args.seed,
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": args.batch_size * accumulation_steps * world_size,
            "accumulation_steps_per_gpu": accumulation_steps,
            "finetune_max_optimizer_steps": args.finetune_max_steps,
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
            "num_classes": NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
            "output_stride": OUTPUT_STRIDE,
            "initialization": f"A-group probe fine-tune (source={args.source_experiment})",
            "backbone_frozen": False,
            "loss": "pixel_cross_entropy_only",
            "knowledge_distillation": False,
            "test_local_evaluated": False,
            "distributed_backend": "nccl" if world_size > 1 else None,
        }
        hashes: Dict[str, object] = {
            "training_script_sha256": t0.sha256_file(Path(__file__).resolve()),
            "initial_model_state_sha256": initial_model_hash,
            "initial_backbone_state_sha256": initial_backbone_hash,
            "source_probe_checkpoint_sha256": t0.verify_checkpoint_sidecar(probe_checkpoint_path),
        }
        t0.write_json_atomic(paths["config"], config)

        # ------------------------------------------------------------------
        # Resume
        # ------------------------------------------------------------------
        history: List[Dict[str, object]] = []
        best_key: Optional[Tuple[float, float, float, float]] = None
        best_epoch: Optional[int] = None
        best_optimizer_step: Optional[int] = None
        best_dev_metrics: Optional[Dict[str, object]] = None
        epoch = 0
        cumulative_optimizer_steps = 0

        if args.resume and paths["last"].is_file():
            resume_payload = t0.safe_torch_load(
                paths["last"], map_location="cpu", weights_only=False
            )
            if resume_payload.get("config") != config:
                raise RuntimeError(
                    f"{EXPERIMENT} resume config differs from current arguments"
                )
            model_to_load = model.module if isinstance(model, DDP) else model
            model_to_load.load_state_dict(
                resume_payload["model_state_dict"], strict=True
            )
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
            amp_scaler.load_state_dict(resume_payload["scaler_state_dict"])
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
                    f"[OK] Resuming {EXPERIMENT} after epoch {epoch}, "
                    f"step {cumulative_optimizer_steps:,}"
                )

        # ------------------------------------------------------------------
        # Training loop — same structure as S2-0 server
        # ------------------------------------------------------------------
        next_eval_step = (
            ((cumulative_optimizer_steps // args.eval_every_steps) + 1)
            * args.eval_every_steps
            if args.eval_every_steps > 0
            else math.inf
        )
        training_started = time.time()

        while cumulative_optimizer_steps < args.finetune_max_steps:
            epoch += 1
            remaining_steps = args.finetune_max_steps - cumulative_optimizer_steps
            train_metrics, optimizer_steps = s2_0_server.train_one_epoch_server(
                model=model,
                loader=train_loader,
                sampler=train_sampler,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=amp_scaler,
                device=device,
                amp_enabled=amp_enabled,
                accumulation_steps=accumulation_steps,
                epoch=epoch,
                remaining_optimizer_steps=remaining_steps,
                rank=rank,
                world_size=world_size,
            )
            cumulative_optimizer_steps += optimizer_steps
            # Evaluate at the first epoch boundary at/after each eval interval.
            should_evaluate = (
                cumulative_optimizer_steps >= next_eval_step
            ) or cumulative_optimizer_steps == args.finetune_max_steps
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
                        split_name=f"dev_local server ({EXPERIMENT})",
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
                        best_dev_metrics = _copy.deepcopy(dev_metrics)
                        best_payload = build_finetune_checkpoint(
                            model=model.module if isinstance(model, DDP) else model,
                            epoch=epoch,
                            optimizer_step=cumulative_optimizer_steps,
                            dev_metrics=dev_metrics,
                            config=config,
                            hashes=hashes,
                            dataset_lock=dataset_lock,
                            shape_audit=shape_audit,
                        )
                        checkpoint_hash = t0.write_checkpoint_with_sidecar(
                            best_payload, paths["best"]
                        )
                        print(
                            f"[OK] {EXPERIMENT} best updated: step={cumulative_optimizer_steps:,}, "
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
                t0.write_json_atomic(paths["history"], history)
                last_payload = {
                    "format_version": ARTIFACT_FORMAT_VERSION,
                    "artifact_type": ARTIFACT_TYPE_FINETUNE,
                    "experiment": EXPERIMENT,
                    "stage": "finetune",
                    "epoch": epoch,
                    "optimizer_steps": cumulative_optimizer_steps,
                    "model_state_dict": t0.cpu_state_dict(
                        model.module if isinstance(model, DDP) else model
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": amp_scaler.state_dict(),
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
                t0.torch_save_atomic(last_payload, paths["last"])
                message = (
                    f"Epoch {epoch}: step={cumulative_optimizer_steps:,}/"
                    f"{args.finetune_max_steps:,}, "
                    f"train_mIoU={train_metrics['mIoU']:.4f}, "
                    f"train_loss={train_metrics['loss']:.4f}"
                )
                if dev_metrics is not None:
                    message += f", dev_mIoU={dev_metrics['mIoU']:.4f}"
                print(message)
            s2_0_server.barrier(world_size)

        # ------------------------------------------------------------------
        # Final evaluation: reload best checkpoint, re-run dev, write metrics
        # ------------------------------------------------------------------
        s2_0_server.barrier(world_size)
        if main_process:
            if best_epoch is None or best_optimizer_step is None or best_dev_metrics is None:
                raise RuntimeError(f"{EXPERIMENT} ended without a selected dev checkpoint")
            selected_model, selected_payload = load_finetune_model(
                paths["best"], device=device
            )
            selected_dev_metrics, per_image_rows = t0.evaluate(
                model=selected_model,
                loader=dev_loader,
                device=device,
                amp_enabled=amp_enabled,
                split_name=f"selected dev_local server ({EXPERIMENT})",
                boundary_tolerance=args.boundary_tolerance,
                collect_per_image=True,
            )
            if not t0.metrics_reproduce(selected_dev_metrics, best_dev_metrics):
                raise RuntimeError(
                    f"Reloaded {EXPERIMENT} checkpoint did not reproduce best dev metrics: "
                    f"saved={best_dev_metrics['mIoU']}, "
                    f"reloaded={selected_dev_metrics['mIoU']}"
                )
            t0.write_jsonl_atomic(paths["per_image"], per_image_rows)
            checkpoint_hash = t0.verify_checkpoint_sidecar(paths["best"])
            final_backbone_hash = t0.state_dict_sha256(
                {
                    k: v
                    for k, v in selected_model.state_dict().items()
                    if k.startswith("backbone.")
                }
            )
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
            t0.write_json_atomic(paths["efficiency"], efficiency)

            results = {
                "experiment": EXPERIMENT,
                "protocol": (
                    f"MobileNetV2+R-ASPP initialized from the best "
                    f"{args.source_experiment} frozen-backbone probe checkpoint, "
                    f"then end-to-end 80k-step pixel CE fine-tuned with the same "
                    f"optimizer/augmentation as S2-0. No KD loss. Best checkpoint "
                    f"selected by dev_local mIoU; test_local not evaluated."
                ),
                "best_epoch": best_epoch,
                "best_optimizer_step": best_optimizer_step,
                "best_dev_metrics": selected_dev_metrics,
                "config": config,
                "shape_audit": shape_audit,
                "dataset_lock": dataset_lock,
                "model": {
                    "model_name": MODEL_NAME,
                    "initialization": f"A-group probe fine-tune ({args.source_experiment})",
                    "head": "R-ASPP",
                    "backbone_frozen": False,
                    "feature_taps": base.FEATURE_TAPS,
                },
                "efficiency": efficiency,
                "hashes": {
                    **hashes,
                    "selected_model_state_sha256": selected_payload["model_state_sha256"],
                    "checkpoint_sha256": checkpoint_hash,
                    "final_backbone_state_sha256": final_backbone_hash,
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
                "artifacts": {key: str(value) for key, value in paths.items()},
            }
            t0.write_json_atomic(paths["dev_metrics"], results)
            print(
                f"[DONE] {EXPERIMENT}: GPUs={world_size}, steps={cumulative_optimizer_steps:,}, "
                f"best dev mIoU={selected_dev_metrics['mIoU']:.6f}, "
                f"source={args.source_experiment!r}"
            )
        s2_0_server.barrier(world_size)
        successful_exit = True

    finally:
        # Ordered teardown (server_training_issues_and_solutions.md):
        # stop workers -> CUDA synchronize -> release DDP/optimizer ->
        # barrier -> destroy process group.
        s2_0_server._shutdown_loader(train_loader)
        s2_0_server._shutdown_loader(dev_loader)
        s2_0_server._synchronize_cuda(device)
        if successful_exit and world_size > 1 and dist.is_initialized():
            s2_0_server.barrier(world_size)
        selected_model = None
        amp_scaler = None
        scheduler = None
        optimizer = None
        model = None
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
