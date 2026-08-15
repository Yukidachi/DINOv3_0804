"""Profile the S2-0 training pipeline and locate its throughput bottleneck.

This is intentionally separate from ``dino_s2_0.py``.  It never writes a
checkpoint and it does not change the training program.  The report contains:

* image decode and augmentation timings;
* DataLoader startup/steady-state timings for a worker-count sweep;
* a synthetic GPU-only train benchmark (no disk or CPU preprocessing);
* an end-to-end pass matching the S2-0 train loop;
* synchronized per-stage timings for data transfer, forward, loss, backward,
  optimizer, and metrics;
* optional PyTorch profiler trace and an ``nvidia-smi`` sample log.

Run this script once on each machine with the same arguments, then compare the
generated JSON files.  The human-readable diagnosis is also printed to stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# Match dino.py: this must be present before CUDA/CUBLAS work is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

# When invoked as ``python scripts/profile_dino_s2_0.py``, Python initially
# puts ``scripts`` on sys.path rather than the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dino as common
import dino_s2_0 as s2


SCRIPT_DIR = REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find S2-0 data, host, CUDA, or metric-processing bottlenecks."
    )
    parser.add_argument("--dataset-root", type=Path, default=common.DEFAULT_DATASET_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--worker-sweep", default="0,2,4,8")
    parser.add_argument("--crop-height", type=int, default=512)
    parser.add_argument("--crop-width", type=int, default=1024)
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--head-channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the same CUDA autocast setting as the training program.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the same deterministic kernel setting as the training program.",
    )
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--multiprocessing-context",
        choices=("auto", "fork", "spawn", "forkserver"),
        default="auto",
        help="DataLoader worker start method. auto keeps the platform default.",
    )
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--profile-batches", type=int, default=20)
    parser.add_argument("--data-batches", type=int, default=40)
    parser.add_argument("--io-samples", type=int, default=20)
    parser.add_argument("--trace", type=Path, default=None, help="Export a Chrome trace to this path.")
    parser.add_argument("--trace-batches", type=int, default=5)
    parser.add_argument("--no-worker-sweep", action="store_true")
    parser.add_argument("--no-io-profile", action="store_true")
    parser.add_argument("--no-synthetic", action="store_true")
    parser.add_argument("--no-pipeline", action="store_true")
    parser.add_argument("--no-gpu-monitor", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    for name in (
        "batch_size",
        "accumulation_steps",
        "num_workers",
        "crop_height",
        "crop_width",
        "warmup_batches",
        "profile_batches",
        "data_batches",
        "io_samples",
        "trace_batches",
    ):
        if getattr(args, name) < 0 or (name in {"batch_size", "accumulation_steps", "crop_height", "crop_width"} and getattr(args, name) == 0):
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if not 0 < args.scale_min <= args.scale_max:
        parser.error("Require 0 < --scale-min <= --scale-max")
    if args.worker_sweep.strip():
        try:
            values = [int(value.strip()) for value in args.worker_sweep.split(",") if value.strip()]
        except ValueError as error:
            parser.error(f"Invalid --worker-sweep: {error}")
        if any(value < 0 for value in values):
            parser.error("Worker counts cannot be negative")
    return args


class Samples:
    """Small numeric accumulator that keeps raw values for p90 reporting."""

    def __init__(self) -> None:
        self.values: List[float] = []

    def add(self, value: float) -> None:
        self.values.append(float(value))

    def summary(self) -> Dict[str, Optional[float]]:
        if not self.values:
            return {"n": 0, "mean_ms": None, "median_ms": None, "p90_ms": None, "min_ms": None, "max_ms": None}
        values = np.asarray(self.values, dtype=np.float64)
        return {
            "n": len(self.values),
            "mean_ms": float(values.mean()),
            "median_ms": float(np.median(values)),
            "p90_ms": float(np.percentile(values, 90)),
            "min_ms": float(values.min()),
            "max_ms": float(values.max()),
        }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn: Callable[[], Any], device: torch.device) -> Tuple[Any, float]:
    """Time one stage with synchronization on both sides.

    This is used only for the diagnostic stage pass.  The report separately
    includes a no-extra-sync end-to-end throughput pass.
    """

    _sync(device)
    started = time.perf_counter()
    value = fn()
    _sync(device)
    return value, (time.perf_counter() - started) * 1000.0


def _safe_shutdown(iterator: Any) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except RuntimeError as error:
            # Some PyTorch 2.x + Linux spawn/persistent-worker combinations
            # abort while joining an already-terminated worker.  The measured
            # batches are still valid; retain the report and surface a warning
            # instead of masking the profiling result.
            print(
                f"[WARN] DataLoader worker cleanup failed after profiling: {error}",
                file=sys.stderr,
            )


def _parse_workers(value: str, current: int) -> List[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if current not in values:
        values.insert(0, current)
    return list(dict.fromkeys(values))


def build_train_loader(
    args: argparse.Namespace,
    dataset_root: Path,
    entries_by_split: Mapping[str, Sequence[Tuple[str, str]]],
    device: torch.device,
    workers: Optional[int] = None,
) -> DataLoader:
    dataset = common.CityscapesManifestDataset(
        dataset_root=dataset_root,
        entries=entries_by_split["train_local"],
        transform=common.CityscapesTrainTransform(
            crop_size=(args.crop_height, args.crop_width),
            scale_range=(args.scale_min, args.scale_max),
        ),
        reject_all_ignore=True,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    worker_count = args.num_workers if workers is None else workers
    pin_memory = device.type == "cuda" if args.pin_memory is None else bool(args.pin_memory)
    loader_kwargs: Dict[str, Any] = {}
    if worker_count > 0 and args.multiprocessing_context != "auto":
        loader_kwargs["multiprocessing_context"] = args.multiprocessing_context
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        num_workers=worker_count,
        pin_memory=pin_memory,
        worker_init_fn=common.seed_worker,
        persistent_workers=bool(args.persistent_workers and worker_count > 0),
        **loader_kwargs,
    )


def _nvidia_device_selector(device: torch.device) -> Optional[str]:
    if device.type != "cuda":
        return None
    logical_index = device.index
    if logical_index is None:
        logical_index = torch.cuda.current_device()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if logical_index < len(tokens):
            return tokens[logical_index]
    return str(logical_index)


def nvidia_smi_snapshot(selector: Optional[str] = None) -> Dict[str, Any]:
    fields = [
        "index",
        "uuid",
        "name",
        "driver_version",
        "compute_mode",
        "pstate",
        "temperature.gpu",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "power.draw",
        "power.limit",
        "clocks.current.sm",
        "clocks.current.memory",
        "pcie.link.gen.current",
        "pcie.link.width.current",
    ]
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    if selector is not None:
        command.insert(1, f"--id={selector}")
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as error:
        return {"available": False, "error": str(error)}
    if completed.returncode != 0:
        return {"available": False, "error": completed.stderr.strip() or f"exit={completed.returncode}"}
    rows = []
    for row in csv.reader(completed.stdout.splitlines()):
        if not row:
            continue
        rows.append({key: value.strip() for key, value in zip(fields, row)})
    return {"available": True, "fields": fields, "gpus": rows}


def nvidia_compute_processes() -> Dict[str, Any]:
    fields = ["pid", "process_name", "used_gpu_memory", "gpu_uuid"]
    command = [
        "nvidia-smi",
        f"--query-compute-apps={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as error:
        return {"available": False, "error": str(error)}
    if completed.returncode != 0:
        return {"available": False, "error": completed.stderr.strip() or f"exit={completed.returncode}"}
    rows = []
    for row in csv.reader(completed.stdout.splitlines()):
        if row:
            rows.append({key: value.strip() for key, value in zip(fields, row)})
    return {"available": True, "processes": rows}


class GpuSampler:
    def __init__(self, enabled: bool, selector: Optional[str]) -> None:
        self.enabled = enabled
        self.selector = selector
        self.rows: List[Dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.enabled:
            return

        def sample_loop() -> None:
            while not self.stop_event.wait(0.5):
                snapshot = nvidia_smi_snapshot(self.selector)
                if snapshot.get("available"):
                    self.rows.append({"time": time.time(), "gpus": snapshot.get("gpus", [])})

        self.thread = threading.Thread(target=sample_loop, name="s2-0-gpu-monitor", daemon=True)
        self.thread.start()

    def stop(self) -> List[Dict[str, Any]]:
        if self.thread is None:
            return []
        self.stop_event.set()
        self.thread.join(timeout=6.0)
        return self.rows


def summarize_gpu_samples(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    columns = (
        "utilization.gpu",
        "utilization.memory",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.current.sm",
        "clocks.current.memory",
    )
    values: Dict[str, List[float]] = {column: [] for column in columns}
    pstates: Dict[str, int] = {}
    for sample in samples:
        for gpu in sample.get("gpus", []):
            pstate = str(gpu.get("pstate", "unknown"))
            pstates[pstate] = pstates.get(pstate, 0) + 1
            for column in columns:
                raw = str(gpu.get(column, "")).replace("%", "").strip()
                try:
                    values[column].append(float(raw))
                except ValueError:
                    pass
    summary: Dict[str, Any] = {"sample_count": len(samples), "pstates": pstates}
    for column, column_values in values.items():
        summary[column] = {
            "mean": statistics.mean(column_values) if column_values else None,
            "min": min(column_values) if column_values else None,
            "max": max(column_values) if column_values else None,
        }
    return summary


def collect_environment(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": str(torch.__version__),
        "torchvision": str(s2.torchvision.__version__),
        "numpy": str(np.__version__),
        "pillow": str(__import__("PIL").__version__),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["gpu"] = {
            "name": props.name,
            "capability": [props.major, props.minor],
            "total_memory_mib": props.total_memory / (1024.0 * 1024.0),
            "multi_processor_count": props.multi_processor_count,
            "logical_index": device.index if device.index is not None else torch.cuda.current_device(),
            "nvidia_smi_selector": _nvidia_device_selector(device),
        }
    return info


def profile_io(
    args: argparse.Namespace,
    dataset_root: Path,
    entries: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    image_decode = Samples()
    label_decode = Samples()
    transform = Samples()
    pipeline = Samples()
    train_transform = common.CityscapesTrainTransform(
        crop_size=(args.crop_height, args.crop_width),
        scale_range=(args.scale_min, args.scale_max),
    )
    for image_rel, label_rel in list(entries)[: args.io_samples]:
        image_path = dataset_root / image_rel
        label_path = dataset_root / label_rel
        start = time.perf_counter()
        with Image.open(image_path) as image_obj:
            image = image_obj.convert("RGB")
        image_decode.add((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        with Image.open(label_path) as label_obj:
            label = label_obj.convert("L")
        label_decode.add((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        train_transform(image, label)
        transform.add((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        with Image.open(image_path) as image_obj:
            image = image_obj.convert("RGB")
        with Image.open(label_path) as label_obj:
            label = label_obj.convert("L")
        train_transform(image, label)
        pipeline.add((time.perf_counter() - start) * 1000.0)
    return {
        "samples": len(pipeline.values),
        "image_decode": image_decode.summary(),
        "label_decode": label_decode.summary(),
        "augmentation_and_tensor": transform.summary(),
        "decode_plus_augmentation": pipeline.summary(),
    }


def profile_data_loader(
    loader: DataLoader,
    batches: int,
) -> Dict[str, Any]:
    iterator = iter(loader)
    first = Samples()
    steady = Samples()
    try:
        for index in range(batches):
            started = time.perf_counter()
            next(iterator)
            elapsed = (time.perf_counter() - started) * 1000.0
            (first if index == 0 else steady).add(elapsed)
    finally:
        _safe_shutdown(iterator)
    measured = steady.values or first.values
    seconds = sum(measured) / 1000.0
    return {
        "batches": len(measured),
        "first_batch": first.summary(),
        "steady_next_batch": steady.summary(),
        "steady_batches_per_second": (len(measured) / seconds if seconds > 0 else None),
        "batch_size": loader.batch_size,
        "samples_per_second": (len(measured) * int(loader.batch_size or 1) / seconds if seconds > 0 else None),
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
        "persistent_workers": loader.persistent_workers,
    }


def _make_train_objects(args: argparse.Namespace, device: torch.device):
    model = s2.build_model(args.head_channels, args.dropout).to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    return model, optimizer, scaler, scheduler


def _run_train_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    images: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    group_end: bool,
    group_size: int,
) -> Dict[str, Any]:
    with common.autocast_context(device, amp_enabled):
        logits = model(images)
    logits_float = logits.float()
    batch_loss_sum = F.cross_entropy(
        logits_float, targets, ignore_index=common.IGNORE_INDEX, reduction="sum"
    )
    batch_valid = int((targets != common.IGNORE_INDEX).sum().item())
    if batch_valid == 0:
        raise RuntimeError("Profile batch contains no valid pixels")
    batch_loss = batch_loss_sum / batch_valid
    scaler.scale(batch_loss / group_size).backward()
    if group_end:
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    return {"logits_float": logits_float, "batch_loss_sum": batch_loss_sum, "batch_valid": batch_valid}


def _metric_update(
    state: Dict[str, Any],
    values: Mapping[str, Any],
) -> None:
    predictions = values["logits_float"].detach().argmax(dim=1)
    state["confusion"] += common.confusion_counts(predictions, values["targets"])
    state["loss_sum"] += float(values["batch_loss_sum"].detach().item())
    state["valid_pixels"] += int(values["batch_valid"])
    common.metrics_from_confusion(state["confusion"], state["loss_sum"], state["valid_pixels"])


def _profile_realistic_pass(
    args: argparse.Namespace,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, Any]:
    """Run the current loop shape without diagnostic synchronization barriers."""

    model, optimizer, scaler, scheduler = _make_train_objects(args, device)
    iterator = iter(loader)
    state: Dict[str, Any] = {
        "confusion": torch.zeros(common.NUM_CLASSES, common.NUM_CLASSES, dtype=torch.int64),
        "loss_sum": 0.0,
        "valid_pixels": 0,
    }
    wait = Samples()
    total = Samples()
    try:
        for index in range(args.warmup_batches):
            try:
                images, targets, _ = next(iterator)
            except StopIteration:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            warmup_group_position = index % args.accumulation_steps
            warmup_group_size = min(
                args.accumulation_steps,
                args.warmup_batches - (index // args.accumulation_steps) * args.accumulation_steps,
            )
            values = _run_train_batch(
                model, optimizer, scaler, scheduler, images, targets, device, amp_enabled,
                group_end=warmup_group_position + 1 == warmup_group_size,
                group_size=warmup_group_size,
            )
            values["targets"] = targets
            _metric_update(state, values)
        _sync(device)
        state = {
            "confusion": torch.zeros(common.NUM_CLASSES, common.NUM_CLASSES, dtype=torch.int64),
            "loss_sum": 0.0,
            "valid_pixels": 0,
        }
        for index in range(args.profile_batches):
            started = time.perf_counter()
            wait_started = time.perf_counter()
            try:
                images, targets, _ = next(iterator)
            except StopIteration:
                break
            wait.add((time.perf_counter() - wait_started) * 1000.0)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            group_position = index % args.accumulation_steps
            group_size = args.accumulation_steps
            if group_position == 0:
                group_size = min(args.accumulation_steps, args.profile_batches - index)
            group_end = group_position + 1 == group_size
            values = _run_train_batch(
                model, optimizer, scaler, scheduler, images, targets, device, amp_enabled,
                group_end=group_end, group_size=group_size,
            )
            values["targets"] = targets
            _metric_update(state, values)
            _sync(device)
            total.add((time.perf_counter() - started) * 1000.0)
    finally:
        _safe_shutdown(iterator)
        del model, optimizer, scaler, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()
    total_summary = total.summary()
    return {
        "batches": len(total.values),
        "data_wait": wait.summary(),
        "iteration": total_summary,
        "batches_per_second": (1000.0 / total_summary["mean_ms"] if total_summary["mean_ms"] else None),
        "note": "Matches the current S2-0 loop and its metric calls; only the final timing synchronization is added.",
    }


def profile_pipeline(
    args: argparse.Namespace,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, Any]:
    model, optimizer, scaler, scheduler = _make_train_objects(args, device)
    model.train()
    iterator = iter(loader)
    for index in range(args.warmup_batches):
        try:
            images, targets, _ = next(iterator)
        except StopIteration:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        warmup_group_position = index % args.accumulation_steps
        warmup_group_size = min(
            args.accumulation_steps,
            args.warmup_batches - (index // args.accumulation_steps) * args.accumulation_steps,
        )
        _run_train_batch(
            model, optimizer, scaler, scheduler, images, targets, device, amp_enabled,
            group_end=warmup_group_position + 1 == warmup_group_size,
            group_size=warmup_group_size,
        )
    _sync(device)

    stage_names = ("data_wait", "host_to_device", "forward", "loss_and_valid", "backward", "optimizer", "metrics")
    stages = {name: Samples() for name in stage_names}
    e2e = Samples()
    state: Dict[str, Any] = {
        "confusion": torch.zeros(common.NUM_CLASSES, common.NUM_CLASSES, dtype=torch.int64),
        "loss_sum": 0.0,
        "valid_pixels": 0,
    }
    measured = 0
    try:
        for batch_index in range(args.profile_batches):
            iteration_started = time.perf_counter()
            wait_started = time.perf_counter()
            try:
                images, targets, _ = next(iterator)
            except StopIteration:
                break
            stages["data_wait"].add((time.perf_counter() - wait_started) * 1000.0)
            group_position = batch_index % args.accumulation_steps
            if group_position == 0:
                group_size = min(args.accumulation_steps, args.profile_batches - batch_index)
            else:
                group_size = args.accumulation_steps
            group_end = group_position + 1 == group_size

            (images, targets), elapsed = _timed(
                lambda: (images.to(device, non_blocking=True), targets.to(device, non_blocking=True)), device
            )
            stages["host_to_device"].add(elapsed)
            logits, elapsed = _timed(lambda: model(images), device)
            stages["forward"].add(elapsed)
            logits_float: torch.Tensor
            batch_loss_sum: torch.Tensor
            batch_valid: int
            def loss_stage() -> Tuple[torch.Tensor, int, torch.Tensor]:
                nonlocal logits_float
                logits_float = logits.float()
                loss_value = F.cross_entropy(
                    logits_float, targets, ignore_index=common.IGNORE_INDEX, reduction="sum"
                )
                valid = int((targets != common.IGNORE_INDEX).sum().item())
                return loss_value, valid, logits_float
            (batch_loss_sum, batch_valid, logits_float), elapsed = _timed(loss_stage, device)
            stages["loss_and_valid"].add(elapsed)
            if batch_valid == 0:
                raise RuntimeError("Profile batch contains no valid pixels")
            batch_loss = batch_loss_sum / batch_valid
            _, elapsed = _timed(
                lambda: scaler.scale(batch_loss / group_size).backward(), device
            )
            stages["backward"].add(elapsed)
            if group_end:
                _, elapsed = _timed(
                    lambda: (
                        scaler.unscale_(optimizer),
                        scaler.step(optimizer),
                        scaler.update(),
                        optimizer.zero_grad(set_to_none=True),
                        scheduler.step(),
                    ), device
                )
                stages["optimizer"].add(elapsed)
            else:
                stages["optimizer"].add(0.0)
            values = {"logits_float": logits_float, "batch_loss_sum": batch_loss_sum, "batch_valid": batch_valid, "targets": targets}
            _, elapsed = _timed(lambda: _metric_update(state, values), device)
            stages["metrics"].add(elapsed)
            e2e.add((time.perf_counter() - iteration_started) * 1000.0)
            measured += 1
    finally:
        _safe_shutdown(iterator)
        del model, optimizer, scaler, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stage_summary = {name: sample.summary() for name, sample in stages.items()}
    stage_total = sum((item["mean_ms"] or 0.0) for item in stage_summary.values())
    for item in stage_summary.values():
        item["share_of_synchronized_stage_sum"] = (item["mean_ms"] or 0.0) / stage_total if stage_total else None
    realistic = _profile_realistic_pass(args, loader, device, amp_enabled)
    return {
        "batches": measured,
        "synchronized_stage_timings": stage_summary,
        "synchronized_stage_sum_mean_ms": stage_total,
        "synchronized_pass_iteration": e2e.summary(),
        "realistic_end_to_end": realistic,
        "end_to_end_batches_per_second": realistic["batches_per_second"],
        "note": "Synchronized stage timings are diagnostic only; realistic_end_to_end matches the current loop without per-stage synchronization.",
    }


def benchmark_synthetic(
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, Any]:
    model, optimizer, scaler, scheduler = _make_train_objects(args, device)
    images = torch.randn(args.batch_size, 3, args.crop_height, args.crop_width, device=device)
    targets = torch.zeros(args.batch_size, args.crop_height, args.crop_width, dtype=torch.long, device=device)
    timings = Samples()
    total_batches = args.warmup_batches + args.profile_batches
    try:
        for index in range(total_batches):
            _sync(device)
            started = time.perf_counter()
            with common.autocast_context(device, amp_enabled):
                logits = model(images)
            loss = F.cross_entropy(logits.float(), targets, reduction="mean")
            scaler.scale(loss / args.accumulation_steps).backward()
            if (index + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            _sync(device)
            if index >= args.warmup_batches:
                timings.add((time.perf_counter() - started) * 1000.0)
    finally:
        del model, optimizer, scaler, scheduler, images, targets
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = timings.summary()
    return {
        "batches": len(timings.values),
        "batch_timing": summary,
        "batches_per_second": (1000.0 / summary["mean_ms"] if summary["mean_ms"] else None),
        "note": "Synthetic train data removes image decode, augmentation, DataLoader, and H2D transfer.",
    }


def run_trace(
    args: argparse.Namespace,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    trace_path: Path,
) -> Dict[str, Any]:
    try:
        from torch.profiler import ProfilerActivity, profile, record_function
    except ImportError as error:
        return {"available": False, "error": str(error)}
    model, optimizer, scaler, scheduler = _make_train_objects(args, device)
    iterator = iter(loader)
    trace_path = trace_path.resolve()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    try:
        with profile(activities=activities, record_shapes=False, profile_memory=True) as prof:
            for index in range(args.trace_batches):
                with record_function("s2_0.data_wait"):
                    try:
                        images, targets, _ = next(iterator)
                    except StopIteration:
                        break
                with record_function("s2_0.host_to_device"):
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                group_end = (index + 1) % args.accumulation_steps == 0
                with record_function("s2_0.forward_loss_backward"):
                    values = _run_train_batch(
                        model, optimizer, scaler, scheduler, images, targets, device, amp_enabled,
                        group_end=group_end, group_size=args.accumulation_steps,
                    )
                with record_function("s2_0.metrics"):
                    state = {"confusion": torch.zeros(common.NUM_CLASSES, common.NUM_CLASSES, dtype=torch.int64), "loss_sum": 0.0, "valid_pixels": 0}
                    values["targets"] = targets
                    _metric_update(state, values)
                prof.step()
        prof.export_chrome_trace(str(trace_path))
        sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        table = prof.key_averages().table(sort_by=sort_key, row_limit=20)
        text_path = trace_path.with_suffix(trace_path.suffix + ".txt")
        text_path.write_text(table, encoding="utf-8")
        return {"available": True, "trace": str(trace_path), "operator_table": str(text_path), "top_operators": table}
    except Exception as error:  # profiler support differs between PyTorch builds
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    finally:
        _safe_shutdown(iterator)
        del model, optimizer, scaler, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()


def diagnose(result: Mapping[str, Any], args: argparse.Namespace) -> List[str]:
    notes: List[str] = []
    pipeline = result.get("pipeline") or {}
    e2e_ips = pipeline.get("end_to_end_batches_per_second")
    stages = (pipeline.get("synchronized_stage_timings") or {})
    stage_values = {name: float(item.get("mean_ms") or 0.0) for name, item in stages.items()}
    total = sum(stage_values.values())
    if total:
        dominant = max(stage_values, key=stage_values.get)
        notes.append(f"Largest synchronized stage: {dominant} ({stage_values[dominant]:.2f} ms, {stage_values[dominant] / total:.1%} of stage sum).")
    data_sweep = result.get("data_loader_sweep") or []
    if data_sweep:
        best = max((row for row in data_sweep if row.get("samples_per_second") is not None), key=lambda row: row["samples_per_second"], default=None)
        current = next((row for row in data_sweep if row.get("num_workers") == args.num_workers), None)
        if best and current and current.get("samples_per_second"):
            ratio = best["samples_per_second"] / current["samples_per_second"]
            notes.append(f"Best worker count in this run: {best['num_workers']} ({best['samples_per_second']:.2f} samples/s); current={current['samples_per_second']:.2f} samples/s ({ratio:.2f}x best/current).")
            required_samples = float(e2e_ips or 0.0) * float(args.batch_size)
            data_wait_share = stage_values.get("data_wait", 0.0) / total if total else 0.0
            if ratio >= 1.20 and (data_wait_share >= 0.20 or current["samples_per_second"] < required_samples * 1.25):
                notes.append("DataLoader worker count is a likely bottleneck; use the fastest sweep value and verify CPU/I/O utilization.")
            elif ratio >= 1.20:
                notes.append(f"The fastest DataLoader setting is {ratio:.2f}x current, but current capacity ({current['samples_per_second']:.2f} samples/s) exceeds measured training demand ({required_samples:.2f} samples/s); it is not limiting this run.")
    synthetic = result.get("synthetic") or {}
    synthetic_ips = synthetic.get("batches_per_second")
    if e2e_ips and synthetic_ips:
        ratio = synthetic_ips / e2e_ips
        notes.append(f"Synthetic/e2e throughput ratio: {synthetic_ips:.2f}/{e2e_ips:.2f} = {ratio:.2f}x.")
        if ratio >= 1.20:
            notes.append("Synthetic throughput is at least 20% higher; host/data/metric work outside CUDA kernels is limiting end-to-end speed.")
        elif ratio < 0.90:
            notes.append("Synthetic throughput was not higher than end-to-end in this short sample; increase --warmup-batches and --profile-batches before drawing a GPU conclusion.")
    if stage_values.get("data_wait", 0.0) > max(total * 0.20, 1.0):
        notes.append("Data wait exceeds 20% of synchronized stages; inspect storage, JPEG/PNG decode, augmentation, and worker count.")
    if stage_values.get("metrics", 0.0) > max(total * 0.15, 1.0):
        notes.append("Per-batch metrics are expensive. S2-0 currently performs argmax, CPU bincount, .item(), and full metrics_from_confusion every batch; profile this before changing the model.")
    if args.deterministic:
        notes.append("Current mode is deterministic: cuDNN benchmark and TF32 are disabled. Run once with --no-deterministic as a speed A/B check if exact reproducibility is not required for diagnosis.")
    if not (args.amp and torch.cuda.is_available()):
        notes.append("AMP is disabled for this run; ensure the real training command is not accidentally using --no-amp on CUDA.")
    before = result.get("nvidia_smi_before") or {}
    before_gpus = before.get("gpus") or []
    if before_gpus:
        try:
            before_utilization = float(str(before_gpus[0].get("utilization.gpu", "")).replace("%", "").strip())
        except ValueError:
            before_utilization = None
        if before_utilization is not None and before_utilization >= 10:
            notes.append(f"GPU was already {before_utilization:.0f}% utilized before profiling; stop other GPU jobs and repeat before comparing machines.")
    processes = (result.get("nvidia_compute_processes_before") or {}).get("processes") or []
    selected_gpu = ((result.get("nvidia_smi_before") or {}).get("gpus") or [{}])[0]
    selected_uuid = selected_gpu.get("uuid")
    if selected_uuid:
        processes = [process for process in processes if process.get("gpu_uuid") == selected_uuid]
    if len(processes) > 1:
        notes.append(f"nvidia-smi reported {len(processes)} compute/display processes before profiling; shared GPU activity can explain unexpectedly low throughput.")
    gpu_summary = result.get("gpu_monitor_summary") or {}
    gpu_utilization = (gpu_summary.get("utilization.gpu") or {}).get("mean")
    if gpu_utilization is not None and gpu_utilization < 60:
        notes.append(f"Average sampled GPU utilization was {gpu_utilization:.1f}%; low utilization with high data_wait usually means the GPU is being starved by the host pipeline.")
    pstates = gpu_summary.get("pstates") or {}
    pstate_samples = sum(int(value) for value in pstates.values())
    if gpu_utilization is not None and gpu_utilization >= 60 and pstate_samples and pstates.get("P0", 0) / pstate_samples < 0.5:
        notes.append(f"GPU was outside P0 for most samples ({pstates}); check server power limits, clocks, virtualization, and thermal throttling.")
    if not notes:
        notes.append("No single bottleneck crossed the heuristic thresholds; compare the stage table and synthetic benchmark between machines.")
    return notes


def print_report(result: Mapping[str, Any]) -> None:
    print("\n=== S2-0 performance profile ===")
    env = result.get("environment", {})
    print(f"device={env.get('device')} gpu={(env.get('gpu') or {}).get('name')} torch={env.get('torch')} cuda={env.get('torch_cuda')}")
    if result.get("data_loader_sweep"):
        print("\nDataLoader sweep (higher samples/s is better):")
        for row in result["data_loader_sweep"]:
            print(f"  workers={row['num_workers']:>2}  first={row['first_batch'].get('mean_ms')} ms  steady={row['steady_next_batch'].get('mean_ms')} ms  samples/s={row.get('samples_per_second')}")
    if result.get("pipeline"):
        pipeline = result["pipeline"]
        print("\nPipeline:")
        print(f"  realistic end-to-end={pipeline.get('end_to_end_batches_per_second')} batches/s")
        realistic = pipeline.get("realistic_end_to_end") or {}
        print(f"  realistic data_wait={((realistic.get('data_wait') or {}).get('mean_ms'))} ms")
        for name, item in (pipeline.get("synchronized_stage_timings") or {}).items():
            print(f"  {name:>16}={item.get('mean_ms')} ms ({(item.get('share_of_synchronized_stage_sum') or 0.0):.1%})")
    if result.get("synthetic"):
        print(f"\nSynthetic GPU-only={result['synthetic'].get('batches_per_second')} batches/s")
    gpu_summary = result.get("gpu_monitor_summary") or {}
    if gpu_summary.get("sample_count"):
        utilization = (gpu_summary.get("utilization.gpu") or {}).get("mean")
        power = (gpu_summary.get("power.draw") or {}).get("mean")
        limit = (gpu_summary.get("power.limit") or {}).get("mean")
        print(f"\nGPU monitor: utilization={utilization}% power={power}/{limit} W pstates={gpu_summary.get('pstates')}")
    print("\nDiagnosis:")
    for note in result.get("diagnosis", []):
        print(f"  - {note}")


def main() -> None:
    args = parse_args()
    if args.profile_batches == 0 and args.trace is None and args.no_pipeline and args.no_synthetic and args.no_worker_sweep and args.no_io_profile:
        raise SystemExit("Nothing to profile; remove one of the --no-* flags or request --trace.")
    common.set_global_seed(args.seed, args.deterministic)
    device = common.resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    dataset_root = args.dataset_root.resolve()
    dataset_lock, entries_by_split = common.validate_dataset_lock(dataset_root)
    environment = collect_environment(device)
    nvidia_selector = _nvidia_device_selector(device)
    before_smi = nvidia_smi_snapshot(nvidia_selector)
    before_processes = nvidia_compute_processes()
    sampler = GpuSampler(
        not args.no_gpu_monitor and device.type == "cuda",
        selector=nvidia_selector,
    )
    result: Dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arguments": vars(args),
        "environment": environment,
        "nvidia_smi_before": before_smi,
        "nvidia_compute_processes_before": before_processes,
        "dataset": {"root": str(dataset_root), "lock": dataset_lock, "train_count": len(entries_by_split["train_local"])},
        "effective": {"device": str(device), "amp": amp_enabled, "pin_memory": device.type == "cuda" if args.pin_memory is None else bool(args.pin_memory)},
    }
    try:
        if not args.no_io_profile:
            result["io_profile"] = profile_io(args, dataset_root, entries_by_split["train_local"])
        if not args.no_worker_sweep:
            result["data_loader_sweep"] = []
            for workers in _parse_workers(args.worker_sweep, args.num_workers):
                loader = build_train_loader(args, dataset_root, entries_by_split, device, workers=workers)
                row = profile_data_loader(loader, args.data_batches)
                result["data_loader_sweep"].append(row)
                del loader
        sampler.start()
        if not args.no_synthetic:
            result["synthetic"] = benchmark_synthetic(args, device, amp_enabled)
        if not args.no_pipeline:
            loader = build_train_loader(args, dataset_root, entries_by_split, device)
            result["pipeline"] = profile_pipeline(args, loader, device, amp_enabled)
            if args.trace is not None:
                trace_loader = build_train_loader(args, dataset_root, entries_by_split, device)
                result["trace"] = run_trace(args, trace_loader, device, amp_enabled, args.trace)
                del trace_loader
            del loader
        elif args.trace is not None:
            loader = build_train_loader(args, dataset_root, entries_by_split, device)
            result["trace"] = run_trace(args, loader, device, amp_enabled, args.trace)
            del loader
    finally:
        result["gpu_samples"] = sampler.stop()
        result["gpu_monitor_summary"] = summarize_gpu_samples(result["gpu_samples"])
        result["nvidia_smi_after"] = nvidia_smi_snapshot(nvidia_selector)
    result["diagnosis"] = diagnose(result, args)
    if args.output is None:
        output = SCRIPT_DIR / "result" / "profiling" / f"s2_0_profile_{time.strftime('%Y%m%d_%H%M%S')}.json"
    else:
        output = args.output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print_report(result)
    print(f"\nJSON report: {output}")


if __name__ == "__main__":
    main()
