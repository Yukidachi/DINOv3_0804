"""K3-G server entry point: K3 gradient-angle audit.

K3-G is a diagnostic rerun of the locked K3 experiment.  It keeps K3's
initialization, data order, objective, weights, optimizer, scheduler, DDP
lifecycle, and 80,000-step budget unchanged.  The only additional work is a
fixed-batch gradient audit at pre-registered optimizer steps::

    1, 4000, 20000, 40000, 60000, 80000

For every OS=4/8/16 student tap the audit records the raw (unweighted)
gradient L2 norms and the pairwise cosine similarities between CE, feature KD,
and pixel-logit KD.  The audit forward is state-preserving: BatchNorm buffers,
module training modes, and RNG states are restored after the diagnostic
forward, so it cannot change the K3 optimization trajectory.

Typical two-GPU server command::

    torchrun --standalone --nproc_per_node=2 dino_k3_g_server.py \
        --seed 42 --batch-size 2 --global-batch-size 8 \
        --num-workers 8 --multiprocessing-context spawn \
        --no-pin-memory --persistent-workers

The output is written below ``result/K_MobileNetV2_RASPP_server/K3-G/seed_42``.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import dino as common
import dino_a0_server as a0
import dino_k0_server as k0
import dino_k2_server as k2
import dino_k3_server as k3


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT = "K3-G"
ARTIFACT_TYPE = "mobilenetv2_cityscapes19_raspp_k3_gradient_audit"
AUDIT_STEPS: Tuple[int, ...] = (1, 4_000, 20_000, 40_000, 60_000, 80_000)
AUDIT_BATCH_INDEX = 0


# The first local training micro-batch is captured independently by each rank.
# It is reused only by the diagnostic forward and is never fed to the
# optimizer a second time.
_FIXED_AUDIT_BATCH: Optional[Dict[str, Any]] = None

_ORIGINAL_K3_FILE = k3.__file__
_ORIGINAL_K3_EXPERIMENT = k3.EXPERIMENT
_ORIGINAL_K3_ARTIFACT_TYPE = k3.ARTIFACT_TYPE
_ORIGINAL_K3_PATHS = k3.k3_paths
_ORIGINAL_K3_BUILD_CONFIG = k3.build_config
_ORIGINAL_K3_BUILD_BEST_CHECKPOINT = k3.build_best_checkpoint
_ORIGINAL_K3_TRAIN_ONE_EPOCH = k3.train_one_epoch_k3
_ORIGINAL_K3_SMOKE_TEST = k3.smoke_test_k3
_ORIGINAL_K3_POSTPROCESS = k3._postprocess_metrics


def parse_args() -> Any:
    """Reuse K3's complete CLI, while locking the K3-G protocol."""

    args = k3.parse_args()
    if args.seed != 42:
        raise SystemExit("K3-G is pre-registered for --seed 42")
    if args.max_steps != 80_000:
        raise SystemExit("K3-G is pre-registered for exactly 80,000 optimizer steps")
    if args.resume:
        raise SystemExit(
            "K3-G must start from step 0 so its fixed audit batch is preserved; "
            "--resume is not supported"
        )
    if args.gradient_log_steps != 500:
        raise SystemExit(
            "K3-G uses its fixed audit schedule; --gradient-log-steps must remain 500"
        )
    return args


def k3g_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    """Use K3's artifact schema in a separate K3-G directory."""

    original = k3._ORIGINAL_K0_PATHS(output_dir, seed)
    run_dir = output_dir.resolve() / "K3-G" / f"seed_{seed}"
    return {
        key: run_dir if key == "run_dir" else run_dir / value.name
        for key, value in original.items()
    }


def _warmup_steps(args: Any) -> int:
    return k3._warmup_steps(args)


def _gradient_l2(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm(2).item())


def _gradient_cosine(first: torch.Tensor, second: torch.Tensor) -> Optional[float]:
    first_flat = first.detach().float().reshape(-1)
    second_flat = second.detach().float().reshape(-1)
    first_norm = first_flat.norm(2)
    second_norm = second_flat.norm(2)
    denominator = first_norm * second_norm
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) == 0.0:
        return None
    value = torch.dot(first_flat, second_flat) / denominator
    if not bool(torch.isfinite(value)):
        return None
    return float(value.item())


@contextlib.contextmanager
def _preserve_student_state(model: torch.nn.Module, device: torch.device):
    """Preserve all state that a train-mode diagnostic forward can mutate."""

    modes = [(module, module.training) for module in model.modules()]
    buffers = [
        (buffer, buffer.detach().clone())
        for buffer in model.buffers()
        if torch.is_tensor(buffer)
    ]
    cpu_rng = torch.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        for module, was_training in modes:
            module.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device=device)


def _audit_fixed_batch(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    projection: torch.nn.ModuleDict,
    images: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    optimizer_step: int,
    learning_rate: float,
    warmup_weight: float,
    rank: int,
    paths: Sequence[str],
) -> Dict[str, object]:
    """Measure raw component gradients on one fixed local batch."""

    args = k3._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("K3-G active arguments were not set")

    with _preserve_student_state(model, device):
        model.train()
        teacher.eval()
        projection.eval()
        model.zero_grad(set_to_none=True)

        with k3.common.autocast_context(device, amp_enabled):
            student_output = model(images)
            if not isinstance(student_output, Mapping):
                raise RuntimeError("K3-G audit forward did not return tapped features")
            logits = student_output["logits"]
            student_features = student_output["features"]
            with torch.no_grad():
                teacher_features, teacher_logits = k3._teacher_features_and_logits(
                    teacher, images
                )
            layer_losses, projected_shapes = k3._feature_kd_losses(
                student_features, teacher_features, projection
            )

        logits_float = logits.float()
        valid_pixels = int((targets != common.IGNORE_INDEX).sum().item())
        if valid_pixels == 0:
            raise RuntimeError("K3-G audit batch contains no valid pixels")
        loss_seg = F.cross_entropy(
            logits_float,
            targets,
            ignore_index=common.IGNORE_INDEX,
            reduction="sum",
        ) / valid_pixels
        loss_feat = sum(layer_losses.values()) / len(a0.A0_LAYER_ORDER)
        loss_logit = k2._masked_pixel_kl(
            teacher_logits, logits_float, targets, args.temperature
        )
        total_loss = loss_seg + warmup_weight * (
            args.lambda_feat * loss_feat + args.lambda_logit * loss_logit
        )
        losses = {
            "ce": loss_seg,
            "feature": loss_feat,
            "logit": loss_logit,
        }
        if not all(
            torch.isfinite(value)
            for value in [loss_seg, loss_feat, loss_logit, total_loss]
        ):
            raise RuntimeError("K3-G audit produced a non-finite loss")

        layer_records: Dict[str, Dict[str, object]] = {}
        for layer in a0.A0_LAYER_ORDER:
            tap = student_features[layer]
            gradients: Dict[str, torch.Tensor] = {}
            for name, loss in losses.items():
                gradients[name] = torch.autograd.grad(
                    loss,
                    tap,
                    retain_graph=True,
                    allow_unused=False,
                )[0].detach().float()

            grad_total = gradients["ce"] + warmup_weight * (
                args.lambda_feat * gradients["feature"]
                + args.lambda_logit * gradients["logit"]
            )
            layer_records[layer] = {
                "tap_shape": list(tap.shape),
                "grad_l2_ce": _gradient_l2(gradients["ce"]),
                "grad_l2_feature": _gradient_l2(gradients["feature"]),
                "grad_l2_logit": _gradient_l2(gradients["logit"]),
                "grad_l2_feature_effective": _gradient_l2(
                    warmup_weight * args.lambda_feat * gradients["feature"]
                ),
                "grad_l2_logit_effective": _gradient_l2(
                    warmup_weight * args.lambda_logit * gradients["logit"]
                ),
                "grad_l2_total_effective": _gradient_l2(grad_total),
                "cos_ce_feature": _gradient_cosine(
                    gradients["ce"], gradients["feature"]
                ),
                "cos_ce_logit": _gradient_cosine(
                    gradients["ce"], gradients["logit"]
                ),
                "cos_feature_logit": _gradient_cosine(
                    gradients["feature"], gradients["logit"]
                ),
            }
            del gradients, grad_total

        student_gradient_present = any(
            parameter.grad is not None for parameter in model.parameters()
        )
        if student_gradient_present:
            raise RuntimeError("K3-G audit unexpectedly populated student parameter grads")

    return {
        "rank": rank,
        "optimizer_step": optimizer_step,
        "state_timing": "after_optimizer_step",
        "audit_batch_index": AUDIT_BATCH_INDEX,
        "audit_batch_paths": list(paths),
        "image_tensor_shape": list(images.shape),
        "target_tensor_shape": list(targets.shape),
        "image_tensor_sha256": k0._tensor_sha256(images),
        "target_tensor_sha256": k0._tensor_sha256(targets),
        "valid_pixels": valid_pixels,
        "learning_rate": learning_rate,
        "warmup_weight": warmup_weight,
        "lambda_feat": args.lambda_feat,
        "lambda_logit": args.lambda_logit,
        "temperature": args.temperature,
        "losses": {
            "ce": float(loss_seg.detach().item()),
            "feature": float(loss_feat.detach().item()),
            "logit": float(loss_logit.detach().item()),
            "total_effective": float(total_loss.detach().item()),
            "feature_by_layer": {
                layer: float(layer_losses[layer].detach().item())
                for layer in a0.A0_LAYER_ORDER
            },
        },
        "projected_teacher_shapes": projected_shapes,
        "layers": layer_records,
        "gradient_definition": "raw component gradients before lambda and warm-up",
    }


def _mean_std(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return {"mean": None, "sample_std": None}
    mean = sum(finite) / len(finite)
    if len(finite) < 2:
        sample_std = 0.0
    else:
        sample_std = math.sqrt(
            sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
        )
    return {"mean": mean, "sample_std": sample_std}


def _aggregate_audit_records(
    rows: Sequence[Optional[Mapping[str, object]]],
    world_size: int,
) -> Dict[str, object]:
    """Make a rank-0 summary while retaining every rank's raw audit row."""

    valid_rows = [row for row in rows if row is not None]
    if len(valid_rows) != world_size:
        raise RuntimeError(
            f"K3-G audit gathered {len(valid_rows)} rank rows; expected {world_size}"
        )
    first = valid_rows[0]
    summary: Dict[str, object] = {
        "experiment": EXPERIMENT,
        "optimizer_step": first["optimizer_step"],
        "state_timing": first["state_timing"],
        "audit_batch_index": AUDIT_BATCH_INDEX,
        "world_size": world_size,
        "rank_aggregation": "mean across ranks; sample_std also reported",
        "per_rank": list(valid_rows),
        "warmup_weight": first["warmup_weight"],
        "lambda_feat": first["lambda_feat"],
        "lambda_logit": first["lambda_logit"],
        "temperature": first["temperature"],
        "gradient_definition": first["gradient_definition"],
        "audit_batch_paths_by_rank": [row["audit_batch_paths"] for row in valid_rows],
        "audit_batch_hashes_by_rank": [
            {
                "image_tensor_sha256": row["image_tensor_sha256"],
                "target_tensor_sha256": row["target_tensor_sha256"],
            }
            for row in valid_rows
        ],
    }

    losses: Dict[str, object] = {}
    first_losses = first["losses"]
    assert isinstance(first_losses, Mapping)
    for name in ("ce", "feature", "logit", "total_effective"):
        losses[name] = _mean_std(
            [
                row["losses"][name]  # type: ignore[index]
                for row in valid_rows
            ]
        )
    losses["feature_by_layer"] = {
        layer: _mean_std(
            [
                row["losses"]["feature_by_layer"][layer]  # type: ignore[index]
                for row in valid_rows
            ]
        )
        for layer in a0.A0_LAYER_ORDER
    }
    summary["losses"] = losses

    layer_summary: Dict[str, object] = {}
    for layer in a0.A0_LAYER_ORDER:
        first_layer = first["layers"][layer]  # type: ignore[index]
        assert isinstance(first_layer, Mapping)
        metrics: Dict[str, object] = {}
        for metric in (
            "grad_l2_ce",
            "grad_l2_feature",
            "grad_l2_logit",
            "grad_l2_feature_effective",
            "grad_l2_logit_effective",
            "grad_l2_total_effective",
            "cos_ce_feature",
            "cos_ce_logit",
            "cos_feature_logit",
        ):
            metrics[metric] = _mean_std(
                [
                    row["layers"][layer][metric]  # type: ignore[index]
                    for row in valid_rows
                ]
            )
        metrics["tap_shape"] = first_layer["tap_shape"]
        layer_summary[layer] = metrics
        # Flat aliases make the JSONL convenient for plotting scripts.
        for metric, value in metrics.items():
            if metric == "tap_shape":
                continue
            assert isinstance(value, Mapping)
            summary[f"{metric}_{layer}"] = value["mean"]
    summary["layers"] = layer_summary
    return summary


def _gather_audit_record(
    local_record: Mapping[str, object], world_size: int
) -> Optional[Dict[str, object]]:
    rows: List[Optional[Mapping[str, object]]] = [None for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather_object(rows, local_record)
    else:
        rows[0] = local_record
    if int(os.environ.get("RANK", "0")) != 0:
        return None
    return _aggregate_audit_records(rows, world_size)


def _capture_fixed_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    paths: Sequence[str],
) -> None:
    global _FIXED_AUDIT_BATCH
    if _FIXED_AUDIT_BATCH is not None:
        return
    _FIXED_AUDIT_BATCH = {
        "images": images.detach().clone(),
        "targets": targets.detach().clone(),
        "paths": list(paths),
    }


def train_one_epoch_k3g(
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
    """K3 training loop with only the fixed-batch audit added."""

    teacher, projection = k3._require_resources()
    args = k3._ACTIVE_ARGS
    if args is None:
        raise RuntimeError("K3-G active arguments were not set")
    warmup_steps = _warmup_steps(args)
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
    logit_sum = 0.0
    total_sum = 0.0
    layer_sums = {layer: 0.0 for layer in a0.A0_LAYER_ORDER}
    batch_count = 0
    optimizer_steps = 0
    last_warmup_weight = 0.0
    gradient_records: List[Dict[str, object]] = []
    first_batch_audit: Optional[Dict[str, object]] = None

    possible_steps = math.ceil(len(loader) / accumulation_steps)
    target_steps = min(possible_steps, remaining_optimizer_steps)
    max_batches = min(len(loader), target_steps * accumulation_steps)
    progress = k3.tqdm(
        loader, desc=f"Epoch {epoch} [K3-G CE+feature+logit]", disable=rank != 0
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
        if starting_optimizer_step == 0 and batch_index == 0:
            _capture_fixed_batch(images, targets, paths)
        next_optimizer_step = starting_optimizer_step + optimizer_steps + 1
        warmup_weight = min(1.0, next_optimizer_step / warmup_steps)

        sync_context = contextlib.nullcontext()
        if isinstance(model, DDP) and not sync_gradients:
            sync_context = model.no_sync()
        with sync_context:
            with k3.common.autocast_context(device, amp_enabled):
                student_output = model(images)
                if not isinstance(student_output, Mapping):
                    raise RuntimeError("K3-G training forward did not return features")
                logits = student_output["logits"]
                student_features = student_output["features"]
                with torch.no_grad():
                    teacher_features, teacher_logits = k3._teacher_features_and_logits(
                        teacher, images
                    )
                layer_losses, projected_shapes = k3._feature_kd_losses(
                    student_features, teacher_features, projection
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
                raise RuntimeError("Training batch contains no valid Cityscapes pixels")
            loss_seg = batch_ce_sum / batch_valid
            loss_feat = sum(layer_losses.values()) / len(a0.A0_LAYER_ORDER)
            loss_logit = k2._masked_pixel_kl(
                teacher_logits, logits_float, targets, args.temperature
            )
            total_loss = loss_seg + warmup_weight * (
                args.lambda_feat * loss_feat + args.lambda_logit * loss_logit
            )
            if not all(
                torch.isfinite(value)
                for value in [
                    loss_seg,
                    loss_feat,
                    loss_logit,
                    total_loss,
                    *layer_losses.values(),
                ]
            ):
                raise RuntimeError("K3-G produced a non-finite CE/feature/KL loss")
            # This is byte-for-byte the K3 optimization loss and reduction.
            scaler.scale(total_loss / group_size).backward()

        if sync_gradients:
            scaler.unscale_(optimizer)
            optimizer_steps += 1
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            completed_step = starting_optimizer_step + optimizer_steps
            if completed_step in AUDIT_STEPS:
                if _FIXED_AUDIT_BATCH is None:
                    raise RuntimeError("K3-G fixed audit batch was not captured")
                local_record = _audit_fixed_batch(
                    model=model,
                    teacher=teacher,
                    projection=projection,
                    images=_FIXED_AUDIT_BATCH["images"],
                    targets=_FIXED_AUDIT_BATCH["targets"],
                    device=device,
                    amp_enabled=amp_enabled,
                    optimizer_step=completed_step,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    warmup_weight=min(1.0, completed_step / warmup_steps),
                    rank=rank,
                    paths=_FIXED_AUDIT_BATCH["paths"],
                )
                summary = _gather_audit_record(local_record, world_size)
                if summary is not None:
                    gradient_records.append(summary)

        if first_batch_audit is None and starting_optimizer_step == 0 and batch_index == 0:
            first_batch_audit = {
                "rank": rank,
                "epoch": epoch,
                "micro_batch_index": 0,
                "paths": list(paths),
                "image_tensor_shape": list(images.shape),
                "target_tensor_shape": list(targets.shape),
                "image_tensor_sha256": k0._tensor_sha256(images),
                "target_tensor_sha256": k0._tensor_sha256(targets),
                "valid_pixels": batch_valid,
                "student_logit_shape": list(logits_float.shape),
                "teacher_logit_shape": list(teacher_logits.shape),
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
                    layer: float(layer_losses[layer].detach().item())
                    for layer in a0.A0_LAYER_ORDER
                },
                "ce_loss": float(loss_seg.detach().item()),
                "feature_loss": float(loss_feat.detach().item()),
                "logit_loss": float(loss_logit.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                "temperature": args.temperature,
                "lambda_feat": args.lambda_feat,
                "lambda_logit": args.lambda_logit,
                "warmup_weight": warmup_weight,
                "teacher_backbone_forward_count": 1,
                **k3._resource_hashes(),
            }

        predictions = logits_float.detach().argmax(dim=1)
        confusion += common.confusion_counts(predictions, targets)
        ce_loss_sum += float(batch_ce_sum.detach().item())
        valid_pixels += batch_valid
        feature_value = float(loss_feat.detach().item())
        logit_value = float(loss_logit.detach().item())
        feature_sum += feature_value
        logit_sum += logit_value
        total_sum += float(total_loss.detach().item())
        for layer in a0.A0_LAYER_ORDER:
            layer_sums[layer] += float(layer_losses[layer].detach().item())
        batch_count += 1
        last_warmup_weight = warmup_weight
        if rank == 0:
            running = common.metrics_from_confusion(
                confusion, ce_loss_sum, valid_pixels
            )
            progress.set_postfix(
                {
                    "CE": f"{running['loss']:.4f}",
                    "feat": f"{feature_value:.4f}",
                    "KL": f"{logit_value:.4f}",
                    "mIoU": f"{running['mIoU']:.4f}",
                    "warm": f"{warmup_weight:.3f}",
                    "steps": optimizer_steps,
                }
            )

    if optimizer_steps != target_steps:
        raise RuntimeError(
            f"K3-G optimizer-step accounting failed: actual={optimizer_steps}, "
            f"expected={target_steps}"
        )
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("K3-G training found a gradient on the frozen teacher")
    if list(projection.parameters()):
        raise RuntimeError("K3-G projection became trainable during training")
    metrics = k3.server_base._reduce_train_metrics(
        confusion, ce_loss_sum, valid_pixels, device, world_size
    )
    (
        layer_means,
        feature_mean,
        logit_mean,
        total_mean,
        global_batches,
    ) = k3._reduce_auxiliary_statistics(
        layer_sums,
        feature_sum,
        logit_sum,
        total_sum,
        batch_count,
        device,
        world_size,
    )
    metrics["loss_schema"] = (
        "hard_label_CE_plus_A0_fixed_PCA_feature_MSE_plus_"
        "full_resolution_masked_pixel_KL"
    )
    metrics["ce_loss"] = metrics["loss"]
    metrics["feature_loss"] = feature_mean
    metrics["feature_loss_by_layer"] = layer_means
    metrics["logit_loss"] = logit_mean
    metrics["total_loss_micro_batch_mean"] = total_mean
    metrics["warmup_weight"] = last_warmup_weight
    metrics["micro_batches_global"] = global_batches
    metrics["gradient_audit_steps"] = list(AUDIT_STEPS)
    return metrics, optimizer_steps, gradient_records, first_batch_audit


def smoke_test_k3g(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    rank: int,
) -> None:
    """Run K3's smoke checks and one local gradient-angle audit."""

    _ORIGINAL_K3_SMOKE_TEST(model, loader, device, amp_enabled, rank)
    teacher, projection = k3._require_resources()
    images, targets, paths = next(iter(loader))
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    record = _audit_fixed_batch(
        model=model,
        teacher=teacher,
        projection=projection,
        images=images,
        targets=targets,
        device=device,
        amp_enabled=amp_enabled,
        optimizer_step=1,
        learning_rate=0.01,
        warmup_weight=1.0 / _warmup_steps(k3._ACTIVE_ARGS),
        rank=rank,
        paths=paths,
    )
    for layer in a0.A0_LAYER_ORDER:
        layer_data = record["layers"][layer]  # type: ignore[index]
        for metric in ("grad_l2_ce", "grad_l2_feature", "grad_l2_logit"):
            if float(layer_data[metric]) <= 0.0:  # type: ignore[index]
                raise RuntimeError(f"K3-G smoke audit found zero {metric} at {layer}")
    if rank == 0:
        print(
            "[OK] K3-G fixed-batch gradient audit smoke: "
            f"sample={paths[0]}, step=1, "
            + ", ".join(
                f"{layer}.cos(CE,feat)={record['layers'][layer]['cos_ce_feature']:.6f}"  # type: ignore[index]
                for layer in a0.A0_LAYER_ORDER
            )
        )


def build_config_k3g(
    args: Any,
    accumulation_steps: int,
    world_size: int,
    device: torch.device,
    shared_init_state_sha256: str,
    shared_init_file_sha256: str,
) -> Dict[str, object]:
    config = _ORIGINAL_K3_BUILD_CONFIG(
        args,
        accumulation_steps,
        world_size,
        device,
        shared_init_state_sha256,
        shared_init_file_sha256,
    )
    config["experiment"] = EXPERIMENT
    config["artifact_type"] = ARTIFACT_TYPE
    config["server_entry_point"] = str(Path(__file__).resolve())
    config["gradient_audit"] = {
        "enabled": True,
        "fixed_batch_index": AUDIT_BATCH_INDEX,
        "fixed_batch_definition": "first local train micro-batch at step 0",
        "fixed_steps": list(AUDIT_STEPS),
        "state_timing": "after_optimizer_step",
        "layers": list(a0.A0_LAYER_ORDER),
        "pairs": [
            "cos(CE,feature)",
            "cos(CE,logit)",
            "cos(feature,logit)",
        ],
        "raw_gradients_before_lambda_and_warmup": True,
        "rank_summary": "mean across ranks with sample_std",
        "preserve_batchnorm_buffers_and_rng": True,
        "training_objective_changed": False,
    }
    return config


def build_best_checkpoint_k3g(*args: Any, **kwargs: Any) -> Dict[str, object]:
    payload = _ORIGINAL_K3_BUILD_BEST_CHECKPOINT(*args, **kwargs)
    payload["experiment"] = EXPERIMENT
    payload["artifact_type"] = ARTIFACT_TYPE
    payload["gradient_audit"] = {
        "fixed_batch_index": AUDIT_BATCH_INDEX,
        "fixed_steps": list(AUDIT_STEPS),
        "state_timing": "after_optimizer_step",
    }
    return payload


def _postprocess_metrics_k3g(args: Any) -> None:
    _ORIGINAL_K3_POSTPROCESS(args)
    if int(os.environ.get("RANK", "0")) != 0:
        return
    metrics_path = k3g_paths(args.output_dir, args.seed)["metrics"]
    if not metrics_path.is_file():
        return
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    results["experiment"] = EXPERIMENT
    results["artifact_type"] = ARTIFACT_TYPE
    results["protocol"] = (
        "K3-G diagnostic rerun of locked K3: same scratch initialization, "
        "data order, CE+feature+logit objective, weights, optimizer, DDP and "
        "80k budget; only fixed-batch gradient-angle audits are added."
    )
    results["gradient_audit"] = {
        "enabled": True,
        "fixed_batch_index": AUDIT_BATCH_INDEX,
        "fixed_steps": list(AUDIT_STEPS),
        "state_timing": "after_optimizer_step",
        "layers": list(a0.A0_LAYER_ORDER),
        "pairs": [
            "cos(CE,feature)",
            "cos(CE,logit)",
            "cos(feature,logit)",
        ],
        "gradient_norm_file": str(k3g_paths(args.output_dir, args.seed)["gradient_norms"]),
        "training_objective_changed": False,
    }
    results["test_local_evaluated"] = False
    common.write_json_atomic(metrics_path, results)


def run_training(args: Any) -> None:
    """Temporarily route K3's shared server runner through the K3-G audit."""

    global _FIXED_AUDIT_BATCH
    _FIXED_AUDIT_BATCH = None
    saved = {
        "__file__": k3.__file__,
        "EXPERIMENT": k3.EXPERIMENT,
        "ARTIFACT_TYPE": k3.ARTIFACT_TYPE,
        "k3_paths": k3.k3_paths,
        "build_config": k3.build_config,
        "build_best_checkpoint": k3.build_best_checkpoint,
        "train_one_epoch_k3": k3.train_one_epoch_k3,
        "smoke_test_k3": k3.smoke_test_k3,
        "_postprocess_metrics": k3._postprocess_metrics,
    }
    k3.__file__ = str(Path(__file__).resolve())
    k3.EXPERIMENT = EXPERIMENT
    k3.ARTIFACT_TYPE = ARTIFACT_TYPE
    k3.k3_paths = k3g_paths
    k3.build_config = build_config_k3g
    k3.build_best_checkpoint = build_best_checkpoint_k3g
    k3.train_one_epoch_k3 = train_one_epoch_k3g
    k3.smoke_test_k3 = smoke_test_k3g
    k3._postprocess_metrics = _postprocess_metrics_k3g
    try:
        k3.run_training(args)
    finally:
        _FIXED_AUDIT_BATCH = None
        for name, value in saved.items():
            setattr(k3, name, value)


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
