"""H-PB: paired-bootstrap evaluation for the H0-H3 activation experiments.

This entry point does not train or load a model.  It consumes the frozen
per-image dev_local confusion matrices produced by the H experiments and
estimates image-level uncertainty for registered same-seed mIoU contrasts.

The default screening run evaluates:

* H1-H0: all in-block Hardswish versus the ReLU6 anchor;
* H2-H0: late expansion+depthwise Hardswish versus the ReLU6 anchor;
* H3-H0: late depthwise-only Hardswish versus the ReLU6 anchor;
* H2-H1: late placement versus full in-block placement;
* H3-H2: depthwise-only versus late expansion+depthwise placement.

The protocol follows dino_d1_pb_server.py:

* 445 paired dev_local images for each training seed;
* 100,000 paired bootstrap resamples;
* bootstrap RNG seed 260820;
* the same sampled image indices are used for every H model in a seed;
* each resample sums 19x19 confusion matrices before computing mIoU;
* different training seeds are never mixed into one bootstrap population.

The default seed is 42 because H1-H3 are screened there first.  Once a
candidate has completed the matched three-seed extension, pass
--seeds 42 3407 260805 with the applicable --comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_H_ROOT = SCRIPT_DIR / "result" / "H_MobileNetV2_RASPP_server"
DEFAULT_OUTPUT_DIR = DEFAULT_H_ROOT / "H-PB"

FORMAL_SEEDS = (42, 3407, 260805)
SCREENING_SEED = 42
REGISTERED_EXPERIMENTS = ("H0", "H1", "H2", "H3")
DEFAULT_COMPARISONS = (
    ("H1", "H0"),
    ("H2", "H0"),
    ("H3", "H0"),
    ("H2", "H1"),
    ("H3", "H2"),
)
NUM_CLASSES = 19
DEFAULT_EXPECTED_IMAGES = 445
DEFAULT_BOOTSTRAP_REPETITIONS = 100_000
DEFAULT_BOOTSTRAP_SEED = 260820
DEFAULT_BOOTSTRAP_BATCH_SIZE = 4096
METRIC_TOLERANCE = 1e-12
K1_THREE_SEED_SAMPLE_STD = 0.002194


class HPBError(RuntimeError):
    """Raised when an H paired-bootstrap input or protocol audit fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, np.asarray(values, dtype=np.float64), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _coerce_confusion(value: Any, *, path: Path, line_number: int) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise HPBError(f"{path}:{line_number}: invalid confusion_matrix") from exc
    if array.shape != (NUM_CLASSES, NUM_CLASSES):
        raise HPBError(
            f"{path}:{line_number}: expected a {NUM_CLASSES}x{NUM_CLASSES} "
            f"confusion matrix, got shape {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        numeric = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise HPBError(f"{path}:{line_number}: confusion counts must be integers")
    array = np.asarray(array, dtype=np.int64)
    if np.any(array < 0):
        raise HPBError(f"{path}:{line_number}: confusion counts cannot be negative")
    return array


def load_per_image_confusion(
    path: Path, *, expected_images: Optional[int] = DEFAULT_EXPECTED_IMAGES
) -> Dict[str, Any]:
    """Load and audit one dev_per_image_confusion.jsonl file."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise HPBError(f"Missing per-image confusion file: {path}")

    images: List[str] = []
    image_set = set()
    valid_pixels: List[int] = []
    matrices: List[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise HPBError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, Mapping):
                raise HPBError(f"{path}:{line_number}: JSON row must be an object")

            image = row.get("image")
            if not isinstance(image, str) or not image:
                raise HPBError(f"{path}:{line_number}: missing non-empty image name")
            if image in image_set:
                raise HPBError(f"{path}:{line_number}: duplicate image name {image!r}")
            image_set.add(image)

            if "valid_pixels" not in row:
                raise HPBError(f"{path}:{line_number}: missing valid_pixels")
            try:
                valid = int(row["valid_pixels"])
            except (TypeError, ValueError) as exc:
                raise HPBError(f"{path}:{line_number}: invalid valid_pixels") from exc
            if valid < 0:
                raise HPBError(f"{path}:{line_number}: valid_pixels cannot be negative")

            matrix = _coerce_confusion(
                row.get("confusion_matrix"), path=path, line_number=line_number
            )
            matrix_sum = int(matrix.sum())
            if matrix_sum != valid:
                raise HPBError(
                    f"{path}:{line_number}: valid_pixels={valid} does not match "
                    f"confusion sum={matrix_sum}"
                )
            images.append(image)
            valid_pixels.append(valid)
            matrices.append(matrix)

    if not matrices:
        raise HPBError(f"{path}: file contains no image rows")
    if expected_images is not None and len(matrices) != expected_images:
        raise HPBError(
            f"{path}: expected {expected_images} images, found {len(matrices)}"
        )

    matrix_stack = np.stack(matrices, axis=0)
    row_sums = matrix_stack.sum(axis=2, dtype=np.int64)
    column_sums = matrix_stack.sum(axis=1, dtype=np.int64)
    diagonal = np.diagonal(matrix_stack, axis1=1, axis2=2).copy()
    statistics = np.concatenate((row_sums, column_sums, diagonal), axis=1)
    return {
        "path": path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "images": images,
        "valid_pixels": np.asarray(valid_pixels, dtype=np.int64),
        "stats": statistics,
    }


def validate_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_name: str,
    right_name: str,
    seed: int,
) -> Dict[str, Any]:
    left_images = list(left["images"])
    right_images = list(right["images"])
    if left_images != right_images:
        first_difference = next(
            (
                index
                for index, (lhs, rhs) in enumerate(zip(left_images, right_images))
                if lhs != rhs
            ),
            min(len(left_images), len(right_images)),
        )
        raise HPBError(
            f"seed={seed}: {left_name}/{right_name} image order or set mismatch "
            f"at index {first_difference}: "
            f"{left_images[first_difference:first_difference + 1]} vs "
            f"{right_images[first_difference:first_difference + 1]}"
        )
    if not np.array_equal(left["valid_pixels"], right["valid_pixels"]):
        index = int(
            np.flatnonzero(left["valid_pixels"] != right["valid_pixels"])[0]
        )
        raise HPBError(
            f"seed={seed}: valid_pixels mismatch for {left_images[index]!r}: "
            f"{left_name}={int(left['valid_pixels'][index])}, "
            f"{right_name}={int(right['valid_pixels'][index])}"
        )
    if left["stats"].shape != right["stats"].shape:
        raise HPBError(
            f"seed={seed}: {left_name}/{right_name} sufficient-statistic "
            "shapes differ"
        )
    return {
        "seed": int(seed),
        "paired": True,
        "left_experiment": left_name,
        "right_experiment": right_name,
        "image_count": len(left_images),
        "order_equal": True,
        "image_set_equal": True,
        "valid_pixels_equal": True,
        "class_count": NUM_CLASSES,
        "image_name_sha256": hashlib.sha256(
            "\n".join(left_images).encode("utf-8")
        ).hexdigest(),
    }


def class_iou_from_statistics(statistics: np.ndarray) -> np.ndarray:
    """Compute class IoUs from final-dimension row/column/diagonal statistics."""

    values = np.asarray(statistics, dtype=np.float64)
    if values.shape[-1] != NUM_CLASSES * 3:
        raise ValueError(
            f"Expected the final dimension to be {NUM_CLASSES * 3}, "
            f"got {values.shape}"
        )
    rows = values[..., :NUM_CLASSES]
    columns = values[..., NUM_CLASSES : 2 * NUM_CLASSES]
    diagonal = values[..., 2 * NUM_CLASSES :]
    union = rows + columns - diagonal
    iou = np.full(union.shape, np.nan, dtype=np.float64)
    np.divide(diagonal, union, out=iou, where=union > 0)
    return iou


def miou_from_statistics(statistics: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(class_iou_from_statistics(statistics), axis=-1)


def _percentile(values: np.ndarray, percentile: float) -> float:
    try:
        return float(np.percentile(values, percentile, method="linear"))
    except TypeError:
        return float(np.percentile(values, percentile, interpolation="linear"))


def paired_bootstrap_deltas(
    statistics_by_experiment: Mapping[str, np.ndarray],
    comparisons: Sequence[Tuple[str, str]],
    *,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    random_seed: int = DEFAULT_BOOTSTRAP_SEED,
    batch_size: int = DEFAULT_BOOTSTRAP_BATCH_SIZE,
) -> Dict[str, np.ndarray]:
    """Bootstrap all requested H contrasts using one shared index stream."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not comparisons:
        raise ValueError("at least one comparison is required")

    experiments = tuple(
        dict.fromkeys(name for comparison in comparisons for name in comparison)
    )
    values: Dict[str, np.ndarray] = {}
    expected_shape: Optional[Tuple[int, int]] = None
    for experiment in experiments:
        if experiment not in statistics_by_experiment:
            raise ValueError(f"Missing statistics for {experiment}")
        array = np.asarray(statistics_by_experiment[experiment], dtype=np.int64)
        if array.ndim != 2 or array.shape[1] != NUM_CLASSES * 3:
            raise ValueError(
                f"{experiment} statistics must have shape [N, {NUM_CLASSES * 3}]"
            )
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError("All H statistics must have the same shape")
        values[experiment] = array

    assert expected_shape is not None
    n_images = expected_shape[0]
    if n_images == 0:
        raise ValueError("Cannot bootstrap an empty image set")

    comparison_ids = [comparison_id(pair) for pair in comparisons]
    distributions = {
        identifier: np.empty(int(repetitions), dtype=np.float64)
        for identifier in comparison_ids
    }
    rng = np.random.default_rng(int(random_seed))
    for start in range(0, int(repetitions), int(batch_size)):
        count = min(int(batch_size), int(repetitions) - start)
        indices = rng.integers(0, n_images, size=(count, n_images), dtype=np.int64)
        counts = np.zeros((count, n_images), dtype=np.int32)
        row_numbers = np.broadcast_to(
            np.arange(count, dtype=np.int64)[:, None], indices.shape
        )
        np.add.at(counts, (row_numbers, indices), 1)
        bootstrap_miou = {
            experiment: miou_from_statistics(counts @ values[experiment])
            for experiment in experiments
        }
        for (left_name, right_name), identifier in zip(
            comparisons, comparison_ids
        ):
            distributions[identifier][start : start + count] = (
                bootstrap_miou[left_name] - bootstrap_miou[right_name]
            )
    return distributions


def _finite_float(value: Any, *, context: str) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HPBError(f"Invalid numeric value for {context}: {value!r}") from exc
    if not np.isfinite(result):
        raise HPBError(f"Non-finite numeric value for {context}: {result}")
    return result


def load_metrics(path: Path, *, experiment: str, seed: int) -> Dict[str, Any]:
    """Load the frozen best-dev metric snapshot when metrics.json is present."""

    path = path.expanduser().resolve()
    if not path.is_file():
        return {
            "path": path,
            "exists": False,
            "sha256": None,
            "snapshot": {},
            "test_local_evaluated": None,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HPBError(f"Cannot read metrics file {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise HPBError(f"Metrics file must contain a JSON object: {path}")

    test_local_evaluated = payload.get("test_local_evaluated")
    if test_local_evaluated is True:
        raise HPBError(
            f"{experiment} seed={seed}: metrics reports test_local_evaluated=true"
        )

    best = payload.get("best_dev_metrics")
    metric_map = best if isinstance(best, Mapping) else payload
    miou_candidates: Iterable[Any] = (
        metric_map.get("mIoU"),
        payload.get("best_dev_mIoU"),
        payload.get("mIoU"),
    )
    miou: Optional[float] = None
    for value in miou_candidates:
        if value is not None:
            miou = _finite_float(
                value, context=f"{experiment} seed={seed} metrics mIoU"
            )
            break

    snapshot: Dict[str, Any] = {
        "mIoU": miou,
        "mAcc": _finite_float(
            metric_map.get("mAcc"), context=f"{experiment} seed={seed} mAcc"
        ),
        "pixel_accuracy": _finite_float(
            metric_map.get("pixel_accuracy"),
            context=f"{experiment} seed={seed} pixel_accuracy",
        ),
        "small_object_mIoU": _finite_float(
            metric_map.get("small_object_mIoU"),
            context=f"{experiment} seed={seed} small_object_mIoU",
        ),
        "boundary_f1": _finite_float(
            metric_map.get("boundary_f1"),
            context=f"{experiment} seed={seed} boundary_f1",
        ),
    }
    per_class = metric_map.get("per_class")
    if isinstance(per_class, Mapping):
        snapshot["per_class_iou"] = {
            str(class_name): _finite_float(
                class_metrics.get("iou"),
                context=f"{experiment} seed={seed} {class_name} IoU",
            )
            for class_name, class_metrics in per_class.items()
            if isinstance(class_metrics, Mapping) and class_metrics.get("iou") is not None
        }
    else:
        snapshot["per_class_iou"] = {}

    return {
        "path": path,
        "exists": True,
        "sha256": sha256_file(path),
        "snapshot": snapshot,
        "test_local_evaluated": test_local_evaluated,
    }


def metric_deltas(
    left_metrics: Mapping[str, Any], right_metrics: Mapping[str, Any]
) -> Dict[str, Any]:
    deltas: Dict[str, Any] = {}
    for key in (
        "mIoU",
        "mAcc",
        "pixel_accuracy",
        "small_object_mIoU",
        "boundary_f1",
    ):
        left = left_metrics.get(key)
        right = right_metrics.get(key)
        deltas[key] = (
            float(left) - float(right)
            if left is not None and right is not None
            else None
        )

    left_classes = left_metrics.get("per_class_iou")
    right_classes = right_metrics.get("per_class_iou")
    per_class_delta: Dict[str, float] = {}
    if isinstance(left_classes, Mapping) and isinstance(right_classes, Mapping):
        for class_name in left_classes.keys() & right_classes.keys():
            left_value = left_classes[class_name]
            right_value = right_classes[class_name]
            if left_value is not None and right_value is not None:
                per_class_delta[str(class_name)] = float(left_value) - float(
                    right_value
                )
    deltas["per_class_iou"] = dict(sorted(per_class_delta.items()))
    return deltas


def comparison_id(comparison: Tuple[str, str]) -> str:
    return f"{comparison[0]}-{comparison[1]}"


def parse_comparison(value: str) -> Tuple[str, str]:
    normalized = value.strip().upper().replace("_VS_", "-").replace("VS", "-")
    normalized = normalized.replace(":", "-").replace("/", "-")
    parts = [part for part in normalized.split("-") if part]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid comparison {value!r}; use LEFT-RIGHT, for example H2-H0"
        )
    left_name, right_name = parts
    if left_name not in REGISTERED_EXPERIMENTS:
        raise argparse.ArgumentTypeError(
            f"Unknown H experiment {left_name!r}; expected one of "
            f"{REGISTERED_EXPERIMENTS}"
        )
    if right_name not in REGISTERED_EXPERIMENTS:
        raise argparse.ArgumentTypeError(
            f"Unknown H experiment {right_name!r}; expected one of "
            f"{REGISTERED_EXPERIMENTS}"
        )
    if left_name == right_name:
        raise argparse.ArgumentTypeError("A comparison must use two different models")
    return left_name, right_name


def _default_input_paths(
    h_root: Path, experiment: str, seed: int
) -> Tuple[Path, Path]:
    run_dir = h_root / experiment / f"seed_{seed}"
    return (
        run_dir / "dev_per_image_confusion.jsonl",
        run_dir / "metrics.json",
    )


def _input_audit(
    confusion: Mapping[str, Any], metrics: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "per_image_confusion_path": str(confusion["path"]),
        "per_image_confusion_sha256": confusion["sha256"],
        "per_image_confusion_size_bytes": int(confusion["size_bytes"]),
        "metrics_path": str(metrics["path"]),
        "metrics_exists": bool(metrics["exists"]),
        "metrics_sha256": metrics["sha256"],
        "test_local_evaluated": metrics["test_local_evaluated"],
    }


def run_seed(
    *,
    seed: int,
    h_root: Path,
    output_dir: Path,
    comparisons: Sequence[Tuple[str, str]],
    expected_images: Optional[int],
    repetitions: int,
    bootstrap_seed: int,
    batch_size: int,
) -> Dict[str, Dict[str, Any]]:
    started = time.perf_counter()
    experiments = tuple(
        dict.fromkeys(name for comparison in comparisons for name in comparison)
    )
    confusion_by_experiment: Dict[str, Dict[str, Any]] = {}
    metrics_by_experiment: Dict[str, Dict[str, Any]] = {}
    aggregate_miou: Dict[str, float] = {}
    aggregate_class_iou: Dict[str, np.ndarray] = {}

    for experiment in experiments:
        confusion_path, metrics_path = _default_input_paths(
            h_root, experiment, seed
        )
        confusion = load_per_image_confusion(
            confusion_path, expected_images=expected_images
        )
        metrics = load_metrics(metrics_path, experiment=experiment, seed=seed)
        aggregate_stats = confusion["stats"].sum(axis=0)
        per_class_iou = class_iou_from_statistics(aggregate_stats)
        miou = float(np.nanmean(per_class_iou))
        metric_miou = metrics["snapshot"].get("mIoU")
        if (
            metric_miou is not None
            and abs(float(metric_miou) - miou) > METRIC_TOLERANCE
        ):
            raise HPBError(
                f"{experiment} seed={seed}: metrics mIoU "
                f"{float(metric_miou):.17g} disagrees with per-image aggregate "
                f"{miou:.17g}"
            )
        confusion_by_experiment[experiment] = confusion
        metrics_by_experiment[experiment] = metrics
        aggregate_miou[experiment] = miou
        aggregate_class_iou[experiment] = per_class_iou

    alignment_by_comparison: Dict[str, Dict[str, Any]] = {}
    for left_name, right_name in comparisons:
        identifier = comparison_id((left_name, right_name))
        alignment_by_comparison[identifier] = validate_pair(
            confusion_by_experiment[left_name],
            confusion_by_experiment[right_name],
            left_name=left_name,
            right_name=right_name,
            seed=seed,
        )

    distributions = paired_bootstrap_deltas(
        {
            experiment: confusion_by_experiment[experiment]["stats"]
            for experiment in experiments
        },
        comparisons,
        repetitions=repetitions,
        random_seed=bootstrap_seed,
        batch_size=batch_size,
    )

    per_comparison: Dict[str, Dict[str, Any]] = {}
    for left_name, right_name in comparisons:
        identifier = comparison_id((left_name, right_name))
        deltas = distributions[identifier]
        q025 = _percentile(deltas, 2.5)
        q975 = _percentile(deltas, 97.5)
        comparison_dir = output_dir / identifier
        distribution_name = f"seed_{seed}_bootstrap_deltas.npy"
        distribution_path = comparison_dir / distribution_name
        write_npy_atomic(distribution_path, deltas)

        left_snapshot = dict(metrics_by_experiment[left_name]["snapshot"])
        right_snapshot = dict(metrics_by_experiment[right_name]["snapshot"])
        left_snapshot["per_image_aggregate_mIoU"] = aggregate_miou[left_name]
        right_snapshot["per_image_aggregate_mIoU"] = aggregate_miou[right_name]
        if not left_snapshot.get("per_class_iou"):
            left_snapshot["per_class_iou_by_index"] = [
                None if np.isnan(value) else float(value)
                for value in aggregate_class_iou[left_name]
            ]
        if not right_snapshot.get("per_class_iou"):
            right_snapshot["per_class_iou_by_index"] = [
                None if np.isnan(value) else float(value)
                for value in aggregate_class_iou[right_name]
            ]

        checkpoint_delta = (
            aggregate_miou[left_name] - aggregate_miou[right_name]
        )
        is_anchor_comparison = right_name == "H0" and left_name != "H0"
        summary = {
            "experiment": "H-PB",
            "comparison": identifier,
            "delta_definition": f"mIoU({left_name}) - mIoU({right_name})",
            "method": (
                "paired image-level bootstrap of aggregated 19x19 "
                "confusion matrices"
            ),
            "seed": int(seed),
            "num_images": int(
                len(confusion_by_experiment[left_name]["images"])
            ),
            "num_classes": NUM_CLASSES,
            "bootstrap_repetitions": int(repetitions),
            "bootstrap_random_seed": int(bootstrap_seed),
            "bootstrap_batch_size": int(batch_size),
            "rng": (
                "numpy.default_rng(PCG64).integers; identical image-index "
                "draws shared by all H models and comparisons within this seed"
            ),
            "left_experiment": left_name,
            "right_experiment": right_name,
            "left_metrics": left_snapshot,
            "right_metrics": right_snapshot,
            "checkpoint_miou_delta": float(checkpoint_delta),
            "reported_metric_deltas": metric_deltas(
                left_snapshot, right_snapshot
            ),
            "bootstrap_delta_mean": float(np.mean(deltas)),
            "bootstrap_delta_std": float(np.std(deltas, ddof=1)),
            "bootstrap_delta_min": float(np.min(deltas)),
            "bootstrap_delta_max": float(np.max(deltas)),
            "bootstrap_delta_percentile_2_5": q025,
            "bootstrap_delta_percentile_97_5": q975,
            "bootstrap_ci_95": [q025, q975],
            "bootstrap_ci_excludes_zero": bool(q025 > 0.0 or q975 < 0.0),
            "bootstrap_ci_supports_positive": bool(q025 > 0.0),
            "bootstrap_ci_supports_negative": bool(q975 < 0.0),
            "bootstrap_probability_delta_gt_zero": float(np.mean(deltas > 0.0)),
            "bootstrap_probability_delta_lt_zero": float(np.mean(deltas < 0.0)),
            "bootstrap_delta_distribution": str(
                Path(identifier) / distribution_name
            ),
            "screening_reference": {
                "applies": is_anchor_comparison,
                "name": "K1 three-seed sample standard deviation",
                "threshold": K1_THREE_SEED_SAMPLE_STD,
                "positive_delta_exceeds_threshold": (
                    bool(checkpoint_delta > K1_THREE_SEED_SAMPLE_STD)
                    if is_anchor_comparison
                    else None
                ),
            },
            "left_input": _input_audit(
                confusion_by_experiment[left_name],
                metrics_by_experiment[left_name],
            ),
            "right_input": _input_audit(
                confusion_by_experiment[right_name],
                metrics_by_experiment[right_name],
            ),
            "alignment": alignment_by_comparison[identifier],
            "test_local_evaluated": False,
            "elapsed_seconds_for_seed_all_comparisons": float(
                time.perf_counter() - started
            ),
        }
        write_json_atomic(comparison_dir / f"seed_{seed}.json", summary)
        per_comparison[identifier] = summary
    return per_comparison


def _delta_summary(
    per_seed: Mapping[str, Mapping[str, Mapping[str, Any]]],
    comparison: str,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    values = np.asarray(
        [
            per_seed[str(seed)][comparison]["checkpoint_miou_delta"]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    return {
        "values_by_seed": {
            str(seed): float(
                per_seed[str(seed)][comparison]["checkpoint_miou_delta"]
            )
            for seed in seeds
        },
        "mean": float(np.mean(values)),
        "sample_std": (
            float(np.std(values, ddof=1)) if len(values) > 1 else None
        ),
        "std_ddof": 1,
        "num_seeds": len(values),
        "all_positive": bool(np.all(values > 0.0)),
        "all_negative": bool(np.all(values < 0.0)),
        "same_nonzero_direction": bool(
            np.all(values > 0.0) or np.all(values < 0.0)
        ),
    }


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = tuple(int(seed) for seed in args.seeds)
    invalid_seeds = [seed for seed in seeds if seed not in FORMAL_SEEDS]
    if invalid_seeds:
        raise HPBError(
            f"H-PB seeds must be registered values {FORMAL_SEEDS}; "
            f"got {invalid_seeds}"
        )
    if not seeds:
        raise HPBError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise HPBError("H-PB seed list contains duplicates")

    comparisons = tuple(args.comparisons)
    if not comparisons:
        raise HPBError("At least one comparison is required")
    identifiers = [comparison_id(pair) for pair in comparisons]
    if len(set(identifiers)) != len(identifiers):
        raise HPBError("H-PB comparison list contains duplicates")

    repetitions = int(args.bootstrap_repetitions)
    if args.smoke_test:
        repetitions = min(repetitions, 1_000)
    h_root = args.h_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for seed in seeds:
        print(
            f"[H-PB] seed={seed}: loading and auditing "
            f"{len(comparisons)} comparison(s)"
        )
        per_seed[str(seed)] = run_seed(
            seed=seed,
            h_root=h_root,
            output_dir=output_dir,
            comparisons=comparisons,
            expected_images=(
                None if int(args.expected_images) <= 0 else int(args.expected_images)
            ),
            repetitions=repetitions,
            bootstrap_seed=int(args.bootstrap_seed),
            batch_size=int(args.batch_size),
        )
        for identifier in identifiers:
            result = per_seed[str(seed)][identifier]
            print(
                f"[H-PB] seed={seed} {identifier}: "
                f"delta={result['checkpoint_miou_delta']:+.9f}, "
                f"bootstrap CI=[{result['bootstrap_ci_95'][0]:+.9f}, "
                f"{result['bootstrap_ci_95'][1]:+.9f}]"
            )

    comparison_summaries: Dict[str, Any] = {}
    for identifier in identifiers:
        comparison_summaries[identifier] = {
            "matched_seed_delta": _delta_summary(
                per_seed, identifier, seeds
            ),
            "bootstrap_ci_by_seed": {
                str(seed): per_seed[str(seed)][identifier]["bootstrap_ci_95"]
                for seed in seeds
            },
            "per_seed_json": {
                str(seed): str(Path(identifier) / f"seed_{seed}.json")
                for seed in seeds
            },
            "combined_bootstrap": {
                "computed": False,
                "reason": (
                    "training-seed uncertainty and image-resampling "
                    "uncertainty remain separate; training seeds are not mixed"
                ),
            },
        }

    summary = {
        "experiment": "H-PB",
        "protocol": {
            "description": (
                "H0-H3 same-seed paired bootstrap on frozen dev_local "
                "per-image confusion matrices"
            ),
            "training_performed": False,
            "h_root": str(h_root),
            "seeds": [int(seed) for seed in seeds],
            "screening_seed": SCREENING_SEED,
            "formal_seeds": list(FORMAL_SEEDS),
            "comparisons": identifiers,
            "delta_orientation": "left experiment minus right experiment",
            "expected_images": int(args.expected_images),
            "bootstrap_repetitions": repetitions,
            "bootstrap_random_seed": int(args.bootstrap_seed),
            "bootstrap_batch_size": int(args.batch_size),
            "bootstrap_unit": "image; identical paired indices within a seed",
            "aggregation": (
                "sum 19x19 confusion matrices, then nanmean of 19 class IoUs"
            ),
            "different_training_seeds_mixed": False,
            "test_local_evaluated": False,
        },
        "screening_reference": {
            "name": "K1 three-seed sample standard deviation",
            "value": K1_THREE_SEED_SAMPLE_STD,
            "scope": "candidate-H0 screening comparisons",
        },
        "comparisons": comparison_summaries,
        "per_seed": per_seed,
        "output_directory": str(output_dir),
        "test_local_evaluated": False,
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "H-PB: paired bootstrap of existing H0-H3 per-image dev "
            "confusion matrices; no model training is performed."
        )
    )
    parser.add_argument("--h-root", type=Path, default=DEFAULT_H_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[SCREENING_SEED],
        help=(
            "Matched training seeds. Default: 42. For a completed extension "
            "use: --seeds 42 3407 260805"
        ),
    )
    parser.add_argument(
        "--comparisons",
        type=parse_comparison,
        nargs="+",
        default=list(DEFAULT_COMPARISONS),
        metavar="LEFT-RIGHT",
        help=(
            "mIoU contrasts with left-minus-right orientation. Default: "
            "H1-H0 H2-H0 H3-H0 H2-H1 H3-H2"
        ),
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        "--repetitions",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BOOTSTRAP_BATCH_SIZE,
        help="Number of bootstrap resamples processed at once.",
    )
    parser.add_argument(
        "--expected-images",
        type=int,
        default=DEFAULT_EXPECTED_IMAGES,
        help="Expected dev image count; use 0 to disable the count check.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run at most 1,000 resamples per seed for a quick audit.",
    )
    args = parser.parse_args(argv)
    if args.bootstrap_repetitions <= 0:
        parser.error("--bootstrap-repetitions must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.expected_images < 0:
        parser.error("--expected-images must be non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run_all(args)
    except (HPBError, ValueError) as exc:
        print(f"[H-PB][ERROR] {exc}")
        return 2
    print(
        f"[H-PB] completed {len(args.seeds)} seed(s); results: "
        f"{args.output_dir.expanduser().resolve() / 'summary.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
