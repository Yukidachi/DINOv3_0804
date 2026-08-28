"""D1-PB: paired-bootstrap evaluation for the three D1/K1 seed pairs.

This entry point deliberately does *not* train or load a model.  It consumes
the already-frozen per-image development confusion matrices produced by D1
and its matched K1 run, and estimates the image-level uncertainty of
``mIoU(D1) - mIoU(K1)``.

The formal protocol is fixed by the experiment plans:

* seeds: ``42, 3407, 260805``;
* 445 paired ``dev_local`` images per seed;
* 100,000 paired bootstrap resamples per seed;
* bootstrap RNG seed: ``260820``;
* each resample aggregates the 19x19 confusion matrices first, then computes
  the mean of the 19 class IoUs (never the mean of per-image mIoUs).

The default output directory is
``result/D_MobileNetV2_RASPP_server/D1-PB``.  Each seed receives a JSON
summary and a ``.npy`` delta distribution; ``summary.json`` contains the
three-seed training-delta summary and all input/audit metadata.
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
DEFAULT_D1_ROOT = SCRIPT_DIR / "result" / "D_MobileNetV2_RASPP_server" / "D1"
DEFAULT_K1_ROOT = SCRIPT_DIR / "result" / "K_MobileNetV2_RASPP_server" / "K1"
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "result" / "D_MobileNetV2_RASPP_server" / "D1-PB"
)

FORMAL_SEEDS = (42, 3407, 260805)
NUM_CLASSES = 19
DEFAULT_EXPECTED_IMAGES = 445
DEFAULT_BOOTSTRAP_REPETITIONS = 100_000
DEFAULT_BOOTSTRAP_SEED = 260820
DEFAULT_BOOTSTRAP_BATCH_SIZE = 4096
METRIC_TOLERANCE = 1e-12


class D1PBError(RuntimeError):
    """Raised when a D1/K1 paired-bootstrap input fails an audit."""


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
    except Exception as exc:  # pragma: no cover - defensive error context
        raise D1PBError(f"{path}:{line_number}: invalid confusion_matrix") from exc
    if array.shape != (NUM_CLASSES, NUM_CLASSES):
        raise D1PBError(
            f"{path}:{line_number}: expected a {NUM_CLASSES}x{NUM_CLASSES} "
            f"confusion matrix, got shape {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        numeric = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise D1PBError(f"{path}:{line_number}: confusion counts must be integers")
    array = np.asarray(array, dtype=np.int64)
    if np.any(array < 0):
        raise D1PBError(f"{path}:{line_number}: confusion counts cannot be negative")
    return array


def load_per_image_confusion(
    path: Path, *, expected_images: Optional[int] = DEFAULT_EXPECTED_IMAGES
) -> Dict[str, Any]:
    """Load and audit one ``dev_per_image_confusion.jsonl`` file.

    The returned ``stats`` contains only row sums, column sums and diagonal
    counts.  These 57 sufficient statistics reproduce mIoU exactly while
    making 100,000 bootstrap resamples practical.
    """

    path = path.expanduser().resolve()
    if not path.is_file():
        raise D1PBError(f"Missing per-image confusion file: {path}")
    images: List[str] = []
    valid_pixels: List[int] = []
    matrices: List[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise D1PBError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, Mapping):
                raise D1PBError(f"{path}:{line_number}: JSON row must be an object")
            image = row.get("image")
            if not isinstance(image, str) or not image:
                raise D1PBError(f"{path}:{line_number}: missing non-empty image name")
            if image in images:
                raise D1PBError(f"{path}:{line_number}: duplicate image name {image!r}")
            if "valid_pixels" not in row:
                raise D1PBError(f"{path}:{line_number}: missing valid_pixels")
            try:
                valid = int(row["valid_pixels"])
            except (TypeError, ValueError) as exc:
                raise D1PBError(f"{path}:{line_number}: invalid valid_pixels") from exc
            if valid < 0:
                raise D1PBError(f"{path}:{line_number}: valid_pixels cannot be negative")
            matrix = _coerce_confusion(
                row.get("confusion_matrix"), path=path, line_number=line_number
            )
            matrix_sum = int(matrix.sum())
            if matrix_sum != valid:
                raise D1PBError(
                    f"{path}:{line_number}: valid_pixels={valid} does not match "
                    f"confusion sum={matrix_sum}"
                )
            images.append(image)
            valid_pixels.append(valid)
            matrices.append(matrix)

    if not matrices:
        raise D1PBError(f"{path}: file contains no image rows")
    if expected_images is not None and len(matrices) != expected_images:
        raise D1PBError(
            f"{path}: expected {expected_images} images, found {len(matrices)}"
        )
    matrix_stack = np.stack(matrices, axis=0)
    row_sums = matrix_stack.sum(axis=2, dtype=np.int64)
    column_sums = matrix_stack.sum(axis=1, dtype=np.int64)
    diagonal = np.diagonal(matrix_stack, axis1=1, axis2=2).copy()
    stats = np.concatenate((row_sums, column_sums, diagonal), axis=1)
    return {
        "path": path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "images": images,
        "valid_pixels": np.asarray(valid_pixels, dtype=np.int64),
        "matrices": matrix_stack,
        "stats": stats,
    }


def validate_pair(
    d1: Mapping[str, Any], k1: Mapping[str, Any], *, seed: int
) -> Dict[str, Any]:
    d1_images = list(d1["images"])
    k1_images = list(k1["images"])
    if d1_images != k1_images:
        first_difference = next(
            (
                index
                for index, (left, right) in enumerate(zip(d1_images, k1_images))
                if left != right
            ),
            min(len(d1_images), len(k1_images)),
        )
        raise D1PBError(
            f"seed={seed}: D1/K1 image order or set mismatch at index "
            f"{first_difference}: {d1_images[first_difference:first_difference + 1]} "
            f"vs {k1_images[first_difference:first_difference + 1]}"
        )
    if not np.array_equal(d1["valid_pixels"], k1["valid_pixels"]):
        index = int(np.flatnonzero(d1["valid_pixels"] != k1["valid_pixels"])[0])
        raise D1PBError(
            f"seed={seed}: valid_pixels mismatch for {d1_images[index]!r}: "
            f"D1={int(d1['valid_pixels'][index])}, K1={int(k1['valid_pixels'][index])}"
        )
    if d1["stats"].shape != k1["stats"].shape:
        raise D1PBError(f"seed={seed}: D1/K1 sufficient-statistic shapes differ")
    return {
        "seed": int(seed),
        "paired": True,
        "image_count": len(d1_images),
        "order_equal": True,
        "image_set_equal": True,
        "valid_pixels_equal": True,
        "class_count": NUM_CLASSES,
        "image_name_sha256": hashlib.sha256(
            "\n".join(d1_images).encode("utf-8")
        ).hexdigest(),
    }


def miou_from_statistics(statistics: np.ndarray) -> np.ndarray:
    """Compute mIoU from [..., 57] row/column/diagonal statistics."""

    values = np.asarray(statistics, dtype=np.float64)
    if values.shape[-1] != NUM_CLASSES * 3:
        raise ValueError(
            f"Expected the final dimension to be {NUM_CLASSES * 3}, got {values.shape}"
        )
    rows = values[..., :NUM_CLASSES]
    columns = values[..., NUM_CLASSES : 2 * NUM_CLASSES]
    diagonal = values[..., 2 * NUM_CLASSES :]
    union = rows + columns - diagonal
    iou = np.full(union.shape, np.nan, dtype=np.float64)
    np.divide(diagonal, union, out=iou, where=union > 0)
    with np.errstate(invalid="ignore"):
        return np.nanmean(iou, axis=-1)


def _percentile(values: np.ndarray, percentile: float) -> float:
    try:
        return float(np.percentile(values, percentile, method="linear"))
    except TypeError:  # NumPy < 1.22 compatibility
        return float(np.percentile(values, percentile, interpolation="linear"))


def paired_bootstrap_deltas(
    d1_stats: np.ndarray,
    k1_stats: np.ndarray,
    *,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    random_seed: int = DEFAULT_BOOTSTRAP_SEED,
    batch_size: int = DEFAULT_BOOTSTRAP_BATCH_SIZE,
) -> np.ndarray:
    """Return paired bootstrap mIoU deltas using direct image-index draws.

    Counts are only an implementation optimization: each row is equivalent to
    drawing ``n`` image indices with replacement and using those same indices
    for D1 and K1.  Direct index generation makes the fixed RNG protocol easy
    to audit and reproduce.
    """

    d1_values = np.asarray(d1_stats, dtype=np.int64)
    k1_values = np.asarray(k1_stats, dtype=np.int64)
    if d1_values.ndim != 2 or k1_values.ndim != 2 or d1_values.shape != k1_values.shape:
        raise ValueError("D1/K1 statistics must have equal shape [N, 57]")
    n_images = d1_values.shape[0]
    if n_images == 0:
        raise ValueError("Cannot bootstrap an empty image set")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    rng = np.random.default_rng(int(random_seed))
    deltas = np.empty(int(repetitions), dtype=np.float64)
    row_numbers = None
    for start in range(0, int(repetitions), int(batch_size)):
        count = min(int(batch_size), int(repetitions) - start)
        indices = rng.integers(0, n_images, size=(count, n_images), dtype=np.int64)
        counts = np.zeros((count, n_images), dtype=np.int16)
        if row_numbers is None or row_numbers.shape != indices.shape:
            row_numbers = np.broadcast_to(
                np.arange(count, dtype=np.int64)[:, None], indices.shape
            )
        np.add.at(counts, (row_numbers, indices), 1)
        d1_aggregated = counts @ d1_values
        k1_aggregated = counts @ k1_values
        deltas[start : start + count] = (
            miou_from_statistics(d1_aggregated)
            - miou_from_statistics(k1_aggregated)
        )
    return deltas


def _metric_miou(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D1PBError(f"Cannot read metrics file {path}: {exc}") from exc
    candidates: Iterable[Any] = (
        payload.get("best_dev_metrics", {}).get("mIoU")
        if isinstance(payload.get("best_dev_metrics"), Mapping)
        else None,
        payload.get("best_dev_mIoU"),
        payload.get("mIoU"),
    )
    for value in candidates:
        if value is not None:
            try:
                result = float(value)
            except (TypeError, ValueError) as exc:
                raise D1PBError(f"Invalid mIoU in {path}: {value!r}") from exc
            if not np.isfinite(result):
                raise D1PBError(f"Non-finite mIoU in {path}: {result}")
            return result
    return None


def _default_input_paths(
    d1_root: Path, k1_root: Path, seed: int
) -> Tuple[Path, Path, Path, Path]:
    d1_run = d1_root / f"seed_{seed}_lambda_0.1"
    k1_run = k1_root / f"seed_{seed}"
    return (
        d1_run / "dev_per_image_confusion.jsonl",
        k1_run / "dev_per_image_confusion.jsonl",
        d1_run / "metrics.json",
        k1_run / "metrics.json",
    )


def run_seed(
    *,
    seed: int,
    d1_path: Path,
    k1_path: Path,
    d1_metrics_path: Path,
    k1_metrics_path: Path,
    output_dir: Path,
    expected_images: Optional[int],
    repetitions: int,
    bootstrap_seed: int,
    batch_size: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    d1 = load_per_image_confusion(d1_path, expected_images=expected_images)
    k1 = load_per_image_confusion(k1_path, expected_images=expected_images)
    alignment = validate_pair(d1, k1, seed=seed)
    d1_aggregate_miou = float(miou_from_statistics(d1["stats"].sum(axis=0)))
    k1_aggregate_miou = float(miou_from_statistics(k1["stats"].sum(axis=0)))
    d1_metric = _metric_miou(d1_metrics_path)
    k1_metric = _metric_miou(k1_metrics_path)
    if d1_metric is not None and abs(d1_metric - d1_aggregate_miou) > METRIC_TOLERANCE:
        raise D1PBError(
            f"seed={seed}: D1 metrics mIoU {d1_metric:.17g} disagrees with "
            f"per-image aggregate {d1_aggregate_miou:.17g}"
        )
    if k1_metric is not None and abs(k1_metric - k1_aggregate_miou) > METRIC_TOLERANCE:
        raise D1PBError(
            f"seed={seed}: K1 metrics mIoU {k1_metric:.17g} disagrees with "
            f"per-image aggregate {k1_aggregate_miou:.17g}"
        )
    d1_checkpoint_miou = d1_metric if d1_metric is not None else d1_aggregate_miou
    k1_checkpoint_miou = k1_metric if k1_metric is not None else k1_aggregate_miou

    deltas = paired_bootstrap_deltas(
        d1["stats"],
        k1["stats"],
        repetitions=repetitions,
        random_seed=bootstrap_seed,
        batch_size=batch_size,
    )
    q025 = _percentile(deltas, 2.5)
    q975 = _percentile(deltas, 97.5)
    distribution_name = f"seed_{seed}_bootstrap_deltas.npy"
    distribution_path = output_dir / distribution_name
    write_npy_atomic(distribution_path, deltas)
    summary = {
        "experiment": "D1-PB",
        "method": "paired image-level bootstrap of aggregated 19x19 confusion matrices",
        "seed": int(seed),
        "num_images": int(len(d1["images"])),
        "num_classes": NUM_CLASSES,
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_random_seed": int(bootstrap_seed),
        "bootstrap_batch_size": int(batch_size),
        "rng": "numpy.default_rng(PCG64).integers; paired indices shared by D1/K1",
        "d1_miou": float(d1_checkpoint_miou),
        "k1_miou": float(k1_checkpoint_miou),
        "d1_per_image_aggregate_miou": d1_aggregate_miou,
        "k1_per_image_aggregate_miou": k1_aggregate_miou,
        "checkpoint_miou_delta": float(d1_checkpoint_miou - k1_checkpoint_miou),
        "bootstrap_delta_mean": float(np.mean(deltas)),
        "bootstrap_delta_std": float(np.std(deltas, ddof=1)),
        "bootstrap_delta_min": float(np.min(deltas)),
        "bootstrap_delta_max": float(np.max(deltas)),
        "bootstrap_delta_percentile_2_5": q025,
        "bootstrap_delta_percentile_97_5": q975,
        "bootstrap_ci_95": [q025, q975],
        "bootstrap_ci_excludes_zero": bool(q025 > 0.0 or q975 < 0.0),
        "bootstrap_delta_distribution": distribution_name,
        "d1_input": {
            "path": str(d1["path"]),
            "sha256": d1["sha256"],
            "size_bytes": int(d1["size_bytes"]),
            "metrics_path": str(d1_metrics_path.expanduser().resolve()),
            "metrics_sha256": (
                sha256_file(d1_metrics_path.expanduser().resolve())
                if d1_metrics_path.expanduser().is_file()
                else None
            ),
        },
        "k1_input": {
            "path": str(k1["path"]),
            "sha256": k1["sha256"],
            "size_bytes": int(k1["size_bytes"]),
            "metrics_path": str(k1_metrics_path.expanduser().resolve()),
            "metrics_sha256": (
                sha256_file(k1_metrics_path.expanduser().resolve())
                if k1_metrics_path.expanduser().is_file()
                else None
            ),
        },
        "alignment": alignment,
        "test_local_evaluated": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json_atomic(output_dir / f"seed_{seed}.json", summary)
    return summary


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = tuple(int(seed) for seed in args.seeds)
    invalid = [seed for seed in seeds if seed not in FORMAL_SEEDS]
    if invalid:
        raise D1PBError(
            "D1-PB seeds must be registered values "
            f"{FORMAL_SEEDS}; got {invalid}"
        )
    if len(set(seeds)) != len(seeds):
        raise D1PBError("D1-PB seed list contains duplicates")
    if not seeds:
        raise D1PBError("At least one seed is required")
    repetitions = int(args.bootstrap_repetitions)
    if args.smoke_test:
        repetitions = min(repetitions, 1_000)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed: Dict[str, Dict[str, Any]] = {}
    for seed in seeds:
        d1_path, k1_path, d1_metrics, k1_metrics = _default_input_paths(
            args.d1_root.expanduser().resolve(), args.k1_root.expanduser().resolve(), seed
        )
        print(f"[D1-PB] seed={seed}: loading and auditing paired inputs")
        per_seed[str(seed)] = run_seed(
            seed=seed,
            d1_path=d1_path,
            k1_path=k1_path,
            d1_metrics_path=d1_metrics,
            k1_metrics_path=k1_metrics,
            output_dir=output_dir,
            expected_images=(
                None if int(args.expected_images) <= 0 else int(args.expected_images)
            ),
            repetitions=repetitions,
            bootstrap_seed=int(args.bootstrap_seed),
            batch_size=int(args.batch_size),
        )
        result = per_seed[str(seed)]
        print(
            f"[D1-PB] seed={seed}: delta={result['checkpoint_miou_delta']:+.9f}, "
            f"bootstrap CI=[{result['bootstrap_ci_95'][0]:+.9f}, "
            f"{result['bootstrap_ci_95'][1]:+.9f}]"
        )

    deltas = np.asarray(
        [per_seed[str(seed)]["checkpoint_miou_delta"] for seed in seeds], dtype=np.float64
    )
    summary = {
        "experiment": "D1-PB",
        "protocol": {
            "description": "D1 versus matched same-seed K1 paired-bootstrap on dev_local",
            "training_performed": False,
            "seeds": [int(seed) for seed in seeds],
            "expected_images": int(args.expected_images),
            "bootstrap_repetitions": repetitions,
            "bootstrap_random_seed": int(args.bootstrap_seed),
            "bootstrap_batch_size": int(args.batch_size),
            "bootstrap_unit": "image; paired D1/K1 indices",
            "aggregation": "sum 19x19 confusion matrices, then nanmean of 19 IoUs",
            "test_local_evaluated": False,
        },
        "training_delta": {
            "values_by_seed": {
                str(seed): float(per_seed[str(seed)]["checkpoint_miou_delta"])
                for seed in seeds
            },
            "mean": float(np.mean(deltas)),
            "sample_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
            "std_ddof": 1,
            "num_seeds": len(deltas),
        },
        "per_seed": per_seed,
        "bootstrap_ci_by_seed": {
            str(seed): per_seed[str(seed)]["bootstrap_ci_95"] for seed in seeds
        },
        "combined_bootstrap": {
            "computed": False,
            "reason": "seed uncertainty and image-resampling uncertainty remain separate",
        },
        "output_directory": str(output_dir),
        "test_local_evaluated": False,
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "D1-PB: paired bootstrap of existing D1/K1 per-image confusion "
            "matrices; no model training is performed."
        )
    )
    parser.add_argument("--d1-root", type=Path, default=DEFAULT_D1_ROOT)
    parser.add_argument("--k1-root", type=Path, default=DEFAULT_K1_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
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
        "--batch-size", type=int, default=DEFAULT_BOOTSTRAP_BATCH_SIZE,
        help="Number of bootstrap resamples processed at once.",
    )
    parser.add_argument(
        "--expected-images", type=int, default=DEFAULT_EXPECTED_IMAGES,
        help="Expected dev image count; use 0 to disable the count check.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
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
    except D1PBError as exc:
        print(f"[D1-PB][ERROR] {exc}")
        return 2
    print(
        f"[D1-PB] completed {len(args.seeds)} seed(s); "
        f"results: {(args.output_dir.expanduser().resolve() / 'summary.json')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
