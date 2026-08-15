#!/usr/bin/env python3
"""Create or verify locked Cityscapes local split manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


SPLITS: Mapping[str, Mapping[str, object]] = {
    "train_local": {
        "source": "official train",
        "count": 2530,
        "cities": [
            "aachen",
            "bochum",
            "bremen",
            "cologne",
            "dusseldorf",
            "erfurt",
            "hamburg",
            "hanover",
            "monchengladbach",
            "strasbourg",
            "stuttgart",
            "tubingen",
            "ulm",
            "zurich",
        ],
    },
    "dev_local": {
        "source": "official train",
        "count": 445,
        "cities": ["darmstadt", "jena", "krefeld", "weimar"],
    },
    "test_local": {
        "source": "official val",
        "count": 500,
        "cities": ["frankfurt", "lindau", "munster"],
    },
}

LOCK_FILENAME = "local_splits.lock.json"
CHECKSUM_FILENAME = "local_splits.sha256"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unique_by_id(paths: Iterable[Path], suffix: str, split: str) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for path in paths:
        if not path.name.endswith(suffix):
            continue
        sample_id = path.name[: -len(suffix)]
        if sample_id in result:
            raise ValueError(f"Duplicate sample ID in {split}: {sample_id}")
        result[sample_id] = path
    return result


def build_split(dataset_root: Path, split: str) -> Tuple[bytes, Dict[str, object], set[str]]:
    spec = SPLITS[split]
    image_root = dataset_root / "leftImg8bit" / split
    label_root = dataset_root / "gtFine" / split
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(f"Missing split directories for {split}")

    images = unique_by_id(
        image_root.glob("*/*_leftImg8bit.png"), "_leftImg8bit.png", split
    )
    labels = unique_by_id(
        label_root.glob("*/*_gtFine_labelIds.png"), "_gtFine_labelIds.png", split
    )
    missing_labels = sorted(set(images) - set(labels))
    orphan_labels = sorted(set(labels) - set(images))
    if missing_labels or orphan_labels:
        raise ValueError(
            f"Pairing failure for {split}: missing_labels={missing_labels[:5]}, "
            f"orphan_labels={orphan_labels[:5]}"
        )

    expected_count = int(spec["count"])
    if len(images) != expected_count:
        raise ValueError(
            f"Count mismatch for {split}: expected {expected_count}, found {len(images)}"
        )

    actual_cities = sorted({path.parent.name for path in images.values()})
    expected_cities = sorted(str(city) for city in spec["cities"])
    if actual_cities != expected_cities:
        raise ValueError(
            f"City mismatch for {split}: expected {expected_cities}, found {actual_cities}"
        )

    lines: List[str] = []
    for sample_id in sorted(images):
        image = images[sample_id]
        label = labels[sample_id]
        if image.parent.name != label.parent.name:
            raise ValueError(
                f"Image/label city mismatch for {sample_id}: "
                f"{image.parent.name} != {label.parent.name}"
            )

        label_prefix = label.name[: -len("_gtFine_labelIds.png")]
        required_annotations = [
            label.with_name(f"{label_prefix}_gtFine_color.png"),
            label.with_name(f"{label_prefix}_gtFine_instanceIds.png"),
            label.with_name(f"{label_prefix}_gtFine_polygons.json"),
        ]
        missing_annotations = [
            path.name for path in required_annotations if not path.is_file()
        ]
        if missing_annotations:
            raise FileNotFoundError(
                f"Missing annotations for {sample_id}: {missing_annotations}"
            )

        image_rel = image.relative_to(dataset_root).as_posix()
        label_rel = label.relative_to(dataset_root).as_posix()
        lines.append(f"{image_rel}\t{label_rel}")

    payload = ("\n".join(lines) + "\n").encode("utf-8")
    metadata: Dict[str, object] = {
        "source": spec["source"],
        "count": len(lines),
        "cities": actual_cities,
        "manifest": f"{split}.txt",
        "sha256": sha256_bytes(payload),
    }
    return payload, metadata, set(images)


def build_artifacts(
    dataset_root: Path, lock_date: str, timezone: str
) -> Tuple[Dict[str, bytes], Dict[str, object]]:
    manifests: Dict[str, bytes] = {}
    split_metadata: Dict[str, object] = {}
    split_ids: Dict[str, set[str]] = {}

    for split in SPLITS:
        payload, metadata, sample_ids = build_split(dataset_root, split)
        manifests[f"{split}.txt"] = payload
        split_metadata[split] = metadata
        split_ids[split] = sample_ids

    split_names = list(SPLITS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = sorted(split_ids[left] & split_ids[right])
            if overlap:
                raise ValueError(
                    f"Sample overlap between {left} and {right}: {overlap[:5]}"
                )

    combined = hashlib.sha256()
    for filename in sorted(manifests):
        combined.update(filename.encode("utf-8"))
        combined.update(b"\0")
        combined.update(manifests[filename])

    lock: Dict[str, object] = {
        "schema_version": 1,
        "lock_date": lock_date,
        "timezone": timezone,
        "dataset_root": "datasets/cityscapes",
        "official_test": "omitted from the local protocol",
        "manifest_format": (
            "UTF-8 without BOM, LF endings, one tab-separated "
            "image-relative-path and label-relative-path pair per line"
        ),
        "sha256_scope": (
            "Manifest file bytes. This locks split membership and image/label "
            "pairing, not PNG or JSON file contents."
        ),
        "rule_zh": {
            "train_local": (
                "官方 train 中除 darmstadt、jena、krefeld、weimar 外的 14 个城市"
            ),
            "dev_local": "官方 train 中的 darmstadt、jena、krefeld、weimar",
            "test_local": "官方 val 中的 frankfurt、lindau、munster",
            "official_test": "暂不纳入本地协议，不生成清单",
        },
        "splits": split_metadata,
        "combined_manifest_sha256": combined.hexdigest(),
        "generator": "scripts/lock_cityscapes_local_splits.py",
    }
    return manifests, lock


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def checksum_bytes(manifests: Mapping[str, bytes], lock_payload: bytes) -> bytes:
    lines = [
        f"{sha256_bytes(payload)}  {filename}"
        for filename, payload in sorted(manifests.items())
    ]
    lines.append(f"{sha256_bytes(lock_payload)}  {LOCK_FILENAME}")
    return ("\n".join(lines) + "\n").encode("ascii")


def atomic_write(path: Path, payload: bytes) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_bytes(payload)
    os.replace(temp_path, path)


def write_artifacts(
    dataset_root: Path, manifests: Mapping[str, bytes], lock: Mapping[str, object]
) -> None:
    lock_payload = json_bytes(lock)
    for filename, payload in manifests.items():
        atomic_write(dataset_root / filename, payload)
    atomic_write(dataset_root / LOCK_FILENAME, lock_payload)
    atomic_write(
        dataset_root / CHECKSUM_FILENAME,
        checksum_bytes(manifests, lock_payload),
    )


def verify_artifacts(
    dataset_root: Path, manifests: Mapping[str, bytes], lock: Mapping[str, object]
) -> None:
    expected_lock = json_bytes(lock)
    expected_files = dict(manifests)
    expected_files[LOCK_FILENAME] = expected_lock
    expected_files[CHECKSUM_FILENAME] = checksum_bytes(manifests, expected_lock)
    failures: List[str] = []
    for filename, expected in expected_files.items():
        path = dataset_root / filename
        if not path.is_file():
            failures.append(f"missing {filename}")
        elif path.read_bytes() != expected:
            failures.append(f"content mismatch: {filename}")
    if failures:
        raise ValueError("Lock verification failed: " + "; ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("write", "verify"))
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("datasets/cityscapes")
    )
    parser.add_argument("--lock-date")
    parser.add_argument("--timezone")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if args.mode == "verify" and (args.lock_date is None or args.timezone is None):
        lock_path = dataset_root / LOCK_FILENAME
        if not lock_path.is_file():
            raise FileNotFoundError(f"Missing lock metadata: {lock_path}")
        saved_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_date = args.lock_date or str(saved_lock["lock_date"])
        timezone = args.timezone or str(saved_lock["timezone"])
    else:
        lock_date = args.lock_date or date.today().isoformat()
        timezone = args.timezone or "Asia/Tokyo"

    manifests, lock = build_artifacts(dataset_root, lock_date, timezone)
    if args.mode == "write":
        write_artifacts(dataset_root, manifests, lock)
    else:
        verify_artifacts(dataset_root, manifests, lock)

    for split in SPLITS:
        metadata = lock["splits"][split]
        print(f"{split}: {metadata['count']} samples, sha256={metadata['sha256']}")
    print(f"combined_manifest_sha256={lock['combined_manifest_sha256']}")
    print(f"mode={args.mode}: OK")


if __name__ == "__main__":
    main()
