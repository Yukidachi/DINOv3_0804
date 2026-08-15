"""T0 baseline: frozen DINOv3 ConvNeXt-T + trainable R-ASPP on Cityscapes.

Protocol implemented from:

* ``知识蒸馏实验分析与后续实验方向.md``
* ``plan_markdown/Cityscapes知识蒸馏实验详单.md``

T0 freezes the complete DINOv3 backbone, trains only a 19-class R-ASPP head
on ``train_local``, and selects the best artifact using ``dev_local`` mIoU.
``test_local`` is validated as part of the locked split protocol but is never
loaded or evaluated by this training entry point.

Typical commands (run in the ``pytorch`` conda environment):

    python -B dino.py --smoke-test --seed 42
    python -B dino.py --seed 42
    python -B dino.py --seed 3407
    python -B dino.py --seed 260805
    python -B dino.py --verify-only --seed 42

Future K2/K3 code should call ``load_teacher_for_distillation`` instead of
constructing a new segmentation head.
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
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# Set before CUDA/CUBLAS initialization.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "datasets" / "cityscapes"
DEFAULT_REPO_DIR = SCRIPT_DIR / "dinov3-main"
DEFAULT_WEIGHTS_PATH = SCRIPT_DIR / "models" / "dinov3_convnext_t.pth"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "T0_DINOv3_RASPP"

MODEL_NAME = "dinov3_convnext_tiny"
NUM_CLASSES = 19
IGNORE_INDEX = 255
OUTPUT_STRIDE = 16
ARTIFACT_TYPE = "dinov3_cityscapes19_frozen_raspp_t0"
ARTIFACT_FORMAT_VERSION = 1
SPLIT_LOCK_FILENAME = "local_splits.lock.json"
SPLIT_CHECKSUM_FILENAME = "local_splits.sha256"

CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

# Official Cityscapes labelId -> trainId mapping.
LABEL_ID_TO_TRAIN_ID = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}
LABEL_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _label_id, _train_id in LABEL_ID_TO_TRAIN_ID.items():
    LABEL_LUT[_label_id] = _train_id

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure T0: frozen DINOv3 ConvNeXt-T + R-ASPP on Cityscapes."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--head-channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--crop-height", type=int, default=512)
    parser.add_argument("--crop-width", type=int, default=1024)
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request deterministic kernels where PyTorch provides them.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the seed run's last checkpoint.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate data, shape, freezing, forward, loss, and backward without training.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the selected T0 artifact and its current dataset/backbone hashes.",
    )
    args = parser.parse_args()

    positive_int_fields = (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "accumulation_steps",
        "head_channels",
        "crop_height",
        "crop_width",
        "eval_every",
    )
    for field in positive_int_fields:
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if not 0 < args.min_lr_ratio <= 1:
        parser.error("--min-lr-ratio must be in (0, 1]")
    if args.poly_power <= 0:
        parser.error("--poly-power must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if not 0 < args.scale_min <= args.scale_max:
        parser.error("Require 0 < --scale-min <= --scale-max")
    if args.crop_height % OUTPUT_STRIDE or args.crop_width % OUTPUT_STRIDE:
        parser.error(f"Crop dimensions must be divisible by output stride {OUTPUT_STRIDE}")
    if args.boundary_tolerance < 0:
        parser.error("--boundary-tolerance cannot be negative")
    if args.resume and (args.smoke_test or args.verify_only):
        parser.error("--resume cannot be combined with --smoke-test or --verify-only")
    return args


def resolve_device(device_string: str) -> torch.device:
    if device_string == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_string)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_global_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"State entry {name!r} is not a tensor")
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        # Flatten first so scalar buffers (for example BatchNorm's
        # num_batches_tracked) can also be reinterpreted as raw bytes.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def cpu_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def write_json_atomic(path: Path, value: object) -> None:
    path = Path(path)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as file_obj:
        json.dump(value, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    os.replace(temp_path, path)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp_path, path)


def torch_save_atomic(value: object, path: Path) -> None:
    path = Path(path)
    temp_path = path.with_name(f".{path.name}.tmp")
    torch.save(value, temp_path)
    os.replace(temp_path, path)


def safe_torch_load(path: Path, map_location: object = "cpu", weights_only: bool = True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_under_root(dataset_root: Path, relative_path: str) -> Path:
    candidate = (dataset_root / relative_path).resolve()
    try:
        candidate.relative_to(dataset_root)
    except ValueError as error:
        raise RuntimeError(f"Manifest path escapes dataset root: {relative_path}") from error
    return candidate


def _manifest_payloads_and_entries(
    dataset_root: Path, lock: Mapping[str, object]
) -> Tuple[Dict[str, bytes], Dict[str, List[Tuple[str, str]]]]:
    payloads: Dict[str, bytes] = {}
    entries_by_split: Dict[str, List[Tuple[str, str]]] = {}
    split_ids: Dict[str, set[str]] = {}

    for split in ("train_local", "dev_local", "test_local"):
        split_spec = lock["splits"][split]
        manifest_name = str(split_spec["manifest"])
        manifest_path = dataset_root / manifest_name
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing locked manifest: {manifest_path}")
        payload = manifest_path.read_bytes()
        actual_hash = sha256_bytes(payload)
        expected_hash = str(split_spec["sha256"])
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{manifest_name} SHA-256 mismatch: actual={actual_hash}, expected={expected_hash}"
            )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"Manifest is not UTF-8: {manifest_path}") from error
        if text.startswith("\ufeff") or "\r" in text:
            raise RuntimeError(f"Manifest must use UTF-8 without BOM and LF endings: {manifest_path}")

        rows: List[Tuple[str, str]] = []
        sample_ids: set[str] = set()
        cities: set[str] = set()
        for line_number, line in enumerate(text.splitlines(), start=1):
            fields = line.split("\t")
            if len(fields) != 2:
                raise RuntimeError(
                    f"Expected image<TAB>label at {manifest_name}:{line_number}"
                )
            image_rel, label_rel = fields
            image_path = _resolve_under_root(dataset_root, image_rel)
            label_path = _resolve_under_root(dataset_root, label_rel)
            if not image_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing manifest pair at {manifest_name}:{line_number}: "
                    f"{image_rel}, {label_rel}"
                )
            image_suffix = "_leftImg8bit.png"
            label_suffix = "_gtFine_labelIds.png"
            if not image_path.name.endswith(image_suffix) or not label_path.name.endswith(label_suffix):
                raise RuntimeError(f"Unexpected Cityscapes filename at {manifest_name}:{line_number}")
            image_id = image_path.name[: -len(image_suffix)]
            label_id = label_path.name[: -len(label_suffix)]
            if image_id != label_id:
                raise RuntimeError(
                    f"Image/label ID mismatch at {manifest_name}:{line_number}: "
                    f"{image_id} != {label_id}"
                )
            if image_id in sample_ids:
                raise RuntimeError(f"Duplicate sample in {manifest_name}: {image_id}")
            if image_path.parent.name != label_path.parent.name:
                raise RuntimeError(f"Image/label city mismatch for {image_id}")
            sample_ids.add(image_id)
            cities.add(image_path.parent.name)
            rows.append((image_rel, label_rel))

        expected_count = int(split_spec["count"])
        expected_cities = sorted(str(city) for city in split_spec["cities"])
        if len(rows) != expected_count:
            raise RuntimeError(
                f"{split} count mismatch: actual={len(rows)}, expected={expected_count}"
            )
        if sorted(cities) != expected_cities:
            raise RuntimeError(
                f"{split} city mismatch: actual={sorted(cities)}, expected={expected_cities}"
            )
        if rows != sorted(rows):
            raise RuntimeError(f"Manifest is not in stable lexical order: {manifest_name}")
        payloads[manifest_name] = payload
        entries_by_split[split] = rows
        split_ids[split] = sample_ids

    split_names = list(split_ids)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise RuntimeError(
                    f"Locked splits overlap: {left}/{right}: {sorted(overlap)[:5]}"
                )
    return payloads, entries_by_split


def validate_dataset_lock(dataset_root: Path) -> Tuple[Dict[str, object], Dict[str, List[Tuple[str, str]]]]:
    dataset_root = Path(dataset_root).resolve()
    lock_path = dataset_root / SPLIT_LOCK_FILENAME
    checksum_path = dataset_root / SPLIT_CHECKSUM_FILENAME
    if not lock_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(
            f"Locked split metadata is required: {lock_path}, {checksum_path}"
        )
    with lock_path.open("r", encoding="utf-8") as file_obj:
        lock = json.load(file_obj)
    if lock.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported split lock schema: {lock.get('schema_version')}")

    payloads, entries_by_split = _manifest_payloads_and_entries(dataset_root, lock)
    combined = hashlib.sha256()
    for filename in sorted(payloads):
        combined.update(filename.encode("utf-8"))
        combined.update(b"\0")
        combined.update(payloads[filename])
    combined_hash = combined.hexdigest()
    if combined_hash != lock.get("combined_manifest_sha256"):
        raise RuntimeError(
            f"Combined manifest hash mismatch: actual={combined_hash}, "
            f"expected={lock.get('combined_manifest_sha256')}"
        )

    checksum_rows: Dict[str, str] = {}
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise RuntimeError(f"Malformed checksum row: {line!r}")
        checksum_rows[fields[1]] = fields[0].lower()
    files_to_check = {**payloads, SPLIT_LOCK_FILENAME: lock_path.read_bytes()}
    for filename, payload in files_to_check.items():
        actual = sha256_bytes(payload)
        if checksum_rows.get(filename) != actual:
            raise RuntimeError(
                f"{SPLIT_CHECKSUM_FILENAME} mismatch for {filename}: "
                f"actual={actual}, listed={checksum_rows.get(filename)}"
            )

    info = {
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "checksum_path": str(checksum_path),
        "checksum_sha256": sha256_file(checksum_path),
        "lock_date": lock["lock_date"],
        "timezone": lock["timezone"],
        "combined_manifest_sha256": combined_hash,
        "splits": copy.deepcopy(lock["splits"]),
    }
    return info, entries_by_split


def label_ids_to_train_ids(label: Image.Image) -> torch.Tensor:
    array = np.asarray(label, dtype=np.uint8)
    mapped = LABEL_LUT[array]
    return torch.from_numpy(mapped.copy()).long()


def image_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class CityscapesTrainTransform:
    def __init__(
        self,
        crop_size: Tuple[int, int],
        scale_range: Tuple[float, float],
        flip_probability: float = 0.5,
    ) -> None:
        self.crop_height, self.crop_width = crop_size
        self.scale_min, self.scale_max = scale_range
        self.flip_probability = flip_probability

    def __call__(self, image: Image.Image, label: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = random.uniform(self.scale_min, self.scale_max)
        new_width = max(1, int(round(image.width * scale)))
        new_height = max(1, int(round(image.height * scale)))
        image = image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
        label = label.resize((new_width, new_height), resample=Image.Resampling.NEAREST)

        pad_right = max(0, self.crop_width - new_width)
        pad_bottom = max(0, self.crop_height - new_height)
        if pad_right or pad_bottom:
            image = ImageOps.expand(image, border=(0, 0, pad_right, pad_bottom), fill=(0, 0, 0))
            label = ImageOps.expand(label, border=(0, 0, pad_right, pad_bottom), fill=IGNORE_INDEX)

        max_left = image.width - self.crop_width
        max_top = image.height - self.crop_height
        left = random.randint(0, max_left) if max_left > 0 else 0
        top = random.randint(0, max_top) if max_top > 0 else 0
        box = (left, top, left + self.crop_width, top + self.crop_height)
        image = image.crop(box)
        label = label.crop(box)

        if random.random() < self.flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image_to_normalized_tensor(image), label_ids_to_train_ids(label)


class CityscapesEvalTransform:
    def __call__(self, image: Image.Image, label: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return image_to_normalized_tensor(image), label_ids_to_train_ids(label)


class CityscapesManifestDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        entries: Sequence[Tuple[str, str]],
        transform,
        reject_all_ignore: bool,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.entries = list(entries)
        self.transform = transform
        self.reject_all_ignore = reject_all_ignore

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        image_rel, label_rel = self.entries[index]
        image_path = self.dataset_root / image_rel
        label_path = self.dataset_root / label_rel
        with Image.open(image_path) as image_obj:
            image = image_obj.convert("RGB")
        with Image.open(label_path) as label_obj:
            label = label_obj.convert("L")

        attempts = 10 if self.reject_all_ignore else 1
        for _ in range(attempts):
            image_tensor, target = self.transform(image, label)
            if not self.reject_all_ignore or bool((target != IGNORE_INDEX).any()):
                return image_tensor, target, image_rel
        raise RuntimeError(f"Could not draw a valid non-ignore crop from {image_rel}")


def make_data_loaders(
    dataset_root: Path,
    entries_by_split: Mapping[str, Sequence[Tuple[str, str]]],
    args: argparse.Namespace,
    device: torch.device,
):
    train_dataset = CityscapesManifestDataset(
        dataset_root=dataset_root,
        entries=entries_by_split["train_local"],
        transform=CityscapesTrainTransform(
            crop_size=(args.crop_height, args.crop_width),
            scale_range=(args.scale_min, args.scale_max),
        ),
        reject_all_ignore=True,
    )
    dev_dataset = CityscapesManifestDataset(
        dataset_root=dataset_root,
        entries=entries_by_split["dev_local"],
        transform=CityscapesEvalTransform(),
        reject_all_ignore=False,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        **common,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, dev_loader, generator


class SamePadDownsample(nn.Module):
    """Run a pretrained 2x2 downsample convolution at stride 1 with SAME padding."""

    def __init__(self, convolution: nn.Conv2d) -> None:
        super().__init__()
        if convolution.kernel_size != (2, 2) or convolution.stride != (2, 2):
            raise ValueError(
                f"Expected pretrained 2x2/stride2 conv, got "
                f"kernel={convolution.kernel_size}, stride={convolution.stride}"
            )
        convolution.stride = (1, 1)
        convolution.padding = (0, 0)
        self.convolution = convolution

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Right/bottom padding preserves HxW for an even 2x2 kernel.
        return self.convolution(F.pad(inputs, (0, 1, 0, 1)))


def validate_model_paths(repo_dir: Path, weights_path: Path) -> Tuple[Path, Path]:
    repo_dir = Path(repo_dir).resolve()
    weights_path = Path(weights_path).resolve()
    if not repo_dir.is_dir() or not (repo_dir / "hubconf.py").is_file():
        raise FileNotFoundError(f"Invalid local DINOv3 repository: {repo_dir}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
    return repo_dir, weights_path


def convert_backbone_to_output_stride16(backbone: nn.Module) -> None:
    if getattr(backbone, "_cityscapes_output_stride", None) == OUTPUT_STRIDE:
        return
    if not hasattr(backbone, "downsample_layers") or not hasattr(backbone, "stages"):
        raise RuntimeError("Loaded DINOv3 backbone is not the expected ConvNeXt model")
    downsample = backbone.downsample_layers[3]
    if not isinstance(downsample, nn.Sequential) or len(downsample) != 2:
        raise RuntimeError("Unexpected final ConvNeXt downsample structure")
    convolution = downsample[1]
    if not isinstance(convolution, nn.Conv2d):
        raise RuntimeError("Final ConvNeXt downsample is not Conv2d")
    downsample[1] = SamePadDownsample(convolution)

    for block in backbone.stages[3]:
        depthwise = getattr(block, "dwconv", None)
        if not isinstance(depthwise, nn.Conv2d) or depthwise.kernel_size != (7, 7):
            raise RuntimeError("Unexpected ConvNeXt final-stage depthwise convolution")
        depthwise.dilation = (2, 2)
        depthwise.padding = (6, 6)
    backbone._cityscapes_output_stride = OUTPUT_STRIDE


def load_backbone(repo_dir: Path, weights_path: Path) -> nn.Module:
    repo_dir, weights_path = validate_model_paths(repo_dir, weights_path)
    backbone = torch.hub.load(
        repo_or_dir=str(repo_dir),
        model=MODEL_NAME,
        source="local",
        weights=str(weights_path),
    )
    if list(getattr(backbone, "embed_dims", [])) != [96, 192, 384, 768]:
        raise RuntimeError(
            f"Unexpected DINOv3 ConvNeXt-T dimensions: {getattr(backbone, 'embed_dims', None)}"
        )
    convert_backbone_to_output_stride16(backbone)
    backbone.requires_grad_(False)
    backbone.eval()
    return backbone


class RASPPHead(nn.Module):
    """Reduced ASPP: high-level 1x1 branch plus image-level context gating."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        inter_channels: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU6(inplace=True),
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(inter_channels, num_classes, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, high_feature: torch.Tensor) -> torch.Tensor:
        local = self.project(high_feature)
        context_gate = self.image_pool(high_feature)
        return self.classifier(self.dropout(local * context_gate))


class FrozenDINOv3RASPPTeacher(nn.Module):
    def __init__(self, backbone: nn.Module, head: RASPPHead, freeze_head: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.freeze_head = freeze_head
        self.backbone.requires_grad_(False)
        if freeze_head:
            self.head.requires_grad_(False)

    @property
    def teacher_head(self) -> RASPPHead:
        return self.head

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.head.train(False if self.freeze_head else mode)
        return self

    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.backbone.get_intermediate_layers(
                images,
                n=[0, 1, 2, 3],
                reshape=True,
                return_class_token=False,
                norm=True,
            )
        if len(outputs) != 4:
            raise RuntimeError(f"Expected four ConvNeXt stage outputs, got {len(outputs)}")
        return {
            "os4": outputs[0],
            "os8": outputs[1],
            "os16_mid": outputs[2],
            "os16": outputs[3],
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        input_size = images.shape[-2:]
        high_feature = self.extract_features(images)["os16"]
        logits = self.head(high_feature)
        return F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

    def freeze_for_distillation(self):
        self.freeze_head = True
        self.requires_grad_(False)
        self.eval()
        return self


def build_model(backbone: nn.Module, head_channels: int, dropout: float) -> FrozenDINOv3RASPPTeacher:
    head = RASPPHead(
        in_channels=768,
        num_classes=NUM_CLASSES,
        inter_channels=head_channels,
        dropout=dropout,
    )
    model = FrozenDINOv3RASPPTeacher(backbone=backbone, head=head, freeze_head=False)
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("Backbone freeze failed")
    if not all(parameter.requires_grad for parameter in model.head.parameters()):
        raise RuntimeError("Every R-ASPP parameter must be trainable in T0")
    return model


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def audit_model_shapes(
    model: FrozenDINOv3RASPPTeacher,
    device: torch.device,
    crop_height: int,
    crop_width: int,
    amp_enabled: bool,
) -> Dict[str, object]:
    model.eval()
    sample = torch.zeros(1, 3, crop_height, crop_width, device=device)
    with torch.inference_mode(), autocast_context(device, amp_enabled):
        features = model.extract_features(sample)
        logits = model(sample)
    expected = {
        "os4": (crop_height // 4, crop_width // 4, 96),
        "os8": (crop_height // 8, crop_width // 8, 192),
        "os16_mid": (crop_height // 16, crop_width // 16, 384),
        "os16": (crop_height // 16, crop_width // 16, 768),
    }
    feature_shapes: Dict[str, List[int]] = {}
    for name, tensor in features.items():
        height, width, channels = expected[name]
        if tensor.shape != (1, channels, height, width):
            raise RuntimeError(
                f"Shape audit failed for {name}: actual={tuple(tensor.shape)}, "
                f"expected={(1, channels, height, width)}"
            )
        feature_shapes[name] = list(tensor.shape)
    if logits.shape != (1, NUM_CLASSES, crop_height, crop_width):
        raise RuntimeError(f"Logit shape audit failed: {tuple(logits.shape)}")
    return {
        "input_shape": [1, 3, crop_height, crop_width],
        "feature_shapes": feature_shapes,
        "logit_shape": list(logits.shape),
        "output_stride": OUTPUT_STRIDE,
        "conversion": (
            "final downsample stride 2->1 with SAME padding; final-stage "
            "7x7 depthwise convolutions use dilation=2, padding=6"
        ),
    }


def confusion_counts(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    valid = targets != IGNORE_INDEX
    encoded = targets[valid].to(torch.int64) * NUM_CLASSES + predictions[valid].to(torch.int64)
    counts = torch.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES)
    return counts.reshape(NUM_CLASSES, NUM_CLASSES).cpu()


def semantic_boundaries(labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    boundaries = torch.zeros_like(valid, dtype=torch.bool)
    horizontal_valid = valid[:, :, 1:] & valid[:, :, :-1]
    horizontal_diff = (labels[:, :, 1:] != labels[:, :, :-1]) & horizontal_valid
    boundaries[:, :, 1:] |= horizontal_diff
    boundaries[:, :, :-1] |= horizontal_diff
    vertical_valid = valid[:, 1:, :] & valid[:, :-1, :]
    vertical_diff = (labels[:, 1:, :] != labels[:, :-1, :]) & vertical_valid
    boundaries[:, 1:, :] |= vertical_diff
    boundaries[:, :-1, :] |= vertical_diff
    return boundaries


def boundary_match_counts(
    predictions: torch.Tensor, targets: torch.Tensor, tolerance: int
) -> Tuple[int, int, int, int]:
    valid = targets != IGNORE_INDEX
    predicted_boundary = semantic_boundaries(predictions, valid)
    target_boundary = semantic_boundaries(targets, valid)
    if tolerance > 0:
        kernel_size = 2 * tolerance + 1
        predicted_dilated = F.max_pool2d(
            predicted_boundary.float().unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=tolerance,
        ).squeeze(1).bool()
        target_dilated = F.max_pool2d(
            target_boundary.float().unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=tolerance,
        ).squeeze(1).bool()
    else:
        predicted_dilated = predicted_boundary
        target_dilated = target_boundary
    matched_predicted = int((predicted_boundary & target_dilated).sum().item())
    matched_target = int((target_boundary & predicted_dilated).sum().item())
    return (
        matched_predicted,
        int(predicted_boundary.sum().item()),
        matched_target,
        int(target_boundary.sum().item()),
    )


def metrics_from_confusion(
    confusion: torch.Tensor,
    loss_sum: float,
    valid_pixels: int,
    boundary_counts: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, object]:
    matrix = confusion.to(torch.float64)
    intersection = matrix.diag()
    target_count = matrix.sum(dim=1)
    prediction_count = matrix.sum(dim=0)
    union = target_count + prediction_count - intersection
    iou = torch.where(union > 0, intersection / union, torch.nan)
    class_accuracy = torch.where(
        target_count > 0, intersection / target_count, torch.nan
    )
    per_class = {
        class_name: {
            "iou": None if torch.isnan(iou[index]) else float(iou[index]),
            "accuracy": (
                None
                if torch.isnan(class_accuracy[index])
                else float(class_accuracy[index])
            ),
            "target_pixels": int(target_count[index].item()),
            "predicted_pixels": int(prediction_count[index].item()),
        }
        for index, class_name in enumerate(CITYSCAPES_CLASSES)
    }
    small_object_indices = list(range(11, 19))
    small_ious = iou[small_object_indices]
    metrics: Dict[str, object] = {
        "loss": loss_sum / max(valid_pixels, 1),
        "mIoU": float(torch.nanmean(iou)),
        "mAcc": float(torch.nanmean(class_accuracy)),
        "pixel_accuracy": float(intersection.sum() / target_count.sum()),
        "small_object_mIoU": float(torch.nanmean(small_ious)),
        "valid_pixels": int(valid_pixels),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
    if boundary_counts is not None:
        matched_predicted, predicted_total, matched_target, target_total = boundary_counts
        precision = matched_predicted / predicted_total if predicted_total else 1.0
        recall = matched_target / target_total if target_total else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics["boundary_precision"] = precision
        metrics["boundary_recall"] = recall
        metrics["boundary_f1"] = f1
    return metrics


def evaluate(
    model: FrozenDINOv3RASPPTeacher,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    split_name: str,
    boundary_tolerance: int,
    collect_per_image: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    loss_sum = 0.0
    valid_pixels = 0
    boundary_totals = [0, 0, 0, 0]
    per_image_rows: List[Dict[str, object]] = []

    progress = tqdm(loader, desc=f"Evaluating {split_name}")
    with torch.inference_mode():
        for images, targets, paths in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast_context(device, amp_enabled):
                logits = model(images)
            logits_float = logits.float()
            batch_loss_sum = F.cross_entropy(
                logits_float,
                targets,
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            batch_valid = int((targets != IGNORE_INDEX).sum().item())
            predictions = logits_float.argmax(dim=1)
            batch_confusion = confusion_counts(predictions, targets)
            confusion += batch_confusion
            loss_sum += float(batch_loss_sum.item())
            valid_pixels += batch_valid

            boundary_batch = boundary_match_counts(
                predictions, targets, boundary_tolerance
            )
            for index, value in enumerate(boundary_batch):
                boundary_totals[index] += value

            if collect_per_image:
                for item_index, path in enumerate(paths):
                    item_confusion = confusion_counts(
                        predictions[item_index : item_index + 1],
                        targets[item_index : item_index + 1],
                    )
                    per_image_rows.append(
                        {
                            "image": path,
                            "valid_pixels": int(
                                (targets[item_index] != IGNORE_INDEX).sum().item()
                            ),
                            "confusion_matrix": item_confusion.tolist(),
                        }
                    )
            running = metrics_from_confusion(
                confusion, loss_sum, valid_pixels, tuple(boundary_totals)
            )
            progress.set_postfix(
                {
                    "mIoU": f"{running['mIoU']:.4f}",
                    "bF1": f"{running['boundary_f1']:.4f}",
                }
            )

    if valid_pixels == 0:
        raise RuntimeError(f"No valid pixels found in {split_name}")
    return (
        metrics_from_confusion(
            confusion, loss_sum, valid_pixels, tuple(boundary_totals)
        ),
        per_image_rows,
    )


def train_one_epoch(
    model: FrozenDINOv3RASPPTeacher,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    accumulation_steps: int,
    epoch: int,
    epochs: int,
) -> Tuple[Dict[str, object], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    loss_sum = 0.0
    valid_pixels = 0
    optimizer_steps = 0
    group_size = accumulation_steps
    progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [T0 R-ASPP]")

    for batch_index, (images, targets, _) in enumerate(progress):
        group_position = batch_index % accumulation_steps
        if group_position == 0:
            group_size = min(accumulation_steps, len(loader) - batch_index)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
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
            raise RuntimeError("Training batch contains no valid Cityscapes pixels")
        batch_loss = batch_loss_sum / batch_valid
        scaler.scale(batch_loss / group_size).backward()

        is_group_end = group_position + 1 == group_size
        if is_group_end:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1

        predictions = logits_float.detach().argmax(dim=1)
        confusion += confusion_counts(predictions, targets)
        loss_sum += float(batch_loss_sum.detach().item())
        valid_pixels += batch_valid
        running = metrics_from_confusion(confusion, loss_sum, valid_pixels)
        progress.set_postfix(
            {
                "loss": f"{running['loss']:.4f}",
                "mIoU": f"{running['mIoU']:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
        )

    return metrics_from_confusion(confusion, loss_sum, valid_pixels), optimizer_steps


def build_best_checkpoint(
    model: FrozenDINOv3RASPPTeacher,
    epoch: int,
    dev_metrics: Mapping[str, object],
    config: Mapping[str, object],
    hashes: Mapping[str, object],
    dataset_lock: Mapping[str, object],
    shape_audit: Mapping[str, object],
) -> Dict[str, object]:
    head_state = cpu_state_dict(model.head)
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "frozen_for_distillation": True,
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "class_names": list(CITYSCAPES_CLASSES),
        "output_stride": OUTPUT_STRIDE,
        "head_type": "R-ASPP",
        "head_state_dict": head_state,
        "head_state_sha256": state_dict_sha256(head_state),
        "best_epoch": epoch,
        "best_dev_metrics": copy.deepcopy(dev_metrics),
        "config": copy.deepcopy(config),
        "hashes": copy.deepcopy(hashes),
        "dataset_lock": copy.deepcopy(dataset_lock),
        "shape_audit": copy.deepcopy(shape_audit),
    }


def _checkpoint_sidecar_path(checkpoint_path: Path) -> Path:
    return Path(str(checkpoint_path) + ".sha256")


def write_checkpoint_with_sidecar(payload: object, checkpoint_path: Path) -> str:
    torch_save_atomic(payload, checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    sidecar = _checkpoint_sidecar_path(checkpoint_path)
    sidecar.write_text(
        f"{checkpoint_hash}  {checkpoint_path.name}\n",
        encoding="ascii",
        newline="\n",
    )
    return checkpoint_hash


def verify_checkpoint_sidecar(checkpoint_path: Path) -> str:
    sidecar = _checkpoint_sidecar_path(checkpoint_path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing checkpoint sidecar: {sidecar}")
    fields = sidecar.read_text(encoding="ascii").split()
    if len(fields) < 1:
        raise RuntimeError(f"Empty checkpoint sidecar: {sidecar}")
    expected = fields[0].lower()
    actual = sha256_file(checkpoint_path)
    if actual != expected:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch: actual={actual}, expected={expected}"
        )
    return actual


def load_teacher_for_distillation(
    checkpoint_path: Path,
    repo_dir: Path = DEFAULT_REPO_DIR,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    device: object = None,
    verify_checkpoint_file: bool = True,
) -> Tuple[FrozenDINOv3RASPPTeacher, Dict[str, object]]:
    checkpoint_path = Path(checkpoint_path).resolve()
    if verify_checkpoint_file:
        verify_checkpoint_sidecar(checkpoint_path)
    payload = safe_torch_load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "frozen_for_distillation": True,
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "output_stride": OUTPUT_STRIDE,
        "head_type": "R-ASPP",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"Incompatible teacher artifact field {key}: "
                f"actual={payload.get(key)!r}, expected={expected!r}"
            )
    if payload.get("class_names") != CITYSCAPES_CLASSES:
        raise RuntimeError("Teacher artifact Cityscapes class mapping is incompatible")

    head_state = payload["head_state_dict"]
    actual_head_hash = state_dict_sha256(head_state)
    if actual_head_hash != payload.get("head_state_sha256"):
        raise RuntimeError(
            f"R-ASPP state hash mismatch: actual={actual_head_hash}, "
            f"expected={payload.get('head_state_sha256')}"
        )
    weights_path = Path(weights_path).resolve()
    weights_hash = sha256_file(weights_path)
    if weights_hash != payload["hashes"]["backbone_weights_sha256"]:
        raise RuntimeError("Current DINOv3 weights differ from the T0 artifact")

    backbone = load_backbone(repo_dir, weights_path)
    backbone_hash = state_dict_sha256(backbone.state_dict())
    if backbone_hash != payload["hashes"]["converted_backbone_state_sha256"]:
        raise RuntimeError("Loaded OS=16 DINOv3 backbone differs from the T0 artifact")
    config = payload["config"]
    head = RASPPHead(
        in_channels=768,
        num_classes=NUM_CLASSES,
        inter_channels=int(config["head_channels"]),
        dropout=float(config["dropout"]),
    )
    head.load_state_dict(head_state, strict=True)
    teacher = FrozenDINOv3RASPPTeacher(backbone, head, freeze_head=True)
    if device is None:
        target_device = resolve_device("auto")
    elif isinstance(device, str):
        target_device = resolve_device(device)
    else:
        target_device = torch.device(device)
    teacher = teacher.to(target_device).freeze_for_distillation()
    return teacher, payload


def metrics_reproduce(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    if left["confusion_matrix"] != right["confusion_matrix"]:
        return False
    for key in ("loss", "mIoU", "mAcc", "pixel_accuracy", "boundary_f1"):
        if not math.isclose(float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1e-6):
            return False
    return True


def run_smoke_test(
    model: FrozenDINOv3RASPPTeacher,
    train_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> None:
    model.train()
    images, targets, paths = next(iter(train_loader))
    images = images[:1].to(device)
    targets = targets[:1].to(device)
    with autocast_context(device, amp_enabled):
        logits = model(images)
    loss = F.cross_entropy(
        logits.float(), targets, ignore_index=IGNORE_INDEX, reduction="mean"
    )
    model.zero_grad(set_to_none=True)
    loss.backward()
    backbone_gradients = [
        parameter.grad for parameter in model.backbone.parameters() if parameter.grad is not None
    ]
    head_gradient_count = sum(
        parameter.grad is not None for parameter in model.head.parameters()
    )
    if backbone_gradients:
        raise RuntimeError("Smoke test found gradients in the frozen backbone")
    if head_gradient_count == 0:
        raise RuntimeError("Smoke test found no gradients in R-ASPP")
    print("[OK] T0 smoke test passed")
    print(f"   - sample: {paths[0]}")
    print(f"   - logits: {tuple(logits.shape)}")
    print(f"   - valid pixels: {int((targets != IGNORE_INDEX).sum().item())}")
    print(f"   - loss: {loss.item():.6f}")
    print(f"   - backbone gradients: 0")
    print(f"   - head tensors with gradients: {head_gradient_count}")


def run_training(args: argparse.Namespace) -> None:
    set_global_seed(args.seed, args.deterministic)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    dataset_root = args.dataset_root.resolve()
    dataset_lock, entries_by_split = validate_dataset_lock(dataset_root)
    print(
        "[OK] Locked Cityscapes splits: "
        f"train={len(entries_by_split['train_local'])}, "
        f"dev={len(entries_by_split['dev_local'])}, "
        f"test={len(entries_by_split['test_local'])} (not evaluated)"
    )

    weights_path = args.weights_path.resolve()
    weights_hash = sha256_file(weights_path)
    backbone = load_backbone(args.repo_dir, weights_path)
    converted_backbone_hash = state_dict_sha256(backbone.state_dict())
    model = build_model(backbone, args.head_channels, args.dropout).to(device)
    shape_audit = audit_model_shapes(
        model,
        device,
        args.crop_height,
        args.crop_width,
        amp_enabled,
    )
    train_loader, dev_loader, train_generator = make_data_loaders(
        dataset_root, entries_by_split, args, device
    )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    backbone_parameters = sum(parameter.numel() for parameter in model.backbone.parameters())
    head_parameters = sum(parameter.numel() for parameter in model.head.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters != head_parameters:
        raise RuntimeError(
            f"T0 must train only R-ASPP: trainable={trainable_parameters}, head={head_parameters}"
        )
    print(f"[INFO] Device={device}; AMP={amp_enabled}")
    print(
        f"[OK] Frozen backbone params={backbone_parameters:,}; "
        f"trainable R-ASPP params={head_parameters:,}"
    )
    print(f"[OK] Feature shapes: {shape_audit['feature_shapes']}")

    if args.smoke_test:
        run_smoke_test(model, train_loader, device, amp_enabled)
        return

    run_dir = args.output_dir.resolve() / f"seed_{args.seed}"
    best_checkpoint_path = run_dir / "t0_dinov3_raspp_teacher.pth"
    last_checkpoint_path = run_dir / "t0_last_checkpoint.pth"
    history_path = run_dir / "training_history.json"
    results_path = run_dir / "t0_metrics.json"
    per_image_path = run_dir / "dev_per_image_confusion.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        teacher, payload = load_teacher_for_distillation(
            best_checkpoint_path,
            repo_dir=args.repo_dir,
            weights_path=weights_path,
            device=device,
        )
        if (
            payload["dataset_lock"]["combined_manifest_sha256"]
            != dataset_lock["combined_manifest_sha256"]
        ):
            raise RuntimeError("Current locked split differs from the T0 artifact")
        verified_shape = audit_model_shapes(
            teacher,
            device,
            args.crop_height,
            args.crop_width,
            amp_enabled,
        )
        del teacher
        print("[OK] T0 artifact verified")
        print(f"   - checkpoint: {best_checkpoint_path}")
        print(f"   - checkpoint SHA-256: {verify_checkpoint_sidecar(best_checkpoint_path)}")
        print(f"   - head SHA-256: {payload['head_state_sha256']}")
        print(f"   - best epoch: {payload['best_epoch']}")
        print(f"   - best dev mIoU: {payload['best_dev_metrics']['mIoU']:.6f}")
        print(f"   - shapes: {verified_shape['feature_shapes']}")
        return

    if not args.resume and any(
        path.exists()
        for path in (
            best_checkpoint_path,
            last_checkpoint_path,
            history_path,
            results_path,
            per_image_path,
        )
    ):
        raise FileExistsError(
            f"Run artifacts already exist in {run_dir}. Use --resume or a different --output-dir."
        )

    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    total_optimizer_steps = steps_per_epoch * args.epochs

    def lr_factor(step: int) -> float:
        progress = min(step, total_optimizer_steps) / max(total_optimizer_steps, 1)
        return max((1.0 - progress) ** args.poly_power, args.min_lr_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    config = {
        "experiment": "T0",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "accumulation_steps": args.accumulation_steps,
        "global_batch_size": args.batch_size * args.accumulation_steps,
        "num_workers": args.num_workers,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": "polynomial",
        "poly_power": args.poly_power,
        "min_lr_ratio": args.min_lr_ratio,
        "eval_every": args.eval_every,
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
        "backbone_frozen": True,
        "test_local_evaluated": False,
    }
    hashes = {
        "backbone_weights_sha256": weights_hash,
        "converted_backbone_state_sha256": converted_backbone_hash,
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
    }

    history: List[Dict[str, object]] = []
    best_key: Optional[Tuple[float, float, float, float]] = None
    best_epoch: Optional[int] = None
    best_dev_metrics: Optional[Dict[str, object]] = None
    start_epoch = 1
    cumulative_optimizer_steps = 0

    if args.resume:
        if not last_checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {last_checkpoint_path}")
        resume_payload = safe_torch_load(
            last_checkpoint_path, map_location="cpu", weights_only=False
        )
        if resume_payload.get("config") != config:
            raise RuntimeError("Resume configuration differs from the current T0 arguments")
        resume_hashes = resume_payload.get("hashes", {})
        for hash_name in (
            "backbone_weights_sha256",
            "converted_backbone_state_sha256",
        ):
            if resume_hashes.get(hash_name) != hashes[hash_name]:
                raise RuntimeError(f"Resume {hash_name} differs from the current run")
        if resume_hashes.get("training_script_sha256") != hashes["training_script_sha256"]:
            print(
                "[WARN] Training script SHA-256 differs from the resume checkpoint; "
                "backbone hashes and the remaining resume invariants still match."
            )
        if (
            resume_payload["dataset_lock"]["combined_manifest_sha256"]
            != dataset_lock["combined_manifest_sha256"]
        ):
            raise RuntimeError("Resume dataset lock differs from the current split")
        model.head.load_state_dict(resume_payload["head_state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        scaler.load_state_dict(resume_payload["scaler_state_dict"])
        train_generator.set_state(resume_payload["train_generator_state"])
        history = resume_payload["history"]
        best_key = resume_payload["best_key"]
        best_epoch = resume_payload["best_epoch"]
        best_dev_metrics = resume_payload["best_dev_metrics"]
        cumulative_optimizer_steps = int(resume_payload["optimizer_steps"])
        start_epoch = int(resume_payload["epoch"]) + 1
        print(f"[OK] Resuming T0 at epoch {start_epoch}")

    training_started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics, optimizer_steps = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            accumulation_steps=args.accumulation_steps,
            epoch=epoch,
            epochs=args.epochs,
        )
        cumulative_optimizer_steps += optimizer_steps
        should_evaluate = epoch % args.eval_every == 0 or epoch == args.epochs
        dev_metrics = None
        if should_evaluate:
            dev_metrics, _ = evaluate(
                model=model,
                loader=dev_loader,
                device=device,
                amp_enabled=amp_enabled,
                split_name="dev_local",
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
                best_dev_metrics = copy.deepcopy(dev_metrics)
                best_payload = build_best_checkpoint(
                    model=model,
                    epoch=epoch,
                    dev_metrics=dev_metrics,
                    config=config,
                    hashes=hashes,
                    dataset_lock=dataset_lock,
                    shape_audit=shape_audit,
                )
                checkpoint_hash = write_checkpoint_with_sidecar(
                    best_payload, best_checkpoint_path
                )
                print(
                    f"[OK] Best T0 updated: epoch={epoch}, "
                    f"dev_mIoU={dev_metrics['mIoU']:.6f}, sha256={checkpoint_hash}"
                )

        epoch_record = {
            "epoch": epoch,
            "optimizer_steps": cumulative_optimizer_steps,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "dev": dev_metrics,
        }
        history.append(epoch_record)
        write_json_atomic(history_path, history)
        last_payload = {
            "epoch": epoch,
            "head_state_dict": cpu_state_dict(model.head),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_generator_state": train_generator.get_state(),
            "history": history,
            "best_key": best_key,
            "best_epoch": best_epoch,
            "best_dev_metrics": best_dev_metrics,
            "optimizer_steps": cumulative_optimizer_steps,
            "config": config,
            "hashes": hashes,
            "dataset_lock": dataset_lock,
        }
        torch_save_atomic(last_payload, last_checkpoint_path)
        message = (
            f"Epoch {epoch}: train_mIoU={train_metrics['mIoU']:.4f}, "
            f"train_loss={train_metrics['loss']:.4f}"
        )
        if dev_metrics is not None:
            message += (
                f", dev_mIoU={dev_metrics['mIoU']:.4f}, "
                f"dev_bF1={dev_metrics['boundary_f1']:.4f}"
            )
        print(message)

    if best_epoch is None or best_dev_metrics is None:
        raise RuntimeError("Training ended without a dev evaluation or best T0 checkpoint")
    if state_dict_sha256(model.backbone.state_dict()) != converted_backbone_hash:
        raise RuntimeError("Frozen DINOv3 backbone state changed during T0 training")

    selected_payload = safe_torch_load(
        best_checkpoint_path, map_location="cpu", weights_only=True
    )
    if state_dict_sha256(selected_payload["head_state_dict"]) != selected_payload["head_state_sha256"]:
        raise RuntimeError("Selected R-ASPP state failed its SHA-256 verification")
    model.head.load_state_dict(selected_payload["head_state_dict"], strict=True)
    model.freeze_for_distillation()
    selected_dev_metrics, per_image_rows = evaluate(
        model=model,
        loader=dev_loader,
        device=device,
        amp_enabled=amp_enabled,
        split_name="selected dev_local",
        boundary_tolerance=args.boundary_tolerance,
        collect_per_image=True,
    )
    if not metrics_reproduce(selected_dev_metrics, best_dev_metrics):
        raise RuntimeError(
            "Reloaded T0 checkpoint did not reproduce best dev metrics: "
            f"saved={best_dev_metrics['mIoU']}, reloaded={selected_dev_metrics['mIoU']}"
        )
    write_jsonl_atomic(per_image_path, per_image_rows)
    checkpoint_hash = verify_checkpoint_sidecar(best_checkpoint_path)
    head_hash = selected_payload["head_state_sha256"]

    results = {
        "experiment": "T0",
        "protocol": (
            "DINOv3 ConvNeXt-T is frozen; only R-ASPP is trained on train_local; "
            "best checkpoint is selected by dev_local mIoU; test_local is not evaluated."
        ),
        "best_epoch": best_epoch,
        "best_dev_metrics": selected_dev_metrics,
        "class_names": CITYSCAPES_CLASSES,
        "config": config,
        "shape_audit": shape_audit,
        "dataset_lock": dataset_lock,
        "model": {
            "model_name": MODEL_NAME,
            "head": "R-ASPP",
            "total_parameters": total_parameters,
            "backbone_parameters": backbone_parameters,
            "head_parameters": head_parameters,
            "trainable_parameters": trainable_parameters,
            "frozen_for_distillation": True,
        },
        "hashes": {
            **hashes,
            "head_state_sha256": head_hash,
            "checkpoint_sha256": checkpoint_hash,
        },
        "training": {
            "elapsed_seconds": time.time() - training_started,
            "optimizer_steps": cumulative_optimizer_steps,
        },
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pillow": __import__("PIL").__version__,
            "platform": platform.platform(),
        },
        "artifacts": {
            "checkpoint": str(best_checkpoint_path),
            "checkpoint_sha256": str(_checkpoint_sidecar_path(best_checkpoint_path)),
            "last_checkpoint": str(last_checkpoint_path),
            "history": str(history_path),
            "dev_per_image_confusion": str(per_image_path),
        },
    }
    write_json_atomic(results_path, results)

    print("\n[DONE] T0 DINOv3+R-ASPP baseline selected and frozen")
    print(f"   - seed: {args.seed}")
    print(f"   - best epoch: {best_epoch}")
    print(f"   - dev mIoU: {selected_dev_metrics['mIoU']:.6f}")
    print(f"   - dev mAcc: {selected_dev_metrics['mAcc']:.6f}")
    print(f"   - dev pixel accuracy: {selected_dev_metrics['pixel_accuracy']:.6f}")
    print(f"   - dev boundary F1: {selected_dev_metrics['boundary_f1']:.6f}")
    print(f"   - test_local evaluated: False")
    print(f"   - R-ASPP SHA-256: {head_hash}")
    print(f"   - checkpoint SHA-256: {checkpoint_hash}")
    print(f"   - metrics: {results_path}")


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
