"""A4 server entry point: raw-feature PCA without ``StandardScaler``.

A4 keeps the locked A0 protocol unchanged except for the teacher-side PCA
coordinate system:

    A0: ((x - scaler_mean) / scaler_scale - pca_mean) @ components.T
    A4: (x - pca_mean) @ components.T

The raw teacher features are fitted with the same deterministic 200k-token
manifest and PCA view used by A0, but A4 has its own PCA directory and never
reads or reuses A0's scaler or components.  The raw PCA transform is fused
into three fixed 1x1 convolutions for the training path.  The explicit
per-pixel expression is retained for the numerical audit.

The long-running DDP/data/checkpoint implementation is delegated to
``dino_a0_server`` after its projection, PCA-stage and artifact hooks are
replaced.  This preserves the server lifecycle fixes documented in
``plan_markdown/server_training_issues_and_solutions.md``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader
from tqdm import tqdm

import dino as t0
import dino_a0_server as a0


EXPERIMENT = "A4"
ARTIFACT_TYPE_PRETRAIN = "a4_pretrain_mobilenetv2_backbone_raw_pca"
ARTIFACT_TYPE_PROBE = "a4_probe_mobilenetv2_raspp_raw_pca"
ARTIFACT_FORMAT_VERSION = 1

DEFAULT_OUTPUT_DIR = a0.DEFAULT_OUTPUT_DIR
DEFAULT_PCA_DIR = DEFAULT_OUTPUT_DIR / "pca_shared_A4_raw"
DEFAULT_TEACHER_CHECKPOINT = a0.DEFAULT_TEACHER_CHECKPOINT

# The Conv and explicit token/matmul paths can accumulate in a different
# order.  Keep the same small audited margin used by the existing fixed-
# projection server entries while retaining the stricter mathematical gate.
EQUIVALENCE_MAX_ABS_ERROR = 1.5e-5
EQUIVALENCE_RELATIVE_L2_ERROR = 2e-6

_EQUIVALENCE_REPORT: Optional[Dict[str, Dict[str, object]]] = None

_A0_BUILD_CONFIG = a0.build_config
_A0_BUILD_PRETRAIN_CHECKPOINT = a0.build_pretrain_checkpoint
_A0_BUILD_PROBE_BEST_CHECKPOINT = a0.build_probe_best_checkpoint


class FixedRawPCAProjection(nn.Module):
    """Fixed raw-feature PCA fused into a bias-enabled 1x1 convolution."""

    def __init__(self, pca_mean: np.ndarray, components: np.ndarray, layer: str) -> None:
        super().__init__()
        if (
            pca_mean.ndim != 1
            or components.ndim != 2
            or components.shape[1] != pca_mean.shape[0]
        ):
            raise RuntimeError("A4 raw PCA parameter shapes are inconsistent")

        mean = torch.as_tensor(pca_mean, dtype=torch.float32).contiguous()
        components_tensor = torch.as_tensor(components, dtype=torch.float32).contiguous()
        weight = components_tensor
        bias = -(mean @ components_tensor.t())

        self.register_buffer("pca_mean", mean.view(1, 1, -1))
        self.register_buffer("components", components_tensor)
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias.contiguous())
        self.layer = str(layer)
        self.c_in = int(components.shape[1])
        self.d_out = int(components.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.c_in:
            raise RuntimeError(
                f"A4 raw PCA projection expects [B,{self.c_in},H,W], "
                f"got {tuple(x.shape)}"
            )
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def explicit_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reference ``PCA.transform`` expression in NCHW form."""

        if x.ndim != 4 or x.shape[1] != self.c_in:
            raise RuntimeError(
                f"A4 explicit raw PCA expects [B,{self.c_in},H,W], "
                f"got {tuple(x.shape)}"
            )
        tokens = x.permute(0, 2, 3, 1)
        projected = torch.matmul(tokens - self.pca_mean, self.components.t())
        return projected.permute(0, 3, 1, 2)

    def parameter_sha256(self) -> str:
        return a0.numpy_arrays_sha256(
            self.pca_mean.detach().cpu().numpy(),
            self.components.detach().cpu().numpy(),
            self.weight.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy(),
        )


def build_projection_bundle(
    _scalers: Mapping[str, Mapping[str, np.ndarray]],
    pcas: Mapping[str, Mapping[str, np.ndarray]],
) -> nn.ModuleDict:
    """Build fixed raw PCA projections; the scaler argument is intentionally ignored."""

    projections: Dict[str, FixedRawPCAProjection] = {}
    for layer in a0.A0_LAYER_ORDER:
        pca = pcas[layer]
        projection = FixedRawPCAProjection(
            pca_mean=np.asarray(pca["mean_"]),
            components=np.asarray(pca["components_"]),
            layer=layer,
        )
        if (
            projection.c_in != a0.TEACHER_CHANNELS[layer]
            or projection.d_out != a0.STUDENT_CHANNELS[layer]
        ):
            raise RuntimeError(
                f"A4 projection contract mismatch for {layer}: "
                f"got d_out={projection.d_out}, c_in={projection.c_in}; "
                f"expected d_out={a0.STUDENT_CHANNELS[layer]}, "
                f"c_in={a0.TEACHER_CHANNELS[layer]}"
            )
        if any(parameter.requires_grad for parameter in projection.parameters()):
            raise RuntimeError("A4 fixed raw PCA projection unexpectedly has parameters")
        projections[layer] = projection
    return nn.ModuleDict(projections)


def _compare_projection(
    projection: FixedRawPCAProjection,
    sample: torch.Tensor,
) -> Dict[str, object]:
    """Compare fused Conv output with explicit raw PCA output."""

    sample32 = sample.detach().cpu().float()
    with torch.inference_mode():
        explicit32 = projection.explicit_forward(sample32).float()
        fused32 = projection(sample32).float()
    difference32 = (fused32 - explicit32).abs()
    denominator32 = float(explicit32.norm().clamp_min(1e-12).item())

    projection64 = copy.deepcopy(projection).double()
    sample64 = sample32.double()
    with torch.inference_mode():
        explicit64 = projection64.explicit_forward(sample64)
        fused64 = projection64(sample64)
    difference64 = (fused64 - explicit64).abs()
    denominator64 = float(explicit64.norm().clamp_min(1e-12).item())
    fp32_relative = float(difference32.norm().item() / denominator32)
    fp64_relative = float(difference64.norm().item() / denominator64)

    return {
        "input_shape": list(sample32.shape),
        "max_abs_error": float(difference32.max().item()),
        "mean_abs_error": float(difference32.mean().item()),
        "relative_l2_error": fp32_relative,
        "fp32_max_abs_error": float(difference32.max().item()),
        "fp32_mean_abs_error": float(difference32.mean().item()),
        "fp32_relative_l2_error": fp32_relative,
        "fp64_max_abs_error": float(difference64.max().item()),
        "fp64_mean_abs_error": float(difference64.mean().item()),
        "fp64_relative_l2_error": fp64_relative,
        "mathematical_equivalence_passed": bool(
            difference64.max().item() <= 1e-10 and fp64_relative <= 1e-12
        ),
        "passed": bool(
            difference32.max().item() <= EQUIVALENCE_MAX_ABS_ERROR
            and fp32_relative <= EQUIVALENCE_RELATIVE_L2_ERROR
        ),
        "configured_max_abs_error": EQUIVALENCE_MAX_ABS_ERROR,
        "configured_relative_l2_error": EQUIVALENCE_RELATIVE_L2_ERROR,
        "note": (
            "The candidate is the fixed A4 raw-PCA 1x1 Conv; the reference is "
            "the explicit (x - pca.mean_) @ pca.components_.T path."
        ),
    }


def _make_real_teacher_features(
    teacher: torch.nn.Module,
    dataset_root: Path,
    entries: Sequence[Tuple[str, str]],
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    if not entries:
        raise RuntimeError("A4 equivalence test could not find a train_local image")
    image_rel = entries[0][0]
    with Image.open(dataset_root / image_rel) as image_obj:
        image = image_obj.convert("RGB")
    image = image.resize(
        (a0.PCA_VIEW_WIDTH, a0.PCA_VIEW_HEIGHT),
        resample=Image.Resampling.BILINEAR,
    )
    image_tensor = t0.image_to_normalized_tensor(image).unsqueeze(0).to(device)
    teacher.eval()
    with torch.inference_mode():
        features = teacher.extract_features(image_tensor)
    return {layer: features[layer].detach().cpu() for layer in a0.A0_LAYER_ORDER}


def _channel_statistics(
    projection: FixedRawPCAProjection,
    features: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    output = projection(features[projection.layer].float())
    channel_mean = output.mean(dim=(0, 2, 3))
    channel_std = output.std(dim=(0, 2, 3), unbiased=False)
    return {
        "shape": list(output.shape),
        "mean": [float(value) for value in channel_mean.tolist()],
        "std": [float(value) for value in channel_std.tolist()],
        "mean_of_channel_std": float(channel_std.mean().item()),
        "global_rms": float(output.pow(2).mean().sqrt().item()),
    }


def build_equivalence_report(
    pcas: Mapping[str, Mapping[str, np.ndarray]],
    teacher: torch.nn.Module,
    dataset_root: Path,
    entries: Sequence[Tuple[str, str]],
    device: torch.device,
) -> Dict[str, Dict[str, object]]:
    """Run A4's random-input and real-teacher fixed-Conv audit."""

    projections = build_projection_bundle({}, pcas)
    real_features = _make_real_teacher_features(teacher, dataset_root, entries, device)
    report: Dict[str, Dict[str, object]] = {}
    for index, layer in enumerate(a0.A0_LAYER_ORDER):
        projection = projections[layer]
        generator = torch.Generator(device="cpu").manual_seed(4_001 + index)
        random_sample = torch.randn(
            2,
            a0.TEACHER_CHANNELS[layer],
            32,
            64,
            generator=generator,
        ) * 0.05
        random_result = _compare_projection(projection, random_sample)
        real_result = _compare_projection(projection, real_features[layer])
        passed = bool(random_result["passed"] and real_result["passed"])
        row: Dict[str, object] = {
            "reference": "explicit raw PCA transform (x - pca.mean_) @ components_.T",
            "candidate": "A4 fixed Conv2d(C_t,C_s,1), fused raw PCA",
            "random_input_scale": 0.05,
            "random_tensor": random_result,
            "real_teacher_feature": real_result,
            "channel_statistics_real_teacher": _channel_statistics(projection, real_features),
            "pca_mean_sha256": a0.numpy_arrays_sha256(np.asarray(pcas[layer]["mean_"])),
            "components_sha256": a0.numpy_arrays_sha256(
                np.asarray(pcas[layer]["components_"])
            ),
            "weight_sha256": projection.parameter_sha256(),
            "passed": passed,
        }
        report[layer] = row
        if not passed:
            raise RuntimeError(
                f"A4 raw PCA projection equivalence failed for {layer}: "
                f"random={random_result['max_abs_error']}, "
                f"real={real_result['max_abs_error']}"
            )
    return report


def check_projection_conv_equivalence(
    reference: a0.FixedPCAProjection,
    sample: torch.Tensor,
) -> Dict[str, object]:
    """Compatibility hook used by A0's delegated run loop."""

    if _EQUIVALENCE_REPORT is not None:
        layer = next(
            (
                candidate
                for candidate in a0.A0_LAYER_ORDER
                if a0.TEACHER_CHANNELS[candidate] == reference.c_in
                and a0.STUDENT_CHANNELS[candidate] == reference.d_out
            ),
            None,
        )
        if layer is not None:
            return _EQUIVALENCE_REPORT[layer]

    # This fallback is useful for direct unit-level calls before the full
    # preflight.  A0's reference object is constructed with identity scaler
    # buffers by A4's loader, so its fused coefficients are exactly the raw
    # PCA coefficients used here.
    del sample
    raise RuntimeError("A4 projection equivalence preflight was not run")


def _manifest_sha256(manifest: Mapping[str, object]) -> str:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _save_raw_pca_stage(
    args: Any,
    device: torch.device,
    dataset_root: Path,
    dataset_lock: Mapping[str, object],
    entries_by_split: Mapping[str, Sequence[Tuple[str, str]]],
) -> None:
    """Fit raw-feature IncrementalPCA on the locked deterministic manifest."""

    pca_dir = Path(args.pca_dir).resolve()
    pca_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pca_dir / "sampling_manifest.json"
    parameter_hashes_path = pca_dir / "pca_parameters_sha256.json"
    pca_paths = {layer: pca_dir / f"pca_{layer}.npz" for layer in a0.A0_LAYER_ORDER}
    expected_artifacts = [manifest_path, parameter_hashes_path, *pca_paths.values()]
    if any(path.exists() for path in expected_artifacts) and not args.force_pca:
        raise FileExistsError(
            f"A4 raw PCA artifacts already exist in {pca_dir}; use --force-pca to overwrite"
        )

    entries = list(entries_by_split["train_local"])
    if len(entries) != 2530:
        raise RuntimeError(f"Expected 2530 train_local images for A4 PCA, found {len(entries)}")

    teacher, _teacher_payload = a0.load_teacher_for_distillation(
        args.teacher_checkpoint,
        repo_dir=args.teacher_repo_dir,
        weights_path=args.teacher_weights_path,
        device=device,
        verify_checkpoint_file=True,
    )
    teacher.eval()
    layer_specs = {
        layer: {
            "height": a0.PCA_VIEW_SHAPES[layer][0],
            "width": a0.PCA_VIEW_SHAPES[layer][1],
            "channels": a0.TEACHER_CHANNELS[layer],
            "d_out": a0.STUDENT_CHANNELS[layer],
        }
        for layer in a0.A0_LAYER_ORDER
    }
    manifest = a0.build_pca_sampling_manifest(
        entries,
        total_tokens=args.pca_total_tokens,
        selection_seed=args.pca_selection_seed,
        layer_specs=layer_specs,
    )
    manifest["dataset_combined_manifest_sha256"] = dataset_lock["combined_manifest_sha256"]
    manifest["teacher_checkpoint_sha256"] = t0.verify_checkpoint_sidecar(
        args.teacher_checkpoint
    )
    manifest["teacher_checkpoint_path"] = str(Path(args.teacher_checkpoint).resolve())
    manifest["pca_view"] = {
        "height": a0.PCA_VIEW_HEIGHT,
        "width": a0.PCA_VIEW_WIDTH,
        "resample": "bilinear",
        "normalization": "imagenet_mean_std",
        "crop": "none",
        "flip": "none",
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    grouped = a0._group_selections_by_path(manifest)

    dataset = a0.PCASamplingViewDataset(
        dataset_root, entries, (a0.PCA_VIEW_HEIGHT, a0.PCA_VIEW_WIDTH)
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    pcas: Dict[str, IncrementalPCA] = {
        layer: IncrementalPCA(
            n_components=a0.STUDENT_CHANNELS[layer],
            batch_size=args.pca_batch_size,
            whiten=False,
        )
        for layer in a0.A0_LAYER_ORDER
    }
    buffers = {layer: a0._ChunkBuffer(args.pca_chunk_size) for layer in a0.A0_LAYER_ORDER}
    input_hashes = {layer: hashlib.sha256() for layer in a0.A0_LAYER_ORDER}
    sample_counts = {layer: 0 for layer in a0.A0_LAYER_ORDER}

    def consume_features(features: Mapping[str, torch.Tensor], path: str) -> None:
        for layer in a0.A0_LAYER_ORDER:
            selections = grouped[layer].get(path)
            if not selections:
                raise RuntimeError(f"A4 manifest has no selections for {path} in {layer}")
            feature_map = features[layer][0].float().cpu().numpy()
            tokens = np.stack(
                [feature_map[:, sel["row"], sel["col"]] for sel in selections]
            )
            chunk = buffers[layer].push(tokens)
            if chunk is not None:
                chunk = np.ascontiguousarray(chunk, dtype=np.float32)
                input_hashes[layer].update(chunk.tobytes(order="C"))
                pcas[layer].partial_fit(chunk.astype(np.float64))
                sample_counts[layer] += int(chunk.shape[0])

    print("[INFO] A4 PCA: fitting IncrementalPCA directly on raw teacher features")
    with torch.inference_mode():
        for images, paths in tqdm(loader, desc="A4 raw PCA", disable=False):
            images = images.to(device, non_blocking=True)
            features = teacher.extract_features(images)
            consume_features(features, paths[0])
    for layer in a0.A0_LAYER_ORDER:
        tail = buffers[layer].flush()
        if tail is not None:
            tail = np.ascontiguousarray(tail, dtype=np.float32)
            input_hashes[layer].update(tail.tobytes(order="C"))
            pcas[layer].partial_fit(tail.astype(np.float64))
            sample_counts[layer] += int(tail.shape[0])

    layers_record: Dict[str, object] = {}
    for layer in a0.A0_LAYER_ORDER:
        pca = pcas[layer]
        if sample_counts[layer] != args.pca_total_tokens:
            raise RuntimeError(
                f"A4 raw PCA token accounting failed for {layer}: "
                f"{sample_counts[layer]} != {args.pca_total_tokens}"
            )
        expected_components = (
            a0.STUDENT_CHANNELS[layer],
            a0.TEACHER_CHANNELS[layer],
        )
        if tuple(pca.components_.shape) != expected_components:
            raise RuntimeError(
                f"A4 raw PCA component shape mismatch for {layer}: "
                f"{tuple(pca.components_.shape)} != {expected_components}"
            )
        arrays = {
            "components_": np.ascontiguousarray(pca.components_),
            "mean_": np.ascontiguousarray(pca.mean_),
            "explained_variance_": np.ascontiguousarray(pca.explained_variance_),
            "explained_variance_ratio_": np.ascontiguousarray(
                pca.explained_variance_ratio_
            ),
            "singular_values_": np.ascontiguousarray(pca.singular_values_),
            "n_samples_seen_": np.asarray([pca.n_samples_seen_]),
        }
        np.savez_compressed(pca_paths[layer], **arrays)
        layers_record[layer] = {
            "feature_input_sha256": input_hashes[layer].hexdigest(),
            "pca_sha256": a0.numpy_arrays_sha256(*arrays.values()),
            "d_out": int(a0.STUDENT_CHANNELS[layer]),
            "teacher_channels": int(a0.TEACHER_CHANNELS[layer]),
            "n_tokens": int(args.pca_total_tokens),
            "explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()
            ),
            "explained_variance_ratio_head": [
                float(value)
                for value in pca.explained_variance_ratio_[
                    : min(10, pca.explained_variance_ratio_.size)
                ]
            ],
            "pca_path": str(pca_paths[layer]),
        }

    parameter_record: Dict[str, object] = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "sampling_manifest_path": str(manifest_path),
        "sampling_manifest_sha256": manifest["manifest_sha256"],
        "dataset_combined_manifest_sha256": dataset_lock["combined_manifest_sha256"],
        "teacher_checkpoint_sha256": manifest["teacher_checkpoint_sha256"],
        "fit_algorithm": (
            "IncrementalPCA.partial_fit (raw teacher tokens, no StandardScaler)"
        ),
        "pca_fit_space": "raw_teacher_feature_space",
        "standard_scaler_used": False,
        "pca_dtype": "float64 fit from float32 feature input",
        "whiten": False,
        "layers": layers_record,
    }
    t0.write_json_atomic(manifest_path, manifest)
    t0.write_json_atomic(parameter_hashes_path, parameter_record)
    print("[DONE] A4 raw PCA stage complete")
    for layer in a0.A0_LAYER_ORDER:
        record = layers_record[layer]
        print(
            f"   - {layer}: d={record['d_out']}, "
            f"explained_var_ratio={record['explained_variance_ratio_sum']:.4f}, "
            f"feature_input_sha256={record['feature_input_sha256'][:16]}"
        )
    print(f"   - manifest SHA-256: {manifest['manifest_sha256']}")


def load_raw_pca_parameters(
    pca_dir: Path,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Dict[str, object]]:
    """Load and verify A4 raw PCA artifacts through A0's runner interface."""

    pca_dir = Path(pca_dir).resolve()
    record_path = pca_dir / "pca_parameters_sha256.json"
    manifest_path = pca_dir / "sampling_manifest.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"Missing A4 PCA parameter record: {record_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing A4 sampling manifest: {manifest_path}")
    with record_path.open("r", encoding="utf-8") as file_obj:
        record = json.load(file_obj)
    with manifest_path.open("r", encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    if record.get("experiment") != EXPERIMENT or record.get("standard_scaler_used") is not False:
        raise RuntimeError(
            "The selected PCA directory is not an A4 raw-PCA artifact; "
            "A4 must not read A0 StandardScaler/PCA artifacts"
        )
    if manifest.get("manifest_sha256") != _manifest_sha256(manifest):
        raise RuntimeError("A4 sampling manifest SHA-256 verification failed")
    if record.get("sampling_manifest_sha256") != manifest.get("manifest_sha256"):
        raise RuntimeError("A4 PCA record refers to a different sampling manifest")

    # A0's delegated runner expects a scaler-shaped object.  These identity
    # arrays are an interface shim only; the A4 projection ignores them and
    # uses the raw PCA mean/components exclusively.
    scalers: Dict[str, Dict[str, np.ndarray]] = {}
    pcas: Dict[str, Dict[str, np.ndarray]] = {}
    for layer in a0.A0_LAYER_ORDER:
        path = pca_dir / f"pca_{layer}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing A4 PCA artifact for {layer}: {path}")
        with np.load(path) as data:
            pcas[layer] = {key: np.asarray(data[key]) for key in data.files}
        expected_shape = (
            a0.STUDENT_CHANNELS[layer],
            a0.TEACHER_CHANNELS[layer],
        )
        if tuple(pcas[layer]["components_"].shape) != expected_shape:
            raise RuntimeError(
                f"A4 PCA components for {layer} have shape "
                f"{pcas[layer]['components_'].shape}, expected {expected_shape}"
            )
        arrays = {
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
        expected_hash = record.get("layers", {}).get(layer, {}).get("pca_sha256")
        if a0.numpy_arrays_sha256(*arrays.values()) != expected_hash:
            raise RuntimeError(f"A4 raw PCA hash verification failed for {layer}")
        c_in = a0.TEACHER_CHANNELS[layer]
        scalers[layer] = {
            "mean_": np.zeros(c_in, dtype=np.float64),
            "var_": np.ones(c_in, dtype=np.float64),
            "scale_": np.ones(c_in, dtype=np.float64),
            "n_samples_seen_": np.asarray([record["layers"][layer]["n_tokens"]]),
        }
    return scalers, pcas, record


def a4_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    run_dir = Path(output_dir).resolve() / "A4" / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "config": run_dir / "config.json",
        "feature_taps": run_dir / "feature_taps.json",
        "pretrain_last": run_dir / "a4_pretrain_last.pth",
        "pretrain_history": run_dir / "a4_pretrain_history.json",
        "pretrain_gradients": run_dir / "a4_pretrain_gradient_norms.jsonl",
        "pretrain_snapshots": run_dir / "pretrain_snapshots",
        "probe_last": run_dir / "a4_probe_last.pth",
        "probe_history": run_dir / "a4_probe_history.json",
        "best_probe": run_dir / "a4_probe_mobilenetv2_raspp_best.pth",
        "dev_metrics": run_dir / "a4_dev_metrics.json",
        "efficiency": run_dir / "efficiency.json",
        "per_image": run_dir / "a4_dev_per_image_confusion.jsonl",
        "projection_equivalence": run_dir / "projection_equivalence.json",
    }


def build_config(*args, **kwargs):
    config = _A0_BUILD_CONFIG(*args, **kwargs)
    config.update(
        {
            "experiment": EXPERIMENT,
            "projection_implementation": (
                "fixed 1x1 Conv2d fused from raw-feature PCA mean and components"
            ),
            "projection_reference": "explicit raw PCA: (x - pca.mean_) @ components_.T",
            "projection_trainable": False,
            "pca_fit_space": "raw_teacher_feature_space",
            "standard_scaler_used": False,
            "pca_refit": True,
            "pca_resampling": False,
            "pca_components_used": True,
            "equivalence_tolerance_max_abs": EQUIVALENCE_MAX_ABS_ERROR,
            "equivalence_tolerance_relative_l2": EQUIVALENCE_RELATIVE_L2_ERROR,
        }
    )
    return config


def build_pretrain_checkpoint(*args, **kwargs):
    payload = _A0_BUILD_PRETRAIN_CHECKPOINT(*args, **kwargs)
    payload.update(
        {
            "experiment": EXPERIMENT,
            "artifact_type": ARTIFACT_TYPE_PRETRAIN,
            "initialization": "weights=None + A4 raw-feature PCA pretrain",
            "projection": "fixed raw-feature PCA fused into 1x1 Conv",
            "standard_scaler_used": False,
        }
    )
    return payload


def build_probe_best_checkpoint(*args, **kwargs):
    payload = _A0_BUILD_PROBE_BEST_CHECKPOINT(*args, **kwargs)
    payload.update(
        {
            "experiment": EXPERIMENT,
            "artifact_type": ARTIFACT_TYPE_PROBE,
            "initialization": "weights=None + A4 raw-feature PCA feature pretrain",
            "projection": "fixed raw-feature PCA fused into 1x1 Conv",
            "standard_scaler_used": False,
        }
    )
    return payload


def _patch_a0_hooks() -> None:
    """Redirect A0 globals while retaining its tested training lifecycle."""

    a0.__dict__["__file__"] = str(Path(__file__).resolve())
    a0.EXPERIMENT = EXPERIMENT
    a0.ARTIFACT_TYPE_PRETRAIN = ARTIFACT_TYPE_PRETRAIN
    a0.ARTIFACT_TYPE_PROBE = ARTIFACT_TYPE_PROBE
    a0.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    a0.DEFAULT_PCA_DIR = DEFAULT_PCA_DIR
    a0.a0_paths = a4_paths
    a0.build_projection_bundle = build_projection_bundle
    a0.check_projection_conv_equivalence = check_projection_conv_equivalence
    a0.build_config = build_config
    a0.build_pretrain_checkpoint = build_pretrain_checkpoint
    a0.build_probe_best_checkpoint = build_probe_best_checkpoint
    a0.load_pca_parameters = load_raw_pca_parameters
    a0.run_pca_stage = _save_raw_pca_stage


def _rewrite_final_metrics(args: Any) -> None:
    metrics_path = a4_paths(args.output_dir, args.seed)["dev_metrics"]
    if not metrics_path.is_file():
        return
    with metrics_path.open("r", encoding="utf-8") as file_obj:
        results = json.load(file_obj)
    results["experiment"] = EXPERIMENT
    results["protocol"] = (
        "Scratch MobileNetV2 backbone trained label-free for 40k steps with "
        "three fixed raw-feature PCA teacher-side projections (no StandardScaler), "
        "fused into 1x1 Conv2d layers, followed by the common 40k-step "
        "frozen-backbone 19-class R-ASPP probe. Best checkpoint is selected by "
        "dev_local mIoU; test_local is not evaluated."
    )
    results["model"] = {
        **results.get("model", {}),
        "initialization": "weights=None + A4 raw-feature PCA feature pretrain",
        "projection": "fixed raw-feature PCA (no StandardScaler), fused 1x1 Conv",
        "standard_scaler_used": False,
    }
    results["pca_fit_space"] = "raw_teacher_feature_space"
    results["standard_scaler_used"] = False
    t0.write_json_atomic(metrics_path, results)


def _write_config_before_training(
    args: Any,
    pca_record: Mapping[str, object],
    projection: nn.ModuleDict,
) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    accumulation_steps = a0.s2_0_server.effective_accumulation_steps(args, world_size)
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    teacher_hash = t0.verify_checkpoint_sidecar(args.teacher_checkpoint)
    sampling_hash = str(pca_record.get("sampling_manifest_sha256", ""))
    projection_hashes = {
        layer: projection[layer].parameter_sha256() for layer in a0.A0_LAYER_ORDER
    }
    config = build_config(
        args,
        accumulation_steps,
        world_size,
        amp_enabled,
        teacher_hash,
        sampling_hash,
        pca_record,
        projection_hashes,
    )
    paths = a4_paths(args.output_dir, args.seed)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    t0.write_json_atomic(paths["config"], config)


def parse_args() -> Any:
    _patch_a0_hooks()
    return a0.parse_args()


def main() -> None:
    global _EQUIVALENCE_REPORT
    _patch_a0_hooks()
    args = a0.parse_args()

    if args.stage == "pca":
        # The patched A0 runner executes A4's raw-PCA stage on rank 0 and
        # retains its collective/error propagation and ordered teardown.
        a0.run_training(args)
        return

    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

    dataset_root = args.dataset_root.resolve()
    _dataset_lock, entries_by_split = t0.validate_dataset_lock(dataset_root)
    scalers, pcas, pca_record = load_raw_pca_parameters(args.pca_dir.resolve())
    teacher, _teacher_payload = a0.load_teacher_for_distillation(
        args.teacher_checkpoint,
        repo_dir=args.teacher_repo_dir,
        weights_path=args.teacher_weights_path,
        device=device,
        verify_checkpoint_file=True,
    )
    teacher.eval()
    _EQUIVALENCE_REPORT = build_equivalence_report(
        pcas,
        teacher,
        dataset_root,
        entries_by_split["train_local"],
        device,
    )
    print(
        "[OK] A4 raw PCA projection equivalence checks:",
        {layer: _EQUIVALENCE_REPORT[layer]["passed"] for layer in a0.A0_LAYER_ORDER},
    )
    del teacher
    _write_config_before_training(
        args,
        pca_record,
        build_projection_bundle(scalers, pcas),
    )
    a0.run_training(args)
    _rewrite_final_metrics(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
