"""
Experiment A3: fixed random orthogonal low-rank teacher projection.

The A3 protocol is the same as A0 in every training-related respect:

    teacher side : the locked T1 DINOv3 teacher
    normalization : the shared per-layer StandardScaler fitted for A0
    projection : a fixed random row-orthogonal matrix, T -> S
    student side : scratch MobileNetV2 backbone, no adapter
    loss : three-layer dense feature MSE, followed by the common R-ASPP probe

Only the PCA components are replaced.  For a teacher feature vector x, A3
uses

    y = ((x - scaler_mean) / scaler_scale) @ Q.T,

where Q has shape [student_channels, teacher_channels] and QQ.T is the
identity.  The fixed transform is evaluated as a 1x1 convolution during
training; the explicit per-pixel expression is retained for the numerical
equivalence audit.

The random matrices are independent of the training seed.  Layer ``i`` uses
the locked construction seed ``42 + i`` and is generated once from a NumPy
RandomState, followed by a complete QR decomposition and
``Q_full.T[:student_channels]``.  The matrices and their metadata are stored
alongside the shared PCA artifacts when the A3 run starts.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import dino as t0
import dino_a0_server as a0


EXPERIMENT = "A3"
ARTIFACT_TYPE_PRETRAIN = "a3_pretrain_mobilenetv2_backbone_fixed_random_orthogonal"
ARTIFACT_TYPE_PROBE = "a3_probe_mobilenetv2_raspp_fixed_random_orthogonal"
ARTIFACT_FORMAT_VERSION = 1

RANDOM_PROJECTION_SEED_BASE = 42
ORTHOGONALITY_TOLERANCE = 1e-5
EQUIVALENCE_MAX_ABS_ERROR = 1.5e-5
EQUIVALENCE_RELATIVE_L2_ERROR = 2e-6

DEFAULT_OUTPUT_DIR = a0.DEFAULT_OUTPUT_DIR
DEFAULT_PCA_DIR = a0.DEFAULT_PCA_DIR
DEFAULT_TEACHER_CHECKPOINT = a0.DEFAULT_TEACHER_CHECKPOINT

_EQUIVALENCE_REPORT: Optional[Dict[str, Dict[str, object]]] = None
_RANDOM_PROJECTION_METADATA: Dict[str, Dict[str, object]] = {}

_A0_BUILD_CONFIG = a0.build_config
_A0_BUILD_PRETRAIN_CHECKPOINT = a0.build_pretrain_checkpoint
_A0_BUILD_PROBE_BEST_CHECKPOINT = a0.build_probe_best_checkpoint


class FixedRandomOrthoProjection(nn.Module):
    """Fixed StandardScaler + random row-orthogonal projection.

    The registered tensors are buffers rather than parameters, so this module
    cannot accidentally enter the optimizer.  ``weight`` and ``bias`` are
    the fused convolution coefficients, while ``random_matrix`` is retained
    for the explicit audit path and for reproducibility metadata.
    """

    def __init__(
        self,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        random_matrix: np.ndarray,
        seed: int,
        layer: str,
        orthogonality_error: float,
    ) -> None:
        super().__init__()
        if (
            scaler_mean.ndim != 1
            or scaler_scale.ndim != 1
            or random_matrix.ndim != 2
            or scaler_mean.shape != scaler_scale.shape
            or random_matrix.shape[1] != scaler_mean.shape[0]
        ):
            raise RuntimeError("A3 projection parameter shapes are inconsistent")

        mean = torch.as_tensor(scaler_mean, dtype=torch.float32).contiguous()
        scale = torch.as_tensor(scaler_scale, dtype=torch.float32).contiguous()
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.as_tensor(random_matrix, dtype=torch.float32).contiguous()
        weight = q / scale.view(1, -1)
        bias = -(mean / scale) @ q.t()

        self.register_buffer("scaler_mean", mean.view(1, 1, -1))
        self.register_buffer("scaler_scale", scale.view(1, 1, -1))
        self.register_buffer("random_matrix", q)
        self.register_buffer("weight", weight.contiguous())
        self.register_buffer("bias", bias.contiguous())
        self.c_in = int(random_matrix.shape[1])
        self.d_out = int(random_matrix.shape[0])
        self.seed = int(seed)
        self.layer = str(layer)
        self.orthogonality_error = float(orthogonality_error)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.c_in:
            raise RuntimeError(
                f"A3 fixed projection expects [B,{self.c_in},H,W], got {tuple(x.shape)}"
            )
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def explicit_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reference expression used by the A3 Conv audit."""

        if x.ndim != 4 or x.shape[1] != self.c_in:
            raise RuntimeError(
                f"A3 explicit projection expects [B,{self.c_in},H,W], got {tuple(x.shape)}"
            )
        tokens = x.permute(0, 2, 3, 1)
        tokens = (tokens - self.scaler_mean) / self.scaler_scale
        projected = torch.matmul(tokens, self.random_matrix.t())
        return projected.permute(0, 3, 1, 2)

    def parameter_sha256(self) -> str:
        return a0.numpy_arrays_sha256(
            self.scaler_mean.detach().cpu().numpy(),
            self.scaler_scale.detach().cpu().numpy(),
            self.random_matrix.detach().cpu().numpy(),
            self.weight.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy(),
        )

    def random_matrix_sha256(self) -> str:
        return a0.numpy_arrays_sha256(self.random_matrix.detach().cpu().numpy())


def _make_random_orthogonal_matrix(
    c_in: int,
    d_out: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    """Generate the locked A3 ``Q_full.T[:d_out]`` matrix."""

    if d_out > c_in:
        raise RuntimeError(
            f"A3 requires d_out <= c_in, got d_out={d_out}, c_in={c_in}"
        )
    rng = np.random.RandomState(int(seed))
    gaussian = rng.standard_normal((c_in, d_out)).astype(np.float64)
    q_full, _r = np.linalg.qr(gaussian, mode="complete")
    random_matrix = np.ascontiguousarray(q_full.T[:d_out, :], dtype=np.float64)
    identity = np.eye(d_out, dtype=np.float64)
    orthogonality_error = float(
        np.linalg.norm(random_matrix @ random_matrix.T - identity, ord="fro")
    )
    if orthogonality_error > ORTHOGONALITY_TOLERANCE:
        raise RuntimeError(
            f"A3 random projection is not sufficiently orthogonal: "
            f"seed={seed}, error={orthogonality_error}"
        )
    return random_matrix, orthogonality_error


def build_projection_bundle(
    scalers: Mapping[str, Mapping[str, np.ndarray]],
    pcas: Mapping[str, Mapping[str, np.ndarray]],
) -> nn.ModuleDict:
    """Build the fixed StandardScaler + random-orthogonal projections.

    ``pcas`` is accepted for compatibility with A0's delegated training
    runner, but no PCA mean or component is read or used by A3.
    """

    del pcas
    projections: Dict[str, FixedRandomOrthoProjection] = {}
    metadata: Dict[str, Dict[str, object]] = {}
    for layer_index, layer in enumerate(a0.A0_LAYER_ORDER):
        scaler = scalers[layer]
        expected_c_in = a0.TEACHER_CHANNELS[layer]
        expected_d_out = a0.STUDENT_CHANNELS[layer]
        if scaler["mean_"].shape != (expected_c_in,) or scaler["scale_"].shape != (
            expected_c_in,
        ):
            raise RuntimeError(
                f"A3 scaler shape mismatch for {layer}: "
                f"mean={scaler['mean_'].shape}, scale={scaler['scale_'].shape}, "
                f"expected={(expected_c_in,)}"
            )

        construction_seed = RANDOM_PROJECTION_SEED_BASE + layer_index
        random_matrix, orthogonality_error = _make_random_orthogonal_matrix(
            expected_c_in, expected_d_out, construction_seed
        )
        projection = FixedRandomOrthoProjection(
            scaler_mean=scaler["mean_"],
            scaler_scale=scaler["scale_"],
            random_matrix=random_matrix,
            seed=construction_seed,
            layer=layer,
            orthogonality_error=orthogonality_error,
        )
        if projection.c_in != expected_c_in or projection.d_out != expected_d_out:
            raise RuntimeError(
                f"A3 projection contract mismatch for {layer}: "
                f"got d_out={projection.d_out}, c_in={projection.c_in}, "
                f"expected d_out={expected_d_out}, c_in={expected_c_in}"
            )
        if any(parameter.requires_grad for parameter in projection.parameters()):
            raise RuntimeError("A3 fixed random projection unexpectedly has parameters")
        projections[layer] = projection
        metadata[layer] = {
            "layer": layer,
            "layer_index": layer_index,
            "seed": construction_seed,
            "rng": "numpy.random.RandomState",
            "gaussian_shape": [expected_c_in, expected_d_out],
            "q_shape": [expected_d_out, expected_c_in],
            "construction": "complete_qr(Q_full.T[:d_out])",
            "orthogonality_error_frobenius": orthogonality_error,
            "orthogonality_tolerance": ORTHOGONALITY_TOLERANCE,
            "random_matrix_sha256": projection.random_matrix_sha256(),
            "projection_parameter_sha256": projection.parameter_sha256(),
            "scaler_mean_sha256": a0.numpy_arrays_sha256(scaler["mean_"]),
            "scaler_scale_sha256": a0.numpy_arrays_sha256(scaler["scale_"]),
        }

    _RANDOM_PROJECTION_METADATA.clear()
    _RANDOM_PROJECTION_METADATA.update(metadata)
    return nn.ModuleDict(projections)


def _compare_projection(
    projection: FixedRandomOrthoProjection,
    sample: torch.Tensor,
) -> Dict[str, object]:
    """Compare the fused Conv with A3's explicit StandardScaler + Q path."""

    sample32 = sample.detach().cpu().float()
    with torch.inference_mode():
        explicit32 = projection.explicit_forward(sample32).float()
        fused32 = projection(sample32).float()
    difference32 = (fused32 - explicit32).abs()
    denominator32 = float(explicit32.norm().clamp_min(1e-12).item())

    projection64 = copy.deepcopy(projection).double()
    sample64 = sample32.double()
    weight64 = projection64.random_matrix / projection64.scaler_scale[0, 0, :].view(1, -1)
    bias64 = -(
        projection64.scaler_mean[0, 0, :] / projection64.scaler_scale[0, 0, :]
    ) @ projection64.random_matrix.t()
    with torch.inference_mode():
        explicit64 = projection64.explicit_forward(sample64)
        fused64 = F.conv2d(sample64, weight64[:, :, None, None], bias64)
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
            "The candidate is the fixed A3 1x1 Conv; the reference is the "
            "explicit StandardScaler + random row-orthogonal projection."
        ),
    }


def _make_real_teacher_features(
    teacher: torch.nn.Module,
    dataset_root: Path,
    entries: Sequence[Tuple[str, str]],
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    if not entries:
        raise RuntimeError("A3 equivalence test could not find a train_local image")
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


def build_equivalence_report(
    scalers: Mapping[str, Mapping[str, np.ndarray]],
    teacher: torch.nn.Module,
    dataset_root: Path,
    entries: Sequence[Tuple[str, str]],
    device: torch.device,
) -> Dict[str, Dict[str, object]]:
    """Run a fixed-Conv audit on random tensors and real teacher features."""

    projections = build_projection_bundle(scalers, {})
    real_features = _make_real_teacher_features(
        teacher, dataset_root, entries, device
    )
    report: Dict[str, Dict[str, object]] = {}
    for index, layer in enumerate(a0.A0_LAYER_ORDER):
        projection = projections[layer]
        generator = torch.Generator(device="cpu").manual_seed(2_001 + index)
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
        layer_metadata = copy.deepcopy(_RANDOM_PROJECTION_METADATA[layer])
        layer_metadata.update(
            {
                "reference": "explicit StandardScaler + random row-orthogonal Q",
                "candidate": "A3 fixed Conv2d(C_t,C_s,1)",
                "random_input_scale": 0.05,
                "random_tensor": random_result,
                "real_teacher_feature": real_result,
                "max_abs_error": max(
                    float(random_result["max_abs_error"]),
                    float(real_result["max_abs_error"]),
                ),
                "passed": passed,
            }
        )
        report[layer] = layer_metadata
        if not passed:
            raise RuntimeError(
                f"A3 fixed projection equivalence failed for {layer}: "
                f"random={random_result['max_abs_error']}, "
                f"real={real_result['max_abs_error']}"
            )
    return report


def check_projection_conv_equivalence(
    reference: a0.FixedPCAProjection,
    sample: torch.Tensor,
) -> Dict[str, object]:
    """Compatibility hook for A0's delegated run loop."""

    del sample
    if _EQUIVALENCE_REPORT is None:
        raise RuntimeError("A3 projection equivalence preflight was not run")
    layer = next(
        (
            candidate
            for candidate in a0.A0_LAYER_ORDER
            if a0.TEACHER_CHANNELS[candidate] == reference.c_in
            and a0.STUDENT_CHANNELS[candidate] == reference.d_out
        ),
        None,
    )
    if layer is None:
        raise RuntimeError(
            f"Cannot identify A3 layer for projection shape "
            f"[{reference.c_in}->{reference.d_out}]"
        )
    return _EQUIVALENCE_REPORT[layer]


def _save_random_projection_artifacts(
    pca_dir: Path,
    projections: nn.ModuleDict,
) -> None:
    """Persist the fixed Q matrices next to the shared PCA artifacts."""

    pca_dir = Path(pca_dir).resolve()
    pca_dir.mkdir(parents=True, exist_ok=True)
    for layer in a0.A0_LAYER_ORDER:
        projection = projections[layer]
        path = pca_dir / f"a3_random_orthogonal_{layer}.npz"
        arrays = {
            "random_matrix": projection.random_matrix.detach().cpu().numpy(),
            "scaler_mean": projection.scaler_mean.detach().cpu().numpy().reshape(-1),
            "scaler_scale": projection.scaler_scale.detach().cpu().numpy().reshape(-1),
            "seed": np.asarray([projection.seed], dtype=np.int64),
            "orthogonality_error_frobenius": np.asarray(
                [projection.orthogonality_error], dtype=np.float64
            ),
        }
        if path.is_file():
            with np.load(path) as existing:
                for name, value in arrays.items():
                    if name not in existing.files or not np.array_equal(existing[name], value):
                        raise RuntimeError(
                            f"Existing A3 random projection artifact differs for {layer}: {path}"
                        )
            continue
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(temp_path, **arrays)
        os.replace(temp_path, path)

    manifest_path = pca_dir / "a3_random_orthogonal_manifest.json"
    manifest = {
        "experiment": EXPERIMENT,
        "seed_base": RANDOM_PROJECTION_SEED_BASE,
        "layers": copy.deepcopy(_RANDOM_PROJECTION_METADATA),
        "artifacts": {
            layer: str(pca_dir / f"a3_random_orthogonal_{layer}.npz")
            for layer in a0.A0_LAYER_ORDER
        },
    }
    t0.write_json_atomic(manifest_path, manifest)


def a3_paths(output_dir: Path, seed: int) -> Dict[str, Path]:
    run_dir = output_dir.resolve() / "A3" / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "config": run_dir / "config.json",
        "feature_taps": run_dir / "feature_taps.json",
        "pretrain_last": run_dir / "a3_pretrain_last.pth",
        "pretrain_history": run_dir / "a3_pretrain_history.json",
        "pretrain_gradients": run_dir / "a3_pretrain_gradient_norms.jsonl",
        "pretrain_snapshots": run_dir / "pretrain_snapshots",
        "probe_last": run_dir / "a3_probe_last.pth",
        "probe_history": run_dir / "a3_probe_history.json",
        "best_probe": run_dir / "a3_probe_mobilenetv2_raspp_best.pth",
        "dev_metrics": run_dir / "a3_dev_metrics.json",
        "efficiency": run_dir / "efficiency.json",
        "per_image": run_dir / "a3_dev_per_image_confusion.jsonl",
        "projection_equivalence": run_dir / "projection_equivalence.json",
    }


def build_config(*args, **kwargs):
    config = _A0_BUILD_CONFIG(*args, **kwargs)
    config.update(
        {
            "experiment": EXPERIMENT,
            "projection_implementation": (
                "fixed 1x1 Conv2d fused from shared StandardScaler and "
                "fixed random row-orthogonal T-to-S matrix"
            ),
            "projection_reference": "explicit StandardScaler + random Q",
            "projection_trainable": False,
            "random_projection_seed_base": RANDOM_PROJECTION_SEED_BASE,
            "random_projection_rng": "numpy.random.RandomState",
            "random_projection_construction": (
                "Gaussian[C_t,d_l] -> complete QR -> Q_full.T[:d_l]"
            ),
            "random_projection_orthogonality_tolerance": ORTHOGONALITY_TOLERANCE,
            "random_projection_layers": copy.deepcopy(_RANDOM_PROJECTION_METADATA),
            "pca_refit": False,
            "pca_resampling": False,
            "pca_components_used": False,
        }
    )
    return config


def build_pretrain_checkpoint(*args, **kwargs):
    payload = _A0_BUILD_PRETRAIN_CHECKPOINT(*args, **kwargs)
    payload.update(
        {
            "experiment": EXPERIMENT,
            "artifact_type": ARTIFACT_TYPE_PRETRAIN,
            "projection": "fixed StandardScaler + random row-orthogonal T-to-S",
            "random_projection_seed_base": RANDOM_PROJECTION_SEED_BASE,
            "random_projection_layers": copy.deepcopy(_RANDOM_PROJECTION_METADATA),
        }
    )
    return payload


def build_probe_best_checkpoint(*args, **kwargs):
    payload = _A0_BUILD_PROBE_BEST_CHECKPOINT(*args, **kwargs)
    payload.update(
        {
            "experiment": EXPERIMENT,
            "artifact_type": ARTIFACT_TYPE_PROBE,
            "initialization": "weights=None + A3 fixed-random-orthogonal feature pretrain",
            "projection": "fixed StandardScaler + random row-orthogonal T-to-S",
            "random_projection_seed_base": RANDOM_PROJECTION_SEED_BASE,
            "random_projection_layers": copy.deepcopy(_RANDOM_PROJECTION_METADATA),
        }
    )
    return payload


def _patch_a0_hooks() -> None:
    # Redirect A0's module-global artifact and path helpers so delegated
    # training writes an auditable A3 run rather than an A0 run.
    a0.__dict__["__file__"] = str(Path(__file__).resolve())
    a0.EXPERIMENT = EXPERIMENT
    a0.ARTIFACT_TYPE_PRETRAIN = ARTIFACT_TYPE_PRETRAIN
    a0.ARTIFACT_TYPE_PROBE = ARTIFACT_TYPE_PROBE
    a0.a0_paths = a3_paths
    a0.build_projection_bundle = build_projection_bundle
    a0.check_projection_conv_equivalence = check_projection_conv_equivalence
    a0.build_config = build_config
    a0.build_pretrain_checkpoint = build_pretrain_checkpoint
    a0.build_probe_best_checkpoint = build_probe_best_checkpoint


def _rewrite_final_metrics(args) -> None:
    metrics_path = a3_paths(args.output_dir, args.seed)["dev_metrics"]
    if not metrics_path.is_file():
        return
    with metrics_path.open("r", encoding="utf-8") as file_obj:
        results = json.load(file_obj)
    results["experiment"] = EXPERIMENT
    results["protocol"] = (
        "Scratch MobileNetV2 backbone trained label-free for 40k steps with "
        "three fixed StandardScaler + random row-orthogonal teacher-side "
        "projections (seeds 42/43/44), followed by the common 40k-step "
        "frozen-backbone 19-class R-ASPP probe. Best checkpoint is selected "
        "by dev_local mIoU; test_local is not evaluated."
    )
    results["model"] = {
        **results.get("model", {}),
        "initialization": "weights=None + A3 fixed-random-orthogonal feature pretrain",
        "projection": "fixed StandardScaler + random row-orthogonal T-to-S",
        "projection_trainable": False,
        "random_projection_seed_base": RANDOM_PROJECTION_SEED_BASE,
    }
    results["random_projection"] = copy.deepcopy(_RANDOM_PROJECTION_METADATA)
    t0.write_json_atomic(metrics_path, results)


def _write_config_before_training(
    args,
    pca_record: Mapping[str, object],
    projection: nn.ModuleDict,
) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    accumulation_steps = a0.s2_0_server.effective_accumulation_steps(args, world_size)
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device("cuda", local_rank)
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
    paths = a3_paths(args.output_dir, args.seed)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    t0.write_json_atomic(paths["config"], config)
    if int(os.environ.get("RANK", "0")) == 0:
        _save_random_projection_artifacts(args.pca_dir, projection)


def parse_args() -> object:
    args = a0.parse_args()
    if args.stage == "pca":
        raise RuntimeError(
            "A3 does not refit PCA or StandardScaler. Run "
            "dino_a0_server.py --stage pca once, then point A3 --pca-dir "
            "at the shared pca_shared directory."
        )
    return args


def main() -> None:
    global _EQUIVALENCE_REPORT
    _patch_a0_hooks()
    args = parse_args()

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
    scalers, _pcas, pca_record = a0.load_pca_parameters(args.pca_dir.resolve())
    teacher, _teacher_payload = a0.load_teacher_for_distillation(
        args.teacher_checkpoint,
        repo_dir=args.teacher_repo_dir,
        weights_path=args.teacher_weights_path,
        device=device,
        verify_checkpoint_file=True,
    )
    teacher.eval()
    _EQUIVALENCE_REPORT = build_equivalence_report(
        scalers,
        teacher,
        dataset_root,
        entries_by_split["train_local"],
        device,
    )
    print(
        "[OK] A3 fixed random projection equivalence checks:",
        {
            layer: _EQUIVALENCE_REPORT[layer]["passed"]
            for layer in a0.A0_LAYER_ORDER
        },
    )
    del teacher
    projection = build_projection_bundle(scalers, {}).to(device)
    _write_config_before_training(args, pca_record, projection)
    a0.run_training(args)
    _rewrite_final_metrics(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
