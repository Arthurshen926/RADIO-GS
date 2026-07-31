#!/usr/bin/env python3
"""Evaluate the official Easy3D checkpoint on AGILE3D ScanNet40 objects.

The Easy3D repository does not ship a quantitative evaluation entry point.
This adapter keeps its released model and preprocessing, then applies the
frozen AGILE3D single-object interaction/point-IoU contract.  Scene shards
make the long GPU run resumable without overwriting other experiment output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .protocol import Agile3DObject, load_official_object_list


EASY3D_REPOSITORY = "https://github.com/facebookresearch/easy3d"
EASY3D_CHECKPOINT_URL = (
    "https://github.com/facebookresearch/easy3d/releases/download/v1.0/"
    "pretrained_easy3d.pth"
)
EASY3D_CHECKPOINT_SHA256 = (
    "4a13d16ba2f2470031287812dbbdf1ec6aa14097cb3738e0fe596bb708dc475f"
)
EASY3D_AUDITED_COMMIT = "b3f5bd70defaa9a601edb0975802775b056c784a"
AGILE3D_AUDITED_COMMIT = "b73638da41edbabe52a1b578d52ddeb8fa552173"
OFFICIAL_OBJECT_IDS_SHA256 = (
    "d734230755fdde72ee04f8ca199b15c19f330233588eba1e24021fe36459a037"
)
OFFICIAL_OBJECT_CLASSES_SHA256 = (
    "f2bf7241e0cbad22056c9fe9b029818ac3b7d1d9846a96399eee68bac2fef537"
)
PAPER_IOU = {
    "IoU@1": 0.682,
    "IoU@2": 0.746,
    "IoU@3": 0.773,
    "IoU@5": 0.796,
    "IoU@10": 0.817,
}
IOU_CLICK_COUNTS = (1, 2, 3, 5, 10, 15)
NOC_THRESHOLDS = (0.50, 0.65, 0.80, 0.85, 0.90)
EVALUATOR_SCHEMA_VERSION = "easy3d-agile3d-protocol-audit-v5"
EASY3D_PREPROCESSING_ID = (
    "official_Easy3D_DataLoader_worker1thread_stable_last_write"
)
OFFICIAL_WORKER_CACHE_SCHEMA = "official-easy3d-worker-preprocessing-v1"
OFFICIAL_VOXEL_DATASET_SHA256 = (
    "7cef7619b21c5e93f0035fbe444d151d7d4adef6b1bb20157778fb003b2c11a8"
)


@dataclass(frozen=True)
class Easy3DScene:
    """One scene quantized exactly like Easy3D's released VoxelDataset."""

    coordinates: np.ndarray
    features: np.ndarray
    voxel_labels: np.ndarray
    voxel_valid: np.ndarray
    point_labels: np.ndarray
    inverse_map: np.ndarray
    quantization_diagnostics: Mapping[str, float | int]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: str | Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(path)), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        values = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(str(values.shape).encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def runtime_versions() -> dict[str, Any]:
    """Record the exact isolated runtime used for checkpoint inference."""

    import torch

    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "--id=0",
            ],
            text=True,
        ).splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        driver = None
    packages = {}
    for name in ("torchvision", "spconv-cu121", "numpy", "scipy", "plyfile"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_built_cuda": torch.version.cuda,
        "nvidia_driver": driver,
        "packages": packages,
    }


def existing_scene_shard_matches(
    path: str | Path,
    expected: Mapping[str, Any],
    *,
    no_resume: bool = False,
) -> bool:
    """Resume an identical shard or fail closed without overwriting it."""

    shard = Path(path)
    if not shard.exists():
        return False
    if bool(no_resume):
        raise FileExistsError(
            f"scene shard already exists and overwrite is disabled: {shard}"
        )
    payload = json.loads(shard.read_text(encoding="utf-8"))
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"existing scene shard has incompatible provenance; use a new "
            f"output directory: {shard}: {mismatches}"
        )
    if len(payload.get("rows", [])) != len(expected["object_keys"]):
        raise ValueError(
            f"existing scene shard row count is incomplete; use a new output "
            f"directory: {shard}"
        )
    return True


def quantize_easy3d_scene(
    xyz: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    *,
    voxel_size: float = 0.05,
    max_scene_size_m: float = 40.0,
) -> Easy3DScene:
    """Reproduce the public Easy3D data loader, including last-row writes.

    ``torch.unique(..., dim=0)`` lexicographically sorts coordinates.  The
    released indexed assignments choose the last point presented for duplicate
    voxels; this differs from AGILE3D/MinkowskiEngine's first-row feature map.
    """

    xyz = np.asarray(xyz, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be [N,3]")
    if colors.shape != xyz.shape or labels.shape != (len(xyz),):
        raise ValueError("xyz, RGB, and instance labels must align")
    if not np.isfinite(xyz).all() or not np.isfinite(colors).all():
        raise ValueError("scene coordinates and colors must be finite")
    if float(voxel_size) <= 0 or float(max_scene_size_m) <= 0:
        raise ValueError("voxel and maximum scene sizes must be positive")
    shifted = xyz - xyz.min(axis=0, keepdims=True)
    discrete = np.trunc(shifted / float(voxel_size)).astype(np.int32)
    coordinates, first, inverse = np.unique(
        discrete, axis=0, return_index=True, return_inverse=True
    )
    # Match ``dst[inverse] = src``: the final point in input order wins.
    last = np.zeros(len(coordinates), dtype=np.int64)
    np.maximum.at(last, inverse, np.arange(len(inverse), dtype=np.int64))
    voxel_colors = colors[last]
    voxel_labels = labels[last]
    diagnostics = assignment_quantization_diagnostics(
        colors,
        labels,
        inverse,
        voxel_colors,
        voxel_labels,
        first_indices=first,
    )
    normalized_xyz = coordinates.astype(np.float32) / (
        float(max_scene_size_m) / float(voxel_size)
    )
    features = np.concatenate(
        (normalized_xyz, voxel_colors.astype(np.float32) * 2.0 - 1.0),
        axis=1,
    )
    return Easy3DScene(
        coordinates=np.ascontiguousarray(coordinates.astype(np.float32)),
        features=np.ascontiguousarray(features.astype(np.float32)),
        voxel_labels=np.ascontiguousarray(voxel_labels),
        voxel_valid=np.ascontiguousarray(voxel_labels != -1),
        point_labels=np.ascontiguousarray(labels),
        inverse_map=np.ascontiguousarray(inverse.astype(np.int64)),
        quantization_diagnostics=diagnostics,
    )


def assignment_quantization_diagnostics(
    colors: np.ndarray,
    labels: np.ndarray,
    inverse_map: np.ndarray,
    voxel_colors: np.ndarray,
    voxel_labels: np.ndarray,
    *,
    first_indices: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compare an actual indexed-write result with mean/first alternatives."""

    colors = np.asarray(colors, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    inverse = np.asarray(inverse_map, dtype=np.int64).reshape(-1)
    voxel_colors = np.asarray(voxel_colors, dtype=np.float32)
    voxel_labels = np.asarray(voxel_labels, dtype=np.int32).reshape(-1)
    voxel_count = len(voxel_colors)
    if (
        colors.shape != (len(labels), 3)
        or inverse.shape != labels.shape
        or voxel_colors.shape != (voxel_count, 3)
        or voxel_labels.shape != (voxel_count,)
        or bool((inverse < 0).any())
        or bool((inverse >= voxel_count).any())
    ):
        raise ValueError("point/voxel assignment diagnostics do not align")
    if first_indices is None:
        first = np.full(voxel_count, len(inverse), dtype=np.int64)
        np.minimum.at(first, inverse, np.arange(len(inverse), dtype=np.int64))
    else:
        first = np.asarray(first_indices, dtype=np.int64).reshape(-1)
        if first.shape != (voxel_count,):
            raise ValueError("first point indices must align with voxels")
    counts = np.bincount(inverse, minlength=voxel_count).astype(np.int64)
    color_sums = np.zeros((voxel_count, 3), dtype=np.float64)
    np.add.at(color_sums, inverse, colors.astype(np.float64))
    mean_colors = color_sums / counts[:, None]
    mean_assigned_rgb_l2 = np.linalg.norm(
        mean_colors - voxel_colors.astype(np.float64), axis=1
    )
    duplicate = counts > 1
    first_assigned_rgb_l2 = np.linalg.norm(
        colors[first].astype(np.float64) - voxel_colors.astype(np.float64),
        axis=1,
    )
    label_difference = labels[first] != voxel_labels
    return {
        "point_count": int(len(labels)),
        "voxel_count": int(voxel_count),
        "duplicate_voxel_count": int(duplicate.sum()),
        "duplicate_voxel_fraction": float(duplicate.mean()),
        "points_in_duplicate_voxels": int(counts[duplicate].sum()),
        "mean_rgb_vs_last_l2_mean_all_voxels": float(
            mean_assigned_rgb_l2.mean()
        ),
        "mean_rgb_vs_last_l2_mean_duplicate_voxels": float(
            mean_assigned_rgb_l2[duplicate].mean() if duplicate.any() else 0.0
        ),
        "mean_rgb_vs_last_l2_fraction_above_one_8bit_step": float(
            (mean_assigned_rgb_l2 > (1.0 / 255.0)).mean()
        ),
        "mean_rgb_vs_last_l2_max": float(
            mean_assigned_rgb_l2.max(initial=0.0)
        ),
        "first_vs_last_rgb_l2_mean_all_voxels": float(
            first_assigned_rgb_l2.mean()
        ),
        "first_vs_last_label_difference_count": int(label_difference.sum()),
        "first_vs_last_label_difference_fraction": float(
            label_difference.mean()
        ),
    }


def load_easy3d_scene(path: str | Path) -> Easy3DScene:
    """Load an AGILE3D PLY using the field names expected by Easy3D."""

    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover - exercised in Easy3D env
        raise RuntimeError("Easy3D evaluation requires the plyfile package") from exc
    vertex = PlyData.read(str(path))["vertex"]
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"]))
    colors = np.column_stack((vertex["R"], vertex["G"], vertex["B"])) / 255.0
    labels = np.asarray(vertex["label"], dtype=np.int32)
    return quantize_easy3d_scene(xyz, colors, labels)


def load_official_worker_cached_scene(
    cache_root: str | Path,
    scene_id: str,
    *,
    easy3d_commit: str,
    object_ids_sha256: str,
) -> tuple[Easy3DScene, dict[str, Any]]:
    """Load and validate preprocessing emitted by the official worker path."""

    import torch

    root = Path(cache_root)
    npz_path = root / "scenes" / f"{scene_id}.npz"
    metadata_path = root / "scenes" / f"{scene_id}.json"
    if not npz_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"missing official-worker Easy3D cache for {scene_id}: {root}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "cache_schema": OFFICIAL_WORKER_CACHE_SCHEMA,
        "scene_id": scene_id,
        "easy3d_commit": easy3d_commit,
        "voxel_dataset_sha256": OFFICIAL_VOXEL_DATASET_SHA256,
        "object_ids_sha256": object_ids_sha256,
        "torch_version": torch.__version__,
        "torch_built_cuda": torch.version.cuda,
        "configured_num_workers": 4,
        "worker_torch_num_threads": 1,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{scene_id}: official-worker preprocessing provenance mismatch: "
            f"{mismatches}"
        )
    if _sha256(npz_path) != metadata.get("npz_sha256"):
        raise ValueError(f"{scene_id}: preprocessing NPZ hash failed")
    with np.load(npz_path, allow_pickle=False) as payload:
        required = {
            "coordinates",
            "features",
            "voxel_labels",
            "voxel_valid",
            "point_labels",
            "inverse_map",
        }
        if set(payload.files) != required:
            raise ValueError(f"{scene_id}: preprocessing arrays are incomplete")
        arrays = {name: np.asarray(payload[name]) for name in required}
    if _array_digest(arrays) != metadata.get("array_content_sha256"):
        raise ValueError(f"{scene_id}: preprocessing array-content hash failed")
    coordinates = np.asarray(arrays["coordinates"], dtype=np.float32)
    features = np.asarray(arrays["features"], dtype=np.float32)
    voxel_labels = np.asarray(arrays["voxel_labels"], dtype=np.int32).reshape(-1)
    voxel_valid = np.asarray(arrays["voxel_valid"], dtype=bool).reshape(-1)
    point_labels = np.asarray(arrays["point_labels"], dtype=np.int32).reshape(-1)
    inverse = np.asarray(arrays["inverse_map"], dtype=np.int64).reshape(-1)
    voxel_count = len(coordinates)
    if (
        coordinates.shape != (voxel_count, 3)
        or features.shape != (voxel_count, 6)
        or voxel_labels.shape != (voxel_count,)
        or voxel_valid.shape != (voxel_count,)
        or inverse.shape != point_labels.shape
        or not np.array_equal(voxel_valid, voxel_labels != -1)
        or bool((inverse < 0).any())
        or bool((inverse >= voxel_count).any())
    ):
        raise ValueError(f"{scene_id}: preprocessing array contract failed")
    return (
        Easy3DScene(
            coordinates=np.ascontiguousarray(coordinates),
            features=np.ascontiguousarray(features),
            voxel_labels=np.ascontiguousarray(voxel_labels),
            voxel_valid=np.ascontiguousarray(voxel_valid),
            point_labels=np.ascontiguousarray(point_labels),
            inverse_map=np.ascontiguousarray(inverse),
            quantization_diagnostics=dict(
                metadata["quantization_diagnostics"]
            ),
        ),
        metadata,
    )


def validate_official_worker_cache_manifest(
    cache_root: str | Path,
    *,
    data_root: str | Path,
    easy3d_commit: str,
    object_ids_sha256: str,
    required_scene_ids: Iterable[str],
    formal: bool,
) -> tuple[dict[str, Any], str, dict[str, dict[str, Any]]]:
    """Validate the immutable worker-cache manifest and selected scene rows."""

    import torch

    root = Path(cache_root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing official-worker preprocessing manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "cache_schema": OFFICIAL_WORKER_CACHE_SCHEMA,
        "easy3d_commit": easy3d_commit,
        "voxel_dataset_sha256": OFFICIAL_VOXEL_DATASET_SHA256,
        "object_ids_sha256": object_ids_sha256,
        "data_root": str(Path(data_root).resolve()),
        "torch_version": torch.__version__,
        "torch_built_cuda": torch.version.cuda,
        "configured_num_workers": 4,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "official-worker preprocessing manifest provenance mismatch: "
            f"{mismatches}"
        )
    rows = manifest.get("scenes")
    if not isinstance(rows, list):
        raise ValueError("preprocessing manifest scenes must be a list")
    by_scene: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(
            row.get("scene_id"), str
        ):
            raise ValueError("preprocessing manifest has an invalid scene row")
        scene_id = str(row["scene_id"])
        if scene_id in by_scene:
            raise ValueError(
                f"preprocessing manifest repeats scene {scene_id}"
            )
        if (
            not isinstance(row.get("npz_sha256"), str)
            or not isinstance(row.get("array_content_sha256"), str)
            or int(row.get("worker_torch_num_threads", -1)) != 1
        ):
            raise ValueError(
                f"preprocessing manifest has incomplete hashes for {scene_id}"
            )
        by_scene[scene_id] = row
    if int(manifest.get("scene_count", -1)) != len(by_scene):
        raise ValueError("preprocessing manifest scene count is inconsistent")
    required = set(required_scene_ids)
    missing = sorted(required - set(by_scene))
    if missing:
        raise ValueError(
            f"preprocessing manifest lacks selected scenes: {missing}"
        )
    if formal and (
        manifest.get("selection") != "formal_312"
        or len(by_scene) != 312
        or len(required) != 312
        or set(by_scene) != required
    ):
        raise ValueError(
            "formal Easy3D evaluation requires the exact 312-scene cache "
            "manifest"
        )
    return manifest, _sha256(manifest_path), by_scene


class NearestComplementIndex:
    """Exact error-center search with one reusable scene KD-tree."""

    def __init__(self, coordinates: np.ndarray) -> None:
        values = np.asarray(coordinates, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3 or not len(values):
            raise ValueError("coordinates must be a nonempty [V,3] array")
        self.coordinates = values
        self.tree = cKDTree(values)

    def error_center(self, error_mask: np.ndarray) -> tuple[int, float]:
        error = np.asarray(error_mask, dtype=bool).reshape(-1)
        if error.shape != (len(self.coordinates),):
            raise ValueError("error mask must align with scene coordinates")
        error_indices = np.flatnonzero(error)
        if not len(error_indices) or len(error_indices) == len(error):
            raise ValueError("an error region needs error and complement voxels")
        unresolved = np.arange(len(error_indices), dtype=np.int64)
        nearest = np.full(len(error_indices), np.inf, dtype=np.float64)
        k = min(8, len(error))
        while len(unresolved):
            distances, neighbors = self.tree.query(
                self.coordinates[error_indices[unresolved]],
                k=k,
                workers=-1,
            )
            if k == 1:
                distances = distances[:, None]
                neighbors = neighbors[:, None]
            complement = ~error[neighbors]
            found = complement.any(axis=1)
            if found.any():
                rows = np.flatnonzero(found)
                first = np.argmax(complement[rows], axis=1)
                nearest[unresolved[rows]] = distances[rows, first]
            unresolved = unresolved[~found]
            if not len(unresolved):
                break
            if k == len(error):
                raise AssertionError("KD-tree did not expose an existing complement")
            k = min(len(error), k * 2)
        local = int(np.argmax(nearest))
        return int(error_indices[local]), float(nearest[local])

    def next_click(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        *,
        valid: np.ndarray | None,
    ) -> tuple[int, bool] | None:
        """Return the Easy3D/AGILE largest-error click (FP wins exact ties)."""

        prediction = np.asarray(prediction, dtype=bool).reshape(-1)
        target = np.asarray(target, dtype=bool).reshape(-1)
        if prediction.shape != target.shape or len(target) != len(self.coordinates):
            raise ValueError("prediction and target must align with coordinates")
        usable = (
            np.ones_like(target)
            if valid is None
            else np.asarray(valid, dtype=bool).reshape(-1)
        )
        if usable.shape != target.shape:
            raise ValueError("valid mask must align with target")
        chosen: tuple[int, bool] | None = None
        chosen_radius = -np.inf
        for error, positive in (
            (prediction & ~target & usable, False),
            (~prediction & target, True),
        ):
            if not error.any() or error.all():
                continue
            point_index, radius = self.error_center(error)
            # Released Easy3D loops FP then FN and updates only on strict >.
            if radius > chosen_radius:
                chosen = (point_index, positive)
                chosen_radius = radius
        return chosen


def _iou(
    prediction: np.ndarray, target: np.ndarray, *, epsilon: float = 0.0
) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    union = int(np.logical_or(prediction, target).sum())
    return (
        float(np.logical_and(prediction, target).sum() / (union + epsilon))
        if union
        else 0.0
    )


def _model_class(easy3d_repo: str | Path) -> Any:
    repo = str(Path(easy3d_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return importlib.import_module("easy3d.model.model").Easy3DModel


def load_official_model(
    easy3d_repo: str | Path,
    checkpoint: str | Path,
    *,
    device: str,
) -> Any:
    import torch

    model = _model_class(easy3d_repo)(
        embedding_dim=256,
        mlp_dim=1024,
        voxel_size=0.05,
        max_scene_size=40.0,
        num_clicks=10,
    )
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(torch.device(device))


def _decode_batch(
    model: Any,
    voxel_embedding: Any,
    coordinates: Any,
    click_indices: np.ndarray,
    click_labels: np.ndarray,
) -> Any:
    """Run the released Easy3D decoder for a batch of object interactions."""

    import torch

    device = coordinates.device
    query_count, click_count = click_indices.shape
    voxel_count, feature_count = voxel_embedding.shape
    indices = torch.as_tensor(click_indices, dtype=torch.long, device=device)
    labels = torch.as_tensor(click_labels, dtype=torch.long, device=device)
    first = coordinates[indices[:, 0]]
    centered = coordinates.unsqueeze(0) - first.unsqueeze(1)
    normalized = centered / (model.max_scene_size / model.voxel_size)
    query_pe = model.position_proj(model.pe(normalized))
    click_embedding = query_pe.new_zeros(
        (query_count, click_count, feature_count)
    )
    rows = torch.arange(query_count, device=device).unsqueeze(1)
    click_embedding += query_pe[rows, indices] * (labels != 2).unsqueeze(-1)
    click_embedding += model.interaction_embeddings(labels)
    output_tokens = model.output_mask_tokens.weight.unsqueeze(0).expand(
        query_count, -1, -1
    )
    click_embedding = torch.cat((output_tokens, click_embedding), dim=1)
    scene_embedding = voxel_embedding.unsqueeze(0).expand(
        query_count, voxel_count, feature_count
    )
    updated_scene, updated_clicks = model.decoder(
        scene_embedding, query_pe, click_embedding
    )
    masks = torch.einsum(
        "qcf,qvf->qcv", updated_clicks[:, :2], updated_scene
    )
    return masks[:, 0] - masks[:, 1]


def evaluate_scene_objects(
    model: Any,
    scene: Easy3DScene,
    objects: Sequence[Agile3DObject],
    *,
    device: str,
    object_batch_size: int = 4,
    max_clicks: int = 10,
    amp_bfloat16: bool = True,
    interaction_contract: str = "agile3d_release",
) -> list[dict[str, Any]]:
    """Encode one scene once and evaluate all requested object trajectories."""

    import torch

    if interaction_contract not in {
        "agile3d_release",
        "easy3d_released_code",
    }:
        raise ValueError("unknown Easy3D interaction contract")
    if int(object_batch_size) <= 0 or int(max_clicks) <= 0:
        raise ValueError("batch size and maximum clicks must be positive")
    torch_device = torch.device(device)
    coordinates = torch.from_numpy(scene.coordinates).to(torch_device)
    features = torch.from_numpy(scene.features).to(torch_device)
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if amp_bfloat16 and torch_device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), amp:
        voxel_features = model.encoder(coordinates, features)
        voxel_embedding = model.encoder_projection(voxel_features)
    click_index = NearestComplementIndex(scene.coordinates)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(objects), int(object_batch_size)):
        requested_batch = list(
            objects[start : start + int(object_batch_size)]
        )
        targets_voxel = np.stack(
            [scene.voxel_labels == item.object_id for item in requested_batch]
        )
        present = targets_voxel.any(axis=1)
        for item in (
            requested_batch[index]
            for index in np.flatnonzero(~present).tolist()
        ):
            rows.append(
                {
                    "key": item.key,
                    "scene_id": item.scene_id,
                    "object_id": int(item.object_id),
                    "semantic_class": item.semantic_class,
                    "status": "not_evaluable",
                    "failure_reason": (
                        "object_absent_after_released_Easy3D_last_write_voxelization"
                    ),
                }
            )
        batch = [
            requested_batch[index]
            for index in np.flatnonzero(present).tolist()
        ]
        if not batch:
            continue
        targets_voxel = targets_voxel[present]
        query_count = len(batch)
        targets_point = np.stack(
            [scene.point_labels == item.object_id for item in batch]
        )
        predictions_for_click = np.zeros_like(targets_voxel, dtype=bool)
        metric_predictions = np.zeros_like(targets_voxel, dtype=bool)
        frozen = np.zeros(query_count, dtype=bool)
        clicks: list[list[int]] = [[] for _ in batch]
        labels: list[list[int]] = [[] for _ in batch]
        trajectories = [dict() for _ in batch]
        for click_count in range(1, int(max_clicks) + 1):
            decode_queries: list[int] = []
            for query_index in range(query_count):
                if frozen[query_index]:
                    continue
                click = click_index.next_click(
                    predictions_for_click[query_index],
                    targets_voxel[query_index],
                    valid=(
                        None
                        if interaction_contract == "agile3d_release"
                        else scene.voxel_valid
                    ),
                )
                if click is None:
                    if interaction_contract == "easy3d_released_code":
                        clicks[query_index].append(0)
                        labels[query_index].append(2)
                    else:
                        # AGILE3D holds the previous prediction and interaction
                        # set fixed once no error remains.
                        frozen[query_index] = True
                        continue
                else:
                    point_index, positive = click
                    clicks[query_index].append(point_index)
                    labels[query_index].append(1 if positive else 0)
                decode_queries.append(query_index)
            if decode_queries:
                with torch.inference_mode(), (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if amp_bfloat16 and torch_device.type == "cuda"
                    else nullcontext()
                ):
                    logits = _decode_batch(
                        model,
                        voxel_embedding,
                        coordinates,
                        np.asarray(
                            [clicks[index] for index in decode_queries],
                            dtype=np.int64,
                        ),
                        np.asarray(
                            [labels[index] for index in decode_queries],
                            dtype=np.int64,
                        ),
                    )
                raw_metric = (logits >= 0).cpu().numpy()
                raw_click = (logits > 0).cpu().numpy()
                for local_index, query_index in enumerate(decode_queries):
                    metric_prediction = raw_metric[local_index]
                    next_prediction = raw_click[local_index]
                    if interaction_contract == "agile3d_release":
                        for point_index, label in zip(
                            clicks[query_index], labels[query_index]
                        ):
                            metric_prediction[point_index] = label == 1
                            next_prediction[point_index] = label == 1
                    metric_predictions[query_index] = metric_prediction
                    predictions_for_click[query_index] = next_prediction
            for query_index in range(query_count):
                trajectories[query_index][click_count] = _iou(
                    metric_predictions[query_index][scene.inverse_map],
                    targets_point[query_index],
                    epsilon=(
                        1e-6
                        if interaction_contract == "easy3d_released_code"
                        else 0.0
                    ),
                )
        for item, trajectory, item_clicks, item_labels in zip(
            batch, trajectories, clicks, labels
        ):
            rows.append(
                {
                    "key": item.key,
                    "scene_id": item.scene_id,
                    "object_id": int(item.object_id),
                    "semantic_class": item.semantic_class,
                    "status": "evaluated",
                    "trajectory": {
                        str(key): float(value)
                        for key, value in trajectory.items()
                    },
                    "positive_clicks": int(sum(value == 1 for value in item_labels)),
                    "negative_clicks": int(sum(value == 0 for value in item_labels)),
                    "invalid_or_repeated_clicks": int(
                        sum(value == 2 for value in item_labels)
                    ),
                    "click_point_indices": [int(value) for value in item_clicks],
                    "click_labels": [int(value) for value in item_labels],
                }
            )
    return rows


def _normalized_trajectory(row: Mapping[str, Any]) -> dict[int, float]:
    values = row.get("trajectory", row)
    return {int(key): float(value) for key, value in values.items()}


def aggregate_trajectory_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_clicks: int,
) -> dict[str, float]:
    trajectories = [_normalized_trajectory(row) for row in rows]
    if not trajectories:
        raise ValueError("no Easy3D trajectories to aggregate")
    metrics: dict[str, float] = {}
    for count in IOU_CLICK_COUNTS:
        if count <= int(max_clicks):
            if any(count not in row for row in trajectories):
                raise ValueError(f"trajectory misses IoU@{count}")
            metrics[f"IoU@{count}"] = float(
                np.mean([row[count] for row in trajectories])
            )
    for threshold in NOC_THRESHOLDS:
        values = [
            next(
                (
                    count
                    for count in range(1, int(max_clicks) + 1)
                    if row[count] >= threshold
                ),
                int(max_clicks),
            )
            for row in trajectories
        ]
        metrics[f"NoC@{round(threshold * 100):.0f}"] = float(np.mean(values))
    return metrics


def load_agile3d_result_csv(
    path: str | Path,
) -> dict[str, dict[int, float]]:
    """Load AGILE3D's bundled single-object CSV using its recorded IDs."""

    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            columns = line.split()
            if len(columns) != 5:
                raise ValueError(f"invalid AGILE3D CSV row {line_number}")
            _row_id, scene, object_id, click_count, iou = columns
            key = f"scene{scene}_obj_{int(object_id)}"
            click = int(click_count)
            if click in grouped[key]:
                raise ValueError(f"duplicate AGILE3D result for {key} click {click}")
            grouped[key][click] = float(iou)
    if not grouped:
        raise ValueError("AGILE3D result CSV is empty")
    return dict(grouped)


def reference_cohort_audit(
    official_objects: Sequence[Agile3DObject],
    agile_csv: str | Path,
) -> dict[str, Any]:
    """Expose the official evaluator's silent release/CSV key intersection."""

    csv_rows = load_agile3d_result_csv(agile_csv)
    release_keys = {item.key for item in official_objects}
    csv_keys = set(csv_rows)
    matched = release_keys & csv_keys
    max_clicks = max(max(row) for row in csv_rows.values())
    matched_rows = [csv_rows[key] for key in sorted(matched)]
    complete_rows = [row for _key, row in sorted(csv_rows.items())]
    return {
        "release_object_count": len(release_keys),
        "csv_object_count": len(csv_keys),
        "legacy_matched_object_count": len(matched),
        "release_objects_silently_unmatched": len(release_keys - csv_keys),
        "csv_objects_not_in_release_list": len(csv_keys - release_keys),
        "legacy_matched_keys": sorted(matched),
        "examples_release_only": sorted(release_keys - csv_keys)[:20],
        "examples_csv_only": sorted(csv_keys - release_keys)[:20],
        "bundled_csv_metrics_legacy_match": aggregate_trajectory_rows(
            matched_rows, max_clicks=max_clicks
        ),
        "bundled_csv_metrics_all_recorded_rows": aggregate_trajectory_rows(
            complete_rows, max_clicks=max_clicks
        ),
    }


def _scene_macro(
    rows: Sequence[Mapping[str, Any]], *, max_clicks: int
) -> dict[str, float]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scene_id"])].append(row)
    per_scene = {
        scene_id: aggregate_trajectory_rows(scene_rows, max_clicks=max_clicks)
        for scene_id, scene_rows in sorted(grouped.items())
    }
    keys = next(iter(per_scene.values()))
    return {
        key: float(np.mean([values[key] for values in per_scene.values()]))
        for key in keys
    }


def aggregate_quantization_diagnostics(
    rows: Sequence[Mapping[str, float | int]],
) -> dict[str, float | int]:
    """Aggregate paper-mean versus released-last-write RGB diagnostics."""

    if not rows:
        raise ValueError("no quantization diagnostics to aggregate")
    point_count = sum(int(row["point_count"]) for row in rows)
    voxel_count = sum(int(row["voxel_count"]) for row in rows)
    duplicate_count = sum(int(row["duplicate_voxel_count"]) for row in rows)
    label_difference_count = sum(
        int(row["first_vs_last_label_difference_count"]) for row in rows
    )

    def weighted(name: str, weight_name: str) -> float:
        denominator = sum(int(row[weight_name]) for row in rows)
        return (
            float(
                sum(
                    float(row[name]) * int(row[weight_name])
                    for row in rows
                )
                / denominator
            )
            if denominator
            else 0.0
        )

    return {
        "scene_count": len(rows),
        "point_count": point_count,
        "voxel_count": voxel_count,
        "duplicate_voxel_count": duplicate_count,
        "duplicate_voxel_fraction": float(
            duplicate_count / voxel_count if voxel_count else 0.0
        ),
        "points_in_duplicate_voxels": sum(
            int(row["points_in_duplicate_voxels"]) for row in rows
        ),
        "mean_rgb_vs_last_l2_mean_all_voxels": weighted(
            "mean_rgb_vs_last_l2_mean_all_voxels", "voxel_count"
        ),
        "mean_rgb_vs_last_l2_mean_duplicate_voxels": weighted(
            "mean_rgb_vs_last_l2_mean_duplicate_voxels",
            "duplicate_voxel_count",
        ),
        "mean_rgb_vs_last_l2_fraction_above_one_8bit_step": weighted(
            "mean_rgb_vs_last_l2_fraction_above_one_8bit_step",
            "voxel_count",
        ),
        "mean_rgb_vs_last_l2_max": max(
            float(row["mean_rgb_vs_last_l2_max"]) for row in rows
        ),
        "first_vs_last_rgb_l2_mean_all_voxels": weighted(
            "first_vs_last_rgb_l2_mean_all_voxels", "voxel_count"
        ),
        "first_vs_last_label_difference_count": label_difference_count,
        "first_vs_last_label_difference_fraction": float(
            label_difference_count / voxel_count if voxel_count else 0.0
        ),
    }


def _result_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_clicks: int,
    provenance: Mapping[str, Any],
    cohort_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    evaluated = [
        row for row in rows if str(row.get("status", "evaluated")) == "evaluated"
    ]
    failures = [
        row for row in rows if str(row.get("status", "evaluated")) != "evaluated"
    ]
    full_metrics = aggregate_trajectory_rows(evaluated, max_clicks=max_clicks)
    by_key = {str(row["key"]): row for row in evaluated}
    selected_keys = {str(row["key"]) for row in rows}
    cohorts: dict[str, Any] = {
        "complete_release_selection": {
            "object_count": len(rows),
            "evaluated_object_count": len(evaluated),
            "failed_object_count": len(failures),
            "metrics_query_micro": full_metrics,
            "metrics_scene_macro": _scene_macro(
                evaluated, max_clicks=max_clicks
            ),
        }
    }
    if cohort_audit is not None:
        selected_legacy_keys = sorted(
            selected_keys & set(cohort_audit["legacy_matched_keys"])
        )
        matched_rows = [
            by_key[key]
            for key in selected_legacy_keys
            if key in by_key
        ]
        cohorts["agile3d_legacy_paper_script_intersection"] = {
            "object_count": len(selected_legacy_keys),
            "evaluated_object_count": len(matched_rows),
            "failed_object_count": (
                len(selected_legacy_keys) - len(matched_rows)
            ),
            "metrics_query_micro": aggregate_trajectory_rows(
                matched_rows, max_clicks=max_clicks
            ),
            "warning": (
                "Reference-only legacy cohort: AGILE3D's bundled result IDs "
                "silently match only part of the released object list."
            ),
        }
    comparisons = {
        cohort_name: {
            key: {
                "paper": value,
                "reproduced": cohort["metrics_query_micro"].get(key),
                "delta": (
                    None
                    if key not in cohort["metrics_query_micro"]
                    else float(cohort["metrics_query_micro"][key] - value)
                ),
            }
            for key, value in PAPER_IOU.items()
        }
        for cohort_name, cohort in cohorts.items()
    }
    return {
        "benchmark": "AGILE3D-ScanNet40-single-object",
        "method": "official_Easy3D_pretrained_checkpoint",
        "status": (
            "formal_complete"
            if len(rows) == 10357 and not failures
            else (
                "formal_with_explicit_object_failures"
                if len(rows) == 10357
                else "declared_pilot_or_partial"
            )
        ),
        "provenance": dict(provenance),
        "cohort_audit": (
            None
            if cohort_audit is None
            else {
                key: value
                for key, value in cohort_audit.items()
                if key != "legacy_matched_keys"
            }
        ),
        "cohorts": cohorts,
        "paper_comparison_by_cohort": comparisons,
        "object_failures": failures,
        "rows": list(rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root).resolve()
    easy3d_repo = Path(args.easy3d_repo).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    preprocessing_cache_root = Path(
        args.preprocessing_cache_root
    ).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = _sha256(checkpoint)
    if (
        checkpoint_sha != EASY3D_CHECKPOINT_SHA256
        and not bool(args.allow_unverified_checkpoint)
    ):
        raise ValueError(
            "checkpoint SHA256 is not the official Easy3D v1.0 release"
        )
    easy3d_commit = _git_commit(easy3d_repo)
    objects = load_official_object_list(data_root)
    object_ids_sha = _sha256(data_root / "single" / "object_ids.npy")
    object_classes_sha = _sha256(data_root / "single" / "object_classes.txt")
    requested = {
        value
        for value in str(args.scene_names).replace(",", " ").split()
        if value
    }
    by_scene: dict[str, list[Agile3DObject]] = defaultdict(list)
    for item in objects:
        if requested and item.scene_id not in requested:
            continue
        by_scene[item.scene_id].append(item)
    if requested - set(by_scene):
        raise ValueError(f"unknown requested scenes: {sorted(requested - set(by_scene))}")
    if int(args.object_limit_per_scene) > 0:
        by_scene = {
            key: values[: int(args.object_limit_per_scene)]
            for key, values in by_scene.items()
        }
    selected_count = sum(len(values) for values in by_scene.values())
    formal = not requested and int(args.object_limit_per_scene) == 0
    if formal:
        if len(by_scene) != 312 or selected_count != 10357:
            raise ValueError("formal Easy3D run requires 312 scenes / 10,357 objects")
        if easy3d_commit != EASY3D_AUDITED_COMMIT:
            raise ValueError(
                "formal Easy3D run requires the audited official repository commit"
            )
        if (
            object_ids_sha != OFFICIAL_OBJECT_IDS_SHA256
            or object_classes_sha != OFFICIAL_OBJECT_CLASSES_SHA256
        ):
            raise ValueError("formal Easy3D run has noncanonical object-list hashes")
    (
        preprocessing_manifest,
        preprocessing_manifest_sha,
        preprocessing_scene_rows,
    ) = validate_official_worker_cache_manifest(
        preprocessing_cache_root,
        data_root=data_root,
        easy3d_commit=easy3d_commit,
        object_ids_sha256=object_ids_sha,
        required_scene_ids=by_scene,
        formal=formal,
    )
    provenance = {
        "easy3d_repository": EASY3D_REPOSITORY,
        "easy3d_commit": easy3d_commit,
        "audited_easy3d_commit": EASY3D_AUDITED_COMMIT,
        "checkpoint_url": EASY3D_CHECKPOINT_URL,
        "checkpoint_sha256": checkpoint_sha,
        "dataset_root": str(data_root),
        "object_ids_sha256": object_ids_sha,
        "object_classes_sha256": object_classes_sha,
        "scene_count": len(by_scene),
        "object_count": selected_count,
        "voxel_size_m": 0.05,
        "max_scene_size_m": 40.0,
        "preprocessing": (
            EASY3D_PREPROCESSING_ID
        ),
        "preprocessing_cache_root": str(preprocessing_cache_root),
        "preprocessing_cache_schema": OFFICIAL_WORKER_CACHE_SCHEMA,
        "preprocessing_manifest_sha256": preprocessing_manifest_sha,
        "preprocessing_manifest_selection": preprocessing_manifest["selection"],
        "preprocessing_manifest_scene_count": preprocessing_manifest[
            "scene_count"
        ],
        "paper_preprocessing_prose": (
            "RGB_features_within_each_voxel_are_averaged"
        ),
        "paper_code_preprocessing_mismatch": True,
        "interaction_contract": str(args.interaction_contract),
        "interaction_contract_scope": (
            "AGILE3D_click_overwrite_and_all_voxel_error_policy_on_released_"
            "Easy3D_preprocessing"
            if str(args.interaction_contract) == "agile3d_release"
            else "released_Easy3D_forward_click_and_metric_behavior"
        ),
        "click_coordinate_basis": "released_Easy3D_integer_voxel_coordinates",
        "false_positive_error_excludes_invalid_voxels": (
            str(args.interaction_contract) == "easy3d_released_code"
        ),
        "first_click": "largest_false_negative_error_region_center",
        "corrective_click": (
            "largest_FP_or_FN_error_region_center_FP_wins_exact_tie"
        ),
        "clicked_voxel_overwrite": (
            str(args.interaction_contract) == "agile3d_release"
        ),
        "prediction_threshold_for_metric": "sigmoid_logit_greater_equal_0.5",
        "point_iou": True,
        "point_iou_denominator_epsilon": (
            1e-6
            if str(args.interaction_contract) == "easy3d_released_code"
            else 0.0
        ),
        "max_clicks": int(args.max_clicks),
        "noc_cap": int(args.max_clicks),
        "amp_bfloat16": bool(args.amp_bfloat16),
        "object_batch_size": int(args.object_batch_size),
        "formal_selection": formal,
        "runtime": runtime_versions(),
    }
    model = load_official_model(
        easy3d_repo, checkpoint, device=str(args.device)
    )
    for scene_index, (scene_id, scene_objects) in enumerate(
        sorted(by_scene.items()), start=1
    ):
        shard = output_dir / "scene_shards" / f"{scene_id}.json"
        expected_object_keys = [item.key for item in scene_objects]
        preprocessing_row = preprocessing_scene_rows[scene_id]
        expected_shard = {
            "evaluator_schema_version": EVALUATOR_SCHEMA_VERSION,
            "scene_id": scene_id,
            "checkpoint_sha256": checkpoint_sha,
            "easy3d_commit": easy3d_commit,
            "object_ids_sha256": object_ids_sha,
            "preprocessing": EASY3D_PREPROCESSING_ID,
            "preprocessing_manifest_sha256": preprocessing_manifest_sha,
            "preprocessing_npz_sha256": preprocessing_row["npz_sha256"],
            "preprocessing_array_content_sha256": preprocessing_row[
                "array_content_sha256"
            ],
            "interaction_contract": str(args.interaction_contract),
            "max_clicks": int(args.max_clicks),
            "amp_bfloat16": bool(args.amp_bfloat16),
            "object_batch_size": int(args.object_batch_size),
            "object_keys": expected_object_keys,
        }
        if existing_scene_shard_matches(
            shard, expected_shard, no_resume=bool(args.no_resume)
        ):
            print(
                f"[{scene_index}/{len(by_scene)}] {scene_id}: resume",
                flush=True,
            )
            continue
        started = time.time()
        scene, cache_metadata = load_official_worker_cached_scene(
            preprocessing_cache_root,
            scene_id,
            easy3d_commit=easy3d_commit,
            object_ids_sha256=object_ids_sha,
        )
        for hash_key in ("npz_sha256", "array_content_sha256"):
            if cache_metadata.get(hash_key) != preprocessing_row.get(hash_key):
                raise ValueError(
                    f"{scene_id}: cache metadata disagrees with manifest "
                    f"for {hash_key}"
                )
        rows = evaluate_scene_objects(
            model,
            scene,
            scene_objects,
            device=str(args.device),
            object_batch_size=int(args.object_batch_size),
            max_clicks=int(args.max_clicks),
            amp_bfloat16=bool(args.amp_bfloat16),
            interaction_contract=str(args.interaction_contract),
        )
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(
            json.dumps(
                {
                    **expected_shard,
                    "elapsed_seconds": time.time() - started,
                    "voxel_count": len(scene.coordinates),
                    "point_count": len(scene.point_labels),
                    "quantization_diagnostics": dict(
                        scene.quantization_diagnostics
                    ),
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[{scene_index}/{len(by_scene)}] {scene_id}: "
            f"{len(rows)} objects, {time.time() - started:.1f}s",
            flush=True,
        )
    merged_rows: list[dict[str, Any]] = []
    quantization_rows: list[dict[str, float | int]] = []
    for scene_id in sorted(by_scene):
        shard = output_dir / "scene_shards" / f"{scene_id}.json"
        payload = json.loads(shard.read_text(encoding="utf-8"))
        merged_rows.extend(payload["rows"])
        quantization_rows.append(payload["quantization_diagnostics"])
    provenance["quantization_diagnostics"] = (
        aggregate_quantization_diagnostics(quantization_rows)
    )
    cohort_audit = (
        None
        if not str(args.agile_reference_csv).strip()
        else reference_cohort_audit(objects, args.agile_reference_csv)
    )
    report = _result_report(
        merged_rows,
        max_clicks=int(args.max_clicks),
        provenance=provenance,
        cohort_audit=cohort_audit,
    )
    results_path = output_dir / "results.json"
    if results_path.exists():
        existing = json.loads(results_path.read_text(encoding="utf-8"))
        if existing != report:
            raise ValueError(
                "existing Easy3D results differ; use a new output directory"
            )
    else:
        results_path.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--easy3d-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessing-cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--agile-reference-csv", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--object-batch-size", type=int, default=4)
    parser.add_argument("--max-clicks", type=int, default=10)
    parser.add_argument("--scene-names", default="")
    parser.add_argument("--object-limit-per-scene", type=int, default=0)
    parser.add_argument(
        "--interaction-contract",
        choices=("agile3d_release", "easy3d_released_code"),
        default="agile3d_release",
    )
    parser.add_argument(
        "--amp-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-unverified-checkpoint", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "results_path": str(
                    (Path(args.output_dir).resolve() / "results.json")
                ),
                "cohorts": {
                    name: {
                        key: value
                        for key, value in cohort.items()
                        if key
                        in {
                            "object_count",
                            "evaluated_object_count",
                            "failed_object_count",
                            "metrics_query_micro",
                            "metrics_scene_macro",
                        }
                    }
                    for name, cohort in report["cohorts"].items()
                },
                "object_failure_count": len(report["object_failures"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
