#!/usr/bin/env python3
"""Train a standard 3D Gaussian Splatting model from COLMAP sparse reconstruction.

Initializes Gaussians from COLMAP points3D.ply, trains with gsplat rasterization
and adaptive densification (clone + split), and saves a standard 3DGS PLY file.

Usage:
    python radio_gs/scripts/train_colmap_gs.py \
        --scene_root /mnt/pool/sqy/lerf_ovs/figurines \
        --output_dir output/3dgs_models/figurines \
        --iters 30000 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gsplat import DefaultStrategy, rasterization
from radio_gs.data.lerf_dataset import (
    _camera_params_to_intrinsics,
    _parse_colmap_sparse,
    _qvec_to_rotmat,
    _read_cameras_binary,
    _read_images_binary,
)
from radio_gs.data.view_split import (
    load_excluded_image_stems,
    select_image_indices,
)

# SH basis constant for degree-0
C0 = 0.28209479177387814
_CAMERA_MAP_SCHEMA_VERSION = 1
_CAMERA_MAP_RULES = frozenset(
    {
        "exact_case_sensitive_basename_stem",
        "strip_official_0_or_1_split_prefix_then_exact_stem",
        "imageNNN_canonical_index_to_lexicographic_colmap_camera",
    }
)


def _load_colmap_camera_rgb_map(
    source: str | Path,
    *,
    scene_root: Path,
) -> tuple[dict[str, dict], dict]:
    """Load an explicit COLMAP-camera-to-RGB mapping, never infer one here."""

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image-map JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Image-map JSON must contain an object")
    if int(payload.get("schema_version", -1)) != _CAMERA_MAP_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing image-map schema_version")
    declared_root = Path(str(payload.get("scene_root") or "")).expanduser().resolve()
    if declared_root != scene_root.resolve():
        raise ValueError(
            f"Image-map scene_root differs from requested scene: "
            f"{declared_root} != {scene_root.resolve()}"
        )
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Image-map JSON must contain a non-empty records list")

    lookup: dict[str, dict] = {}
    rgb_names: set[str] = set()
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise ValueError(f"Image-map record {index} is not an object")
        colmap_name = str(raw.get("colmap_camera_name") or "")
        rgb_name = str(raw.get("rgb_camera_name") or "")
        rgb_path = Path(str(raw.get("rgb_path") or "")).expanduser()
        colmap_path = Path(str(raw.get("colmap_file_path") or ""))
        rule = str(raw.get("match_rule") or "")
        if not colmap_name or Path(colmap_name).name != colmap_name:
            raise ValueError(f"Invalid COLMAP camera name in image-map record {index}")
        if not rgb_name or Path(rgb_name).name != rgb_name:
            raise ValueError(f"Invalid RGB camera name in image-map record {index}")
        if colmap_path.stem != colmap_name:
            raise ValueError(
                f"Image-map COLMAP stem mismatch: {colmap_path.stem!r} != "
                f"{colmap_name!r}"
            )
        if not rgb_path.is_absolute() or rgb_path.stem != rgb_name:
            raise ValueError(
                f"Image-map RGB path/name mismatch for {rgb_name!r}: {rgb_path}"
            )
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Mapped RGB image not found: {rgb_path}")
        if rule not in _CAMERA_MAP_RULES:
            raise ValueError(
                f"Image-map record {rgb_name!r} has unsupported match_rule {rule!r}"
            )
        if colmap_name in lookup or rgb_name in rgb_names:
            raise ValueError(
                f"Image map is not one-to-one at RGB {rgb_name!r} / "
                f"COLMAP {colmap_name!r}"
            )
        lookup[colmap_name] = dict(raw)
        rgb_names.add(rgb_name)

    declared_camera_to_rgb = payload.get("colmap_camera_to_rgb_path")
    if declared_camera_to_rgb is not None:
        if not isinstance(declared_camera_to_rgb, dict):
            raise ValueError("colmap_camera_to_rgb_path must be a JSON object")
        expected_camera_to_rgb = {
            camera: str(record["rgb_path"]) for camera, record in lookup.items()
        }
        normalized_camera_to_rgb = {
            str(camera): str(rgb_path)
            for camera, rgb_path in declared_camera_to_rgb.items()
        }
        if normalized_camera_to_rgb != expected_camera_to_rgb:
            raise ValueError(
                "colmap_camera_to_rgb_path disagrees with image-map records"
            )

    return lookup, {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "record_count": len(lookup),
        "nearest_or_fuzzy_matching": "forbidden",
    }


# ──────────────────────────────────────────────────────────────────
# COLMAP points3D.ply loader
# ──────────────────────────────────────────────────────────────────

def load_colmap_points(
    scene_root: Path,
    *,
    excluded_image_ids: frozenset[int] = frozenset(),
    return_metadata: bool = False,
):
    """Load COLMAP sparse points from points3D.ply.

    Returns:
        xyz: (N, 3) float32 positions
        rgb: (N, 3) float32 colours in [0, 1]
    """
    sparse_root = scene_root / "sparse" / "0"
    ply_path = sparse_root / "points3D.ply"
    binary_path = sparse_root / "points3D.bin"
    # A PLY does not retain observation tracks.  When a benchmark view is
    # held out, use points3D.bin and drop every point whose track contains the
    # held-out image.  Otherwise target pixels could still enter through the
    # sparse initialization even though their RGB frame is absent from the
    # photometric loss.
    if excluded_image_ids:
        if not binary_path.exists():
            raise FileNotFoundError(
                "Track-safe sparse initialization requires points3D.bin when "
                f"views are excluded; not found: {binary_path}"
            )
        return _load_colmap_points_binary(
            binary_path,
            excluded_image_ids=excluded_image_ids,
            return_metadata=return_metadata,
        )

    if not ply_path.exists():
        if binary_path.exists():
            return _load_colmap_points_binary(
                binary_path,
                return_metadata=return_metadata,
            )
        raise FileNotFoundError(
            f"COLMAP points not found: expected {ply_path} or {binary_path}"
        )

    from plyfile import PlyData

    plydata = PlyData.read(str(ply_path))
    vertex = plydata.elements[0]
    xyz = np.stack(
        [np.asarray(vertex["x"], dtype=np.float32),
         np.asarray(vertex["y"], dtype=np.float32),
         np.asarray(vertex["z"], dtype=np.float32)],
        axis=1,
    )
    rgb = np.stack(
        [np.asarray(vertex["red"], dtype=np.float32),
         np.asarray(vertex["green"], dtype=np.float32),
         np.asarray(vertex["blue"], dtype=np.float32)],
        axis=1,
    ) / 255.0
    if not return_metadata:
        return xyz, rgb
    return xyz, rgb, {
        "source": str(ply_path.resolve()),
        "source_point_count": int(xyz.shape[0]),
        "retained_point_count": int(xyz.shape[0]),
        "removed_point_count": 0,
        "excluded_image_ids": [],
        "track_filter": "none",
    }


def _read_exact(handle, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise EOFError("Unexpected end of COLMAP points3D.bin")
    return value


def _load_colmap_points_binary(
    path: Path,
    *,
    excluded_image_ids: frozenset[int] = frozenset(),
    return_metadata: bool = False,
):
    """Read XYZ/RGB and optionally remove points seen by excluded images."""

    xyz_values: list[tuple[float, float, float]] = []
    rgb_values: list[tuple[int, int, int]] = []
    removed_points = 0
    with path.open("rb") as handle:
        (num_points,) = struct.unpack("<Q", _read_exact(handle, 8))
        for _ in range(num_points):
            _point_id, x, y, z = struct.unpack("<Qddd", _read_exact(handle, 32))
            red, green, blue = struct.unpack("<BBB", _read_exact(handle, 3))
            _error = struct.unpack("<d", _read_exact(handle, 8))[0]
            (track_length,) = struct.unpack("<Q", _read_exact(handle, 8))
            # Each track element is (image_id:uint32, point2D_idx:uint32).
            track_bytes = _read_exact(handle, int(track_length) * 8)
            track_image_ids = {
                int(image_id)
                for image_id, _point2d_idx in struct.iter_unpack("<II", track_bytes)
            }
            if excluded_image_ids.intersection(track_image_ids):
                removed_points += 1
                continue
            xyz_values.append((x, y, z))
            rgb_values.append((red, green, blue))
        if handle.read(1):
            raise ValueError(f"Trailing bytes after {num_points} points in {path}")
    if not xyz_values:
        raise ValueError(f"COLMAP point cloud is empty: {path}")
    xyz = np.asarray(xyz_values, dtype=np.float32)
    rgb = np.asarray(rgb_values, dtype=np.float32) / 255.0
    if not return_metadata:
        return xyz, rgb
    return xyz, rgb, {
        "source": str(path.resolve()),
        "source_point_count": int(num_points),
        "retained_point_count": int(xyz.shape[0]),
        "removed_point_count": int(removed_points),
        "excluded_image_ids": sorted(int(value) for value in excluded_image_ids),
        "track_filter": (
            "drop_any_point_observed_by_excluded_image"
            if excluded_image_ids
            else "none"
        ),
    }


# ──────────────────────────────────────────────────────────────────
# Scene data loading
# ──────────────────────────────────────────────────────────────────

def load_scene(
    scene_root: str,
    device: torch.device,
    *,
    excluded_image_stems: tuple[str, ...] = (),
    image_dir: str | None = None,
    image_map_json: str | None = None,
    image_scale: float = 1.0,
    image_width: int | None = None,
    image_height: int | None = None,
    return_view_metadata: bool = False,
):
    """Load images, w2c matrices, and intrinsics from a COLMAP scene.

    Images are kept on CPU to save GPU memory; only the active frame is
    moved to *device* during training.

    Returns:
        images: list of [H, W, 3] uint8 tensors on **CPU**
        w2cs:   list of [4, 4] float32 w2c tensors on *device*
        K:      [3, 3] intrinsics tensor on *device*
        W, H:   image width and height
        camera_extent: float, NeRF++ style camera radius for LR scaling
    """
    scene_root = Path(scene_root)
    if image_dir and image_map_json:
        raise ValueError("Use either image_dir or image_map_json, not both")
    colmap = _parse_colmap_sparse(scene_root)

    calibration_w, calibration_h = int(colmap["w"]), int(colmap["h"])
    if not math.isfinite(float(image_scale)) or float(image_scale) <= 0:
        raise ValueError("image_scale must be finite and positive")
    if (image_width is None) != (image_height is None):
        raise ValueError("image_width and image_height must be specified together")
    if image_width is not None and (int(image_width) <= 0 or int(image_height) <= 0):
        raise ValueError("image_width and image_height must be positive")
    if image_width is not None and float(image_scale) != 1.0:
        raise ValueError("Use either explicit image_width/image_height or image_scale")

    fx, fy, cx, cy = colmap["fl_x"], colmap["fl_y"], colmap["cx"], colmap["cy"]

    all_file_paths = list(colmap["file_paths"])
    colmap_index_by_stem: dict[str, int] = {}
    for index, file_path in enumerate(all_file_paths):
        stem = Path(str(file_path)).stem
        if stem in colmap_index_by_stem:
            raise ValueError(f"Duplicate COLMAP camera basename stem {stem!r}")
        colmap_index_by_stem[stem] = index

    image_map_lookup: dict[str, dict] | None = None
    image_map_metadata: dict | None = None
    if image_map_json:
        image_map_lookup, image_map_metadata = _load_colmap_camera_rgb_map(
            image_map_json,
            scene_root=scene_root,
        )
        unknown_mapped_cameras = sorted(set(image_map_lookup) - set(colmap_index_by_stem))
        if unknown_mapped_cameras:
            raise ValueError(
                f"Image map references cameras absent from COLMAP: "
                f"{unknown_mapped_cameras}"
            )
        candidate_indices = [
            index
            for index, file_path in enumerate(all_file_paths)
            if Path(str(file_path)).stem in image_map_lookup
        ]
    else:
        candidate_indices = list(range(len(all_file_paths)))
    candidate_paths = [all_file_paths[index] for index in candidate_indices]
    retained_candidate_indices, excluded_names = select_image_indices(
        candidate_paths,
        excluded_image_stems,
        min_remaining=2,
    )
    retained_indices = [candidate_indices[index] for index in retained_candidate_indices]

    # c2w → w2c. Keep image paths and cameras under the same exact filter.
    c2w_list = [colmap["c2w_list"][index] for index in retained_indices]
    file_paths = [all_file_paths[index] for index in retained_indices]
    w2cs = []
    for c2w in c2w_list:
        w2c = np.linalg.inv(c2w).astype(np.float32)
        w2cs.append(torch.from_numpy(w2c).to(device))

    # Resolve an optional exact image-directory override. This supports
    # downsampled/cropped benchmark RGB pyramids while retaining COLMAP poses.
    override_lookup: dict[str, Path] | None = None
    if image_dir:
        override_root = Path(image_dir).expanduser().resolve()
        if not override_root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {override_root}")
        override_lookup = {}
        for path in sorted(override_root.iterdir(), key=lambda value: value.name):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if path.stem in override_lookup:
                raise ValueError(
                    f"Duplicate image stem {path.stem!r} in {override_root}"
                )
            override_lookup[path.stem] = path

    # Load images (kept on CPU), then scale the calibration to their exact
    # dimensions. All selected views must share one resolution.
    images = []
    resolved_paths: list[Path] = []
    output_size: tuple[int, int] | None = None
    for fp in file_paths:
        full = scene_root / fp
        if image_map_lookup is not None:
            stem = Path(fp).stem
            full = Path(str(image_map_lookup[stem]["rgb_path"]))
        elif override_lookup is not None:
            stem = Path(fp).stem
            if stem not in override_lookup:
                raise FileNotFoundError(
                    f"No exact image stem {stem!r} in override directory {image_dir}"
                )
            full = override_lookup[stem]
        if not full.exists():
            raise FileNotFoundError(f"Image not found: {full}")
        pil_image = Image.open(str(full)).convert("RGB")
        if image_width is not None:
            target_size = (int(image_width), int(image_height))
        elif float(image_scale) != 1.0:
            target_size = (
                max(1, int(round(pil_image.width * float(image_scale)))),
                max(1, int(round(pil_image.height * float(image_scale)))),
            )
        else:
            target_size = pil_image.size
        if output_size is None:
            output_size = target_size
        elif output_size != target_size:
            raise ValueError(
                f"Selected RGB views have inconsistent output sizes: "
                f"{output_size} versus {target_size} for {full}"
            )
        if pil_image.size != target_size:
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            pil_image = pil_image.resize(target_size, resample=resampling)
        # Keep the resident training set compact.  Only one sampled frame is
        # converted to float on the GPU per step below.
        img = np.array(pil_image, dtype=np.uint8)
        images.append(torch.from_numpy(img))  # CPU
        resolved_paths.append(full.resolve())

    assert output_size is not None
    W, H = output_size
    scale_x = float(W) / float(calibration_w)
    scale_y = float(H) / float(calibration_h)
    fx, cx = float(fx) * scale_x, float(cx) * scale_x
    fy, cy = float(fy) * scale_y, float(cy) * scale_y
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device,
    )

    print(f"Loaded {len(images)} images at {W}×{H}, "
          f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    if excluded_names:
        print(f"  Excluded RGB training views: {', '.join(excluded_names)}")
    camera_extent = compute_camera_extent(c2w_list)
    print(f"  Camera extent (NeRF++ radius): {camera_extent:.2f}")
    result = (images, w2cs, K, W, H, camera_extent)
    if not return_view_metadata:
        return result
    metadata = {
        "selection": (
            "locked_explicit_colmap_camera_to_rgb_path_map"
            if image_map_lookup is not None
            else "exact_case_sensitive_basename_stem"
        ),
        "source_view_count": len(all_file_paths),
        "mapped_view_count": (
            len(image_map_lookup) if image_map_lookup is not None else len(all_file_paths)
        ),
        "training_view_count": len(file_paths),
        "training_image_names": [Path(path).name for path in file_paths],
        "training_image_paths": [str(path) for path in resolved_paths],
        "excluded_image_stems": list(excluded_image_stems),
        "excluded_image_names": excluded_names,
        "calibration_source": colmap.get("calibration_source"),
        "calibration_resolution": [calibration_w, calibration_h],
        "training_resolution": [W, H],
        "intrinsics_scale_xy": [scale_x, scale_y],
        "image_map": image_map_metadata,
    }
    canonical = json.dumps(
        metadata["training_image_names"], separators=(",", ":"), ensure_ascii=False
    )
    metadata["training_image_names_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return (*result, metadata)


# ──────────────────────────────────────────────────────────────────
# Gaussian initialisation
# ──────────────────────────────────────────────────────────────────

def compute_scene_scale(xyz: np.ndarray) -> float:
    """Compute scene scale as the extent of the point cloud bounding box."""
    pmin = xyz.min(axis=0)
    pmax = xyz.max(axis=0)
    return float(np.linalg.norm(pmax - pmin))


def compute_camera_extent(c2w_list: list) -> float:
    """Compute camera extent (NeRF++ style) for position LR scaling.

    Returns the radius of the smallest sphere centred at the camera centroid
    that contains all cameras, times 1.1 (standard 3DGS padding).
    """
    cam_centers = np.array([c2w[:3, 3] for c2w in c2w_list])
    centroid = cam_centers.mean(axis=0)
    dists = np.linalg.norm(cam_centers - centroid, axis=1)
    return float(np.max(dists) * 1.1)


def _excluded_colmap_image_ids(
    scene_root: Path,
    excluded_image_stems: tuple[str, ...],
) -> frozenset[int]:
    """Resolve exact held-out stems to COLMAP image IDs for track filtering."""

    if not excluded_image_stems:
        return frozenset()
    images_path = scene_root / "sparse" / "0" / "images.bin"
    if not images_path.is_file():
        raise FileNotFoundError(
            f"Track-safe sparse initialization requires {images_path}"
        )
    by_stem: dict[str, int] = {}
    for entry in _read_images_binary(images_path):
        stem = Path(str(entry["name"])).stem
        if stem in by_stem:
            raise ValueError(
                f"Duplicate COLMAP image basename stem {stem!r} in {images_path}"
            )
        by_stem[stem] = int(entry["image_id"])
    unknown = sorted(set(excluded_image_stems) - set(by_stem))
    if unknown:
        raise ValueError(
            f"Excluded image stems are absent from COLMAP images.bin: {unknown}"
        )
    return frozenset(by_stem[stem] for stem in excluded_image_stems)


def init_gaussians(
    scene_root: str,
    device: torch.device,
    *,
    excluded_image_stems: tuple[str, ...] = (),
):
    """Create initial Gaussian parameters from COLMAP points3D.ply.

    Returns:
        params: dict of nn.Parameter with keys
            means, scales, quats, opacities, sh0, shN
        scene_scale: float
    """
    source_root = Path(scene_root)
    excluded_image_ids = _excluded_colmap_image_ids(
        source_root,
        excluded_image_stems,
    )
    xyz, rgb, sparse_metadata = load_colmap_points(
        source_root,
        excluded_image_ids=excluded_image_ids,
        return_metadata=True,
    )
    N = xyz.shape[0]
    scene_scale = compute_scene_scale(xyz)
    print(f"Initialising {N:,} Gaussians from COLMAP points "
          f"(scene scale={scene_scale:.2f})")

    means = torch.from_numpy(xyz).float().to(device)

    # SH DC from colours: colour = C0 * sh_dc + 0.5  →  sh_dc = (colour - 0.5) / C0
    sh_dc = (torch.from_numpy(rgb).float().to(device) - 0.5) / C0  # [N, 3]
    sh0 = sh_dc.unsqueeze(1)  # [N, 1, 3]

    # Higher-order SH (degree 3 → 15 extra coefficients per channel)
    shN = torch.zeros(N, 15, 3, device=device)

    # Initial log-scales: use a fraction of the mean nearest-neighbour distance
    log_scales = _estimate_initial_scales(means, scene_scale)

    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = 1.0  # identity rotation

    # Logit of initial opacity 0.1 → logit(0.1) ≈ −2.197
    opacities = torch.full((N, 1), math.log(0.1 / 0.9), device=device)

    params = {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(log_scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    }
    sparse_metadata.update(
        {
            "excluded_image_stems": list(excluded_image_stems),
            "target_rgb_observation_policy": (
                "points_with_any_excluded_view_track_removed"
                if excluded_image_stems
                else "no_views_excluded"
            ),
            # Camera poses/intrinsics are retained so the held-out view can be
            # rendered.  They originate from the upstream reconstruction and
            # may have been jointly estimated using all capture views.
            "camera_calibration_prior": "upstream_shared_full_capture",
            "camera_calibration_shared_exception": bool(excluded_image_stems),
        }
    )
    return params, scene_scale, sparse_metadata


def _estimate_initial_scales(means: Tensor, scene_scale: float) -> Tensor:
    """Estimate per-Gaussian initial log-scale from nearest-neighbour distances."""
    N = means.shape[0]
    device = means.device
    cpu_means = means.detach().cpu()

    # For large point clouds, subsample to estimate median NN distance
    max_sample = min(N, 50_000)
    idx = torch.randperm(N)[:max_sample]
    subset = cpu_means[idx]

    # An exact KD-tree computes the same nearest-neighbour statistic without
    # materialising the former O(S^2) 50k-by-50k distance matrix.
    # Use one worker deliberately: large shared hosts can expose hundreds of
    # CPUs, where ``workers=-1`` spends far longer in thread scheduling than
    # this small 3-D query needs.
    distances, _ = cKDTree(subset.numpy()).query(subset.numpy(), k=2, workers=1)
    median_nn = float(np.median(np.asarray(distances, dtype=np.float32)[:, 1]))
    init_scale = max(median_nn * 0.5, scene_scale * 1e-4)
    log_scale = math.log(init_scale)
    print(f"  Median NN distance: {median_nn:.4f}, "
          f"init scale: {init_scale:.4f} (log={log_scale:.3f})")
    return torch.full((N, 3), log_scale, device=device)


# ──────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────

def l1_loss(pred: Tensor, gt: Tensor) -> Tensor:
    return (pred - gt).abs().mean()


_ssim_window_cache: dict = {}


def _gaussian_window(window_size: int, sigma: float, channels: int,
                     dtype: torch.dtype, device: torch.device) -> Tensor:
    """Create a 2-D Gaussian window for SSIM (cached across calls)."""
    key = (window_size, sigma, channels, dtype, device)
    if key not in _ssim_window_cache:
        coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
        _ssim_window_cache[key] = window_2d.unsqueeze(0).unsqueeze(0).expand(
            channels, 1, -1, -1
        ).contiguous()
    return _ssim_window_cache[key]


def ssim_loss(pred: Tensor, gt: Tensor, window_size: int = 11) -> Tensor:
    """Structural similarity loss: 1 − SSIM (Gaussian-weighted window)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    p = pred.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    g = gt.permute(2, 0, 1).unsqueeze(0)
    C = p.shape[1]
    pad = window_size // 2

    # Reflect-pad to avoid border bias from zero-padding
    p = F.pad(p, [pad, pad, pad, pad], mode="reflect")
    g = F.pad(g, [pad, pad, pad, pad], mode="reflect")

    window = _gaussian_window(window_size, 1.5, C, p.dtype, p.device)

    mu_p = F.conv2d(p, window, groups=C)
    mu_g = F.conv2d(g, window, groups=C)
    mu_pp = mu_p * mu_p
    mu_gg = mu_g * mu_g
    mu_pg = mu_p * mu_g

    sigma_pp = F.conv2d(p * p, window, groups=C) - mu_pp
    sigma_gg = F.conv2d(g * g, window, groups=C) - mu_gg
    sigma_pg = F.conv2d(p * g, window, groups=C) - mu_pg

    # Clamp variances to avoid numerical artifacts from E[x²] − E[x]²
    sigma_pp = sigma_pp.clamp(min=0.0)
    sigma_gg = sigma_gg.clamp(min=0.0)

    ssim = ((2.0 * mu_pg + C1) * (2.0 * sigma_pg + C2)) / \
           ((mu_pp + mu_gg + C1) * (sigma_pp + sigma_gg + C2))
    return 1.0 - ssim.mean()


# ──────────────────────────────────────────────────────────────────
# PLY export (standard 3DGS format)
# ──────────────────────────────────────────────────────────────────

def save_ply(path: str, params: dict, sh_degree: int = 3) -> None:
    """Save Gaussians to standard 3DGS PLY format.

    The PLY contains per-vertex properties:
        x, y, z, nx, ny, nz,
        f_dc_0..2, f_rest_0..44,
        opacity, scale_0..2, rot_0..3
    """
    from plyfile import PlyData, PlyElement

    means = params["means"].detach().cpu().numpy()         # [N, 3]
    scales = params["scales"].detach().cpu().numpy()        # [N, 3]
    quats = params["quats"].detach().cpu().numpy()          # [N, 4]
    opacities = params["opacities"].detach().cpu().numpy()  # [N, 1]
    sh0 = params["sh0"].detach().cpu().numpy()              # [N, 1, 3]
    shN = params["shN"].detach().cpu().numpy()              # [N, K, 3]

    N = means.shape[0]
    n_rest = shN.shape[1]  # 15 for degree 3

    # Build structured numpy array
    dtype_list = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ]
    for i in range(3):
        dtype_list.append((f"f_dc_{i}", "f4"))
    # f_rest: stored as [ch0_c0, ch0_c1, ..., ch0_cK, ch1_c0, ..., ch2_cK]
    for i in range(n_rest * 3):
        dtype_list.append((f"f_rest_{i}", "f4"))
    dtype_list.append(("opacity", "f4"))
    for i in range(3):
        dtype_list.append((f"scale_{i}", "f4"))
    for i in range(4):
        dtype_list.append((f"rot_{i}", "f4"))

    arr = np.empty(N, dtype=dtype_list)
    arr["x"] = means[:, 0]
    arr["y"] = means[:, 1]
    arr["z"] = means[:, 2]
    arr["nx"] = 0.0
    arr["ny"] = 0.0
    arr["nz"] = 0.0

    # SH DC
    for i in range(3):
        arr[f"f_dc_{i}"] = sh0[:, 0, i]

    # SH rest — original 3DGS stores as transpose(1,2).flatten():
    #   [ch0_c0, ch0_c1, ..., ch0_cK, ch1_c0, ..., ch2_cK]
    # shN is [N, K, 3], so we need [N, 3, K] flattened to [N, 3*K]
    sh_rest_flat = shN.transpose(0, 2, 1).reshape(N, -1)  # [N, 3*K]
    for i in range(n_rest * 3):
        arr[f"f_rest_{i}"] = sh_rest_flat[:, i]

    arr["opacity"] = opacities[:, 0]
    for i in range(3):
        arr[f"scale_{i}"] = scales[:, i]
    for i in range(4):
        arr[f"rot_{i}"] = quats[:, i]

    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(str(path))
    print(f"Saved {N:,} Gaussians to {path}")


# ──────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load scene data
    excluded_image_stems = load_excluded_image_stems(
        args.exclude_image_stem,
        args.exclude_image_stems_file,
    )
    images, w2cs, K, W, H, camera_extent, view_metadata = load_scene(
        args.scene_root,
        device,
        excluded_image_stems=excluded_image_stems,
        image_dir=args.image_dir or None,
        image_map_json=args.image_map_json or None,
        image_scale=args.image_scale,
        image_width=args.image_width,
        image_height=args.image_height,
        return_view_metadata=True,
    )
    n_frames = len(images)

    # Initialise Gaussians from COLMAP points
    params, scene_scale, sparse_metadata = init_gaussians(
        args.scene_root,
        device,
        excluded_image_stems=excluded_image_stems,
    )
    n_init = params["means"].shape[0]
    view_metadata["sparse_initialization"] = sparse_metadata

    # Background colour
    if args.white_bg:
        bg = torch.ones(3, device=device)
    else:
        bg = torch.zeros(3, device=device)

    # Optimiser with per-parameter learning rates
    # Scale position LR by camera extent (standard 3DGS practice — NeRF++ radius)
    lr_map = {
        "means": args.lr_means * camera_extent,
        "scales": args.lr_scale,
        "quats": args.lr_quat,
        "opacities": args.lr_opacity,
        "sh0": args.lr_sh,
        "shN": args.lr_sh,
    }
    optimizers = {}
    for name, param in params.items():
        optimizers[name] = torch.optim.Adam(
            [param], lr=lr_map[name], eps=1e-15,
        )

    # LR schedulers for positions: warmup (1% → 100% over 1000 steps) + exponential decay
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizers["means"], start_factor=0.01, total_iters=1000,
    )
    decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"],
        gamma=(args.lr_means_final_factor) ** (1.0 / max(args.iters, 1)),
    )
    pos_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [warmup_scheduler, decay_scheduler]
    )

    # Densification strategy
    strategy = DefaultStrategy(
        grow_grad2d=args.densify_grad_thresh,
        refine_start_iter=args.densify_from,
        refine_stop_iter=args.densify_until,
        refine_every=args.densify_every,
        reset_every=args.opacity_reset_every,
        prune_opa=0.005,
        absgrad=True,
        verbose=True,
    )
    state = strategy.initialize_state(scene_scale=scene_scale)

    # SH degree schedule: start at 0, increase every 1000 iters up to sh_degree
    max_sh_degree = args.sh_degree

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_views.json").write_text(
        json.dumps(view_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ply_dir = out_dir / "point_cloud" / f"iteration_{args.iters}"
    ply_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training 3DGS: {n_init:,} initial Gaussians, {args.iters} iterations")
    print(f"Scene: {args.scene_root}")
    print(f"Output: {out_dir}")
    print(f"Scene scale (pts bbox): {scene_scale:.2f}, Camera extent: {camera_extent:.2f}")
    print(f"Position LR: {args.lr_means} × {camera_extent:.2f} = {args.lr_means * camera_extent:.4e}")
    print(f"{'='*60}\n")

    for step in range(args.iters):
        # Current SH degree
        cur_sh_degree = min(step // 1000, max_sh_degree)

        # Random training frame
        idx = random.randint(0, n_frames - 1)
        gt_img = images[idx].to(device=device, dtype=torch.float32).div_(255.0)
        viewmat = w2cs[idx]              # [4, 4]

        # Build SH colour tensor: [N, K_cur, 3]
        n_cur_sh = (cur_sh_degree + 1) ** 2
        if cur_sh_degree == 0:
            colors = params["sh0"]  # [N, 1, 3]
        else:
            colors = torch.cat(
                [params["sh0"], params["shN"][:, : n_cur_sh - 1, :]],
                dim=1,
            )  # [N, K_cur, 3]

        scales = torch.exp(params["scales"])
        opacities = torch.sigmoid(params["opacities"]).squeeze(-1)  # [N]
        quats = F.normalize(params["quats"], p=2, dim=-1)

        # Render
        renders, alphas, info = rasterization(
            means=params["means"],
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=W,
            height=H,
            near_plane=0.01,
            far_plane=1000.0,
            backgrounds=bg.unsqueeze(0),
            sh_degree=cur_sh_degree,
            packed=False,
            absgrad=True,
        )
        pred_rgb = renders[0].clamp(0.0, 1.0)  # [H, W, 3]

        # Loss: L1 + SSIM
        loss_l1 = l1_loss(pred_rgb, gt_img)
        loss_ssim = ssim_loss(pred_rgb, gt_img)
        loss = (1.0 - args.lambda_ssim) * loss_l1 + args.lambda_ssim * loss_ssim

        # Densification: pre-backward
        strategy.step_pre_backward(
            params=params, optimizers=optimizers, state=state,
            step=step, info=info,
        )

        loss.backward()

        # Densification: post-backward
        strategy.step_post_backward(
            params=params, optimizers=optimizers, state=state,
            step=step, info=info, packed=False,
        )

        # Optimiser step
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        pos_scheduler.step()

        # Logging
        if step % args.log_every == 0 or step == args.iters - 1:
            with torch.no_grad():
                mse = ((pred_rgb - gt_img) ** 2).mean()
                psnr = -10.0 * math.log10(mse.item() + 1e-10)
            n_gs = params["means"].shape[0]
            lr_pos = optimizers["means"].param_groups[0]["lr"]
            print(
                f"[Iter {step:>5d}/{args.iters}] "
                f"loss={loss.item():.4f}  PSNR={psnr:.2f} dB  "
                f"#GS={n_gs:,}  SH={cur_sh_degree}  lr_pos={lr_pos:.2e}"
            )

        # Intermediate checkpoint
        if args.save_every > 0 and (step + 1) % args.save_every == 0 and step > 0:
            mid_dir = out_dir / "point_cloud" / f"iteration_{step + 1}"
            mid_dir.mkdir(parents=True, exist_ok=True)
            save_ply(str(mid_dir / "point_cloud.ply"), params, max_sh_degree)

    # Final save
    save_ply(str(ply_dir / "point_cloud.ply"), params, max_sh_degree)
    print(f"\nTraining complete. Final #GS: {params['means'].shape[0]:,}")
    print(f"Output saved to {ply_dir / 'point_cloud.ply'}")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train 3D Gaussian Splatting from COLMAP sparse reconstruction",
    )
    # I/O
    parser.add_argument("--scene_root", required=True,
                        help="Path to COLMAP scene (with sparse/0/ and images/)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for trained model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-dir",
        default="",
        help=(
            "Optional RGB directory override matched to COLMAP cameras by exact "
            "basename stem"
        ),
    )
    parser.add_argument(
        "--image-map-json",
        default="",
        help=(
            "Explicit queue-audited JSON mapping from COLMAP camera names to RGB "
            "paths. This is mutually exclusive with --image-dir."
        ),
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
        help="Resize RGB training views and intrinsics by this scale",
    )
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument(
        "--exclude-image-stem",
        action="append",
        default=[],
        help=(
            "Exact case-sensitive image basename stem to exclude from RGB training; "
            "repeat for multiple held-out views"
        ),
    )
    parser.add_argument(
        "--exclude-image-stems-file",
        default="",
        help="Optional JSON/text file of exact image stems to exclude",
    )

    # Training
    parser.add_argument("--iters", type=int, default=30000)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--lambda_ssim", type=float, default=0.2,
                        help="SSIM weight in combined loss (L1 weight = 1 − this)")

    # Learning rates
    parser.add_argument("--lr_means", type=float, default=1.6e-4)
    parser.add_argument("--lr_means_final_factor", type=float, default=0.01,
                        help="Final LR = lr_means × this factor (exponential decay)")
    parser.add_argument("--lr_sh", type=float, default=2.5e-3)
    parser.add_argument("--lr_scale", type=float, default=5e-3)
    parser.add_argument("--lr_quat", type=float, default=1e-3)
    parser.add_argument("--lr_opacity", type=float, default=0.05)

    # Densification
    parser.add_argument("--densify_from", type=int, default=500)
    parser.add_argument("--densify_until", type=int, default=15000)
    parser.add_argument("--densify_every", type=int, default=100)
    parser.add_argument("--densify_grad_thresh", type=float, default=0.0008)
    parser.add_argument("--opacity_reset_every", type=int, default=3000)

    # Logging / checkpoints
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=0,
                        help="Save intermediate checkpoint every N iters (0=off)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
