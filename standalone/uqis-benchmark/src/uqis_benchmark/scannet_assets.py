"""Minimal official ScanNet asset adapter used by the UQIS constructor.

Keeping ScanNet parsing behind this internal seam makes the benchmark core
exportable without importing RADIO-GS or the older PFIR benchmark package.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree


STRUCTURAL_NYU40_IDS = frozenset({1, 2, 22})
STRUCTURAL_LABELS = frozenset({"wall", "floor", "ceiling"})


@dataclass(frozen=True)
class FramePaths:
    frame_id: str
    rgb: Path
    depth: Path
    instance: Path
    label: Path
    pose: Path


@dataclass
class FrameInstanceObservation:
    encoded_2d_id: int
    nyu40_class_id: int
    pixel_count: int
    image_fraction: float
    border_fraction: float
    bbox_xyxy: tuple[int, int, int, int]
    instance_id_3d: int
    resolution_purity: float
    valid_depth_votes: int
    observed_world_xyz: np.ndarray


def find_scene_annotations(
    scene_id: str, roots: Iterable[str | Path]
) -> tuple[Path, Path, Path]:
    for source_root in roots:
        root = Path(source_root)
        for scene_root in (root / scene_id, root):
            mesh_candidates = (
                scene_root / f"{scene_id}_vh_clean_2.ply",
                scene_root / f"{scene_id}_vh_clean_2.labels.ply",
            )
            for annotation_root in (
                scene_root / "instance_annotations",
                scene_root,
            ):
                aggregation = annotation_root / f"{scene_id}.aggregation.json"
                segmentation_candidates = (
                    annotation_root / f"{scene_id}_vh_clean_2.0.010000.segs.json",
                    annotation_root / f"{scene_id}_vh_clean_2.segs.json",
                )
                mesh = next((path for path in mesh_candidates if path.is_file()), None)
                segmentation = next(
                    (path for path in segmentation_candidates if path.is_file()), None
                )
                if mesh is not None and aggregation.is_file() and segmentation is not None:
                    return mesh, aggregation, segmentation
    raise FileNotFoundError(f"{scene_id}: incomplete official ScanNet annotations")


def load_matrix(path: str | Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid finite 4x4 matrix: {path}")
    return matrix


def discover_frames(scene_dir: str | Path) -> list[FramePaths]:
    root = Path(scene_dir)
    frames: list[FramePaths] = []
    for rgb in sorted((root / "color").glob("*.jpg")):
        paths = {
            "depth": root / "depth" / f"{rgb.stem}.png",
            "instance": root / "instance" / f"{rgb.stem}.png",
            "label": root / "label" / f"{rgb.stem}.png",
            "pose": root / "pose" / f"{rgb.stem}.txt",
        }
        if all(path.is_file() for path in paths.values()):
            try:
                load_matrix(paths["pose"])
            except ValueError:
                continue
            frames.append(FramePaths(rgb.stem, rgb, **paths))
    if not frames:
        raise FileNotFoundError(f"no complete finite ScanNet frames in {root}")
    return frames


def load_mesh_instances(
    mesh_path: str | Path,
    aggregation_path: str | Path,
    segmentation_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
    ply = PlyData.read(str(mesh_path))
    vertex = ply["vertex"].data
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(
        np.float32
    )
    segmentation = json.loads(Path(segmentation_path).read_text(encoding="utf-8"))
    aggregation = json.loads(Path(aggregation_path).read_text(encoding="utf-8"))
    segment_ids = np.asarray(segmentation.get("segIndices", []), dtype=np.int64)
    if segment_ids.shape != (xyz.shape[0],):
        raise ValueError("segIndices and mesh vertex rows do not align")
    instance_ids = np.zeros(xyz.shape[0], dtype=np.int32)
    metadata: dict[int, dict[str, Any]] = {}
    for group in aggregation.get("segGroups", []):
        instance_id = int(group["objectId"]) + 1
        selected = np.isin(
            segment_ids, np.asarray(group.get("segments", []), dtype=np.int64)
        )
        if bool((instance_ids[selected] != 0).any()):
            raise ValueError(f"overlapping ScanNet segments for instance {instance_id}")
        instance_ids[selected] = instance_id
        metadata[instance_id] = {
            "object_id": int(group["objectId"]),
            "label": str(group.get("label", "")),
            "num_vertices": int(selected.sum()),
        }
    if not metadata:
        raise ValueError("aggregation contains no 3-D instances")
    return xyz, instance_ids, metadata


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    y, x = np.nonzero(mask)
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def _border_fraction(mask: np.ndarray, width: int = 2) -> float:
    border = np.zeros_like(mask, dtype=bool)
    border[:width] = border[-width:] = True
    border[:, :width] = border[:, -width:] = True
    return float(np.logical_and(mask, border).sum() / max(int(mask.sum()), 1))


def _depth_to_labeled_world(
    depth: np.ndarray,
    instance_image: np.ndarray,
    depth_intrinsic: np.ndarray,
    color_intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    yy, xx = np.mgrid[0:height:int(stride), 0:width:int(stride)]
    z = depth[yy, xx].astype(np.float64) / 1000.0
    valid = np.isfinite(z) & (z > 0.20) & (z < 8.0)
    x = (xx - depth_intrinsic[0, 2]) * z / depth_intrinsic[0, 0]
    y = (yy - depth_intrinsic[1, 2]) * z / depth_intrinsic[1, 1]
    uc = color_intrinsic[0, 0] * x / np.maximum(z, 1e-8) + color_intrinsic[0, 2]
    vc = color_intrinsic[1, 1] * y / np.maximum(z, 1e-8) + color_intrinsic[1, 2]
    ui, vi = np.rint(uc).astype(np.int64), np.rint(vc).astype(np.int64)
    valid &= (ui >= 0) & (vi >= 0) & (ui < instance_image.shape[1]) & (
        vi < instance_image.shape[0]
    )
    encoded = instance_image[vi[valid], ui[valid]].astype(np.int64)
    camera = np.stack(
        [x[valid], y[valid], z[valid], np.ones(int(valid.sum()))], axis=1
    )
    return (camera @ camera_to_world.T)[:, :3].astype(np.float32), encoded


def resolve_frame_observations(
    frame: FramePaths,
    mesh_xyz: np.ndarray,
    mesh_instance_ids: np.ndarray,
    depth_intrinsic: np.ndarray,
    color_intrinsic: np.ndarray,
    *,
    depth_stride: int = 2,
    maximum_mesh_distance_m: float = 0.08,
    mesh_tree: cKDTree | None = None,
) -> dict[int, FrameInstanceObservation]:
    instance_image = np.asarray(Image.open(frame.instance), dtype=np.int64)
    label_image = np.asarray(Image.open(frame.label), dtype=np.int64)
    depth = np.asarray(Image.open(frame.depth), dtype=np.uint16)
    if instance_image.shape != label_image.shape:
        raise ValueError(f"instance/label mismatch in {frame.frame_id}")
    world, encoded = _depth_to_labeled_world(
        depth,
        instance_image,
        depth_intrinsic,
        color_intrinsic,
        load_matrix(frame.pose),
        stride=depth_stride,
    )
    nonzero = encoded > 0
    world, encoded = world[nonzero], encoded[nonzero]
    if world.size:
        tree = mesh_tree if mesh_tree is not None else cKDTree(mesh_xyz)
        distance, nearest = tree.query(world, k=1)
        matched = mesh_instance_ids[nearest]
        valid_match = (
            np.isfinite(distance)
            & (distance <= float(maximum_mesh_distance_m))
            & (matched > 0)
        )
    else:
        matched, valid_match = np.empty(0, dtype=np.int32), np.empty(0, dtype=bool)
    result: dict[int, FrameInstanceObservation] = {}
    for encoded_id in np.unique(instance_image):
        encoded_id = int(encoded_id)
        if encoded_id <= 0:
            continue
        mask = instance_image == encoded_id
        labels, counts = np.unique(label_image[mask], return_counts=True)
        selected_labels = labels > 0
        nyu40_id = (
            int(labels[selected_labels][np.argmax(counts[selected_labels])])
            if bool(selected_labels.any())
            else 0
        )
        selected = (encoded == encoded_id) & valid_match
        votes = matched[selected]
        if votes.size:
            vote_ids, vote_counts = np.unique(votes, return_counts=True)
            best = int(np.argmax(vote_counts))
            instance_id = int(vote_ids[best])
            purity = float(vote_counts[best] / votes.size)
            points = world[selected][votes == instance_id]
        else:
            instance_id, purity = 0, 0.0
            points = np.empty((0, 3), dtype=np.float32)
        result[encoded_id] = FrameInstanceObservation(
            encoded_2d_id=encoded_id,
            nyu40_class_id=nyu40_id,
            pixel_count=int(mask.sum()),
            image_fraction=float(mask.mean()),
            border_fraction=_border_fraction(mask),
            bbox_xyxy=_tight_bbox(mask),
            instance_id_3d=instance_id,
            resolution_purity=purity,
            valid_depth_votes=int(votes.size),
            observed_world_xyz=points.astype(np.float32, copy=False),
        )
    return result
