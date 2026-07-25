#!/usr/bin/env python3
"""Materialize a strict, RGB-D-only field source for ScanNet-PFPR.

The pose-free query benchmark deliberately withholds every RGB-D frame that
produced one of its public RGB patches.  Earlier PFIR materialization also
withheld neighbouring prompt frames that are irrelevant to PFPR and AGILE3D.
This utility produces the narrower, auditable contract used by the next
canonical-field rebuild:

``all dense RGB-D observations - exact PFPR source frames``.

It reads only the private evaluator's ``scene_id`` and ``source_frame_id`` to
make that split.  It never reads or writes anchors, depth pixels, masks,
instance IDs, semantic labels, click trajectories, or metrics.  The materialized
directory contains RGB, depth, pose, and camera calibration files only.
"""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
from PIL import Image

from radio_gs.benchmarks.scannet_pfir.protocol import canonical_json_sha256, sha256_file
from radio_gs.scripts.prepare_scannet_scene import SensorData

from .protocol import (
    PFPR_V2_BENCHMARK_VERSION,
    query_frame_exclusion_digest,
    validate_benchmark_version,
)


FRAME_MODALITIES = {
    "color": (".jpg", ".jpeg", ".png"),
    "depth": (".png",),
    "pose": (".txt",),
}
CAMERA_FILES = (
    "intrinsics_color.txt",
    "intrinsics_depth.txt",
    "extrinsics_color.txt",
    "extrinsics_depth.txt",
)
SCANNET_CAMERA_LAYOUT = {
    "intrinsics_color.txt": "intrinsic_color.txt",
    "intrinsics_depth.txt": "intrinsic_depth.txt",
    "extrinsics_color.txt": "extrinsic_color.txt",
    "extrinsics_depth.txt": "extrinsic_depth.txt",
}
FIELD_CONTRACT_VERSION = "scannet-pfpr-query-heldout-field-v1"
AGILE_DENSE_FIELD_CONTRACT_VERSION = "scannet-agile-dense-observation-field-v1"
# This contract is intentionally about the *source sequence*, not the number
# of decoded frames.  A full ScanNet .sens stream may be deterministically
# subsampled by a query-free view selector before field training; it still has
# materially different provenance from the sparse scannet_frames_25k helper.
SCANNET_FULL_OBSERVATION_FIELD_CONTRACT_VERSION = "scannet_full_observation_v1"
# PFPR must withhold the source RGB-D frame of each pose-free patch.  This
# version retains the same complete `.sens` provenance as the AGILE contract,
# but makes the exact, anchor-free source-frame exclusion auditable.
SCANNET_FULL_OBSERVATION_PFPR_FIELD_CONTRACT_VERSION = (
    "scannet_full_observation_pfpr_queryheldout_v1"
)
FIELD_SOURCE_CONTRACT_FILENAME = "field_source_contract.json"


def _numeric_files(directory: Path, suffixes: Iterable[str]) -> dict[str, Path]:
    allowed = {value.lower() for value in suffixes}
    output: dict[str, Path] = {}
    if not directory.is_dir():
        raise FileNotFoundError(f"frame modality directory is missing: {directory}")
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        try:
            frame_id = f"{int(path.stem):06d}"
        except ValueError:
            continue
        if frame_id in output:
            raise ValueError(f"duplicate numeric frame {frame_id} under {directory}")
        output[frame_id] = path
    return output


def _place(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"refusing to replace existing field input: {destination}")
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"unsupported materialization mode: {mode}")


def _finite_pose_frame_ids(poses: dict[str, Path]) -> set[str]:
    """Keep only poses accepted identically by geometry and MPR stages."""

    valid: set[str] = set()
    for frame_id, path in poses.items():
        try:
            pose = np.loadtxt(path, dtype=np.float64)
        except (OSError, ValueError):
            continue
        if pose.shape == (4, 4) and bool(np.isfinite(pose).all()):
            valid.add(frame_id)
    return valid


def _query_source_frames(
    method_manifest: dict,
    evaluator_manifest: dict,
) -> dict[str, set[str]]:
    """Return only source-frame exclusions, with no anchor dependence."""

    method_scenes = {
        str(row["scene_id"])
        for row in method_manifest.get("queries", [])
        if str(row.get("scene_id", ""))
    }
    if not method_scenes:
        raise ValueError("PFPR method manifest contains no scenes")
    exclusions: dict[str, set[str]] = {scene: set() for scene in method_scenes}
    for row in evaluator_manifest.get("queries", []):
        scene = str(row.get("scene_id", ""))
        if scene not in exclusions:
            continue
        # Access only the source-frame key.  Explicitly do not inspect private
        # anchor/depth-pixel fields carried by the evaluator manifest.
        frame = row.get("source_frame_id")
        if frame is None:
            raise ValueError(f"{scene}: evaluator query lacks source_frame_id")
        try:
            exclusions[scene].add(f"{int(str(frame)):06d}")
        except ValueError as error:
            raise ValueError(f"{scene}: invalid PFPR source frame {frame!r}") from error
    empty = sorted(scene for scene, frames in exclusions.items() if not frames)
    if empty:
        raise ValueError(f"PFPR evaluator has no source-frame exclusions for {empty}")
    return exclusions


def _validate_public_exclusion_commitments(
    evaluator_manifest: Mapping[str, object],
    exclusions: Mapping[str, set[str]],
) -> None:
    """Check v2's public held-out-frame commitments before decoding.

    This uses only scene IDs, source-frame IDs, and public one-way digests.
    It deliberately avoids evaluator-only anchors and depth pixels while
    proving that the field split is the one advertised by the release.
    """

    version = validate_benchmark_version(
        str(evaluator_manifest.get("benchmark_version", ""))
    )
    if version != PFPR_V2_BENCHMARK_VERSION:
        return
    committed: dict[str, str] = {}
    for item in evaluator_manifest.get("scene_domains", []):
        if not isinstance(item, Mapping):
            raise ValueError("PFPR v2 scene-domain record is invalid")
        scene = str(item.get("scene_id", ""))
        digest = str(item.get("excluded_query_source_frame_ids_sha256", ""))
        if not scene or not digest or scene in committed:
            raise ValueError("PFPR v2 public exclusion commitment is invalid")
        committed[scene] = digest
    if set(committed) != set(exclusions):
        raise ValueError("PFPR v2 public exclusion commitments do not cover its scenes")
    for scene, frames in exclusions.items():
        if query_frame_exclusion_digest(sorted(frames, key=int)) != committed[scene]:
            raise ValueError(
                f"{scene}: PFPR v2 held-out source frames disagree with its public commitment"
            )


def _cli_scene_ids(values: Iterable[str], scene_file: str = "") -> tuple[str, ...]:
    """Read explicit scene IDs without letting a file path become one ID."""

    result = [str(value) for value in values if str(value).strip()]
    if str(scene_file).strip():
        path = Path(scene_file)
        if not path.is_file():
            raise FileNotFoundError(f"scene list is missing: {path}")
        result.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return tuple(sorted(set(result)))


def _assert_manifest_versions(method_manifest: dict, evaluator_manifest: dict) -> None:
    method_version = str(method_manifest.get("benchmark_version", ""))
    evaluator_version = str(evaluator_manifest.get("benchmark_version", ""))
    if not method_version or method_version != evaluator_version:
        raise ValueError("method/evaluator PFPR benchmark versions disagree")


def materialize_pfpr_field_contract(
    method_manifest_path: str | Path,
    evaluator_manifest_path: str | Path,
    dense_root: str | Path,
    output_root: str | Path,
    *,
    mode: str = "symlink",
    scenes: Iterable[str] = (),
) -> dict:
    """Create all-dense-except-query-source field inputs without labels."""

    method_path = Path(method_manifest_path).resolve()
    evaluator_path = Path(evaluator_manifest_path).resolve()
    dense_root = Path(dense_root).resolve()
    output_root = Path(output_root).resolve()
    method_manifest = json.loads(method_path.read_text(encoding="utf-8"))
    evaluator_manifest = json.loads(evaluator_path.read_text(encoding="utf-8"))
    _assert_manifest_versions(method_manifest, evaluator_manifest)
    exclusions = _query_source_frames(method_manifest, evaluator_manifest)
    requested = {str(value) for value in scenes}
    if requested:
        missing = sorted(requested - set(exclusions))
        if missing:
            raise KeyError(f"requested scenes absent from PFPR manifest: {missing}")
        exclusions = {scene: exclusions[scene] for scene in sorted(requested)}

    reports: list[dict] = []
    for scene, query_frames in sorted(exclusions.items()):
        source_root = dense_root / scene
        destination_root = output_root / scene
        if not source_root.is_dir():
            raise FileNotFoundError(f"missing dense ScanNet scene: {source_root}")
        available = {
            modality: _numeric_files(source_root / modality, suffixes)
            for modality, suffixes in FRAME_MODALITIES.items()
        }
        common = set.intersection(*(set(values) for values in available.values()))
        finite_pose = _finite_pose_frame_ids(available["pose"])
        common &= finite_pose
        if not common:
            raise ValueError(f"{scene}: no common RGB-D-pose observations")
        missing_query = sorted(query_frames - common)
        if missing_query:
            raise FileNotFoundError(
                f"{scene}: PFPR source frames are absent from dense observations: "
                f"{missing_query[:5]}"
            )
        field_ids = sorted(common - query_frames, key=int)
        if not field_ids:
            raise ValueError(f"{scene}: excluding PFPR source frames leaves no field observations")

        placed: dict[str, int] = {}
        for modality in FRAME_MODALITIES:
            for frame_id in field_ids:
                source = available[modality][frame_id]
                _place(source, destination_root / modality / source.name, mode)
            placed[modality] = len(field_ids)
        intrinsic_dir = destination_root / "intrinsic"
        for name in CAMERA_FILES:
            source = source_root / name
            if not source.is_file():
                raise FileNotFoundError(f"{scene}: missing camera calibration {source}")
            _place(source, destination_root / name, mode)
            _place(source, intrinsic_dir / SCANNET_CAMERA_LAYOUT[name], mode)

        # The contract is intentionally anchor-free.  A digest proves the
        # exact excluded source-frame set without exporting any private
        # per-query coordinate or pixel record to the field directory.
        query_frame_digest = hashlib.sha256(
            canonical_json_sha256(sorted(query_frames, key=int)).encode("utf-8")
        ).hexdigest()
        record = {
            "field_contract_version": FIELD_CONTRACT_VERSION,
            "scene_id": scene,
            "source_root": str(source_root),
            "output_root": str(destination_root),
            "materialization_mode": str(mode),
            "field_frame_count": len(field_ids),
            "field_frame_manifest_sha256": canonical_json_sha256(field_ids),
            "excluded_query_source_frame_count": len(query_frames),
            "invalid_or_nonfinite_pose_frame_count": len(
                set(available["pose"]) - finite_pose
            ),
            "excluded_query_source_frame_ids_sha256": query_frame_digest,
            "placed_frame_counts": placed,
            "uses_private_anchor": False,
            "uses_private_depth_pixel": False,
            "uses_instances_or_semantic_labels": False,
            "contains_instance_or_label_directories": any(
                (destination_root / name).exists() for name in ("instance", "label")
            ),
        }
        if record["contains_instance_or_label_directories"]:
            raise RuntimeError(f"{scene}: field materialization unexpectedly contains labels")
        destination_root.mkdir(parents=True, exist_ok=True)
        (destination_root / "pfpr_field_contract.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        reports.append(record)

    report = {
        "field_contract_version": FIELD_CONTRACT_VERSION,
        "benchmark_version": str(method_manifest["benchmark_version"]),
        "method_manifest": str(method_path),
        "method_manifest_sha256": sha256_file(method_path),
        "private_evaluator_manifest_sha256": sha256_file(evaluator_path),
        "dense_root": str(dense_root),
        "output_root": str(output_root),
        "scene_count": len(reports),
        "scenes": reports,
        "uses_private_anchor": False,
        "uses_private_depth_pixel": False,
        "uses_instances_or_semantic_labels": False,
        "valid": bool(reports) and all(
            not row["contains_instance_or_label_directories"] for row in reports
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "materialization_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def materialize_agile_dense_field_contract(
    dense_root: str | Path,
    output_root: str | Path,
    *,
    mode: str = "symlink",
    scenes: Iterable[str],
) -> dict:
    """Materialize every valid dense RGB-D observation for an AGILE pilot.

    Unlike pose-free patch retrieval, an AGILE3D world-click query has no RGB
    source frame to hold out.  This function therefore copies *all* valid
    registered RGB/depth/pose triples from an explicitly supplied dense pilot
    root while excluding its label and instance directories.  It never opens
    the AGILE object list, PLY labels, click trajectories, or any prediction.

    The result is deliberately named an all-observations *pilot* contract: it
    is suitable for the fixed dense-view overlap subset, but does not claim
    the unavailable full-312 ScanNet observation source.
    """

    dense_root = Path(dense_root).resolve()
    output_root = Path(output_root).resolve()
    requested = sorted({str(scene) for scene in scenes})
    if not requested:
        raise ValueError("an AGILE dense field contract requires at least one scene")

    reports: list[dict] = []
    for scene in requested:
        source_root = dense_root / scene
        destination_root = output_root / scene
        if not source_root.is_dir():
            raise FileNotFoundError(f"missing dense ScanNet scene: {source_root}")
        available = {
            modality: _numeric_files(source_root / modality, suffixes)
            for modality, suffixes in FRAME_MODALITIES.items()
        }
        common = set.intersection(*(set(values) for values in available.values()))
        finite_pose = _finite_pose_frame_ids(available["pose"])
        common &= finite_pose
        field_ids = sorted(common, key=int)
        if not field_ids:
            raise ValueError(f"{scene}: no common finite RGB-D-pose observations")

        placed: dict[str, int] = {}
        for modality in FRAME_MODALITIES:
            for frame_id in field_ids:
                source = available[modality][frame_id]
                _place(source, destination_root / modality / source.name, mode)
            placed[modality] = len(field_ids)
        intrinsic_dir = destination_root / "intrinsic"
        for name in CAMERA_FILES:
            source = source_root / name
            if not source.is_file():
                raise FileNotFoundError(f"{scene}: missing camera calibration {source}")
            _place(source, destination_root / name, mode)
            _place(source, intrinsic_dir / SCANNET_CAMERA_LAYOUT[name], mode)

        record = {
            "field_contract_version": AGILE_DENSE_FIELD_CONTRACT_VERSION,
            "scene_id": scene,
            "source_root": str(source_root),
            "output_root": str(destination_root),
            "materialization_mode": str(mode),
            "source_policy": "all_valid_dense_rgbd_observations",
            "field_frame_count": len(field_ids),
            "field_frame_manifest_sha256": canonical_json_sha256(field_ids),
            "excluded_query_source_frame_count": 0,
            "invalid_or_nonfinite_pose_frame_count": len(
                set(available["pose"]) - finite_pose
            ),
            "placed_frame_counts": placed,
            "uses_private_anchor": False,
            "uses_private_depth_pixel": False,
            "uses_instances_or_semantic_labels": False,
            "contains_instance_or_label_directories": any(
                (destination_root / name).exists() for name in ("instance", "label")
            ),
        }
        if record["contains_instance_or_label_directories"]:
            raise RuntimeError(f"{scene}: AGILE field materialization contains labels")
        destination_root.mkdir(parents=True, exist_ok=True)
        (destination_root / FIELD_SOURCE_CONTRACT_FILENAME).write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        reports.append(record)

    report = {
        "field_contract_version": AGILE_DENSE_FIELD_CONTRACT_VERSION,
        "dense_root": str(dense_root),
        "output_root": str(output_root),
        "scene_count": len(reports),
        "scenes": reports,
        "source_policy": "all_valid_dense_rgbd_observations",
        "uses_private_anchor": False,
        "uses_private_depth_pixel": False,
        "uses_instances_or_semantic_labels": False,
        "valid": bool(reports) and all(
            not row["contains_instance_or_label_directories"] for row in reports
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "materialization_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def select_pose_diverse_frame_indices(
    poses: dict[int, np.ndarray],
    *,
    max_frames: int,
    candidate_stride: int = 1,
    orientation_weight: float = 0.25,
) -> list[int]:
    """Select query-free registered views by deterministic pose-space FPS.

    The selector sees only finite camera-to-world transforms.  It neither
    decodes labels nor consults evaluator geometry, clicks, objects, or any
    downstream score.  Translation is normalized by the source trajectory
    diameter and the camera forward axis adds a fixed orientation term, which
    prevents a long in-place rotation from collapsing to a single view.

    ``max_frames == 0`` means retain every valid candidate.  Returned indices
    are sorted chronologically; the selection order is deliberately irrelevant
    to geometry/MPR training and is recorded separately in the field contract.
    """

    maximum = int(max_frames)
    stride = int(candidate_stride)
    if maximum < 0:
        raise ValueError("max_frames must be non-negative")
    if stride <= 0:
        raise ValueError("candidate_stride must be positive")
    if float(orientation_weight) < 0:
        raise ValueError("orientation_weight must be non-negative")

    candidates: list[tuple[int, np.ndarray]] = []
    for index, pose in sorted(poses.items()):
        matrix = np.asarray(pose, dtype=np.float64)
        if (
            int(index) % stride != 0
            or matrix.shape != (4, 4)
            or not bool(np.isfinite(matrix).all())
        ):
            continue
        candidates.append((int(index), matrix))
    if not candidates:
        return []
    if maximum == 0 or maximum >= len(candidates):
        return [index for index, _pose in candidates]

    centers = np.stack([pose[:3, 3] for _index, pose in candidates])
    forward = np.stack([pose[:3, 2] for _index, pose in candidates])
    forward_norm = np.linalg.norm(forward, axis=1, keepdims=True)
    forward = forward / np.maximum(forward_norm, 1e-12)
    trajectory_diameter = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    translation = (centers - centers.mean(axis=0, keepdims=True)) / max(
        trajectory_diameter, 1e-6
    )
    embedding = np.concatenate(
        [translation, forward * float(orientation_weight)], axis=1
    )

    # Start at the earliest available frame.  np.argmax deterministically
    # resolves later ties to the first remaining chronological candidate.
    selected = [0]
    nearest_distance2 = ((embedding - embedding[0]) ** 2).sum(axis=1)
    nearest_distance2[0] = -np.inf
    while len(selected) < maximum:
        next_index = int(np.argmax(nearest_distance2))
        if not np.isfinite(nearest_distance2[next_index]):
            break
        selected.append(next_index)
        distance2 = ((embedding - embedding[next_index]) ** 2).sum(axis=1)
        nearest_distance2 = np.minimum(nearest_distance2, distance2)
        nearest_distance2[selected] = -np.inf
    return sorted(candidates[index][0] for index in selected)


def _pack_voxel_coordinates(indices: np.ndarray) -> np.ndarray:
    """Losslessly pack a bounded signed voxel triplet into one ``int64``.

    ScanNet rooms are many orders of magnitude smaller than the 20-bit signed
    range at the 5 cm coverage resolution.  Packing keeps the greedy selector
    memory-bounded without relying on a hash (and therefore without accidental
    voxel collisions).
    """

    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("voxel indices must be [N,3]")
    offset = 1 << 19
    limit = offset - 1
    if values.size and (int(values.min()) < -offset or int(values.max()) > limit):
        raise ValueError("coverage voxel index exceeds the supported signed 20-bit range")
    shifted = values + offset
    return (shifted[:, 0] << 40) | (shifted[:, 1] << 20) | shifted[:, 2]


def _depth_coverage_voxels(
    depth: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic_depth: np.ndarray,
    *,
    depth_scale: float,
    voxel_size_m: float,
    depth_stride: int,
) -> np.ndarray:
    """Return unique query-free world-space surface voxels from one RGB-D view."""

    image = np.asarray(depth)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_depth, dtype=np.float64)
    if image.ndim != 2 or pose.shape != (4, 4) or intrinsic.shape != (4, 4):
        raise ValueError("depth coverage expects [H,W] depth and finite 4x4 camera matrices")
    if not bool(np.isfinite(pose).all()) or not bool(np.isfinite(intrinsic).all()):
        return np.empty(0, dtype=np.int64)
    if depth_scale <= 0 or voxel_size_m <= 0 or depth_stride <= 0:
        raise ValueError("depth coverage scale, voxel size, and stride must be positive")
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    if fx <= 0 or fy <= 0:
        raise ValueError("depth intrinsics need positive focal lengths")
    yy, xx = np.mgrid[0 : image.shape[0] : int(depth_stride), 0 : image.shape[1] : int(depth_stride)]
    z = image[yy, xx].astype(np.float64) / float(depth_scale)
    valid = np.isfinite(z) & (z > 0.20) & (z < 8.0)
    if not bool(valid.any()):
        return np.empty(0, dtype=np.int64)
    z = z[valid]
    x = (xx[valid] - float(intrinsic[0, 2])) * z / fx
    y = (yy[valid] - float(intrinsic[1, 2])) * z / fy
    camera = np.stack([x, y, z, np.ones_like(z)], axis=1)
    world = (camera @ pose.T)[:, :3]
    return np.unique(_pack_voxel_coordinates(np.floor(world / float(voxel_size_m)).astype(np.int64)))


def select_depth_coverage_frame_indices(
    frames: Iterable[object],
    poses: dict[int, np.ndarray],
    *,
    intrinsic_depth: np.ndarray,
    depth_width: int,
    depth_height: int,
    depth_compression_type: str,
    depth_scale: float = 1000.0,
    max_frames: int,
    candidate_stride: int = 1,
    voxel_size_m: float = 0.05,
    depth_stride: int = 8,
) -> tuple[list[int], list[int], int]:
    """Select views by greedy world-surface coverage without evaluator data.

    A lazy greedy set-cover implementation avoids the quadratic cost of
    repeatedly rescoring every ScanNet frame.  It sees only depth, calibration
    and registered poses; masks, object lists, query inputs and labels are
    never opened.  The second return value preserves the selection order for
    auditability while the first is chronological for the renderer.
    """

    maximum = int(max_frames)
    stride = int(candidate_stride)
    if maximum < 0 or stride <= 0:
        raise ValueError("max_frames must be non-negative and candidate_stride positive")
    if voxel_size_m <= 0 or depth_stride <= 0 or depth_scale <= 0:
        raise ValueError("coverage voxel size, depth stride, and scale must be positive")
    frame_list = list(frames)
    candidate_sets: list[tuple[int, set[int]]] = []
    for index, pose in sorted(poses.items()):
        if index < 0 or index >= len(frame_list) or int(index) % stride != 0:
            continue
        matrix = np.asarray(pose, dtype=np.float64)
        if matrix.shape != (4, 4) or not bool(np.isfinite(matrix).all()):
            continue
        depth = np.asarray(
            frame_list[int(index)].decompress_depth(depth_compression_type),
            dtype=np.uint16,
        ).reshape(int(depth_height), int(depth_width))
        keys = _depth_coverage_voxels(
            depth,
            matrix,
            intrinsic_depth,
            depth_scale=float(depth_scale),
            voxel_size_m=float(voxel_size_m),
            depth_stride=int(depth_stride),
        )
        if keys.size:
            candidate_sets.append((int(index), {int(key) for key in keys.tolist()}))
    if not candidate_sets:
        raise ValueError("full .sens contains no finite RGB-D surface samples for coverage selection")
    if maximum == 0 or maximum >= len(candidate_sets):
        ordered = [index for index, _keys in candidate_sets]
        return ordered, ordered, len(set().union(*(keys for _index, keys in candidate_sets)))

    # Every cached gain is an upper bound after coverage only grows.  Lazy
    # rescoring therefore returns the same deterministic greedy solution while
    # keeping full ScanNet selection practical on CPU.
    heap = [(-len(keys), index, position) for position, (index, keys) in enumerate(candidate_sets)]
    heapq.heapify(heap)
    covered: set[int] = set()
    selected_order: list[int] = []
    while heap and len(selected_order) < maximum:
        _negative_bound, index, position = heapq.heappop(heap)
        keys = candidate_sets[position][1]
        exact_gain = sum(key not in covered for key in keys)
        next_bound = -heap[0][0] if heap else -1
        if exact_gain < next_bound:
            heapq.heappush(heap, (-exact_gain, index, position))
            continue
        selected_order.append(index)
        covered.update(keys)
    if not selected_order:
        raise RuntimeError("coverage selector unexpectedly chose no RGB-D frame")
    return sorted(selected_order), selected_order, len(covered)


def _resolve_sens_path(sens_root: Path, scene_id: str) -> Path:
    """Resolve either the standard ``root/scene/scene.sens`` or flat layout."""

    candidates = (
        sens_root / scene_id / f"{scene_id}.sens",
        sens_root / f"{scene_id}.sens",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"missing ScanNet .sens for {scene_id}; checked "
        + ", ".join(str(path) for path in candidates)
    )


def _save_matrix(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(matrix, dtype=np.float64), fmt="%.8f")


def materialize_scannet_full_observation_field_contract(
    sens_root: str | Path,
    output_root: str | Path,
    *,
    scenes: Iterable[str],
    max_frames: int = 240,
    candidate_stride: int = 1,
    orientation_weight: float = 0.25,
    selection_policy: str = "depth_voxel_coverage",
    coverage_voxel_size_m: float = 0.05,
    coverage_depth_stride: int = 8,
    sensor_factory: Callable[[Path], object] = SensorData,
    force: bool = False,
    excluded_query_frame_indices: Mapping[str, Iterable[int]] | None = None,
    field_contract_version: str = SCANNET_FULL_OBSERVATION_FIELD_CONTRACT_VERSION,
) -> dict:
    """Decode query-free RGB-D fields from full registered ScanNet ``.sens``.

    This is the formal-source counterpart to the dense-pilot materializer.
    It requires only a registered sensor stream and writes no instance/semantic
    projections.  The deterministic pose-space selection bounds storage and
    field-training cost while retaining an auditable link to the complete
    sequence.  The default selector greedily covers the scene's own RGB-D
    surface at 5 cm; ``pose_diverse`` remains as a reproducible lightweight
    ablation.  A subsequent direct AGILE evaluator must still enforce its
    pre-label continuous-support gate before aggregating objects.
    """

    source_root = Path(sens_root).resolve()
    destination_root = Path(output_root).resolve()
    requested = sorted({str(scene) for scene in scenes})
    if not requested:
        raise ValueError("a full-observation field contract requires at least one scene")
    selection_policy = str(selection_policy)
    if selection_policy not in {"depth_voxel_coverage", "pose_diverse"}:
        raise ValueError("selection_policy must be depth_voxel_coverage or pose_diverse")
    field_contract_version = str(field_contract_version)
    if field_contract_version not in {
        SCANNET_FULL_OBSERVATION_FIELD_CONTRACT_VERSION,
        SCANNET_FULL_OBSERVATION_PFPR_FIELD_CONTRACT_VERSION,
    }:
        raise ValueError("unsupported full-observation field contract version")
    exclusions = {
        str(scene): {int(index) for index in indices}
        for scene, indices in (excluded_query_frame_indices or {}).items()
    }
    unknown_exclusion_scenes = set(exclusions) - set(requested)
    if unknown_exclusion_scenes:
        raise KeyError(
            "full-observation exclusions contain unrequested scenes: "
            f"{sorted(unknown_exclusion_scenes)}"
        )
    if (
        field_contract_version
        == SCANNET_FULL_OBSERVATION_PFPR_FIELD_CONTRACT_VERSION
        and not exclusions
    ):
        raise ValueError("the PFPR full-observation contract requires source-frame exclusions")

    reports: list[dict] = []
    for scene in requested:
        sens_path = _resolve_sens_path(source_root, scene)
        scene_root = destination_root / scene
        contract_path = scene_root / FIELD_SOURCE_CONTRACT_FILENAME
        if scene_root.exists() and any(scene_root.iterdir()):
            if not force:
                raise FileExistsError(
                    f"refusing to replace existing full-observation field input: {scene_root}"
                )
            shutil.rmtree(scene_root)
        sensor = sensor_factory(sens_path)
        frames = list(getattr(sensor, "frames"))
        poses = {
            index: np.asarray(getattr(frame, "camera_to_world"), dtype=np.float64)
            for index, frame in enumerate(frames)
        }
        excluded = exclusions.get(scene, set())
        invalid_exclusions = sorted(
            index for index in excluded if index < 0 or index >= len(frames)
        )
        if invalid_exclusions:
            raise ValueError(
                f"{scene}: query source frame is absent from full .sens: "
                f"{invalid_exclusions[:5]}"
            )
        valid_pose_indices = [
            index
            for index, pose in poses.items()
            if pose.shape == (4, 4) and bool(np.isfinite(pose).all())
        ]
        candidate_poses = {
            index: pose for index, pose in poses.items() if index not in excluded
        }
        candidate_valid_pose_indices = [
            index
            for index, pose in candidate_poses.items()
            if pose.shape == (4, 4) and bool(np.isfinite(pose).all())
        ]
        selection_order: list[int]
        coverage_voxel_count: int | None = None
        if selection_policy == "depth_voxel_coverage":
            selected, selection_order, coverage_voxel_count = select_depth_coverage_frame_indices(
                frames,
                candidate_poses,
                intrinsic_depth=np.asarray(getattr(sensor, "intrinsic_depth")),
                depth_width=int(getattr(sensor, "depth_width")),
                depth_height=int(getattr(sensor, "depth_height")),
                depth_compression_type=str(getattr(sensor, "depth_compression_type")),
                depth_scale=float(getattr(sensor, "depth_shift", 1000.0)),
                max_frames=int(max_frames),
                candidate_stride=int(candidate_stride),
                voxel_size_m=float(coverage_voxel_size_m),
                depth_stride=int(coverage_depth_stride),
            )
        else:
            selected = select_pose_diverse_frame_indices(
                candidate_poses,
                max_frames=int(max_frames),
                candidate_stride=int(candidate_stride),
                orientation_weight=float(orientation_weight),
            )
            selection_order = list(selected)
        if not selected:
            raise ValueError(f"{scene}: full .sens contains no valid selected RGB-D poses")
        if set(selected) & excluded:
            raise AssertionError("full-observation field selected a withheld PFPR query frame")

        for name in ("color", "depth", "pose", "intrinsic"):
            (scene_root / name).mkdir(parents=True, exist_ok=True)
        color_size: tuple[int, int] | None = None
        raw_color_size: tuple[int, int] | None = None
        target_color_size = (
            int(getattr(sensor, "depth_width")),
            int(getattr(sensor, "depth_height")),
        )
        for index in selected:
            frame = frames[index]
            stem = f"{index:06d}"
            color = frame.decompress_color(getattr(sensor, "color_compression_type"))
            if not isinstance(color, Image.Image):
                color = Image.fromarray(np.asarray(color))
            raw_color_size = tuple(map(int, color.size))
            # Existing RADIO-GS ScanNet loaders pair color and depth pixels.
            # ``.sens`` color is normally higher resolution than depth, so
            # preserve the standard exported layout rather than emitting an
            # internally inconsistent RGB-D field source.
            if color.size != target_color_size:
                color = color.resize(target_color_size, Image.Resampling.BILINEAR)
            depth = np.asarray(
                frame.decompress_depth(getattr(sensor, "depth_compression_type")),
                dtype=np.uint16,
            ).reshape(
                int(getattr(sensor, "depth_height")),
                int(getattr(sensor, "depth_width")),
            )
            color_size = tuple(map(int, color.size))
            color.save(scene_root / "color" / f"{stem}.jpg", quality=95)
            Image.fromarray(depth).save(scene_root / "depth" / f"{stem}.png")
            _save_matrix(poses[index], scene_root / "pose" / f"{stem}.txt")

        camera_matrices = {
            "intrinsics_color.txt": getattr(sensor, "intrinsic_color"),
            "intrinsics_depth.txt": getattr(sensor, "intrinsic_depth"),
            "extrinsics_color.txt": getattr(sensor, "extrinsic_color"),
            "extrinsics_depth.txt": getattr(sensor, "extrinsic_depth"),
        }
        for name, matrix in camera_matrices.items():
            _save_matrix(matrix, scene_root / name)
            _save_matrix(matrix, scene_root / "intrinsic" / SCANNET_CAMERA_LAYOUT[name])

        excluded_digest = (
            query_frame_exclusion_digest(sorted(excluded)) if excluded else ""
        )
        source_policy = (
            "full_sens_greedy_depth_voxel_coverage_query_free_frames"
            if selection_policy == "depth_voxel_coverage"
            else "full_sens_pose_diverse_query_free_frames"
        )
        if excluded:
            source_policy = source_policy.replace(
                "_query_free_frames", "_excluding_pfpr_query_source_frames"
            )
        record = {
            "field_contract_version": field_contract_version,
            "scene_id": scene,
            "source_root": str(sens_path.parent.resolve()),
            "source_sens": str(sens_path.resolve()),
            "source_sens_sha256": sha256_file(sens_path),
            "output_root": str(scene_root),
            "materialization_mode": "decoded_sens",
            "source_policy": source_policy,
            "full_sens_frame_count": len(frames),
            "all_valid_pose_frame_count": len(valid_pose_indices),
            "candidate_stride": int(candidate_stride),
            "candidate_frame_count": len(
                [
                    index
                    for index in candidate_valid_pose_indices
                    if index % int(candidate_stride) == 0
                ]
            ),
            "max_field_frames": int(max_frames),
            "frame_selection_policy": selection_policy,
            "selection_order_frame_indices": selection_order,
            "pose_orientation_weight": float(orientation_weight),
            "coverage_voxel_size_m": (
                float(coverage_voxel_size_m)
                if selection_policy == "depth_voxel_coverage"
                else None
            ),
            "coverage_depth_stride": (
                int(coverage_depth_stride)
                if selection_policy == "depth_voxel_coverage"
                else None
            ),
            "coverage_voxel_count": coverage_voxel_count,
            "selected_frame_indices": selected,
            "field_frame_count": len(selected),
            "field_frame_manifest_sha256": canonical_json_sha256(selected),
            "excluded_query_source_frame_count": len(excluded),
            "excluded_query_source_frame_ids_sha256": excluded_digest,
            "invalid_or_nonfinite_pose_frame_count": len(frames) - len(valid_pose_indices),
            "color_size": list(color_size or ()),
            "source_color_size": list(raw_color_size or ()),
            "uses_private_anchor": False,
            "uses_private_depth_pixel": False,
            "uses_instances_or_semantic_labels": False,
            "contains_instance_or_label_directories": False,
        }
        contract_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        reports.append(record)

    report = {
        "field_contract_version": field_contract_version,
        "sens_root": str(source_root),
        "output_root": str(destination_root),
        "scene_count": len(reports),
        "scenes": reports,
        "source_policy": (
            reports[0]["source_policy"] if reports else ""
        ),
        "uses_private_anchor": False,
        "uses_private_depth_pixel": False,
        "uses_instances_or_semantic_labels": False,
        "valid": bool(reports),
    }
    destination_root.mkdir(parents=True, exist_ok=True)
    (destination_root / "materialization_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def materialize_scannet_full_observation_pfpr_field_contract(
    method_manifest_path: str | Path,
    evaluator_manifest_path: str | Path,
    sens_root: str | Path,
    output_root: str | Path,
    *,
    scenes: Iterable[str] = (),
    max_frames: int = 240,
    candidate_stride: int = 1,
    orientation_weight: float = 0.25,
    selection_policy: str = "depth_voxel_coverage",
    coverage_voxel_size_m: float = 0.05,
    coverage_depth_stride: int = 8,
    sensor_factory: Callable[[Path], object] = SensorData,
    force: bool = False,
) -> dict:
    """Build a full-.sens PFPR field while withholding only query source frames.

    The private evaluator manifest is opened solely to obtain its
    ``source_frame_id`` values through :func:`_query_source_frames`.  Anchor
    coordinates, depth pixels, masks, labels, and instance IDs are neither
    inspected nor written.  The returned field source retains the full sensor
    digest and a digest of the withheld frame-ID set, never the IDs themselves.
    """

    method_path = Path(method_manifest_path).resolve()
    evaluator_path = Path(evaluator_manifest_path).resolve()
    method_manifest = json.loads(method_path.read_text(encoding="utf-8"))
    evaluator_manifest = json.loads(evaluator_path.read_text(encoding="utf-8"))
    _assert_manifest_versions(method_manifest, evaluator_manifest)
    source_frames = _query_source_frames(method_manifest, evaluator_manifest)
    _validate_public_exclusion_commitments(evaluator_manifest, source_frames)
    requested = {str(scene) for scene in scenes}
    if requested:
        missing = sorted(requested - set(source_frames))
        if missing:
            raise KeyError(f"requested PFPR scenes absent from the manifest: {missing}")
        source_frames = {
            scene: source_frames[scene] for scene in sorted(requested)
        }
    else:
        requested = set(source_frames)
    excluded_indices = {
        scene: {int(frame) for frame in frames}
        for scene, frames in source_frames.items()
    }
    report = materialize_scannet_full_observation_field_contract(
        sens_root,
        output_root,
        scenes=sorted(requested),
        max_frames=int(max_frames),
        candidate_stride=int(candidate_stride),
        orientation_weight=float(orientation_weight),
        selection_policy=str(selection_policy),
        coverage_voxel_size_m=float(coverage_voxel_size_m),
        coverage_depth_stride=int(coverage_depth_stride),
        sensor_factory=sensor_factory,
        force=bool(force),
        excluded_query_frame_indices=excluded_indices,
        field_contract_version=(
            SCANNET_FULL_OBSERVATION_PFPR_FIELD_CONTRACT_VERSION
        ),
    )
    # These hashes make the source split auditable without copying any
    # evaluator-only coordinate into a trainable field directory.
    report.update(
        {
            "benchmark_version": str(method_manifest["benchmark_version"]),
            "method_manifest": str(method_path),
            "method_manifest_sha256": sha256_file(method_path),
            "private_evaluator_manifest_sha256": sha256_file(evaluator_path),
            "query_source_frame_exclusions_only": True,
        }
    )
    output = Path(output_root).resolve() / "materialization_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-manifest", default="")
    parser.add_argument("--evaluator-manifest", default="")
    parser.add_argument(
        "--dense-root",
        default="",
        help="registered RGB-D frame root for PFPR or the dense AGILE pilot",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--scenes", nargs="*", default=())
    parser.add_argument(
        "--scenes-file",
        default="",
        help="optional newline-delimited scene IDs; blank lines and # comments are ignored",
    )
    parser.add_argument(
        "--all-dense-agile-observations",
        action="store_true",
        help="materialize all valid RGB-D frames for an AGILE dense-view pilot",
    )
    parser.add_argument(
        "--full-scannet-observations",
        action="store_true",
        help="decode query-free RGB-D fields from full ScanNet .sens streams",
    )
    parser.add_argument(
        "--full-scannet-pfpr-queryheldout-observations",
        action="store_true",
        help=(
            "decode full ScanNet .sens fields while withholding only each "
            "PFPR query source frame"
        ),
    )
    parser.add_argument("--sens-root", default="")
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--candidate-stride", type=int, default=1)
    parser.add_argument("--pose-orientation-weight", type=float, default=0.25)
    parser.add_argument(
        "--frame-selection-policy",
        choices=("depth_voxel_coverage", "pose_diverse"),
        default="depth_voxel_coverage",
        help="query-free full-.sens view selector; coverage is the formal default",
    )
    parser.add_argument("--coverage-voxel-size-m", type=float, default=0.05)
    parser.add_argument("--coverage-depth-stride", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    scenes = _cli_scene_ids(args.scenes, args.scenes_file)
    if args.all_dense_agile_observations:
        if not str(args.dense_root).strip():
            parser.error("--dense-root is required with --all-dense-agile-observations")
        print(
            json.dumps(
                materialize_agile_dense_field_contract(
                    args.dense_root,
                    args.output_root,
                    mode=args.mode,
                    scenes=scenes,
                ),
                indent=2,
            )
        )
        return
    if args.full_scannet_observations:
        if not str(args.sens_root).strip():
            parser.error("--sens-root is required with --full-scannet-observations")
        print(
            json.dumps(
                materialize_scannet_full_observation_field_contract(
                    args.sens_root,
                    args.output_root,
                    scenes=scenes,
                    max_frames=args.max_frames,
                    candidate_stride=args.candidate_stride,
                    orientation_weight=args.pose_orientation_weight,
                    selection_policy=args.frame_selection_policy,
                    coverage_voxel_size_m=args.coverage_voxel_size_m,
                    coverage_depth_stride=args.coverage_depth_stride,
                    force=args.force,
                ),
                indent=2,
            )
        )
        return
    if args.full_scannet_pfpr_queryheldout_observations:
        if not str(args.sens_root).strip():
            parser.error(
                "--sens-root is required with "
                "--full-scannet-pfpr-queryheldout-observations"
            )
        if not str(args.method_manifest).strip() or not str(args.evaluator_manifest).strip():
            parser.error(
                "--method-manifest and --evaluator-manifest are required with "
                "--full-scannet-pfpr-queryheldout-observations"
            )
        print(
            json.dumps(
                materialize_scannet_full_observation_pfpr_field_contract(
                    args.method_manifest,
                    args.evaluator_manifest,
                    args.sens_root,
                    args.output_root,
                    scenes=scenes,
                    max_frames=args.max_frames,
                    candidate_stride=args.candidate_stride,
                    orientation_weight=args.pose_orientation_weight,
                    selection_policy=args.frame_selection_policy,
                    coverage_voxel_size_m=args.coverage_voxel_size_m,
                    coverage_depth_stride=args.coverage_depth_stride,
                    force=args.force,
                ),
                indent=2,
            )
        )
        return
    if not str(args.method_manifest).strip() or not str(args.evaluator_manifest).strip():
        parser.error(
            "--method-manifest and --evaluator-manifest are required unless "
            "--all-dense-agile-observations is selected"
        )
    if not str(args.dense_root).strip():
        parser.error("--dense-root is required for PFPR field materialization")
    print(
        json.dumps(
            materialize_pfpr_field_contract(
                args.method_manifest,
                args.evaluator_manifest,
            args.dense_root,
            args.output_root,
            mode=args.mode,
            scenes=scenes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
