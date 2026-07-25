#!/usr/bin/env python3
"""Build versioned ScanNet-PFPR releases from held-out RGB-D frames.

The builder intentionally reuses only the *field-frame exclusions* from a
frozen PFIR manifest.  It never reads 2-D instances, semantic labels, or
aggregation data.  Annotation meshes supply public geometry only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

from radio_gs.benchmarks.scannet_pfir.build_benchmark import find_scene_annotations
from radio_gs.benchmarks.scannet_pfir.protocol import load_matrix, sha256_file

from .protocol import (
    BENCHMARK_VERSION,
    DEPTH_ALIGNED_QUERY_RASTER_V2,
    NATIVE_COLOR_QUERY_RASTER_V1,
    ProtocolConfig,
    canonical_json_sha256,
    method_query_record,
    protocol_record,
    query_frame_exclusion_digest,
    stable_voxel_domain,
    validate_benchmark_version,
    validate_release_config,
)


def _read_mesh_xyz(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"annotation mesh lacks xyz: {path}")
    return np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(
        np.float32
    )


def _frame_ids(scene_dir: Path) -> list[str]:
    return sorted(path.stem for path in (scene_dir / "color").glob("*.jpg"))


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _scene_field_frames(source_manifest: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    by_scene: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for query in source_manifest.get("queries", []):
        scene = str(query.get("scene_id", ""))
        frames = tuple(str(item) for item in query.get("field_frame_ids", ()))
        if not scene or not frames:
            raise ValueError("source PFIR public manifest lacks scene field-frame records")
        by_scene[scene].add(frames)
    resolved: dict[str, tuple[str, ...]] = {}
    for scene, values in by_scene.items():
        if len(values) != 1:
            raise ValueError(f"source PFIR field frames disagree within {scene}")
        resolved[scene] = next(iter(values))
    if not resolved:
        raise ValueError("source PFIR manifest contains no scenes")
    return resolved


def _assert_method_manifest_has_no_private_registration(
    method_manifest: dict[str, Any],
) -> None:
    """Reject raw evaluator registration while allowing v2's one-way digest.

    The source-frame exclusion digest is public provenance, not a source-frame
    identifier.  Check structured keys instead of a broad string search so a
    safe hash can coexist with a strict prohibition on query pose/depth/anchor
    data.
    """

    forbidden = {
        "anchor_world_xyz",
        "source_frame_id",
        "source_depth_pixel_uv",
        "pose_path",
        "depth_path",
    }
    for collection_name in ("queries", "scene_domains"):
        rows = method_manifest.get(collection_name, [])
        if not isinstance(rows, list):
            raise AssertionError(f"PFPR method manifest {collection_name} is invalid")
        for row in rows:
            if not isinstance(row, dict):
                raise AssertionError(f"PFPR method manifest {collection_name} is invalid")
            if forbidden & set(row):
                raise AssertionError("PFPR method manifest leaks evaluator-private registration")


def _depth_world_point(
    depth_intrinsic: np.ndarray,
    pose_c2w: np.ndarray,
    *,
    u: int,
    v: int,
    depth_m: float,
) -> np.ndarray:
    camera = np.asarray(
        [
            (float(u) - depth_intrinsic[0, 2]) * depth_m / depth_intrinsic[0, 0],
            (float(v) - depth_intrinsic[1, 2]) * depth_m / depth_intrinsic[1, 1],
            depth_m,
            1.0,
        ],
        dtype=np.float64,
    )
    return np.asarray((camera @ pose_c2w.T)[:3], dtype=np.float32)


def _color_pixel_from_depth(
    depth_intrinsic: np.ndarray,
    color_intrinsic: np.ndarray,
    *,
    u: int,
    v: int,
    depth_m: float,
) -> tuple[float, float]:
    x = (float(u) - depth_intrinsic[0, 2]) * depth_m / depth_intrinsic[0, 0]
    y = (float(v) - depth_intrinsic[1, 2]) * depth_m / depth_intrinsic[1, 1]
    return (
        float(color_intrinsic[0, 0] * x / depth_m + color_intrinsic[0, 2]),
        float(color_intrinsic[1, 1] * y / depth_m + color_intrinsic[1, 2]),
    )


def query_raster_geometry(
    *,
    depth_u: int,
    depth_v: int,
    depth_m: float,
    depth_intrinsic: np.ndarray,
    color_intrinsic: np.ndarray,
    depth_size: tuple[int, int],
    color_size: tuple[int, int],
    query_raster_contract: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Map a depth anchor to the public PFPR crop raster.

    The canonical field is constructed on ScanNet's depth-aligned RGB-D
    raster.  V2 therefore exposes a patch from that same raster, with the
    center remaining the exact depth pixel used for the evaluator-private
    anchor.  V1's native-color mapping is retained solely to read/reproduce
    its frozen query release.
    """

    contract = str(query_raster_contract)
    depth_width, depth_height = (int(depth_size[0]), int(depth_size[1]))
    color_width, color_height = (int(color_size[0]), int(color_size[1]))
    if min(depth_width, depth_height, color_width, color_height) <= 0:
        raise ValueError("PFPR query raster dimensions must be positive")
    if not (0 <= int(depth_u) < depth_width and 0 <= int(depth_v) < depth_height):
        raise ValueError("PFPR depth anchor lies outside the depth raster")
    if contract == DEPTH_ALIGNED_QUERY_RASTER_V2:
        return (int(depth_u), int(depth_v)), (depth_width, depth_height)
    if contract != NATIVE_COLOR_QUERY_RASTER_V1:
        raise ValueError(f"unsupported PFPR query raster contract: {contract!r}")
    color_u, color_v = _color_pixel_from_depth(
        depth_intrinsic,
        color_intrinsic,
        u=int(depth_u),
        v=int(depth_v),
        depth_m=float(depth_m),
    )
    return (
        (int(np.rint(color_u)), int(np.rint(color_v))),
        (color_width, color_height),
    )


def _load_query_raster(
    color_path: Path,
    *,
    target_size: tuple[int, int],
    query_raster_contract: str,
) -> Image.Image:
    """Load one method-visible source raster under the frozen crop contract."""

    with Image.open(color_path) as source:
        image = source.convert("RGB")
    if str(query_raster_contract) == DEPTH_ALIGNED_QUERY_RASTER_V2:
        if image.size != tuple(map(int, target_size)):
            image = image.resize(tuple(map(int, target_size)), Image.Resampling.BILINEAR)
    elif str(query_raster_contract) != NATIVE_COLOR_QUERY_RASTER_V1:
        raise ValueError("unsupported PFPR query raster contract")
    return image


def _valid_depth_window(depth_m: np.ndarray, u: int, v: int, config: ProtocolConfig) -> bool:
    half = config.depth_window_size_px // 2
    if u < half or v < half or u + half >= depth_m.shape[1] or v + half >= depth_m.shape[0]:
        return False
    window = depth_m[v - half : v + half + 1, u - half : u + half + 1]
    valid = (
        np.isfinite(window)
        & (window >= float(config.minimum_depth_m))
        & (window <= float(config.maximum_depth_m))
    )
    return bool(valid.mean() >= float(config.minimum_window_valid_fraction))


def _eligible_anchors(
    scene_dir: Path,
    held_out_frame_ids: Iterable[str],
    candidate_tree: cKDTree,
    *,
    depth_intrinsic: np.ndarray,
    color_intrinsic: np.ndarray,
    config: ProtocolConfig,
) -> list[dict[str, Any]]:
    """Enumerate geometry-valid anchors before deterministic sampling."""

    candidates: list[dict[str, Any]] = []
    half_patch = config.patch_size_px // 2
    for frame_id in sorted(set(str(value) for value in held_out_frame_ids)):
        color_path = scene_dir / "color" / f"{frame_id}.jpg"
        depth_path = scene_dir / "depth" / f"{frame_id}.png"
        pose_path = scene_dir / "pose" / f"{frame_id}.txt"
        if not (color_path.is_file() and depth_path.is_file() and pose_path.is_file()):
            continue
        try:
            pose = load_matrix(pose_path)
        except ValueError:
            continue
        with Image.open(color_path) as color:
            color_width, color_height = color.size
        depth = np.asarray(Image.open(depth_path), dtype=np.float32) / 1000.0
        valid_depth = (
            np.isfinite(depth)
            & (depth >= float(config.minimum_depth_m))
            & (depth <= float(config.maximum_depth_m))
        )
        window_fraction = uniform_filter(
            valid_depth.astype(np.float32),
            size=int(config.depth_window_size_px),
            mode="constant",
            cval=0.0,
        )
        half_window = config.depth_window_size_px // 2
        yy, xx = np.mgrid[
            half_window : depth.shape[0] - half_window : int(config.depth_grid_stride_px),
            half_window : depth.shape[1] - half_window : int(config.depth_grid_stride_px),
        ]
        yy, xx = yy.reshape(-1), xx.reshape(-1)
        keep = valid_depth[yy, xx] & (
            window_fraction[yy, xx] >= float(config.minimum_window_valid_fraction)
        )
        yy, xx = yy[keep], xx[keep]
        if not len(xx):
            continue
        z = depth[yy, xx]
        camera_x = (xx - depth_intrinsic[0, 2]) * z / depth_intrinsic[0, 0]
        camera_y = (yy - depth_intrinsic[1, 2]) * z / depth_intrinsic[1, 1]
        if config.query_raster_contract == DEPTH_ALIGNED_QUERY_RASTER_V2:
            crop_u, crop_v = xx.astype(np.int64), yy.astype(np.int64)
            crop_width, crop_height = int(depth.shape[1]), int(depth.shape[0])
        else:
            color_u = color_intrinsic[0, 0] * camera_x / z + color_intrinsic[0, 2]
            color_v = color_intrinsic[1, 1] * camera_y / z + color_intrinsic[1, 2]
            crop_u, crop_v = (
                np.rint(color_u).astype(np.int64),
                np.rint(color_v).astype(np.int64),
            )
            crop_width, crop_height = int(color_width), int(color_height)
        crop_keep = (
            (crop_u - half_patch >= 0)
            & (crop_v - half_patch >= 0)
            & (crop_u + half_patch <= crop_width)
            & (crop_v + half_patch <= crop_height)
        )
        yy, xx, z, camera_x, camera_y, crop_u, crop_v = (
            values[crop_keep]
            for values in (yy, xx, z, camera_x, camera_y, crop_u, crop_v)
        )
        if not len(xx):
            continue
        homogeneous = np.column_stack(
            [camera_x, camera_y, z, np.ones(len(z), dtype=np.float32)]
        )
        world = (homogeneous @ pose.T)[:, :3].astype(np.float32)
        distance, _nearest = candidate_tree.query(world, k=1)
        domain_keep = np.isfinite(distance) & (
            distance <= float(config.maximum_anchor_to_domain_distance_m)
        )
        for u, v, crop_x, crop_y, point, value in zip(
            xx[domain_keep],
            yy[domain_keep],
            crop_u[domain_keep],
            crop_v[domain_keep],
            world[domain_keep],
            np.asarray(distance)[domain_keep],
        ):
            candidates.append(
                {
                    "frame_id": frame_id,
                    "depth_u": int(u),
                    "depth_v": int(v),
                    "crop_u": int(crop_x),
                    "crop_v": int(crop_y),
                    "crop_raster_size": [int(crop_width), int(crop_height)],
                    "anchor_world_xyz": [float(item) for item in point],
                    "anchor_to_domain_distance_m": float(value),
                }
            )
    return candidates


def _deterministic_choice(
    scene_id: str,
    candidates: list[dict[str, Any]],
    count: int,
    *,
    benchmark_version: str,
) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise ValueError(f"{scene_id}: only {len(candidates)} valid held-out anchors for {count}")
    payload = f"{validate_benchmark_version(benchmark_version)}:{scene_id}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    chosen = sorted(int(index) for index in rng.choice(len(candidates), size=count, replace=False))
    return [candidates[index] for index in chosen]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    frames_root = Path(args.frames_root)
    annotations_roots = [Path(value) for value in args.annotations_root]
    source_manifest_path = Path(args.source_pfir_public_manifest)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    scene_field_frames = _scene_field_frames(source_manifest)
    benchmark_version = validate_benchmark_version(str(args.benchmark_version))
    config = ProtocolConfig(
        anchors_per_scene=int(args.anchors_per_scene),
        query_raster_contract=str(args.query_raster_contract),
    )
    validate_release_config(benchmark_version, config)
    output_dir = Path(args.output_dir)
    query_dir = output_dir / "queries" / "rgb"
    candidate_dir = output_dir / "candidates"
    query_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    method_queries: list[dict[str, Any]] = []
    evaluator_queries: list[dict[str, Any]] = []
    public_queries: list[dict[str, Any]] = []
    scene_domains: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for scene_id in sorted(scene_field_frames):
        scene_dir = frames_root / scene_id
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"PFPR frames unavailable: {scene_dir}")
        mesh_path, _aggregation_path, _segmentation_path = find_scene_annotations(
            scene_id, annotations_roots
        )
        candidate_xyz = stable_voxel_domain(
            _read_mesh_xyz(mesh_path), voxel_size_m=config.candidate_voxel_size_m
        )
        candidate_path = candidate_dir / f"{scene_id}.npy"
        np.save(candidate_path, candidate_xyz)
        depth_intrinsic = load_matrix(scene_dir / "intrinsics_depth.txt")
        color_intrinsic = load_matrix(scene_dir / "intrinsics_color.txt")
        all_ids = set(_frame_ids(scene_dir))
        field_ids = set(scene_field_frames[scene_id])
        held_out_ids = sorted(all_ids - field_ids)
        if not held_out_ids:
            raise ValueError(f"{scene_id}: no held-out RGB-D frames after field exclusion")
        anchors = _eligible_anchors(
            scene_dir,
            held_out_ids,
            cKDTree(candidate_xyz),
            depth_intrinsic=depth_intrinsic,
            color_intrinsic=color_intrinsic,
            config=config,
        )
        selected = _deterministic_choice(
            scene_id,
            anchors,
            config.anchors_per_scene,
            benchmark_version=benchmark_version,
        )
        for index, anchor in enumerate(selected):
            query_id = f"{scene_id}_pfpr_{index:03d}"
            color_path = scene_dir / "color" / f"{anchor['frame_id']}.jpg"
            crop_path = query_dir / f"{query_id}.png"
            image = _load_query_raster(
                color_path,
                target_size=tuple(int(value) for value in anchor["crop_raster_size"]),
                query_raster_contract=config.query_raster_contract,
            )
            try:
                half = config.patch_size_px // 2
                crop = image.crop(
                    (
                        int(anchor["crop_u"]) - half,
                        int(anchor["crop_v"]) - half,
                        int(anchor["crop_u"]) + half,
                        int(anchor["crop_v"]) + half,
                    )
                )
                if crop.size != (config.patch_size_px, config.patch_size_px):
                    raise AssertionError("PFPR crop dimensions disagree with frozen protocol")
                crop.save(crop_path)
            finally:
                image.close()
            crop_hash = _sha256(crop_path)
            method_queries.append(
                method_query_record(
                    query_id=query_id,
                    scene_id=scene_id,
                    crop_rgb_path=str(crop_path.resolve()),
                    crop_rgb_sha256=crop_hash,
                    benchmark_version=benchmark_version,
                )
            )
            evaluator_queries.append(
                {
                    "query_id": query_id,
                    "scene_id": scene_id,
                    "anchor_world_xyz": anchor["anchor_world_xyz"],
                    "source_frame_id": anchor["frame_id"],
                    "source_depth_pixel_uv": [anchor["depth_u"], anchor["depth_v"]],
                    "anchor_to_domain_distance_m": anchor["anchor_to_domain_distance_m"],
                }
            )
            public_queries.append(
                {
                    "query_id": query_id,
                    "scene_id": scene_id,
                    "crop_rgb_sha256": crop_hash,
                    "query_pose_used_by_method": False,
                    "query_depth_used_by_method": False,
                    "query_mask_used_by_method": False,
                }
            )
        scene_domains.append(
            {
                "scene_id": scene_id,
                "candidate_xyz_path": str(candidate_path.resolve()),
                "candidate_xyz_sha256": _sha256(candidate_path),
                "candidate_points": int(len(candidate_xyz)),
                "geometry_only": True,
                # A public one-way commitment lets a scorer reject a field
                # built with a different held-out source-frame split without
                # revealing the evaluator-only frame IDs or anchor pixels.
                "excluded_query_source_frame_ids_sha256": query_frame_exclusion_digest(
                    [str(anchor["frame_id"]) for anchor in selected]
                ),
            }
        )
        reports.append(
            {
                "scene_id": scene_id,
                "field_frame_count": len(field_ids),
                "held_out_frame_count": len(held_out_ids),
                "eligible_anchor_count": len(anchors),
                "selected_anchor_count": len(selected),
            }
        )

    common = protocol_record(config, benchmark_version=benchmark_version)
    common["source_pfir_public_manifest"] = str(source_manifest_path.resolve())
    common["source_pfir_public_manifest_sha256"] = _sha256(source_manifest_path)
    method_manifest = {**common, "queries": method_queries, "scene_domains": scene_domains}
    evaluator_manifest = {**common, "queries": evaluator_queries, "scene_domains": scene_domains}
    public_manifest = {**common, "queries": public_queries, "scene_domains": scene_domains, "scene_reports": reports}
    _assert_method_manifest_has_no_private_registration(method_manifest)
    _write_json(output_dir / "manifest.method.json", method_manifest)
    _write_json(output_dir / "manifest.evaluator.json", evaluator_manifest)
    _write_json(output_dir / "manifest.public.json", public_manifest)
    release = {
        "benchmark_version": benchmark_version,
        "method_manifest_sha256": _sha256(output_dir / "manifest.method.json"),
        "evaluator_manifest_sha256": _sha256(output_dir / "manifest.evaluator.json"),
        "public_manifest_sha256": _sha256(output_dir / "manifest.public.json"),
        "query_count": len(method_queries),
        "scene_count": len(scene_domains),
        "method_inputs": ["scene_id", "crop_rgb"],
        "instance_labels_used": False,
        "source_query_pose_used_by_method": False,
        "source_query_depth_used_by_method": False,
        "query_raster_contract": config.query_raster_contract,
    }
    _write_json(output_dir / "release.json", release)
    return {**release, "scene_reports": reports, "config": asdict(config)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--annotations-root", nargs="+", required=True)
    parser.add_argument("--source-pfir-public-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--anchors-per-scene", type=int, default=10)
    parser.add_argument(
        "--benchmark-version",
        choices=("scannet-pfpr-small-v1", "scannet-pfpr-small-v2"),
        default=BENCHMARK_VERSION,
    )
    parser.add_argument(
        "--query-raster-contract",
        choices=(NATIVE_COLOR_QUERY_RASTER_V1, DEPTH_ALIGNED_QUERY_RASTER_V2),
        default=DEPTH_ALIGNED_QUERY_RASTER_V2,
        help="v2 crops the depth-aligned RGB raster used by canonical-field construction",
    )
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
