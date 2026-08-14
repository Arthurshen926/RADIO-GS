#!/usr/bin/env python3
"""Materialize dense evaluator-only Query Camera candidates from ScanNet assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from radio_gs.benchmarks.scannet_pfir.build_benchmark import find_scene_annotations
from radio_gs.benchmarks.scannet_pfir.preparation.prepare_full_scene import (
    load_raw_to_nyu40,
)
from radio_gs.benchmarks.scannet_pfir.protocol import load_mesh_instances
from radio_gs.scripts.prepare_scannet_scene import SensorData

from .protocol import BENCHMARK_VERSION, canonical_json_sha256, sha256_file


def prepare_geometry_query_frames(
    *,
    scene_id: str,
    sens_path: str | Path,
    annotation_roots: list[str | Path],
    label_map_path: str | Path,
    output_root: str | Path,
    frame_stride: int = 20,
    maximum_mesh_distance_m: float = 0.08,
) -> dict:
    """Project official mesh instances into a dense RGB-D camera subset.

    These projections are evaluator construction artifacts.  They are never
    authorized Mapping Observations or method-visible query inputs.
    """

    if frame_stride <= 0 or maximum_mesh_distance_m <= 0:
        raise ValueError("projection stride/distance must be positive")
    sens = Path(sens_path).resolve()
    label_map = Path(label_map_path).resolve()
    mesh, aggregation, segmentation = find_scene_annotations(scene_id, annotation_roots)
    mesh_xyz, mesh_ids, metadata = load_mesh_instances(mesh, aggregation, segmentation)
    raw_to_nyu40 = load_raw_to_nyu40(label_map)
    # ScanNet label TSV exposes raw category string -> NYU40 id in addition to
    # the numeric raw-id mapping used by PFIR preparation.
    import csv
    with label_map.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    name_to_nyu40 = {
        str(row.get("raw_category", "")).strip().lower(): int(row["nyu40id"])
        for row in rows
        if str(row.get("raw_category", "")).strip()
        and str(row.get("nyu40id", "")).strip()
    }
    if not raw_to_nyu40 or not name_to_nyu40:
        raise ValueError("ScanNet label map lacks raw/NYU40 bindings")
    instance_to_nyu40 = {
        int(instance_id): int(name_to_nyu40.get(str(meta["label"]).strip().lower(), 0))
        for instance_id, meta in metadata.items()
    }

    output = Path(output_root).resolve() / scene_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    for name in ("color", "depth", "pose", "instance", "label"):
        (output / name).mkdir(parents=True, exist_ok=True)
    sensor = SensorData(sens)
    tree = cKDTree(mesh_xyz)
    kd = sensor.intrinsic_depth[:3, :3].astype(np.float64)
    exported = []
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    for index in range(0, len(sensor.frames), frame_stride):
        frame = sensor.frames[index]
        pose = np.asarray(frame.camera_to_world, dtype=np.float64)
        if not np.isfinite(pose).all():
            continue
        depth = frame.decompress_depth(sensor.depth_compression_type).reshape(
            sensor.depth_height, sensor.depth_width
        )
        yy, xx = np.mgrid[0:sensor.depth_height, 0:sensor.depth_width]
        z = depth.astype(np.float64) / float(sensor.depth_shift)
        valid = np.isfinite(z) & (z > 0.20) & (z < 8.0)
        x = (xx - kd[0, 2]) * z / kd[0, 0]
        y = (yy - kd[1, 2]) * z / kd[1, 1]
        camera = np.stack(
            [x[valid], y[valid], z[valid], np.ones(int(valid.sum()))], axis=1
        )
        world = (camera @ pose.T)[:, :3]
        distance, nearest = tree.query(world, k=1)
        accepted = np.isfinite(distance) & (distance <= maximum_mesh_distance_m)
        encoded = np.zeros(depth.shape, dtype=np.uint16)
        flat_valid = np.flatnonzero(valid)
        encoded.reshape(-1)[flat_valid[accepted]] = mesh_ids[nearest[accepted]].astype(np.uint16)
        semantic = np.zeros(depth.shape, dtype=np.uint16)
        for instance_id in np.unique(encoded):
            if instance_id:
                semantic[encoded == instance_id] = instance_to_nyu40.get(int(instance_id), 0)
        stem = f"{index:06d}"
        color = frame.decompress_color(sensor.color_compression_type).resize(
            (sensor.depth_width, sensor.depth_height), resampling
        )
        color.save(output / "color" / f"{stem}.jpg", quality=95)
        Image.fromarray(depth).save(output / "depth" / f"{stem}.png")
        Image.fromarray(encoded).save(output / "instance" / f"{stem}.png")
        Image.fromarray(semantic).save(output / "label" / f"{stem}.png")
        np.savetxt(output / "pose" / f"{stem}.txt", pose, fmt="%.8f")
        exported.append(index)
    np.savetxt(output / "intrinsics_depth.txt", sensor.intrinsic_depth, fmt="%.8f")
    # RGB was explicitly resized onto the depth raster.
    np.savetxt(output / "intrinsics_color.txt", sensor.intrinsic_depth, fmt="%.8f")
    body = {
        "schema_version": "scannet_uqis_geometry_query_frames_v1",
        "benchmark_version": BENCHMARK_VERSION,
        "scene_id": scene_id,
        "visibility": "evaluator_construction_only",
        "authorized_mapping_observations": False,
        "method_visible": False,
        "sources": {
            "sens": {"path": str(sens), "sha256": sha256_file(sens)},
            "mesh": {"path": str(mesh.resolve()), "sha256": sha256_file(mesh)},
            "aggregation": {"path": str(aggregation.resolve()), "sha256": sha256_file(aggregation)},
            "segmentation": {"path": str(segmentation.resolve()), "sha256": sha256_file(segmentation)},
            "label_map": {"path": str(label_map), "sha256": sha256_file(label_map)},
        },
        "frame_stride": int(frame_stride),
        "maximum_mesh_distance_m": float(maximum_mesh_distance_m),
        "total_sensor_frames": len(sensor.frames),
        "exported_frame_ids": [f"{value:06d}" for value in exported],
        "raster_size": [sensor.depth_width, sensor.depth_height],
    }
    manifest = {**body, "receipt_sha256": canonical_json_sha256(body)}
    (output / "query_frame_derivation_receipt.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--sens", dest="sens_path", required=True)
    parser.add_argument(
        "--annotation-root", dest="annotation_roots", action="append", required=True
    )
    parser.add_argument("--label-map", dest="label_map_path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frame-stride", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(prepare_geometry_query_frames(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
