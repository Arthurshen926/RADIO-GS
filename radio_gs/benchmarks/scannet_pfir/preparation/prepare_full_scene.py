#!/usr/bin/env python3
"""Prepare dense, frame-aligned ScanNet assets for a formal PFIR scene."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np
from PIL import Image

from radio_gs.benchmarks.scannet_pfir.protocol import sha256_file
from radio_gs.scripts.prepare_scannet_scene import SensorData


def zip_frame_members(archive: zipfile.ZipFile) -> dict[int, str]:
    """Map the final numeric filename token to a unique archive member."""

    result: dict[int, str] = {}
    for member in archive.namelist():
        if member.endswith("/"):
            continue
        match = re.search(r"(\d+)(?=\.[^.]+$)", Path(member).name)
        if match is None:
            continue
        index = int(match.group(1))
        if index in result:
            raise ValueError(f"duplicate projected frame {index} in {archive.filename}")
        result[index] = member
    return result


def _save_matrix(matrix: np.ndarray, path: Path) -> None:
    np.savetxt(path, np.asarray(matrix), fmt="%.8f")


def load_raw_to_nyu40(path: str | Path) -> dict[int, int]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        mapping = {
            int(row["id"]): int(row["nyu40id"])
            for row in rows
            if row.get("id", "").strip() and row.get("nyu40id", "").strip()
        }
    if not mapping or any(value < 0 or value > 40 for value in mapping.values()):
        raise ValueError("invalid ScanNet raw-id -> NYU40 label mapping")
    return mapping


def remap_raw_labels(values: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    source = np.asarray(values, dtype=np.int64)
    maximum = max(max(mapping), int(source.max(initial=0)))
    lookup = np.zeros(maximum + 1, dtype=np.uint16)
    for raw_id, nyu40_id in mapping.items():
        lookup[int(raw_id)] = int(nyu40_id)
    valid = (source >= 0) & (source < lookup.size)
    output = np.zeros(source.shape, dtype=np.uint16)
    output[valid] = lookup[source[valid]]
    return output


def prepare(
    *,
    scene_id: str,
    sens_path: str | Path,
    instance_zip_path: str | Path,
    label_zip_path: str | Path,
    label_map_path: str | Path,
    output_root: str | Path,
    frame_skip: int = 20,
    force: bool = False,
) -> dict:
    sens_path = Path(sens_path)
    instance_zip_path = Path(instance_zip_path)
    label_zip_path = Path(label_zip_path)
    label_map_path = Path(label_map_path)
    for path in (sens_path, instance_zip_path, label_zip_path, label_map_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    scene_root = Path(output_root) / scene_id
    manifest_path = scene_root / "pfir_source_manifest.json"
    requested = {
        "scene_id": scene_id,
        "frame_skip": int(frame_skip),
        "sens_sha256": sha256_file(sens_path),
        "instance_zip_sha256": sha256_file(instance_zip_path),
        "label_zip_sha256": sha256_file(label_zip_path),
        "label_map_sha256": sha256_file(label_map_path),
    }
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in requested.items()):
            return existing
        raise RuntimeError(
            f"{scene_root} was prepared under a different source/stride; "
            "use a new output root or explicitly pass --force"
        )
    if force:
        for name in ("color", "depth", "pose", "instance", "label"):
            directory = scene_root / name
            if directory.exists():
                shutil.rmtree(directory)
    for name in ("color", "depth", "pose", "instance", "label"):
        (scene_root / name).mkdir(parents=True, exist_ok=True)

    sensor = SensorData(sens_path)
    raw_to_nyu40 = load_raw_to_nyu40(label_map_path)
    with zipfile.ZipFile(instance_zip_path) as instance_archive, zipfile.ZipFile(
        label_zip_path
    ) as label_archive:
        instance_members = zip_frame_members(instance_archive)
        label_members = zip_frame_members(label_archive)
        selected = [
            index
            for index in range(0, len(sensor.frames), max(1, int(frame_skip)))
            if index in instance_members and index in label_members
        ]
        if not selected:
            raise RuntimeError("no .sens/projection frame intersection")
        color_shape = None
        projection_shape = None
        for index in selected:
            stem = f"{index:06d}"
            frame = sensor.frames[index]
            color = frame.decompress_color(sensor.color_compression_type)
            depth = frame.decompress_depth(sensor.depth_compression_type).reshape(
                sensor.depth_height, sensor.depth_width
            )
            instance = Image.open(
                io.BytesIO(instance_archive.read(instance_members[index]))
            )
            raw_label = Image.open(io.BytesIO(label_archive.read(label_members[index])))
            if instance.size != raw_label.size:
                raise ValueError(f"{scene_id}/{stem}: instance/label sizes differ")
            if color.size != instance.size:
                raise ValueError(
                    f"{scene_id}/{stem}: RGB {color.size} and projections "
                    f"{instance.size} differ"
                )
            color_shape = color.size
            projection_shape = instance.size
            color.save(scene_root / "color" / f"{stem}.jpg", quality=95)
            Image.fromarray(depth).save(scene_root / "depth" / f"{stem}.png")
            instance.save(scene_root / "instance" / f"{stem}.png")
            nyu40_label = remap_raw_labels(np.asarray(raw_label), raw_to_nyu40)
            Image.fromarray(nyu40_label).save(
                scene_root / "label" / f"{stem}.png"
            )
            _save_matrix(frame.camera_to_world, scene_root / "pose" / f"{stem}.txt")

    _save_matrix(sensor.intrinsic_color, scene_root / "intrinsics_color.txt")
    _save_matrix(sensor.intrinsic_depth, scene_root / "intrinsics_depth.txt")
    _save_matrix(sensor.extrinsic_color, scene_root / "extrinsics_color.txt")
    _save_matrix(sensor.extrinsic_depth, scene_root / "extrinsics_depth.txt")
    finite = sum(
        np.isfinite(sensor.frames[index].camera_to_world).all() for index in selected
    )
    manifest = {
        **requested,
        "source_contract": "official_sens_plus_2d_instance_filt_plus_2d_label_filt",
        "semantic_label_space": "nyu40id",
        "total_sens_frames": len(sensor.frames),
        "projection_intersection_frames": len(
            set(instance_members).intersection(label_members)
        ),
        "exported_frame_count": len(selected),
        "finite_pose_frame_count": int(finite),
        "exported_frame_indices": selected,
        "color_size": list(color_shape or ()),
        "projection_size": list(projection_shape or ()),
        "depth_size": [sensor.depth_width, sensor.depth_height],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--sens", dest="sens_path", required=True)
    parser.add_argument("--instance-zip", dest="instance_zip_path", required=True)
    parser.add_argument("--label-zip", dest="label_zip_path", required=True)
    parser.add_argument("--label-map", dest="label_map_path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frame-skip", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
