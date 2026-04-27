#!/usr/bin/env python3
"""Prepare a ScanNet scene for RADIO-GS training.

This script converts a downloaded ScanNet scan into the directory layout expected
by the existing ScanNet dataset loader:

  scene_root/
    color/
    depth/
    pose/
    intrinsic/
    label-filt/

It exports RGB, depth, pose, and intrinsics from ``.sens`` and extracts the
filtered 2D semantic labels from ``*_2d-label-filt.zip`` when available.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import struct
import tempfile
import zipfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image


COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


class RGBDFrame:
    def __init__(self) -> None:
        self.camera_to_world: np.ndarray | None = None
        self.color_data: bytes = b""
        self.depth_data: bytes = b""

    def load(self, file_handle) -> None:
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        file_handle.read(8)  # timestamp_color
        file_handle.read(8)  # timestamp_depth
        color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = file_handle.read(color_size_bytes)
        self.depth_data = file_handle.read(depth_size_bytes)

    def decompress_depth(self, compression_type: str) -> np.ndarray:
        if compression_type == "zlib_ushort":
            raw = zlib.decompress(self.depth_data)
            return np.frombuffer(raw, dtype=np.uint16)
        if compression_type == "raw_ushort":
            return np.frombuffer(self.depth_data, dtype=np.uint16)
        raise ValueError(f"Unsupported depth compression type: {compression_type}")

    def decompress_color(self, compression_type: str) -> Image.Image:
        if compression_type == "jpeg" or compression_type == "png":
            return Image.open(io.BytesIO(self.color_data)).convert("RGB")
        if compression_type == "raw":
            raise ValueError("Raw ScanNet color streams are not supported by this script")
        raise ValueError(f"Unsupported color compression type: {compression_type}")


class SensorData:
    def __init__(self, filename: Path) -> None:
        self.version = 4
        self.load(filename)

    def load(self, filename: Path) -> None:
        with filename.open("rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            if version != self.version:
                raise ValueError(f"Unexpected .sens version {version}, expected {self.version}")
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = f.read(strlen).decode("utf-8", errors="ignore")
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack("i", f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack("i", f.read(4))[0]]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames: list[RGBDFrame] = []
            for _ in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)


def save_matrix(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt="%.8f")


def iter_frame_indices(num_frames: int, frame_skip: int, max_frames: int | None) -> list[int]:
    indices = list(range(0, num_frames, max(1, frame_skip)))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def export_sens(scene_root: Path, sens_path: Path, frame_skip: int, max_frames: int | None, force: bool) -> dict[str, object]:
    sensor_data = SensorData(sens_path)
    color_dir = scene_root / "color"
    depth_dir = scene_root / "depth"
    pose_dir = scene_root / "pose"
    intrinsic_dir = scene_root / "intrinsic"

    frame_indices = iter_frame_indices(len(sensor_data.frames), frame_skip, max_frames)
    target_size = (sensor_data.depth_width, sensor_data.depth_height)

    if force:
        for directory in (color_dir, depth_dir, pose_dir, intrinsic_dir):
            if directory.exists():
                shutil.rmtree(directory)

    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in frame_indices:
        frame = sensor_data.frames[frame_idx]
        color_path = color_dir / f"{frame_idx}.jpg"
        depth_path = depth_dir / f"{frame_idx}.png"
        pose_path = pose_dir / f"{frame_idx}.txt"

        if force or not color_path.exists():
            color = frame.decompress_color(sensor_data.color_compression_type)
            if color.size != target_size:
                color = color.resize(target_size, Image.Resampling.BILINEAR)
            color.save(color_path, quality=95)

        if force or not depth_path.exists():
            depth = frame.decompress_depth(sensor_data.depth_compression_type).reshape(
                sensor_data.depth_height, sensor_data.depth_width
            )
            Image.fromarray(depth).save(depth_path)

        if force or not pose_path.exists():
            save_matrix(frame.camera_to_world, pose_path)

    save_matrix(sensor_data.intrinsic_color, intrinsic_dir / "intrinsic_color.txt")
    save_matrix(sensor_data.extrinsic_color, intrinsic_dir / "extrinsic_color.txt")
    save_matrix(sensor_data.intrinsic_depth, intrinsic_dir / "intrinsic_depth.txt")
    save_matrix(sensor_data.extrinsic_depth, intrinsic_dir / "extrinsic_depth.txt")

    return {
        "sensor_name": sensor_data.sensor_name,
        "num_frames_total": len(sensor_data.frames),
        "num_frames_exported": len(frame_indices),
        "frame_indices": frame_indices,
        "depth_width": sensor_data.depth_width,
        "depth_height": sensor_data.depth_height,
        "color_width_raw": sensor_data.color_width,
        "color_height_raw": sensor_data.color_height,
        "color_width_exported": sensor_data.depth_width,
        "color_height_exported": sensor_data.depth_height,
    }


def _locate_label_dir(root: Path) -> Path | None:
    direct = root / "label-filt"
    if direct.exists():
        return direct
    matches = [path for path in root.rglob("label-filt") if path.is_dir()]
    return matches[0] if matches else None


def extract_label_zip(scene_root: Path, label_zip: Path, force: bool) -> Path | None:
    if not label_zip.exists():
        return None

    label_dir = scene_root / "label-filt"
    if label_dir.exists() and not force:
        return label_dir
    if label_dir.exists() and force:
        shutil.rmtree(label_dir)

    with tempfile.TemporaryDirectory(dir=scene_root) as tmp_dir:
        tmp_root = Path(tmp_dir)
        with zipfile.ZipFile(label_zip, "r") as zf:
            zf.extractall(tmp_root)
        extracted = _locate_label_dir(tmp_root)
        if extracted is None:
            raise FileNotFoundError(f"Could not locate label-filt/ in extracted archive: {label_zip}")
        shutil.move(str(extracted), str(label_dir))
    return label_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a ScanNet scene for RADIO-GS")
    parser.add_argument("--scene_root", required=True, help="Scene directory, e.g. .../scans/scene0000_00")
    parser.add_argument("--sens_path", default=None, help="Optional explicit path to sceneXXXX_XX.sens")
    parser.add_argument("--label_zip", default=None, help="Optional explicit path to sceneXXXX_XX_2d-label-filt.zip")
    parser.add_argument("--frame_skip", type=int, default=10, help="Export every Nth frame from .sens")
    parser.add_argument("--max_frames", type=int, default=250, help="Maximum exported frames")
    parser.add_argument("--force", action="store_true", help="Overwrite exported color/depth/pose/labels")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).resolve()
    scene_root.mkdir(parents=True, exist_ok=True)
    scene_name = scene_root.name
    sens_path = Path(args.sens_path).resolve() if args.sens_path else scene_root / f"{scene_name}.sens"
    label_zip = Path(args.label_zip).resolve() if args.label_zip else scene_root / f"{scene_name}_2d-label-filt.zip"

    if not sens_path.exists():
        raise FileNotFoundError(f"Missing ScanNet .sens file: {sens_path}")

    manifest = export_sens(
        scene_root=scene_root,
        sens_path=sens_path,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        force=args.force,
    )
    label_dir = extract_label_zip(scene_root, label_zip, args.force)
    manifest["label_dir"] = str(label_dir) if label_dir is not None else None
    manifest["sens_path"] = str(sens_path)
    manifest["label_zip"] = str(label_zip) if label_zip.exists() else None

    manifest_path = scene_root / "radio_gs_scannet_prep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared ScanNet scene: {scene_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
