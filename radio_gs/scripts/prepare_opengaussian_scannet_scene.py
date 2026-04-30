#!/usr/bin/env python3
"""Prepare OpenGaussian ScanNet scene zips for RADIO-GS.

The OpenGaussian release packs each scene as ``sceneXXXX_YY.zip`` with
``transforms_train.json``, ``transforms_test.json``, ``points3d.ply``, a label
PLY, and either a nested ``color.zip`` or an extracted ``color/`` directory.
This script unpacks that bundle into a RADIO-GS friendly layout:

    output/scannet_og_prepared/{scene}/
        color/
        transforms_train.json
        transforms_test.json
        transforms.json
        traj_w_c.txt
        traj_w_c_train.txt
        traj_w_c_val.txt
        splits/train_frames.txt
        splits/val_frames.txt
        points3d.ply
        {scene}_vh_clean_2.labels.ply
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from radio_gs.data.benchmark_paths import extract_feature_frame_index
from radio_gs.data.lerf_dataset import _parse_transforms_json


DEFAULT_DATA_ROOT = Path("/mnt/pool/sqy/3d_understanding/scannet")
DEFAULT_OUTPUT_ROOT = Path("dataset/scannet_og")


@dataclass
class PreparedScene:
    scene: str
    scene_root: Path
    num_train_frames: int
    num_val_frames: int
    num_all_frames: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    points_ply: Path
    labels_ply: Path


def _scene_zip(data_root: Path, scene: str) -> Path:
    zip_path = data_root / f"{scene}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"OpenGaussian ScanNet zip not found: {zip_path}")
    return zip_path


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _copy_or_link_tree(src: Path, dst: Path, copy_mode: str, overwrite: bool) -> None:
    if dst.exists():
        if overwrite:
            _remove_path(dst)
        else:
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if overwrite:
            _remove_path(dst)
        else:
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _materialize_existing_scene(
    src_scene: Path,
    scene_root: Path,
    scene: str,
    copy_mode: str,
    overwrite: bool,
) -> Path:
    """Create a prepared scene from an already-extracted source directory.

    Transform JSONs are always copied, never symlinked, because preparation may
    filter invalid poses and write the filtered transforms into *scene_root*.
    Heavy immutable assets such as color images and PLYs can still be symlinked.
    """
    if scene_root.exists() and overwrite:
        _remove_path(scene_root)
    scene_root.mkdir(parents=True, exist_ok=True)

    for name in ("transforms_train.json", "transforms_test.json"):
        _copy_file(src_scene / name, scene_root / name, overwrite=True)

    for name in ("points3d.ply", "language_features.zip", "color.zip"):
        src = src_scene / name
        if src.exists():
            _copy_or_link_tree(src, scene_root / name, copy_mode=copy_mode, overwrite=overwrite)

    label_src = _find_label_ply(src_scene, scene)
    _copy_or_link_tree(
        label_src,
        scene_root / label_src.name,
        copy_mode=copy_mode,
        overwrite=overwrite,
    )

    color_src = src_scene / "color"
    if color_src.exists():
        _copy_or_link_tree(
            color_src,
            scene_root / "color",
            copy_mode=copy_mode,
            overwrite=overwrite,
        )
    return scene_root


def _extract_top_zip(zip_path: Path, output_root: Path, scene: str, overwrite: bool) -> Path:
    scene_root = output_root / scene
    if scene_root.exists() and not overwrite:
        return scene_root
    if scene_root.exists():
        shutil.rmtree(scene_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_root)
    if not scene_root.exists():
        raise FileNotFoundError(
            f"Expected extracted scene directory {scene_root} after unpacking {zip_path}"
        )
    return scene_root


def _ensure_color_dir(scene_root: Path, overwrite: bool = False) -> Path:
    color_dir = scene_root / "color"
    if color_dir.exists() and any(color_dir.iterdir()):
        return color_dir

    color_zip = scene_root / "color.zip"
    if not color_zip.exists():
        if color_dir.exists():
            return color_dir
        raise FileNotFoundError(f"Neither color/ nor color.zip found in {scene_root}")

    if color_dir.exists() and overwrite:
        _remove_path(color_dir)
    color_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(color_zip, "r") as zf:
        zf.extractall(color_dir)

    nested = color_dir / "color"
    if nested.exists() and nested.is_dir():
        for child in nested.iterdir():
            shutil.move(str(child), str(color_dir / child.name))
        nested.rmdir()
    return color_dir


def _frame_id_from_path(file_path: str, fallback: int) -> int:
    try:
        return extract_feature_frame_index(Path(file_path))
    except ValueError:
        return int(fallback)


def _load_transform_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _frame_has_finite_transform(frame: dict) -> bool:
    matrix = np.asarray(frame.get("transform_matrix", []), dtype=np.float32)
    return matrix.size in {12, 16} and np.isfinite(matrix).all()


def _filter_transforms(data: dict) -> tuple[dict, int]:
    frames = data.get("frames", [])
    kept = [frame for frame in frames if _frame_has_finite_transform(frame)]
    filtered = dict(data)
    filtered["frames"] = kept
    return filtered, len(frames) - len(kept)


def _frames_with_ids(transforms: dict) -> list[tuple[int, dict]]:
    frames = []
    for idx, frame in enumerate(transforms.get("frames", [])):
        frames.append((_frame_id_from_path(str(frame.get("file_path", "")), idx), frame))
    frames.sort(key=lambda item: item[0])
    return frames


def _write_frame_ids(path: Path, frame_ids: Iterable[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(fid)) for fid in frame_ids) + "\n", encoding="utf-8")


def _write_flat_poses(path: Path, c2ws: Iterable[np.ndarray]) -> None:
    arr = np.stack([np.asarray(c2w, dtype=np.float32) for c2w in c2ws], axis=0)
    np.savetxt(str(path), arr.reshape(arr.shape[0], 16), fmt="%.9g")


def _parsed_pose_by_frame(path: Path) -> dict[int, np.ndarray]:
    parsed = _parse_transforms_json(str(path))
    pose_by_id = {}
    for idx, (c2w, file_path) in enumerate(zip(parsed["c2w_list"], parsed["file_paths"])):
        frame_id = _frame_id_from_path(str(file_path), idx)
        pose_by_id[int(frame_id)] = np.asarray(c2w, dtype=np.float32)
    return pose_by_id


def _write_combined_transforms(
    scene_root: Path,
    train_data: dict,
    val_data: dict,
) -> list[int]:
    frame_by_id: dict[int, dict] = {}
    for source in (train_data, val_data):
        for frame_id, frame in _frames_with_ids(source):
            frame_by_id[int(frame_id)] = frame

    combined = {
        key: train_data[key]
        for key in ("camera_angle_x", "fl_x", "fl_y", "cx", "cy", "w", "h")
        if key in train_data
    }
    combined["frames"] = [frame_by_id[fid] for fid in sorted(frame_by_id)]
    (scene_root / "transforms.json").write_text(
        json.dumps(combined, indent=2),
        encoding="utf-8",
    )
    return sorted(frame_by_id)


def _find_label_ply(scene_root: Path, scene: str) -> Path:
    preferred = scene_root / f"{scene}_vh_clean_2.labels.ply"
    if preferred.exists():
        return preferred
    matches = sorted(scene_root.glob("*.labels.ply"))
    if not matches:
        raise FileNotFoundError(f"No *.labels.ply file found in {scene_root}")
    return matches[0]


def prepare_scene(
    scene: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    copy_mode: str = "symlink",
    overwrite: bool = False,
) -> PreparedScene:
    """Prepare one OpenGaussian ScanNet scene and return its manifest."""
    data_root = Path(data_root)
    output_root = Path(output_root)
    if copy_mode not in {"symlink", "copy"}:
        raise ValueError("--copy_mode must be 'symlink' or 'copy'")

    src_scene = data_root / scene
    if src_scene.exists() and src_scene.is_dir():
        scene_root = _materialize_existing_scene(
            src_scene=src_scene,
            scene_root=output_root / scene,
            scene=scene,
            copy_mode=copy_mode,
            overwrite=overwrite,
        )
    else:
        scene_root = _extract_top_zip(_scene_zip(data_root, scene), output_root, scene, overwrite)

    _ensure_color_dir(scene_root, overwrite=overwrite)

    train_path = scene_root / "transforms_train.json"
    val_path = scene_root / "transforms_test.json"
    points_ply = scene_root / "points3d.ply"
    labels_ply = _find_label_ply(scene_root, scene)
    for required in (train_path, val_path, points_ply, labels_ply):
        if not required.exists():
            raise FileNotFoundError(f"Required OpenGaussian file missing: {required}")

    train_data, dropped_train = _filter_transforms(_load_transform_json(train_path))
    val_data, dropped_val = _filter_transforms(_load_transform_json(val_path))
    if dropped_train or dropped_val:
        print(
            f"[{scene}] dropped non-finite pose frames: "
            f"train={dropped_train} val={dropped_val}"
        )
        train_path.write_text(json.dumps(train_data, indent=2), encoding="utf-8")
        val_path.write_text(json.dumps(val_data, indent=2), encoding="utf-8")
    train_ids = [fid for fid, _ in _frames_with_ids(train_data)]
    val_ids = [fid for fid, _ in _frames_with_ids(val_data)]
    all_ids = _write_combined_transforms(scene_root, train_data, val_data)

    train_pose_by_id = _parsed_pose_by_frame(train_path)
    val_pose_by_id = _parsed_pose_by_frame(val_path)
    all_pose_by_id = {**train_pose_by_id, **val_pose_by_id}
    _write_flat_poses(scene_root / "traj_w_c_train.txt", [train_pose_by_id[fid] for fid in train_ids])
    _write_flat_poses(scene_root / "traj_w_c_val.txt", [val_pose_by_id[fid] for fid in val_ids])
    _write_flat_poses(scene_root / "traj_w_c.txt", [all_pose_by_id[fid] for fid in all_ids])

    splits_dir = scene_root / "splits"
    _write_frame_ids(splits_dir / "train_frames.txt", train_ids)
    _write_frame_ids(splits_dir / "val_frames.txt", val_ids)
    _write_frame_ids(splits_dir / "all_frames.txt", all_ids)

    prepared = PreparedScene(
        scene=scene,
        scene_root=scene_root,
        num_train_frames=len(train_ids),
        num_val_frames=len(val_ids),
        num_all_frames=len(all_ids),
        width=int(train_data.get("w", val_data.get("w", 0))),
        height=int(train_data.get("h", val_data.get("h", 0))),
        fx=float(train_data.get("fl_x", val_data.get("fl_x", 0.0))),
        fy=float(train_data.get("fl_y", val_data.get("fl_y", train_data.get("fl_x", 0.0)))),
        cx=float(train_data.get("cx", val_data.get("cx", 0.0))),
        cy=float(train_data.get("cy", val_data.get("cy", 0.0))),
        points_ply=points_ply,
        labels_ply=labels_ply,
    )

    manifest = asdict(prepared)
    for key in ("scene_root", "points_ply", "labels_ply"):
        manifest[key] = str(manifest[key])
    (scene_root / "radio_gs_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare OpenGaussian ScanNet zips for RADIO-GS."
    )
    parser.add_argument("--scene", default="scene0000_00", help="Scene id or 'all'")
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--copy_mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.scene == "all":
        scenes = sorted(path.stem for path in data_root.glob("scene*.zip"))
    else:
        scenes = [args.scene]
    if not scenes:
        raise FileNotFoundError(f"No scene*.zip files found under {data_root}")

    for scene in scenes:
        prepared = prepare_scene(
            scene=scene,
            data_root=data_root,
            output_root=args.output_root,
            copy_mode=args.copy_mode,
            overwrite=args.overwrite,
        )
        print(
            f"[{scene}] prepared at {prepared.scene_root} | "
            f"train={prepared.num_train_frames} val={prepared.num_val_frames} "
            f"size={prepared.width}x{prepared.height}"
        )


if __name__ == "__main__":
    main()
