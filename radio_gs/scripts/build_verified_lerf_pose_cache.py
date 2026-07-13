#!/usr/bin/env python3
"""Export frame-ID aligned COLMAP poses with an auditable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from radio_gs.data.lerf_dataset import LERFDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict:
    scene_root = Path(args.scene_root)
    feature_manifest_path = Path(args.feature_manifest)
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    dataset = LERFDataset(
        str(scene_root),
        str(Path(args.output_dir) / "__no_features__"),
        annotation_dir=None,
        allow_empty_features=True,
    )
    source_by_id = {}
    for source_path in dataset.file_paths:
        name = Path(source_path).name
        stem = Path(name).stem
        suffix = stem.split("_")[-1]
        if suffix.isdigit():
            source_by_id[int(suffix)] = name
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in feature_manifest["frames"]:
        frame_id = int(frame["frame_idx"])
        source_file = str(frame["source_file"])
        mapped_file = source_by_id.get(frame_id)
        if mapped_file != source_file:
            raise ValueError(
                f"frame {frame_id} feature/COLMAP source mismatch: "
                f"{source_file!r} vs {mapped_file!r}"
            )
        w2c = dataset.pose_by_frame_idx.get(frame_id)
        if w2c is None:
            raise KeyError(f"COLMAP has no pose for frame {frame_id}")
        c2w = np.linalg.inv(w2c).astype(np.float32)
        pose_path = output / f"{frame_id}.txt"
        np.savetxt(pose_path, c2w, fmt="%.9g")
        rows.append(
            {
                "frame_id": frame_id,
                "source_file": source_file,
                "pose_file": pose_path.name,
                "c2w_sha256": _sha256(pose_path),
            }
        )
    sparse = scene_root / "sparse" / "0"
    manifest = {
        "schema_version": 1,
        "scene_root": str(scene_root.resolve()),
        "feature_manifest": str(feature_manifest_path.resolve()),
        "pose_source": "COLMAP sparse/0, exact source filename mapping",
        "stored_convention": "camera_to_world; SimpleRadioDataset inverts to w2c",
        "images_bin_sha256": _sha256(sparse / "images.bin"),
        "cameras_bin_sha256": _sha256(sparse / "cameras.bin"),
        "points3d_bin_sha256": _sha256(sparse / "points3D.bin"),
        "num_frames": len(rows),
        "frames": rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = build(args)
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
