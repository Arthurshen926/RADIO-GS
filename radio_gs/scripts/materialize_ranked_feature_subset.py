#!/usr/bin/env python3
"""Materialize a frame-ID feature subset with dense rank-aligned poses."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--pose-file", required=True)
    parser.add_argument("--frame-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    selected = [int(value) for value in args.frame_ids.replace(",", " ").split()]
    images = sorted(
        (path for path in Path(args.image_dir).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda path: int(path.stem),
    )
    rank_by_frame = {int(path.stem): rank for rank, path in enumerate(images)}
    pose_lines = Path(args.pose_file).read_text(encoding="utf-8").splitlines()
    if len(pose_lines) != len(images):
        raise ValueError("pose/image count mismatch")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mapping = []
    selected_pose_lines = []
    for dense_rank, frame_id in enumerate(selected):
        source_rank = rank_by_frame[frame_id]
        source = Path(args.feature_dir) / f"rgb_{frame_id}.pt"
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / f"rgb_{dense_rank}.pt"
        if destination.exists():
            destination.unlink()
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        selected_pose_lines.append(pose_lines[source_rank])
        mapping.append({"dense_rank": dense_rank, "frame_id": frame_id, "source_rank": source_rank})
    pose_output = output / "traj_w_c.txt"
    pose_output.write_text("\n".join(selected_pose_lines) + "\n", encoding="utf-8")
    (output / "rank_mapping.json").write_text(
        json.dumps({"schema_version": 1, "frames": mapping}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "num_frames": len(mapping)}, indent=2))


if __name__ == "__main__":
    main()
