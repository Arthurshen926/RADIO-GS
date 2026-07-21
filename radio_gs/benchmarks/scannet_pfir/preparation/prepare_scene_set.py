#!/usr/bin/env python3
"""Prepare a fixed set of downloaded ScanNet scenes under one dense contract."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from radio_gs.benchmarks.scannet_pfir.preparation.prepare_full_scene import prepare


def _scene_names(raw: str) -> list[str]:
    path = Path(raw)
    if not any(character.isspace() for character in raw) and path.is_file():
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return [value for value in raw.replace(",", " ").split() if value]


def _prepare_job(job: dict) -> dict:
    return prepare(**job)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-names", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--frame-skip", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    raw_root = Path(args.raw_root)
    jobs = []
    for scene_id in _scene_names(args.scene_names):
        scene_root = raw_root / scene_id
        jobs.append(
            {
                "scene_id": scene_id,
                "sens_path": scene_root / f"{scene_id}.sens",
                "instance_zip_path": scene_root
                / f"{scene_id}_2d-instance-filt.zip",
                "label_zip_path": scene_root / f"{scene_id}_2d-label-filt.zip",
                "label_map_path": args.label_map,
                "output_root": args.output_root,
                "frame_skip": args.frame_skip,
                "force": args.force,
            }
        )
    if int(args.workers) > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            manifests = list(executor.map(_prepare_job, jobs))
    else:
        manifests = [_prepare_job(job) for job in jobs]
    report = {
        "scene_count": len(manifests),
        "frame_skip": int(args.frame_skip),
        "source_contract": "official_sens_plus_filtered_2d_projections",
        "scenes": manifests,
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "scene_count": len(manifests),
                "finite_pose_frame_count": sum(
                    row["finite_pose_frame_count"] for row in manifests
                ),
                "report": str(output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
