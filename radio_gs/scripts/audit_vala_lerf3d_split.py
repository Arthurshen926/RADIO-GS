#!/usr/bin/env python3
"""Fail closed when a VALA LERF-3D test split would leak label frames.

VALA's COLMAP reader strips the image extension before comparing camera names
with ``sparse/0/test.txt``.  A list copied from OccamLGS or another 3DGS fork
may contain ``.jpg`` and silently match zero cameras.  This audit freezes the
four released LERF-OVS annotation-frame sets in the exact stem-only form VALA
expects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES


def validate_vala_test_file(path: Path, scene: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    entries = [entry for entry in entries if entry]
    if len(entries) != len(set(entries)):
        raise ValueError(f"{path}: duplicate test-frame entries")
    suffixed = [entry for entry in entries if Path(entry).suffix]
    if suffixed:
        raise ValueError(
            f"{path}: VALA compares extensionless camera stems; remove suffixes from "
            f"{suffixed}"
        )
    expected = list(SCENE_GT_FRAMES[scene])
    missing = sorted(set(expected) - set(entries))
    extra = sorted(set(entries) - set(expected))
    if missing or extra:
        raise ValueError(f"{path}: split mismatch; missing={missing}, extra={extra}")
    return {
        "scene": scene,
        "test_file": str(path.resolve()),
        "test_frames": entries,
        "test_frame_count": len(entries),
        "status": "exact_extensionless_vala_holdout",
    }


def audit_dataset(dataset_root: Path, scenes: Sequence[str]) -> dict[str, object]:
    reports = {
        scene: validate_vala_test_file(
            dataset_root / scene / "sparse" / "0" / "test.txt", scene
        )
        for scene in scenes
    }
    return {
        "dataset_root": str(dataset_root.resolve()),
        "reader_contract": (
            "VALA scene.dataset_readers.readColmapCameras strips the image "
            "extension before test.txt membership"
        ),
        "scenes": reports,
        "status": "pass",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=sorted(SCENE_GT_FRAMES),
        default=sorted(SCENE_GT_FRAMES),
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = audit_dataset(args.dataset_root, args.scenes)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
