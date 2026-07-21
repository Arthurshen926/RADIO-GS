#!/usr/bin/env python3
"""Materialize query-free PFIR reconstruction inputs from a frozen manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

from radio_gs.benchmarks.scannet_pfir.protocol import (
    canonical_json_sha256,
    sha256_file,
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


def _numeric_files(directory: Path, suffixes: Iterable[str]) -> dict[str, Path]:
    allowed = {value.lower() for value in suffixes}
    output: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        try:
            key = f"{int(path.stem):06d}"
        except ValueError:
            continue
        if key in output:
            raise ValueError(
                f"duplicate numeric frame {key} under {directory}: "
                f"{output[key].name}, {path.name}"
            )
        output[key] = path
    return output


def _place(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"refusing to replace {destination}")
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"unsupported materialization mode: {mode}")


def _scene_contracts(manifest: dict) -> dict[str, dict]:
    contracts: dict[str, dict] = {}
    for record in manifest.get("queries", []):
        scene = str(record["scene_id"])
        field_ids = [f"{int(value):06d}" for value in record["field_frame_ids"]]
        excluded = [f"{int(value):06d}" for value in record["query_exclusion_frames"]]
        expected_hash = str(record["field_frame_manifest_sha256"])
        if canonical_json_sha256(field_ids) != expected_hash:
            raise ValueError(f"{scene}: invalid field-frame manifest hash")
        current = {
            "field_frame_ids": field_ids,
            "excluded_frame_ids": excluded,
            "field_frame_manifest_sha256": expected_hash,
        }
        previous = contracts.setdefault(scene, current)
        if previous != current:
            raise ValueError(f"{scene}: query records disagree on reconstruction inputs")
    if not contracts:
        raise ValueError("manifest contains no PFIR queries")
    return contracts


def materialize(
    manifest_path: str | Path,
    dense_root: str | Path,
    output_root: str | Path,
    *,
    mode: str = "symlink",
    scenes: Iterable[str] = (),
) -> dict:
    manifest_path = Path(manifest_path).resolve()
    dense_root = Path(dense_root).resolve()
    output_root = Path(output_root).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    contracts = _scene_contracts(payload)
    requested = {str(value) for value in scenes}
    if requested:
        missing = sorted(requested - set(contracts))
        if missing:
            raise KeyError(f"requested scenes absent from manifest: {missing}")
        contracts = {key: value for key, value in contracts.items() if key in requested}

    reports = []
    for scene, contract in sorted(contracts.items()):
        source_root = dense_root / scene
        destination_root = output_root / scene
        if not source_root.is_dir():
            raise FileNotFoundError(f"missing dense PFIR scene: {source_root}")
        field_ids = list(contract["field_frame_ids"])
        exclusions = set(contract["excluded_frame_ids"])
        if exclusions.intersection(field_ids):
            raise ValueError(f"{scene}: excluded query-neighbour frame enters field")

        placed: dict[str, int] = {}
        for modality, suffixes in FRAME_MODALITIES.items():
            available = _numeric_files(source_root / modality, suffixes)
            missing = sorted(set(field_ids) - set(available))
            if missing:
                raise FileNotFoundError(
                    f"{scene}/{modality}: missing {len(missing)} field frames; "
                    f"first={missing[:5]}"
                )
            for frame_id in field_ids:
                source = available[frame_id]
                _place(
                    source,
                    destination_root / modality / source.name,
                    mode,
                )
            placed[modality] = len(field_ids)

        intrinsic_dir = destination_root / "intrinsic"
        for name in CAMERA_FILES:
            source = source_root / name
            if not source.is_file():
                raise FileNotFoundError(f"missing PFIR camera file: {source}")
            _place(source, destination_root / name, mode)
            _place(source, intrinsic_dir / SCANNET_CAMERA_LAYOUT[name], mode)

        source_manifest = source_root / "pfir_source_manifest.json"
        if source_manifest.is_file():
            _place(source_manifest, destination_root / source_manifest.name, mode)
        scene_report = {
            "scene_id": scene,
            "source_root": str(source_root),
            "output_root": str(destination_root),
            "mode": mode,
            "field_frame_ids": field_ids,
            "field_frame_count": len(field_ids),
            "excluded_frame_count": len(exclusions),
            "field_frame_manifest_sha256": contract[
                "field_frame_manifest_sha256"
            ],
            "placed_frame_counts": placed,
            "contains_instance_or_label_directories": any(
                (destination_root / value).exists() for value in ("instance", "label")
            ),
        }
        (destination_root / "pfir_field_contract.json").write_text(
            json.dumps(scene_report, indent=2) + "\n", encoding="utf-8"
        )
        reports.append(scene_report)

    report = {
        "schema_version": 1,
        "benchmark_version": payload.get("benchmark_version"),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dense_root": str(dense_root),
        "output_root": str(output_root),
        "scene_count": len(reports),
        "scenes": reports,
        "query_supervision_materialized": False,
        "valid": all(
            not row["contains_instance_or_label_directories"] for row in reports
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "materialization_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dense-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--mode", choices=("symlink", "hardlink", "copy"), default="symlink"
    )
    parser.add_argument("--scenes", nargs="*", default=())
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                args.manifest,
                args.dense_root,
                args.output_root,
                mode=args.mode,
                scenes=args.scenes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
