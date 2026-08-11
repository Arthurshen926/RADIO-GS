"""Build an explicit source-frame inventory from one frozen SAM relation cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.scripts.build_source_official_sam_authority import INVENTORY_SCHEMA
from radio_gs.utils.immutable_artifacts import (
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
)


def _frame_stems(path: Path) -> set[str]:
    values: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if token:
            values.add(Path(token).stem)
    if not values:
        raise ValueError("source frame split is empty")
    return values


def build_inventory(
    relation: dict,
    *,
    scene_id: str,
    source_frame_list: Path,
    source_frame_list_sha256: str,
) -> dict:
    metadata = relation.get("metadata")
    if not isinstance(metadata, dict) or any(
        metadata.get(name) is not wanted
        for name, wanted in {
            "query_free": True,
            "labels_opened": False,
            "instances_opened": False,
            "text_opened": False,
        }.items()
    ):
        raise ValueError("source SAM relation provenance differs")
    names = metadata.get("mask_frames")
    roots = metadata.get("mask_roots")
    if not isinstance(names, list) or not names or not isinstance(roots, list) or not roots:
        raise ValueError("source SAM relation lacks its mask inventory")
    split = _frame_stems(source_frame_list)
    by_name: dict[str, list[Path]] = {}
    for root_value in roots:
        root = Path(str(root_value))
        if not root.is_dir():
            raise FileNotFoundError(f"source SAM mask root does not exist: {root}")
        for name in names:
            candidate = root / Path(str(name)).name
            if candidate.is_file():
                by_name.setdefault(Path(str(name)).name, []).append(candidate)
    records: list[dict[str, str]] = []
    frames: list[str] = []
    for name_value in names:
        name = Path(str(name_value)).name
        matches = by_name.get(name, [])
        if len(matches) != 1:
            raise ValueError(f"source SAM mask cache must resolve exactly once: {name}")
        stem = Path(name).stem
        if stem not in split:
            raise ValueError(f"source SAM frame is outside the frozen mapping split: {stem}")
        frames.append(stem)
        records.append(
            {"path": str(matches[0].resolve()), "sha256": sha256_file(matches[0])}
        )
    if len(frames) != len(set(frames)):
        raise ValueError("source SAM relation repeats a frame")
    return {
        "schema": INVENTORY_SCHEMA,
        "scene_id": str(scene_id),
        "source_frames": frames,
        "source_split_authority": {
            "path": str(source_frame_list.resolve()),
            "sha256": str(source_frame_list_sha256),
        },
        "target_or_evaluation_frames_excluded": True,
        "benchmark_queries_excluded": True,
        "mask_caches": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation-cache", required=True)
    parser.add_argument("--expected-relation-cache-sha256", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--source-frame-list", required=True)
    parser.add_argument("--expected-source-frame-list-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    relation, _digest, _source = load_torch_mapping(
        args.relation_cache,
        expected_sha256=args.expected_relation_cache_sha256,
        map_location="cpu",
        label="source official-SAM relation cache",
    )
    frame_list = Path(args.source_frame_list).resolve()
    if sha256_file(frame_list) != args.expected_source_frame_list_sha256:
        raise ValueError("source frame split SHA-256 differs")
    inventory = build_inventory(
        relation,
        scene_id=args.scene_id,
        source_frame_list=frame_list,
        source_frame_list_sha256=args.expected_source_frame_list_sha256,
    )
    write_frozen_json(args.output, inventory)
    print(
        json.dumps(
            {
                "status": "sealed_source_official_sam_inventory",
                "scene_id": inventory["scene_id"],
                "source_frames": len(inventory["source_frames"]),
                "output": str(Path(args.output).resolve()),
                "sha256": sha256_file(args.output),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
