"""Combine independently sealed ScanNet source-SAM scene inventories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.scripts.build_scannet_source_sam_rollout_inventory import SCHEMA
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    sha256_file,
    write_frozen_json,
)


def combine(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("at least one scene inventory is required")
    scenes: list[dict] = []
    authorities: list[dict[str, str]] = []
    shared_method = promotion_gate = access_audit = None
    for index, path in enumerate(paths):
        payload, digest, source = load_json_object(
            path, label=f"source-SAM scene inventory {index}"
        )
        if (
            payload.get("schema") != SCHEMA
            or payload.get("status") != "frozen_before_source_sam_generation"
            or not isinstance(payload.get("scenes"), list)
            or len(payload["scenes"]) != 1
        ):
            raise ValueError("source-SAM scene inventory contract differs")
        current = (
            payload.get("shared_method"),
            payload.get("promotion_gate"),
            payload.get("access_audit"),
        )
        if index == 0:
            shared_method, promotion_gate, access_audit = current
        elif current != (shared_method, promotion_gate, access_audit):
            raise ValueError("source-SAM scene inventories use different frozen methods")
        scenes.append(dict(payload["scenes"][0]))
        authorities.append({"path": str(source), "sha256": digest})
    scene_ids = [str(value.get("scene_id", "")) for value in scenes]
    if not all(scene_ids) or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("combined source-SAM scene axis differs")
    return {
        "schema": SCHEMA,
        "status": "frozen_before_source_sam_generation",
        "scenes": scenes,
        "scene_inventory_authorities": authorities,
        "shared_method": shared_method,
        "promotion_gate": promotion_gate,
        "access_audit": access_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-inventories", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = combine([Path(value) for value in args.scene_inventories])
    write_frozen_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "scenes": {
            item["scene_id"]: item["source_frame_count"] for item in payload["scenes"]
        },
        "output": str(Path(args.output).resolve()),
        "sha256": sha256_file(args.output),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
