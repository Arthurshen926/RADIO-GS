"""Combine disjoint official-SAM generation shards without touching caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.utils.immutable_artifacts import load_json_object, write_frozen_json


def combine(paths: list[Path], *, output_root: Path) -> dict:
    if len(paths) < 2:
        raise ValueError("at least two official-SAM generation shards are required")
    reports: list[dict] = []
    stems: set[str] = set()
    for path in paths:
        payload, _digest, _source = load_json_object(
            path, label="official-SAM generation shard"
        )
        if Path(str(payload.get("output_root", ""))).resolve() != output_root.resolve():
            raise ValueError("official-SAM shard output roots differ")
        images = payload.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError("official-SAM generation shard is empty")
        for report in images:
            if not isinstance(report, dict):
                raise ValueError("official-SAM generation report differs")
            image = Path(str(report.get("image", "")))
            output = Path(str(report.get("output", ""))).resolve()
            if image.stem in stems:
                raise ValueError(f"duplicate official-SAM frame across shards: {image.stem}")
            if output != (output_root / f"{image.stem}.pt").resolve() or not output.is_file():
                raise ValueError("official-SAM shard output cache differs")
            stems.add(image.stem)
            reports.append(dict(report))
    reports.sort(key=lambda value: (
        (0, int(Path(str(value["image"])).stem))
        if Path(str(value["image"])).stem.isdigit()
        else (1, Path(str(value["image"])).stem)
    ))
    return {"output_root": str(output_root.resolve()), "images": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = combine(
        [Path(value).resolve() for value in args.inputs],
        output_root=Path(args.output_root).resolve(),
    )
    write_frozen_json(args.output, payload)
    print(json.dumps({
        "status": "combined_disjoint_official_sam_generation",
        "images": len(payload["images"]), "output": str(Path(args.output).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
