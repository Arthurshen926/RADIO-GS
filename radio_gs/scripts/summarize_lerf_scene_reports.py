#!/usr/bin/env python3
"""Merge independently rendered LERF scene reports without rescoring masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    protocol = None
    for value in args.reports:
        path = Path(value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        current_protocol = {
            key: payload["args"].get(key)
            for key in (
                "render_readout", "scoring", "relevancy_temp",
                "threshold_mode", "iou_threshold", "localization_mode",
                "mask_refinement", "prompt_templates",
            )
        }
        if protocol is None:
            protocol = current_protocol
        elif current_protocol != protocol:
            raise ValueError(f"protocol mismatch in {path}")
        if len(payload["scenes"]) != 1:
            raise ValueError(f"expected one scene in {path}")
        scene = next(iter(payload["scenes"]))
        aggregate = payload["aggregates"]["rendered"]
        rows.append({"scene": scene, "report": str(path), **aggregate})

    count = sum(int(row["sample_count"]) for row in rows)
    summary = {
        "schema_version": 1,
        "protocol": protocol,
        "scenes": rows,
        "aggregate": {
            "scene_macro_miou": sum(row["sample_micro_miou"] for row in rows) / len(rows),
            "scene_macro_localization_accuracy": sum(row["localization_accuracy"] for row in rows) / len(rows),
            "sample_micro_miou": sum(row["sample_micro_miou"] * row["sample_count"] for row in rows) / count,
            "sample_micro_localization_accuracy": sum(row["localization_accuracy"] * row["sample_count"] for row in rows) / count,
            "sample_count": count,
            "scene_count": len(rows),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
