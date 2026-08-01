#!/usr/bin/env python3
"""Merge independently rendered LERF scene reports without rescoring masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROTOCOL_ARGUMENT_KEYS = (
    "rendered_only",
    "render_readout",
    "primitive_valid_normalization",
    "primitive_valid_coverage_power",
    "feature_contribution_gamma",
    "scoring",
    "relevancy_temp",
    "threshold_mode",
    "iou_threshold",
    "heatmap_upsample",
    "localization_mode",
    "mask_refinement",
    "prompt_templates",
)
OBSERVATION_OPERATOR_KEYS = (
    "type",
    "gamma",
    "primitive_valid_normalization",
    "semantic_score_formula",
    "semantic_coverage_power",
    "query_dependent",
    "changes_geometry_or_alpha",
)


def _protocol(payload: dict) -> dict:
    arguments = payload.get("args")
    if not isinstance(arguments, dict):
        raise ValueError("LERF report does not contain an args mapping")
    operator = payload.get("feature_observation_operator")
    if not isinstance(operator, dict):
        raise ValueError(
            "LERF report does not declare its feature observation operator"
        )
    return {
        "args": {
            key: arguments.get(key)
            for key in PROTOCOL_ARGUMENT_KEYS
        },
        "prompt_templates": payload.get("prompt_templates"),
        "feature_observation_operator": {
            key: operator.get(key)
            for key in OBSERVATION_OPERATOR_KEYS
        },
    }


def summarize_reports(
    report_paths: list[str | Path],
    *,
    expected_scenes: set[str] | None = None,
) -> dict:
    if not report_paths:
        raise ValueError("at least one LERF scene report is required")
    rows = []
    protocol = None
    seen_scenes: set[str] = set()
    for value in report_paths:
        path = Path(value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        current_protocol = _protocol(payload)
        if protocol is None:
            protocol = current_protocol
        elif current_protocol != protocol:
            raise ValueError(f"protocol mismatch in {path}")
        if len(payload["scenes"]) != 1:
            raise ValueError(f"expected one scene in {path}")
        scene = next(iter(payload["scenes"]))
        if scene in seen_scenes:
            raise ValueError(f"duplicate LERF scene report: {scene}")
        seen_scenes.add(scene)
        aggregate = payload["aggregates"]["rendered"]
        rows.append({"scene": scene, "report": str(path), **aggregate})

    if expected_scenes is not None and seen_scenes != expected_scenes:
        missing = sorted(expected_scenes - seen_scenes)
        unexpected = sorted(seen_scenes - expected_scenes)
        raise ValueError(
            "LERF scene set mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    count = sum(int(row["sample_count"]) for row in rows)
    if count <= 0:
        raise ValueError("LERF reports contain no scored samples")
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-scenes",
        nargs="*",
        default=None,
        help="Optional exact scene set; duplicates and missing scenes fail closed.",
    )
    args = parser.parse_args()
    summary = summarize_reports(
        args.reports,
        expected_scenes=(
            set(args.expected_scenes)
            if args.expected_scenes is not None
            else None
        ),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
