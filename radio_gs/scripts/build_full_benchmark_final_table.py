#!/usr/bin/env python3
"""Build the final PFPR/AGILE3D Markdown metric table from frozen JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result must contain one JSON object: {path}")
    return payload


def _fmt(value: object, *, percent: bool = False) -> str:
    number = float(value)
    return f"{100.0 * number:.2f}" if percent else f"{number:.3f}"


def build_table(
    pfpr_result: str | Path,
    agile_result: str | Path,
) -> str:
    pfpr = _load(pfpr_result)
    agile = _load(agile_result)
    if pfpr.get("benchmark") != "scannet-pfpr-small-v2":
        raise ValueError("PFPR result has the wrong benchmark identity")
    if agile.get("benchmark") != "AGILE3D ScanNet40 single-object":
        raise ValueError("AGILE result has the wrong benchmark identity")

    pfpr_protocol = dict(pfpr.get("protocol", {}))
    pfpr_metrics = dict(pfpr.get("metrics_query_micro", {}))
    agile_protocol = dict(agile.get("protocol", {}))
    agile_metrics = dict(agile.get("metrics", {}))
    pfpr_scope = (
        int(pfpr_protocol.get("scene_count", 0)),
        int(pfpr_protocol.get("query_count", 0)),
    )
    if pfpr_scope != (20, 200):
        raise ValueError(
            "final PFPR table requires exactly 20 scenes and 200 queries, "
            f"got {pfpr_scope[0]} scenes and {pfpr_scope[1]} queries"
        )
    agile_scope = (
        int(agile_protocol.get("scenes", 0)),
        int(agile_protocol.get("objects", 0)),
    )
    if agile_scope != (312, 10357):
        raise ValueError(
            "final AGILE table requires exactly 312 scenes and 10357 objects, "
            f"got {agile_scope[0]} scenes and {agile_scope[1]} objects"
        )
    required_agile_method = {
        "world_query": "compile_world_3d_query",
        "observation_lift": "none",
        "official_point_readout": "continuous_opacity_weighted_gaussian",
        "background_centroids": 4,
    }
    for key, required in required_agile_method.items():
        if agile_protocol.get(key) != required:
            raise ValueError(
                f"final AGILE table requires {key}={required!r}, "
                f"got {agile_protocol.get(key)!r}"
            )

    lines = [
        "# Full benchmark results",
        "",
        "All values below are read directly from the frozen evaluator JSON files.",
        "The no-support-gate runs are diagnostic and are not presented as formal",
        "leaderboard-comparable numbers.",
        "",
        "## ScanNet-PFPR-Small v2",
        "",
        "| Scenes | Queries | Top-1 mean error ↓ (m) | Top-1 median error ↓ (m) | R@1 / 5 cm ↑ | R@5 / 5 cm ↑ | R@10 / 5 cm ↑ | MRR / 5 cm ↑ | R@1 / 10 cm ↑ | R@5 / 10 cm ↑ | R@10 / 10 cm ↑ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            [
                str(int(pfpr_protocol.get("scene_count", 0))),
                str(int(pfpr_protocol.get("query_count", 0))),
                _fmt(pfpr_metrics["top1_mean_error_m"]),
                _fmt(pfpr_metrics["top1_median_error_m"]),
                _fmt(pfpr_metrics["R@1_5cm"], percent=True),
                _fmt(pfpr_metrics["R@5_5cm"], percent=True),
                _fmt(pfpr_metrics["R@10_5cm"], percent=True),
                _fmt(pfpr_metrics["MRR_5cm"], percent=True),
                _fmt(pfpr_metrics["R@1_10cm"], percent=True),
                _fmt(pfpr_metrics["R@5_10cm"], percent=True),
                _fmt(pfpr_metrics["R@10_10cm"], percent=True),
            ]
        )
        + " |",
        "",
        "## AGILE3D ScanNet40",
        "",
        "| Status | Scenes | Objects | IoU@1 ↑ | IoU@5 ↑ | IoU@10 ↑ | IoU@15 ↑ | NoC@50 ↓ | NoC@65 ↓ | NoC@80 ↓ | NoC@90 ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            [
                str(agile_protocol.get("result_status", "unknown")),
                str(int(agile_protocol.get("scenes", 0))),
                str(int(agile_protocol.get("objects", 0))),
                _fmt(agile_metrics["IoU@1"], percent=True),
                _fmt(agile_metrics["IoU@5"], percent=True),
                _fmt(agile_metrics["IoU@10"], percent=True),
                _fmt(agile_metrics["IoU@15"], percent=True),
                _fmt(agile_metrics["NoC@50"]),
                _fmt(agile_metrics["NoC@65"]),
                _fmt(agile_metrics["NoC@80"]),
                _fmt(agile_metrics["NoC@90"]),
            ]
        )
        + " |",
        "",
        "## Provenance",
        "",
        f"- PFPR result: `{Path(pfpr_result).resolve()}`",
        f"- AGILE result: `{Path(agile_result).resolve()}`",
        f"- AGILE observation contract: `{agile_protocol.get('observation_contract', '')}`",
        f"- AGILE canonical MPR contract: `{agile_protocol.get('canonical_mpr_contract', '')}`",
        f"- AGILE world-query compiler: `{agile_protocol.get('world_query', '')}`",
        f"- AGILE observation lift: `{agile_protocol.get('observation_lift', '')}`",
        f"- AGILE output readout: `{agile_protocol.get('official_point_readout', '')}`",
        f"- AGILE query-free scene background modes: `{int(agile_protocol.get('background_centroids', 0))}`",
        f"- AGILE formal comparable: `{bool(agile_protocol.get('formal_comparable', False))}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfpr-result", required=True)
    parser.add_argument("--agile-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    text = build_table(args.pfpr_result, args.agile_result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
