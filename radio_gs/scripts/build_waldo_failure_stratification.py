"""Build a Waldo Kitchen direct-3D failure stratification report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_JSON = (
    REPO_ROOT
    / "output"
    / "radio_gs"
    / "lerf_direct_3d_selection_threshold_grabcut_20260515"
    / "waldo_kitchen"
    / "lerf_direct_3d_selection_results.json"
)
DEFAULT_MARKDOWN = REPO_ROOT / "output" / "radio_gs" / "reports" / "waldo_failure_stratification.md"
DEFAULT_JSON = REPO_ROOT / "output" / "radio_gs" / "reports" / "waldo_failure_stratification.json"


def _round4(value: float) -> float:
    return round(float(value), 4)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bucket(gt_pixels: int) -> str:
    if gt_pixels < 5_000:
        return "small"
    if gt_pixels < 15_000:
        return "medium"
    return "large"


def _bucket_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "mean_iou": 0.0, "zero_prediction_rate": 0.0, "mean_overselect": 0.0}
    return {
        "n": len(rows),
        "mean_iou": _round4(_mean([float(row.get("iou", 0.0)) for row in rows])),
        "zero_prediction_rate": _round4(
            sum(1 for row in rows if int(row.get("pred_pixels", 0)) == 0) / len(rows)
        ),
        "mean_overselect": _round4(
            _mean([float(row.get("overselect_ratio", 0.0)) for row in rows])
        ),
    }


def summarize_result(path: str | Path, *, selection: str) -> dict[str, Any]:
    result_path = Path(path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    scene_payload = payload.get("scene", {})
    results = scene_payload.get("results", {})
    if selection not in results:
        raise KeyError(f"Selection {selection!r} not found in {result_path}")
    metrics = results[selection]
    details = list(metrics.get("query_details", []))

    buckets = {"small": [], "medium": [], "large": []}
    for row in details:
        buckets[_bucket(int(row.get("gt_pixels", 0)))].append(row)

    zero_categories = [
        str(row.get("category", "unknown"))
        for row in details
        if int(row.get("pred_pixels", 0)) == 0
    ]
    return {
        "scene": scene_payload.get("scene", "waldo_kitchen"),
        "selection": selection,
        "source": str(result_path),
        "query_count": len(details),
        "mean_iou": _round4(float(metrics.get("miou", _mean([float(row.get("iou", 0.0)) for row in details])))),
        "acc025": _round4(float(metrics.get("acc025", 0.0))),
        "zero_prediction_rate": _round4(
            sum(1 for row in details if int(row.get("pred_pixels", 0)) == 0) / len(details)
            if details
            else 0.0
        ),
        "mean_overselect": _round4(
            _mean([float(row.get("overselect_ratio", 0.0)) for row in details])
        ),
        "size_buckets": {name: _bucket_summary(rows) for name, rows in buckets.items()},
        "worst_zero_prediction_categories": zero_categories[:8],
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Waldo Kitchen Failure Stratification",
        "",
        f"- Source: `{summary['source']}`",
        f"- Selection: `{summary['selection']}`",
        f"- Queries: `{summary['query_count']}`",
        f"- Mean IoU: `{float(summary['mean_iou']):.4f}`",
        f"- Acc@0.25: `{float(summary['acc025']):.4f}`",
        f"- Zero-prediction rate: `{float(summary['zero_prediction_rate']):.4f}`",
        f"- Mean overselect ratio: `{float(summary['mean_overselect']):.4f}`",
        "",
        "## Object-Size Buckets",
        "",
        "| Bucket | Queries | Mean IoU | Zero-pred rate | Mean overselect |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("small", "medium", "large"):
        bucket = summary["size_buckets"][name]
        lines.append(
            "| {name} | {n} | {iou:.4f} | {zero:.4f} | {over:.4f} |".format(
                name=name,
                n=int(bucket["n"]),
                iou=float(bucket["mean_iou"]),
                zero=float(bucket["zero_prediction_rate"]),
                over=float(bucket["mean_overselect"]),
            )
        )
    zeros = summary.get("worst_zero_prediction_categories", [])
    lines.extend(
        [
            "",
            "## Zero-Prediction Categories",
            "",
            ", ".join(zeros) if zeros else "none",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    summary: dict[str, Any],
    markdown_path: str | Path,
    json_path: str | Path,
) -> dict[str, Path]:
    markdown_out = Path(markdown_path)
    json_out = Path(json_path)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(build_markdown(summary), encoding="utf-8")
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"markdown": markdown_out, "json": json_out}


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_json", default=str(DEFAULT_RESULT_JSON))
    parser.add_argument("--selection", default="thr0p25")
    parser.add_argument("--output_md", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--output_json", default=str(DEFAULT_JSON))
    args = parser.parse_args(argv)

    summary = summarize_result(args.result_json, selection=args.selection)
    paths = write_outputs(summary, args.output_md, args.output_json)
    print(f"Wrote {paths['markdown']}")
    print(f"Wrote {paths['json']}")
    return paths


if __name__ == "__main__":
    main()
