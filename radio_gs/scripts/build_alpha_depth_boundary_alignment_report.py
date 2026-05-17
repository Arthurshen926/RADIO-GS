"""Build alpha/depth discontinuity alignment evidence for SAM3-box Direct3D."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP = (
    REPO_ROOT / "output" / "radio_gs" / "reports" / "lerf_sam3_box_global_threshold_sweep_20260516.json"
)
DEFAULT_MARKDOWN = REPO_ROOT / "output" / "radio_gs" / "reports" / "alpha_depth_boundary_alignment_report.md"
DEFAULT_JSON = REPO_ROOT / "output" / "radio_gs" / "reports" / "alpha_depth_boundary_alignment_report.json"
DEFAULT_LATEX = REPO_ROOT / "paper" / "alpha_depth_boundary_alignment_table.tex"

GEOMETRY_METRICS = (
    "alpha_edge_gt_boundary_mean",
    "depth_edge_gt_boundary_mean",
    "discontinuity_gt_boundary_mean",
    "discontinuity_error_boundary_mean",
    "discontinuity_pred_boundary_mean",
)


def _round4(value: float) -> float:
    return round(float(value), 4)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return _round4(cov / math.sqrt(vx * vy))


def _scene_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _display_scene(value: str) -> str:
    return {
        "figurines": "Figurines",
        "ramen": "Ramen",
        "teatime": "Teatime",
        "waldo_kitchen": "Waldo Kitchen",
    }.get(_scene_key(value), value)


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _resolve_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    return base_dir / path


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _select_run(payload: dict[str, Any], run_label: str) -> dict[str, Any]:
    for run in payload.get("runs", []):
        if str(run.get("label", "")) == run_label:
            return run
    raise KeyError(f"Run label {run_label!r} not found")


def _has_geometry_metrics(row: dict[str, Any]) -> bool:
    return bool(int(row.get("geometry_valid", 0))) and any(key in row for key in GEOMETRY_METRICS)


def _load_query_details(source: str | Path, *, selection: str, fallback_scene: str) -> list[dict[str, Any]]:
    payload = _load_json(source)
    scene_payload = payload.get("scene", {})
    scene_name = _display_scene(str(scene_payload.get("scene", fallback_scene)))
    result = scene_payload.get("results", {}).get(selection, {})
    rows = []
    for raw in result.get("query_details", []):
        row: dict[str, Any] = {
            "scene": scene_name,
            "frame": str(raw.get("frame", "")),
            "category": str(raw.get("category", "")),
            "iou": _round4(float(raw.get("iou", 0.0))),
            "boundary_f": _round4(float(raw.get("boundary_f", 0.0))),
            "boundary_error": _round4(1.0 - float(raw.get("boundary_f", 0.0))),
            "trimap_iou": _round4(float(raw.get("trimap_iou", 0.0))),
            "geometry_valid": int(raw.get("geometry_valid", 0)),
            "geometry_overlay_path": str(raw.get("geometry_overlay_path", "")),
        }
        for key in GEOMETRY_METRICS:
            if key in raw:
                row[key] = _round4(float(raw[key]))
        rows.append(row)
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "mean_iou": _round4(_mean([float(row.get("iou", 0.0)) for row in rows])),
        "mean_boundary_error": _round4(_mean([float(row.get("boundary_error", 0.0)) for row in rows])),
        "mean_discontinuity_gt_boundary": _round4(
            _mean([float(row.get("discontinuity_gt_boundary_mean", 0.0)) for row in rows])
        ),
        "mean_discontinuity_error_boundary": _round4(
            _mean([float(row.get("discontinuity_error_boundary_mean", 0.0)) for row in rows])
        ),
    }


def _bucket_by_discontinuity(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets = {"low": [], "mid": [], "high": []}
    if not rows:
        return {name: _summarize_rows(bucket) for name, bucket in buckets.items()}
    sorted_rows = sorted(rows, key=lambda row: float(row.get("discontinuity_gt_boundary_mean", 0.0)))
    n = len(sorted_rows)
    for idx, row in enumerate(sorted_rows):
        if idx < n / 3:
            buckets["low"].append(row)
        elif idx < (2 * n) / 3:
            buckets["mid"].append(row)
        else:
            buckets["high"].append(row)
    return {name: _summarize_rows(bucket) for name, bucket in buckets.items()}


def build_summary(
    sweep_path: str | Path = DEFAULT_SWEEP,
    *,
    run_label: str = "pad16",
    selection: str = "thr0p25",
) -> dict[str, Any]:
    sweep = _load_json(sweep_path)
    run = _select_run(sweep, run_label)
    fixed = run.get("fixed_global_threshold", {})
    base_dir = Path(sweep_path).resolve().parent
    scene_rows: list[dict[str, Any]] = []
    all_queries: list[dict[str, Any]] = []
    geometry_queries: list[dict[str, Any]] = []

    for raw in fixed.get("rows", []):
        scene = str(raw.get("scene", "unknown"))
        source = _resolve_path(str(raw.get("source", "")), base_dir=base_dir)
        queries = _load_query_details(source, selection=selection, fallback_scene=scene) if source.exists() else []
        scene_geometry = [row for row in queries if _has_geometry_metrics(row)]
        all_queries.extend(queries)
        geometry_queries.extend(scene_geometry)
        scene_rows.append(
            {
                "scene": _display_scene(scene),
                "n": int(raw.get("n", 0)),
                "query_count": len(queries),
                "geometry_query_count": len(scene_geometry),
                "miou": _round4(float(raw.get("miou", 0.0))),
                "boundary_f": _round4(float(raw.get("boundary_f", 0.0))),
                "source": str(source.relative_to(REPO_ROOT) if source.is_relative_to(REPO_ROOT) else source),
            }
        )

    boundary_errors = [float(row["boundary_error"]) for row in geometry_queries]
    correlations = {
        "boundary_error_vs_alpha_edge_gt_boundary_mean": _pearson(
            boundary_errors,
            [float(row.get("alpha_edge_gt_boundary_mean", 0.0)) for row in geometry_queries],
        ),
        "boundary_error_vs_depth_edge_gt_boundary_mean": _pearson(
            boundary_errors,
            [float(row.get("depth_edge_gt_boundary_mean", 0.0)) for row in geometry_queries],
        ),
        "boundary_error_vs_discontinuity_gt_boundary_mean": _pearson(
            boundary_errors,
            [float(row.get("discontinuity_gt_boundary_mean", 0.0)) for row in geometry_queries],
        ),
        "boundary_error_vs_discontinuity_error_boundary_mean": _pearson(
            boundary_errors,
            [float(row.get("discontinuity_error_boundary_mean", 0.0)) for row in geometry_queries],
        ),
    }
    worst = sorted(
        geometry_queries,
        key=lambda row: (
            -float(row.get("boundary_error", 0.0)),
            -float(row.get("discontinuity_error_boundary_mean", 0.0)),
        ),
    )[:10]
    return {
        "sweep_source": str(sweep_path),
        "run_label": run_label,
        "selection": selection,
        "geometry_alignment_status": "available" if geometry_queries else "not_available",
        "query_count": len(all_queries),
        "geometry_query_count": len(geometry_queries),
        "map_artifact_count": sum(1 for row in geometry_queries if row.get("geometry_overlay_path")),
        "scene_rows": scene_rows,
        "query_correlations": correlations,
        "discontinuity_buckets": _bucket_by_discontinuity(geometry_queries),
        "worst_geometry_cases": worst,
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Alpha/Depth Boundary Alignment Report",
        "",
        f"- Sweep source: `{summary.get('sweep_source', '')}`",
        f"- Run: `{summary.get('run_label', '')}`; selection: `{summary.get('selection', '')}`",
        f"- Query records: `{summary.get('query_count', 0)}`",
        f"- Query records with alpha/depth geometry metrics: `{summary.get('geometry_query_count', 0)}`",
        f"- Query overlay artifacts: `{summary.get('map_artifact_count', 0)}`",
        f"- Status: `{summary.get('geometry_alignment_status', 'unknown')}`",
        "",
    ]
    if summary.get("geometry_alignment_status") != "available":
        lines.extend(
            [
                "Alpha/depth geometry maps are not available in the current frozen SAM3-box result JSONs.",
                "This report therefore records an instrumentation gap rather than a causal occlusion/discontinuity result.",
                "Regenerate the Direct3D SAM3-box readout with `--save_geometry_maps` to populate per-query metrics and overlays.",
                "",
            ]
        )
    lines.extend(
        [
            "## Scene Coverage",
            "",
            "| Scene | N | Query records | Geometry records | mIoU | Boundary-F |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("scene_rows", []):
        lines.append(
            "| {scene} | {n} | {query_count} | {geometry_query_count} | {miou:.4f} | {boundary:.4f} |".format(
                scene=row["scene"],
                n=int(row.get("n", 0)),
                query_count=int(row.get("query_count", 0)),
                geometry_query_count=int(row.get("geometry_query_count", 0)),
                miou=float(row.get("miou", 0.0)),
                boundary=float(row.get("boundary_f", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Query Correlations",
            "",
            "| Pair | Pearson r |",
            "|---|---:|",
        ]
    )
    for key, value in summary.get("query_correlations", {}).items():
        lines.append(f"| {key.replace('_', ' ')} | {float(value):.4f} |")
    lines.extend(
        [
            "",
            "## Discontinuity Buckets",
            "",
            "| Bucket | Count | Mean IoU | Mean boundary error | Mean disc. on GT boundary | Mean disc. on error boundary |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary.get("discontinuity_buckets", {}).items():
        lines.append(
            "| {name} | {count} | {iou:.4f} | {berr:.4f} | {disc_gt:.4f} | {disc_err:.4f} |".format(
                name=name,
                count=int(row.get("count", 0)),
                iou=float(row.get("mean_iou", 0.0)),
                berr=float(row.get("mean_boundary_error", 0.0)),
                disc_gt=float(row.get("mean_discontinuity_gt_boundary", 0.0)),
                disc_err=float(row.get("mean_discontinuity_error_boundary", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Worst Geometry-Aligned Cases",
            "",
            "| Scene | Frame | Category | IoU | Boundary error | Disc. GT boundary | Disc. error boundary | Overlay |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary.get("worst_geometry_cases", []):
        lines.append(
            "| {scene} | `{frame}` | {category} | {iou:.4f} | {berr:.4f} | {disc_gt:.4f} | {disc_err:.4f} | `{overlay}` |".format(
                scene=row.get("scene", ""),
                frame=row.get("frame", ""),
                category=row.get("category", ""),
                iou=float(row.get("iou", 0.0)),
                berr=float(row.get("boundary_error", 0.0)),
                disc_gt=float(row.get("discontinuity_gt_boundary_mean", 0.0)),
                disc_err=float(row.get("discontinuity_error_boundary_mean", 0.0)),
                overlay=row.get("geometry_overlay_path", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_latex_table(summary: dict[str, Any]) -> str:
    rows = summary.get("scene_rows", [])
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Alpha/depth discontinuity alignment coverage for the strict SAM3-box Direct3D row. Geometry records count per-query alpha/depth edge metrics and overlays.}",
        "\\label{tab:alpha_depth_boundary_alignment}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Scene & Queries & Geometry records & mIoU & Boundary-F \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            "{scene} & {queries} & {geom} & {miou:.4f} & {boundary:.4f} \\\\".format(
                scene=_latex_escape(str(row.get("scene", ""))),
                queries=int(row.get("query_count", 0)),
                geom=int(row.get("geometry_query_count", 0)),
                miou=float(row.get("miou", 0.0)),
                boundary=float(row.get("boundary_f", 0.0)),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def write_outputs(
    summary: dict[str, Any],
    markdown_path: str | Path = DEFAULT_MARKDOWN,
    json_path: str | Path = DEFAULT_JSON,
    latex_path: str | Path = DEFAULT_LATEX,
) -> dict[str, Path]:
    markdown_out = Path(markdown_path)
    json_out = Path(json_path)
    latex_out = Path(latex_path)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    latex_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(build_markdown(summary), encoding="utf-8")
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    latex_out.write_text(build_latex_table(summary), encoding="utf-8")
    return {"markdown": markdown_out, "json": json_out, "latex": latex_out}


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default=str(DEFAULT_SWEEP))
    parser.add_argument("--run_label", default="pad16")
    parser.add_argument("--selection", default="thr0p25")
    parser.add_argument("--output_md", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--output_json", default=str(DEFAULT_JSON))
    parser.add_argument("--output_tex", default=str(DEFAULT_LATEX))
    args = parser.parse_args(argv)

    summary = build_summary(args.sweep, run_label=args.run_label, selection=args.selection)
    paths = write_outputs(summary, args.output_md, args.output_json, args.output_tex)
    print(f"Wrote {paths['markdown']}")
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['latex']}")
    return paths


if __name__ == "__main__":
    main()
