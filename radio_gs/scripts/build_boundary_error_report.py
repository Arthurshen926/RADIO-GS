"""Build measured SAM3-box boundary-error evidence for the paper audit."""

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
DEFAULT_MARKDOWN = REPO_ROOT / "output" / "radio_gs" / "reports" / "boundary_error_readout_report.md"
DEFAULT_JSON = REPO_ROOT / "output" / "radio_gs" / "reports" / "boundary_error_readout_report.json"
DEFAULT_LATEX = REPO_ROOT / "paper" / "boundary_error_readout_table.tex"


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
    key = _scene_key(value)
    return {
        "figurines": "Figurines",
        "ramen": "Ramen",
        "teatime": "Teatime",
        "waldo_kitchen": "Waldo Kitchen",
    }.get(key, value)


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


def _summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "mean_iou": _round4(_mean([float(row.get("iou", 0.0)) for row in rows])),
        "mean_boundary_f": _round4(_mean([float(row.get("boundary_f", 0.0)) for row in rows])),
        "mean_boundary_error": _round4(_mean([1.0 - float(row.get("boundary_f", 0.0)) for row in rows])),
        "mean_trimap_iou": _round4(_mean([float(row.get("trimap_iou", 0.0)) for row in rows])),
        "mean_gt_pixels": _round4(_mean([float(row.get("gt_pixels", 0.0)) for row in rows])),
        "mean_overselect_ratio": _round4(_mean([float(row.get("overselect_ratio", 0.0)) for row in rows])),
    }


def _bucket_by_overselect(queries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets = {"under": [], "balanced": [], "over": []}
    for row in queries:
        ratio = float(row.get("overselect_ratio", 0.0))
        if ratio < 0.67:
            buckets["under"].append(row)
        elif ratio <= 1.5:
            buckets["balanced"].append(row)
        else:
            buckets["over"].append(row)
    return {name: _summarize_bucket(rows) for name, rows in buckets.items()}


def _bucket_by_area(queries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets = {"small": [], "mid": [], "large": []}
    if not queries:
        return {name: _summarize_bucket(rows) for name, rows in buckets.items()}
    sorted_rows = sorted(queries, key=lambda row: float(row.get("gt_pixels", 0.0)))
    n = len(sorted_rows)
    for idx, row in enumerate(sorted_rows):
        if idx < n / 3:
            buckets["small"].append(row)
        elif idx < (2 * n) / 3:
            buckets["mid"].append(row)
        else:
            buckets["large"].append(row)
    return {name: _summarize_bucket(rows) for name, rows in buckets.items()}


def _load_query_details(source: str | Path, *, selection: str, fallback_scene: str) -> list[dict[str, Any]]:
    payload = _load_json(source)
    scene_payload = payload.get("scene", {})
    scene_name = str(scene_payload.get("scene", fallback_scene))
    result = scene_payload.get("results", {}).get(selection, {})
    rows = []
    for raw in result.get("query_details", []):
        ratio = float(raw.get("overselect_ratio", 0.0))
        rows.append(
            {
                "scene": _display_scene(scene_name),
                "frame": str(raw.get("frame", "")),
                "category": str(raw.get("category", "")),
                "iou": _round4(float(raw.get("iou", 0.0))),
                "boundary_f": _round4(float(raw.get("boundary_f", 0.0))),
                "boundary_error": _round4(1.0 - float(raw.get("boundary_f", 0.0))),
                "trimap_iou": _round4(float(raw.get("trimap_iou", 0.0))),
                "trimap_error": _round4(1.0 - float(raw.get("trimap_iou", 0.0))),
                "gt_pixels": int(raw.get("gt_pixels", 0)),
                "pred_pixels": int(raw.get("pred_pixels", 0)),
                "overselect_ratio": _round4(ratio),
                "overselect_log_abs": _round4(abs(math.log(max(ratio, 1e-6)))),
                "selected_gaussians": int(raw.get("selected_gaussians", 0)),
            }
        )
    return rows


def build_summary(
    sweep_path: str | Path = DEFAULT_SWEEP,
    *,
    run_label: str = "pad16",
    selection: str = "thr0p25",
) -> dict[str, Any]:
    sweep = _load_json(sweep_path)
    run = _select_run(sweep, run_label)
    fixed = run.get("fixed_global_threshold", {})
    scene_rows: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    base_dir = Path(sweep_path).resolve().parent
    for raw in fixed.get("rows", []):
        scene = str(raw.get("scene", "unknown"))
        source = _resolve_path(str(raw.get("source", "")), base_dir=base_dir)
        scene_rows.append(
            {
                "scene": _display_scene(scene),
                "selection": str(raw.get("selection", selection)),
                "miou": _round4(float(raw.get("miou", 0.0))),
                "boundary_f": _round4(float(raw.get("boundary_f", 0.0))),
                "boundary_error": _round4(1.0 - float(raw.get("boundary_f", 0.0))),
                "trimap_iou": _round4(float(raw.get("trimap_iou", 0.0))),
                "trimap_error": _round4(1.0 - float(raw.get("trimap_iou", 0.0))),
                "n": int(raw.get("n", 0)),
                "source": str(source.relative_to(REPO_ROOT) if source.is_relative_to(REPO_ROOT) else source),
            }
        )
        if source.exists():
            queries.extend(_load_query_details(source, selection=selection, fallback_scene=scene))

    ious = [float(row["iou"]) for row in queries]
    boundary = [float(row["boundary_f"]) for row in queries]
    trimap = [float(row["trimap_iou"]) for row in queries]
    overselect = [float(row["overselect_log_abs"]) for row in queries]
    worst = sorted(queries, key=lambda row: (float(row["boundary_f"]), float(row["iou"])))[:10]
    return {
        "sweep_source": str(sweep_path),
        "run_label": run_label,
        "source_root": str(run.get("source_root", "")),
        "selection": selection,
        "macro": {
            "miou": _round4(float(fixed.get("macro_miou", 0.0))),
            "boundary_f": _round4(float(fixed.get("macro_boundary_f", 0.0))),
            "boundary_error": _round4(1.0 - float(fixed.get("macro_boundary_f", 0.0))),
            "trimap_iou": _round4(float(fixed.get("macro_trimap_iou", 0.0))),
            "trimap_error": _round4(1.0 - float(fixed.get("macro_trimap_iou", 0.0))),
        },
        "scene_rows": scene_rows,
        "query_count": len(queries),
        "query_correlations": {
            "iou_vs_boundary_f": _pearson(ious, boundary),
            "iou_vs_trimap_iou": _pearson(ious, trimap),
            "iou_vs_abs_log_overselect": _pearson(ious, overselect),
        },
        "overselect_buckets": _bucket_by_overselect(queries),
        "area_buckets": _bucket_by_area(queries),
        "worst_boundary_cases": worst,
        "alpha_depth_status": "not_available",
    }


def _bucket_lines(title: str, buckets: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Bucket | Count | Mean IoU | Mean boundary-F | Mean boundary error | Mean trimap IoU | Mean overselect |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in buckets.items():
        lines.append(
            "| {name} | {count} | {iou:.4f} | {bf:.4f} | {berr:.4f} | {trimap:.4f} | {over:.4f} |".format(
                name=name,
                count=int(row.get("count", 0)),
                iou=float(row.get("mean_iou", 0.0)),
                bf=float(row.get("mean_boundary_f", 0.0)),
                berr=float(row.get("mean_boundary_error", 0.0)),
                trimap=float(row.get("mean_trimap_iou", 0.0)),
                over=float(row.get("mean_overselect_ratio", 0.0)),
            )
        )
    lines.append("")
    return lines


def build_markdown(summary: dict[str, Any]) -> str:
    macro = summary.get("macro", {})
    corr = summary.get("query_correlations", {})
    lines = [
        "# Boundary Error Readout",
        "",
        f"- Sweep source: `{summary.get('sweep_source', '')}`",
        f"- Run: `{summary.get('run_label', '')}`; selection: `{summary.get('selection', '')}`",
        f"- Query count with per-query boundary details: `{summary.get('query_count', 0)}`",
        "- Boundary error is `1 - boundary_f`; trimap error is `1 - trimap_iou`.",
        "",
        "## Strict Readout",
        "",
        "| Macro mIoU | Boundary-F | Boundary error | Trimap IoU | Trimap error |",
        "|---:|---:|---:|---:|---:|",
        "| {miou:.4f} | {bf:.4f} | {berr:.4f} | {trimap:.4f} | {terr:.4f} |".format(
            miou=float(macro.get("miou", 0.0)),
            bf=float(macro.get("boundary_f", 0.0)),
            berr=float(macro.get("boundary_error", 0.0)),
            trimap=float(macro.get("trimap_iou", 0.0)),
            terr=float(macro.get("trimap_error", 0.0)),
        ),
        "",
        "## Scene Rows",
        "",
        "| Scene | mIoU | Boundary-F | Boundary error | Trimap IoU | Trimap error | N |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("scene_rows", []):
        lines.append(
            "| {scene} | {miou:.4f} | {bf:.4f} | {berr:.4f} | {trimap:.4f} | {terr:.4f} | {n} |".format(
                scene=row["scene"],
                miou=float(row["miou"]),
                bf=float(row["boundary_f"]),
                berr=float(row["boundary_error"]),
                trimap=float(row["trimap_iou"]),
                terr=float(row["trimap_error"]),
                n=int(row["n"]),
            )
        )
    lines.extend(
        [
            "",
            "## Query Correlations",
            "",
            "| Pair | Pearson r |",
            "|---|---:|",
            f"| IoU vs boundary-F | {float(corr.get('iou_vs_boundary_f', 0.0)):.4f} |",
            f"| IoU vs trimap IoU | {float(corr.get('iou_vs_trimap_iou', 0.0)):.4f} |",
            f"| IoU vs abs(log overselect ratio) | {float(corr.get('iou_vs_abs_log_overselect', 0.0)):.4f} |",
            "",
        ]
    )
    lines.extend(_bucket_lines("Overselect Buckets", summary.get("overselect_buckets", {})))
    lines.extend(_bucket_lines("GT-Area Buckets", summary.get("area_buckets", {})))
    lines.extend(
        [
            "## Worst Boundary Cases",
            "",
            "| Scene | Frame | Category | IoU | Boundary-F | Trimap IoU | GT px | Pred px | Overselect |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("worst_boundary_cases", []):
        lines.append(
            "| {scene} | `{frame}` | {cat} | {iou:.4f} | {bf:.4f} | {trimap:.4f} | {gt} | {pred} | {over:.4f} |".format(
                scene=row["scene"],
                frame=row["frame"],
                cat=row["category"],
                iou=float(row["iou"]),
                bf=float(row["boundary_f"]),
                trimap=float(row["trimap_iou"]),
                gt=int(row["gt_pixels"]),
                pred=int(row["pred_pixels"]),
                over=float(row["overselect_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Alpha/Depth Discontinuity Status",
            "",
            "- Alpha/depth discontinuity maps are not present in the frozen SAM3-box result JSONs.",
            "- The report therefore supports a measured boundary-error readout, not a causal occlusion/discontinuity claim.",
            "- The current occlusion-adjacent evidence remains protocol metadata and the negative alpha/alpha-depth registration-weight ablation; a stronger audit needs saved per-query alpha/depth edges aligned to the official masks.",
            "",
        ]
    )
    return "\n".join(lines)


def build_latex_table(summary: dict[str, Any]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Measured boundary-error readout for the strict SAM3-box Direct3D row. Boundary error is $1-$Boundary-F and trimap error is $1-$trimap IoU.}",
        "\\label{tab:boundary_error_readout}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Scene & mIoU & Boundary-F & Boundary err. & Trimap IoU & Trimap err. \\\\",
        "\\midrule",
    ]
    for row in summary.get("scene_rows", []):
        lines.append(
            "{scene} & {miou:.4f} & {bf:.4f} & {berr:.4f} & {trimap:.4f} & {terr:.4f} \\\\".format(
                scene=_latex_escape(str(row["scene"])),
                miou=float(row["miou"]),
                bf=float(row["boundary_f"]),
                berr=float(row["boundary_error"]),
                trimap=float(row["trimap_iou"]),
                terr=float(row["trimap_error"]),
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
