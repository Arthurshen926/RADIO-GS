"""Build a Direct3D view-coverage and teacher-confidence analysis report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = (
    REPO_ROOT
    / "output"
    / "radio_gs"
    / "lerf_direct_3d_selection_threshold_grabcut_20260515"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT
    / "output"
    / "radio_gs"
    / "reports"
    / "lerf_direct3d_confidence_coverage_analysis.md"
)
DEFAULT_JSON = (
    REPO_ROOT
    / "output"
    / "radio_gs"
    / "reports"
    / "lerf_direct3d_confidence_coverage_analysis.json"
)
DEFAULT_LATEX = REPO_ROOT / "paper" / "lerf_direct3d_confidence_coverage_table.tex"


def _round4(value: float) -> float:
    return round(float(value), 4)


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return _round4(cov / math.sqrt(var_x * var_y))


def _result_paths(result_root: Path) -> list[Path]:
    return sorted(result_root.glob("*/lerf_direct_3d_selection_results.json"))


def _resolve_path(path: str | Path, *, base: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    return base / candidate


def _load_score_stats(
    cache_path: Path,
    categories: list[str],
    *,
    top_score_ratio: float,
) -> dict[str, dict[str, float]]:
    payload = torch.load(cache_path, map_location="cpu")
    scores = payload.get("scores")
    if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
        raise ValueError(f"Score cache {cache_path} must contain a 2D 'scores' tensor")
    if scores.shape[1] != len(categories):
        raise ValueError(
            f"Score cache {cache_path} has {scores.shape[1]} columns for "
            f"{len(categories)} categories"
        )
    scores = scores.float().cpu()
    top_k = max(1, min(int(scores.shape[0]), int(round(scores.shape[0] * top_score_ratio))))
    result: dict[str, dict[str, float]] = {}
    for idx, category in enumerate(categories):
        column = scores[:, idx]
        top_values, top_indices = torch.topk(column, k=top_k, largest=True)
        if scores.shape[1] > 1:
            competitor_scores = scores.clone()
            competitor_scores[:, idx] = float("-inf")
            top_competitors = competitor_scores[top_indices].max(dim=1).values
            top_margin = top_values - top_competitors
            probs = scores[top_indices].clamp_min(1e-8)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
            entropy = -(probs * probs.log()).sum(dim=1) / math.log(float(scores.shape[1]))
        else:
            top_margin = top_values
            entropy = torch.zeros_like(top_values)
        result[str(category)] = {
            "max_score": _round4(float(column.max().item())),
            "mean_score": _round4(float(column.mean().item())),
            "top1pct_mean_score": _round4(float(top_values.mean().item())),
            "top1pct_min_score": _round4(float(top_values.min().item())),
            "top1pct_mean_margin": _round4(float(top_margin.mean().item())),
            "top1pct_mean_entropy": _round4(float(entropy.mean().item())),
        }
    return result


def _zero_prediction_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if int(row.get("pred_pixels", 0)) == 0) / len(rows)


def _summarize_rows(rows: list[dict[str, Any]], *, score_field: str) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "mean_iou": 0.0,
            "acc025": 0.0,
            "zero_prediction_rate": 0.0,
            "mean_score": 0.0,
            "score_min": 0.0,
            "score_max": 0.0,
            "mean_gt_pixels": 0.0,
            "mean_selected_gaussians": 0.0,
        }
    scores = [float(row[score_field]) for row in rows]
    ious = [float(row.get("iou", 0.0)) for row in rows]
    return {
        "n": len(rows),
        "mean_iou": _round4(_mean(ious)),
        "acc025": _round4(sum(1 for value in ious if value >= 0.25) / len(ious)),
        "zero_prediction_rate": _round4(_zero_prediction_rate(rows)),
        "mean_score": _round4(_mean(scores)),
        "score_min": _round4(min(scores)),
        "score_max": _round4(max(scores)),
        "mean_gt_pixels": _round4(_mean([float(row.get("gt_pixels", 0.0)) for row in rows])),
        "mean_selected_gaussians": _round4(
            _mean([float(row.get("selected_gaussians", 0.0)) for row in rows])
        ),
    }


def _confidence_buckets(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
) -> dict[str, dict[str, Any]]:
    if not rows:
        return {name: _summarize_rows([], score_field=score_field) for name in ("low", "mid", "high")}
    ordered = sorted(rows, key=lambda row: (float(row[score_field]), row["scene"], row["category"]))
    buckets = {"low": [], "mid": [], "high": []}
    n = len(ordered)
    for idx, row in enumerate(ordered):
        if idx < n / 3:
            buckets["low"].append(row)
        elif idx < (2 * n) / 3:
            buckets["mid"].append(row)
        else:
            buckets["high"].append(row)
    return {
        name: _summarize_rows(bucket_rows, score_field=score_field)
        for name, bucket_rows in buckets.items()
    }


def _text_ambiguity_buckets(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
) -> dict[str, dict[str, Any]]:
    if not rows:
        return {
            name: _summarize_rows([], score_field=score_field)
            for name in ("ambiguous", "mixed", "distinct")
        }
    ordered = sorted(rows, key=lambda row: (float(row[score_field]), row["scene"], row["category"]))
    buckets = {"ambiguous": [], "mixed": [], "distinct": []}
    n = len(ordered)
    for idx, row in enumerate(ordered):
        if idx < n / 3:
            buckets["ambiguous"].append(row)
        elif idx < (2 * n) / 3:
            buckets["mixed"].append(row)
        else:
            buckets["distinct"].append(row)
    return {
        name: _summarize_rows(bucket_rows, score_field=score_field)
        for name, bucket_rows in buckets.items()
    }


def _low_confidence_failures(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: (float(row[score_field]), float(row.get("iou", 0.0))))
    cutoff = max(1, math.ceil(len(ordered) / 3))
    failures = [
        row
        for row in ordered[:cutoff]
        if float(row.get("iou", 0.0)) < 0.25 or int(row.get("pred_pixels", 0)) == 0
    ]
    return [
        {
            "scene": row["scene"],
            "frame": row.get("frame", ""),
            "category": row["category"],
            "iou": _round4(float(row.get("iou", 0.0))),
            "pred_pixels": int(row.get("pred_pixels", 0)),
            "gt_pixels": int(row.get("gt_pixels", 0)),
            score_field: _round4(float(row[score_field])),
        }
        for row in failures[:limit]
    ]


def build_summary(
    result_root: str | Path = DEFAULT_RESULT_ROOT,
    *,
    selection: str = "thr0p25",
    top_score_ratio: float = 0.01,
) -> dict[str, Any]:
    root = Path(result_root)
    paths = _result_paths(root)
    if not paths:
        raise FileNotFoundError(f"No Direct3D result JSON files under {root}")

    scene_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scene_payload = payload.get("scene", {})
        scene_name = str(scene_payload.get("scene", path.parent.name))
        results = scene_payload.get("results", {})
        if selection not in results:
            raise KeyError(f"Selection {selection!r} not found in {path}")
        metrics = results[selection]
        categories = [str(category) for category in scene_payload.get("categories", [])]
        score_cache = scene_payload.get("score_cache", {})
        cache_path = _resolve_path(str(score_cache.get("path", "")), base=REPO_ROOT)
        score_stats = _load_score_stats(cache_path, categories, top_score_ratio=top_score_ratio)
        details = list(metrics.get("query_details", []))
        registration = scene_payload.get("registration", {})

        per_category = metrics.get("per_category", {})
        scene_category_rows: list[dict[str, Any]] = []
        for category, cat_metrics in per_category.items():
            stats = score_stats.get(str(category), {})
            row = {
                "scene": scene_name,
                "category": str(category),
                "miou": _round4(float(cat_metrics.get("miou", 0.0))),
                "n": int(cat_metrics.get("n", 0)),
                "selected_gaussians": int(cat_metrics.get("selected_gaussians", 0)),
                **stats,
            }
            scene_category_rows.append(row)
            category_rows.append(row)

        for detail in details:
            category = str(detail.get("category", ""))
            stats = score_stats.get(category, {})
            query_rows.append(
                {
                    "scene": scene_name,
                    "frame": str(detail.get("frame", "")),
                    "frame_id": int(detail.get("frame_id", -1)),
                    "category": category,
                    "iou": _round4(float(detail.get("iou", 0.0))),
                    "pred_pixels": int(detail.get("pred_pixels", 0)),
                    "gt_pixels": int(detail.get("gt_pixels", 0)),
                    "selected_gaussians": int(detail.get("selected_gaussians", 0)),
                    **stats,
                }
            )

        scene_rows.append(
            {
                "scene": scene_name,
                "miou": _round4(float(metrics.get("miou", 0.0))),
                "acc025": _round4(float(metrics.get("acc025", 0.0))),
                "n": int(metrics.get("n", len(details))),
                "mean_valid_views": _round4(float(registration.get("mean_valid_views", 0.0))),
                "max_valid_views": _round4(float(registration.get("max_valid_views", 0.0))),
                "registered_fraction": _round4(float(registration.get("registered_fraction", 0.0))),
                "registered_gaussians": int(registration.get("registered_gaussians", 0)),
                "total_gaussians": int(registration.get("total_gaussians", 0)),
                "num_registration_frames": int(registration.get("num_frames", 0)),
                "zero_prediction_rate": _round4(_zero_prediction_rate(details)),
                "mean_top1pct_score": _round4(
                    _mean([float(row["top1pct_mean_score"]) for row in scene_category_rows])
                ),
                "mean_top1pct_margin": _round4(
                    _mean([float(row["top1pct_mean_margin"]) for row in scene_category_rows])
                ),
                "score_cache": str(cache_path),
                "source": str(path),
            }
        )

    scene_rows.sort(key=lambda row: row["scene"])
    query_rows.sort(key=lambda row: (row["scene"], row["frame"], row["category"]))
    category_rows.sort(key=lambda row: (row["scene"], row["category"]))
    score_field = "top1pct_mean_score"
    margin_field = "top1pct_mean_margin"
    return {
        "selection": selection,
        "source_root": str(root),
        "top_score_ratio": float(top_score_ratio),
        "scene_rows": scene_rows,
        "category_rows": category_rows,
        "query_rows": query_rows,
        "scene_correlations": {
            "mean_valid_views_vs_miou": _pearson(
                [float(row["mean_valid_views"]) for row in scene_rows],
                [float(row["miou"]) for row in scene_rows],
            ),
            "registered_fraction_vs_miou": _pearson(
                [float(row["registered_fraction"]) for row in scene_rows],
                [float(row["miou"]) for row in scene_rows],
            ),
            "mean_top1pct_score_vs_miou": _pearson(
                [float(row["mean_top1pct_score"]) for row in scene_rows],
                [float(row["miou"]) for row in scene_rows],
            ),
            "mean_top1pct_margin_vs_miou": _pearson(
                [float(row["mean_top1pct_margin"]) for row in scene_rows],
                [float(row["miou"]) for row in scene_rows],
            ),
        },
        "confidence_buckets": _confidence_buckets(query_rows, score_field=score_field),
        "text_ambiguity_buckets": _text_ambiguity_buckets(query_rows, score_field=margin_field),
        "low_confidence_failures": _low_confidence_failures(query_rows, score_field=score_field),
        "ambiguous_failures": _low_confidence_failures(query_rows, score_field=margin_field),
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Direct3D Confidence and Coverage Analysis",
        "",
        f"- Source root: `{summary['source_root']}`",
        f"- Selection: `{summary['selection']}`",
        f"- Teacher-score proxy: mean of top {float(summary['top_score_ratio']) * 100:.2f}% primitive scores per category",
        "",
        "## Scene View-Coverage",
        "",
        "| Scene | Mean valid views | Registered fraction | mIoU | Acc@0.25 | Zero-pred rate | Mean top-score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["scene_rows"]:
        lines.append(
            "| {scene} | {views:.4f} | {frac:.4f} | {miou:.4f} | {acc:.4f} | {zero:.4f} | {score:.4f} |".format(
                scene=row["scene"],
                views=float(row["mean_valid_views"]),
                frac=float(row["registered_fraction"]),
                miou=float(row["miou"]),
                acc=float(row["acc025"]),
                zero=float(row["zero_prediction_rate"]),
                score=float(row["mean_top1pct_score"]),
            )
        )
    corr = summary.get("scene_correlations", {})
    lines.extend(
        [
            "",
            "## Scene Correlations",
            "",
            "| Pair | Pearson r |",
            "|---|---:|",
            f"| mean valid views vs mIoU | {float(corr.get('mean_valid_views_vs_miou', 0.0)):.4f} |",
            f"| registered fraction vs mIoU | {float(corr.get('registered_fraction_vs_miou', 0.0)):.4f} |",
            f"| mean top-score vs mIoU | {float(corr.get('mean_top1pct_score_vs_miou', 0.0)):.4f} |",
            f"| mean text margin vs mIoU | {float(corr.get('mean_top1pct_margin_vs_miou', 0.0)):.4f} |",
            "",
            "## Teacher-Score Confidence Buckets",
            "",
            "| Bucket | Queries | Score range | Mean IoU | Acc@0.25 | Zero-pred rate | Mean GT pixels |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("low", "mid", "high"):
        bucket = summary["confidence_buckets"].get(name, {})
        lines.append(
            "| {name} | {n} | {lo:.4f}-{hi:.4f} | {iou:.4f} | {acc:.4f} | {zero:.4f} | {gt:.1f} |".format(
                name=name,
                n=int(bucket.get("n", 0)),
                lo=float(bucket.get("score_min", 0.0)),
                hi=float(bucket.get("score_max", 0.0)),
                iou=float(bucket.get("mean_iou", 0.0)),
                acc=float(bucket.get("acc025", 0.0)),
                zero=float(bucket.get("zero_prediction_rate", 0.0)),
                gt=float(bucket.get("mean_gt_pixels", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Text-Ambiguity Buckets",
            "",
            "| Bucket | Queries | Margin range | Mean IoU | Acc@0.25 | Zero-pred rate | Mean GT pixels |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("ambiguous", "mixed", "distinct"):
        bucket = summary["text_ambiguity_buckets"].get(name, {})
        lines.append(
            "| {name} | {n} | {lo:.4f}-{hi:.4f} | {iou:.4f} | {acc:.4f} | {zero:.4f} | {gt:.1f} |".format(
                name=name,
                n=int(bucket.get("n", 0)),
                lo=float(bucket.get("score_min", 0.0)),
                hi=float(bucket.get("score_max", 0.0)),
                iou=float(bucket.get("mean_iou", 0.0)),
                acc=float(bucket.get("acc025", 0.0)),
                zero=float(bucket.get("zero_prediction_rate", 0.0)),
                gt=float(bucket.get("mean_gt_pixels", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Low-Confidence Failure Examples",
            "",
            "| Scene | Frame | Category | IoU | Pred px | GT px | Top-score |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    failures = summary.get("low_confidence_failures", [])
    if failures:
        for row in failures:
            lines.append(
                "| {scene} | {frame} | {cat} | {iou:.4f} | {pred} | {gt} | {score:.4f} |".format(
                    scene=row["scene"],
                    frame=row["frame"],
                    cat=row["category"],
                    iou=float(row["iou"]),
                    pred=int(row["pred_pixels"]),
                    gt=int(row["gt_pixels"]),
                    score=float(row["top1pct_mean_score"]),
                )
            )
    else:
        lines.append("| none |  |  | 0.0000 | 0 | 0 | 0.0000 |")
    lines.extend(
        [
            "",
            "## Ambiguous-Text Failure Examples",
            "",
            "| Scene | Frame | Category | IoU | Pred px | GT px | Text margin |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    ambiguous_failures = summary.get("ambiguous_failures", [])
    if ambiguous_failures:
        for row in ambiguous_failures:
            lines.append(
                "| {scene} | {frame} | {cat} | {iou:.4f} | {pred} | {gt} | {score:.4f} |".format(
                    scene=row["scene"],
                    frame=row["frame"],
                    cat=row["category"],
                    iou=float(row["iou"]),
                    pred=int(row["pred_pixels"]),
                    gt=int(row["gt_pixels"]),
                    score=float(row["top1pct_mean_margin"]),
                )
            )
    else:
        lines.append("| none |  |  | 0.0000 | 0 | 0 | 0.0000 |")
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- View coverage is scene-level registration support from the VPR score cache; it is a GT-free mechanism proxy, not a causal ablation.",
            "- Teacher-score confidence is computed only from primitive text-score caches and is independent of LERF masks.",
            "- Text ambiguity uses the margin between a query score and the strongest competing scene category on the same top-scoring primitives.",
            "- Query rows are evaluation instances, so repeated categories across frames receive the same category-level score proxy.",
            "",
        ]
    )
    return "\n".join(lines)


def build_latex_table(summary: dict[str, Any]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Direct3D mechanism audit using GT-free VPR view coverage, primitive teacher-score confidence, and text-margin ambiguity. Scene rows report mean valid registered views; teacher-bucket rows report mean top-1\\% primitive text score; text-margin rows report score margin against the strongest competing category.}",
        "\\label{tab:direct3d_confidence_coverage}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Group & Item & Support & Coverage/score & mIoU & Acc@0.25 \\\\",
        "\\midrule",
    ]
    for row in summary.get("scene_rows", []):
        lines.append(
            "Scene & {scene} & {n} & {views:.4f} & {miou:.4f} & {acc:.4f} \\\\".format(
                scene=_latex_escape(str(row["scene"])),
                n=int(row.get("n", 0)),
                views=float(row.get("mean_valid_views", 0.0)),
                miou=float(row.get("miou", 0.0)),
                acc=float(row.get("acc025", 0.0)),
            )
        )
    lines.append("\\midrule")
    for name in ("low", "mid", "high"):
        bucket = summary.get("confidence_buckets", {}).get(name)
        if not bucket:
            continue
        lines.append(
            "Teacher bucket & {name} & {n} & {score:.4f} & {miou:.4f} & {acc:.4f} \\\\".format(
                name=_latex_escape(name),
                n=int(bucket.get("n", 0)),
                score=float(bucket.get("mean_score", 0.0)),
                miou=float(bucket.get("mean_iou", 0.0)),
                acc=float(bucket.get("acc025", 0.0)),
            )
        )
    for name in ("ambiguous", "mixed", "distinct"):
        bucket = summary.get("text_ambiguity_buckets", {}).get(name)
        if not bucket:
            continue
        lines.append(
            "Text margin & {name} & {n} & {score:.4f} & {miou:.4f} & {acc:.4f} \\\\".format(
                name=_latex_escape(name),
                n=int(bucket.get("n", 0)),
                score=float(bucket.get("mean_score", 0.0)),
                miou=float(bucket.get("mean_iou", 0.0)),
                acc=float(bucket.get("acc025", 0.0)),
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
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
    parser.add_argument("--result_root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--selection", default="thr0p25")
    parser.add_argument("--top_score_ratio", type=float, default=0.01)
    parser.add_argument("--output_md", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--output_json", default=str(DEFAULT_JSON))
    parser.add_argument("--output_tex", default=str(DEFAULT_LATEX))
    args = parser.parse_args(argv)

    summary = build_summary(
        args.result_root,
        selection=args.selection,
        top_score_ratio=float(args.top_score_ratio),
    )
    paths = write_outputs(summary, args.output_md, args.output_json, args.output_tex)
    print(f"Wrote {paths['markdown']}")
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['latex']}")
    return paths


if __name__ == "__main__":
    main()
