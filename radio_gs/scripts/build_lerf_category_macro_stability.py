"""Build category-macro stability diagnostics for LERF result files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


def _round4(value: float) -> float:
    return round(float(value), 4)


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(mean(values)) if values else 0.0


def _safe_std(values: Iterable[float]) -> float:
    values = list(values)
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_scene_rows(paths: list[str | Path], *, readout: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scene_rows: list[dict[str, Any]] = []
    scene_category_rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = _load_json(path)
        scenes = payload.get("scenes", {})
        if not isinstance(scenes, dict):
            continue
        for scene, scene_payload in scenes.items():
            if not isinstance(scene_payload, dict) or readout not in scene_payload:
                continue
            metrics = scene_payload[readout]
            if not isinstance(metrics, dict):
                continue
            n_iou = int(metrics.get("n_iou_samples", metrics.get("loc_total", 0)) or 0)
            n_loc = int(metrics.get("loc_total", n_iou) or 0)
            scene_rows.append(
                {
                    "scene": str(scene),
                    "source": str(path),
                    "loc_acc": float(metrics.get("loc_acc", 0.0)),
                    "miou": float(metrics.get("miou", 0.0)),
                    "loc_total": n_loc,
                    "n_iou_samples": n_iou,
                }
            )
            per_category = metrics.get("per_category", {})
            if not isinstance(per_category, dict):
                continue
            for category, cat_metrics in per_category.items():
                if not isinstance(cat_metrics, dict):
                    continue
                scene_category_rows.append(
                    {
                        "scene": str(scene),
                        "category": str(category),
                        "loc_acc": float(cat_metrics.get("loc_acc", 0.0)),
                        "miou": float(cat_metrics.get("miou", 0.0)),
                        "n_samples": int(cat_metrics.get("n_samples", 0) or 0),
                    }
                )
    return scene_rows, scene_category_rows


def _weighted_mean(rows: list[dict[str, Any]], metric: str, weight_key: str) -> float:
    denom = sum(max(0, int(row.get(weight_key, 0) or 0)) for row in rows)
    if denom <= 0:
        return 0.0
    return sum(float(row.get(metric, 0.0)) * max(0, int(row.get(weight_key, 0) or 0)) for row in rows) / denom


def _aggregate_categories(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["category"]), []).append(row)
    category_rows: list[dict[str, Any]] = []
    for category, cat_rows in grouped.items():
        n = sum(int(row.get("n_samples", 0) or 0) for row in cat_rows)
        miou = _weighted_mean(cat_rows, "miou", "n_samples") if n > 0 else _safe_mean(row["miou"] for row in cat_rows)
        loc = _weighted_mean(cat_rows, "loc_acc", "n_samples") if n > 0 else _safe_mean(row["loc_acc"] for row in cat_rows)
        category_rows.append(
            {
                "category": category,
                "miou": miou,
                "loc_acc": loc,
                "n_samples": n,
                "scene_count": len({str(row["scene"]) for row in cat_rows}),
            }
        )
    return sorted(category_rows, key=lambda row: row["category"])


def _bootstrap_scene_macro(
    scene_rows: list[dict[str, Any]],
    *,
    metric: str,
    iters: int,
    seed: int,
) -> list[float]:
    if not scene_rows or iters <= 0:
        value = _safe_mean(row.get(metric, 0.0) for row in scene_rows)
        return [_round4(value), _round4(value)]
    rng = random.Random(seed)
    values = []
    for _ in range(int(iters)):
        sample = [rng.choice(scene_rows) for _ in scene_rows]
        values.append(_safe_mean(float(row.get(metric, 0.0)) for row in sample))
    values.sort()
    lo_idx = int(0.025 * (len(values) - 1))
    hi_idx = int(0.975 * (len(values) - 1))
    return [_round4(values[lo_idx]), _round4(values[hi_idx])]


def build_summary(
    paths: list[str | Path],
    *,
    readout: str = "rendered",
    bootstrap_iters: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    scene_rows, scene_category_rows = _extract_scene_rows(paths, readout=readout)
    category_rows = _aggregate_categories(scene_category_rows)
    scene_mious = [float(row["miou"]) for row in scene_rows]
    scene_locs = [float(row["loc_acc"]) for row in scene_rows]
    scene_category_mious = [float(row["miou"]) for row in scene_category_rows]
    scene_category_locs = [float(row["loc_acc"]) for row in scene_category_rows]
    sample_weighted_miou = _weighted_mean(scene_rows, "miou", "n_iou_samples")
    sample_weighted_loc = _weighted_mean(scene_rows, "loc_acc", "loc_total")
    scene_category_macro_miou = _safe_mean(scene_category_mious)
    scene_category_macro_loc = _safe_mean(scene_category_locs)
    worst_categories = sorted(
        category_rows,
        key=lambda row: (float(row["miou"]), float(row["loc_acc"]), -int(row["n_samples"])),
    )[:10]
    return {
        "readout": readout,
        "sources": [str(Path(path)) for path in paths],
        "scene_count": len(scene_rows),
        "scene_rows": [
            {
                **row,
                "loc_acc": _round4(row["loc_acc"]),
                "miou": _round4(row["miou"]),
            }
            for row in scene_rows
        ],
        "scene_macro": {
            "loc_acc": _round4(_safe_mean(scene_locs)),
            "miou": _round4(_safe_mean(scene_mious)),
        },
        "sample_weighted": {
            "loc_acc": _round4(sample_weighted_loc),
            "miou": _round4(sample_weighted_miou),
        },
        "scene_category_macro": {
            "loc_acc": _round4(scene_category_macro_loc),
            "miou": _round4(scene_category_macro_miou),
        },
        "stability": {
            "scene_miou_std": _round4(_safe_std(scene_mious)),
            "scene_loc_acc_std": _round4(_safe_std(scene_locs)),
            "scene_category_miou_std": _round4(_safe_std(scene_category_mious)),
            "scene_category_loc_acc_std": _round4(_safe_std(scene_category_locs)),
            "category_minus_sample_miou": _round4(scene_category_macro_miou - sample_weighted_miou),
            "category_minus_scene_miou": _round4(scene_category_macro_miou - _safe_mean(scene_mious)),
        },
        "bootstrap": {
            "iters": int(bootstrap_iters),
            "seed": int(seed),
            "scene_macro_miou_ci95": _bootstrap_scene_macro(
                scene_rows,
                metric="miou",
                iters=bootstrap_iters,
                seed=seed,
            ),
            "scene_macro_loc_acc_ci95": _bootstrap_scene_macro(
                scene_rows,
                metric="loc_acc",
                iters=bootstrap_iters,
                seed=seed + 17,
            ),
        },
        "category_rows": [
            {
                **row,
                "loc_acc": _round4(row["loc_acc"]),
                "miou": _round4(row["miou"]),
            }
            for row in category_rows
        ],
        "worst_categories": [
            {
                **row,
                "loc_acc": _round4(row["loc_acc"]),
                "miou": _round4(row["miou"]),
            }
            for row in worst_categories
        ],
    }


def _format_metric(value: Any) -> str:
    return f"{float(value):.4f}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LERF Category-Macro Stability",
        "",
        f"Readout: `{summary.get('readout', '')}`",
        "",
        "| Aggregation | LocAcc | mIoU |",
        "|---|---:|---:|",
    ]
    for key, label in [
        ("scene_macro", "Scene macro"),
        ("sample_weighted", "Sample weighted"),
        ("scene_category_macro", "Scene-category macro"),
    ]:
        row = summary.get(key, {})
        lines.append(
            f"| {label} | {_format_metric(row.get('loc_acc', 0.0))} | {_format_metric(row.get('miou', 0.0))} |"
        )
    stability = summary.get("stability", {})
    bootstrap = summary.get("bootstrap", {})
    lines.extend(
        [
            "",
            "## Stability",
            "",
            f"- Scene mIoU std: `{_format_metric(stability.get('scene_miou_std', 0.0))}`",
            f"- Scene-category mIoU std: `{_format_metric(stability.get('scene_category_miou_std', 0.0))}`",
            f"- Category minus sample-weighted mIoU: `{_format_metric(stability.get('category_minus_sample_miou', 0.0))}`",
            f"- Scene macro mIoU 95% bootstrap CI: `{bootstrap.get('scene_macro_miou_ci95', [0.0, 0.0])}`",
            "",
            "## Worst Categories",
            "",
            "| Category | LocAcc | mIoU | Samples | Scenes |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("worst_categories", []):
        lines.append(
            "| {category} | {loc} | {miou} | {samples} | {scenes} |".format(
                category=str(row.get("category", "")),
                loc=_format_metric(row.get("loc_acc", 0.0)),
                miou=_format_metric(row.get("miou", 0.0)),
                samples=int(row.get("n_samples", 0)),
                scenes=int(row.get("scene_count", 0)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(
    summary: dict[str, Any],
    *,
    output_json: str | Path,
    output_md: str | Path,
) -> dict[str, Path]:
    output_json = Path(output_json)
    output_md = Path(output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    return {"json": output_json, "markdown": output_md}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="LERF result JSON files")
    parser.add_argument("--readout", default="rendered", help="Metric branch to read")
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_json",
        default="output/radio_gs/reports/lerf_category_macro_stability.json",
    )
    parser.add_argument(
        "--output_md",
        default="output/radio_gs/reports/lerf_category_macro_stability.md",
    )
    args = parser.parse_args()

    summary = build_summary(
        [Path(path) for path in args.paths],
        readout=args.readout,
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
    )
    paths = write_reports(
        summary,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['markdown']}")


if __name__ == "__main__":
    main()
