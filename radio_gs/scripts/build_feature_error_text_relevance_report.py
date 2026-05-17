"""Build feature-reconstruction error vs text-relevance error evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RENDERED = (
    REPO_ROOT / "output" / "radio_gs" / "reports" / "lerf_rendered_grounding_paper_ckpt_threshold_sweep.json"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT / "output" / "radio_gs" / "reports" / "feature_error_text_relevance_report.md"
)
DEFAULT_JSON = (
    REPO_ROOT / "output" / "radio_gs" / "reports" / "feature_error_text_relevance_report.json"
)
DEFAULT_LATEX = REPO_ROOT / "paper" / "feature_error_text_relevance_table.tex"


DEFAULT_LOG_PATHS = {
    "figurines": REPO_ROOT / "output" / "radio_gs" / "lerf_figurines_v14_fdh_ws240_240ep" / "logs" / "training.log",
    "ramen": REPO_ROOT / "output" / "radio_gs" / "lerf_ramen_v14_fdh_ws240_240ep_seed7" / "logs" / "training.log",
    "teatime": REPO_ROOT / "output" / "radio_gs" / "lerf_teatime_v14_fdh_ws240_240ep_seed7" / "logs" / "training.log",
    "waldo_kitchen": REPO_ROOT / "output" / "radio_gs" / "lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7" / "logs" / "training.log",
}


def _round4(value: float) -> float:
    return round(float(value), 4)


def _scene_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _display_scene(key: str) -> str:
    return {
        "figurines": "Figurines",
        "ramen": "Ramen",
        "teatime": "Teatime",
        "waldo_kitchen": "Waldo Kitchen",
    }.get(key, key)


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
    mx = _mean(xs)
    my = _mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return _round4(cov / math.sqrt(vx * vy))


def parse_best_val_cosine(path: str | Path) -> dict[str, Any]:
    pattern = re.compile(
        r"\[Val E(?P<epoch>\d+)\].*?cos_decoded=(?P<cos>[0-9.]+).*?psnr=(?P<psnr>[0-9.]+)"
    )
    best: dict[str, Any] | None = None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        row = {
            "epoch": int(match.group("epoch")),
            "cos_decoded": _round4(float(match.group("cos"))),
            "psnr": _round4(float(match.group("psnr"))),
        }
        if best is None or float(row["cos_decoded"]) > float(best["cos_decoded"]):
            best = row
    if best is None:
        raise ValueError(f"No validation cos_decoded entries found in {path}")
    return best


def parse_rendered_rows(path: str | Path, *, variant: str) -> dict[str, dict[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    variants = payload.get("variants", {})
    if variant not in variants:
        raise KeyError(f"Rendered variant {variant!r} not found in {path}")
    rows: dict[str, dict[str, float]] = {}
    for row in variants[variant].get("rows", []):
        scene = _scene_key(str(row.get("scene", "")))
        rows[scene] = {
            "rendered_miou": _round4(float(row.get("miou", 0.0))),
            "rendered_locacc": _round4(float(row.get("loc", row.get("loc_acc", 0.0)))),
        }
    return rows


def build_summary(
    rendered_path: str | Path = DEFAULT_RENDERED,
    *,
    log_paths: dict[str, str | Path] | None = None,
    rendered_variant: str = "0.60",
) -> dict[str, Any]:
    logs = log_paths or DEFAULT_LOG_PATHS
    rendered = parse_rendered_rows(rendered_path, variant=rendered_variant)
    rows: list[dict[str, Any]] = []
    for raw_scene, log_path in sorted(logs.items()):
        scene = _scene_key(raw_scene)
        if scene not in rendered:
            continue
        best = parse_best_val_cosine(log_path)
        cos = float(best["cos_decoded"])
        rendered_row = rendered[scene]
        miou = float(rendered_row["rendered_miou"])
        locacc = float(rendered_row["rendered_locacc"])
        rows.append(
            {
                "scene": _display_scene(scene),
                "log_path": str(log_path),
                "best_epoch": int(best["epoch"]),
                "best_val_cos_decoded": _round4(cos),
                "feature_error": _round4(1.0 - cos),
                "psnr": _round4(float(best["psnr"])),
                "rendered_miou": _round4(miou),
                "miou_error": _round4(1.0 - miou),
                "rendered_locacc": _round4(locacc),
                "loc_error": _round4(1.0 - locacc),
            }
        )
    feature_errors = [float(row["feature_error"]) for row in rows]
    miou_errors = [float(row["miou_error"]) for row in rows]
    loc_errors = [float(row["loc_error"]) for row in rows]
    return {
        "rendered_source": str(rendered_path),
        "rendered_variant": rendered_variant,
        "rows": rows,
        "correlations": {
            "feature_error_vs_miou_error": _pearson(feature_errors, miou_errors),
            "feature_error_vs_loc_error": _pearson(feature_errors, loc_errors),
        },
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Feature Error vs Text Relevance Error",
        "",
        f"- Rendered source: `{summary['rendered_source']}` at variant `{summary['rendered_variant']}`",
        "- Feature error proxy: `1 - best validation cos_decoded` from the frozen scene training log.",
        "- Text relevance error proxies: `1 - rendered mIoU` and `1 - rendered LocAcc` under the frozen LERF evaluator.",
        "",
        "| Scene | Best val cosine | Feature error | Rendered mIoU | mIoU error | Rendered LocAcc | Loc error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {scene} | {cos:.4f} | {ferr:.4f} | {miou:.4f} | {miouerr:.4f} | {loc:.4f} | {locerr:.4f} |".format(
                scene=row["scene"],
                cos=float(row["best_val_cos_decoded"]),
                ferr=float(row["feature_error"]),
                miou=float(row["rendered_miou"]),
                miouerr=float(row["miou_error"]),
                loc=float(row["rendered_locacc"]),
                locerr=float(row["loc_error"]),
            )
        )
    corr = summary.get("correlations", {})
    lines.extend(
        [
            "",
            "## Correlations",
            "",
            "| Pair | Pearson r |",
            "|---|---:|",
            f"| feature error vs mIoU error | {float(corr.get('feature_error_vs_miou_error', 0.0)):.4f} |",
            f"| feature error vs LocAcc error | {float(corr.get('feature_error_vs_loc_error', 0.0)):.4f} |",
            "",
            "## Interpretation",
            "",
            "- This is a scene-level mechanism audit, not a per-query causal proof.",
            "- If correlations are weak, the paper should avoid claiming that lower global reconstruction error alone explains text grounding.",
            "- Stronger future evidence would require per-view or per-query feature residuals aligned with text heatmap failures.",
            "",
        ]
    )
    return "\n".join(lines)


def build_latex_table(summary: dict[str, Any]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Feature reconstruction error versus text relevance error. Feature error is $1-$ best validation decoded cosine from the scene training log; text relevance errors are derived from the frozen rendered-grounding readout.}",
        "\\label{tab:feature_error_text_relevance}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Scene & Feature err. & mIoU err. & LocAcc err. & Best cosine \\\\",
        "\\midrule",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "{scene} & {ferr:.4f} & {miouerr:.4f} & {locerr:.4f} & {cos:.4f} \\\\".format(
                scene=_latex_escape(str(row["scene"])),
                ferr=float(row["feature_error"]),
                miouerr=float(row["miou_error"]),
                locerr=float(row["loc_error"]),
                cos=float(row["best_val_cos_decoded"]),
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
    parser.add_argument("--rendered", default=str(DEFAULT_RENDERED))
    parser.add_argument("--rendered_variant", default="0.60")
    parser.add_argument("--output_md", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--output_json", default=str(DEFAULT_JSON))
    parser.add_argument("--output_tex", default=str(DEFAULT_LATEX))
    args = parser.parse_args(argv)

    summary = build_summary(args.rendered, rendered_variant=args.rendered_variant)
    paths = write_outputs(summary, args.output_md, args.output_json, args.output_tex)
    print(f"Wrote {paths['markdown']}")
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['latex']}")
    return paths


if __name__ == "__main__":
    main()
