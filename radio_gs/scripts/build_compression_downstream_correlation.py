"""Build compression-ratio vs downstream-mIoU correlation evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE = REPO_ROOT / "output" / "radio_gs" / "reports" / "storage_footprint_report.md"
DEFAULT_RENDERED = (
    REPO_ROOT / "output" / "radio_gs" / "reports" / "lerf_rendered_grounding_paper_ckpt_threshold_sweep.json"
)
DEFAULT_DIRECT3D_ROOT = (
    REPO_ROOT / "output" / "radio_gs" / "lerf_direct_3d_selection_threshold_grabcut_20260515"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT / "output" / "radio_gs" / "reports" / "compression_downstream_correlation.md"
)
DEFAULT_JSON = (
    REPO_ROOT / "output" / "radio_gs" / "reports" / "compression_downstream_correlation.json"
)
DEFAULT_LATEX = REPO_ROOT / "paper" / "compression_downstream_correlation_table.tex"


def _round4(value: float) -> float:
    return round(float(value), 4)


def _scene_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


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


def parse_storage_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or " MiB " not in stripped or "x" not in stripped:
            continue
        columns = [part.strip() for part in stripped.strip("|").split("|")]
        if len(columns) < 5 or columns[0] == "Scene":
            continue
        scene = columns[0]
        saving_match = re.search(r"([0-9.]+)x", columns[4])
        direct_match = re.search(r"([0-9.]+)\s+MiB", columns[2])
        compact_match = re.search(r"([0-9.]+)\s+MiB", columns[3])
        if not saving_match:
            continue
        rows[_scene_key(scene)] = {
            "scene": scene,
            "saving_ratio": _round4(float(saving_match.group(1))),
            "direct_mib": _round4(float(direct_match.group(1))) if direct_match else 0.0,
            "compact_mib": _round4(float(compact_match.group(1))) if compact_match else 0.0,
        }
    return rows


def parse_rendered_rows(path: str | Path, *, variant: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    variants = payload.get("variants", {})
    if variant not in variants:
        raise KeyError(f"Rendered variant {variant!r} not found in {path}")
    rows: dict[str, dict[str, Any]] = {}
    for row in variants[variant].get("rows", []):
        scene = str(row.get("scene", ""))
        rows[_scene_key(scene)] = {
            "rendered_miou": _round4(float(row.get("miou", 0.0))),
            "rendered_locacc": _round4(float(row.get("loc", row.get("loc_acc", 0.0)))),
        }
    return rows


def parse_direct3d_rows(root: str | Path, *, selection: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(root).glob("*/lerf_direct_3d_selection_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scene_payload = payload.get("scene", {})
        scene = str(scene_payload.get("scene", path.parent.name))
        metrics = scene_payload.get("results", {}).get(selection)
        if not metrics:
            continue
        rows[_scene_key(scene)] = {
            "direct3d_miou": _round4(float(metrics.get("miou", 0.0))),
            "direct3d_acc025": _round4(float(metrics.get("acc025", 0.0))),
        }
    return rows


def build_summary(
    storage_path: str | Path = DEFAULT_STORAGE,
    rendered_path: str | Path = DEFAULT_RENDERED,
    *,
    direct3d_root: str | Path = DEFAULT_DIRECT3D_ROOT,
    rendered_variant: str = "0.60",
    selection: str = "thr0p25",
) -> dict[str, Any]:
    storage = parse_storage_rows(storage_path)
    rendered = parse_rendered_rows(rendered_path, variant=rendered_variant)
    direct3d = parse_direct3d_rows(direct3d_root, selection=selection)

    rows: list[dict[str, Any]] = []
    for key in sorted(storage):
        if key not in rendered or key not in direct3d:
            continue
        rows.append({**storage[key], **rendered[key], **direct3d[key]})

    savings = [float(row["saving_ratio"]) for row in rows]
    rendered_miou = [float(row["rendered_miou"]) for row in rows]
    direct3d_miou = [float(row["direct3d_miou"]) for row in rows]
    return {
        "storage_source": str(storage_path),
        "rendered_source": str(rendered_path),
        "direct3d_root": str(direct3d_root),
        "rendered_variant": rendered_variant,
        "selection": selection,
        "rows": rows,
        "correlations": {
            "saving_vs_rendered_miou": _pearson(savings, rendered_miou),
            "saving_vs_direct3d_miou": _pearson(savings, direct3d_miou),
        },
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Compression vs Downstream Correlation",
        "",
        f"- Storage source: `{summary['storage_source']}`",
        f"- Rendered source: `{summary['rendered_source']}` at variant `{summary['rendered_variant']}`",
        f"- Direct3D root: `{summary['direct3d_root']}` with selection `{summary['selection']}`",
        "",
        "| Scene | Saving ratio | Rendered mIoU | Direct3D mIoU | Direct3D Acc@0.25 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {scene} | {saving:.2f}x | {rendered:.4f} | {direct:.4f} | {acc:.4f} |".format(
                scene=row["scene"],
                saving=float(row["saving_ratio"]),
                rendered=float(row["rendered_miou"]),
                direct=float(row["direct3d_miou"]),
                acc=float(row["direct3d_acc025"]),
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
            f"| saving ratio vs rendered mIoU | {float(corr.get('saving_vs_rendered_miou', 0.0)):.4f} |",
            f"| saving ratio vs Direct3D mIoU | {float(corr.get('saving_vs_direct3d_miou', 0.0)):.4f} |",
            "",
            "## Interpretation",
            "",
            "- The saving ratio is a compact-storage accounting ratio, not an accuracy predictor.",
            "- A weak or negative correlation means higher compression on a larger scene should not be framed as causing stronger downstream mIoU.",
            "- This table supports separating the compactness claim from the direct-query robustness claim.",
            "",
        ]
    )
    return "\n".join(lines)


def build_latex_table(summary: dict[str, Any]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Compression ratio versus downstream mIoU. Saving ratio compares direct per-Gaussian 1280-D fp16 teacher-feature storage with the compact checkpoint footprint; downstream metrics use the frozen rendered-grounding and Direct3D readouts.}",
        "\\label{tab:compression_downstream_correlation}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Scene & Saving & Rendered mIoU & Direct3D mIoU & Direct3D Acc@0.25 \\\\",
        "\\midrule",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "{scene} & {saving:.2f}$\\times$ & {rendered:.4f} & {direct:.4f} & {acc:.4f} \\\\".format(
                scene=_latex_escape(str(row["scene"])),
                saving=float(row["saving_ratio"]),
                rendered=float(row["rendered_miou"]),
                direct=float(row["direct3d_miou"]),
                acc=float(row["direct3d_acc025"]),
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
    parser.add_argument("--storage", default=str(DEFAULT_STORAGE))
    parser.add_argument("--rendered", default=str(DEFAULT_RENDERED))
    parser.add_argument("--direct3d_root", default=str(DEFAULT_DIRECT3D_ROOT))
    parser.add_argument("--rendered_variant", default="0.60")
    parser.add_argument("--selection", default="thr0p25")
    parser.add_argument("--output_md", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--output_json", default=str(DEFAULT_JSON))
    parser.add_argument("--output_tex", default=str(DEFAULT_LATEX))
    args = parser.parse_args(argv)

    summary = build_summary(
        args.storage,
        args.rendered,
        direct3d_root=args.direct3d_root,
        rendered_variant=args.rendered_variant,
        selection=args.selection,
    )
    paths = write_outputs(summary, args.output_md, args.output_json, args.output_tex)
    print(f"Wrote {paths['markdown']}")
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['latex']}")
    return paths


if __name__ == "__main__":
    main()
