#!/usr/bin/env python3
"""Summarize OpenGaussian baseline reproduction against RADIO-GS."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


OPENGAUSSIAN_PAPER_SCANNET = {
    "19": {"miou": 0.2473, "macc": 0.4154},
    "15": {"miou": 0.3013, "macc": 0.4825},
    "10": {"miou": 0.3829, "macc": 0.5519},
}

OPENGAUSSIAN_PAPER_LERF = {
    "figurines": {"miou": 0.3929, "macc025": 0.5536},
    "teatime": {"miou": 0.6044, "macc025": 0.7627},
    "ramen": {"miou": 0.3101, "macc025": 0.4225},
    "waldo_kitchen": {"miou": 0.2270, "macc025": 0.3182},
    "macro": {"miou": 0.3836, "macc025": 0.5143},
}


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_radio_lerf(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _scan_table_lines(radio: dict[str, Any] | None, og: dict[str, Any] | None) -> list[str]:
    lines = [
        "## ScanNet 3D Segmentation",
        "",
        "| Method | Source | split19 mIoU | split19 mAcc | split15 mIoU | split15 mAcc | split10 mIoU | split10 mAcc |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    paper = OPENGAUSSIAN_PAPER_SCANNET
    lines.append(
        "| OpenGaussian | paper Table 2 | "
        f"{_fmt(paper['19']['miou'])} | {_fmt(paper['19']['macc'])} | "
        f"{_fmt(paper['15']['miou'])} | {_fmt(paper['15']['macc'])} | "
        f"{_fmt(paper['10']['miou'])} | {_fmt(paper['10']['macc'])} |"
    )
    if og is not None:
        macro = og["macro"]
        lines.append(
            "| OpenGaussian | local reproduction | "
            f"{_fmt(macro['19']['miou'])} | {_fmt(macro['19']['macc'])} | "
            f"{_fmt(macro['15']['miou'])} | {_fmt(macro['15']['macc'])} | "
            f"{_fmt(macro['10']['miou'])} | {_fmt(macro['10']['macc'])} |"
        )
    if radio is not None:
        macro = radio["macro"]
        lines.append(
            "| RADIO-GS | local v67 direct point-query | "
            f"{_fmt(macro['19']['miou'])} | {_fmt(macro['19']['macc'])} | "
            f"{_fmt(macro['15']['miou'])} | {_fmt(macro['15']['macc'])} | "
            f"{_fmt(macro['10']['miou'])} | {_fmt(macro['10']['macc'])} |"
        )
    lines.append("")
    if og is None:
        lines.append("Local OpenGaussian ScanNet reproduction is still pending.")
        lines.append("")
        return lines

    lines.extend(
        [
            "### Per-Scene Local Reproduction",
            "",
            "| Scene | OpenGaussian 19 mIoU/mAcc | OpenGaussian 15 mIoU/mAcc | OpenGaussian 10 mIoU/mAcc | RADIO-GS 19 mIoU/mAcc | RADIO-GS 15 mIoU/mAcc | RADIO-GS 10 mIoU/mAcc |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    radio_scenes = radio.get("scenes", {}) if radio is not None else {}
    for scene, entry in sorted(og.get("scenes", {}).items()):
        og_splits = entry["splits"]
        radio_entry = radio_scenes.get(scene, {})
        radio_splits = radio_entry.get("splits", {})

        def pair(splits: dict[str, Any], split: str) -> str:
            if split not in splits:
                return "-"
            return f"{_fmt(splits[split]['miou'])}/{_fmt(splits[split]['macc'])}"

        lines.append(
            f"| {scene} | {pair(og_splits, '19')} | {pair(og_splits, '15')} | "
            f"{pair(og_splits, '10')} | {pair(radio_splits, '19')} | "
            f"{pair(radio_splits, '15')} | {pair(radio_splits, '10')} |"
        )
    lines.append("")
    return lines


def _lerf_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "## LERF-OVS",
        "",
        "OpenGaussian reports LeRF as 3D object selection mIoU and mAcc@0.25. RADIO-GS currently reports rendered-feature 2D grounding LocAcc and heatmap mIoU. These are both useful qualitative/quantitative evidence, but they are not a single identical metric protocol.",
        "",
        "| Method | Protocol | Figurines | Ramen | Teatime | Waldo Kitchen | Macro |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lerf = OPENGAUSSIAN_PAPER_LERF
    lines.append(
        "| OpenGaussian | paper object-selection mIoU | "
        f"{_fmt(lerf['figurines']['miou'])} | {_fmt(lerf['ramen']['miou'])} | "
        f"{_fmt(lerf['teatime']['miou'])} | {_fmt(lerf['waldo_kitchen']['miou'])} | "
        f"{_fmt(lerf['macro']['miou'])} |"
    )
    lines.append(
        "| OpenGaussian | paper object-selection mAcc@0.25 | "
        f"{_fmt(lerf['figurines']['macc025'])} | {_fmt(lerf['ramen']['macc025'])} | "
        f"{_fmt(lerf['teatime']['macc025'])} | {_fmt(lerf['waldo_kitchen']['macc025'])} | "
        f"{_fmt(lerf['macro']['macc025'])} |"
    )
    if rows:
        by_scene = {row["scene"]: row for row in rows}
        macro = by_scene.get("macro", {})
        lines.append(
            "| RADIO-GS | rendered-feature LocAcc | "
            f"{_fmt(float(by_scene['figurines']['loc_acc']))} | {_fmt(float(by_scene['ramen']['loc_acc']))} | "
            f"{_fmt(float(by_scene['teatime']['loc_acc']))} | {_fmt(float(by_scene['waldo_kitchen']['loc_acc']))} | "
            f"{_fmt(float(macro['loc_acc']))} |"
        )
        lines.append(
            "| RADIO-GS | rendered-feature heatmap mIoU | "
            f"{_fmt(float(by_scene['figurines']['miou']))} | {_fmt(float(by_scene['ramen']['miou']))} | "
            f"{_fmt(float(by_scene['teatime']['miou']))} | {_fmt(float(by_scene['waldo_kitchen']['miou']))} | "
            f"{_fmt(float(macro['miou']))} |"
        )
    lines.extend(
        [
            "",
            "Local OpenGaussian LeRF reproduction is blocked until the official LangSplat-reannotated `language_features.zip` is available. The local LERF folders only contain images/COLMAP/labels, while OpenGaussian's LeRF recipe expects precomputed `language_features/` together with the reannotated image package.",
            "",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opengaussian-scannet-json", default="output/baselines/opengaussian/scannet_eval/opengaussian_scannet_results.json")
    parser.add_argument("--radio-scannet-json", default="output/scannet_pointcloud_eval/freeze_v67_all_eval_20260502/scannet_pointcloud_radio_gs_results.json")
    parser.add_argument("--radio-lerf-csv", default="output/radio_gs/lerf_summary_tables/current_best_lerf_ovs_per_scene.csv")
    parser.add_argument("--qualitative-image", default="output/baselines/opengaussian/scannet_qualitative_comparison.png")
    parser.add_argument("--output", default="output/baselines/opengaussian/opengaussian_vs_radio_gs_report.md")
    args = parser.parse_args()

    og = _load_json(Path(args.opengaussian_scannet_json))
    radio = _load_json(Path(args.radio_scannet_json))
    lerf_rows = _load_radio_lerf(Path(args.radio_lerf_csv))

    lines = [
        "# OpenGaussian vs RADIO-GS Baseline Report",
        "",
        "Baseline selected: OpenGaussian, because its official release covers both ScanNet open-vocabulary point-cloud understanding and LeRF object selection, and the repository provides reproducible training/evaluation code.",
        "",
        "Sources: OpenGaussian project page `https://3d-aigc.github.io/OpenGaussian/`, official code `https://github.com/yanmin-wu/OpenGaussian`, and arXiv `https://arxiv.org/abs/2406.02058`.",
        "",
    ]
    lines.extend(_scan_table_lines(radio, og))
    lines.extend(_lerf_lines(lerf_rows))
    q = Path(args.qualitative_image)
    lines.extend(
        [
            "## Qualitative Artifacts",
            "",
            f"- ScanNet GT/RADIO-GS/OpenGaussian montage: `{q}`" if q.exists() else f"- ScanNet montage pending: `{q}`",
            "- Per-scene OpenGaussian PLY/PNG files: `output/baselines/opengaussian/scannet_eval/visualizations/{scene}/`",
            "- RADIO-GS v67 per-scene PLY files: `output/scannet_pointcloud_eval/{scene}_v67_teacherbalanced_fromv63_best_gidx_labelpoint/visualizations/{scene}/`",
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
