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
LERF_SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")


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


def inspect_opengaussian_lerf_assets(lerf_root: Path) -> dict[str, dict[str, int | bool]]:
    """Inspect whether local LERF folders can run OpenGaussian's LeRF recipe."""
    status: dict[str, dict[str, int | bool]] = {}
    for scene in LERF_SCENES:
        scene_root = lerf_root / scene
        image_dir = scene_root / "images"
        language_dir = scene_root / "language_features"
        label_dir = lerf_root / "label" / scene
        status[scene] = {
            "scene_root_exists": scene_root.exists(),
            "images": len(list(image_dir.glob("*"))) if image_dir.exists() else 0,
            "language_feature_masks": len(list(language_dir.glob("*_s.npy"))) if language_dir.exists() else 0,
            "language_feature_vectors": len(list(language_dir.glob("*_f.npy"))) if language_dir.exists() else 0,
            "labels": len(list(label_dir.rglob("*.jpg"))) if label_dir.exists() else 0,
            "ready": (
                scene_root.exists()
                and image_dir.exists()
                and language_dir.exists()
                and any(language_dir.glob("*_s.npy"))
                and any(language_dir.glob("*_f.npy"))
            ),
        }
    return status


def _load_radio_lerf_threshold_sweep(path: Path, threshold: str | float) -> list[dict[str, str]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = f"{float(threshold):.2f}"
    variant = payload.get("variants", {}).get(key)
    if variant is None:
        return []
    rows: list[dict[str, str]] = []
    for row in variant.get("rows", []):
        rows.append(
            {
                "scene": str(row["scene"]),
                "loc_acc": _fmt(float(row["loc"])),
                "miou": _fmt(float(row["miou"])),
            }
        )
    macro = variant.get("macro", {})
    rows.append(
        {
            "scene": "macro",
            "loc_acc": _fmt(float(macro.get("loc", 0.0))),
            "miou": _fmt(float(macro.get("miou", 0.0))),
        }
    )
    return rows


def _load_radio_direct_lerf_results(path: Path, tag: str) -> dict[str, dict[str, float]] | None:
    if not path.exists():
        return None
    direct: dict[str, dict[str, float]] = {}
    for scene in LERF_SCENES:
        result_path = path / scene / "lerf_direct_3d_selection_results.json"
        payload = _load_json(result_path)
        if payload is None:
            return None
        scene_payload = payload.get("scene", {})
        result = scene_payload.get("results", {}).get(tag)
        if result is None:
            return None
        direct[scene] = {
            "miou": float(result["miou"]),
            "acc025": float(result["acc025"]),
        }
    direct["macro"] = {
        "miou": sum(direct[scene]["miou"] for scene in LERF_SCENES) / len(LERF_SCENES),
        "acc025": sum(direct[scene]["acc025"] for scene in LERF_SCENES) / len(LERF_SCENES),
    }
    return direct


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


def _lerf_lines(
    rows: list[dict[str, str]],
    direct: dict[str, dict[str, float]] | None = None,
    asset_status: dict[str, dict[str, int | bool]] | None = None,
) -> list[str]:
    lines = [
        "## LERF-OVS",
        "",
        "OpenGaussian reports LeRF as 3D object selection mIoU and mAcc@0.25. RADIO-GS reports rendered-feature 2D grounding and, when available, VPR-backed direct 3D primitive selection. The direct 3D rows follow the same query-select-render metric family, while rendered-feature rows are a different protocol.",
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
    if direct is not None:
        lines.append(
            "| RADIO-GS/CTF-GS | VPR direct 3D selection mIoU | "
            f"{_fmt(direct['figurines']['miou'])} | {_fmt(direct['ramen']['miou'])} | "
            f"{_fmt(direct['teatime']['miou'])} | {_fmt(direct['waldo_kitchen']['miou'])} | "
            f"{_fmt(direct['macro']['miou'])} |"
        )
        lines.append(
            "| RADIO-GS/CTF-GS | VPR direct 3D selection Acc@0.25 | "
            f"{_fmt(direct['figurines']['acc025'])} | {_fmt(direct['ramen']['acc025'])} | "
            f"{_fmt(direct['teatime']['acc025'])} | {_fmt(direct['waldo_kitchen']['acc025'])} | "
            f"{_fmt(direct['macro']['acc025'])} |"
        )
    lines.append("")
    if asset_status:
        ready = all(bool(asset_status.get(scene, {}).get("ready")) for scene in LERF_SCENES)
        lines.extend(
            [
                "### Local OpenGaussian LeRF Asset Check",
                "",
                "| Scene | Images | Language masks | Language feats | Labels | Ready |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for scene in LERF_SCENES:
            row = asset_status.get(scene, {})
            lines.append(
                f"| {scene} | {row.get('images', 0)} | "
                f"{row.get('language_feature_masks', 0)} | "
                f"{row.get('language_feature_vectors', 0)} | "
                f"{row.get('labels', 0)} | {bool(row.get('ready'))} |"
            )
        lines.append("")
        if ready:
            lines.append(
                "Local OpenGaussian LeRF reproduction assets appear complete; run the official `train_lerf.sh`, `render_lerf_by_text.py`, and `compute_lerf_iou.py` flow under the local paths."
            )
        else:
            lines.append(
                "Local OpenGaussian LeRF reproduction is blocked: OpenGaussian's LeRF recipe requires per-frame `language_features/*_s.npy` SAM masks and `language_features/*_f.npy` CLIP features. The inspected local LERF folders have images/COLMAP/labels but no complete `language_features/` assets."
            )
    else:
        lines.append(
            "Local OpenGaussian LeRF reproduction asset status was not inspected; without `language_features/`, the official OpenGaussian LeRF recipe cannot be rerun locally."
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opengaussian-scannet-json", default="output/baselines/opengaussian/scannet_eval/opengaussian_scannet_results.json")
    parser.add_argument("--radio-scannet-json", default="output/scannet_pointcloud_eval/freeze_v67_all_eval_20260502/scannet_pointcloud_radio_gs_results.json")
    parser.add_argument("--radio-lerf-csv", default="output/radio_gs/lerf_summary_tables/current_best_lerf_ovs_per_scene.csv")
    parser.add_argument("--radio-lerf-threshold-sweep-json", default="output/radio_gs/reports/lerf_rendered_grounding_paper_ckpt_threshold_sweep.json")
    parser.add_argument("--radio-lerf-threshold", default="0.60")
    parser.add_argument("--radio-direct-lerf-root", default="output/radio_gs/lerf_direct_3d_selection_max128_cap0p018_cache_20260514")
    parser.add_argument("--radio-direct-lerf-tag", default="meanstd2p5")
    parser.add_argument("--opengaussian-lerf-root", default="/mnt/pool/sqy/3d_understanding/lerf_ovs")
    parser.add_argument("--qualitative-image", default="output/baselines/opengaussian/scannet_qualitative_comparison.png")
    parser.add_argument("--output", default="output/baselines/opengaussian/opengaussian_vs_radio_gs_report.md")
    args = parser.parse_args()

    og = _load_json(Path(args.opengaussian_scannet_json))
    radio = _load_json(Path(args.radio_scannet_json))
    lerf_rows = _load_radio_lerf_threshold_sweep(
        Path(args.radio_lerf_threshold_sweep_json),
        args.radio_lerf_threshold,
    )
    if not lerf_rows:
        lerf_rows = _load_radio_lerf(Path(args.radio_lerf_csv))
    direct_lerf = _load_radio_direct_lerf_results(Path(args.radio_direct_lerf_root), args.radio_direct_lerf_tag)
    lerf_assets = inspect_opengaussian_lerf_assets(Path(args.opengaussian_lerf_root))

    lines = [
        "# OpenGaussian vs RADIO-GS Baseline Report",
        "",
        "Baseline selected: OpenGaussian, because its official release covers both ScanNet open-vocabulary point-cloud understanding and LeRF object selection, and the repository provides reproducible training/evaluation code.",
        "",
        "Sources: OpenGaussian project page `https://3d-aigc.github.io/OpenGaussian/`, official code `https://github.com/yanmin-wu/OpenGaussian`, and arXiv `https://arxiv.org/abs/2406.02058`.",
        "",
    ]
    lines.extend(_scan_table_lines(radio, og))
    lines.extend(_lerf_lines(lerf_rows, direct_lerf, lerf_assets))
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
