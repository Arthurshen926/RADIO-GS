#!/usr/bin/env python3
"""Compose a LERF direct-3D support-policy ablation qualitative figure."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from radio_gs.scripts.compose_lerf_main_qualitative import (
    REPO_ROOT,
    add_zoom_inset,
    build_gt_mask,
    compute_iou,
    fit_panel,
    load_binary_mask,
    overlay_mask,
    put_text_box,
    read_image,
    rel_or_str,
    resize_mask,
)


@dataclass(frozen=True)
class AblationCase:
    scene: str
    frame_id: str
    query: str


DEFAULT_CASES = [
    AblationCase("waldo_kitchen", "00053", "knife"),
    AblationCase("waldo_kitchen", "00140", "spoon"),
    AblationCase("ramen", "00024", "wavy noodles"),
    AblationCase("teatime", "00140", "plate"),
]


def parse_cases(items: Iterable[str]) -> list[AblationCase]:
    cases: list[AblationCase] = []
    for item in items:
        parts = item.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"Expected scene:frame:query, got {item!r}")
        cases.append(AblationCase(parts[0], parts[1], parts[2]))
    return cases


def mask_path(root: Path, case: AblationCase) -> Path:
    return root / case.scene / f"frame_{case.frame_id}_{case.query}.png"


def make_panel_title(panel: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    lines = [title] if not subtitle else [title, subtitle]
    return put_text_box(panel, lines, origin=(10, 10), font_scale=0.54)


def make_row(
    case: AblationCase,
    *,
    label_root: Path,
    base_root: Path,
    full_root: Path,
    panel_width: int,
    panel_height: int,
) -> tuple[np.ndarray, dict[str, object]]:
    rgb_path = label_root / case.scene / f"frame_{case.frame_id}.jpg"
    label_json = label_root / case.scene / f"frame_{case.frame_id}.json"
    base_path = mask_path(base_root, case)
    full_path = mask_path(full_root, case)
    rgb = read_image(rgb_path)
    gt = build_gt_mask(label_json, case.query)
    base = resize_mask(load_binary_mask(base_path), gt.shape)
    full = resize_mask(load_binary_mask(full_path), gt.shape)
    base_iou = compute_iou(gt, base)
    full_iou = compute_iou(gt, full)

    rgb_panel = make_panel_title(
        fit_panel(rgb, width=panel_width, height=panel_height),
        case.scene.replace("_", " "),
        f'"{case.query}"',
    )
    gt_panel = make_panel_title(
        fit_panel(overlay_mask(rgb, gt, (65, 185, 85)), width=panel_width, height=panel_height),
        "GT",
    )
    base_panel = make_panel_title(
        fit_panel(overlay_mask(rgb, base, (225, 125, 65)), width=panel_width, height=panel_height),
        "Base compact",
        f"IoU {base_iou:.3f}",
    )
    full_panel = make_panel_title(
        fit_panel(overlay_mask(rgb, full, (55, 115, 225)), width=panel_width, height=panel_height),
        "+ support policy",
        f"IoU {full_iou:.3f}",
    )
    cutout = np.full_like(rgb, 255)
    cutout[full] = rgb[full]
    contours, _ = cv2.findContours(full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cutout, contours, -1, (55, 115, 225), 3, cv2.LINE_AA)
    cutout = add_zoom_inset(cutout, full)
    cutout_panel = make_panel_title(
        fit_panel(cutout, width=panel_width, height=panel_height),
        "Selected 3D support",
        "rendered mask",
    )
    row = np.hstack([rgb_panel, gt_panel, base_panel, full_panel, cutout_panel])
    manifest = {
        "scene": case.scene,
        "frame_id": case.frame_id,
        "query": case.query,
        "base_iou": round(base_iou, 4),
        "full_iou": round(full_iou, 4),
        "delta_iou": round(full_iou - base_iou, 4),
        "rgb": rel_or_str(rgb_path),
        "label_json": rel_or_str(label_json),
        "base_mask": rel_or_str(base_path),
        "full_mask": rel_or_str(full_path),
    }
    return row, manifest


def add_column_header(image: np.ndarray, titles: list[str], *, panel_width: int) -> np.ndarray:
    header_h = 40
    canvas = np.full((image.shape[0] + header_h, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[header_h:] = image
    for idx, title in enumerate(titles):
        x = idx * panel_width + 12
        cv2.putText(canvas, title, (x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, header_h - 1), (canvas.shape[1], header_h - 1), (220, 220, 220), 1)
    return canvas


def write_markdown(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# LERF Direct3D Support-Policy Ablation Qualitative",
        "",
        "Qualitative ablation for the compact direct-field support policy. The base column uses the prior compact direct mask; the final column uses the current prompt-ensemble + component-support policy. No VPR cache or official RGB SAM3 readout is used by either row.",
        "",
        f"Figure: `{manifest['output']}`",
        "",
        "| Scene | Frame | Query | Base IoU | Support-policy IoU | Delta |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for case in manifest["cases"]:
        lines.append(
            "| {scene} | {frame_id} | {query} | {base_iou:.4f} | {full_iou:.4f} | {delta_iou:+.4f} |".format(
                **case
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-root", default="/mnt/pool/sqy/3d_understanding/lerf_ovs/label")
    parser.add_argument(
        "--base-root",
        default="output/radio_gs/lerf_direct3d_deployed_opacity_gate_masks_20260528/pred_masks/thr0p25",
    )
    parser.add_argument(
        "--full-root",
        default="output/radio_gs/lerf_direct3d_prompt_ensemble_policy_masks_20260528/pred_masks/thr0p65",
    )
    parser.add_argument("--output", default="paper/figures/lerf_direct3d_support_policy_ablation_qualitative.png")
    parser.add_argument("--manifest", default="paper/artifacts/lerf_direct3d_support_policy_ablation_qualitative_manifest.json")
    parser.add_argument("--report", default="paper/artifacts/lerf_direct3d_support_policy_ablation_qualitative.md")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--panel-width", type=int, default=300)
    parser.add_argument("--panel-height", type=int, default=205)
    args = parser.parse_args()

    cases = parse_cases(args.cases) if args.cases else DEFAULT_CASES
    rows: list[np.ndarray] = []
    manifests: list[dict[str, object]] = []
    for case in cases:
        row, manifest = make_row(
            case,
            label_root=Path(args.label_root),
            base_root=Path(args.base_root),
            full_root=Path(args.full_root),
            panel_width=args.panel_width,
            panel_height=args.panel_height,
        )
        rows.append(row)
        manifests.append(manifest)

    gutter = 10
    figure_h = sum(row.shape[0] for row in rows) + gutter * (len(rows) - 1)
    figure_w = rows[0].shape[1]
    figure = np.full((figure_h, figure_w, 3), 255, dtype=np.uint8)
    y = 0
    for row in rows:
        figure[y : y + row.shape[0]] = row
        y += row.shape[0] + gutter
    figure = add_column_header(
        figure,
        ["RGB/query", "GT", "Base compact", "Ours w/ support", "3D support"],
        panel_width=args.panel_width,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), figure)
    payload = {
        "output": rel_or_str(output),
        "base_root": rel_or_str(Path(args.base_root)),
        "full_root": rel_or_str(Path(args.full_root)),
        "layout": "RGB/query | GT | base compact direct field | prompt-ensemble support policy | rendered selected support",
        "cases": manifests,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report_path, payload)
    print(f"Wrote {output}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
