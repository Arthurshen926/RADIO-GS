#!/usr/bin/env python3
"""Compose rendered-view boundary calibration qualitative examples."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BoundaryCase:
    scene: str
    mode: str
    stem: str
    query: str


DEFAULT_CASES = (
    BoundaryCase("ramen", "rendered", "frame_00065_bowl", "bowl"),
    BoundaryCase("ramen", "rendered", "frame_00006_bowl", "bowl"),
    BoundaryCase("ramen", "rendered", "frame_00081_onion_segments", "onion segments"),
    BoundaryCase("ramen", "rendered", "frame_00128_spoon", "spoon"),
)


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def resize_panel(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def mask_overlay(mask: np.ndarray, label_color: tuple[int, int, int], size: tuple[int, int]) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    panel = np.full((mask.shape[0], mask.shape[1], 3), 245, dtype=np.uint8)
    panel[mask > 0] = label_color
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, (40, 40, 40), 2)
    return resize_panel(panel, size)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    return float(np.logical_and(aa, bb).sum() / max(1, np.logical_or(aa, bb).sum()))


def add_label(panel: np.ndarray, text: str, *, top: bool = True) -> np.ndarray:
    out = panel.copy()
    h, w = out.shape[:2]
    band_h = 32
    y0 = 0 if top else h - band_h
    overlay = out.copy()
    overlay[y0 : y0 + band_h, :] = (18, 18, 18)
    out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)
    cv2.putText(
        out,
        text,
        (10, y0 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def compose_case(root: Path, case: BoundaryCase, size: tuple[int, int]) -> tuple[np.ndarray, dict[str, object]]:
    case_root = root / case.scene / case.mode
    gt_path = case_root / f"{case.stem}_gt.png"
    initial_path = case_root / f"{case.stem}_initial.png"
    final_path = case_root / f"{case.stem}_final.png"
    initial_overlay_path = case_root / f"{case.stem}_initial_overlay.png"
    final_overlay_path = case_root / f"{case.stem}_final_overlay.png"

    gt = read_image(gt_path, cv2.IMREAD_GRAYSCALE)
    initial = read_image(initial_path, cv2.IMREAD_GRAYSCALE)
    final = read_image(final_path, cv2.IMREAD_GRAYSCALE)
    initial_overlay = resize_panel(read_image(initial_overlay_path), size)
    final_overlay = resize_panel(read_image(final_overlay_path), size)
    gt_panel = mask_overlay(gt, (80, 200, 80), size)
    initial_panel = add_label(initial_overlay, f"Coarse IoU {iou(initial, gt):.2f}", top=False)
    final_panel = add_label(final_overlay, f"Boundary IoU {iou(final, gt):.2f}", top=False)
    query_panel = np.full((size[1], size[0], 3), 248, dtype=np.uint8)
    cv2.putText(query_panel, case.scene, (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (70, 90, 150), 2, cv2.LINE_AA)
    cv2.putText(query_panel, f'"{case.query}"', (14, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(
        query_panel,
        f"+{(iou(final, gt) - iou(initial, gt)):.2f} IoU",
        (14, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (30, 120, 80),
        2,
        cv2.LINE_AA,
    )
    row = np.hstack(
        [
            add_label(query_panel, "Query"),
            add_label(gt_panel, "GT"),
            add_label(initial_panel, "Heatmap support"),
            add_label(final_panel, "Feature-only boundary"),
        ]
    )
    manifest = {
        "scene": case.scene,
        "query": case.query,
        "stem": case.stem,
        "initial_iou": iou(initial, gt),
        "final_iou": iou(final, gt),
        "delta_iou": iou(final, gt) - iou(initial, gt),
        "gt": rel(gt_path),
        "initial": rel(initial_path),
        "final": rel(final_path),
    }
    return row, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-root",
        default=(
            "output/radio_gs/lerf2d_heatmap_guard_sam3_20260525_masks/"
            "ramen_peakinit_base_lerf2dcoarse/pred_masks"
        ),
    )
    parser.add_argument("--output", default="paper/figures/lerf_rendered_boundary_calibration_qualitative.png")
    parser.add_argument(
        "--manifest",
        default="paper/artifacts/lerf_rendered_boundary_calibration_qualitative_manifest.json",
    )
    parser.add_argument(
        "--report",
        default="paper/artifacts/lerf_rendered_boundary_calibration_qualitative.md",
    )
    parser.add_argument("--panel-width", type=int, default=300)
    parser.add_argument("--panel-height", type=int, default=220)
    args = parser.parse_args()

    rows: list[np.ndarray] = []
    entries: list[dict[str, object]] = []
    size = (int(args.panel_width), int(args.panel_height))
    for case in DEFAULT_CASES:
        row, manifest = compose_case(Path(args.mask_root), case, size)
        rows.append(row)
        entries.append(manifest)

    gap = np.full((12, rows[0].shape[1], 3), 255, dtype=np.uint8)
    figure = []
    header = np.full((44, rows[0].shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        "Rendered-view boundary calibration: coarse heatmap support -> feature-only boundary mask",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    figure.append(header)
    for idx, row in enumerate(rows):
        if idx:
            figure.append(gap)
        figure.append(row)
    output_image = np.vstack(figure)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), output_image)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "figure": rel(output),
        "mask_root": rel(Path(args.mask_root)),
        "cases": entries,
        "notes": (
            "The figure uses masks exported by eval_lerf_grounding --save_pred_masks. "
            "The feature-only boundary head is driven by reconstructed RADIO/GaussFM "
            "features and prompt embeddings; it does not call the official RGB SAM3 decoder."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = Path(args.report)
    lines = [
        "# Rendered Boundary Calibration Qualitative",
        "",
        f"Figure: `{rel(output)}`",
        "",
        "| Scene | Query | Initial IoU | Boundary IoU | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for entry in entries:
        lines.append(
            f"| {entry['scene']} | {entry['query']} | "
            f"{entry['initial_iou']:.4f} | {entry['final_iou']:.4f} | {entry['delta_iou']:+.4f} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {manifest_path}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
