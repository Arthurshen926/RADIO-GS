#!/usr/bin/env python3
"""Compose the paper-facing foundation-feature downstream qualitative figure."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RowSpec:
    task: str
    query: str
    input_panel: Path
    teacher_panel: Path
    rendered_panel: Path


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def crop_equal_column(image: np.ndarray, index: int, count: int) -> np.ndarray:
    h, w = image.shape[:2]
    col_w = w // count
    x0 = index * col_w
    x1 = w if index == count - 1 else (index + 1) * col_w
    return image[:, x0:x1]


def crop_input_gt(image: np.ndarray, cols: int = 4) -> np.ndarray:
    rgb = crop_equal_column(image, 0, cols)
    gt = crop_equal_column(image, 1, cols)
    return np.hstack([rgb, gt])


def crop_lerf_grounding_cell(path: Path, column: int) -> np.ndarray:
    image = read_image(path)
    # LERF grounding visual columns: query, GT mask, heatmap, RGB, GT/RGB, heatmap/RGB.
    return crop_equal_column(image, column, 6)


def crop_task_cell(path: Path, column: int) -> np.ndarray:
    image = read_image(path)
    # SAM/DINO task visuals: RGB, GT, teacher, rendered.
    return crop_equal_column(image, column, 4)


def resize_crop(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max(width / max(w, 1), height / max(h, 1))
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    y0 = max(0, (resized.shape[0] - height) // 2)
    x0 = max(0, (resized.shape[1] - width) // 2)
    return resized[y0 : y0 + height, x0 : x0 + width]


def resize_fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y0 = max(0, (height - resized.shape[0]) // 2)
    x0 = max(0, (width - resized.shape[1]) // 2)
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def add_cell_label(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    band_h = 30
    overlay = out.copy()
    overlay[:band_h, :] = (20, 20, 20)
    out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)
    cv2.putText(out, label, (9, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def task_panel(task: str, query: str, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.putText(panel, task, (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (55, 80, 150), 2, cv2.LINE_AA)
    cv2.putText(panel, query, (14, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (45, 45, 45), 1, cv2.LINE_AA)
    return panel


def build_rows(args: argparse.Namespace) -> tuple[list[np.ndarray], list[dict[str, str]]]:
    cell_w = int(args.cell_width)
    cell_h = int(args.cell_height)
    task_w = int(args.task_width)
    manifest: list[dict[str, str]] = []

    siglip_teacher = Path(args.siglip_teacher)
    siglip_rendered = Path(args.siglip_rendered)
    specs: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]] = [
        (
            "SigLIP2 grounding",
            '"wavy noodles"',
            crop_lerf_grounding_cell(siglip_rendered, 3),
            crop_lerf_grounding_cell(siglip_rendered, 4),
            crop_lerf_grounding_cell(siglip_teacher, 5),
            crop_lerf_grounding_cell(siglip_rendered, 5),
            {
                "task": "SigLIP2 grounding",
                "teacher_panel": rel(siglip_teacher),
                "rendered_panel": rel(siglip_rendered),
            },
        ),
    ]

    sam = Path(args.sam_visual)
    specs.append(
        (
            "SAM3 mask propagation",
            '"plate"',
            crop_task_cell(sam, 0),
            crop_task_cell(sam, 1),
            crop_task_cell(sam, 2),
            crop_task_cell(sam, 3),
            {"task": "SAM3 mask propagation", "panel": rel(sam)},
        )
    )

    dino_mask = Path(args.dino_mask_visual)
    specs.append(
        (
            "DINOv3 mask prop.",
            '"bowl"',
            crop_task_cell(dino_mask, 0),
            crop_task_cell(dino_mask, 1),
            crop_task_cell(dino_mask, 2),
            crop_task_cell(dino_mask, 3),
            {"task": "DINO mask propagation", "panel": rel(dino_mask)},
        )
    )

    dino_teacher = Path(args.dino_match_teacher)
    dino_rendered = Path(args.dino_match_rendered)
    teacher_match = read_image(dino_teacher)
    rendered_match = read_image(dino_rendered)
    specs.append(
        (
            "DINOv3 matching",
            '"bowl", 6 -> 24',
            crop_equal_column(rendered_match, 0, 2),
            crop_equal_column(rendered_match, 1, 2),
            teacher_match,
            rendered_match,
            {
                "task": "DINO dense matching",
                "teacher_panel": rel(dino_teacher),
                "rendered_panel": rel(dino_rendered),
            },
        )
    )

    rows: list[np.ndarray] = []
    for task, query, input_img, gt_img, teacher_img, rendered_img, meta in specs:
        task_cell = add_cell_label(task_panel(task, query, task_w, cell_h), "Task")
        input_cell = add_cell_label(resize_crop(input_img, cell_w, cell_h), "Input / source")
        gt_cell = add_cell_label(resize_crop(gt_img, cell_w, cell_h), "GT / target")
        if "matching" in task.lower():
            teacher_cell = add_cell_label(resize_fit(teacher_img, cell_w, cell_h), "Frame-wise RADIO")
            rendered_cell = add_cell_label(resize_fit(rendered_img, cell_w, cell_h), "GaussFM field")
        else:
            teacher_cell = add_cell_label(resize_crop(teacher_img, cell_w, cell_h), "Frame-wise RADIO")
            rendered_cell = add_cell_label(resize_crop(rendered_img, cell_w, cell_h), "GaussFM field")
        rows.append(np.hstack([task_cell, input_cell, gt_cell, teacher_cell, rendered_cell]))
        manifest.append(meta)
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--siglip-teacher",
        default=(
            "output/radio_gs/freeze_eval/lerf_ramen_overlay_calibrated_thr0p60_vis_20260514/"
            "visualisations/ramen/lerf_grounding_frame_00024_teacher_wavy_noodles.png"
        ),
    )
    parser.add_argument(
        "--siglip-rendered",
        default=(
            "output/radio_gs/freeze_eval/lerf_ramen_overlay_calibrated_thr0p60_vis_20260514/"
            "visualisations/ramen/lerf_grounding_frame_00024_rendered_wavy_noodles.png"
        ),
    )
    parser.add_argument(
        "--sam-visual",
        default=(
            "output/lerf_sam_dino_tasks/mainline_fixed_vis/waldo_kitchen/visualizations/"
            "waldo_kitchen/waldo_kitchen_sam3_mask_prompt_propagation_00154_plate.png"
        ),
    )
    parser.add_argument(
        "--dino-mask-visual",
        default=(
            "output/lerf_sam_dino_tasks/formal_v9_dino_bg110_vis/ramen/visualizations/"
            "ramen/ramen_dino_v3_mask_propagation_00024_bowl.png"
        ),
    )
    parser.add_argument(
        "--dino-match-teacher",
        default=(
            "output/lerf_sam_dino_tasks/formal_v8_mutual_homography_ransac_vis/ramen/visualizations/"
            "ramen/ramen_dino_v3_dense_matching_teacher_00006_00024_bowl.png"
        ),
    )
    parser.add_argument(
        "--dino-match-rendered",
        default=(
            "output/lerf_sam_dino_tasks/formal_v8_mutual_homography_ransac_vis/ramen/visualizations/"
            "ramen/ramen_dino_v3_dense_matching_rendered_00006_00024_bowl.png"
        ),
    )
    parser.add_argument("--output", default="paper/figures/lerf_sam_dino_tasks_qualitative.png")
    parser.add_argument(
        "--manifest",
        default="paper/artifacts/lerf_sam_dino_tasks_qualitative_manifest.json",
    )
    parser.add_argument(
        "--report",
        default="paper/artifacts/lerf_sam_dino_tasks_qualitative.md",
    )
    parser.add_argument("--task-width", type=int, default=320)
    parser.add_argument("--cell-width", type=int, default=360)
    parser.add_argument("--cell-height", type=int, default=230)
    args = parser.parse_args()

    rows, entries = build_rows(args)
    gap = np.full((12, rows[0].shape[1], 3), 255, dtype=np.uint8)
    header = np.full((48, rows[0].shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        "Reconstructed scene features vs. frame-wise RADIO across frozen-head tasks",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    stack: list[np.ndarray] = [header]
    for idx, row in enumerate(rows):
        if idx:
            stack.append(gap)
        stack.append(row)
    figure = np.vstack(stack)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), figure)

    manifest = {
        "figure": rel(output),
        "rows": entries,
        "notes": (
            "Rows are qualitative examples for the same quantitative frozen-head probes. "
            "The GaussFM field columns are reconstructed from the compact scene feature field; "
            "the frame-wise RADIO columns are per-image foundation features under the same task heads."
        ),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = Path(args.report)
    report.write_text(
        "# Reconstructed Scene Features vs. Frame-wise RADIO Qualitative\n\n"
        f"Figure: `{rel(output)}`\n\n"
        "The figure summarizes SigLIP2 text grounding, SAM3-adaptor mask propagation, "
        "DINOv3 mask propagation, and DINOv3 dense matching under the same frozen-head "
        "evaluation family used in the paper.\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"wrote {manifest_path}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
