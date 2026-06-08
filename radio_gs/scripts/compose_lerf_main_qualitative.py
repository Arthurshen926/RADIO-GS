#!/usr/bin/env python3
"""Compose the main-paper LERF qualitative comparison figure.

The default figure is intentionally narrow: RGB/query, GT, one reproduced
baseline, and the deployed compact direct-field readout. It avoids the legacy
VPR-cache and official RGB SAM3 qualitative rows so the main figure matches the
current compact-field claim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class QualCase:
    scene: str
    frame_id: str
    query: str


DEFAULT_CASES = [
    QualCase("figurines", "00041", "old camera"),
    QualCase("ramen", "00024", "onion segments"),
    QualCase("teatime", "00140", "bag of cookies"),
    QualCase("waldo_kitchen", "00140", "plate"),
]

DEFAULT_OVS_2D3D_GROUPS = [
    [
        QualCase("figurines", "00105", "old camera"),
        QualCase("figurines", "00105", "green apple"),
        QualCase("figurines", "00105", "pumpkin"),
    ],
    [
        QualCase("teatime", "00140", "tea in a glass"),
        QualCase("teatime", "00140", "apple"),
        QualCase("teatime", "00140", "bag of cookies"),
    ],
]

SCENE_COLORS = {
    "figurines": (255, 236, 224),
    "ramen": (232, 242, 226),
    "teatime": (226, 244, 224),
    "waldo_kitchen": (245, 235, 224),
}


DR_SPLAT_SCENE_DIR = {
    "figurines": "figurines_1_lerfcompat_topk45_weight_128",
    "ramen": "ramen_1_lerfcompat_topk45_weight_128",
    "teatime": "teatime_1_lerfcompat_topk45_weight_128",
    "waldo_kitchen": "waldo_kitchen_1_lerfcompat_topk45_weight_128",
}


def rel_or_str(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_cases(items: Iterable[str]) -> list[QualCase]:
    cases: list[QualCase] = []
    for item in items:
        parts = item.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"Expected scene:frame:query, got {item!r}")
        cases.append(QualCase(parts[0], parts[1], parts[2]))
    return cases


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def build_gt_mask(label_json: Path, category: str) -> np.ndarray:
    payload = json.loads(label_json.read_text(encoding="utf-8"))
    height = int(payload["info"]["height"])
    width = int(payload["info"]["width"])
    mask = np.zeros((height, width), dtype=np.uint8)
    for obj in payload.get("objects", []):
        if obj.get("category") != category:
            continue
        points = np.asarray(obj.get("segmentation", []), dtype=np.float32)
        if points.size == 0:
            continue
        points = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def load_binary_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3 and image.shape[2] == 4:
        gray = image[:, :, 3]
        if int(gray.max()) == 255 and int(gray.min()) == 255:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    threshold = 0 if int(gray.max()) <= 1 else 127
    return gray > threshold


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    resized = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def compute_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    union = np.logical_or(gt, pred).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(gt, pred).sum() / union)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    out[mask] = (0.55 * out[mask].astype(np.float32) + 0.45 * color_arr).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 3, lineType=cv2.LINE_AA)
    return out


def fit_panel(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    new_w = max(1, int(round(image.shape[1] * scale)))
    new_h = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def put_text_box(
    image: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int] = (24, 24, 24),
    bg: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    out = image.copy()
    if not lines:
        return out
    thickness = 1
    pad_x = 8
    pad_y = 6
    line_h = int(round(21 * font_scale / 0.55))
    sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        for line in lines
    ]
    box_w = max(size[0] for size in sizes) + 2 * pad_x
    box_h = len(lines) * line_h + 2 * pad_y
    x, y = origin
    x = min(x, max(0, out.shape[1] - box_w - 1))
    y = min(y, max(0, out.shape[0] - box_h - 1))
    roi = out[y : y + box_h, x : x + box_w].copy()
    fill = np.full_like(roi, bg)
    out[y : y + box_h, x : x + box_w] = cv2.addWeighted(roi, 0.18, fill, 0.82, 0)
    cv2.rectangle(out, (x, y), (x + box_w, y + box_h), (230, 230, 230), 1, cv2.LINE_AA)
    text_y = y + pad_y + line_h - 6
    for line in lines:
        cv2.putText(
            out,
            line,
            (x + pad_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        text_y += line_h
    return out


def add_header(panel: np.ndarray, title: str, *, header_h: int = 38) -> np.ndarray:
    canvas = np.full((panel.shape[0] + header_h, panel.shape[1], 3), 255, dtype=np.uint8)
    canvas[header_h:] = panel
    cv2.putText(canvas, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, header_h - 1), (panel.shape[1], header_h - 1), (225, 225, 225), 1)
    return canvas


def query_slug(query: str) -> str:
    return query.replace(" ", "_").replace("/", "_")


def mask_bbox(mask: np.ndarray, *, pad: int = 8) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1] - 1, int(xs.max()) + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0] - 1, int(ys.max()) + pad)
    return x0, y0, x1, y1


def draw_dashed_rect(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int] = (40, 40, 230),
    dash: int = 10,
    gap: int = 6,
) -> None:
    x0, y0, x1, y1 = bbox
    for x in range(x0, x1, dash + gap):
        cv2.line(image, (x, y0), (min(x + dash, x1), y0), color, 2, cv2.LINE_AA)
        cv2.line(image, (x, y1), (min(x + dash, x1), y1), color, 2, cv2.LINE_AA)
    for y in range(y0, y1, dash + gap):
        cv2.line(image, (x0, y), (x0, min(y + dash, y1)), color, 2, cv2.LINE_AA)
        cv2.line(image, (x1, y), (x1, min(y + dash, y1)), color, 2, cv2.LINE_AA)


def add_zoom_inset(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bbox = mask_bbox(mask, pad=20)
    if bbox is None:
        return image.copy()
    out = image.copy()
    x0, y0, x1, y1 = bbox
    crop = image[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0:
        return out
    max_w = max(80, int(round(image.shape[1] * 0.34)))
    max_h = max(70, int(round(image.shape[0] * 0.34)))
    scale = min(max_w / crop.shape[1], max_h / crop.shape[0])
    inset_w = max(1, int(round(crop.shape[1] * scale)))
    inset_h = max(1, int(round(crop.shape[0] * scale)))
    inset = cv2.resize(crop, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
    ox = image.shape[1] - inset_w - 12
    oy = 12
    draw_dashed_rect(out, bbox)
    cv2.rectangle(out, (ox - 3, oy - 3), (ox + inset_w + 3, oy + inset_h + 3), (40, 40, 230), 2, cv2.LINE_AA)
    out[oy : oy + inset_h, ox : ox + inset_w] = inset
    cv2.line(out, (x1, y0), (ox, oy), (40, 40, 230), 1, cv2.LINE_AA)
    cv2.line(out, (x1, y1), (ox, oy + inset_h), (40, 40, 230), 1, cv2.LINE_AA)
    return out


def object_cutout_panel(image: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int]) -> np.ndarray:
    out = np.full_like(image, 255)
    out[mask] = image[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 3, lineType=cv2.LINE_AA)
    return add_zoom_inset(out, mask)


def crop_wide_visual_panel(image: np.ndarray, *, index: int = 5, columns: int = 6) -> np.ndarray:
    if image.shape[1] < columns:
        return image
    panel_w = image.shape[1] // columns
    x0 = min(index, columns - 1) * panel_w
    x1 = image.shape[1] if index == columns - 1 else x0 + panel_w
    return image[:, x0:x1].copy()


def baseline_mask_path(root: Path, case: QualCase, baseline: str) -> Path:
    if baseline == "dr_splat":
        scene_dir = DR_SPLAT_SCENE_DIR[case.scene]
        return (
            root
            / scene_dir
            / "predictions_mask_0.4"
            / "renders_silhouette"
            / f"frame_{case.frame_id}"
            / f"{case.query}.png"
        )
    if baseline == "gags":
        return (
            root
            / case.scene
            / "train"
            / "ours_30000"
            / "eval"
            / case.frame_id
            / "heatmap"
            / f"{case.query}.png"
        )
    raise ValueError(f"Unsupported baseline: {baseline}")


def ours_mask_path(root: Path, case: QualCase, selection: str) -> Path:
    return root / "pred_masks" / selection / case.scene / f"frame_{case.frame_id}_{case.query}.png"


def langsplat_2d_mask_path(root: Path, case: QualCase) -> Path:
    return root / "eval" / case.scene / case.frame_id / f"chosen_{case.query}.png"


def dr_splat_3d_render_path(root: Path, case: QualCase) -> Path:
    scene_dir = DR_SPLAT_SCENE_DIR[case.scene]
    return (
        root
        / scene_dir
        / "predictions_mask_0.4"
        / "renders"
        / f"frame_{case.frame_id}"
        / f"{case.query}.png"
    )


def ctf_gs_2d_visual_path(root: Path, case: QualCase) -> Path:
    candidates = [
        root
        / f"lerf_{case.scene}_overlay_calibrated_thr0p60_vis_20260514"
        / "visualisations"
        / case.scene
        / f"lerf_grounding_frame_{case.frame_id}_rendered_{query_slug(case.query)}.png",
        root
        / f"lerf_{case.scene}_overlay_20260502"
        / "visualisations"
        / case.scene
        / f"lerf_grounding_frame_{case.frame_id}_rendered_{query_slug(case.query)}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_case_masks(
    case: QualCase,
    *,
    label_root: Path,
    baseline_root: Path,
    baseline: str,
    ours_root: Path,
    ours_selection: str,
) -> dict[str, object]:
    rgb_path = label_root / case.scene / f"frame_{case.frame_id}.jpg"
    label_json = label_root / case.scene / f"frame_{case.frame_id}.json"
    baseline_path = baseline_mask_path(baseline_root, case, baseline)
    ours_path = ours_mask_path(ours_root, case, ours_selection)

    rgb = read_image(rgb_path)
    gt = build_gt_mask(label_json, case.query)
    baseline_pred = resize_mask(load_binary_mask(baseline_path), gt.shape)
    ours_pred = resize_mask(load_binary_mask(ours_path), gt.shape)
    return {
        "rgb": rgb,
        "gt": gt,
        "baseline_pred": baseline_pred,
        "ours_pred": ours_pred,
        "rgb_path": rgb_path,
        "label_json": label_json,
        "baseline_path": baseline_path,
        "ours_path": ours_path,
        "baseline_iou": compute_iou(gt, baseline_pred),
        "ours_iou": compute_iou(gt, ours_pred),
    }


def make_case_row(
    case: QualCase,
    *,
    label_root: Path,
    baseline_root: Path,
    baseline: str,
    baseline_label: str,
    ours_root: Path,
    ours_selection: str,
    panel_width: int,
    panel_height: int,
) -> tuple[np.ndarray, dict[str, object]]:
    data = load_case_masks(
        case,
        label_root=label_root,
        baseline_root=baseline_root,
        baseline=baseline,
        ours_root=ours_root,
        ours_selection=ours_selection,
    )
    rgb = data["rgb"]
    gt = data["gt"]
    baseline_pred = data["baseline_pred"]
    ours_pred = data["ours_pred"]
    baseline_iou = float(data["baseline_iou"])
    ours_iou = float(data["ours_iou"])

    rgb_panel = put_text_box(
        fit_panel(rgb, width=panel_width, height=panel_height),
        [case.scene.replace("_", " "), case.query],
        origin=(10, 10),
        font_scale=0.56,
    )
    gt_panel = fit_panel(overlay_mask(rgb, gt, (70, 190, 80)), width=panel_width, height=panel_height)
    baseline_panel = fit_panel(
        overlay_mask(rgb, baseline_pred, (55, 125, 230)),
        width=panel_width,
        height=panel_height,
    )
    baseline_panel = put_text_box(
        baseline_panel,
        [f"IoU {baseline_iou:.3f}"],
        origin=(10, 10),
        font_scale=0.52,
    )
    ours_panel = fit_panel(
        overlay_mask(rgb, ours_pred, (220, 145, 50)),
        width=panel_width,
        height=panel_height,
    )
    ours_panel = put_text_box(ours_panel, [f"IoU {ours_iou:.3f}"], origin=(10, 10), font_scale=0.52)

    row = np.hstack([rgb_panel, gt_panel, baseline_panel, ours_panel])
    manifest = {
        "scene": case.scene,
        "frame_id": case.frame_id,
        "query": case.query,
        "baseline": baseline_label,
        "baseline_iou": round(baseline_iou, 4),
        "ours_iou": round(ours_iou, 4),
        "rgb": rel_or_str(data["rgb_path"]),
        "label_json": rel_or_str(data["label_json"]),
        "baseline_mask": rel_or_str(data["baseline_path"]),
        "ours_mask": rel_or_str(data["ours_path"]),
    }
    return row, manifest


def load_ovs_2d3d_case(
    case: QualCase,
    *,
    label_root: Path,
    baseline_2d_root: Path,
    baseline_3d_root: Path,
    ours_2d_root: Path,
    ours_3d_root: Path,
    ours_3d_selection: str,
) -> dict[str, object]:
    rgb_path = label_root / case.scene / f"frame_{case.frame_id}.jpg"
    label_json = label_root / case.scene / f"frame_{case.frame_id}.json"
    baseline_2d_path = langsplat_2d_mask_path(baseline_2d_root, case)
    baseline_3d_mask_path = baseline_mask_path(baseline_3d_root, case, "dr_splat")
    baseline_3d_render_path = dr_splat_3d_render_path(baseline_3d_root, case)
    ours_2d_path = ctf_gs_2d_visual_path(ours_2d_root, case)
    ours_3d_path = ours_mask_path(ours_3d_root, case, ours_3d_selection)

    rgb = read_image(rgb_path)
    gt = build_gt_mask(label_json, case.query)
    baseline_2d_mask = resize_mask(load_binary_mask(baseline_2d_path), gt.shape)
    baseline_3d_mask = resize_mask(load_binary_mask(baseline_3d_mask_path), gt.shape)
    ours_3d_mask = resize_mask(load_binary_mask(ours_3d_path), gt.shape)
    ours_2d_visual = crop_wide_visual_panel(read_image(ours_2d_path), index=5, columns=6)
    baseline_3d_render = read_image(baseline_3d_render_path)
    if baseline_3d_render.shape[:2] != gt.shape:
        baseline_3d_render = cv2.resize(
            baseline_3d_render,
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return {
        "rgb": rgb,
        "gt": gt,
        "baseline_2d_mask": baseline_2d_mask,
        "baseline_3d_mask": baseline_3d_mask,
        "baseline_3d_render": baseline_3d_render,
        "ours_2d_visual": ours_2d_visual,
        "ours_3d_mask": ours_3d_mask,
        "paths": {
            "rgb": rgb_path,
            "label_json": label_json,
            "baseline_2d_mask": baseline_2d_path,
            "baseline_3d_mask": baseline_3d_mask_path,
            "baseline_3d_render": baseline_3d_render_path,
            "ours_2d_visual": ours_2d_path,
            "ours_3d_mask": ours_3d_path,
        },
        "baseline_2d_iou": compute_iou(gt, baseline_2d_mask),
        "baseline_3d_iou": compute_iou(gt, baseline_3d_mask),
        "ours_3d_iou": compute_iou(gt, ours_3d_mask),
    }


def make_ovs_2d3d_scene_block(
    cases: list[QualCase],
    *,
    label_root: Path,
    baseline_2d_root: Path,
    baseline_3d_root: Path,
    ours_2d_root: Path,
    ours_3d_root: Path,
    ours_3d_selection: str,
    panel_width: int,
    panel_height: int,
    reference_width: int,
    row_label_width: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if not cases:
        raise ValueError("Expected at least one case for an OVS scene block")
    scene = cases[0].scene
    if any(case.scene != scene for case in cases):
        raise ValueError("All OVS cases in a scene block must share the same scene")

    header_h = 34
    query_h = 34
    margin = 12
    block_w = reference_width + row_label_width + len(cases) * 2 * panel_width + 2 * margin
    block_h = header_h + 2 * panel_height + query_h + 2 * margin
    canvas = np.full((block_h, block_w, 3), 255, dtype=np.uint8)
    border_color = SCENE_COLORS.get(scene, (226, 226, 226))
    cv2.rectangle(canvas, (2, 2), (block_w - 3, block_h - 3), border_color, 3, cv2.LINE_AA)

    first_rgb = read_image(label_root / scene / f"frame_{cases[0].frame_id}.jpg")
    ref_panel = fit_panel(first_rgb, width=reference_width - 18, height=panel_height - 6)
    cv2.putText(
        canvas,
        scene.replace("_", " ").title(),
        (margin + 4, margin + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (210, 105, 65) if scene == "figurines" else (80, 160, 65),
        1,
        cv2.LINE_AA,
    )
    ref_y = margin + header_h + (2 * panel_height - ref_panel.shape[0]) // 2 - 4
    canvas[ref_y : ref_y + ref_panel.shape[0], margin + 4 : margin + 4 + ref_panel.shape[1]] = ref_panel
    cv2.putText(
        canvas,
        "GT RGB",
        (margin + 36, ref_y + ref_panel.shape[0] + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )

    x0 = margin + reference_width
    baseline_y = margin + header_h
    ours_y = baseline_y + panel_height
    cv2.putText(canvas, "Prior", (x0 + 4, baseline_y + panel_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Ours", (x0 + 4, ours_y + panel_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)

    manifests: list[dict[str, object]] = []
    for idx, case in enumerate(cases):
        data = load_ovs_2d3d_case(
            case,
            label_root=label_root,
            baseline_2d_root=baseline_2d_root,
            baseline_3d_root=baseline_3d_root,
            ours_2d_root=ours_2d_root,
            ours_3d_root=ours_3d_root,
            ours_3d_selection=ours_3d_selection,
        )
        rgb = data["rgb"]
        base_x = x0 + row_label_width + idx * 2 * panel_width
        for col, title in enumerate(["2D OVS", "3D OVS"]):
            tx = base_x + col * panel_width + 48
            cv2.putText(canvas, title, (tx, margin + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (15, 15, 15), 1, cv2.LINE_AA)

        baseline_2d = fit_panel(
            overlay_mask(rgb, data["baseline_2d_mask"], (230, 125, 55)),
            width=panel_width,
            height=panel_height,
        )
        baseline_3d = fit_panel(
            add_zoom_inset(data["baseline_3d_render"], data["baseline_3d_mask"]),
            width=panel_width,
            height=panel_height,
        )
        ours_2d = fit_panel(data["ours_2d_visual"], width=panel_width, height=panel_height)
        ours_3d = fit_panel(
            object_cutout_panel(rgb, data["ours_3d_mask"], color=(50, 145, 220)),
            width=panel_width,
            height=panel_height,
        )

        canvas[baseline_y : baseline_y + panel_height, base_x : base_x + panel_width] = baseline_2d
        canvas[
            baseline_y : baseline_y + panel_height,
            base_x + panel_width : base_x + 2 * panel_width,
        ] = baseline_3d
        canvas[ours_y : ours_y + panel_height, base_x : base_x + panel_width] = ours_2d
        canvas[ours_y : ours_y + panel_height, base_x + panel_width : base_x + 2 * panel_width] = ours_3d
        cv2.rectangle(canvas, (base_x, baseline_y), (base_x + 2 * panel_width, ours_y + panel_height), (230, 230, 230), 1)
        query_text = f'"{case.query.title()}"'
        text_size = cv2.getTextSize(query_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
        qx = base_x + panel_width - text_size[0] // 2
        qy = ours_y + panel_height + 24
        cv2.putText(canvas, query_text, (qx, qy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)

        paths = data["paths"]
        manifests.append(
            {
                "scene": case.scene,
                "frame_id": case.frame_id,
                "query": case.query,
                "baseline_2d": "LangSplat (repro.)",
                "baseline_3d": "Dr. Splat (repro.)",
                "baseline_2d_iou": round(float(data["baseline_2d_iou"]), 4),
                "baseline_3d_iou": round(float(data["baseline_3d_iou"]), 4),
                "ours_3d_iou": round(float(data["ours_3d_iou"]), 4),
                "rgb": rel_or_str(paths["rgb"]),
                "label_json": rel_or_str(paths["label_json"]),
                "baseline_2d_mask": rel_or_str(paths["baseline_2d_mask"]),
                "baseline_3d_mask": rel_or_str(paths["baseline_3d_mask"]),
                "baseline_3d_render": rel_or_str(paths["baseline_3d_render"]),
                "ours_2d_visual": rel_or_str(paths["ours_2d_visual"]),
                "ours_3d_mask": rel_or_str(paths["ours_3d_mask"]),
            }
        )

    return canvas, manifests


def write_ovs_2d3d_markdown(path: Path, manifest: dict[str, object]) -> None:
    rows = [
        "# LERF 2D/3D OVS Qualitative Figure",
        "",
        f"- Figure: `{manifest['figure']}`",
        "- 2D reproduced prior: `LangSplat` rendered-view mask.",
        "- 3D reproduced prior: `Dr. Splat` direct-selection render and silhouette.",
        f"- Ours 3D source root: `{manifest['ours_3d_root']}`",
        "",
        "| Scene | Frame | Query | Prior 2D IoU | Prior 3D IoU | Ours 3D IoU |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for case in manifest["cases"]:
        rows.append(
            "| {scene} | {frame_id} | `{query}` | {baseline_2d_iou:.4f} | {baseline_3d_iou:.4f} | {ours_3d_iou:.4f} |".format(
                **case
            )
        )
    rows.extend(
        [
            "",
            "Protocol note: 2D panels visualize rendered-view OVS masks/heatmaps on RGB, "
            "while 3D panels visualize primitives selected by direct Gaussian-level query "
            "and rendered/cut out on a white background. The CTF-GS 3D panels use the "
            "compact direct-field mask source only; no VPR cache or official RGB SAM3 "
            "decoder is called when producing these panels.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_markdown_summary(path: Path, manifest: dict[str, object]) -> None:
    rows = [
        "# LERF Main Qualitative Comparison",
        "",
        f"- Figure: `{manifest['figure']}`",
        f"- Baseline: `{manifest['baseline']}`",
        f"- Ours: `{manifest['ours']}`",
        f"- Ours source root: `{manifest['ours_root']}`",
        "",
        "| Scene | Frame | Query | Baseline IoU | Ours IoU |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for case in manifest["cases"]:
        rows.append(
            "| {scene} | {frame_id} | `{query}` | {baseline_iou:.4f} | {ours_iou:.4f} |".format(
                **case
            )
        )
    rows.extend(
        [
            "",
            "Protocol note: the Ours panels use compact direct-field primitive scores "
            "with a frozen SigLIP2 prompt ensemble, opacity-gated point-adapter "
            "blending, and support-aware component cleanup. No VPR cache or official "
            "RGB SAM3 readout is used for these panels.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=["direct_grid", "ovs_2d3d"], default="direct_grid")
    parser.add_argument("--label_root", default="/mnt/pool/sqy/3d_understanding/lerf_ovs/label")
    parser.add_argument("--baseline", choices=["dr_splat", "gags"], default="dr_splat")
    parser.add_argument("--baseline_label", default="Dr. Splat (repro.)")
    parser.add_argument("--baseline_root", default="output/baselines/dr_splat/lerf_compat_20260519")
    parser.add_argument("--baseline_2d_root", default="output/baselines/langsplat/lerf_compat_20260518")
    parser.add_argument("--baseline_3d_root", default="output/baselines/dr_splat/lerf_compat_20260519")
    parser.add_argument("--ours_2d_root", default="output/radio_gs/freeze_eval")
    parser.add_argument("--ours_root", default="output/radio_gs/lerf_direct3d_prompt_ensemble_policy_masks_20260528")
    parser.add_argument("--ours_selection", default="thr0p65")
    parser.add_argument("--ours_label", default="CTF-GS compact")
    parser.add_argument("--case", action="append", default=[], help="Override/add case as scene:frame:query")
    parser.add_argument("--panel_width", type=int, default=300)
    parser.add_argument("--panel_height", type=int, default=214)
    parser.add_argument("--output", default="paper/figures/lerf_main_qualitative_comparison.png")
    parser.add_argument("--manifest", default="paper/artifacts/lerf_main_qualitative_comparison_manifest.json")
    parser.add_argument("--markdown", default="paper/artifacts/lerf_main_qualitative_comparison.md")
    args = parser.parse_args()

    label_root = Path(args.label_root)
    baseline_root = Path(args.baseline_root)
    ours_root = Path(args.ours_root)

    if args.layout == "ovs_2d3d":
        groups = DEFAULT_OVS_2D3D_GROUPS
        blocks: list[np.ndarray] = []
        manifests: list[dict[str, object]] = []
        for group in groups:
            block, block_manifest = make_ovs_2d3d_scene_block(
                group,
                label_root=label_root,
                baseline_2d_root=Path(args.baseline_2d_root),
                baseline_3d_root=Path(args.baseline_3d_root),
                ours_2d_root=Path(args.ours_2d_root),
                ours_3d_root=ours_root,
                ours_3d_selection=args.ours_selection,
                panel_width=args.panel_width,
                panel_height=args.panel_height,
                reference_width=190,
                row_label_width=58,
            )
            blocks.append(block)
            manifests.extend(block_manifest)

        sep = np.full((12, blocks[0].shape[1], 3), 255, dtype=np.uint8)
        grid = np.vstack([item for block in blocks for item in (block, sep)][:-1])
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), grid)

        manifest = {
            "figure": rel_or_str(output),
            "script": "radio_gs/scripts/compose_lerf_main_qualitative.py",
            "layout": "ovs_2d3d",
            "baseline_2d": "LangSplat (repro.)",
            "baseline_2d_root": rel_or_str(Path(args.baseline_2d_root)),
            "baseline_3d": "Dr. Splat (repro.)",
            "baseline_3d_root": rel_or_str(Path(args.baseline_3d_root)),
            "ours_2d": "CTF-GS rendered-view heatmap/RGB",
            "ours_2d_root": rel_or_str(Path(args.ours_2d_root)),
            "ours_3d": args.ours_label,
            "ours_3d_root": rel_or_str(ours_root),
            "ours_3d_selection": args.ours_selection,
            "protocol": (
                "2D panels show rendered-view OVS outputs. 3D panels show direct "
                "Gaussian-level selection outputs rendered or cut out on white. "
                "The Ours 3D panels do not use a VPR cache or official RGB SAM3 readout."
            ),
            "cases": manifests,
        }
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        write_ovs_2d3d_markdown(Path(args.markdown), manifest)
        print(f"wrote {output}")
        print(f"wrote {manifest_path}")
        print(f"wrote {args.markdown}")
        return

    cases = parse_cases(args.case) if args.case else DEFAULT_CASES

    rows: list[np.ndarray] = []
    manifests: list[dict[str, object]] = []
    for case in cases:
        row, case_manifest = make_case_row(
            case,
            label_root=label_root,
            baseline_root=baseline_root,
            baseline=args.baseline,
            baseline_label=args.baseline_label,
            ours_root=ours_root,
            ours_selection=args.ours_selection,
            panel_width=args.panel_width,
            panel_height=args.panel_height,
        )
        rows.append(row)
        manifests.append(case_manifest)

    headers = [
        "RGB + query",
        "GT mask",
        args.baseline_label,
        args.ours_label,
    ]
    header_row = np.hstack(
        [
            add_header(np.full((1, args.panel_width, 3), 255, dtype=np.uint8), header)
            for header in headers
        ]
    )[:38]
    sep = np.full((8, header_row.shape[1], 3), 255, dtype=np.uint8)
    grid = np.vstack([header_row, sep] + rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), grid)

    manifest = {
        "figure": rel_or_str(output),
        "script": "radio_gs/scripts/compose_lerf_main_qualitative.py",
        "baseline": args.baseline_label,
        "baseline_root": rel_or_str(baseline_root),
        "ours": args.ours_label,
        "ours_root": rel_or_str(ours_root),
        "ours_selection": args.ours_selection,
        "ours_protocol": (
            "compact direct-field primitive scores; frozen SigLIP2 prompt ensemble; "
            "opacity-gated point-adapter valid mask; support-aware component guard; "
            "no VPR feature cache or official RGB SAM3 readout at evaluation"
        ),
        "cases": manifests,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_markdown_summary(Path(args.markdown), manifest)
    print(f"wrote {output}")
    print(f"wrote {manifest_path}")
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
