#!/usr/bin/env python3
"""Compose a paper-facing VPR direct-3D qualitative grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CASES = [
    ("figurines", "00152", "green apple"),
    ("ramen", "00024", "wavy noodles"),
    ("teatime", "00025", "coffee mug"),
    ("waldo_kitchen", "00089", "knife"),
]


def rel_or_str(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_rgb(path: Path) -> np.ndarray:
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


def load_pred_mask(mask_root: Path, selection: str, scene: str, frame_id: str, query: str) -> np.ndarray:
    path = mask_root / "pred_masks" / selection / scene / f"frame_{frame_id}_{query}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image > 127


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = rgb.copy()
    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    out[mask] = (0.45 * out[mask].astype(np.float32) + 0.55 * color_arr).astype(np.uint8)
    return out


def error_map(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    tp = gt & pred
    fp = ~gt & pred
    fn = gt & ~pred
    out[tp] = (60, 190, 90)
    out[fp] = (40, 40, 230)
    out[fn] = (230, 90, 40)
    return out


def add_title(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    pad = 40 if subtitle else 28
    canvas = np.full((image.shape[0] + pad, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[pad:, :, :] = image
    cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (65, 65, 65), 1, cv2.LINE_AA)
    return canvas


def resize_panel(image: np.ndarray, width: int = 300) -> np.ndarray:
    scale = width / float(image.shape[1])
    height = int(round(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    pad = np.full((height - image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
    return np.vstack([image, pad])


def compute_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    union = np.logical_or(gt, pred).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(gt, pred).sum() / union)


def make_case(
    scene: str,
    frame_id: str,
    query: str,
    *,
    label_root: Path,
    mask_root: Path,
    selection: str,
    selection_label: str,
) -> tuple[np.ndarray, dict[str, object]]:
    rgb_path = label_root / scene / f"frame_{frame_id}.jpg"
    json_path = label_root / scene / f"frame_{frame_id}.json"
    rgb = read_rgb(rgb_path)
    gt = build_gt_mask(json_path, query)
    pred = load_pred_mask(mask_root, selection, scene, frame_id, query)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    iou = compute_iou(gt, pred)

    panels = [
        add_title(resize_panel(rgb), scene.replace("_", " "), query),
        add_title(resize_panel(overlay_mask(rgb, gt, (40, 190, 60))), "GT mask"),
        add_title(resize_panel(overlay_mask(rgb, pred, (45, 120, 230))), selection_label),
        add_title(resize_panel(error_map(rgb, gt, pred)), f"error IoU={iou:.3f}", "green TP / red FP / blue FN"),
    ]
    height = max(panel.shape[0] for panel in panels)
    row = np.hstack([pad_to_height(panel, height) for panel in panels])
    manifest = {
        "scene": scene,
        "frame_id": frame_id,
        "query": query,
        "iou": round(iou, 4),
        "rgb": rel_or_str(rgb_path),
        "pred_mask": rel_or_str(mask_root / "pred_masks" / selection / scene / f"frame_{frame_id}_{query}.png"),
    }
    return row, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask_root", default="output/radio_gs/lerf_direct_3d_selection_vpr_masks96_vm80b50_20260513")
    parser.add_argument("--label_root", default="/mnt/pool/sqy/3d_understanding/lerf_ovs/label")
    parser.add_argument("--selection", default="top0p02")
    parser.add_argument("--selection_label", default="VPR top2%")
    parser.add_argument("--protocol", default="96-view VPR + voxel_max(res=80, blend=0.50)")
    parser.add_argument("--output", default="paper/figures/lerf_vpr_direct_3d_qualitative.png")
    parser.add_argument("--manifest", default="output/radio_gs/reports/lerf_vpr_direct_3d_qualitative_manifest.json")
    args = parser.parse_args()

    mask_root = Path(args.mask_root)
    label_root = Path(args.label_root)
    rows: list[np.ndarray] = []
    manifests: list[dict[str, object]] = []
    for scene, frame_id, query in DEFAULT_CASES:
        row, manifest = make_case(
            scene,
            frame_id,
            query,
            label_root=label_root,
            mask_root=mask_root,
            selection=args.selection,
            selection_label=args.selection_label,
        )
        rows.append(row)
        manifests.append(manifest)

    width = max(row.shape[1] for row in rows)
    rows = [np.hstack([row, np.full((row.shape[0], width - row.shape[1], 3), 255, dtype=np.uint8)]) for row in rows]
    grid = np.vstack(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), grid)

    manifest = {
        "figure": str(output),
        "mask_root": str(mask_root),
        "label_root": str(label_root),
        "selection": args.selection,
        "protocol": args.protocol,
        "cases": manifests,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
