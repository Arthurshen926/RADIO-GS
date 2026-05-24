#!/usr/bin/env python3
"""Diagnose official SAM3 state injection and box-prompt conventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    resolve_sam3_amp_dtype,
    sam3_autocast_context,
)
from radio_gs.scripts.eval_lerf_grounding import (
    build_gt_masks,
    load_lerf_ovs_labels,
    load_lerf_rgb_frame,
    resolve_lerf_label_dir,
)


def clone_sam3_state(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_sam3_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_sam3_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_sam3_state(item) for item in value)
    return value


def summarize_tensors(value: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if torch.is_tensor(value):
        tensor = value.detach()
        summary = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
        if tensor.numel():
            tf = tensor.float()
            summary.update(
                {
                    "mean": float(tf.mean().cpu()),
                    "std": float(tf.std(unbiased=False).cpu()),
                    "min": float(tf.min().cpu()),
                    "max": float(tf.max().cpu()),
                }
            )
        out[prefix.rstrip(".")] = summary
    elif isinstance(value, dict):
        for key, item in value.items():
            out.update(summarize_tensors(item, f"{prefix}{key}."))
    elif isinstance(value, (list, tuple)):
        stem = prefix.rstrip(".")
        for idx, item in enumerate(value):
            out.update(summarize_tensors(item, f"{stem}[{idx}]."))
    return out


def binary_mask_iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred_b = np.asarray(pred).astype(bool)
    target_b = np.asarray(target).astype(bool)
    union = np.logical_or(pred_b, target_b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_b, target_b).sum()) / float(union)


def mask_to_sam3_box_prompt(mask: np.ndarray, *, padding_pixels: int = 0) -> list[float] | None:
    pred = np.asarray(mask).astype(bool)
    if pred.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {pred.shape}")
    if not pred.any():
        return None
    height, width = pred.shape
    ys, xs = np.nonzero(pred)
    pad = max(int(padding_pixels), 0)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, width)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, height)
    box_w = max(float(x1 - x0), 1.0)
    box_h = max(float(y1 - y0), 1.0)
    return [
        float(np.clip((float(x0) + box_w * 0.5) / float(max(width, 1)), 0.0, 1.0)),
        float(np.clip((float(y0) + box_h * 0.5) / float(max(height, 1)), 0.0, 1.0)),
        float(np.clip(box_w / float(max(width, 1)), 0.0, 1.0)),
        float(np.clip(box_h / float(max(height, 1)), 0.0, 1.0)),
    ]


def sam3_box_prompt_to_xyxy_pixels(box: list[float], *, height: int, width: int) -> list[float]:
    cx, cy, bw, bh = [float(v) for v in box]
    abs_w = bw * float(width)
    abs_h = bh * float(height)
    x0 = cx * float(width) - abs_w * 0.5
    y0 = cy * float(height) - abs_h * 0.5
    return [x0, y0, x0 + abs_w, y0 + abs_h]


def best_output_mask(output: dict[str, Any], height: int, width: int) -> np.ndarray:
    masks = output.get("masks")
    if masks is None:
        logits = output.get("masks_logits")
        if torch.is_tensor(logits):
            masks = logits.float() > 0.5
    if masks is None or not torch.is_tensor(masks) or masks.numel() == 0:
        return np.zeros((height, width), dtype=bool)
    if masks.dim() == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    scores = output.get("scores")
    idx = int(scores.argmax().item()) if torch.is_tensor(scores) and scores.numel() else 0
    return masks[idx].detach().cpu().numpy().astype(bool)


def output_mask_summary(output: dict[str, Any], height: int, width: int) -> dict[str, Any]:
    mask = best_output_mask(output, height, width)
    scores = output.get("scores")
    best_score = 0.0
    if torch.is_tensor(scores) and scores.numel():
        best_score = float(scores.detach().float().max().cpu())
    masks = output.get("masks")
    if masks is None:
        masks = output.get("masks_logits")
    return {
        "best_area": int(mask.sum()),
        "best_score": best_score,
        "num_masks": int(masks.shape[0]) if torch.is_tensor(masks) else 0,
    }


def _load_rgb_image(scene: str, frame_id: int, scene_root_hint: str = "") -> Image.Image:
    bgr = load_lerf_rgb_frame(scene, frame_id, scene_root_hint)
    if bgr is None:
        raise FileNotFoundError(f"Missing RGB frame: {scene}/{frame_id}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.ascontiguousarray(rgb))


def _run_text(processor: Any, state: dict[str, Any], query: str) -> dict[str, Any]:
    return processor.set_text_prompt(query, clone_sam3_state(state))


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    image = _load_rgb_image(args.scene, int(args.frame_id), args.scene_root)
    width, height = image.size
    processor = _load_sam3_model(
        checkpoint_path=args.sam3_checkpoint_path,
        device=args.device,
        confidence_threshold=args.sam3_confidence_threshold,
        dtype=args.sam3_dtype,
        resolution=args.sam3_resolution,
    )
    amp_dtype = resolve_sam3_amp_dtype(str(processor.device), args.sam3_amp_dtype)
    with torch.no_grad(), sam3_autocast_context(str(processor.device), amp_dtype):
        image_state = processor.set_image(image)
        baseline = _run_text(processor, image_state, args.query)
        identity = _run_text(processor, image_state, args.query)

    baseline_mask = best_output_mask(baseline, height, width)
    identity_mask = best_output_mask(identity, height, width)
    report: dict[str, Any] = {
        "scene": args.scene,
        "frame_id": int(args.frame_id),
        "query": args.query,
        "image_size": [height, width],
        "identity_iou": binary_mask_iou(baseline_mask, identity_mask),
        "baseline_mask": output_mask_summary(baseline, height, width),
        "identity_mask": output_mask_summary(identity, height, width),
        "state_summary": summarize_tensors(image_state),
        "box_format": "normalized_cxcywh",
    }

    if args.gt_box_category:
        label_dir = resolve_lerf_label_dir(args.label_dir)
        annotations_by_frame, _, _, _ = load_lerf_ovs_labels(label_dir, args.scene)
        annotations = annotations_by_frame.get(int(args.frame_id), [])
        gt_masks = build_gt_masks(annotations, [args.gt_box_category], height, width)
        gt_mask = gt_masks.get(args.gt_box_category)
        if gt_mask is None or not np.asarray(gt_mask).astype(bool).any():
            report["gt_box_status"] = "missing_or_empty_gt"
        else:
            box = mask_to_sam3_box_prompt(
                gt_mask,
                padding_pixels=args.gt_box_padding_pixels,
            )
            report["gt_box_prompt_cxcywh_norm"] = box
            report["gt_box_prompt_xyxy_pixels"] = (
                sam3_box_prompt_to_xyxy_pixels(box, height=height, width=width) if box else None
            )
            with torch.no_grad(), sam3_autocast_context(str(processor.device), amp_dtype):
                box_output = processor.add_geometric_prompt(box, True, clone_sam3_state(image_state))
            box_mask = best_output_mask(box_output, height, width)
            report["gt_box_status"] = "ok"
            report["gt_box_iou"] = binary_mask_iou(box_mask, gt_mask)
            report["gt_box_mask"] = output_mask_summary(box_output, height, width)
            report["gt_area"] = int(np.asarray(gt_mask).astype(bool).sum())

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frame_id", type=int, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--scene_root", default="")
    parser.add_argument("--label_dir", default="")
    parser.add_argument("--gt_box_category", default="")
    parser.add_argument("--gt_box_padding_pixels", type=int, default=0)
    parser.add_argument("--sam3_checkpoint_path", default="checkpoints/sam3_modelscope/sam3.pt")
    parser.add_argument("--sam3_resolution", type=int, default=1008)
    parser.add_argument("--sam3_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--sam3_dtype", default="auto")
    parser.add_argument("--sam3_amp_dtype", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_diagnostic(_parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
