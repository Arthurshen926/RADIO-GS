#!/usr/bin/env python3
"""Train/evaluate a RADIO -> official SAM3 decoder bridge on LERF frames.

This is the feature-to-decoder experiment: RADIO or CTF-GS rendered 1280-d
features are projected into the official SAM3 ``backbone_out`` tensors, then the
frozen official SAM3 grounding decoder is called without using SAM3 RGB features
as the mask readout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")

from radio_gs.data.lerf_dataset import LERFDataset
from radio_gs.models.sam3_decoder_bridge import Sam3BackboneBridge, sam3_backbone_bridge_loss
from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    resolve_sam3_amp_dtype,
    sam3_autocast_context,
)
from radio_gs.scripts.eval_lerf_adaptor_downstream import _render_feature
from radio_gs.scripts.eval_lerf_grounding import (
    DEFAULT_LABEL_DIR,
    build_gt_masks,
    load_lerf_ovs_labels,
    load_lerf_rgb_frame,
    load_render_pipeline,
    resolve_lerf_label_dir,
)


def _frame_id_from_path(path: Path) -> int | None:
    match = re.search(r"rgb_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


def _available_feature_frames(feature_dir: Path) -> list[int]:
    frames: list[int] = []
    for path in sorted(feature_dir.glob("rgb_*.pt")):
        frame_id = _frame_id_from_path(path)
        if frame_id is not None:
            frames.append(frame_id)
    return frames


def _load_pil_rgb(scene: str, frame_id: int, scene_root_hint: str | Path = "") -> Image.Image:
    image = load_lerf_rgb_frame(scene, frame_id, scene_root_hint)
    if image is None:
        raise FileNotFoundError(f"Missing RGB frame {scene}/{frame_id}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _load_source_feature(
    *,
    source: str,
    scene: str,
    frame_id: int,
    device: torch.device,
    feature_dir: Path,
    render_pipeline: tuple | None,
    lerf_dataset: LERFDataset | None,
) -> torch.Tensor:
    if source == "teacher":
        path = feature_dir / f"rgb_{frame_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing RADIO feature {path}")
        feature = torch.load(path, map_location=device).float()
        if feature.ndim == 3:
            feature = feature.unsqueeze(0)
        return feature
    if source == "rendered":
        if render_pipeline is None or lerf_dataset is None:
            raise ValueError("rendered source requires --config and --checkpoint")
        return _render_feature(scene, frame_id, render_pipeline, lerf_dataset, device).float()
    raise ValueError(f"Unsupported source: {source}")


def _target_backbone_from_state(state: dict[str, Any]) -> dict[str, Any]:
    backbone = state["backbone_out"]
    return {
        "vision_features": backbone["vision_features"].detach().float(),
        "backbone_fpn": [tensor.detach().float() for tensor in backbone["backbone_fpn"]],
    }


def _bridge_backbone(
    base_backbone: dict[str, Any],
    bridge_out: dict[str, Any],
) -> dict[str, Any]:
    dtype = base_backbone["vision_features"].dtype
    out = dict(base_backbone)
    out["vision_features"] = bridge_out["vision_features"].to(dtype=dtype)
    out["backbone_fpn"] = [tensor.to(dtype=dtype) for tensor in bridge_out["backbone_fpn"]]
    return out


def _mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred_b, gt_b).sum()) / float(union)


def _best_mask(output: dict[str, Any], height: int, width: int) -> np.ndarray:
    scores = output.get("scores")
    masks = output.get("masks")
    if masks is None or masks.numel() == 0:
        return np.zeros((height, width), dtype=bool)
    if masks.dim() == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    idx = int(scores.argmax().item()) if torch.is_tensor(scores) and scores.numel() else 0
    return masks[idx].detach().cpu().numpy().astype(bool)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / max(len(values), 1))


def _normalise_sam3_dtype_name(dtype: str) -> str:
    aliases = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "none": "off",
    }
    return aliases.get(str(dtype), str(dtype))


def pick_frames(
    *,
    scene: str,
    feature_dir: Path,
    label_frames: list[int],
    max_train_frames: int,
    max_eval_frames: int,
) -> tuple[list[int], list[int]]:
    eval_frames = label_frames[: max(1, int(max_eval_frames))]
    eval_set = set(eval_frames)
    train_candidates = [
        frame_id
        for frame_id in _available_feature_frames(feature_dir)
        if frame_id not in eval_set and load_lerf_rgb_frame(scene, frame_id) is not None
    ]
    train_frames = train_candidates[: max(1, int(max_train_frames))]
    if not train_frames:
        raise RuntimeError(f"No train frames with RADIO features and RGB found for {scene}")
    return train_frames, eval_frames


def train_bridge(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    label_dir = resolve_lerf_label_dir(args.label_dir)
    frame_annotations, _, _, _ = load_lerf_ovs_labels(label_dir, args.scene)
    label_frames = sorted(frame_annotations)
    feature_dir = Path(args.radio_feature_dir)
    train_frames, eval_frames = pick_frames(
        scene=args.scene,
        feature_dir=feature_dir,
        label_frames=label_frames,
        max_train_frames=args.max_train_frames,
        max_eval_frames=args.max_eval_frames,
    )

    processor = _load_sam3_model(
        checkpoint_path=args.sam3_checkpoint_path,
        device=str(device),
        confidence_threshold=args.sam3_confidence_threshold,
        dtype=_normalise_sam3_dtype_name(args.sam3_dtype),
        resolution=args.sam3_resolution,
    )
    amp_dtype = resolve_sam3_amp_dtype(
        str(processor.device),
        _normalise_sam3_dtype_name(args.sam3_amp_dtype),
    )
    bridge = Sam3BackboneBridge(input_dim=args.input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    render_pipeline = None
    lerf_dataset = None
    if args.source == "rendered":
        render_pipeline = load_render_pipeline(args.config, args.checkpoint, device)
        lerf_dataset = LERFDataset(render_pipeline[5])

    loss_history: list[dict[str, float]] = []
    bridge.train()
    for epoch in range(int(args.epochs)):
        for frame_id in train_frames:
            image = _load_pil_rgb(args.scene, frame_id)
            feature = _load_source_feature(
                source=args.source,
                scene=args.scene,
                frame_id=frame_id,
                device=device,
                feature_dir=feature_dir,
                render_pipeline=render_pipeline,
                lerf_dataset=lerf_dataset,
            )
            with torch.no_grad(), sam3_autocast_context(str(processor.device), amp_dtype):
                state = processor.set_image(image)
                target = _target_backbone_from_state(state)
            optimizer.zero_grad(set_to_none=True)
            pred = bridge(feature)
            loss, stats = sam3_backbone_bridge_loss(pred, target, cosine_weight=args.cosine_weight)
            loss.backward()
            optimizer.step()
            stats.update({"epoch": float(epoch), "frame_id": float(frame_id)})
            loss_history.append(stats)
            del feature, pred, target, state
            if device.type == "cuda":
                torch.cuda.empty_cache()

    bridge.eval()
    eval_rows: list[dict[str, Any]] = []
    for frame_id in eval_frames:
        image = _load_pil_rgb(args.scene, frame_id)
        feature = _load_source_feature(
            source=args.source,
            scene=args.scene,
            frame_id=frame_id,
            device=device,
            feature_dir=feature_dir,
            render_pipeline=render_pipeline,
            lerf_dataset=lerf_dataset,
        )
        with torch.no_grad(), sam3_autocast_context(str(processor.device), amp_dtype):
            base_state = processor.set_image(image)
            bridge_out = bridge(feature)
        height = int(base_state["original_height"])
        width = int(base_state["original_width"])
        categories = sorted({obj["category"] for obj in frame_annotations[frame_id]})
        gt_masks = build_gt_masks(frame_annotations[frame_id], categories, height, width)
        base_backbone = base_state["backbone_out"]
        bridge_backbone = _bridge_backbone(base_backbone, bridge_out)
        for category in categories:
            with torch.no_grad(), sam3_autocast_context(str(processor.device), amp_dtype):
                official_state = {
                    "original_height": height,
                    "original_width": width,
                    "backbone_out": dict(base_backbone),
                }
                official_out = processor.set_text_prompt(category, official_state)
                bridge_state = {
                    "original_height": height,
                    "original_width": width,
                    "backbone_out": dict(bridge_backbone),
                }
                bridge_pred = processor.set_text_prompt(category, bridge_state)
            gt = gt_masks[category].astype(bool)
            official_mask = _best_mask(official_out, height, width)
            bridge_mask = _best_mask(bridge_pred, height, width)
            eval_rows.append(
                {
                    "frame_id": frame_id,
                    "category": category,
                    "official_rgb_iou": _mask_iou(official_mask, gt),
                    "bridge_iou": _mask_iou(bridge_mask, gt),
                    "official_rgb_area": int(official_mask.sum()),
                    "bridge_area": int(bridge_mask.sum()),
                    "gt_area": int(gt.sum()),
                }
            )
        del feature, bridge_out, base_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene": args.scene,
        "source": args.source,
        "train_frames": train_frames,
        "eval_frames": eval_frames,
        "epochs": int(args.epochs),
        "loss_final": loss_history[-1] if loss_history else {},
        "loss_history_tail": loss_history[-10:],
        "metrics": {
            "official_rgb_miou": _mean(row["official_rgb_iou"] for row in eval_rows),
            "bridge_miou": _mean(row["bridge_iou"] for row in eval_rows),
            "query_count": len(eval_rows),
        },
        "rows": eval_rows,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save({"bridge": bridge.state_dict(), "args": vars(args)}, output_path.with_suffix(".pth"))
    print(json.dumps(payload["metrics"], indent=2))
    print(f"saved {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="figurines")
    parser.add_argument("--source", choices=["teacher", "rendered"], default="teacher")
    parser.add_argument("--config", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--radio_feature_dir", default="output/radio_features_lerf/figurines/backbone")
    parser.add_argument("--label_dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--sam3_checkpoint_path", default="checkpoints/sam3_modelscope/sam3.pt")
    parser.add_argument("--sam3_resolution", type=int, default=1008)
    parser.add_argument("--sam3_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--sam3_dtype", choices=["auto", "float32", "bfloat16", "fp32", "bf16"], default="auto")
    parser.add_argument("--sam3_amp_dtype", choices=["auto", "off", "bfloat16", "none", "bf16"], default="auto")
    parser.add_argument("--input_dim", type=int, default=1280)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_train_frames", type=int, default=4)
    parser.add_argument("--max_eval_frames", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--cosine_weight", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    return parser


def main() -> None:
    train_bridge(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
