#!/usr/bin/env python3
"""Render LERF 2D heatmap-threshold coarse masks on training views.

These masks provide the spatial prompt distribution used by rendered-view
grounding, unlike direct-3D selected-primitive masks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, ".")

from radio_gs.data.lerf_dataset import LERFDataset
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts.eval_lerf_grounding import (
    DEFAULT_GT_FEATURE_ROOT,
    DEFAULT_LABEL_DIR,
    DEFAULT_PROMPT_TEMPLATES,
    compute_relevancy_heatmap,
    heatmap_to_binary_mask,
    load_lerf_ovs_labels,
    load_or_generate_prompt_ensemble_embeddings,
    load_render_pipeline,
    parse_prompt_templates,
    project_to_siglip2,
    render_1280d,
    resolve_lerf_label_dir,
    resolve_lerf_scene_root,
)


def choose_training_frame_ids(
    available_frames: Iterable[int],
    *,
    excluded: Iterable[int],
    max_frames: int,
) -> list[int]:
    """Select evenly spaced non-labelled frames for pseudo-mask training."""

    excluded_set = {int(frame_id) for frame_id in excluded}
    candidates = [int(frame_id) for frame_id in sorted(set(available_frames)) if int(frame_id) not in excluded_set]
    limit = int(max_frames)
    if limit <= 0 or len(candidates) <= limit:
        return candidates
    if limit == 1:
        return [candidates[0]]
    positions = torch.linspace(0, len(candidates) - 1, steps=limit).round().long().tolist()
    selected: list[int] = []
    for pos in positions:
        frame_id = candidates[int(pos)]
        if frame_id not in selected:
            selected.append(frame_id)
    idx = 0
    while len(selected) < limit and idx < len(candidates):
        frame_id = candidates[idx]
        if frame_id not in selected:
            selected.append(frame_id)
        idx += 1
    return sorted(selected)


def coarse_mask_path(root: str | Path, *, frame_id: int, category: str) -> Path:
    safe_category = str(category).replace("/", "_")
    return Path(root) / f"frame_{int(frame_id):05d}_{safe_category}.png"


def _build_lerf_dataset(scene: str, config: object, label_dir: str | Path) -> LERFDataset:
    scene_root = resolve_lerf_scene_root(scene, getattr(config, "scene_root", ""))
    feature_dir = Path(DEFAULT_GT_FEATURE_ROOT) / scene
    return LERFDataset(
        scene_root=str(scene_root),
        feature_dir=str(feature_dir),
        annotation_dir=str(Path(label_dir) / scene),
        feature_height=int(getattr(config, "feature_height", 30)),
        feature_width=int(getattr(config, "feature_width", 40)),
    )


def render_lerf2d_coarse_masks(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    label_dir = resolve_lerf_label_dir(args.label_dir)
    frame_annotations, scene_categories, img_h, img_w = load_lerf_ovs_labels(label_dir, args.scene)
    render_pipeline = load_render_pipeline(args.config, args.checkpoint, device)
    model, codec, renderer, sharpener, refiner, config, is_hybrid = render_pipeline
    dataset = _build_lerf_dataset(args.scene, config, label_dir)

    if args.frame_ids:
        frame_ids = [
            int(part)
            for part in str(args.frame_ids).replace("\n", ",").split(",")
            if part.strip()
        ]
    else:
        frame_ids = choose_training_frame_ids(
            dataset.pose_by_frame_idx.keys(),
            excluded=frame_annotations.keys(),
            max_frames=int(args.max_frames),
        )
    if not frame_ids:
        raise RuntimeError(f"No training frames selected for {args.scene}")

    proj = SigLIP2SummaryHead.from_extracted_weights(str(args.summary_head_weights)).to(device)
    proj = proj.half() if device.type == "cuda" else proj.float()
    proj.eval()
    text_embeddings = load_or_generate_prompt_ensemble_embeddings(
        list(scene_categories),
        device,
        cache_path=args.text_embedding_cache,
        prompt_templates=parse_prompt_templates(args.prompt_templates),
    )
    text_embeddings = text_embeddings.half() if device.type == "cuda" else text_embeddings.float()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for frame_id in tqdm(frame_ids, desc=f"render LERF2D coarse masks {args.scene}", leave=False):
        pose_w2c = dataset.pose_by_frame_idx.get(int(frame_id))
        if pose_w2c is None:
            continue
        viewmat = torch.from_numpy(pose_w2c.copy()).float().to(device).unsqueeze(0)
        feat_1280 = render_1280d(
            model,
            codec,
            renderer,
            sharpener,
            refiner,
            viewmat,
            is_hybrid=is_hybrid,
            config=config,
            device=device,
            rgb_image=None,
        )
        siglip_feat = project_to_siglip2(feat_1280.half() if device.type == "cuda" else feat_1280.float(), proj)
        heatmaps = compute_relevancy_heatmap(
            siglip_feat,
            text_embeddings,
            temperature=float(args.temperature),
            scoring=args.scoring,
            all_scene_emb=text_embeddings if args.scoring == "softmax_scene" else None,
            active_scene_indices=list(range(len(scene_categories))) if args.scoring == "softmax_scene" else None,
        )
        if int(args.heatmap_upsample) > 1:
            heatmaps = torch.nn.functional.interpolate(
                heatmaps.unsqueeze(0),
                scale_factor=int(args.heatmap_upsample),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        for cat_idx, category in enumerate(scene_categories):
            mask = heatmap_to_binary_mask(
                heatmaps[cat_idx],
                threshold_ratio=float(args.iou_threshold),
                threshold_mode=args.threshold_mode,
                threshold_mean_std_k=float(args.threshold_mean_std_k),
                threshold_min_ratio=float(args.threshold_min_ratio),
                threshold_max_ratio=float(args.threshold_max_ratio),
                target_shape=(img_h, img_w),
            )
            cv2.imwrite(str(coarse_mask_path(out_dir, frame_id=int(frame_id), category=category)), mask.astype("uint8") * 255)
            saved += 1
        del feat_1280, siglip_feat, heatmaps
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "scene": args.scene,
        "frames": [int(frame_id) for frame_id in frame_ids],
        "categories": list(scene_categories),
        "output_dir": str(out_dir),
        "n_saved": int(saved),
        "iou_threshold": float(args.iou_threshold),
        "threshold_mode": args.threshold_mode,
        "temperature": float(args.temperature),
        "scoring": args.scoring,
        "heatmap_upsample": int(args.heatmap_upsample),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--summary_head_weights", default="checkpoints/siglip2_summary_head.pth")
    parser.add_argument("--text_embedding_cache", default="checkpoints/siglip2_lerf_text_embeddings.pt")
    parser.add_argument("--prompt_templates", default=DEFAULT_PROMPT_TEMPLATES)
    parser.add_argument("--frame_ids", default="")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--iou_threshold", type=float, default=0.6)
    parser.add_argument("--threshold_mode", choices=("fixed", "mean_std"), default="fixed")
    parser.add_argument("--threshold_mean_std_k", type=float, default=1.0)
    parser.add_argument("--threshold_min_ratio", type=float, default=0.0)
    parser.add_argument("--threshold_max_ratio", type=float, default=1.0)
    parser.add_argument("--scoring", choices=("cosine", "softmax_scene", "relevancy"), default="softmax_scene")
    parser.add_argument("--temperature", type=float, default=50.0)
    parser.add_argument("--heatmap_upsample", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    print(json.dumps(render_lerf2d_coarse_masks(args), indent=2))


if __name__ == "__main__":
    main()
