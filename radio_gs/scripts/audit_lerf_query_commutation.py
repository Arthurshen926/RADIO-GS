#!/usr/bin/env python3
"""Compare decode-after-splat with primitive score-first rendering on LERF.

This is a diagnostic, not a benchmark replacement.  It uses one frozen model,
the same text bank, the same cameras, and no GT-dependent tuning.  Ground truth
is opened only after all heatmaps have been produced to report comparable
category/frame IoU at the frozen relevance threshold 0.5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from radio_gs.data.lerf_dataset import LERFDataset
from radio_gs.evaluation.openclip_readout import NEGATIVE_PROMPTS
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    compute_gaussian_text_scores,
    load_text_projection_head,
)
from radio_gs.scripts.eval_lerf_grounding import (
    build_gt_masks,
    compute_relevancy_heatmap,
    load_lerf_ovs_labels,
    load_or_generate_prompt_ensemble_embeddings,
    load_render_pipeline,
    project_to_siglip2,
    render_1280d,
    resolve_lerf_scene_root,
)


def _iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return float(intersection / union) if union else 1.0


def _pair_stats(left: torch.Tensor, right: torch.Tensor, visible: torch.Tensor) -> dict:
    a = left[:, visible].float()
    b = right[:, visible].float()
    if a.numel() == 0:
        raise ValueError("No visible pixels for commutation audit")
    centered_a = a - a.mean(dim=1, keepdim=True)
    centered_b = b - b.mean(dim=1, keepdim=True)
    pearson = F.cosine_similarity(centered_a, centered_b, dim=1)
    cosine = F.cosine_similarity(a, b, dim=1)
    binary_a = a >= 0.5
    binary_b = b >= 0.5
    intersection = (binary_a & binary_b).sum(dim=1).float()
    union = (binary_a | binary_b).sum(dim=1).float()
    mask_iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return {
        "mae": float((a - b).abs().mean()),
        "mean_query_cosine": float(cosine.mean()),
        "mean_query_pearson": float(pearson.mean()),
        "mean_threshold_mask_iou": float(mask_iou.mean()),
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pipeline = load_render_pipeline(
        args.config, args.checkpoint, device, strict_checkpoint_contract=True
    )
    model, codec, renderer, sharpener, refiner, config, is_hybrid = pipeline
    annotations, categories, image_height, image_width = load_lerf_ovs_labels(
        args.label_dir, args.scene
    )
    scene_root = resolve_lerf_scene_root(args.scene, getattr(config, "scene_root", ""))
    dataset = LERFDataset(
        scene_root=str(scene_root),
        feature_dir=str(Path("output/radio_features_lerf") / args.scene),
        annotation_dir=str(Path(args.label_dir) / args.scene),
        feature_height=getattr(config, "feature_height", 30),
        feature_width=getattr(config, "feature_width", 40),
        allow_empty_features=True,
    )
    text = load_or_generate_prompt_ensemble_embeddings(
        categories,
        device,
        cache_path=args.text_embedding_cache,
        prompt_templates=["{query}"],
    )
    negatives = load_or_generate_prompt_ensemble_embeddings(
        list(NEGATIVE_PROMPTS),
        device,
        cache_path=args.canonical_embedding_cache,
        prompt_templates=["{query}"],
    )
    summary_head = load_text_projection_head(
        text_encoder="siglip2",
        summary_head_weights=args.summary_head_weights,
        device=device,
    )
    gaussian_scores = compute_gaussian_text_scores(
        model,
        codec,
        summary_head,
        text,
        negatives,
        is_hybrid=is_hybrid,
        direct_readout_mode="gaussian",
        direct_readout_k=8,
        direct_readout_candidate_k=0,
        compact_feature_key="features",
        scoring="relevancy",
        softmax_temperature=10.0,
        chunk_size=args.chunk_size,
        device=device,
    ).to(device)

    variants = ("screen_enhanced", "core_decode_after_splat", "primitive_score_first")
    ious: dict[str, list[float]] = {name: [] for name in variants}
    frame_reports = []
    for frame_id, frame_objects in sorted(annotations.items()):
        pose = dataset.pose_by_frame_idx[int(frame_id)]
        viewmat = torch.from_numpy(pose.copy()).float().to(device).unsqueeze(0)
        enhanced = render_1280d(
            model,
            codec,
            renderer,
            sharpener,
            refiner,
            viewmat,
            is_hybrid=is_hybrid,
            config=config,
            device=device,
        )
        core = render_1280d(
            model,
            codec,
            renderer,
            nn.Identity().to(device),
            None,
            viewmat,
            is_hybrid=is_hybrid,
            config=config,
            device=device,
        )
        enhanced_map = compute_relevancy_heatmap(
            project_to_siglip2(enhanced.half(), summary_head),
            text,
            canonical_emb=negatives,
            temperature=0.1,
            scoring="relevancy",
        )
        core_map = compute_relevancy_heatmap(
            project_to_siglip2(core.half(), summary_head),
            text,
            canonical_emb=negatives,
            temperature=0.1,
            scoring="relevancy",
        )
        primitive_result = renderer.render_feature_rows(
            model,
            viewmat.squeeze(0),
            gaussian_scores,
            alpha_normalize=True,
        )
        primitive_map = primitive_result["feature_map"].clamp(0.0, 1.0)
        visible = primitive_result["alpha_map"] > 1e-3
        maps = {
            "screen_enhanced": enhanced_map,
            "core_decode_after_splat": core_map,
            "primitive_score_first": primitive_map,
        }
        gt_masks = build_gt_masks(
            frame_objects, categories, image_height, image_width
        )
        for name, heatmaps in maps.items():
            resized = F.interpolate(
                heatmaps.unsqueeze(0),
                size=(image_height, image_width),
                mode="bilinear",
                align_corners=False,
            )[0]
            for category_index, category in enumerate(categories):
                target = gt_masks.get(category)
                if target is None or not bool(np.asarray(target).any()):
                    continue
                prediction = resized[category_index].cpu().numpy() >= 0.5
                ious[name].append(_iou(prediction, np.asarray(target, dtype=bool)))
        frame_reports.append(
            {
                "frame_id": int(frame_id),
                "enhanced_vs_score_first": _pair_stats(
                    enhanced_map, primitive_map, visible
                ),
                "core_vs_score_first": _pair_stats(core_map, primitive_map, visible),
            }
        )
    return {
        "scene": args.scene,
        "protocol": {
            "threshold": 0.5,
            "primitive_rendering": "alpha_normalized_relevance_probability",
            "text_prompts": "raw query plus fixed four negatives",
            "test_calibration": "none",
        },
        "category_frame_miou": {
            name: float(np.mean(values)) if values else 0.0
            for name, values in ious.items()
        },
        "mean_commutation": {
            comparison: {
                key: float(np.mean([frame[comparison][key] for frame in frame_reports]))
                for key in frame_reports[0][comparison]
            }
            for comparison in ("enhanced_vs_score_first", "core_vs_score_first")
        },
        "frames": frame_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="ramen")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label_dir", default="/mnt/pool/sqy/3d_understanding/lerf_ovs/label")
    parser.add_argument("--text_embedding_cache", required=True)
    parser.add_argument("--canonical_embedding_cache", required=True)
    parser.add_argument("--summary_head_weights", default="checkpoints/siglip2_summary_head.pth")
    parser.add_argument("--chunk_size", type=int, default=8192)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("category_frame_miou", "mean_commutation")}, indent=2))


if __name__ == "__main__":
    main()
