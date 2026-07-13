#!/usr/bin/env python3
"""Audit one-view 2D -> primitive -> same-view registration fidelity."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    _parse_frame_ids,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    raster_adjoint_registered_view_features,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.training.feature_training_utils import SimpleRadioDataset


def audit(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    config = load_config(args.config)
    model, _codec, renderer, _sharpener, _refiner, _cfg, _hybrid = load_render_pipeline(
        args.config,
        args.checkpoint,
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
    )
    height = int(getattr(config, "feature_height", renderer.image_height))
    width = int(getattr(config, "feature_width", renderer.image_width))
    dataset = SimpleRadioDataset(
        feature_dir=str(getattr(config, "feature_dir")),
        pose_file=str(getattr(config, "pose_file", "") or "") or None,
        pose_dir=str(getattr(config, "pose_dir", "") or "") or None,
        feature_size=(height, width),
        split="train",
        dataset_type=str(getattr(config, "dataset_type", "lerf")),
    )
    requested = _parse_frame_ids(args.frame_ids)
    rows = [
        index
        for index, frame_id in enumerate(dataset.frame_indices)
        if not requested or int(frame_id) in requested
    ]
    if not rows:
        raise ValueError("no requested frames exist in the configured cache")
    reports = []
    for index in rows:
        sample = dataset[index]
        target = sample["radio_features"].float().unsqueeze(0).to(device)
        pose = sample["pose_w2c"].float().to(device)
        with torch.inference_mode():
            visibility = renderer.render_features(
                model, pose, feature_height=height, feature_width=width
            )
        sums, counts = raster_adjoint_registered_view_features(
            model=model,
            renderer=renderer,
            viewmat=pose,
            siglip_feat=target,
            alpha_map=visibility["alpha_map"].unsqueeze(0),
            alpha_threshold=float(args.alpha_threshold),
            channel_chunk_size=int(args.channel_chunk_size),
        )
        valid_rows = counts > 0
        primitive = torch.zeros_like(sums)
        primitive[valid_rows] = sums[valid_rows] / counts[valid_rows, None].clamp_min(1e-8)
        with torch.inference_mode():
            reconstructed = renderer.render_feature_rows(
                model,
                pose,
                primitive,
                feature_height=height,
                feature_width=width,
                alpha_normalize=True,
                row_confidence=valid_rows.float(),
            )["feature_map"].float()
        source = target[0]
        visible = visibility["alpha_map"] >= float(args.alpha_threshold)
        source_flat = source[:, visible]
        recon_flat = reconstructed[:, visible]
        source_centered = source_flat - source_flat.mean(dim=1, keepdim=True)
        recon_centered = recon_flat - recon_flat.mean(dim=1, keepdim=True)
        channel_corr = F.cosine_similarity(source_centered, recon_centered, dim=1)
        query_ious = []
        for query in range(source.shape[0]):
            source_mask = source[query] > float(args.mask_peak_ratio) * source[query].max()
            recon_mask = (
                reconstructed[query]
                > float(args.mask_peak_ratio) * reconstructed[query].max()
            )
            union = (source_mask | recon_mask).sum()
            query_ious.append(
                float((source_mask & recon_mask).sum().float() / union.clamp_min(1))
            )
        reports.append(
            {
                "frame_id": int(dataset.frame_indices[index]),
                "registered_gaussians": int(valid_rows.sum()),
                "pixel_cosine": float(
                    F.cosine_similarity(source_flat.T, recon_flat.T, dim=1).mean()
                ),
                "mean_channel_pearson": float(channel_corr.mean()),
                "mean_peak_relative_mask_iou": float(sum(query_ious) / len(query_ious)),
                "per_query_peak_relative_mask_iou": query_ious,
            }
        )
    return {
        "schema_version": 1,
        "operation": "same_view_raster_adjoint_roundtrip",
        "mask_peak_ratio": float(args.mask_peak_ratio),
        "benchmark_masks_opened": False,
        "frames": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--mask-peak-ratio", type=float, default=0.6)
    parser.add_argument("--channel-chunk-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
