#!/usr/bin/env python3
"""Build a depth-checked multiview RADIO teacher cache on Gaussian rows.

The cache is training-only: it opens extracted RADIO maps and camera poses,
but never benchmark masks or text queries.  Each Gaussian receives the mean
of the views in which its center is inside the camera, depth-consistent with
the rendered field, and supported by non-trivial rendered alpha.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from radio_gs.config import load_config
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.training.feature_training_utils import (
    SimpleRadioDataset,
    sample_multiview_radio_targets,
)
from radio_gs.training.primitive_consensus import robust_multiview_consensus


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _select_indices(length: int, max_views: int) -> list[int]:
    if max_views <= 0 or max_views >= length:
        return list(range(length))
    positions = np.linspace(0, length - 1, num=max_views)
    return sorted({int(round(value)) for value in positions})


def _parse_frame_ids(raw: str) -> set[int]:
    value = str(raw or "").strip()
    if not value:
        return set()
    path = Path(value)
    if path.is_dir():
        return {
            int(item.stem.split("_")[-1])
            for item in path.glob("frame_*.json")
        }
    if path.is_file():
        value = path.read_text(encoding="utf-8")
    tokens: list[str] = []
    for line in value.splitlines():
        tokens.extend(line.split("#", 1)[0].replace(",", " ").split())
    return {int(item) for item in tokens}


def _select_candidate_indices(candidates: list[int], max_views: int) -> list[int]:
    chosen = _select_indices(len(candidates), max_views)
    return [candidates[index] for index in chosen]


def build_cache(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    config = load_config(args.config)
    model, _codec, renderer, _sharpener, _refiner, _cfg, _is_hybrid = (
        load_render_pipeline(
            args.config,
            args.checkpoint,
            device,
            strict_checkpoint_contract=True,
            load_ply_rgb_features=False,
        )
    )
    feature_height = int(getattr(config, "feature_height", renderer.image_height))
    feature_width = int(getattr(config, "feature_width", renderer.image_width))
    feature_dir = Path(str(getattr(config, "feature_dir", "")))
    raw_pose_file = str(getattr(config, "pose_file", "") or "").strip()
    configured_pose_file = Path(raw_pose_file) if raw_pose_file else None
    pose_file = (
        str(configured_pose_file)
        if configured_pose_file is not None and configured_pose_file.is_file()
        else None
    )
    raw_pose_dir = str(getattr(config, "pose_dir", "") or "").strip()
    configured_pose_dir = Path(raw_pose_dir) if raw_pose_dir else None
    fallback_pose_dir = feature_dir / "poses_w2c"
    pose_dir = (
        str(configured_pose_dir)
        if configured_pose_dir is not None and configured_pose_dir.is_dir()
        else str(fallback_pose_dir) if fallback_pose_dir.is_dir() else None
    )
    dataset = SimpleRadioDataset(
        feature_dir=str(feature_dir),
        pose_file=pose_file,
        pose_dir=pose_dir,
        feature_size=(feature_height, feature_width),
        split="train",
        dataset_type=str(getattr(config, "dataset_type", "lerf")),
    )
    excluded_frame_ids = _parse_frame_ids(args.exclude_frame_ids)
    candidates = [
        index
        for index, frame_id in enumerate(dataset.frame_indices)
        if int(frame_id) not in excluded_frame_ids
    ]
    if not candidates:
        raise RuntimeError("all available feature frames were excluded")
    selected = _select_candidate_indices(candidates, int(args.max_views))
    teacher_maps = torch.stack(
        [dataset[index]["radio_features"].float().cpu() for index in selected], dim=0
    )
    poses = torch.stack(
        [dataset[index]["pose_w2c"].float().cpu() for index in selected], dim=0
    )
    feature_space = str(args.feature_space).lower()
    summary_head_path = ""
    if feature_space == "siglip_summary":
        summary_head_path = str(Path(args.summary_head_weights).expanduser().resolve())
        summary_head = SigLIP2SummaryHead.from_extracted_weights(summary_head_path).to(device)
        summary_head.eval()
        projected_parts: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in tqdm(
                range(0, teacher_maps.shape[0], int(args.projection_batch_size)),
                desc="project teacher query space",
            ):
                maps = teacher_maps[
                    start : start + int(args.projection_batch_size)
                ].to(device)
                batch, channels, height, width = maps.shape
                tokens = maps.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
                projected = F.normalize(summary_head(tokens.float()), dim=-1)
                projected_parts.append(
                    projected.reshape(batch, height, width, -1)
                    .permute(0, 3, 1, 2)
                    .half()
                    .cpu()
                )
        teacher_maps = torch.cat(projected_parts, dim=0)
        del summary_head
        if device.type == "cuda":
            torch.cuda.empty_cache()
    elif feature_space not in {"radio", "semantic_descriptor"}:
        raise ValueError(f"Unsupported feature space: {feature_space}")

    depth_parts: list[torch.Tensor] = []
    alpha_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(selected), int(args.render_batch_size)),
            desc="render visibility",
        ):
            stop = min(start + int(args.render_batch_size), len(selected))
            result = renderer.render_features_batch(
                model,
                poses[start:stop].to(device),
                feature_height=feature_height,
                feature_width=feature_width,
            )
            depth_parts.append(result["depth_map"].float().cpu())
            alpha_parts.append(result["alpha_map"].float().cpu())
    depth_maps = torch.cat(depth_parts, dim=0)
    alpha_maps = torch.cat(alpha_parts, dim=0)

    xyz_cpu = model.get_xyz().detach().float().cpu().contiguous()
    feature_parts: list[torch.Tensor] = []
    valid_parts: list[torch.Tensor] = []
    count_parts: list[torch.Tensor] = []
    reliability_parts: list[torch.Tensor] = []
    view_chunk = max(1, int(args.view_chunk_size))
    point_chunk = max(1, int(args.point_chunk_size))
    if args.aggregation_mode == "center":
        with torch.inference_mode():
            for point_start in tqdm(
                range(0, xyz_cpu.shape[0], point_chunk),
                desc="aggregate Gaussian teachers",
            ):
                point_stop = min(point_start + point_chunk, xyz_cpu.shape[0])
                points = xyz_cpu[point_start:point_stop].to(device)
                target_sum = torch.zeros(
                    points.shape[0], teacher_maps.shape[1], device=device, dtype=torch.float32
                )
                view_counts = torch.zeros(points.shape[0], device=device, dtype=torch.long)
                for view_start in range(0, len(selected), view_chunk):
                    view_stop = min(view_start + view_chunk, len(selected))
                    targets, _valid, counts = sample_multiview_radio_targets(
                        points,
                        teacher_maps[view_start:view_stop].to(device),
                        poses[view_start:view_stop].to(device),
                        renderer.scaled_intrinsics(feature_width, feature_height).float(),
                        depth_map=depth_maps[view_start:view_stop].to(device),
                        alpha_map=alpha_maps[view_start:view_stop].to(device),
                        depth_tolerance=float(args.depth_tolerance),
                        relative_depth_tolerance=float(args.relative_depth_tolerance),
                        alpha_threshold=float(args.alpha_threshold),
                        normalize_sampled_features=bool(args.normalize_each_view),
                    )
                    counts_f = counts.float().unsqueeze(1)
                    target_sum.add_(targets.float() * counts_f)
                    view_counts.add_(counts.long())
                valid = view_counts > 0
                averaged = target_sum / view_counts.clamp_min(1).float().unsqueeze(1)
                averaged[~valid] = 0.0
                if args.robust_mpr:
                    observations: list[torch.Tensor] = []
                    observation_validity: list[torch.Tensor] = []
                    for view_index in range(len(selected)):
                        view_target, view_valid, _view_count = sample_multiview_radio_targets(
                            points,
                            teacher_maps[view_index : view_index + 1].to(device),
                            poses[view_index : view_index + 1].to(device),
                            renderer.scaled_intrinsics(feature_width, feature_height).float(),
                            depth_map=depth_maps[view_index : view_index + 1].to(device),
                            alpha_map=alpha_maps[view_index : view_index + 1].to(device),
                            depth_tolerance=float(args.depth_tolerance),
                            relative_depth_tolerance=float(args.relative_depth_tolerance),
                            alpha_threshold=float(args.alpha_threshold),
                            normalize_sampled_features=bool(args.normalize_each_view),
                        )
                        observations.append(view_target)
                        observation_validity.append(view_valid)
                    consensus = robust_multiview_consensus(
                        torch.stack(observations),
                        torch.stack(observation_validity),
                        robust_temperature=float(args.robust_temperature),
                        iterations=int(args.robust_iterations),
                        normalize_observations=bool(args.normalize_each_view),
                    )
                    averaged = consensus.targets
                    valid = consensus.valid
                    view_counts = consensus.observation_count
                    reliability = consensus.reliability
                else:
                    reliability = torch.stack(
                        [
                            view_counts.float() / max(1, len(selected)),
                            valid.float(),
                            valid.float(),
                        ],
                        dim=-1,
                    )
                feature_parts.append(averaged.half().cpu())
                valid_parts.append(valid.cpu())
                count_parts.append(view_counts.cpu())
                reliability_parts.append(reliability.half().cpu())
        features = torch.cat(feature_parts, dim=0)
        valid = torch.cat(valid_parts, dim=0)
        view_counts = torch.cat(count_parts, dim=0)
        reliability = torch.cat(reliability_parts, dim=0)
    else:
        from radio_gs.scripts.eval_lerf_direct_3d_selection import (
            raster_adjoint_registered_view_features,
            rasterize_registered_view_features,
        )

        registered_sum = torch.zeros(
            xyz_cpu.shape[0], teacher_maps.shape[1], dtype=torch.float32
        )
        registered_counts = torch.zeros(xyz_cpu.shape[0], dtype=torch.float32)
        observation_counts = torch.zeros(xyz_cpu.shape[0], dtype=torch.long)
        topk_observations = None
        if args.raster_view_fusion == "topk_mean":
            topk_observations = torch.full(
                (
                    xyz_cpu.shape[0],
                    teacher_maps.shape[1],
                    max(1, int(args.raster_topk)),
                ),
                -float("inf"),
                dtype=torch.float32,
            )
        aggregation_context = (
            contextlib.nullcontext()
            if args.aggregation_mode == "raster_adjoint"
            else torch.inference_mode()
        )
        with aggregation_context:
            for view_index in tqdm(
                range(len(selected)), desc="aggregate raster contributions"
            ):
                if args.aggregation_mode == "raster_adjoint":
                    frame_sum, frame_counts = raster_adjoint_registered_view_features(
                        model=model,
                        renderer=renderer,
                        viewmat=poses[view_index].to(device),
                        siglip_feat=teacher_maps[view_index : view_index + 1].to(
                            device=device, dtype=torch.float32
                        ),
                        alpha_map=alpha_maps[view_index : view_index + 1].to(device),
                        alpha_threshold=float(args.alpha_threshold),
                        channel_chunk_size=int(args.adjoint_channel_chunk_size),
                    )
                else:
                    frame_sum, frame_counts = rasterize_registered_view_features(
                        model=model,
                        renderer=renderer,
                        viewmat=poses[view_index].to(device),
                        siglip_feat=teacher_maps[view_index : view_index + 1].to(
                            device=device, dtype=torch.float32
                        ),
                        depth_map=depth_maps[view_index : view_index + 1].to(device),
                        alpha_map=alpha_maps[view_index : view_index + 1].to(device),
                        registration_depth_tolerance=float(args.depth_tolerance),
                        registration_relative_depth_tolerance=float(
                            args.relative_depth_tolerance
                        ),
                        registration_alpha_threshold=float(args.alpha_threshold),
                        registration_weight_mode=args.registration_weight_mode,
                        gaussian_top1=True,
                    )
                counts_cpu = frame_counts.float().cpu()
                frame_valid = counts_cpu > 0
                if bool(frame_valid.any()):
                    frame_sum_cpu = frame_sum.float().cpu()[frame_valid]
                    if args.raster_view_fusion == "contribution_mean":
                        registered_sum[frame_valid] += frame_sum_cpu
                        registered_counts[frame_valid] += counts_cpu[frame_valid]
                    else:
                        frame_observation = frame_sum_cpu / counts_cpu[
                            frame_valid, None
                        ].clamp_min(1e-8)
                        if args.raster_view_fusion == "view_mean":
                            registered_sum[frame_valid] += frame_observation
                            registered_counts[frame_valid] += 1.0
                        elif args.raster_view_fusion == "topk_mean":
                            assert topk_observations is not None
                            current = topk_observations[frame_valid]
                            candidates = torch.cat(
                                [current, frame_observation.unsqueeze(-1)], dim=-1
                            )
                            topk_observations[frame_valid] = torch.topk(
                                candidates,
                                k=topk_observations.shape[-1],
                                dim=-1,
                            ).values
                        else:
                            raise ValueError(
                                f"unsupported raster view fusion: {args.raster_view_fusion}"
                            )
                    observation_counts[frame_valid] += 1
                del frame_sum, frame_counts
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if args.raster_view_fusion == "topk_mean":
            assert topk_observations is not None
            finite = torch.isfinite(topk_observations)
            valid = finite.any(dim=2).any(dim=1)
            safe = torch.where(finite, topk_observations, torch.zeros_like(topk_observations))
            features = (
                safe.sum(dim=-1) / finite.sum(dim=-1).clamp_min(1)
            ).half()
            features[~valid] = 0.0
            del topk_observations, safe, finite
        else:
            valid = registered_counts > 0
            features = torch.zeros_like(registered_sum, dtype=torch.float16)
            features[valid] = (
                registered_sum[valid]
                / registered_counts[valid].clamp_min(1e-8).unsqueeze(1)
            ).half()
        view_counts = observation_counts
        reliability = torch.stack(
            [
                view_counts.float() / max(1, len(selected)),
                valid.float(),
                valid.float(),
            ],
            dim=-1,
        ).half()
    metadata = {
        "schema_version": 1,
        "feature_space": feature_space,
        "construction": (
            f"{feature_space}_{args.aggregation_mode}_robust_mpr"
            if args.robust_mpr and args.aggregation_mode == "center"
            else (
                f"{feature_space}_{args.aggregation_mode}_{args.raster_view_fusion}"
                if args.aggregation_mode != "center"
                else f"{feature_space}_{args.aggregation_mode}_multiview_mean"
            )
        ),
        "aggregation_mode": args.aggregation_mode,
        "registration_weight_mode": args.registration_weight_mode,
        "raster_view_fusion": args.raster_view_fusion,
        "raster_topk": max(1, int(args.raster_topk)),
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_declared_views": len(selected),
        "selected_dataset_indices": selected,
        "selected_frame_indices": [int(dataset.frame_indices[index]) for index in selected],
        "excluded_frame_ids": sorted(excluded_frame_ids),
        "depth_tolerance": float(args.depth_tolerance),
        "relative_depth_tolerance": float(args.relative_depth_tolerance),
        "alpha_threshold": float(args.alpha_threshold),
        "normalize_each_view": bool(args.normalize_each_view),
        "robust_mpr": bool(args.robust_mpr and args.aggregation_mode == "center"),
        "robust_temperature": float(args.robust_temperature),
        "robust_iterations": int(args.robust_iterations),
        "summary_head_weights": summary_head_path,
        "query_names": [
            value.strip() for value in str(args.query_names).split(",") if value.strip()
        ],
        "xyz_sha256": _sha256_tensor_rows(xyz_cpu),
        "benchmark_masks_opened": False,
        "benchmark_images_opened": bool(args.benchmark_images_opened),
        "text_queries_opened": bool(args.text_queries_opened),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "xyz": xyz_cpu,
            "geometry_fingerprint": {
                "num_gaussians": int(xyz_cpu.shape[0]),
                "xyz_sha256": _sha256_tensor_rows(xyz_cpu),
            },
            "features": features,
            "valid": valid,
            "view_counts": view_counts,
            "reliability": reliability,
            "metadata": metadata,
        },
        output,
    )
    positive = view_counts[valid]
    report = {
        "output": str(output),
        "num_gaussians": int(xyz_cpu.shape[0]),
        "num_views": len(selected),
        "valid_count": int(valid.sum()),
        "valid_ratio": float(valid.float().mean()),
        "mean_views_if_valid": float(positive.float().mean()) if positive.numel() else 0.0,
        "median_views_if_valid": float(positive.float().median()) if positive.numel() else 0.0,
        "max_views": int(positive.max()) if positive.numel() else 0,
        "metadata": metadata,
    }
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-views", type=int, default=32)
    parser.add_argument(
        "--exclude-frame-ids",
        default="",
        help="Comma list, text file, or LERF label directory of held-out frame IDs.",
    )
    parser.add_argument("--render-batch-size", type=int, default=4)
    parser.add_argument("--view-chunk-size", type=int, default=8)
    parser.add_argument("--point-chunk-size", type=int, default=4096)
    parser.add_argument("--depth-tolerance", type=float, default=0.08)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.02)
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--normalize-each-view", action="store_true")
    parser.add_argument(
        "--robust-mpr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Robustly fuse per-view primitive observations (center aggregation).",
    )
    parser.add_argument("--robust-temperature", type=float, default=0.10)
    parser.add_argument("--robust-iterations", type=int, default=2)
    parser.add_argument(
        "--feature-space",
        choices=["radio", "siglip_summary", "semantic_descriptor"],
        default="radio",
        help=(
            "Aggregate raw RADIO, precomputed semantic descriptors, or first "
            "project every view through a frozen SigLIP2 pointwise head."
        ),
    )
    parser.add_argument(
        "--summary-head-weights",
        default="checkpoints/siglip2_summary_head.pth",
    )
    parser.add_argument("--projection-batch-size", type=int, default=2)
    parser.add_argument(
        "--aggregation-mode",
        choices=["center", "raster_gaussian_top1", "raster_adjoint"],
        default="center",
    )
    parser.add_argument(
        "--registration-weight-mode",
        choices=["uniform", "alpha", "alpha_depth"],
        default="alpha_depth",
    )
    parser.add_argument("--adjoint-channel-chunk-size", type=int, default=32)
    parser.add_argument(
        "--raster-view-fusion",
        choices=["contribution_mean", "view_mean", "topk_mean"],
        default="contribution_mean",
        help="Across-view fusion after each raster registration observation.",
    )
    parser.add_argument(
        "--raster-topk",
        type=int,
        default=3,
        help="Number of strongest view observations retained by topk_mean.",
    )
    parser.add_argument("--query-names", default="")
    parser.add_argument("--text-queries-opened", action="store_true")
    parser.add_argument(
        "--benchmark-images-opened",
        action="store_true",
        help="Mark diagnostic caches built from held-out benchmark RGB/features.",
    )
    args = parser.parse_args()
    print(json.dumps(build_cache(args), indent=2))


if __name__ == "__main__":
    main()
