#!/usr/bin/env python3
"""Measure whether MPR cross-view conflict predicts held-out render error."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.evaluation.render_ceiling import normalize_premultiplied
from radio_gs.evaluation.view_consistency import (
    consistency_from_sums,
    merge_training_partials,
    pearson_spearman,
)
from radio_gs.field import (
    load_canonical_field_checkpoint,
    load_view_residual_checkpoint,
)
from radio_gs.rendering.coefficient_renderer import (
    render_canonical_radio,
    render_view_conditioned_radio,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    rasterize_registered_view_features,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.training.feature_training_utils import SimpleRadioDataset


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_frame_ids(raw: str) -> list[int]:
    value = str(raw or "").strip()
    if not value:
        return []
    path = Path(value)
    if path.is_file():
        value = path.read_text(encoding="utf-8")
    tokens: list[str] = []
    for line in value.splitlines():
        tokens.extend(line.split("#", 1)[0].replace(",", " ").split())
    return sorted({int(token) for token in tokens})


def _dataset(config, renderer) -> SimpleRadioDataset:
    feature_dir = Path(str(getattr(config, "feature_dir", "")))
    raw_pose_file = str(getattr(config, "pose_file", "") or "").strip()
    pose_file = raw_pose_file if raw_pose_file and Path(raw_pose_file).is_file() else None
    raw_pose_dir = str(getattr(config, "pose_dir", "") or "").strip()
    fallback = feature_dir / "poses_w2c"
    pose_dir = (
        raw_pose_dir
        if raw_pose_dir and Path(raw_pose_dir).is_dir()
        else str(fallback) if fallback.is_dir() else None
    )
    return SimpleRadioDataset(
        feature_dir=str(feature_dir),
        pose_file=pose_file,
        pose_dir=pose_dir,
        feature_size=(
            int(getattr(config, "feature_height", renderer.image_height)),
            int(getattr(config, "feature_width", renderer.image_width)),
        ),
        split="validation",
        dataset_type=str(getattr(config, "dataset_type", "lerf")),
    )


def _load_runtime(args: argparse.Namespace):
    device = torch.device(args.device)
    config = load_config(args.config)
    model, _codec, renderer, _sharpener, _refiner, _cfg, _hybrid = load_render_pipeline(
        args.config,
        args.geometry_checkpoint,
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
    )
    geometry_hash = _sha256_tensor_rows(model.get_xyz())
    mpr = torch.load(args.mpr_cache, map_location="cpu")
    fingerprint = dict(mpr.get("geometry_fingerprint", {}))
    if (
        str(fingerprint.get("xyz_sha256", "")) != geometry_hash
        or int(fingerprint.get("num_gaussians", -1)) != int(model.get_xyz().shape[0])
    ):
        raise ValueError("MPR/geometry row fingerprint mismatch")
    metadata = dict(mpr.get("metadata", {}))
    if bool(metadata.get("benchmark_masks_opened", False)) or bool(
        metadata.get("text_queries_opened", False)
    ):
        raise ValueError("MPR provenance is not query/label free")
    return device, config, model, renderer, _dataset(config, renderer), mpr, metadata


def _geometry_maps(renderer, model, pose, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones(
        int(model.get_xyz().shape[0]), 1, device=model.get_xyz().device, dtype=torch.float32
    )
    result = renderer.render_feature_rows(
        model,
        pose,
        ones,
        feature_height=height,
        feature_width=width,
        alpha_normalize=False,
    )
    return result["depth_map"].float(), result["alpha_map"].float()


def _precompute_training_geometry_maps(
    renderer,
    model,
    dataset: SimpleRadioDataset,
    frame_to_index: Mapping[int, int],
    frames: list[int],
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Replay the builder's legacy-feature batch render for depth and alpha."""

    if batch_size <= 0:
        raise ValueError("geometry render batch size must be positive")
    poses = torch.stack(
        [dataset[frame_to_index[frame]]["pose_w2c"].float() for frame in frames]
    )
    sample = dataset[frame_to_index[frames[0]]]["radio_features"]
    height, width = int(sample.shape[1]), int(sample.shape[2])
    depth_parts: list[torch.Tensor] = []
    alpha_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            stop = min(start + batch_size, len(frames))
            result = renderer.render_features_batch(
                model,
                poses[start:stop].to(device),
                feature_height=height,
                feature_width=width,
            )
            depth_parts.append(result["depth_map"].float().cpu())
            alpha_parts.append(result["alpha_map"].float().cpu())
    depths = torch.cat(depth_parts, dim=0)
    alphas = torch.cat(alpha_parts, dim=0)
    return (
        {frame: depths[index] for index, frame in enumerate(frames)},
        {frame: alphas[index] for index, frame in enumerate(frames)},
    )


def _registration_kwargs(metadata: Mapping) -> dict:
    if str(metadata.get("aggregation_mode", "")) != "raster_gaussian_top1":
        raise ValueError("this audit currently requires raster_gaussian_top1 MPR")
    return {
        "registration_depth_tolerance": float(metadata.get("depth_tolerance", 0.08)),
        "registration_relative_depth_tolerance": float(
            metadata.get("relative_depth_tolerance", 0.02)
        ),
        "registration_alpha_threshold": float(metadata.get("alpha_threshold", 0.02)),
        "registration_weight_mode": str(metadata.get("registration_weight_mode", "alpha_depth")),
        "gaussian_top1": True,
    }


def compute_training_shard(args: argparse.Namespace) -> dict:
    device, _config, model, renderer, dataset, mpr, metadata = _load_runtime(args)
    frame_to_index = {int(frame): index for index, frame in enumerate(dataset.frame_indices)}
    all_frames = [int(value) for value in metadata.get("selected_frame_indices", [])]
    if len(all_frames) != len(set(all_frames)) or not all_frames:
        raise ValueError("MPR selected-frame metadata is empty or duplicated")
    missing = sorted(set(all_frames) - set(frame_to_index))
    if missing:
        raise ValueError(f"MPR frames unavailable in feature dataset: {missing}")
    num_shards = int(args.num_shards)
    shard_index = int(args.shard_index)
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard index/count")
    frames = all_frames[shard_index::num_shards]
    depth_by_frame, alpha_by_frame = _precompute_training_geometry_maps(
        renderer,
        model,
        dataset,
        frame_to_index,
        all_frames,
        device,
        batch_size=int(args.geometry_render_batch_size),
    )

    targets = torch.as_tensor(mpr["features"]).to(device=device, dtype=torch.float32)
    target_valid = torch.as_tensor(mpr["valid"]).to(device=device).bool()
    target_norm = F.normalize(targets, dim=-1, eps=1e-8)
    num_rows = int(targets.shape[0])
    observation_count = torch.zeros(num_rows, dtype=torch.long, device=device)
    float_sums = {
        key: torch.zeros(num_rows, dtype=torch.float64, device=device)
        for key in (
            "weight_sum",
            "weight_square_sum",
            "cosine_sum",
            "cosine_square_sum",
            "weighted_cosine_sum",
            "weighted_cosine_square_sum",
        )
    }
    per_view: list[dict] = []
    registration_kwargs = _registration_kwargs(metadata)

    with torch.inference_mode():
        for frame in frames:
            sample = dataset[frame_to_index[frame]]
            teacher = sample["radio_features"].to(device=device, dtype=torch.float32)
            pose = sample["pose_w2c"].to(device)
            height, width = teacher.shape[1:]
            depth = depth_by_frame[frame].to(device)
            alpha = alpha_by_frame[frame].to(device)
            frame_sum, frame_weight = rasterize_registered_view_features(
                model=model,
                renderer=renderer,
                viewmat=pose,
                siglip_feat=teacher.unsqueeze(0),
                depth_map=depth.unsqueeze(0),
                alpha_map=alpha.unsqueeze(0),
                **registration_kwargs,
            )
            valid = (frame_weight > 0) & target_valid
            if bool(valid.any()):
                observation = frame_sum[valid].float() / frame_weight[valid, None].clamp_min(1e-8)
                cosine = (F.normalize(observation, dim=-1, eps=1e-8) * target_norm[valid]).sum(dim=-1)
                weight = frame_weight[valid].double()
                cosine64 = cosine.double()
                observation_count[valid] += 1
                float_sums["weight_sum"][valid] += weight
                float_sums["weight_square_sum"][valid] += weight.square()
                float_sums["cosine_sum"][valid] += cosine64
                float_sums["cosine_square_sum"][valid] += cosine64.square()
                float_sums["weighted_cosine_sum"][valid] += weight * cosine64
                float_sums["weighted_cosine_square_sum"][valid] += weight * cosine64.square()
                per_view.append(
                    {
                        "frame_id": frame,
                        "observed_rows": int(valid.sum()),
                        "mean_cosine": float(cosine.mean()),
                        "weighted_mean_cosine": float((weight * cosine64).sum() / weight.sum()),
                    }
                )
            else:
                per_view.append(
                    {"frame_id": frame, "observed_rows": 0, "mean_cosine": None, "weighted_mean_cosine": None}
                )
            print(
                f"[training shard {shard_index}/{num_shards}] frame {frame}: "
                f"rows={per_view[-1]['observed_rows']} cosine={per_view[-1]['mean_cosine']}",
                flush=True,
            )
            del frame_sum, frame_weight, teacher, depth, alpha
            if device.type == "cuda":
                torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "kind": "mpr_training_view_consistency_partial",
        "geometry_xyz_sha256": str(metadata.get("xyz_sha256", "")),
        "mpr_cache": str(Path(args.mpr_cache).resolve()),
        "num_declared_views": len(all_frames),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "selected_frame_ids": frames,
        "observation_count": observation_count.cpu(),
        **{key: value.float().cpu() for key, value in float_sums.items()},
        "per_view": per_view,
        "protocol": {
            "aggregation_mode": metadata.get("aggregation_mode"),
            "registration_weight_mode": metadata.get("registration_weight_mode"),
            "gaussian_top1": True,
            "geometry_visibility_render": "builder_legacy_feature_batch",
            "geometry_render_batch_size": int(args.geometry_render_batch_size),
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "output": str(output.resolve()),
                "shard_index": shard_index,
                "num_shards": num_shards,
                "selected_frame_ids": frames,
                "mean_view_cosine": sum(
                    item["mean_cosine"] * item["observed_rows"]
                    for item in per_view
                    if item["mean_cosine"] is not None
                )
                / max(1, sum(item["observed_rows"] for item in per_view)),
                "per_view": per_view,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return payload


def compute_heldout(args: argparse.Namespace) -> dict:
    device, _config, model, renderer, dataset, mpr, metadata = _load_runtime(args)
    frame_to_index = {int(frame): index for index, frame in enumerate(dataset.frame_indices)}
    frames = _parse_frame_ids(args.frame_ids)
    if not frames:
        frames = [int(value) for value in metadata.get("excluded_frame_ids", [])]
    overlap = sorted(set(frames).intersection(metadata.get("selected_frame_indices", [])))
    if overlap:
        raise ValueError(f"held-out frames overlap MPR views: {overlap}")
    missing = sorted(set(frames) - set(frame_to_index))
    if missing:
        raise ValueError(f"held-out frames unavailable: {missing}")

    row_valid = torch.as_tensor(mpr["valid"]).to(device=device).bool()
    num_rows = int(row_valid.numel())
    field = None
    residual = None
    representation = "mpr_full1280_oracle"
    if str(args.field_checkpoint).strip():
        field, field_payload = load_canonical_field_checkpoint(
            args.field_checkpoint, map_location="cpu"
        )
        if str(field_payload.get("geometry_fingerprint", {}).get("xyz_sha256", "")) != str(
            metadata.get("xyz_sha256", "")
        ):
            raise ValueError("held-out field/geometry fingerprint mismatch")
        field = field.to(device).eval()
        representation = "canonical_field"
        if str(args.view_residual_checkpoint).strip():
            residual, residual_payload = load_view_residual_checkpoint(
                args.view_residual_checkpoint, map_location="cpu"
            )
            if str(
                residual_payload.get("geometry_fingerprint", {}).get("xyz_sha256", "")
            ) != str(metadata.get("xyz_sha256", "")):
                raise ValueError("held-out residual/geometry fingerprint mismatch")
            if str(residual_payload.get("base_field_sha256", "")) != _sha256_file(
                args.field_checkpoint
            ):
                raise ValueError("held-out residual was trained over a different field")
            residual = residual.to(device).eval()
            representation = "canonical_field_plus_zero_mean_view_residual"
    elif str(args.view_residual_checkpoint).strip():
        raise ValueError("--view-residual-checkpoint requires --field-checkpoint")
    if field is None:
        row_features = torch.as_tensor(mpr["features"]).to(
            device=device, dtype=torch.float32
        )
        row_features.mul_(row_valid[:, None])
    else:
        row_features = None
    error_sum = torch.zeros(num_rows, dtype=torch.float64, device=device)
    error_weight = torch.zeros(num_rows, dtype=torch.float64, device=device)
    observation_count = torch.zeros(num_rows, dtype=torch.long, device=device)
    per_view: list[dict] = []
    registration_kwargs = _registration_kwargs(metadata)

    with torch.inference_mode():
        for frame in frames:
            sample = dataset[frame_to_index[frame]]
            teacher = sample["radio_features"].to(device=device, dtype=torch.float32)
            pose = sample["pose_w2c"].to(device)
            height, width = teacher.shape[1:]
            if field is None:
                rendered = renderer.render_feature_rows(
                    model,
                    pose,
                    row_features,
                    feature_height=height,
                    feature_width=width,
                    alpha_normalize=False,
                )
                prediction = normalize_premultiplied(
                    rendered["feature_map"].float(), rendered["alpha_map"].float()
                )
            elif residual is None:
                rendered = render_canonical_radio(
                    renderer,
                    model,
                    field,
                    pose,
                    feature_height=height,
                    feature_width=width,
                )
                prediction = rendered["feature_map"].float()
            else:
                rendered = render_view_conditioned_radio(
                    renderer,
                    model,
                    field,
                    residual,
                    pose,
                    feature_height=height,
                    feature_width=width,
                )
                prediction = rendered["feature_map"].float()
            visible = rendered["alpha_map"] >= float(metadata.get("alpha_threshold", 0.02))
            pixel_cosine = F.cosine_similarity(prediction, teacher, dim=0, eps=1e-8)
            pixel_error = (1.0 - pixel_cosine).clamp_min(0.0)
            frame_sum, frame_weight = rasterize_registered_view_features(
                model=model,
                renderer=renderer,
                viewmat=pose,
                siglip_feat=pixel_error[None, None],
                depth_map=rendered["depth_map"][None],
                alpha_map=rendered["alpha_map"][None],
                **registration_kwargs,
            )
            valid = (frame_weight > 0) & row_valid
            error_sum[valid] += frame_sum[valid, 0].double()
            error_weight[valid] += frame_weight[valid].double()
            observation_count[valid] += 1
            per_view.append(
                {
                    "frame_id": frame,
                    "visible_pixels": int(visible.sum()),
                    "mean_pixel_cosine": float(pixel_cosine[visible].mean()),
                    "registered_rows": int(valid.sum()),
                }
            )
            print(
                f"[heldout] frame {frame}: cosine={per_view[-1]['mean_pixel_cosine']:.4f} "
                f"rows={per_view[-1]['registered_rows']}",
                flush=True,
            )
            del frame_sum, frame_weight, teacher, rendered, prediction
            if device.type == "cuda":
                torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "kind": "mpr_heldout_primitive_error",
        "representation": representation,
        "geometry_xyz_sha256": str(metadata.get("xyz_sha256", "")),
        "mpr_cache": str(Path(args.mpr_cache).resolve()),
        "selected_frame_ids": frames,
        "error_sum": error_sum.float().cpu(),
        "error_weight": error_weight.float().cpu(),
        "observation_count": observation_count.cpu(),
        "per_view": per_view,
        "protocol": {
            "raw_teacher_metric_target_only": True,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "field_checkpoint": (
                str(Path(args.field_checkpoint).resolve()) if field is not None else ""
            ),
            "view_residual_checkpoint": (
                str(Path(args.view_residual_checkpoint).resolve())
                if residual is not None
                else ""
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "output": str(output.resolve()),
                "selected_frame_ids": frames,
                "representation": representation,
                "mean_pixel_cosine": sum(
                    item["mean_pixel_cosine"] * item["visible_pixels"] for item in per_view
                )
                / max(1, sum(item["visible_pixels"] for item in per_view)),
                "per_view": per_view,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return payload


def compute_direction_stats(args: argparse.Namespace) -> dict:
    """Compute training-only per-row mean viewing directions for zero-mean residuals."""

    device, _config, model, renderer, dataset, mpr, metadata = _load_runtime(args)
    frame_to_index = {int(frame): index for index, frame in enumerate(dataset.frame_indices)}
    frames = [int(value) for value in metadata.get("selected_frame_indices", [])]
    if not frames or len(frames) != len(set(frames)):
        raise ValueError("MPR selected-frame metadata is empty or duplicated")
    missing = sorted(set(frames) - set(frame_to_index))
    if missing:
        raise ValueError(f"MPR frames unavailable: {missing}")
    depth_by_frame, alpha_by_frame = _precompute_training_geometry_maps(
        renderer,
        model,
        dataset,
        frame_to_index,
        frames,
        device,
        batch_size=int(args.geometry_render_batch_size),
    )
    num_rows = int(model.get_xyz().shape[0])
    xyz = model.get_xyz().detach().float()
    direction_sum = torch.zeros(num_rows, 3, dtype=torch.float64, device=device)
    direction_square_sum = torch.zeros(num_rows, 3, dtype=torch.float64, device=device)
    weight_sum = torch.zeros(num_rows, dtype=torch.float64, device=device)
    observation_count = torch.zeros(num_rows, dtype=torch.long, device=device)
    registration_kwargs = _registration_kwargs(metadata)

    with torch.inference_mode():
        for frame in frames:
            sample = dataset[frame_to_index[frame]]
            pose = sample["pose_w2c"].to(device)
            height, width = sample["radio_features"].shape[1:]
            signal = torch.ones(1, 1, height, width, device=device)
            frame_sum, frame_weight = rasterize_registered_view_features(
                model=model,
                renderer=renderer,
                viewmat=pose,
                siglip_feat=signal,
                depth_map=depth_by_frame[frame].to(device)[None],
                alpha_map=alpha_by_frame[frame].to(device)[None],
                **registration_kwargs,
            )
            valid = frame_weight > 0
            camera_center = torch.linalg.inv(pose.float())[:3, 3]
            direction = F.normalize(camera_center[None] - xyz, dim=-1, eps=1e-8)
            weight = frame_weight[valid].double()
            direction64 = direction[valid].double()
            direction_sum[valid] += weight[:, None] * direction64
            direction_square_sum[valid] += weight[:, None] * direction64.square()
            weight_sum[valid] += weight
            observation_count[valid] += 1
            print(
                f"[direction stats] frame {frame}: rows={int(valid.sum())}",
                flush=True,
            )
            del frame_sum, frame_weight, direction
            if device.type == "cuda":
                torch.cuda.empty_cache()

    expected_counts = torch.as_tensor(mpr["view_counts"]).long().to(device)
    count_delta = observation_count - expected_counts
    mismatch_rows = int((count_delta != 0).sum())
    mismatch_fraction = mismatch_rows / max(1, num_rows)
    max_delta = int(count_delta.abs().max())
    validity_flips = int(((observation_count > 0) != (expected_counts > 0)).sum())
    if (
        mismatch_fraction > float(args.max_count_mismatch_fraction)
        or max_delta > int(args.max_count_delta)
        or validity_flips > 0
    ):
        raise ValueError(
            "direction-stat replay exceeds count tolerance: "
            f"rows={mismatch_rows}, max_delta={max_delta}, flips={validity_flips}"
        )
    mean_direction = direction_sum / weight_sum.clamp_min(1e-12)[:, None]
    direction_variance = (
        direction_square_sum / weight_sum.clamp_min(1e-12)[:, None]
        - mean_direction.square()
    ).clamp_min(0.0)
    valid_rows = weight_sum > 0
    mean_direction[~valid_rows] = 0.0
    direction_variance[~valid_rows] = 0.0
    payload = {
        "schema_version": 1,
        "kind": "mpr_training_view_direction_stats",
        "geometry_xyz_sha256": str(metadata.get("xyz_sha256", "")),
        "mpr_cache": str(Path(args.mpr_cache).resolve()),
        "training_frame_ids": frames,
        "mean_view_direction": mean_direction.float().cpu(),
        "view_direction_variance": direction_variance.float().cpu(),
        "view_direction_resultant_length": mean_direction.norm(dim=-1).float().cpu(),
        "weight_sum": weight_sum.float().cpu(),
        "observation_count": observation_count.cpu(),
        "view_count_replay": {
            "exact": mismatch_rows == 0,
            "mismatch_rows": mismatch_rows,
            "mismatch_fraction": mismatch_fraction,
            "max_absolute_delta": max_delta,
            "validity_flip_rows": validity_flips,
        },
        "protocol": {
            "training_views_only": True,
            "direction": "normalize(camera_center_world - primitive_center_world)",
            "centering": "per_primitive_alpha_depth_weighted_mean",
            "geometry_visibility_render": "builder_legacy_feature_batch",
            "geometry_render_batch_size": int(args.geometry_render_batch_size),
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "output": str(output.resolve()),
                "training_views": len(frames),
                "valid_rows": int(valid_rows.sum()),
                "mean_resultant_length": float(
                    payload["view_direction_resultant_length"][valid_rows.cpu()].mean()
                ),
                "view_count_replay": payload["view_count_replay"],
                "protocol": payload["protocol"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return payload


def _bucket_report(
    view_counts: torch.Tensor,
    disagreement: torch.Tensor,
    heldout_error: torch.Tensor,
    valid: torch.Tensor,
) -> list[dict]:
    buckets = ((1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 10**9))
    rows: list[dict] = []
    for lower, upper in buckets:
        mask = valid & (view_counts >= lower) & (view_counts <= upper)
        rows.append(
            {
                "view_count": f"{lower}" if lower == upper else f"{lower}-{upper if upper < 10**9 else 'inf'}",
                "rows": int(mask.sum()),
                "mean_view_disagreement": float(disagreement[mask].mean()) if bool(mask.any()) else None,
                "mean_heldout_error": float(heldout_error[mask].mean()) if bool(mask.any()) else None,
            }
        )
    return rows


def summarize(args: argparse.Namespace) -> dict:
    paths = [Path(value) for value in str(args.training_partials).split(",") if value.strip()]
    partials = [torch.load(path, map_location="cpu") for path in paths]
    if not partials:
        raise ValueError("--training-partials is empty")
    geometry_hashes = {str(item.get("geometry_xyz_sha256", "")) for item in partials}
    mpr_paths = {str(item.get("mpr_cache", "")) for item in partials}
    if len(geometry_hashes) != 1 or len(mpr_paths) != 1:
        raise ValueError("training partial provenance differs")
    selected_lists = [list(map(int, item.get("selected_frame_ids", []))) for item in partials]
    all_selected = [frame for values in selected_lists for frame in values]
    if len(all_selected) != len(set(all_selected)):
        raise ValueError("training partial frame shards overlap")
    declared = {int(item.get("num_declared_views", -1)) for item in partials}
    if len(declared) != 1 or len(all_selected) != next(iter(declared)):
        raise ValueError("training shards do not cover all declared MPR views")

    sums = merge_training_partials(partials)
    consistency = consistency_from_sums(sums)
    mpr = torch.load(next(iter(mpr_paths)), map_location="cpu")
    expected_counts = torch.as_tensor(mpr["view_counts"]).long()
    replay_counts = torch.as_tensor(sums["observation_count"]).long()
    count_match = replay_counts == expected_counts
    mismatch_rows = int((~count_match).sum())
    mismatch_fraction = mismatch_rows / max(1, int(expected_counts.numel()))
    max_count_delta = int((replay_counts - expected_counts).abs().max())
    validity_flip_rows = int(((replay_counts > 0) != (expected_counts > 0)).sum())
    if (
        mismatch_fraction > float(args.max_count_mismatch_fraction)
        or max_count_delta > int(args.max_count_delta)
        or validity_flip_rows > 0
    ):
        raise ValueError(
            "replayed MPR view-count discrepancy exceeds declared numerical "
            f"tolerance: rows={mismatch_rows}, fraction={mismatch_fraction}, "
            f"max_delta={max_count_delta}, validity_flips={validity_flip_rows}"
        )
    heldout = torch.load(args.heldout_cache, map_location="cpu")
    if str(heldout.get("geometry_xyz_sha256", "")) not in geometry_hashes:
        raise ValueError("held-out cache geometry differs")
    heldout_weight = torch.as_tensor(heldout["error_weight"]).float()
    heldout_error = torch.as_tensor(heldout["error_sum"]).float() / heldout_weight.clamp_min(1e-8)
    valid = (
        (torch.as_tensor(sums["observation_count"]) >= 2)
        & (heldout_weight > 0)
        & torch.as_tensor(mpr["valid"]).bool()
    )
    disagreement = consistency["view_disagreement"]
    weighted_disagreement = consistency["weighted_view_disagreement"]
    correlation = pearson_spearman(disagreement[valid], heldout_error[valid])
    weighted_correlation = pearson_spearman(
        weighted_disagreement[valid], heldout_error[valid]
    )

    valid_disagreement = disagreement[valid]
    valid_error = heldout_error[valid]
    threshold_high = torch.quantile(valid_disagreement, 0.9)
    threshold_low = torch.quantile(valid_disagreement, 0.1)
    high = valid_disagreement >= threshold_high
    low = valid_disagreement <= threshold_low
    high_error = float(valid_error[high].mean())
    low_error = float(valid_error[low].mean())
    spearman = correlation["spearman"]
    context_material = (
        spearman is not None
        and float(spearman) >= float(args.material_spearman)
        and high_error - low_error >= float(args.material_error_gap)
    )
    report = {
        "schema_version": 1,
        "audit": "mpr_view_consistency_v1",
        "protocol": {
            "replays_exact_mpr_registration_mode": True,
            "training_frames": sorted(all_selected),
            "heldout_frames": list(map(int, heldout.get("selected_frame_ids", []))),
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
        "provenance": {
            "geometry_xyz_sha256": next(iter(geometry_hashes)),
            "mpr_cache": next(iter(mpr_paths)),
            "training_partials": [str(path.resolve()) for path in paths],
            "heldout_cache": str(Path(args.heldout_cache).resolve()),
            "view_count_replay": {
                "exact": mismatch_rows == 0,
                "mismatch_rows": mismatch_rows,
                "mismatch_fraction": mismatch_fraction,
                "max_absolute_delta": max_count_delta,
                "validity_flip_rows": validity_flip_rows,
                "allowed_mismatch_fraction": float(args.max_count_mismatch_fraction),
                "allowed_max_absolute_delta": int(args.max_count_delta),
                "interpretation": (
                    "independent CUDA raster threshold/top-1 tie replay tolerance"
                    if mismatch_rows
                    else "exact"
                ),
            },
        },
        "rows": {
            "total": int(expected_counts.numel()),
            "mpr_valid": int(torch.as_tensor(mpr["valid"]).sum()),
            "correlation_support_train_views_ge_2": int(valid.sum()),
        },
        "training_agreement": {
            "mean_view_disagreement": float(disagreement[expected_counts >= 2].mean()),
            "p90_view_disagreement": float(torch.quantile(disagreement[expected_counts >= 2], 0.9)),
            "mean_weighted_view_disagreement": float(
                weighted_disagreement[expected_counts >= 2].mean()
            ),
            "mean_effective_views": float(consistency["effective_views"][expected_counts > 0].mean()),
        },
        "heldout_prediction": {
            "correlation_unweighted_disagreement_vs_error": correlation,
            "correlation_weighted_disagreement_vs_error": weighted_correlation,
            "top10_disagreement_mean_error": high_error,
            "bottom10_disagreement_mean_error": low_error,
            "top_minus_bottom_error": high_error - low_error,
        },
        "by_training_view_count": _bucket_report(
            expected_counts, disagreement, heldout_error, valid
        ),
        "decision": {
            "material_spearman_threshold": float(args.material_spearman),
            "material_top_bottom_error_gap": float(args.material_error_gap),
            "view_context_is_material": context_material,
            "next_causal_test": (
                "zero_mean_view_conditioned_residual"
                if context_material
                else "compositing_topk_sharpen_depth_band"
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tensor_output = output.with_suffix(".pt")
    torch.save(
        {
            "schema_version": 1,
            "geometry_xyz_sha256": next(iter(geometry_hashes)),
            "mpr_cache": next(iter(mpr_paths)),
            "observation_count": torch.as_tensor(sums["observation_count"]).long(),
            **consistency,
            "heldout_error": heldout_error,
            "heldout_weight": heldout_weight,
            "correlation_valid": valid,
        },
        tensor_output,
    )
    return report


def _common_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--mpr-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training_parser = subparsers.add_parser("training-shard")
    _common_runtime_arguments(training_parser)
    training_parser.add_argument("--num-shards", type=int, default=1)
    training_parser.add_argument("--shard-index", type=int, default=0)
    training_parser.add_argument("--geometry-render-batch-size", type=int, default=4)
    heldout_parser = subparsers.add_parser("heldout")
    _common_runtime_arguments(heldout_parser)
    heldout_parser.add_argument("--frame-ids", default="")
    heldout_parser.add_argument("--field-checkpoint", default="")
    heldout_parser.add_argument("--view-residual-checkpoint", default="")
    direction_parser = subparsers.add_parser("direction-stats")
    _common_runtime_arguments(direction_parser)
    direction_parser.add_argument("--geometry-render-batch-size", type=int, default=4)
    direction_parser.add_argument("--max-count-mismatch-fraction", type=float, default=1e-4)
    direction_parser.add_argument("--max-count-delta", type=int, default=1)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--training-partials", required=True)
    summarize_parser.add_argument("--heldout-cache", required=True)
    summarize_parser.add_argument("--output", required=True)
    summarize_parser.add_argument("--material-spearman", type=float, default=0.20)
    summarize_parser.add_argument("--material-error-gap", type=float, default=0.02)
    summarize_parser.add_argument("--max-count-mismatch-fraction", type=float, default=1e-4)
    summarize_parser.add_argument("--max-count-delta", type=int, default=1)
    args = parser.parse_args()
    if args.command == "training-shard":
        payload = compute_training_shard(args)
        summary = {
            "output": str(Path(args.output).resolve()),
            "shard_index": payload["shard_index"],
            "views": len(payload["selected_frame_ids"]),
        }
    elif args.command == "heldout":
        payload = compute_heldout(args)
        summary = {
            "output": str(Path(args.output).resolve()),
            "views": len(payload["selected_frame_ids"]),
        }
    elif args.command == "direction-stats":
        payload = compute_direction_stats(args)
        summary = {
            "output": str(Path(args.output).resolve()),
            "views": len(payload["training_frame_ids"]),
            "valid_rows": int((payload["weight_sum"] > 0).sum()),
            "view_count_replay": payload["view_count_replay"],
        }
    else:
        report = summarize(args)
        summary = {
            "output": str(Path(args.output).resolve()),
            "training_agreement": report["training_agreement"],
            "heldout_prediction": report["heldout_prediction"],
            "decision": report["decision"],
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
