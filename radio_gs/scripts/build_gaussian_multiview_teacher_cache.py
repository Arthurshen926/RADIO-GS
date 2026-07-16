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
from radio_gs.field.observation_lifting_contract import (
    CANONICAL_OBSERVATION_CONTRACT_NAME,
    apply_canonical_observation_contract,
    observation_contract_sha256,
)
from radio_gs.models.radio_adaptors import (
    load_radio_adaptor_from_checkpoint,
    project_feature_map_with_adaptor,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.training.feature_training_utils import (
    SimpleRadioDataset,
    sample_multiview_radio_targets,
)
from radio_gs.training.primitive_consensus import robust_multiview_consensus


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def project_official_capability_maps(
    teacher_maps: torch.Tensor,
    adaptor: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Project complete 2-D RADIO grids before any 3-D aggregation.

    This order is intentional: for a nonlinear official adaptor,
    ``MPR(A(f_v))`` is not equivalent to ``A(MPR(f_v))``.  Returned maps are
    normalized official spatial features and contain no query information.
    """

    maps = torch.as_tensor(teacher_maps)
    if maps.ndim != 4 or maps.shape[1] != 1280:
        raise ValueError("teacher RADIO maps must be [V,1280,H,W]")
    if int(batch_size) <= 0:
        raise ValueError("projection batch size must be positive")
    parts: list[torch.Tensor] = []
    for start in tqdm(
        range(0, maps.shape[0], int(batch_size)),
        desc="project teacher capability space",
    ):
        projected = project_feature_map_with_adaptor(
            maps[start : start + int(batch_size)].to(device),
            adaptor,
            normalize=True,
        )
        parts.append(projected.half().cpu())
    return torch.cat(parts, dim=0)


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


def merge_topk_view_observations(
    current_features: torch.Tensor,
    current_responsibility: torch.Tensor,
    observation: torch.Tensor,
    responsibility: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain whole view vectors ranked by compositing responsibility.

    Selecting top-k independently for each feature channel creates a synthetic
    descriptor whose channels come from different views.  The ranking scalar
    is therefore kept separately and its selected slot indexes are applied to
    every channel of the corresponding view observation.
    """
    if current_features.ndim != 3 or current_responsibility.ndim != 2:
        raise ValueError("current top-k state must be [N,D,K] and [N,K]")
    if current_features.shape[0] != current_responsibility.shape[0] or (
        current_features.shape[2] != current_responsibility.shape[1]
    ):
        raise ValueError("top-k feature and responsibility slots do not align")
    if observation.shape != current_features.shape[:2]:
        raise ValueError("new view observation must be [N,D]")
    if responsibility.shape != (current_features.shape[0],):
        raise ValueError("new view responsibility must be [N]")
    candidates = torch.cat(
        [current_responsibility, responsibility[:, None]], dim=1
    )
    selected_responsibility, selected_slots = torch.topk(
        candidates, k=current_responsibility.shape[1], dim=1
    )
    feature_candidates = torch.cat(
        [current_features, observation.unsqueeze(-1)], dim=-1
    )
    selected_features = torch.gather(
        feature_candidates,
        2,
        selected_slots[:, None, :].expand(-1, current_features.shape[1], -1),
    )
    return selected_features, selected_responsibility


def accumulate_contribution_mean_channel_chunked(
    feature_map: torch.Tensor,
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    weights: torch.Tensor,
    registered_sum: torch.Tensor,
    registered_counts: torch.Tensor,
    *,
    channel_chunk_size: int,
) -> torch.Tensor:
    """Accumulate one view without materializing a dense ``N x C`` CUDA tensor.

    The operation is mathematically identical to full-channel weighted
    ``index_add``.  Only channel blocks are materialized on the GPU; the global
    contribution sum remains on CPU.  This is required for official 4096-D
    DINO targets on million-primitive scenes.
    """

    features = feature_map[0] if feature_map.ndim == 4 else feature_map
    if features.ndim != 3:
        raise ValueError("feature_map must be [C,H,W] or [1,C,H,W]")
    num_rows, channels = registered_sum.shape
    if registered_counts.shape != (num_rows,) or channels != features.shape[0]:
        raise ValueError("registered accumulation tensors do not align")
    if channel_chunk_size <= 0:
        raise ValueError("channel_chunk_size must be positive")
    device = features.device
    gids = torch.as_tensor(gaussian_ids, device=device).long()
    pids = torch.as_tensor(pixel_ids, device=device).long()
    weight = torch.as_tensor(weights, device=device).float()
    height, width = features.shape[1:]
    valid = (
        (gids >= 0)
        & (gids < num_rows)
        & (pids >= 0)
        & (pids < height * width)
        & (weight > 0)
    )
    gids, pids, weight = gids[valid], pids[valid], weight[valid]
    frame_counts = torch.zeros(num_rows, dtype=torch.float32, device=device)
    if gids.numel():
        frame_counts.index_add_(0, gids, weight)
    counts_cpu = frame_counts.cpu()
    registered_counts.add_(counts_cpu)
    if not gids.numel():
        return counts_cpu
    for start in range(0, channels, int(channel_chunk_size)):
        stop = min(start + int(channel_chunk_size), channels)
        flat = features[start:stop].float().reshape(stop - start, height * width).t()
        sampled = flat[pids] * weight[:, None]
        sums = torch.zeros(
            num_rows, stop - start, dtype=torch.float32, device=device
        )
        sums.index_add_(0, gids, sampled)
        registered_sum[:, start:stop].add_(sums.cpu())
        del flat, sampled, sums
    return counts_cpu


def _gaussian_state_sha256(model: torch.nn.Module) -> str:
    """Hash every primitive attribute that affects raster responsibility."""

    digest = hashlib.sha256()
    for name, values in (
        ("xyz", model.get_xyz()),
        ("rotation", model.get_rotation()),
        ("scaling", model.get_scaling()),
        ("opacity", model.get_opacity()),
    ):
        tensor = values.detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _responsibility_contract(
    *,
    args: argparse.Namespace,
    selected: list[int],
    selected_frame_indices: list[int],
    poses: torch.Tensor,
    renderer,
    model: torch.nn.Module,
    feature_height: int,
    feature_width: int,
) -> dict:
    """Build the exact feature-independent registration contract."""

    return {
        "schema_version": 1,
        "assignment_mode": "raster_gaussian_top1",
        "registration_weight_mode": str(args.registration_weight_mode),
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "selected_dataset_indices": list(selected),
        "selected_frame_indices": list(selected_frame_indices),
        "excluded_frame_ids": sorted(_parse_frame_ids(args.exclude_frame_ids)),
        "feature_height": int(feature_height),
        "feature_width": int(feature_width),
        "depth_tolerance": float(args.depth_tolerance),
        "relative_depth_tolerance": float(args.relative_depth_tolerance),
        "alpha_threshold": float(args.alpha_threshold),
        "pose_sha256": _sha256_tensor_rows(poses),
        "intrinsics_sha256": _sha256_tensor_rows(
            renderer.scaled_intrinsics(feature_width, feature_height)
        ),
        "xyz_sha256": _sha256_tensor_rows(model.get_xyz()),
        "gaussian_state_sha256": _gaussian_state_sha256(model),
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }


def _load_responsibility_cache(
    path: str | Path,
    *,
    expected_contract: dict,
    num_gaussians: int,
) -> tuple[list[dict[str, torch.Tensor]], str]:
    """Load a shared registration sidecar and fail closed on any mismatch."""

    cache_path = Path(path)
    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("responsibility cache must use schema version 1")
    metadata = dict(payload.get("metadata", {}))
    # ``config`` and dataset-local row indices are feature-source aliases, not
    # registration identity.  A semantic directory may deliberately omit held
    # out images, shifting local indices.  Frame IDs plus pose/intrinsics and
    # complete Gaussian-state hashes are the fail-closed geometric identity.
    alias_fields = {"config", "selected_dataset_indices"}
    mismatched = [
        key
        for key, expected in expected_contract.items()
        if key not in alias_fields and metadata.get(key) != expected
    ]
    if mismatched:
        raise ValueError(
            f"responsibility cache contract differs: {sorted(mismatched)}"
        )
    assignments = payload.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(
        expected_contract["selected_frame_indices"]
    ):
        raise ValueError("responsibility cache does not cover the selected views")
    checked: list[dict[str, torch.Tensor]] = []
    for view_index, item in enumerate(assignments):
        if not isinstance(item, dict):
            raise ValueError(f"responsibility view {view_index} is malformed")
        gaussian_ids = torch.as_tensor(item.get("gaussian_ids")).long().cpu()
        pixel_ids = torch.as_tensor(item.get("pixel_ids")).long().cpu()
        weights = torch.as_tensor(item.get("weights")).float().cpu()
        if (
            gaussian_ids.ndim != 1
            or pixel_ids.shape != gaussian_ids.shape
            or weights.shape != gaussian_ids.shape
        ):
            raise ValueError(f"responsibility view {view_index} tensors do not align")
        if gaussian_ids.numel() and (
            int(gaussian_ids.min()) < 0
            or int(gaussian_ids.max()) >= int(num_gaussians)
        ):
            raise ValueError(f"responsibility view {view_index} has invalid Gaussian IDs")
        num_pixels = int(expected_contract["feature_height"]) * int(
            expected_contract["feature_width"]
        )
        if pixel_ids.numel() and (
            int(pixel_ids.min()) < 0 or int(pixel_ids.max()) >= num_pixels
        ):
            raise ValueError(f"responsibility view {view_index} has invalid pixel IDs")
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError(f"responsibility view {view_index} has invalid weights")
        checked.append(
            {
                "gaussian_ids": gaussian_ids,
                "pixel_ids": pixel_ids,
                "weights": weights,
            }
        )
    return checked, _sha256_file(cache_path)


def build_cache(args: argparse.Namespace) -> dict:
    observation_contract = None
    if str(getattr(args, "observation_contract", "legacy")) == CANONICAL_OBSERVATION_CONTRACT_NAME:
        observation_contract = apply_canonical_observation_contract(args)
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
    included_frame_ids = _parse_frame_ids(
        str(getattr(args, "include_frame_ids", "") or "")
    )
    dataset = SimpleRadioDataset(
        feature_dir=str(feature_dir),
        pose_file=pose_file,
        pose_dir=pose_dir,
        feature_size=(feature_height, feature_width),
        split="train",
        dataset_type=str(getattr(config, "dataset_type", "lerf")),
        frame_ids=sorted(included_frame_ids) if included_frame_ids else None,
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
    adaptor_name = ""
    adaptor_checkpoint_path = ""
    adaptor_checkpoint_sha256 = ""
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
    elif feature_space in {"dino_v3", "sam3"}:
        adaptor_name = "dino_v3_7b" if feature_space == "dino_v3" else "sam3"
        adaptor_checkpoint_path = str(
            Path(args.radio_checkpoint).expanduser().resolve()
        )
        adaptor_checkpoint_sha256 = _sha256_file(adaptor_checkpoint_path)
        adaptor = load_radio_adaptor_from_checkpoint(
            adaptor_checkpoint_path,
            adaptor_name,
            kind="feature_projection",
        ).to(device).eval()
        adaptor.requires_grad_(False)
        teacher_maps = project_official_capability_maps(
            teacher_maps,
            adaptor,
            device=device,
            batch_size=int(args.projection_batch_size),
        )
        del adaptor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    elif feature_space not in {"radio", "semantic_descriptor"}:
        raise ValueError(f"Unsupported feature space: {feature_space}")

    xyz_cpu = model.get_xyz().detach().float().cpu().contiguous()
    selected_frame_indices = [
        int(dataset.frame_indices[index]) for index in selected
    ]
    responsibility_assignments: list[dict[str, torch.Tensor]] | None = None
    responsibility_cache_path = ""
    responsibility_cache_sha256 = ""
    responsibility_contract: dict = {}
    if args.responsibility_cache or args.save_responsibility_cache:
        if args.aggregation_mode != "raster_gaussian_top1":
            raise ValueError(
                "shared responsibility caches require raster_gaussian_top1 aggregation"
            )
        if args.responsibility_cache and args.save_responsibility_cache:
            raise ValueError(
                "load and save responsibility cache options are mutually exclusive"
            )
        responsibility_contract = _responsibility_contract(
            args=args,
            selected=selected,
            selected_frame_indices=selected_frame_indices,
            poses=poses,
            renderer=renderer,
            model=model,
            feature_height=feature_height,
            feature_width=feature_width,
        )
        if args.responsibility_cache:
            responsibility_cache_path = str(
                Path(args.responsibility_cache).expanduser().resolve()
            )
            responsibility_assignments, responsibility_cache_sha256 = (
                _load_responsibility_cache(
                    responsibility_cache_path,
                    expected_contract=responsibility_contract,
                    num_gaussians=int(xyz_cpu.shape[0]),
                )
            )

    # Visibility is already encoded in a loaded sidecar.  Otherwise render it
    # once, and optionally freeze the resulting registration assignments for
    # all feature spaces.
    depth_maps = None
    alpha_maps = None
    if responsibility_assignments is None:
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

    feature_parts: list[torch.Tensor] = []
    valid_parts: list[torch.Tensor] = []
    count_parts: list[torch.Tensor] = []
    reliability_parts: list[torch.Tensor] = []
    view_chunk = max(1, int(args.view_chunk_size))
    point_chunk = max(1, int(args.point_chunk_size))
    if args.aggregation_mode == "center":
        if depth_maps is None or alpha_maps is None:
            raise RuntimeError("center aggregation requires rendered visibility maps")
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
            accumulate_raster_contribution_features,
            raster_adjoint_registered_view_features,
            rasterize_registered_view_assignments,
        )

        registered_sum = torch.zeros(
            xyz_cpu.shape[0], teacher_maps.shape[1], dtype=torch.float32
        )
        registered_counts = torch.zeros(xyz_cpu.shape[0], dtype=torch.float32)
        observation_counts = torch.zeros(xyz_cpu.shape[0], dtype=torch.long)
        topk_observations = None
        topk_responsibility = None
        if args.raster_view_fusion == "topk_mean":
            topk_observations = torch.zeros(
                (
                    xyz_cpu.shape[0],
                    teacher_maps.shape[1],
                    max(1, int(args.raster_topk)),
                ),
                dtype=torch.float32,
            )
            topk_responsibility = torch.full(
                (xyz_cpu.shape[0], max(1, int(args.raster_topk))),
                -float("inf"),
                dtype=torch.float32,
            )
        aggregation_context = (
            contextlib.nullcontext()
            if args.aggregation_mode == "raster_adjoint"
            else torch.inference_mode()
        )
        captured_assignments: list[dict[str, torch.Tensor]] = []
        with aggregation_context:
            for view_index in tqdm(
                range(len(selected)), desc="aggregate raster contributions"
            ):
                if args.aggregation_mode == "raster_adjoint":
                    if alpha_maps is None:
                        raise RuntimeError(
                            "raster adjoint aggregation requires rendered alpha maps"
                        )
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
                    if responsibility_assignments is not None:
                        assignment = responsibility_assignments[view_index]
                        gaussian_ids = assignment["gaussian_ids"].to(device)
                        pixel_ids = assignment["pixel_ids"].to(device)
                        weights = assignment["weights"].to(device)
                    else:
                        if depth_maps is None or alpha_maps is None:
                            raise RuntimeError(
                                "raster registration requires visibility maps or a sidecar"
                            )
                        gaussian_ids, pixel_ids, weights = (
                            rasterize_registered_view_assignments(
                                model=model,
                                renderer=renderer,
                                viewmat=poses[view_index].to(device),
                                image_height=feature_height,
                                image_width=feature_width,
                                depth_map=depth_maps[
                                    view_index : view_index + 1
                                ].to(device),
                                alpha_map=alpha_maps[
                                    view_index : view_index + 1
                                ].to(device),
                                registration_depth_tolerance=float(
                                    args.depth_tolerance
                                ),
                                registration_relative_depth_tolerance=float(
                                    args.relative_depth_tolerance
                                ),
                                registration_alpha_threshold=float(
                                    args.alpha_threshold
                                ),
                                registration_weight_mode=args.registration_weight_mode,
                                gaussian_top1=True,
                            )
                        )
                        if args.save_responsibility_cache:
                            captured_assignments.append(
                                {
                                    "gaussian_ids": gaussian_ids.int().cpu(),
                                    "pixel_ids": pixel_ids.int().cpu(),
                                    "weights": weights.float().cpu(),
                                }
                            )
                    view_features = teacher_maps[
                        view_index : view_index + 1
                    ].to(device=device, dtype=torch.float32)
                    if bool(args.normalize_each_view):
                        view_features = F.normalize(view_features, dim=1, eps=1e-8)
                    if args.raster_view_fusion == "contribution_mean":
                        counts_cpu = accumulate_contribution_mean_channel_chunked(
                            view_features,
                            gaussian_ids,
                            pixel_ids,
                            weights,
                            registered_sum,
                            registered_counts,
                            channel_chunk_size=int(args.raster_channel_chunk_size),
                        )
                        frame_valid = counts_cpu > 0
                        observation_counts[frame_valid] += 1
                        del view_features, counts_cpu
                        continue
                    frame_sum, frame_counts = accumulate_raster_contribution_features(
                        view_features,
                        gaussian_ids,
                        pixel_ids,
                        weights,
                        n_gaussians=int(xyz_cpu.shape[0]),
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
                            assert topk_responsibility is not None
                            selected_features, selected_responsibility = (
                                merge_topk_view_observations(
                                    topk_observations[frame_valid],
                                    topk_responsibility[frame_valid],
                                    frame_observation,
                                    counts_cpu[frame_valid],
                                )
                            )
                            topk_observations[frame_valid] = selected_features
                            topk_responsibility[frame_valid] = selected_responsibility
                        else:
                            raise ValueError(
                                f"unsupported raster view fusion: {args.raster_view_fusion}"
                            )
                    observation_counts[frame_valid] += 1
                del frame_sum, frame_counts
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if args.save_responsibility_cache:
            if len(captured_assignments) != len(selected):
                raise RuntimeError("failed to capture every registration view")
            responsibility_output = Path(
                args.save_responsibility_cache
            ).expanduser()
            responsibility_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": 1,
                    "metadata": responsibility_contract,
                    "assignments": captured_assignments,
                },
                responsibility_output,
            )
            responsibility_cache_path = str(responsibility_output.resolve())
            responsibility_cache_sha256 = _sha256_file(responsibility_output)
        if args.raster_view_fusion == "topk_mean":
            assert topk_observations is not None
            assert topk_responsibility is not None
            finite = torch.isfinite(topk_responsibility)
            valid = finite.any(dim=1)
            features = (
                (topk_observations * finite[:, None, :]).sum(dim=-1)
                / finite.sum(dim=-1, keepdim=True).clamp_min(1)
            ).half()
            features[~valid] = 0.0
            del topk_observations, topk_responsibility, finite
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
        "raster_topk_ranking": (
            "whole_view_compositing_responsibility"
            if args.raster_view_fusion == "topk_mean"
            else "not_applicable"
        ),
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_declared_views": len(selected),
        "selected_dataset_indices": selected,
        "selected_frame_indices": selected_frame_indices,
        "excluded_frame_ids": sorted(excluded_frame_ids),
        "depth_tolerance": float(args.depth_tolerance),
        "relative_depth_tolerance": float(args.relative_depth_tolerance),
        "alpha_threshold": float(args.alpha_threshold),
        "normalize_each_view": bool(args.normalize_each_view),
        "per_view_normalization_applied": bool(args.normalize_each_view),
        "per_view_normalization_stage": (
            "pixel_feature_before_raster_lifting"
            if args.aggregation_mode != "center" and args.normalize_each_view
            else "sampled_feature_before_center_fusion"
            if args.normalize_each_view
            else "disabled"
        ),
        "robust_mpr": bool(args.robust_mpr and args.aggregation_mode == "center"),
        "robust_temperature": float(args.robust_temperature),
        "robust_iterations": int(args.robust_iterations),
        "summary_head_weights": summary_head_path,
        "official_adaptor_name": adaptor_name,
        "official_adaptor_checkpoint": adaptor_checkpoint_path,
        "official_adaptor_checkpoint_sha256": adaptor_checkpoint_sha256,
        "registration_responsibility_cache": responsibility_cache_path,
        "registration_responsibility_cache_sha256": responsibility_cache_sha256,
        "shared_registration_responsibility": bool(
            responsibility_cache_sha256
        ),
        "registration_responsibility_contract": responsibility_contract,
        "capability_projection_before_mpr": feature_space
        in {"dino_v3", "sam3"},
        "custom_adaptor_head": False,
        "query_names": [
            value.strip() for value in str(args.query_names).split(",") if value.strip()
        ],
        "xyz_sha256": _sha256_tensor_rows(xyz_cpu),
        "benchmark_masks_opened": False,
        "benchmark_images_opened": bool(args.benchmark_images_opened),
        "text_queries_opened": bool(args.text_queries_opened),
    }
    if observation_contract is not None:
        metadata["observation_lifting_contract"] = observation_contract
        metadata["observation_lifting_contract_sha256"] = observation_contract_sha256(
            observation_contract
        )
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
    parser.add_argument(
        "--observation-contract",
        choices=["legacy", CANONICAL_OBSERVATION_CONTRACT_NAME],
        default=CANONICAL_OBSERVATION_CONTRACT_NAME,
        help=(
            "Versioned dataset-independent lifting policy. canonical-mpr-v1 "
            "overrides all policy knobs while retaining dataset provenance."
        ),
    )
    parser.add_argument("--max-views", type=int, default=32)
    parser.add_argument(
        "--exclude-frame-ids",
        default="",
        help="Comma list, text file, or LERF label directory of held-out frame IDs.",
    )
    parser.add_argument(
        "--include-frame-ids",
        default="",
        help=(
            "Optional comma list/file restricting observations before pose loading; "
            "use for RGB frames without registered cameras."
        ),
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
        choices=[
            "radio",
            "dino_v3",
            "sam3",
            "siglip_summary",
            "semantic_descriptor",
        ],
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
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
        help="Official C-RADIO checkpoint used only for frozen capability projections.",
    )
    parser.add_argument("--projection-batch-size", type=int, default=2)
    parser.add_argument(
        "--responsibility-cache",
        default="",
        help=(
            "Feature-independent raster_gaussian_top1 assignment sidecar. "
            "Using the same sidecar makes raw and capability MPR observation "
            "support exactly identical."
        ),
    )
    parser.add_argument(
        "--save-responsibility-cache",
        default="",
        help=(
            "Save the query-free pixel-to-Gaussian assignment generated by "
            "this run for exact reuse by other feature spaces."
        ),
    )
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
        "--raster-channel-chunk-size",
        type=int,
        default=256,
        help="Exact channel chunking for contribution-mean raster accumulation.",
    )
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
