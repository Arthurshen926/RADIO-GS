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
    CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    CANONICAL_OBSERVATION_CONTRACT_NAME,
    apply_canonical_observation_contract,
    observation_contract_sha256,
    select_full_observation_coverage_ranked_dataset_indices,
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


def raster_fusion_reliability(
    features: torch.Tensor,
    valid: torch.Tensor,
    view_counts: torch.Tensor,
    *,
    num_views: int,
    mode: str,
    normalized_observations: bool,
) -> torch.Tensor:
    """Build query-free reliability for a raster-fused primitive target.

    For unit-normalized observations, the norm of their weighted mean is the
    mean resultant length: one for perfect directional agreement and lower as
    observations disagree. The legacy mode is retained for frozen contracts.
    """

    values = torch.as_tensor(features).float()
    active = torch.as_tensor(valid).bool().reshape(-1)
    counts = torch.as_tensor(view_counts).float().reshape(-1)
    if values.ndim != 2 or active.shape[0] != values.shape[0]:
        raise ValueError("raster features and validity must align by primitive")
    if counts.shape != active.shape or num_views <= 0:
        raise ValueError("raster view counts must align and num_views be positive")
    if mode not in {"legacy_valid", "mean_resultant"}:
        raise ValueError("unsupported raster reliability mode")
    coverage = (counts / float(num_views)).clamp(0.0, 1.0)
    if mode == "legacy_valid":
        agreement = active.float()
    else:
        if not normalized_observations:
            raise ValueError(
                "mean-resultant reliability requires normalized observations"
            )
        agreement = values.norm(dim=-1).clamp(0.0, 1.0)
        agreement = agreement.masked_fill(~active, 0.0)
    return torch.stack([coverage, agreement, active.float()], dim=-1).half()


# These names and dimensions are those emitted by the official C-RADIOv4-H
# runtime.  The direct path deliberately consumes the saved official adaptor
# output, rather than applying an MLP to an already interpolated raw map.
_EXTRACTED_CAPABILITY_SPECS: dict[str, dict[str, object]] = {
    "dino_v3": {
        "adaptor_name": "dino_v3_7b",
        "subdir": "dino_v3_7b",
        "output_dim": 4096,
    },
    "sam3": {
        "adaptor_name": "sam3",
        "subdir": "sam3",
        "output_dim": 1024,
    },
}


def _resolve_extracted_capability_source(
    feature_dir: str | Path,
    feature_space: str,
) -> dict[str, object]:
    """Verify an official C-RADIO adaptor map advertised by a frame manifest.

    A capability MPR may only consume this route when the same feature
    extraction run explicitly recorded the official adaptor output.  This is
    intentionally stricter than accepting a directory with compatible tensors:
    a forgotten ``--extract_adaptors`` flag must fail closed instead of
    silently reintroducing the legacy project-after-interpolation route.
    """

    space = str(feature_space).lower()
    if space not in _EXTRACTED_CAPABILITY_SPECS:
        raise ValueError(f"no extracted official capability source for {feature_space!r}")
    expected = dict(_EXTRACTED_CAPABILITY_SPECS[space])
    root = Path(feature_dir)
    manifest_path = root / "frame_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"official extracted capability source needs {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid feature frame manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("feature frame manifest must contain an object")
    features = manifest.get("features")
    adapters = features.get("adaptors") if isinstance(features, dict) else None
    if not isinstance(adapters, list):
        raise ValueError(
            f"feature manifest does not declare official adaptor maps for {space}"
        )
    matches = [
        value
        for value in adapters
        if isinstance(value, dict)
        and value.get("name") == expected["adaptor_name"]
        and value.get("subdir") == expected["subdir"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"feature manifest does not declare the matching official {space} adaptor"
        )
    record = matches[0]
    if int(record.get("dim", -1)) != int(expected["output_dim"]):
        raise ValueError(
            f"official {space} adaptor dimension differs from C-RADIO contract"
        )
    grid = record.get("grid")
    if (
        not isinstance(grid, list)
        or len(grid) != 2
        or any(int(value) <= 0 for value in grid)
    ):
        raise ValueError(f"official {space} adaptor manifest has an invalid spatial grid")
    subdir = root / str(expected["subdir"])
    if not subdir.is_dir():
        raise FileNotFoundError(
            f"feature manifest declares {space}, but its map directory is missing: {subdir}"
        )
    return {
        **expected,
        "native_grid": [int(grid[0]), int(grid[1])],
        "frame_manifest": str(manifest_path.resolve()),
        "frame_manifest_sha256": _sha256_file(manifest_path),
        "radio_version": str(dict(manifest.get("radio", {})).get("version", "")),
        "execution": "official_c_radio_runtime_adaptor_output",
    }


def _load_extracted_capability_maps(
    *,
    feature_dir: str | Path,
    feature_space: str,
    pose_file: str | None,
    pose_dir: str | None,
    feature_size: tuple[int, int],
    dataset_type: str,
    selected_frame_indices: list[int],
) -> tuple[torch.Tensor, dict[str, object]]:
    """Load selected native official-adaptor maps then resample for registration.

    The frozen adaptor is evaluated by the official C-RADIO runtime at its
    native token locations.  Interpolation is applied only afterwards to match
    the fixed Gaussian raster grid, which preserves the intended ordering
    ``MPR(resample(A_official(f_v)))``.
    """

    source = _resolve_extracted_capability_source(feature_dir, feature_space)
    dataset = SimpleRadioDataset(
        feature_dir=str(feature_dir),
        pose_file=pose_file,
        pose_dir=pose_dir,
        feature_size=feature_size,
        feature_subdir=str(source["subdir"]),
        split="train",
        dataset_type=dataset_type,
        frame_ids=list(selected_frame_indices),
    )
    if [int(value) for value in dataset.frame_indices] != [
        int(value) for value in selected_frame_indices
    ]:
        raise ValueError("official capability frame order differs from raw RADIO MPR")
    maps = None
    for index in range(len(dataset)):
        item = dataset[index]["radio_features"].float().cpu()
        if item.ndim != 3 or item.shape[0] != int(source["output_dim"]):
            raise ValueError(
                f"official {feature_space} map {index} has unexpected shape "
                f"{tuple(item.shape)}"
            )
        if maps is None:
            maps = torch.empty(
                (len(dataset), *item.shape),
                dtype=torch.float32,
            )
        elif item.shape != maps.shape[1:]:
            raise ValueError(
                f"official {feature_space} maps do not share one spatial shape"
            )
        maps[index].copy_(item)
    if maps is None:
        raise ValueError(f"official {feature_space} map selection is empty")
    if maps.ndim != 4 or maps.shape[1] != int(source["output_dim"]):
        raise ValueError(
            f"official {feature_space} maps have unexpected shape {tuple(maps.shape)}"
        )
    norms = torch.linalg.vector_norm(maps, ord=2, dim=1, keepdim=True)
    maps.div_(norms.clamp_min_(1e-8))
    return maps, source


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
    try:
        is_dir = path.is_dir()
        is_file = path.is_file()
    except OSError:
        # Long comma-separated CLI lists are values, not filesystem paths.
        is_dir = False
        is_file = False
    if is_dir:
        return {
            int(item.stem.split("_")[-1])
            for item in path.glob("frame_*.json")
        }
    if is_file:
        value = path.read_text(encoding="utf-8")
    tokens: list[str] = []
    for line in value.splitlines():
        tokens.extend(line.split("#", 1)[0].replace(",", " ").split())
    return {int(item) for item in tokens}


def _select_candidate_indices(candidates: list[int], max_views: int) -> list[int]:
    chosen = _select_indices(len(candidates), max_views)
    return [candidates[index] for index in chosen]


def _select_full_observation_coverage_views(
    *,
    scene_root: str | Path,
    dataset_frame_ids: list[int],
    candidates: list[int],
    maximum_views: int,
    minimum_source_views: int = 0,
) -> tuple[list[int], dict[str, object]]:
    """Restore a full-.sens field's label-free greedy coverage order.

    A numeric-sorted RGB-D directory is deliberately convenient for normal
    loaders, but it must not erase the coverage ranking that created it.  The
    source manifest has no target/object/click information and is checked here
    before it can influence MPR frame selection.
    """

    manifest_path = Path(scene_root) / "field_source_contract.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "canonical full-observation MPR requires "
            f"{manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("full-observation field-source contract must be an object")
    allowed_versions = {
        "scannet_full_observation_v1",
        "scannet_full_observation_pfpr_queryheldout_v1",
    }
    if str(payload.get("field_contract_version", "")) not in allowed_versions:
        raise ValueError(
            "full-observation MPR source contract has an unsupported provenance version"
        )
    for key in (
        "uses_private_anchor",
        "uses_private_depth_pixel",
        "uses_instances_or_semantic_labels",
        "contains_instance_or_label_directories",
    ):
        if bool(payload.get(key, False)):
            raise ValueError(
                "full-observation MPR source contract is not query/label free: "
                f"{key}"
            )
    selected_frame_ids = [int(value) for value in payload.get("selected_frame_indices", [])]
    ranked_frame_ids = [int(value) for value in payload.get("selection_order_frame_indices", [])]
    if len(set(selected_frame_ids)) != len(selected_frame_ids) or set(
        selected_frame_ids
    ) != set(ranked_frame_ids):
        raise ValueError(
            "full-observation source contract has an incomplete coverage ranking"
        )
    minimum_source_views = int(minimum_source_views)
    if minimum_source_views > 0 and len(selected_frame_ids) < minimum_source_views:
        raise ValueError(
            "full-observation MPR contract requires an independently "
            f"materialized {minimum_source_views}-view source prefix"
        )
    selected = select_full_observation_coverage_ranked_dataset_indices(
        dataset_frame_ids=dataset_frame_ids,
        candidate_dataset_indices=candidates,
        ranked_frame_ids=ranked_frame_ids,
        maximum_views=int(maximum_views),
    )
    return selected, {
        "full_observation_source_contract": str(manifest_path.resolve()),
        "full_observation_source_contract_sha256": _sha256_file(manifest_path),
        "full_observation_source_contract_version": str(
            payload["field_contract_version"]
        ),
        "full_observation_coverage_order_applied": True,
        "full_observation_source_view_count": int(len(selected_frame_ids)),
    }


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
    cpu_sum_staging: torch.Tensor | None = None,
    cpu_count_staging: torch.Tensor | None = None,
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
    maximum_chunk = min(int(channel_chunk_size), int(channels))
    if cpu_sum_staging is not None and (
        cpu_sum_staging.device.type != "cpu"
        or cpu_sum_staging.dtype != torch.float32
        or cpu_sum_staging.shape != (num_rows, maximum_chunk)
    ):
        raise ValueError("cpu_sum_staging must be float32 [N,min(C,chunk)] on CPU")
    if cpu_count_staging is not None and (
        cpu_count_staging.device.type != "cpu"
        or cpu_count_staging.dtype != torch.float32
        or cpu_count_staging.shape != (num_rows,)
    ):
        raise ValueError("cpu_count_staging must be float32 [N] on CPU")
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
    if cpu_count_staging is None:
        counts_cpu = frame_counts.cpu()
    else:
        cpu_count_staging.copy_(frame_counts)
        counts_cpu = cpu_count_staging
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
        if cpu_sum_staging is None:
            sums_cpu = sums.cpu()
        else:
            sums_cpu = cpu_sum_staging[:, : stop - start]
            sums_cpu.copy_(sums)
        registered_sum[:, start:stop].add_(sums_cpu)
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
    if str(getattr(args, "observation_contract", "legacy")) in {
        CANONICAL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    }:
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
    full_observation_source_metadata: dict[str, object] = {}
    if observation_contract is not None and bool(
        observation_contract["requires_full_observation_source_contract"]
    ):
        selected, full_observation_source_metadata = (
            _select_full_observation_coverage_views(
                scene_root=str(getattr(config, "scene_root", "")),
                dataset_frame_ids=[int(value) for value in dataset.frame_indices],
                candidates=candidates,
                maximum_views=int(args.max_views),
                minimum_source_views=int(
                    observation_contract.get("minimum_source_views", 0)
                ),
            )
        )
    else:
        selected = _select_candidate_indices(candidates, int(args.max_views))
    poses = torch.stack(
        [dataset[index]["pose_w2c"].float().cpu() for index in selected], dim=0
    )
    feature_space = str(args.feature_space).lower()
    capability_map_source = str(args.capability_map_source).lower()
    if capability_map_source not in {"project_raw", "official_extracted"}:
        raise ValueError(
            "capability_map_source must be project_raw or official_extracted"
        )
    selected_frame_indices = [
        int(dataset.frame_indices[index]) for index in selected
    ]
    summary_head_path = ""
    adaptor_name = ""
    adaptor_checkpoint_path = ""
    adaptor_checkpoint_sha256 = ""
    capability_source_metadata: dict[str, object] = {
        "capability_map_source": "not_applicable",
        "capability_native_map_manifest": "",
        "capability_native_map_manifest_sha256": "",
        "capability_native_map_grid": [],
        "capability_adaptor_execution": "not_applicable",
    }
    if feature_space in {"dino_v3", "sam3"} and capability_map_source == "official_extracted":
        teacher_maps, extracted_source = _load_extracted_capability_maps(
            feature_dir=feature_dir,
            feature_space=feature_space,
            pose_file=pose_file,
            pose_dir=pose_dir,
            feature_size=(feature_height, feature_width),
            dataset_type=str(getattr(config, "dataset_type", "lerf")),
            selected_frame_indices=selected_frame_indices,
        )
        adaptor_name = str(extracted_source["adaptor_name"])
        adaptor_checkpoint_path = str(Path(args.radio_checkpoint).expanduser().resolve())
        adaptor_checkpoint_sha256 = _sha256_file(adaptor_checkpoint_path)
        capability_source_metadata = {
            "capability_map_source": "official_extracted",
            "capability_native_map_manifest": str(
                extracted_source["frame_manifest"]
            ),
            "capability_native_map_manifest_sha256": str(
                extracted_source["frame_manifest_sha256"]
            ),
            "capability_native_map_grid": list(extracted_source["native_grid"]),
            "capability_adaptor_execution": str(extracted_source["execution"]),
        }
    else:
        teacher_maps = torch.stack(
            [dataset[index]["radio_features"].float().cpu() for index in selected],
            dim=0,
        )
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
    elif (
        feature_space in {"dino_v3", "sam3"}
        and capability_map_source != "official_extracted"
    ):
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
        capability_source_metadata = {
            "capability_map_source": "project_raw",
            "capability_native_map_manifest": "",
            "capability_native_map_manifest_sha256": "",
            "capability_native_map_grid": [],
            "capability_adaptor_execution": "frozen_official_feature_projection_after_raw_resampling",
        }
    elif feature_space not in {"radio", "semantic_descriptor", "dino_v3", "sam3"}:
        raise ValueError(f"Unsupported feature space: {feature_space}")

    xyz_cpu = model.get_xyz().detach().float().cpu().contiguous()
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
        contribution_sum_staging = None
        contribution_count_staging = None
        if args.raster_view_fusion == "contribution_mean":
            contribution_sum_staging = torch.empty(
                xyz_cpu.shape[0],
                min(
                    int(args.raster_channel_chunk_size),
                    int(teacher_maps.shape[1]),
                ),
                dtype=torch.float32,
            )
            contribution_count_staging = torch.empty(
                xyz_cpu.shape[0], dtype=torch.float32
            )
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
                            cpu_sum_staging=contribution_sum_staging,
                            cpu_count_staging=contribution_count_staging,
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
        reliability = raster_fusion_reliability(
            features,
            valid,
            view_counts,
            num_views=max(1, len(selected)),
            mode=str(getattr(args, "raster_reliability_mode", "legacy_valid")),
            normalized_observations=bool(args.normalize_each_view),
        )
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
        "raster_reliability_mode": str(
            getattr(args, "raster_reliability_mode", "legacy_valid")
        ),
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
        **full_observation_source_metadata,
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
        **capability_source_metadata,
        "registration_responsibility_cache": responsibility_cache_path,
        "registration_responsibility_cache_sha256": responsibility_cache_sha256,
        "shared_registration_responsibility": bool(
            responsibility_cache_sha256
        ),
        "registration_responsibility_contract": responsibility_contract,
        "capability_projection_before_mpr": feature_space
        in {"dino_v3", "sam3"},
        "custom_adaptor_head": False,
        "source": (
            "official_crop_summary_mpr"
            if feature_space == "semantic_descriptor"
            and args.semantic_descriptor_source == "official_siglip2_crop_summary"
            else ""
        ),
        "official_summary_head": (
            True
            if feature_space == "semantic_descriptor"
            and args.semantic_descriptor_source == "official_siglip2_crop_summary"
            else None
        ),
        "custom_text_projection": (
            False
            if feature_space == "semantic_descriptor"
            and args.semantic_descriptor_source == "official_siglip2_crop_summary"
            else None
        ),
        "semantic_alignment_level": (
            2
            if feature_space == "semantic_descriptor"
            and args.semantic_descriptor_source == "official_siglip2_crop_summary"
            else None
        ),
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
        choices=[
            "legacy",
            CANONICAL_OBSERVATION_CONTRACT_NAME,
            CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
            CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
            CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
        ],
        default=CANONICAL_OBSERVATION_CONTRACT_NAME,
        help=(
            "Versioned query-free lifting policy. canonical-full-observation "
            "variants restore the field-source coverage ranking; v1 is the "
            "frozen 240-view control; v2/v3 preserve independent 480/960-view "
            "source prefixes."
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
        "--semantic-descriptor-source",
        choices=["unspecified", "official_siglip2_crop_summary"],
        default="unspecified",
        help="Auditable provenance for precomputed semantic_descriptor maps.",
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
    parser.add_argument(
        "--capability-map-source",
        choices=["project_raw", "official_extracted"],
        default="project_raw",
        help=(
            "For DINO/SAM MPR, either apply the frozen feature projection to "
            "the raw map (legacy) or consume the matching native official "
            "C-RADIO adaptor maps emitted by extract_radio_features."
        ),
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
        "--raster-reliability-mode",
        choices=["legacy_valid", "mean_resultant"],
        default="legacy_valid",
        help=(
            "Keep frozen [coverage,valid,valid] reliability or record the "
            "mean-resultant directional agreement of normalized observations."
        ),
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
    if (
        args.raster_reliability_mode != "legacy_valid"
        and args.aggregation_mode == "center"
    ):
        parser.error(
            "--raster-reliability-mode applies only to raster aggregation"
        )
    if (
        args.raster_reliability_mode == "mean_resultant"
        and not args.normalize_each_view
    ):
        parser.error(
            "--raster-reliability-mode mean_resultant requires "
            "--normalize-each-view"
        )
    print(json.dumps(build_cache(args), indent=2))


if __name__ == "__main__":
    main()
