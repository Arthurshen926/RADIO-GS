#!/usr/bin/env python3
"""Evaluate LERF-OVS direct 3D object selection.

This script implements an OpenGaussian-style protocol for RADIO-GS:

1. Decode pre-refiner Gaussian/primitive features at 3D Gaussian centers, or
   register rendered SigLIP2-aligned features back to visible primitives.
2. Select 3D primitives from text-Gaussian similarity scores.
3. Render selected primitives as binary masks on the official LERF-OVS views.
4. Report mIoU, Acc@0.25, and Acc@0.50 against LERF-OVS masks.

The registered-view path is View-to-Primitive Registration (VPR): text queries
still operate on Gaussian primitives, while rendered teacher-compatible features
provide the registration signal.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, ".")

from radio_gs.data.lerf_dataset import LERFDataset
from radio_gs.data.benchmark_paths import (
    extract_feature_frame_index,
    list_feature_paths,
    resolve_split_frame_ids,
    resolve_split_feature_dir,
)
from radio_gs.geometry_utils import resolve_use_2dgs
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts.eval_lerf_grounding import (
    DEFAULT_GT_FEATURE_ROOT,
    DEFAULT_LABEL_DIR,
    DEFAULT_PROMPT_TEMPLATES,
    LERF_OVS_SCENES,
    build_gt_masks,
    load_lerf_rgb_frame,
    load_lerf_ovs_labels,
    load_or_generate_prompt_ensemble_embeddings,
    load_render_pipeline,
    parse_prompt_templates,
    project_to_siglip2,
    render_1280d,
    resolve_lerf_label_dir,
    resolve_lerf_scene_root,
)
from radio_gs.scripts.train_feature_field import sample_multiview_radio_targets
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer

logger = logging.getLogger(__name__)
SCORE_CACHE_VERSION = 1


OPEN_GAUSSIAN_LERF_FRAMES: Dict[str, List[int]] = {
    "waldo_kitchen": [53, 66, 89, 140, 154],
    "ramen": [6, 24, 60, 65, 81, 119, 128],
    "figurines": [41, 105, 152, 195],
    "teatime": [2, 25, 43, 107, 129, 140],
}


@dataclass(frozen=True)
class SelectionSpec:
    mode: str
    value: float

    @property
    def tag(self) -> str:
        if self.mode == "top_ratio":
            return f"top{self.value:g}".replace(".", "p")
        if self.mode == "score_threshold":
            return f"thr{self.value:g}".replace(".", "p")
        if self.mode == "mean_std":
            return f"meanstd{self.value:g}".replace(".", "p")
        return f"{self.mode}_{self.value:g}".replace(".", "p")


class GaussianSelectionProxy:
    """Geometry proxy whose feature vectors are per-query selection masks."""

    def __init__(self, base_model: torch.nn.Module, features: torch.Tensor) -> None:
        self.base_model = base_model
        self.features = features

    def get_xyz(self) -> torch.Tensor:
        return self.base_model.get_xyz()

    def get_rotation(self) -> torch.Tensor:
        return self.base_model.get_rotation()

    def get_scaling(self) -> torch.Tensor:
        return self.base_model.get_scaling()

    def get_opacity(self) -> torch.Tensor:
        return self.base_model.get_opacity()

    def get_features(self) -> torch.Tensor:
        return self.features


def _canonical_cache_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _canonical_cache_value(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_cache_value(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_score_cache_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return JSON-stable metadata used to guard VPR score-cache reuse."""
    return {
        str(key): _canonical_cache_value(value)
        for key, value in sorted(metadata.items())
    }


def save_score_cache(
    path: str | Path,
    scores: torch.Tensor,
    *,
    metadata: Dict[str, Any],
    registration_stats: Dict[str, Any],
) -> None:
    """Persist pre-aggregation primitive text scores with protocol metadata."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": SCORE_CACHE_VERSION,
            "metadata": canonical_score_cache_metadata(metadata),
            "registration_stats": _canonical_cache_value(registration_stats),
            "scores": scores.detach().cpu(),
        },
        cache_path,
    )


def load_score_cache(
    path: str | Path,
    *,
    expected_metadata: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load cached primitive text scores after exact protocol matching."""
    cache_path = Path(path)
    payload = torch.load(cache_path, map_location="cpu")
    version = int(payload.get("version", -1))
    if version != SCORE_CACHE_VERSION:
        raise ValueError(
            f"unsupported score cache version {version}; expected {SCORE_CACHE_VERSION}"
        )
    expected = canonical_score_cache_metadata(expected_metadata)
    cached = canonical_score_cache_metadata(payload.get("metadata", {}))
    if cached != expected:
        mismatched_keys = [
            key
            for key in sorted(set(cached) | set(expected))
            if cached.get(key) != expected.get(key)
        ]
        preview = ", ".join(
            f"{key}: cached={cached.get(key)!r}, expected={expected.get(key)!r}"
            for key in mismatched_keys[:6]
        )
        raise ValueError(f"score cache metadata mismatch ({preview})")
    scores = payload.get("scores")
    if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
        raise ValueError("score cache payload must contain a 2D tensor under 'scores'")
    registration_stats = payload.get("registration_stats", {})
    if not isinstance(registration_stats, dict):
        raise ValueError("score cache payload registration_stats must be a dict")
    return scores.float().cpu(), registration_stats


def parse_float_list(raw: str | None) -> List[float]:
    if raw is None or not str(raw).strip():
        return []
    parts = re.split(r"[,| ]+", str(raw).strip())
    return [float(part) for part in parts if part]


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_u8 = (pred > 0).astype(np.uint8)
    gt_u8 = (gt > 0).astype(np.uint8)
    if pred_u8.shape != gt_u8.shape:
        pred_u8 = cv2.resize(
            pred_u8,
            (gt_u8.shape[1], gt_u8.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    inter = np.logical_and(pred_u8, gt_u8).sum()
    union = np.logical_or(pred_u8, gt_u8).sum()
    return float(inter / union) if union > 0 else 0.0


def summarize_ious(ious: Sequence[float]) -> Dict[str, float | int]:
    if not ious:
        return {"miou": 0.0, "acc025": 0.0, "acc050": 0.0, "n": 0}
    arr = np.asarray(ious, dtype=np.float32)
    return {
        "miou": float(arr.mean()),
        "acc025": float((arr > 0.25).mean()),
        "acc050": float((arr > 0.50).mean()),
        "n": int(arr.size),
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    num_samples: int = 1000,
    seed: int = 13,
    alpha: float = 0.05,
) -> Dict[str, float | int]:
    """Return a deterministic bootstrap CI for a small query-level mean."""
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 1 or num_samples <= 0:
        mean = float(arr.mean())
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": int(arr.size)}
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(arr, size=(int(num_samples), int(arr.size)), replace=True)
    means = samples.mean(axis=1)
    low = float(np.quantile(means, float(alpha) * 0.5))
    high = float(np.quantile(means, 1.0 - float(alpha) * 0.5))
    return {"mean": float(arr.mean()), "ci_low": low, "ci_high": high, "n": int(arr.size)}


def mask_overlap_stats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float | int]:
    pred_u8 = (pred > 0).astype(np.uint8)
    gt_u8 = (gt > 0).astype(np.uint8)
    if pred_u8.shape != gt_u8.shape:
        pred_u8 = cv2.resize(
            pred_u8,
            (gt_u8.shape[1], gt_u8.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    inter = int(np.logical_and(pred_u8, gt_u8).sum())
    union = int(np.logical_or(pred_u8, gt_u8).sum())
    pred_pixels = int(pred_u8.sum())
    gt_pixels = int(gt_u8.sum())
    return {
        "iou": float(inter / union) if union > 0 else 0.0,
        "intersection_pixels": inter,
        "union_pixels": union,
        "pred_pixels": pred_pixels,
        "gt_pixels": gt_pixels,
        "overselect_ratio": float(pred_pixels / max(gt_pixels, 1)),
    }


def select_gaussians_from_scores(
    scores: torch.Tensor,
    spec: SelectionSpec,
    *,
    min_select: int = 1,
) -> torch.Tensor:
    """Return a float selection matrix [N, K] from score matrix [N, K]."""
    if scores.ndim != 2:
        raise ValueError(f"Expected score matrix [N,K], got {tuple(scores.shape)}")
    n_gaussians, n_queries = scores.shape
    if n_gaussians == 0 or n_queries == 0:
        return scores.new_zeros(scores.shape)

    selected = torch.zeros_like(scores, dtype=torch.float32)
    if spec.mode == "top_ratio":
        ratio = min(max(float(spec.value), 0.0), 1.0)
        k = max(int(round(n_gaussians * ratio)), int(min_select))
        k = min(k, n_gaussians)
        if k <= 0:
            return selected
        _, idx = torch.topk(scores.float(), k=k, dim=0, largest=True)
        selected.scatter_(0, idx, 1.0)
        return selected

    if spec.mode == "score_threshold":
        return (scores.float() > float(spec.value)).float()

    if spec.mode == "mean_std":
        mean = scores.float().mean(dim=0, keepdim=True)
        std = scores.float().std(dim=0, keepdim=True, unbiased=False)
        return (scores.float() > mean + float(spec.value) * std).float()

    raise ValueError(f"Unsupported selection mode: {spec.mode}")


def _ratio_to_count(n_items: int, ratio: float, min_select: int) -> int:
    ratio = min(max(float(ratio), 0.0), 1.0)
    count = max(int(round(n_items * ratio)), int(min_select))
    return min(max(count, 0), int(n_items))


def apply_selection_ratio_bounds(
    selected: torch.Tensor,
    scores: torch.Tensor,
    *,
    min_ratio: float = 0.0,
    max_ratio: float = 0.0,
    min_select: int = 1,
) -> torch.Tensor:
    """Apply GT-free per-query floor/cap ratios to a selection matrix.

    The floor unions in top-scoring primitives when a distribution threshold is
    too strict. The cap keeps only the strongest selected primitives when a
    threshold over-selects clutter.
    """
    if selected.ndim != 2 or scores.ndim != 2 or selected.shape != scores.shape:
        raise ValueError(
            f"Expected selected/scores [N,K] with same shape, got "
            f"{tuple(selected.shape)} and {tuple(scores.shape)}"
        )
    n_gaussians, n_queries = scores.shape
    if n_gaussians == 0 or n_queries == 0:
        return selected
    if min_ratio <= 0 and max_ratio <= 0:
        return selected

    result = selected.float().clone()
    scores_f = scores.float()
    if min_ratio > 0:
        floor_count = _ratio_to_count(n_gaussians, min_ratio, min_select)
        if floor_count > 0:
            _, floor_idx = torch.topk(scores_f, k=floor_count, dim=0, largest=True)
            result.scatter_(0, floor_idx, 1.0)

    if max_ratio > 0:
        cap_count = _ratio_to_count(n_gaussians, max_ratio, min_select)
        capped = torch.zeros_like(result)
        for query_idx in range(n_queries):
            idx = torch.nonzero(result[:, query_idx] > 0, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            if idx.numel() <= cap_count:
                capped[idx, query_idx] = 1.0
                continue
            local_scores = scores_f[idx, query_idx]
            _, order = torch.topk(local_scores, k=cap_count, largest=True)
            capped[idx[order], query_idx] = 1.0
        result = capped

    return result.to(device=selected.device, dtype=selected.dtype)


def select_gaussians_with_seed_expand_components(
    seed_selected: torch.Tensor,
    scores: torch.Tensor,
    xyz: torch.Tensor,
    *,
    support_ratio: float,
    resolution: int,
    keep_components: int,
    min_component_size: int,
    rank_by: str,
) -> torch.Tensor:
    """Expand high-confidence primitive seeds to connected support components.

    The seed set is normally a strict top-ratio selector. The support set is a
    wider GT-free top-ratio candidate pool. Only connected support components
    that contain at least one seed survive, which makes the selector closer to
    object-proposal selection without using LERF masks.
    """
    if rank_by not in {"mean_score", "score_sum", "size"}:
        raise ValueError(f"Unsupported component rank_by: {rank_by}")
    if seed_selected.ndim != 2 or scores.ndim != 2 or seed_selected.shape != scores.shape:
        raise ValueError(
            f"Expected seed_selected/scores [N,K] with same shape, got "
            f"{tuple(seed_selected.shape)} and {tuple(scores.shape)}"
        )
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Expected xyz [N,3] aligned with scores [N,K], got "
            f"{tuple(xyz.shape)} and {tuple(scores.shape)}"
        )
    if resolution <= 1 or support_ratio <= 0:
        return seed_selected

    try:
        from scipy import ndimage
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("seed-expand component selection requires scipy") from exc

    support = select_gaussians_from_scores(
        scores,
        SelectionSpec("top_ratio", float(support_ratio)),
        min_select=1,
    )
    support = torch.maximum(support.float(), seed_selected.float())

    device = seed_selected.device
    seed_cpu = seed_selected.detach().float().cpu()
    support_cpu = support.detach().float().cpu()
    scores_cpu = scores.detach().float().cpu()
    xyz_cpu = xyz.detach().float().cpu()
    lo = xyz_cpu.min(dim=0).values
    hi = xyz_cpu.max(dim=0).values
    extent = (hi - lo).clamp_min(1e-6)
    coords = ((xyz_cpu - lo) / extent * float(resolution)).floor().long()
    coords = coords.clamp_(0, resolution - 1).numpy()

    expanded = torch.zeros_like(seed_cpu)
    structure = ndimage.generate_binary_structure(3, 1)
    grid_shape = (int(resolution), int(resolution), int(resolution))
    keep_count = max(int(keep_components), 1)
    min_size = max(int(min_component_size), 1)

    for query_idx in range(seed_cpu.shape[1]):
        seed_idx = torch.nonzero(seed_cpu[:, query_idx] > 0, as_tuple=False).flatten()
        support_idx = torch.nonzero(support_cpu[:, query_idx] > 0, as_tuple=False).flatten()
        if seed_idx.numel() == 0 or support_idx.numel() == 0:
            continue

        support_np = support_idx.numpy()
        support_coords = coords[support_np]
        occupancy = np.zeros(grid_shape, dtype=bool)
        occupancy[support_coords[:, 0], support_coords[:, 1], support_coords[:, 2]] = True
        labels, num_components = ndimage.label(occupancy, structure=structure)
        if num_components <= 0:
            expanded[seed_idx, query_idx] = seed_cpu[seed_idx, query_idx]
            continue

        support_labels = labels[
            support_coords[:, 0],
            support_coords[:, 1],
            support_coords[:, 2],
        ].astype(np.int64)
        seed_np = seed_idx.numpy()
        seed_coords = coords[seed_np]
        seed_labels = labels[
            seed_coords[:, 0],
            seed_coords[:, 1],
            seed_coords[:, 2],
        ].astype(np.int64)
        seeded_labels = np.array(sorted({int(label) for label in seed_labels if label > 0}))
        if seeded_labels.size == 0:
            expanded[seed_idx, query_idx] = seed_cpu[seed_idx, query_idx]
            continue

        sizes = np.bincount(support_labels, minlength=num_components + 1).astype(np.float64)
        score_values = scores_cpu[support_idx, query_idx].numpy().astype(np.float64)
        score_sums = np.bincount(
            support_labels,
            weights=score_values,
            minlength=num_components + 1,
        )
        valid_labels = seeded_labels[sizes[seeded_labels] >= min_size]
        if valid_labels.size == 0:
            valid_labels = seeded_labels

        if rank_by == "size":
            ranks = sizes[valid_labels]
        elif rank_by == "mean_score":
            ranks = score_sums[valid_labels] / np.maximum(sizes[valid_labels], 1.0)
        else:
            ranks = score_sums[valid_labels]
        order = np.argsort(-ranks, kind="stable")
        kept_labels = set(int(label) for label in valid_labels[order[:keep_count]])
        keep_mask = np.array([int(label) in kept_labels for label in support_labels], dtype=bool)
        if keep_mask.any():
            expanded[support_idx[torch.from_numpy(keep_mask)], query_idx] = 1.0
        else:
            expanded[seed_idx, query_idx] = seed_cpu[seed_idx, query_idx]

    return expanded.to(device=device, dtype=seed_selected.dtype)


def refine_selection_by_voxel_components(
    selected: torch.Tensor,
    scores: torch.Tensor,
    xyz: torch.Tensor,
    *,
    mode: str,
    resolution: int,
    keep_components: int,
    min_component_size: int,
    rank_by: str,
) -> torch.Tensor:
    """Filter selected primitives by GT-free 3D connected components.

    This is an instance-consistency diagnostic inspired by Gaussian grouping
    methods: selection still comes from text scores, then disconnected tiny
    fragments can be removed before rendering masks.
    """
    if mode == "none":
        return selected
    if mode not in {"top_score_components", "largest_components"}:
        raise ValueError(f"Unsupported selection refinement mode: {mode}")
    if rank_by not in {"mean_score", "score_sum", "size"}:
        raise ValueError(f"Unsupported component rank_by: {rank_by}")
    if selected.ndim != 2 or scores.ndim != 2 or selected.shape != scores.shape:
        raise ValueError(
            f"Expected selected/scores [N,K] with same shape, got "
            f"{tuple(selected.shape)} and {tuple(scores.shape)}"
        )
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] != selected.shape[0]:
        raise ValueError(
            f"Expected xyz [N,3] aligned with selected [N,K], got "
            f"{tuple(xyz.shape)} and {tuple(selected.shape)}"
        )
    if resolution <= 1 or keep_components <= 0:
        return selected

    try:
        from scipy import ndimage
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("selection component refinement requires scipy") from exc

    device = selected.device
    selected_cpu = selected.detach().float().cpu()
    scores_cpu = scores.detach().float().cpu()
    xyz_cpu = xyz.detach().float().cpu()
    lo = xyz_cpu.min(dim=0).values
    hi = xyz_cpu.max(dim=0).values
    extent = (hi - lo).clamp_min(1e-6)
    coords = ((xyz_cpu - lo) / extent * float(resolution)).floor().long()
    coords = coords.clamp_(0, resolution - 1).numpy()

    refined = torch.zeros_like(selected_cpu)
    structure = ndimage.generate_binary_structure(3, 1)
    min_size = max(int(min_component_size), 1)
    keep_count = max(int(keep_components), 1)
    grid_shape = (int(resolution), int(resolution), int(resolution))

    for query_idx in range(selected_cpu.shape[1]):
        idx = torch.nonzero(selected_cpu[:, query_idx] > 0, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        idx_np = idx.numpy()
        query_coords = coords[idx_np]
        occupancy = np.zeros(grid_shape, dtype=bool)
        occupancy[query_coords[:, 0], query_coords[:, 1], query_coords[:, 2]] = True
        labels, num_components = ndimage.label(occupancy, structure=structure)
        if num_components <= 0:
            refined[idx, query_idx] = selected_cpu[idx, query_idx]
            continue

        component_labels = labels[
            query_coords[:, 0],
            query_coords[:, 1],
            query_coords[:, 2],
        ].astype(np.int64)
        sizes = np.bincount(component_labels, minlength=num_components + 1).astype(np.float64)
        score_values = scores_cpu[idx, query_idx].numpy().astype(np.float64)
        score_sums = np.bincount(
            component_labels,
            weights=score_values,
            minlength=num_components + 1,
        )
        valid_labels = np.arange(1, num_components + 1, dtype=np.int64)
        valid_labels = valid_labels[sizes[valid_labels] >= min_size]
        if valid_labels.size == 0:
            valid_labels = np.array([int(sizes[1:].argmax()) + 1], dtype=np.int64)

        if mode == "largest_components" or rank_by == "size":
            ranks = sizes[valid_labels]
        elif rank_by == "score_sum":
            ranks = score_sums[valid_labels]
        else:
            ranks = score_sums[valid_labels] / np.maximum(sizes[valid_labels], 1.0)
        order = np.argsort(-ranks, kind="stable")
        kept_labels = set(int(label) for label in valid_labels[order[:keep_count]])
        keep_mask = np.array([int(label) in kept_labels for label in component_labels], dtype=bool)
        if keep_mask.any():
            refined[idx[torch.from_numpy(keep_mask)], query_idx] = 1.0

    return refined.to(device=device, dtype=selected.dtype)


def _sorted_unique_ints(values: Iterable[int]) -> List[int]:
    return sorted({int(value) for value in values})


def _evenly_subsample_frames(frames: List[int], max_frames: int) -> List[int]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return list(frames)
    positions = np.linspace(0, len(frames) - 1, num=max_frames)
    selected: List[int] = []
    for pos in positions:
        frame = frames[int(round(float(pos)))]
        if frame not in selected:
            selected.append(frame)
    return selected


def select_registration_frame_ids(
    *,
    available_pose_ids: Iterable[int],
    annotated_frame_ids: Iterable[int],
    official_frame_ids: Iterable[int],
    train_frame_ids: Optional[Iterable[int]] = None,
    val_frame_ids: Optional[Iterable[int]] = None,
    mode: str,
    max_frames: int = 0,
) -> List[int]:
    """Select GT-free posed RGB/rendered views for primitive registration.

    ``official`` means the OpenGaussian official annotated subset when poses are
    available. If a scene has no matching official IDs, the function falls back
    to annotated IDs so the evaluator fails less often on local data variants.
    """
    available = set(_sorted_unique_ints(available_pose_ids))
    annotated = set(_sorted_unique_ints(annotated_frame_ids))
    official = set(_sorted_unique_ints(official_frame_ids))
    train = set(_sorted_unique_ints(train_frame_ids or []))
    val = set(_sorted_unique_ints(val_frame_ids or []))

    if mode == "official":
        frames = sorted(available & annotated & official)
        if not frames:
            frames = sorted(available & annotated)
    elif mode == "annotated":
        frames = sorted(available & annotated)
    elif mode == "all_poses":
        frames = sorted(available)
    elif mode == "train":
        frames = sorted(available & train)
    elif mode == "val":
        frames = sorted(available & val)
    else:
        raise ValueError(f"Unsupported registration frame mode: {mode}")

    return _evenly_subsample_frames(frames, int(max_frames))


def resolve_registration_split_frame_ids(config: object, split: str) -> Optional[List[int]]:
    """Return the frame ids used by train/val splits for LERF-style configs.

    If explicit split files are absent, mirror the generic split logic in
    ``train_feature_field.py``: sort feature files, apply a seeded random split,
    and return frame ids from the chosen side.
    """
    explicit = resolve_split_frame_ids(config, split)
    if explicit is not None:
        return [int(frame_id) for frame_id in explicit]

    feature_dir = resolve_split_feature_dir(config, "train")
    feature_paths = list_feature_paths(feature_dir)
    if not feature_paths:
        return None

    frame_ids = [extract_feature_frame_index(path) for path in feature_paths]
    generator = torch.Generator().manual_seed(int(getattr(config, "mixed_seed", 42)))
    perm = torch.randperm(len(frame_ids), generator=generator).tolist()
    train_cut = int(float(getattr(config, "mixed_train_ratio", 0.8)) * len(frame_ids))
    selected_positions = perm[:train_cut] if split == "train" else perm[train_cut:]
    return sorted(frame_ids[pos] for pos in selected_positions)


def choose_registration_refiner(
    refiner: Optional[torch.nn.Module],
    *,
    disable_registered_refiner: bool,
) -> Optional[torch.nn.Module]:
    """Optionally remove VFA from the VPR feature source for ablation."""
    return None if disable_registered_refiner else refiner


def score_text_aligned_embeddings(
    embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    canonical_embeddings: Optional[torch.Tensor] = None,
    scoring: str,
    softmax_temperature: float = 50.0,
) -> torch.Tensor:
    """Score normalized visual/text embeddings with direct-eval scoring modes."""
    if embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError(
            "Expected embeddings [N,D] and text_embeddings [K,D], got "
            f"{tuple(embeddings.shape)} and {tuple(text_embeddings.shape)}"
        )
    visual = F.normalize(embeddings.float(), dim=-1)
    text = F.normalize(text_embeddings.float(), dim=-1).to(visual.device)

    if scoring == "softmax_scene":
        logits = visual @ text.T
        return torch.softmax(logits * float(softmax_temperature), dim=-1)

    if scoring == "cosine":
        return visual @ text.T

    if scoring == "relevancy":
        if canonical_embeddings is None or canonical_embeddings.numel() == 0:
            raise ValueError("scoring='relevancy' requires canonical embeddings")
        canonical = F.normalize(canonical_embeddings.float(), dim=-1).to(visual.device)
        sim = visual @ text.T
        canon = visual @ canonical.T
        canon_max = canon.max(dim=1, keepdim=True).values
        sim_scaled = sim * float(softmax_temperature)
        canon_scaled = canon_max.expand_as(sim) * float(softmax_temperature)
        max_val = torch.maximum(sim_scaled, canon_scaled)
        return torch.exp(sim_scaled - max_val) / (
            torch.exp(sim_scaled - max_val)
            + torch.exp(canon_scaled - max_val)
            + 1e-8
        )

    raise ValueError(f"Unsupported scoring mode: {scoring}")


def merge_registered_scores(
    registered_embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    fallback_scores: Optional[torch.Tensor] = None,
    canonical_embeddings: Optional[torch.Tensor] = None,
    scoring: str,
    softmax_temperature: float = 50.0,
) -> torch.Tensor:
    """Score registered primitive embeddings and fill unregistered rows."""
    scores = score_text_aligned_embeddings(
        registered_embeddings,
        text_embeddings,
        canonical_embeddings=canonical_embeddings,
        scoring=scoring,
        softmax_temperature=softmax_temperature,
    )
    valid = valid_mask.to(device=scores.device, dtype=torch.bool)
    if valid.shape != (scores.shape[0],):
        raise ValueError(
            f"Expected valid_mask [{scores.shape[0]}], got {tuple(valid.shape)}"
        )
    if valid.all():
        return scores

    merged = scores.clone()
    if fallback_scores is None:
        merged[~valid] = -1.0e4
    else:
        fallback = fallback_scores.to(device=scores.device, dtype=scores.dtype)
        if fallback.shape != scores.shape:
            raise ValueError(
                f"Expected fallback_scores {tuple(scores.shape)}, got {tuple(fallback.shape)}"
            )
        merged[~valid] = fallback[~valid]
    return merged


def apply_registration_confidence(
    scores: torch.Tensor,
    registered_counts: torch.Tensor,
    *,
    blend: float = 0.0,
    mode: str = "log",
) -> torch.Tensor:
    """Downweight primitive scores with weak multi-view registration support.

    The calibration is GT-free: it only uses how many valid rendered-view
    samples contributed to each Gaussian.  ``blend=0`` preserves the original
    scores, while larger values keep all scores but reduce low-support
    primitives by a bounded row-wise confidence scale.
    """
    blend_f = min(max(float(blend), 0.0), 1.0)
    if blend_f <= 0.0:
        return scores
    if scores.ndim != 2:
        raise ValueError(f"Expected scores [N,K], got {tuple(scores.shape)}")
    if registered_counts.shape != (scores.shape[0],):
        raise ValueError(
            f"Expected registered_counts [{scores.shape[0]}], got "
            f"{tuple(registered_counts.shape)}"
        )
    counts = registered_counts.to(device=scores.device, dtype=scores.dtype).clamp_min(0.0)
    max_count = counts.max().clamp_min(1.0)
    if mode == "log":
        confidence = torch.log1p(counts) / torch.log1p(max_count)
    elif mode == "linear":
        confidence = counts / max_count
    else:
        raise ValueError(f"Unsupported registration confidence mode: {mode}")
    scale = (1.0 - blend_f) + blend_f * confidence.clamp(0.0, 1.0)
    return scores * scale.unsqueeze(1)


def sample_registration_view_weights(
    points_xyz: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    depth_map: Optional[torch.Tensor] = None,
    alpha_map: Optional[torch.Tensor] = None,
    depth_tolerance: float = 0.08,
    relative_depth_tolerance: float = 0.02,
    alpha_threshold: float = 0.0,
    mode: str = "uniform",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GT-free visibility weights for one registered-view sample.

    ``uniform`` matches the original VPR behavior. ``alpha`` weights registered
    samples by rendered opacity. ``alpha_depth`` additionally downweights
    samples farther from the rendered depth surface, echoing dominant-ray
    registration without requiring per-Gaussian rasterization contributions.
    """
    if mode not in {"uniform", "alpha", "alpha_depth"}:
        raise ValueError(f"Unsupported registration weight mode: {mode}")
    if points_xyz.dim() != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected points_xyz [N,3], got {tuple(points_xyz.shape)}")
    if pose_w2c.dim() != 3 or pose_w2c.shape[-2:] != (4, 4):
        raise ValueError(f"Expected pose_w2c [B,4,4], got {tuple(pose_w2c.shape)}")
    if pose_w2c.shape[0] != 1:
        raise ValueError("sample_registration_view_weights expects a single view")

    device = points_xyz.device
    n_points = int(points_xyz.shape[0])
    if n_points == 0:
        return (
            torch.empty(0, dtype=torch.bool, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

    points = points_xyz.to(device=device, dtype=torch.float32)
    pose = pose_w2c.to(device=device, dtype=torch.float32)
    intrinsics = K.to(device=device, dtype=torch.float32)
    ones = torch.ones(n_points, 1, device=device, dtype=torch.float32)
    points_h = torch.cat([points, ones], dim=1)
    cam = torch.einsum("bij,nj->bni", pose, points_h)
    z = cam[0, :, 2]
    z_safe = z.clamp_min(1e-6)
    u = intrinsics[0, 0] * (cam[0, :, 0] / z_safe) + intrinsics[0, 2]
    v = intrinsics[1, 1] * (cam[0, :, 1] / z_safe) + intrinsics[1, 2]
    height = int(image_height)
    width = int(image_width)
    valid = (
        (z > 1e-6)
        & (u >= 0.0)
        & (u <= float(width - 1))
        & (v >= 0.0)
        & (v <= float(height - 1))
    )
    grid_x = 2.0 * u / float(max(width - 1, 1)) - 1.0 if width > 1 else torch.zeros_like(u)
    grid_y = 2.0 * v / float(max(height - 1, 1)) - 1.0 if height > 1 else torch.zeros_like(v)
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, n_points, 1, 2)
    weights = torch.ones(n_points, dtype=torch.float32, device=device)

    sampled_depth = None
    tolerance = None
    if depth_map is not None:
        depth = depth_map.to(device=device, dtype=torch.float32)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        if depth.dim() == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if depth.shape != (1, height, width):
            depth = F.interpolate(
                depth[:, None],
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )[:, 0]
        sampled_depth = F.grid_sample(
            depth[:, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0, :, 0]
        tolerance = torch.maximum(
            torch.full_like(z, float(depth_tolerance)),
            z.abs() * float(relative_depth_tolerance),
        )
        valid = valid & (sampled_depth > 0.0) & ((sampled_depth - z).abs() <= tolerance)

    sampled_alpha = None
    if alpha_map is not None:
        alpha = alpha_map.to(device=device, dtype=torch.float32)
        if alpha.dim() == 2:
            alpha = alpha.unsqueeze(0)
        if alpha.dim() == 4 and alpha.shape[1] == 1:
            alpha = alpha[:, 0]
        if alpha.shape != (1, height, width):
            alpha = F.interpolate(
                alpha[:, None],
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )[:, 0]
        sampled_alpha = F.grid_sample(
            alpha[:, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0, :, 0].clamp_min(0.0)
        if alpha_threshold > 0:
            valid = valid & (sampled_alpha >= float(alpha_threshold))
        if mode in {"alpha", "alpha_depth"}:
            weights = weights * sampled_alpha.clamp_min(1e-6)

    if mode == "alpha_depth" and sampled_depth is not None and tolerance is not None:
        depth_error = (sampled_depth - z).abs()
        weights = weights * torch.exp(-depth_error / tolerance.clamp_min(1e-6))

    weights = torch.where(valid, weights, torch.zeros_like(weights))
    return valid, weights


def aggregate_scores_by_voxel(
    scores: torch.Tensor,
    xyz: torch.Tensor,
    *,
    mode: str,
    resolution: int,
    blend: float,
) -> torch.Tensor:
    """Blend per-Gaussian scores with local voxel spatial context.

    This is a GT-free primitive aggregation diagnostic.  It keeps the query in
    3D, but tests whether isolated Gaussian-center scores are too fragmented for
    object-level selection.
    """
    if mode == "none" or resolution <= 1 or blend <= 0:
        return scores
    if scores.ndim != 2:
        raise ValueError(f"Expected score matrix [N,K], got {tuple(scores.shape)}")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Expected xyz [N,3] aligned with scores [N,K], got {tuple(xyz.shape)} "
            f"and {tuple(scores.shape)}"
        )
    if mode not in {"voxel_mean", "voxel_max", "voxel_max_dilate"}:
        raise ValueError(f"Unsupported score aggregation mode: {mode}")

    scores_f = scores.float()
    xyz_f = xyz.float()
    lo = xyz_f.min(dim=0).values
    hi = xyz_f.max(dim=0).values
    extent = (hi - lo).clamp_min(1e-6)
    coords = ((xyz_f - lo) / extent * float(resolution)).floor().long()
    coords = coords.clamp_(0, resolution - 1)
    linear = (
        coords[:, 0] * resolution * resolution
        + coords[:, 1] * resolution
        + coords[:, 2]
    )
    unique, inverse = torch.unique(linear, sorted=False, return_inverse=True)
    num_voxels = int(unique.numel())
    expanded = inverse.view(-1, 1).expand(-1, scores_f.shape[1])

    if mode == "voxel_mean":
        voxel_scores = torch.zeros(
            num_voxels,
            scores_f.shape[1],
            dtype=scores_f.dtype,
            device=scores_f.device,
        )
        voxel_scores.scatter_add_(0, expanded, scores_f)
        counts = torch.bincount(inverse, minlength=num_voxels).to(
            dtype=scores_f.dtype,
            device=scores_f.device,
        )
        voxel_scores = voxel_scores / counts.clamp_min(1.0).view(-1, 1)
    else:
        voxel_scores = torch.full(
            (num_voxels, scores_f.shape[1]),
            -float("inf"),
            dtype=scores_f.dtype,
            device=scores_f.device,
        )
        voxel_scores.scatter_reduce_(0, expanded, scores_f, reduce="amax", include_self=True)
        voxel_scores = torch.where(
            torch.isfinite(voxel_scores),
            voxel_scores,
            torch.zeros_like(voxel_scores),
        )
        if mode == "voxel_max_dilate":
            # Sparse one-hop voxel dilation: each occupied voxel receives the
            # strongest score from its 26-neighborhood, avoiding dense component
            # selection while reducing primitive-level fragmentation.
            num_dense = int(resolution) ** 3
            dense_scores = torch.full(
                (num_dense, scores_f.shape[1]),
                -float("inf"),
                dtype=scores_f.dtype,
                device=scores_f.device,
            )
            dense_scores[unique] = voxel_scores
            unique_coords = torch.stack(
                (
                    unique // (resolution * resolution),
                    (unique // resolution) % resolution,
                    unique % resolution,
                ),
                dim=1,
            )
            dilated_scores = voxel_scores.clone()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor_coords = unique_coords + unique_coords.new_tensor([dx, dy, dz])
                        valid = (
                            (neighbor_coords >= 0).all(dim=1)
                            & (neighbor_coords < resolution).all(dim=1)
                        )
                        if not bool(valid.any()):
                            continue
                        neighbor_linear = (
                            neighbor_coords[valid, 0] * resolution * resolution
                            + neighbor_coords[valid, 1] * resolution
                            + neighbor_coords[valid, 2]
                        )
                        neighbor_scores = dense_scores[neighbor_linear]
                        dilated_scores[valid] = torch.maximum(
                            dilated_scores[valid],
                            neighbor_scores,
                        )
            voxel_scores = torch.where(
                torch.isfinite(dilated_scores),
                dilated_scores,
                voxel_scores,
            )

    blend = min(max(float(blend), 0.0), 1.0)
    return scores_f * (1.0 - blend) + voxel_scores[inverse] * blend


def load_summary_head(weights_path: str, device: torch.device) -> SigLIP2SummaryHead:
    path = Path(weights_path)
    if path.exists():
        head = SigLIP2SummaryHead.from_extracted_weights(str(path))
        print(f"Loaded SigLIP2 summary head from {path}")
    else:
        head = SigLIP2SummaryHead.from_radio_checkpoint(
            "/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar"
        )
        print("Loaded SigLIP2 summary head from RADIO checkpoint")
    head = head.to(device)
    head = head.half() if device.type == "cuda" else head.float()
    return head.eval()


def build_mask_renderer(
    config: object,
    *,
    height: int,
    width: int,
    device: torch.device,
) -> FeatureFieldRenderer:
    image_width = float(getattr(config, "image_width", width) or width)
    image_height = float(getattr(config, "image_height", height) or height)
    fx = float(getattr(config, "fx", width * 0.8)) * width / image_width
    fy = float(getattr(config, "fy", height * 0.8)) * height / image_height
    cx = float(getattr(config, "cx", (image_width - 1.0) * 0.5)) * width / image_width
    cy = float(getattr(config, "cy", (image_height - 1.0) * 0.5)) * height / image_height
    use_2dgs = resolve_use_2dgs(config, getattr(config, "ply_path", ""))
    renderer = FeatureFieldRenderer(
        image_height=height,
        image_width=width,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        max_channels_per_chunk=getattr(config, "max_channels_per_chunk", 32),
        use_2dgs=use_2dgs,
        background_color=0.0,
    )
    return renderer.to(device).eval()


@torch.no_grad()
def compute_gaussian_text_scores(
    model: torch.nn.Module,
    codec: torch.nn.Module,
    summary_head: torch.nn.Module,
    text_embeddings: torch.Tensor,
    canonical_embeddings: Optional[torch.Tensor] = None,
    *,
    is_hybrid: bool,
    direct_readout_mode: str,
    direct_readout_k: int,
    direct_readout_candidate_k: int,
    compact_feature_key: str,
    scoring: str,
    softmax_temperature: float,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Decode Gaussian-center features and score them against scene text queries."""
    n_gaussians = int(model.num_gaussians)
    if n_gaussians <= 0:
        raise RuntimeError("Model has no Gaussians")

    all_scores: List[torch.Tensor] = []
    text = F.normalize(text_embeddings.float(), dim=-1).to(device)
    text_for_compute = text.half() if device.type == "cuda" else text.float()
    canonical = None
    if canonical_embeddings is not None:
        canonical = F.normalize(canonical_embeddings.float(), dim=-1).to(device)
    knn_indices_np: Optional[np.ndarray] = None
    knn_dist_np: Optional[np.ndarray] = None
    knn_latent: Optional[torch.Tensor] = None
    knn_opacity: Optional[torch.Tensor] = None
    if direct_readout_mode == "knn":
        if not is_hybrid or not hasattr(model, "get_latent") or not hasattr(model, "_decode_point_features"):
            raise ValueError("direct_readout_mode=knn requires a HybridFeatureGaussian model")
        try:
            from scipy.spatial import cKDTree
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("direct_readout_mode=knn requires scipy.spatial.cKDTree") from exc
        query_k = max(int(direct_readout_k), 1)
        candidate_k = max(int(direct_readout_candidate_k), query_k)
        xyz_np = model.get_xyz().detach().cpu().float().numpy()
        tree = cKDTree(xyz_np)
        knn_dist_np, knn_indices_np = tree.query(
            xyz_np,
            k=candidate_k,
            workers=-1,
        )
        if candidate_k == 1:
            knn_dist_np = knn_dist_np[:, None]
            knn_indices_np = knn_indices_np[:, None]
        knn_latent = model.get_latent().to(device=device, dtype=torch.float32)
        knn_opacity = model.get_opacity().to(device=device, dtype=torch.float32).squeeze(-1)

    for start in tqdm(range(0, n_gaussians, chunk_size), desc="  decode/score", leave=False):
        end = min(start + chunk_size, n_gaussians)
        idx = torch.arange(start, end, device=device, dtype=torch.long)
        if direct_readout_mode == "knn":
            assert knn_indices_np is not None and knn_dist_np is not None
            assert knn_latent is not None and knn_opacity is not None
            neigh_idx = torch.from_numpy(knn_indices_np[start:end]).to(device=device, dtype=torch.long)
            neigh_dist = torch.from_numpy(knn_dist_np[start:end]).to(device=device, dtype=torch.float32)
            weights = torch.exp(
                -0.5
                * (
                    neigh_dist
                    / neigh_dist[:, -1:].clamp_min(1e-6)
                )
                ** 2
            )
            weights = weights * knn_opacity[neigh_idx].clamp_min(1e-6)
            if neigh_idx.shape[1] > int(direct_readout_k):
                top_weights, order = torch.topk(
                    weights,
                    k=max(int(direct_readout_k), 1),
                    dim=1,
                    largest=True,
                )
                neigh_idx = neigh_idx.gather(1, order)
                weights = top_weights
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            latent_points = (knn_latent[neigh_idx] * weights.unsqueeze(-1)).sum(dim=1)
            points = model.get_xyz()[idx].to(device=device, dtype=torch.float32)
            normalized_points = model.normalize_world_positions(points)
            compact_result = model._decode_point_features(
                latent_points.to(dtype=model.get_latent().dtype),
                normalized_points,
                return_aux=compact_feature_key != "features",
            )
        elif is_hybrid and hasattr(model, "query_gaussian_points"):
            compact_result = model.query_gaussian_points(
                idx,
                return_aux=compact_feature_key != "features",
            )
        else:
            if compact_feature_key != "features":
                raise ValueError(
                    f"compact_feature_key={compact_feature_key!r} requires a hybrid model"
                )
            compact = model.get_features()[idx]
            compact_result = compact
        if isinstance(compact_result, dict):
            if compact_feature_key not in compact_result:
                raise KeyError(
                    f"Compact feature key '{compact_feature_key}' not available; "
                    f"available keys: {sorted(compact_result.keys())}"
                )
            compact = compact_result[compact_feature_key]
        else:
            compact = compact_result
        radio = codec.decode_points(compact.float())
        radio_tokens = radio.unsqueeze(0)
        head_param = next(summary_head.parameters(), None)
        if head_param is not None:
            radio_tokens = radio_tokens.to(dtype=head_param.dtype)
        siglip = summary_head(radio_tokens).squeeze(0)
        siglip = F.normalize(siglip.float(), dim=-1)

        if scoring == "softmax_scene":
            logits = siglip @ text.float().T
            scores = torch.softmax(logits * float(softmax_temperature), dim=-1)
        elif scoring == "cosine":
            scores = siglip @ text.float().T
        elif scoring == "relevancy":
            if canonical is None or canonical.numel() == 0:
                raise ValueError("scoring='relevancy' requires canonical embeddings")
            sim = siglip @ text.float().T
            canon = siglip @ canonical.float().T
            canon_max = canon.max(dim=1, keepdim=True).values
            sim_scaled = sim * float(softmax_temperature)
            canon_scaled = canon_max.expand_as(sim) * float(softmax_temperature)
            max_val = torch.maximum(sim_scaled, canon_scaled)
            scores = torch.exp(sim_scaled - max_val) / (
                torch.exp(sim_scaled - max_val)
                + torch.exp(canon_scaled - max_val)
                + 1e-8
            )
        else:
            raise ValueError(f"Unsupported scoring mode: {scoring}")
        all_scores.append(scores.cpu())

        del compact, radio, radio_tokens, siglip, scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(all_scores, dim=0)


def _load_lerf_rgb_tensor(
    scene: str,
    frame_id: int,
    config: object,
    device: torch.device,
) -> Optional[torch.Tensor]:
    image_bgr = load_lerf_rgb_frame(scene, frame_id, getattr(config, "scene_root", ""))
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1)
    return tensor.float().div(255.0).unsqueeze(0).to(device)


@torch.no_grad()
def compute_registered_view_text_scores(
    *,
    scene: str,
    model: torch.nn.Module,
    codec: torch.nn.Module,
    renderer: FeatureFieldRenderer,
    sharpener: torch.nn.Module,
    refiner: Optional[torch.nn.Module],
    config: object,
    is_hybrid: bool,
    dataset: LERFDataset,
    frame_annotations: Dict[int, List[dict]],
    summary_head: torch.nn.Module,
    text_embeddings: torch.Tensor,
    canonical_embeddings: Optional[torch.Tensor],
    scoring: str,
    softmax_temperature: float,
    registration_frame_mode: str,
    registration_max_frames: int,
    registration_chunk_size: int,
    registration_depth_tolerance: float,
    registration_relative_depth_tolerance: float,
    registration_alpha_threshold: float,
    registration_weight_mode: str,
    registration_confidence_blend: float,
    registration_confidence_mode: str,
    fallback_scores: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Register rendered SigLIP2 features back to Gaussian centers and score text.

    This is the no-training VPR primitive readout: query still happens in 3D
    on Gaussian primitives, while rendered views provide the language-aligned
    registration signal without using LERF labels or masks for training/scoring.
    """
    frame_ids = select_registration_frame_ids(
        available_pose_ids=dataset.pose_by_frame_idx.keys(),
        annotated_frame_ids=frame_annotations.keys(),
        official_frame_ids=OPEN_GAUSSIAN_LERF_FRAMES.get(scene, []),
        train_frame_ids=resolve_registration_split_frame_ids(config, "train"),
        val_frame_ids=resolve_registration_split_frame_ids(config, "val"),
        mode=registration_frame_mode,
        max_frames=registration_max_frames,
    )
    if not frame_ids:
        raise RuntimeError(
            f"No registration frames selected for {scene} with mode={registration_frame_mode}"
        )

    xyz_cpu = model.get_xyz().detach().cpu().float()
    n_gaussians = int(xyz_cpu.shape[0])
    text = F.normalize(text_embeddings.float(), dim=-1).to(device)
    canonical = (
        F.normalize(canonical_embeddings.float(), dim=-1).to(device)
        if canonical_embeddings is not None
        else None
    )
    embedding_dim = int(text.shape[1])
    registered_sum = torch.zeros(n_gaussians, embedding_dim, dtype=torch.float32)
    registered_counts = torch.zeros(n_gaussians, dtype=torch.float32)
    chunk_size = max(int(registration_chunk_size), 1)

    for frame_id in tqdm(frame_ids, desc="  register rendered views", leave=False):
        pose_w2c = dataset.pose_by_frame_idx.get(frame_id)
        if pose_w2c is None:
            continue
        viewmat = torch.from_numpy(pose_w2c.copy()).float().to(device).unsqueeze(0)
        rgb_tensor = None
        if getattr(config, "refiner_rgb_guide", False):
            rgb_tensor = _load_lerf_rgb_tensor(scene, frame_id, config, device)

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
            rgb_image=rgb_tensor,
        )
        head_param = next(summary_head.parameters(), None)
        if head_param is not None:
            feat_1280 = feat_1280.to(dtype=head_param.dtype)
        siglip_feat = project_to_siglip2(feat_1280, summary_head).float()
        siglip_feat = F.normalize(siglip_feat, dim=1)

        aux = renderer.render_features(model, viewmat.squeeze(0))
        depth_map = aux["depth_map"].detach().float().unsqueeze(0)
        alpha_map = aux["alpha_map"].detach().float().unsqueeze(0)

        for start in range(0, n_gaussians, chunk_size):
            end = min(start + chunk_size, n_gaussians)
            points = xyz_cpu[start:end].to(device=device, dtype=torch.float32)
            targets, valid, counts = sample_multiview_radio_targets(
                points,
                siglip_feat,
                viewmat,
                renderer.K,
                depth_map=depth_map,
                alpha_map=alpha_map,
                depth_tolerance=registration_depth_tolerance,
                relative_depth_tolerance=registration_relative_depth_tolerance,
                alpha_threshold=registration_alpha_threshold,
                normalize_sampled_features=True,
            )
            if registration_weight_mode == "uniform":
                weights = counts.to(device=device, dtype=torch.float32)
            else:
                weight_valid, weights = sample_registration_view_weights(
                    points,
                    viewmat,
                    renderer.K,
                    image_height=int(siglip_feat.shape[-2]),
                    image_width=int(siglip_feat.shape[-1]),
                    depth_map=depth_map,
                    alpha_map=alpha_map,
                    depth_tolerance=registration_depth_tolerance,
                    relative_depth_tolerance=registration_relative_depth_tolerance,
                    alpha_threshold=registration_alpha_threshold,
                    mode=registration_weight_mode,
                )
                valid = valid & weight_valid
            valid_cpu = valid.detach().cpu()
            if valid_cpu.any():
                counts_valid = weights[valid].detach().float().cpu()
                registered_sum[start:end][valid_cpu] += (
                    targets[valid].detach().float().cpu()
                    * counts_valid.unsqueeze(1)
                )
                registered_counts[start:end][valid_cpu] += counts_valid

        del feat_1280, siglip_feat, aux, depth_map, alpha_map
        if device.type == "cuda":
            torch.cuda.empty_cache()

    valid_all = registered_counts > 0
    all_scores: List[torch.Tensor] = []
    for start in tqdm(range(0, n_gaussians, chunk_size), desc="  score registered", leave=False):
        end = min(start + chunk_size, n_gaussians)
        counts = registered_counts[start:end].clamp_min(1.0).unsqueeze(1)
        registered = registered_sum[start:end] / counts
        registered = F.normalize(registered, dim=-1).to(device)
        fallback_chunk = None
        if fallback_scores is not None:
            fallback_chunk = fallback_scores[start:end]
        scores = merge_registered_scores(
            registered,
            valid_all[start:end],
            text,
            fallback_scores=fallback_chunk,
            canonical_embeddings=canonical,
            scoring=scoring,
            softmax_temperature=softmax_temperature,
        )
        scores = apply_registration_confidence(
            scores,
            registered_counts[start:end],
            blend=registration_confidence_blend,
            mode=registration_confidence_mode,
        )
        all_scores.append(scores.detach().cpu())
        del registered, scores

    valid_count = int(valid_all.sum().item())
    stats: Dict[str, object] = {
        "frame_mode": registration_frame_mode,
        "frame_ids": frame_ids,
        "num_frames": len(frame_ids),
        "registered_gaussians": valid_count,
        "total_gaussians": n_gaussians,
        "registered_fraction": float(valid_count / max(n_gaussians, 1)),
        "mean_valid_views": float(registered_counts[valid_all].mean().item()) if valid_count else 0.0,
        "max_valid_views": float(registered_counts.max().item()) if n_gaussians else 0.0,
        "fallback": "direct" if fallback_scores is not None else "low",
        "depth_tolerance": float(registration_depth_tolerance),
        "relative_depth_tolerance": float(registration_relative_depth_tolerance),
        "alpha_threshold": float(registration_alpha_threshold),
        "weight_mode": registration_weight_mode,
        "confidence_blend": float(registration_confidence_blend),
        "confidence_mode": registration_confidence_mode,
    }
    return torch.cat(all_scores, dim=0), stats


def build_lerf_dataset_for_scene(
    scene: str,
    config: object,
    label_dir: str,
    *,
    feature_height: int,
    feature_width: int,
) -> LERFDataset:
    scene_root = resolve_lerf_scene_root(scene, getattr(config, "scene_root", ""))
    feat_dir = Path(getattr(config, "feature_dir", "") or "") if getattr(config, "feature_dir", "") else Path()
    if not feat_dir.exists():
        feat_dir = Path(DEFAULT_GT_FEATURE_ROOT) / scene
    return LERFDataset(
        scene_root=str(scene_root),
        feature_dir=str(feat_dir),
        annotation_dir=str(Path(label_dir) / scene),
        feature_height=feature_height,
        feature_width=feature_width,
    )


def save_pred_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def refine_mask_with_rgb_edges(
    rgb_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    iterations: int = 1,
    dilate_pixels: int = 5,
    erode_pixels: int = 2,
) -> np.ndarray:
    """Snap a predicted binary mask to local RGB edges without using GT masks.

    This is an optional 2D readout refinement for visualization/ablation.  The
    initializer is built only from the rendered prediction: eroded prediction is
    sure foreground, the dilated support is probable foreground, and everything
    outside the support is background.
    """
    pred = np.asarray(mask).astype(bool)
    if pred.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {pred.shape}")
    if rgb_bgr.shape[:2] != pred.shape:
        raise ValueError(f"RGB/mask shape mismatch: {rgb_bgr.shape[:2]} vs {pred.shape}")
    if not pred.any() or pred.all():
        return pred.copy()
    kernel_size = max(1, int(dilate_pixels) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    pred_u8 = pred.astype(np.uint8)
    support = cv2.dilate(pred_u8, kernel, iterations=1).astype(bool)
    if int(erode_pixels) > 0:
        erode_size = max(1, int(erode_pixels) * 2 + 1)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        sure_fg = cv2.erode(pred_u8, erode_kernel, iterations=1).astype(bool)
    else:
        sure_fg = pred
    if not sure_fg.any():
        sure_fg = pred

    init = np.full(pred.shape, cv2.GC_BGD, dtype=np.uint8)
    init[support] = cv2.GC_PR_FGD
    init[pred] = cv2.GC_PR_FGD
    init[sure_fg] = cv2.GC_FGD
    init[~support] = cv2.GC_BGD
    if not np.any(init == cv2.GC_BGD) or not np.any((init == cv2.GC_FGD) | (init == cv2.GC_PR_FGD)):
        return pred.copy()

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            rgb_bgr,
            init,
            None,
            bgd_model,
            fgd_model,
            max(1, int(iterations)),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return pred.copy()
    refined = (init == cv2.GC_FGD) | (init == cv2.GC_PR_FGD)
    return refined & support


def evaluate_selection_spec(
    *,
    scene: str,
    scene_categories: List[str],
    frame_annotations: Dict[int, List[dict]],
    img_h: int,
    img_w: int,
    model: torch.nn.Module,
    renderer: FeatureFieldRenderer,
    dataset: LERFDataset,
    scores: torch.Tensor,
    spec: SelectionSpec,
    selection_refinement: str,
    selection_min_ratio: float,
    selection_max_ratio: float,
    component_support_ratio: float,
    component_resolution: int,
    component_keep: int,
    component_min_size: int,
    component_rank_by: str,
    silhouette_threshold: float,
    mask_refinement: str,
    mask_refinement_iters: int,
    mask_refinement_dilate: int,
    mask_refinement_erode: int,
    min_select: int,
    output_dir: Path,
    save_masks: bool,
    device: torch.device,
) -> Dict:
    selected = select_gaussians_from_scores(
        scores,
        spec,
        min_select=min_select,
    )
    selected = apply_selection_ratio_bounds(
        selected,
        scores,
        min_ratio=selection_min_ratio,
        max_ratio=selection_max_ratio,
        min_select=min_select,
    )
    if selection_refinement == "seed_expand_components":
        selected = select_gaussians_with_seed_expand_components(
            selected,
            scores,
            model.get_xyz().detach().cpu(),
            support_ratio=component_support_ratio,
            resolution=component_resolution,
            keep_components=component_keep,
            min_component_size=component_min_size,
            rank_by=component_rank_by,
        )
    elif selection_refinement != "none":
        selected = refine_selection_by_voxel_components(
            selected,
            scores,
            model.get_xyz().detach().cpu(),
            mode=selection_refinement,
            resolution=component_resolution,
            keep_components=component_keep,
            min_component_size=component_min_size,
            rank_by=component_rank_by,
        )
    selected = selected.to(device=device, dtype=torch.float32)
    proxy = GaussianSelectionProxy(model, selected)

    ious: List[float] = []
    per_category: Dict[str, List[float]] = {cat: [] for cat in scene_categories}
    per_frame: Dict[str, Dict[str, float]] = {}
    query_details: List[Dict[str, float | int | str]] = []

    for frame_id, frame_objects in tqdm(
        sorted(frame_annotations.items()),
        desc=f"  render/eval {scene} {spec.tag}",
        leave=False,
    ):
        pose_w2c = dataset.pose_by_frame_idx.get(frame_id)
        if pose_w2c is None:
            logger.warning("No pose for %s frame_%05d; skipping", scene, frame_id)
            continue
        viewmat = torch.from_numpy(pose_w2c.copy()).float().to(device)
        with torch.no_grad():
            rendered = renderer.render_features(proxy, viewmat)
            silhouette = rendered["feature_map"].detach().float().cpu().numpy()
        gt_masks = build_gt_masks(frame_objects, scene_categories, img_h, img_w)
        rgb_for_refinement = None
        if mask_refinement != "none":
            rgb_for_refinement = load_lerf_rgb_frame(scene, frame_id, getattr(dataset, "scene_root", ""))

        active_cats = sorted({obj["category"] for obj in frame_objects})
        frame_scores: Dict[str, float] = {}
        for cat in active_cats:
            if cat not in per_category:
                continue
            cat_idx = scene_categories.index(cat)
            pred = silhouette[cat_idx] > float(silhouette_threshold)
            if mask_refinement == "rgb_grabcut" and rgb_for_refinement is not None:
                pred = refine_mask_with_rgb_edges(
                    rgb_for_refinement,
                    pred,
                    iterations=mask_refinement_iters,
                    dilate_pixels=mask_refinement_dilate,
                    erode_pixels=mask_refinement_erode,
                )
            gt = gt_masks[cat]
            if gt.sum() == 0:
                continue
            overlap = mask_overlap_stats(pred, gt)
            iou = float(overlap["iou"])
            ious.append(iou)
            per_category[cat].append(iou)
            frame_scores[cat] = iou
            query_details.append(
                {
                    "frame": f"frame_{frame_id:05d}",
                    "frame_id": int(frame_id),
                    "category": cat,
                    "iou": iou,
                    "pred_pixels": int(overlap["pred_pixels"]),
                    "gt_pixels": int(overlap["gt_pixels"]),
                    "intersection_pixels": int(overlap["intersection_pixels"]),
                    "union_pixels": int(overlap["union_pixels"]),
                    "overselect_ratio": float(overlap["overselect_ratio"]),
                    "selected_gaussians": int(selected[:, cat_idx].sum().item()),
                }
            )
            if save_masks:
                mask_path = (
                    output_dir
                    / "pred_masks"
                    / spec.tag
                    / scene
                    / f"frame_{frame_id:05d}_{cat.replace('/', '_')}.png"
                )
                save_pred_mask(mask_path, pred)
        per_frame[f"frame_{frame_id:05d}"] = frame_scores

    per_cat_summary = {
        cat: {
            **summarize_ious(vals),
            "selected_gaussians": int(selected[:, ci].sum().item()),
        }
        for ci, (cat, vals) in enumerate(per_category.items())
    }
    summary = summarize_ious(ious)
    summary.update(
        {
            "selection_mode": spec.mode,
            "selection_value": spec.value,
            "selection_tag": spec.tag,
            "selection_refinement": selection_refinement,
            "selection_min_ratio": selection_min_ratio,
            "selection_max_ratio": selection_max_ratio,
            "component_support_ratio": component_support_ratio,
            "component_resolution": component_resolution,
            "component_keep": component_keep,
            "component_min_size": component_min_size,
            "component_rank_by": component_rank_by,
            "silhouette_threshold": silhouette_threshold,
            "mask_refinement": mask_refinement,
            "mask_refinement_iters": mask_refinement_iters,
            "mask_refinement_dilate": mask_refinement_dilate,
            "mask_refinement_erode": mask_refinement_erode,
            "per_category": per_cat_summary,
            "per_frame": per_frame,
            "query_details": query_details,
            "bootstrap_miou": bootstrap_mean_ci(ious),
        }
    )
    return summary


def evaluate_scene(
    *,
    scene: str,
    config_path: str,
    checkpoint_path: str,
    label_dir: str,
    output_dir: Path,
    summary_head: torch.nn.Module,
    summary_head_weights: str,
    text_embedding_cache: Optional[str],
    canonical_embedding_cache: Optional[str],
    score_cache_path: Optional[str],
    prompt_templates: List[str],
    selection_specs: List[SelectionSpec],
    score_source: str,
    scoring: str,
    compact_feature_key: str,
    direct_readout_mode: str,
    direct_readout_k: int,
    direct_readout_candidate_k: int,
    softmax_temperature: float,
    score_aggregation: str,
    score_aggregation_resolution: int,
    score_aggregation_blend: float,
    selection_refinement: str,
    selection_min_ratio: float,
    selection_max_ratio: float,
    component_support_ratio: float,
    component_resolution: int,
    component_keep: int,
    component_min_size: int,
    component_rank_by: str,
    registered_view_fallback: str,
    registration_frame_mode: str,
    registration_max_frames: int,
    registration_chunk_size: int,
    registration_depth_tolerance: float,
    registration_relative_depth_tolerance: float,
    registration_alpha_threshold: float,
    registration_weight_mode: str,
    registration_confidence_blend: float,
    registration_confidence_mode: str,
    disable_registered_refiner: bool,
    silhouette_threshold: float,
    mask_refinement: str,
    mask_refinement_iters: int,
    mask_refinement_dilate: int,
    mask_refinement_erode: int,
    min_select: int,
    chunk_size: int,
    official_frames_only: bool,
    save_masks: bool,
    device: torch.device,
) -> Dict:
    print(f"\n{'=' * 72}\nLERF direct 3D object selection: {scene}\n{'=' * 72}")
    frame_annotations, scene_categories, img_h, img_w = load_lerf_ovs_labels(label_dir, scene)
    if official_frames_only:
        official = set(OPEN_GAUSSIAN_LERF_FRAMES.get(scene, []))
        frame_annotations = {
            frame_id: objects
            for frame_id, objects in frame_annotations.items()
            if frame_id in official
        }
    if not frame_annotations:
        raise RuntimeError(f"No annotated frames selected for scene: {scene}")
    print(f"  categories: {len(scene_categories)} | frames: {len(frame_annotations)}")
    print(f"  mask resolution: {img_w}x{img_h}")

    print("  loading RADIO-GS pipeline")
    model, codec, _renderer, _sharpener, _refiner, config, is_hybrid = load_render_pipeline(
        config_path,
        checkpoint_path,
        device,
    )
    if not is_hybrid:
        logger.warning("Model architecture is explicit; direct readout will use per-Gaussian compact codes")

    dataset = build_lerf_dataset_for_scene(
        scene,
        config,
        label_dir,
        feature_height=img_h,
        feature_width=img_w,
    )
    renderer = build_mask_renderer(config, height=img_h, width=img_w, device=device)

    print("  loading SigLIP2 text embeddings")
    scene_text = load_or_generate_prompt_ensemble_embeddings(
        scene_categories,
        device,
        cache_path=text_embedding_cache,
        prompt_templates=prompt_templates,
    )
    scene_text = F.normalize(scene_text.float(), dim=-1)
    canonical_text = None
    if scoring == "relevancy":
        if not canonical_embedding_cache or not Path(canonical_embedding_cache).exists():
            raise FileNotFoundError(
                "scoring='relevancy' requires --canonical_embedding_cache"
            )
        canon_payload = torch.load(canonical_embedding_cache, map_location="cpu")
        canonical_text = F.normalize(canon_payload["embeddings"].float(), dim=-1)
        print(
            f"  loaded canonical embeddings from {canonical_embedding_cache}: "
            f"{tuple(canonical_text.shape)}"
        )

    registration_stats: Dict[str, object] = {}
    score_cache_info: Dict[str, object] = {"enabled": bool(score_cache_path)}
    score_cache_metadata: Dict[str, object] = {
        "scene": scene,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "label_dir": str(label_dir),
        "summary_head_weights": str(summary_head_weights),
        "text_embedding_cache": str(text_embedding_cache or ""),
        "canonical_embedding_cache": str(canonical_embedding_cache or "")
        if scoring == "relevancy"
        else "",
        "prompt_templates": list(prompt_templates),
        "categories": list(scene_categories),
        "image_height": int(img_h),
        "image_width": int(img_w),
        "num_gaussians": int(model.get_xyz().shape[0]),
        "official_frames_only": bool(official_frames_only),
        "score_source": score_source,
        "scoring": scoring,
        "compact_feature_key": compact_feature_key,
        "direct_readout_mode": direct_readout_mode,
        "direct_readout_k": int(direct_readout_k),
        "direct_readout_candidate_k": int(direct_readout_candidate_k),
        "softmax_temperature": float(softmax_temperature),
        "registered_view_fallback": registered_view_fallback,
        "registration_frame_mode": registration_frame_mode,
        "registration_max_frames": int(registration_max_frames),
        "registration_depth_tolerance": float(registration_depth_tolerance),
        "registration_relative_depth_tolerance": float(registration_relative_depth_tolerance),
        "registration_alpha_threshold": float(registration_alpha_threshold),
        "registration_weight_mode": registration_weight_mode,
        "registration_confidence_blend": float(registration_confidence_blend),
        "registration_confidence_mode": registration_confidence_mode,
        "disable_registered_refiner": bool(disable_registered_refiner),
    }
    scores: Optional[torch.Tensor] = None
    cache_path = Path(score_cache_path) if score_cache_path else None
    if cache_path is not None:
        score_cache_info["path"] = str(cache_path)
        if cache_path.exists():
            print(f"  loading primitive score cache: {cache_path}")
            scores, registration_stats = load_score_cache(
                cache_path,
                expected_metadata=score_cache_metadata,
            )
            score_cache_info["status"] = "hit"
            if tuple(scores.shape) != (int(model.get_xyz().shape[0]), len(scene_categories)):
                raise ValueError(
                    "score cache shape mismatch: "
                    f"got {tuple(scores.shape)}, expected "
                    f"{(int(model.get_xyz().shape[0]), len(scene_categories))}"
                )
        else:
            score_cache_info["status"] = "miss"

    direct_scores: Optional[torch.Tensor] = None
    needs_direct_scores = score_source == "direct" or (
        score_source == "registered_view" and registered_view_fallback == "direct"
    )
    if scores is None and needs_direct_scores:
        print("  computing Gaussian-level text scores")
        direct_scores = compute_gaussian_text_scores(
            model,
            codec,
            summary_head,
            scene_text,
            canonical_text,
            is_hybrid=is_hybrid,
            direct_readout_mode=direct_readout_mode,
            direct_readout_k=direct_readout_k,
            direct_readout_candidate_k=direct_readout_candidate_k,
            compact_feature_key=compact_feature_key,
            scoring=scoring,
            softmax_temperature=softmax_temperature,
            chunk_size=chunk_size,
            device=device,
        )

    if scores is not None:
        print(
            f"  using cached primitive scores "
            f"({scores.shape[0]} Gaussians x {scores.shape[1]} queries)"
        )
    elif score_source == "direct":
        if direct_scores is None:
            raise RuntimeError("Internal error: direct score source missing direct scores")
        scores = direct_scores
    elif score_source == "registered_view":
        print(
            "  computing registered-view primitive scores "
            f"({registration_frame_mode}, max_frames={registration_max_frames or 'all'})"
        )
        scores, registration_stats = compute_registered_view_text_scores(
            scene=scene,
            model=model,
            codec=codec,
            renderer=_renderer,
            sharpener=_sharpener,
            refiner=choose_registration_refiner(
                _refiner,
                disable_registered_refiner=disable_registered_refiner,
            ),
            config=config,
            is_hybrid=is_hybrid,
            dataset=dataset,
            frame_annotations=frame_annotations,
            summary_head=summary_head,
            text_embeddings=scene_text,
            canonical_embeddings=canonical_text,
            scoring=scoring,
            softmax_temperature=softmax_temperature,
            registration_frame_mode=registration_frame_mode,
            registration_max_frames=registration_max_frames,
            registration_chunk_size=registration_chunk_size,
            registration_depth_tolerance=registration_depth_tolerance,
            registration_relative_depth_tolerance=registration_relative_depth_tolerance,
            registration_alpha_threshold=registration_alpha_threshold,
            registration_weight_mode=registration_weight_mode,
            registration_confidence_blend=registration_confidence_blend,
            registration_confidence_mode=registration_confidence_mode,
            fallback_scores=direct_scores if registered_view_fallback == "direct" else None,
            device=device,
        )
        print(
            "  registered "
            f"{registration_stats['registered_gaussians']}/"
            f"{registration_stats['total_gaussians']} Gaussians "
            f"({registration_stats['registered_fraction']:.3f})"
        )
    else:
        raise ValueError(f"Unsupported score source: {score_source}")

    if cache_path is not None and score_cache_info.get("status") == "miss":
        print(f"  writing primitive score cache: {cache_path}")
        save_score_cache(
            cache_path,
            scores,
            metadata=score_cache_metadata,
            registration_stats=registration_stats,
        )
        score_cache_info["status"] = "written"

    if score_aggregation != "none" and score_aggregation_blend > 0:
        print(
            "  aggregating Gaussian scores "
            f"({score_aggregation}, res={score_aggregation_resolution}, "
            f"blend={score_aggregation_blend:g})"
        )
        scores = aggregate_scores_by_voxel(
            scores,
            model.get_xyz().detach().cpu(),
            mode=score_aggregation,
            resolution=score_aggregation_resolution,
            blend=score_aggregation_blend,
        )
    if selection_refinement != "none":
        print(
            "  refining selections by voxel components "
            f"({selection_refinement}, res={component_resolution}, "
            f"support={component_support_ratio:g}, keep={component_keep}, "
            f"min={component_min_size}, rank={component_rank_by})"
        )

    scene_results: Dict[str, Dict] = {}
    for spec in selection_specs:
        scene_results[spec.tag] = evaluate_selection_spec(
            scene=scene,
            scene_categories=scene_categories,
            frame_annotations=frame_annotations,
            img_h=img_h,
            img_w=img_w,
            model=model,
            renderer=renderer,
            dataset=dataset,
            scores=scores,
            spec=spec,
            selection_refinement=selection_refinement,
            selection_min_ratio=selection_min_ratio,
            selection_max_ratio=selection_max_ratio,
            component_support_ratio=component_support_ratio,
            component_resolution=component_resolution,
            component_keep=component_keep,
            component_min_size=component_min_size,
            component_rank_by=component_rank_by,
            silhouette_threshold=silhouette_threshold,
            mask_refinement=mask_refinement,
            mask_refinement_iters=mask_refinement_iters,
            mask_refinement_dilate=mask_refinement_dilate,
            mask_refinement_erode=mask_refinement_erode,
            min_select=min_select,
            output_dir=output_dir,
            save_masks=save_masks,
            device=device,
        )
        m = scene_results[spec.tag]
        print(
            f"  {spec.tag:<14} mIoU={m['miou']:.4f} "
            f"Acc@0.25={m['acc025']:.4f} Acc@0.50={m['acc050']:.4f} n={m['n']}"
        )

    best_tag = max(scene_results, key=lambda tag: scene_results[tag]["miou"])
    return {
        "scene": scene,
        "config": config_path,
        "checkpoint": checkpoint_path,
        "score_source": score_source,
        "compact_feature_key": compact_feature_key,
        "canonical_embedding_cache": canonical_embedding_cache if scoring == "relevancy" else "",
        "registration": registration_stats,
        "score_cache": score_cache_info,
        "categories": scene_categories,
        "image_height": img_h,
        "image_width": img_w,
        "official_frames_only": official_frames_only,
        "official_frames": OPEN_GAUSSIAN_LERF_FRAMES.get(scene, []),
        "results": scene_results,
        "best_by_miou": best_tag,
    }


def build_selection_specs(args: argparse.Namespace) -> List[SelectionSpec]:
    if args.selection_mode == "top_ratio":
        values = parse_float_list(args.ratio_sweep) or [float(args.top_ratio)]
    elif args.selection_mode == "score_threshold":
        values = parse_float_list(args.threshold_sweep) or [float(args.score_threshold)]
    elif args.selection_mode == "mean_std":
        values = parse_float_list(args.mean_std_sweep) or [float(args.mean_std)]
    else:
        raise ValueError(f"Unsupported selection mode: {args.selection_mode}")
    seen = set()
    specs: List[SelectionSpec] = []
    for value in values:
        key = (args.selection_mode, float(value))
        if key in seen:
            continue
        seen.add(key)
        specs.append(SelectionSpec(args.selection_mode, float(value)))
    return specs


def write_scene_report(output_dir: Path, scene: str, report: Dict) -> None:
    scene_dir = output_dir / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    out_json = scene_dir / "lerf_direct_3d_selection_results.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = []
    rows.append("# LERF Direct 3D Object Selection")
    rows.append("")
    rows.append(f"- Scene: `{scene}`")
    rows.append("- Protocol: OpenGaussian-style direct 3D primitive selection; rendering is used only for mask evaluation.")
    rows.append("- Query location: 3D Gaussian primitives.")
    protocol = report.get("protocol", {})
    rows.append(f"- Feature source: {protocol.get('feature_source', 'pre-refiner RADIO-GS Gaussian-center decoded features')}.")
    rows.append(f"- Score source: `{protocol.get('score_source', 'direct')}`.")
    registration = report.get("scene", {}).get("registration", {})
    if registration:
        rows.append(
            "- Registration views: "
            f"{registration.get('num_frames', 0)} frames, "
            f"{registration.get('registered_gaussians', 0)}/"
            f"{registration.get('total_gaussians', 0)} Gaussians "
            f"({float(registration.get('registered_fraction', 0.0)):.3f})."
        )
    if protocol.get("score_aggregation", "none") != "none":
        rows.append(
            "- Score aggregation: "
            f"{protocol.get('score_aggregation')} "
            f"(res={protocol.get('score_aggregation_resolution')}, "
            f"blend={protocol.get('score_aggregation_blend')})."
        )
    if protocol.get("selection_refinement", "none") != "none":
        rows.append(
            "- Selection refinement: "
            f"{protocol.get('selection_refinement')} "
            f"(res={protocol.get('component_resolution')}, "
            f"support={protocol.get('component_support_ratio')}, "
            f"keep={protocol.get('component_keep')}, "
            f"rank={protocol.get('component_rank_by')})."
        )
    if float(protocol.get("selection_min_ratio", 0.0) or 0.0) > 0 or float(
        protocol.get("selection_max_ratio", 0.0) or 0.0
    ) > 0:
        rows.append(
            "- Selection ratio bounds: "
            f"floor={protocol.get('selection_min_ratio')}, "
            f"cap={protocol.get('selection_max_ratio')}."
        )
    rows.append("- Text head: SigLIP2 summary/text space.")
    rows.append("")
    rows.append("| Selection | mIoU | Acc@0.25 | Acc@0.50 | N |")
    rows.append("|---|---:|---:|---:|---:|")
    for tag, metrics in report["scene"]["results"].items():
        ci = metrics.get("bootstrap_miou", {})
        ci_suffix = ""
        if ci:
            ci_suffix = f" [{float(ci.get('ci_low', 0.0)):.4f}, {float(ci.get('ci_high', 0.0)):.4f}]"
        rows.append(
            f"| {tag} | {metrics['miou']:.4f}{ci_suffix} | {metrics['acc025']:.4f} | "
            f"{metrics['acc050']:.4f} | {metrics['n']} |"
        )
    rows.append("")
    rows.append(f"Best diagnostic ratio by mIoU: `{report['scene']['best_by_miou']}`.")
    (scene_dir / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenGaussian-style LERF direct 3D object selection for RADIO-GS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Scene RADIO-GS config YAML")
    parser.add_argument("--checkpoint", required=True, help="Scene RADIO-GS checkpoint")
    parser.add_argument("--scene", required=True, choices=list(LERF_OVS_SCENES), help="LERF scene")
    parser.add_argument("--label_dir", default=DEFAULT_LABEL_DIR, help="LERF-OVS label root")
    parser.add_argument("--output_dir", default="output/radio_gs/lerf_direct_3d_selection", help="Output root")
    parser.add_argument("--summary_head_weights", default="checkpoints/siglip2_summary_head.pth", help="SigLIP2 summary head weights")
    parser.add_argument("--text_embedding_cache", default="checkpoints/siglip2_lerf_text_embeddings.pt", help="SigLIP2 text embedding cache")
    parser.add_argument("--score_cache", default="", help="Optional primitive text-score cache path; loaded if metadata matches, written when missing")
    parser.add_argument("--prompt_templates", default=DEFAULT_PROMPT_TEMPLATES, help="Prompt templates separated by '|'; use {query}")
    parser.add_argument("--selection_mode", choices=["top_ratio", "score_threshold", "mean_std"], default="top_ratio")
    parser.add_argument("--top_ratio", type=float, default=0.02, help="Main fixed Gaussian top-ratio for top_ratio mode")
    parser.add_argument("--ratio_sweep", default="", help="Comma/space separated top-ratio sweep values")
    parser.add_argument("--score_threshold", type=float, default=0.25, help="Main score threshold for score_threshold mode")
    parser.add_argument("--threshold_sweep", default="", help="Comma/space separated score thresholds")
    parser.add_argument("--mean_std", type=float, default=1.0, help="Main mean+std multiplier for mean_std mode")
    parser.add_argument("--mean_std_sweep", default="", help="Comma/space separated mean_std multipliers")
    parser.add_argument("--selection_min_ratio", type=float, default=0.0, help="Optional GT-free top-ratio floor applied after selection")
    parser.add_argument("--selection_max_ratio", type=float, default=0.0, help="Optional GT-free top-ratio cap applied after selection")
    parser.add_argument("--score_source", choices=["direct", "registered_view"], default="direct", help="Primitive score source for direct 3D selection")
    parser.add_argument("--scoring", choices=["cosine", "softmax_scene", "relevancy"], default="cosine", help="Text-Gaussian score")
    parser.add_argument("--canonical_embedding_cache", default="checkpoints/siglip2_canonical_embeddings.pt", help="Canonical text embeddings for relevancy scoring")
    parser.add_argument("--direct_readout_mode", choices=["gaussian", "knn"], default="gaussian", help="Direct 3D compact readout mode before HCD decoding")
    parser.add_argument("--direct_readout_k", type=int, default=8, help="Neighbour count for knn direct_readout_mode")
    parser.add_argument("--direct_readout_candidate_k", type=int, default=0, help="Optional candidate count before scale-aware KNN pruning")
    parser.add_argument("--compact_feature_key", choices=["features", "fused", "semantic", "geometry"], default="features", help="Hybrid compact readout before HCD decoding")
    parser.add_argument("--softmax_temperature", type=float, default=50.0, help="Logit scale for softmax_scene")
    parser.add_argument("--score_aggregation", choices=["none", "voxel_mean", "voxel_max", "voxel_max_dilate"], default="none", help="GT-free spatial aggregation applied to Gaussian text scores")
    parser.add_argument("--score_aggregation_resolution", type=int, default=64, help="Voxel resolution per scene axis for score aggregation")
    parser.add_argument("--score_aggregation_blend", type=float, default=0.0, help="Blend weight for aggregated scores; 0 disables aggregation")
    parser.add_argument("--selection_refinement", choices=["none", "top_score_components", "largest_components", "seed_expand_components"], default="none", help="GT-free connected-component filtering after score-based primitive selection")
    parser.add_argument("--component_support_ratio", type=float, default=0.05, help="Wider top-ratio support pool for seed_expand_components")
    parser.add_argument("--component_resolution", type=int, default=64, help="Voxel resolution for selection_refinement")
    parser.add_argument("--component_keep", type=int, default=1, help="Number of connected components to keep per query")
    parser.add_argument("--component_min_size", type=int, default=8, help="Minimum selected Gaussians per component before ranking")
    parser.add_argument("--component_rank_by", choices=["mean_score", "score_sum", "size"], default="score_sum", help="How top_score_components ranks connected components")
    parser.add_argument("--registered_view_fallback", choices=["direct", "low"], default="direct", help="Fallback score for Gaussians not visible in registration views")
    parser.add_argument("--registration_frame_mode", choices=["official", "annotated", "all_poses", "train", "val"], default="official", help="Views used to register rendered features back to primitives")
    parser.add_argument("--registration_max_frames", type=int, default=0, help="Evenly subsample registration views; 0 uses all selected views")
    parser.add_argument("--registration_chunk_size", type=int, default=32768, help="Gaussian chunk size for rendered-view registration sampling")
    parser.add_argument("--registration_depth_tolerance", type=float, default=0.08, help="Absolute depth tolerance for rendered-view primitive registration")
    parser.add_argument("--registration_relative_depth_tolerance", type=float, default=0.02, help="Relative depth tolerance for rendered-view primitive registration")
    parser.add_argument("--registration_alpha_threshold", type=float, default=0.02, help="Minimum rendered alpha for rendered-view primitive registration")
    parser.add_argument("--registration_weight_mode", choices=["uniform", "alpha", "alpha_depth"], default="uniform", help="Contribution-style weighting for VPR registered samples")
    parser.add_argument("--registration_confidence_blend", type=float, default=0.0, help="Blend weight for GT-free registration-count confidence calibration")
    parser.add_argument("--registration_confidence_mode", choices=["log", "linear"], default="log", help="How registration counts are mapped to confidence")
    parser.add_argument("--disable_registered_refiner", action="store_true", help="Disable VFA/screen refiner only for registered-view primitive scoring")
    parser.add_argument("--silhouette_threshold", type=float, default=0.7, help="OpenGaussian-style rendered silhouette threshold")
    parser.add_argument("--mask_refinement", choices=["none", "rgb_grabcut"], default="none", help="Optional GT-free RGB boundary snapping after rendering selected primitives")
    parser.add_argument("--mask_refinement_iters", type=int, default=1, help="GrabCut iterations for rgb_grabcut mask refinement")
    parser.add_argument("--mask_refinement_dilate", type=int, default=5, help="Pixel dilation radius defining the rgb_grabcut support band")
    parser.add_argument("--mask_refinement_erode", type=int, default=2, help="Pixel erosion radius defining sure foreground for rgb_grabcut")
    parser.add_argument("--min_select", type=int, default=1, help="Minimum selected Gaussians per query")
    parser.add_argument("--chunk_size", type=int, default=8192, help="Gaussian decode/projection chunk size")
    parser.add_argument("--all_labeled_frames", action="store_true", help="Use all local labels instead of OpenGaussian official frames")
    parser.add_argument("--save_masks", action="store_true", help="Save rendered binary prediction masks")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.label_dir = resolve_lerf_label_dir(args.label_dir)
    prompt_templates = parse_prompt_templates(args.prompt_templates)
    specs = build_selection_specs(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("LERF-OVS Direct 3D Object Selection")
    print("=" * 72)
    print(f"Scene:      {args.scene}")
    print(f"Device:     {device}")
    print(f"Selection:  {', '.join(spec.tag for spec in specs)}")
    print(f"Score src:  {args.score_source}")
    print(f"Scoring:    {args.scoring}")
    print(f"Silhouette: > {args.silhouette_threshold}")
    if args.mask_refinement != "none":
        print(f"Mask ref.:  {args.mask_refinement}")
    print()

    summary_head = load_summary_head(args.summary_head_weights, device)
    out_root = Path(args.output_dir)
    t0 = time.time()
    scene_report = evaluate_scene(
        scene=args.scene,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        label_dir=args.label_dir,
        output_dir=out_root,
        summary_head=summary_head,
        summary_head_weights=args.summary_head_weights,
        text_embedding_cache=args.text_embedding_cache,
        canonical_embedding_cache=args.canonical_embedding_cache,
        score_cache_path=args.score_cache or None,
        prompt_templates=prompt_templates,
        selection_specs=specs,
        score_source=args.score_source,
        scoring=args.scoring,
        compact_feature_key=args.compact_feature_key,
        direct_readout_mode=args.direct_readout_mode,
        direct_readout_k=args.direct_readout_k,
        direct_readout_candidate_k=args.direct_readout_candidate_k,
        softmax_temperature=args.softmax_temperature,
        score_aggregation=args.score_aggregation,
        score_aggregation_resolution=args.score_aggregation_resolution,
        score_aggregation_blend=args.score_aggregation_blend,
        selection_refinement=args.selection_refinement,
        selection_min_ratio=args.selection_min_ratio,
        selection_max_ratio=args.selection_max_ratio,
        component_support_ratio=args.component_support_ratio,
        component_resolution=args.component_resolution,
        component_keep=args.component_keep,
        component_min_size=args.component_min_size,
        component_rank_by=args.component_rank_by,
        registered_view_fallback=args.registered_view_fallback,
        registration_frame_mode=args.registration_frame_mode,
        registration_max_frames=args.registration_max_frames,
        registration_chunk_size=args.registration_chunk_size,
        registration_depth_tolerance=args.registration_depth_tolerance,
        registration_relative_depth_tolerance=args.registration_relative_depth_tolerance,
        registration_alpha_threshold=args.registration_alpha_threshold,
        registration_weight_mode=args.registration_weight_mode,
        registration_confidence_blend=args.registration_confidence_blend,
        registration_confidence_mode=args.registration_confidence_mode,
        disable_registered_refiner=args.disable_registered_refiner,
        silhouette_threshold=args.silhouette_threshold,
        mask_refinement=args.mask_refinement,
        mask_refinement_iters=args.mask_refinement_iters,
        mask_refinement_dilate=args.mask_refinement_dilate,
        mask_refinement_erode=args.mask_refinement_erode,
        min_select=args.min_select,
        chunk_size=args.chunk_size,
        official_frames_only=not args.all_labeled_frames,
        save_masks=args.save_masks,
        device=device,
    )
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {key: str(value) for key, value in vars(args).items()},
        "protocol": {
            "name": "OpenGaussian-style LERF-OVS direct 3D object selection",
            "query_location": "3D Gaussian primitives",
            "feature_source": (
                (
                    "pre-refiner rendered SigLIP2 features registered back to Gaussian primitives"
                    if args.disable_registered_refiner
                    else "post-refiner rendered SigLIP2 features registered back to Gaussian primitives"
                )
                if args.score_source == "registered_view"
                else "pre-refiner Gaussian-center decoded RADIO-compatible features"
            ),
            "score_source": args.score_source,
            "score_cache": args.score_cache,
            "compact_feature_key": args.compact_feature_key,
            "direct_readout_mode": args.direct_readout_mode,
            "direct_readout_k": args.direct_readout_k,
            "direct_readout_candidate_k": args.direct_readout_candidate_k,
            "text_head": "SigLIP2 summary/text-aligned head",
            "canonical_embedding_cache": args.canonical_embedding_cache if args.scoring == "relevancy" else "",
            "score_aggregation": args.score_aggregation,
            "score_aggregation_resolution": args.score_aggregation_resolution,
            "score_aggregation_blend": args.score_aggregation_blend,
            "selection_refinement": args.selection_refinement,
            "selection_min_ratio": args.selection_min_ratio,
            "selection_max_ratio": args.selection_max_ratio,
            "component_support_ratio": args.component_support_ratio,
            "component_resolution": args.component_resolution,
            "component_keep": args.component_keep,
            "component_min_size": args.component_min_size,
            "component_rank_by": args.component_rank_by,
            "registered_view_fallback": args.registered_view_fallback,
            "registration_frame_mode": args.registration_frame_mode,
            "registration_max_frames": args.registration_max_frames,
            "registration_depth_tolerance": args.registration_depth_tolerance,
            "registration_relative_depth_tolerance": args.registration_relative_depth_tolerance,
            "registration_alpha_threshold": args.registration_alpha_threshold,
            "registration_weight_mode": args.registration_weight_mode,
            "registration_confidence_blend": args.registration_confidence_blend,
            "registration_confidence_mode": args.registration_confidence_mode,
            "disable_registered_refiner": bool(args.disable_registered_refiner),
            "render_role": "render selected primitives only for mask evaluation",
            "metrics": ["mIoU", "Acc@0.25", "Acc@0.50"],
            "silhouette_threshold": args.silhouette_threshold,
            "mask_refinement": args.mask_refinement,
            "mask_refinement_iters": args.mask_refinement_iters,
            "mask_refinement_dilate": args.mask_refinement_dilate,
            "mask_refinement_erode": args.mask_refinement_erode,
        },
        "prompt_templates": prompt_templates,
        "elapsed_seconds": time.time() - t0,
        "scene": scene_report,
    }
    write_scene_report(out_root, args.scene, report)


if __name__ == "__main__":
    main()
