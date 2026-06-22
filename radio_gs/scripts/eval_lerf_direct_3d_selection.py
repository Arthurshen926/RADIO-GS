#!/usr/bin/env python3
"""Evaluate LERF-OVS direct 3D object selection.

This script implements an OpenGaussian-style protocol for RADIO-GS:

1. Decode pre-refiner Gaussian/primitive features at 3D Gaussian centers, or
   register rendered SigLIP2-aligned features back to visible primitives.
2. Select 3D primitives from text-Gaussian similarity scores.
3. Render selected primitives as binary masks on the official LERF-OVS views.
4. Report mIoU, Acc@0.25, and Acc@0.50 against LERF-OVS masks.

The registered-view path is View-to-Primitive Registration (VPR): text queries
still operate on Gaussian primitives, while rendered RADIO-compatible features
provide the registration signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
from radio_gs.models.point_summary_adapter import (
    CompactToSummaryAdapter,
    append_point_summary_context,
    point_summary_context_dim,
)
from radio_gs.models.proposal_memory import (
    build_voxel_proposal_labels,
    propagate_logits_with_proposals,
)
from radio_gs.models.foundation_cache import load_foundation_cache
from radio_gs.models.radio_adaptors import (
    load_radio_adaptor_from_checkpoint,
    project_feature_map_with_adaptor,
)
from radio_gs.models.sam3_proposal_registration import (
    build_sam3_mask_memberships,
    fuse_scores_with_query_sam3_proposals,
    fuse_scores_with_sam3_proposals,
)
from radio_gs.models.prompt_conditioned_mask_head import PromptConditionedMaskHead
from radio_gs.models.prompt_conditioned_mask_refinement import (
    choose_mask_candidate_by_initial_overlap,
    filter_refined_mask_by_heatmap_support,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.utils.checkpoint_io import load_trusted_checkpoint
from radio_gs.scripts.eval_lerf_grounding import (
    DEFAULT_GT_FEATURE_ROOT,
    DEFAULT_LABEL_DIR,
    DEFAULT_PROMPT_TEMPLATES,
    LERF_OVS_SCENES,
    build_gt_masks,
    heatmap_peak_in_shape,
    keep_peak_connected_component,
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
from radio_gs.scripts.train_prompt_conditioned_sam3_mask_head import (
    _load_text_embedding_map,
    build_coarse_prompt_from_target,
)
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
from radio_gs.training.feature_training_utils import FoundationMaskLogitProjector

logger = logging.getLogger(__name__)
SCORE_CACHE_VERSION = 1
REGISTERED_FEATURE_CACHE_VERSION = 1
DEFAULT_RADIO_ADAPTOR_CHECKPOINT = "/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar"


OPEN_GAUSSIAN_LERF_FRAMES: Dict[str, List[int]] = {
    "waldo_kitchen": [53, 66, 89, 140, 154],
    "ramen": [6, 24, 60, 65, 81, 119, 128],
    "figurines": [41, 105, 152, 195],
    "teatime": [2, 25, 43, 107, 129, 140],
}


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_if_exists(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_file():
        return ""
    return sha256_file(path_obj)


def tensor_sha256_float32(tensor: torch.Tensor) -> str:
    """Stable SHA256 hash for a float32 tensor payload."""
    arr = tensor.detach().cpu().float().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def xyz_geometry_fingerprint(xyz: torch.Tensor) -> Dict[str, Any]:
    """Compact provenance for row-aligned Gaussian primitive caches."""
    xyz_cpu = xyz.detach().cpu().float()
    if xyz_cpu.ndim != 2 or xyz_cpu.shape[-1] != 3:
        raise ValueError(f"Expected xyz [N,3], got {tuple(xyz_cpu.shape)}")
    if xyz_cpu.numel() == 0:
        xyz_min = xyz_max = xyz_mean = [0.0, 0.0, 0.0]
    else:
        xyz_min = [float(v) for v in xyz_cpu.min(dim=0).values.tolist()]
        xyz_max = [float(v) for v in xyz_cpu.max(dim=0).values.tolist()]
        xyz_mean = [float(v) for v in xyz_cpu.mean(dim=0).tolist()]
    return {
        "num_gaussians": int(xyz_cpu.shape[0]),
        "xyz_sha256": tensor_sha256_float32(xyz_cpu),
        "xyz_min": xyz_min,
        "xyz_max": xyz_max,
        "xyz_mean": xyz_mean,
    }


def gaussian_geometry_fingerprint(
    xyz: torch.Tensor,
    *,
    scales: Optional[torch.Tensor] = None,
    rotations: Optional[torch.Tensor] = None,
    opacities: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Fingerprint Gaussian rows and footprint tensors used by VPR caches."""
    fingerprint = xyz_geometry_fingerprint(xyz)
    n_gaussians = int(fingerprint["num_gaussians"])
    optional_tensors = {
        "scales": scales,
        "rotations": rotations,
        "opacities": opacities,
    }
    for name, tensor in optional_tensors.items():
        if tensor is None:
            continue
        tensor_cpu = tensor.detach().cpu().float()
        if tensor_cpu.ndim == 0 or int(tensor_cpu.shape[0]) != n_gaussians:
            raise ValueError(
                f"Expected {name} first dimension {n_gaussians}, "
                f"got {tuple(tensor_cpu.shape)}"
            )
        fingerprint[f"{name}_shape"] = [int(v) for v in tensor_cpu.shape]
        fingerprint[f"{name}_sha256"] = tensor_sha256_float32(tensor_cpu)
    return fingerprint


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
        if self.mode == "score_margin":
            return f"margin{self.value:g}".replace(".", "p")
        if self.mode == "score_ratio":
            return f"ratio{self.value:g}".replace(".", "p")
        if self.mode == "entropy_score":
            return f"entropy{self.value:g}".replace(".", "p")
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
    xyz: Optional[torch.Tensor] = None,
) -> None:
    """Persist pre-aggregation primitive text scores with protocol metadata."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SCORE_CACHE_VERSION,
        "metadata": canonical_score_cache_metadata(metadata),
        "registration_stats": _canonical_cache_value(registration_stats),
        "scores": scores.detach().cpu(),
    }
    if xyz is not None:
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError(f"Expected xyz [N,3], got {tuple(xyz.shape)}")
        if int(xyz.shape[0]) != int(scores.shape[0]):
            raise ValueError(
                "score cache xyz row count must match scores: "
                f"{int(xyz.shape[0])} vs {int(scores.shape[0])}"
            )
        payload["geometry_fingerprint"] = xyz_geometry_fingerprint(xyz)
        payload["xyz"] = xyz.detach().cpu().float()
    torch.save(payload, cache_path)


def save_registered_feature_cache(
    path: str | Path,
    *,
    xyz: torch.Tensor,
    summary_features: torch.Tensor,
    valid: torch.Tensor,
    view_counts: torch.Tensor,
    metadata: Dict[str, Any],
    scales: Optional[torch.Tensor] = None,
    rotations: Optional[torch.Tensor] = None,
    opacities: Optional[torch.Tensor] = None,
) -> None:
    """Persist VPR-registered primitive summary features for distillation.

    The cache stores SigLIP2 summary-space features at Gaussian granularity.
    This gives the direct 3D readout a training target derived from rendered
    VPR registration, without using LERF masks or per-query GT thresholds.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz [N,3], got {tuple(xyz.shape)}")
    if summary_features.ndim != 2 or summary_features.shape[0] != xyz.shape[0]:
        raise ValueError(
            f"Expected summary_features [N,D] aligned with xyz, got "
            f"{tuple(summary_features.shape)} and {tuple(xyz.shape)}"
        )
    if valid.shape != (xyz.shape[0],):
        raise ValueError(f"Expected valid [{xyz.shape[0]}], got {tuple(valid.shape)}")
    if view_counts.shape != (xyz.shape[0],):
        raise ValueError(
            f"Expected view_counts [{xyz.shape[0]}], got {tuple(view_counts.shape)}"
        )

    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": REGISTERED_FEATURE_CACHE_VERSION,
            "metadata": canonical_score_cache_metadata(metadata),
            "feature_space": "siglip_summary",
            "feature_key": "summary_features",
            "geometry_fingerprint": gaussian_geometry_fingerprint(
                xyz,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
            ),
            "xyz": xyz.detach().cpu().float(),
            "summary_features": summary_features.detach().cpu().float(),
            "valid": valid.detach().cpu().bool(),
            "view_counts": view_counts.detach().cpu().float(),
        },
        cache_path,
    )


def load_score_cache(
    path: str | Path,
    *,
    expected_metadata: Dict[str, Any],
    expected_xyz: Optional[torch.Tensor] = None,
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
    if (
        "registration_assignment_mode" in expected
        and "registration_assignment_mode" not in cached
        and expected.get("registration_assignment_mode") == "center"
    ):
        cached["registration_assignment_mode"] = "center"
    default_compatible_keys = {
        "direct_primitive_confidence_mode": "none",
        "direct_primitive_confidence_blend": 0.0,
        "direct_primitive_opacity_threshold": 0.02,
    }
    for key, default in default_compatible_keys.items():
        if key in expected and key not in cached and expected.get(key) == default:
            cached[key] = default
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
    if expected_xyz is not None:
        fingerprint = payload.get("geometry_fingerprint")
        if not isinstance(fingerprint, dict):
            raise ValueError("score cache missing geometry_fingerprint")
        expected_fingerprint = xyz_geometry_fingerprint(expected_xyz)
        cached_hash = str(fingerprint.get("xyz_sha256", ""))
        expected_hash = str(expected_fingerprint.get("xyz_sha256", ""))
        cached_count = int(fingerprint.get("num_gaussians", -1))
        expected_count = int(expected_fingerprint["num_gaussians"])
        if cached_hash != expected_hash or cached_count != expected_count:
            raise ValueError(
                "score cache geometry mismatch: "
                f"cached_count={cached_count} expected_count={expected_count} "
                f"cached_xyz_sha256={cached_hash} expected_xyz_sha256={expected_hash}"
            )
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


def _resize_binary_mask(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred_u8 = (pred > 0).astype(np.uint8)
    gt_u8 = (gt > 0).astype(np.uint8)
    if pred_u8.shape != gt_u8.shape:
        pred_u8 = cv2.resize(
            pred_u8,
            (gt_u8.shape[1], gt_u8.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return pred_u8, gt_u8


def _disk_kernel(radius: int) -> np.ndarray:
    radius_i = max(int(radius), 1)
    size = radius_i * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if not mask_u8.any():
        return np.zeros_like(mask_u8, dtype=np.uint8)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return (mask_u8 != eroded).astype(np.uint8)


def boundary_f_score(pred: np.ndarray, gt: np.ndarray, *, dilation_ratio: float = 0.02) -> float:
    """Return boundary F-score with deterministic morphology tolerance."""
    pred_u8, gt_u8 = _resize_binary_mask(pred, gt)
    pred_boundary = _binary_boundary(pred_u8)
    gt_boundary = _binary_boundary(gt_u8)
    pred_count = int(pred_boundary.sum())
    gt_count = int(gt_boundary.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0

    diag = math.sqrt(float(gt_u8.shape[0] ** 2 + gt_u8.shape[1] ** 2))
    dilation_pixels = max(1, int(round(float(dilation_ratio) * diag)))
    kernel = _disk_kernel(dilation_pixels)
    pred_match = cv2.dilate(pred_boundary, kernel, iterations=1) > 0
    gt_match = cv2.dilate(gt_boundary, kernel, iterations=1) > 0
    precision = float(np.logical_and(pred_boundary > 0, gt_match).sum() / max(pred_count, 1))
    recall = float(np.logical_and(gt_boundary > 0, pred_match).sum() / max(gt_count, 1))
    denom = precision + recall
    return float(2.0 * precision * recall / denom) if denom > 0 else 0.0


def trimap_iou(pred: np.ndarray, gt: np.ndarray, *, dilation_pixels: int = 2) -> float:
    """Return IoU measured only in a GT boundary trimap."""
    pred_u8, gt_u8 = _resize_binary_mask(pred, gt)
    if not gt_u8.any():
        return 0.0
    kernel = _disk_kernel(dilation_pixels)
    dilated = cv2.dilate(gt_u8, kernel, iterations=1) > 0
    eroded = cv2.erode(gt_u8, kernel, iterations=1) > 0
    trimap = np.logical_xor(dilated, eroded)
    if not trimap.any():
        trimap = gt_u8 > 0
    pred_band = np.logical_and(pred_u8 > 0, trimap)
    gt_band = np.logical_and(gt_u8 > 0, trimap)
    union = np.logical_or(pred_band, gt_band).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_band, gt_band).sum() / union)


def _as_numpy_2d(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().float().cpu().numpy()
    else:
        arr = np.asarray(value, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D map after squeezing, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _resize_float_map(value: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if value.shape == shape:
        return value.astype(np.float32, copy=False)
    return cv2.resize(value.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _normalize_edge_map(value: np.ndarray) -> np.ndarray:
    edge = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    edge = np.maximum(edge, 0.0)
    if not np.any(edge > 0):
        return np.zeros_like(edge, dtype=np.float32)
    scale = float(np.percentile(edge[edge > 0], 95))
    if scale <= 1e-8:
        scale = float(edge.max())
    return np.clip(edge / max(scale, 1e-8), 0.0, 1.0).astype(np.float32)


def geometry_discontinuity_maps(
    alpha_map: np.ndarray | torch.Tensor,
    depth_map: np.ndarray | torch.Tensor,
) -> Dict[str, np.ndarray]:
    """Build normalized alpha/depth edge maps for boundary-failure analysis."""
    alpha = _as_numpy_2d(alpha_map)
    depth = _as_numpy_2d(depth_map)
    if depth.shape != alpha.shape:
        depth = _resize_float_map(depth, alpha.shape)

    alpha_smooth = cv2.GaussianBlur(np.nan_to_num(alpha, nan=0.0), (3, 3), 0)
    alpha_dx = cv2.Sobel(alpha_smooth, cv2.CV_32F, 1, 0, ksize=3)
    alpha_dy = cv2.Sobel(alpha_smooth, cv2.CV_32F, 0, 1, ksize=3)
    alpha_edge = _normalize_edge_map(np.sqrt(alpha_dx * alpha_dx + alpha_dy * alpha_dy))

    valid_depth = np.isfinite(depth) & (depth > 0)
    depth_clean = np.where(valid_depth, depth, 0.0).astype(np.float32)
    if valid_depth.any():
        median_depth = float(np.median(depth_clean[valid_depth]))
        depth_clean = np.where(valid_depth, depth_clean, median_depth).astype(np.float32)
    depth_smooth = cv2.GaussianBlur(depth_clean, (3, 3), 0)
    depth_dx = cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=3)
    depth_dy = cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=3)
    depth_edge = _normalize_edge_map(np.sqrt(depth_dx * depth_dx + depth_dy * depth_dy))

    discontinuity = np.maximum(alpha_edge, depth_edge).astype(np.float32)
    return {
        "alpha_edge": alpha_edge,
        "depth_edge": depth_edge,
        "discontinuity": discontinuity,
    }


def _mean_on_mask(value: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return 0.0
    return float(np.asarray(value, dtype=np.float32)[mask_bool].mean())


def compute_geometry_boundary_alignment(
    pred: np.ndarray,
    gt: np.ndarray,
    alpha_map: np.ndarray | torch.Tensor,
    depth_map: np.ndarray | torch.Tensor,
) -> Dict[str, float | int]:
    """Measure alpha/depth discontinuity strength on predicted and GT boundaries."""
    pred_u8, gt_u8 = _resize_binary_mask(pred, gt)
    maps = geometry_discontinuity_maps(alpha_map, depth_map)
    for name, value in list(maps.items()):
        maps[name] = _resize_float_map(value, gt_u8.shape)

    pred_boundary = _binary_boundary(pred_u8)
    gt_boundary = _binary_boundary(gt_u8)
    union_boundary = np.logical_or(pred_boundary > 0, gt_boundary > 0)
    matched_boundary = np.logical_and(pred_boundary > 0, gt_boundary > 0)
    error_boundary = np.logical_xor(pred_boundary > 0, gt_boundary > 0)
    background = ~union_boundary

    metrics: Dict[str, float | int] = {
        "geometry_valid": 1,
        "gt_boundary_pixels": int(gt_boundary.sum()),
        "pred_boundary_pixels": int(pred_boundary.sum()),
        "boundary_union_pixels": int(union_boundary.sum()),
        "boundary_matched_pixels": int(matched_boundary.sum()),
        "boundary_error_pixels": int(error_boundary.sum()),
    }
    for key, value in maps.items():
        metrics[f"{key}_gt_boundary_mean"] = _mean_on_mask(value, gt_boundary)
        metrics[f"{key}_pred_boundary_mean"] = _mean_on_mask(value, pred_boundary)
        metrics[f"{key}_union_boundary_mean"] = _mean_on_mask(value, union_boundary)
        metrics[f"{key}_matched_boundary_mean"] = _mean_on_mask(value, matched_boundary)
        metrics[f"{key}_error_boundary_mean"] = _mean_on_mask(value, error_boundary)
        metrics[f"{key}_background_mean"] = _mean_on_mask(value, background)
    return metrics


def geometry_boundary_score(
    mask: np.ndarray,
    alpha_map: np.ndarray | torch.Tensor,
    depth_map: np.ndarray | torch.Tensor,
) -> float:
    pred = np.asarray(mask).astype(bool)
    if pred.ndim != 2 or not pred.any():
        return 0.0
    maps = geometry_discontinuity_maps(alpha_map, depth_map)
    discontinuity = _resize_float_map(maps["discontinuity"], pred.shape)
    boundary = _binary_boundary(pred.astype(np.uint8)) > 0
    return _mean_on_mask(discontinuity, boundary)


def choose_refined_mask_by_geometry_with_report(
    initial_mask: np.ndarray,
    refined_mask: np.ndarray,
    alpha_map: np.ndarray | torch.Tensor,
    depth_map: np.ndarray | torch.Tensor,
    *,
    min_area_ratio: float = 0.5,
    max_area_ratio: float = 1.5,
    min_boundary_gain: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Accept a refined mask only when a GT-free geometry boundary gate agrees."""
    initial = np.asarray(initial_mask).astype(bool)
    refined = np.asarray(refined_mask).astype(bool)
    if initial.shape != refined.shape:
        refined = _resize_bool_mask(refined, initial.shape)
    initial_pixels = int(initial.sum())
    refined_pixels = int(refined.sum())
    area_ratio = float(refined_pixels / max(initial_pixels, 1))
    initial_score = geometry_boundary_score(initial, alpha_map, depth_map)
    refined_score = geometry_boundary_score(refined, alpha_map, depth_map)
    gain = float(refined_score - initial_score)
    report: Dict[str, Any] = {
        "geometry_gate_enabled": True,
        "geometry_gate_accepted": False,
        "geometry_gate_reason": "",
        "geometry_gate_initial_score": float(initial_score),
        "geometry_gate_refined_score": float(refined_score),
        "geometry_gate_boundary_gain": gain,
        "geometry_gate_area_ratio": area_ratio,
        "geometry_gate_min_area_ratio": float(min_area_ratio),
        "geometry_gate_max_area_ratio": float(max_area_ratio),
        "geometry_gate_min_boundary_gain": float(min_boundary_gain),
    }
    if not initial.any():
        report["geometry_gate_reason"] = "empty_initial_mask"
        return initial.copy(), report
    if not refined.any():
        report["geometry_gate_reason"] = "empty_refined_mask"
        return initial.copy(), report
    if area_ratio < float(min_area_ratio) or area_ratio > float(max_area_ratio):
        report["geometry_gate_reason"] = "area_ratio_out_of_range"
        return initial.copy(), report
    if gain < float(min_boundary_gain):
        report["geometry_gate_reason"] = "insufficient_boundary_gain"
        return initial.copy(), report
    report["geometry_gate_accepted"] = True
    report["geometry_gate_reason"] = "accepted"
    return refined.copy(), report


def apply_geometry_gate_to_sam3_report(
    initial_mask: np.ndarray,
    candidate_mask: np.ndarray,
    sam3_report: Mapping[str, Any],
    alpha_map: np.ndarray | torch.Tensor,
    depth_map: np.ndarray | torch.Tensor,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    min_boundary_gain: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply a GT-free geometry gate while preserving upstream SAM report fields."""

    initial = np.asarray(initial_mask).astype(bool)
    candidate = np.asarray(candidate_mask).astype(bool)
    report = dict(sam3_report)
    upstream_accepted = bool(report.get("accepted", False))
    if not upstream_accepted:
        report.update(
            {
                "geometry_gate_enabled": True,
                "geometry_gate_accepted": False,
                "geometry_gate_reason": "upstream_rejected",
            }
        )
        return initial.copy(), report
    gated, gate_report = choose_refined_mask_by_geometry_with_report(
        initial,
        candidate,
        alpha_map,
        depth_map,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        min_boundary_gain=min_boundary_gain,
    )
    report.update(gate_report)
    gate_accepted = bool(gate_report.get("geometry_gate_accepted", False))
    report["accepted"] = bool(upstream_accepted and gate_accepted)
    if not gate_accepted:
        report["fallback_reason"] = str(gate_report.get("geometry_gate_reason", ""))
    return gated, report


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


INITIAL_IOU_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("lt_0p25", float("-inf"), 0.25),
    ("0p25_0p50", 0.25, 0.50),
    ("0p50_0p75", 0.50, 0.75),
    ("gte_0p75", 0.75, float("inf")),
)


def summarize_initial_iou_buckets(
    query_details: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, float | int]]:
    """Summarize refinement deltas by initial coarse-mask IoU buckets."""

    summary: Dict[str, Dict[str, float | int]] = {}
    for label, low, high in INITIAL_IOU_BUCKETS:
        rows = [
            row
            for row in query_details
            if float(row.get("initial_iou", 0.0)) >= low
            and float(row.get("initial_iou", 0.0)) < high
        ]
        if not rows:
            summary[label] = {
                "n": 0,
                "initial_miou": 0.0,
                "miou": 0.0,
                "delta_miou": 0.0,
                "delta_boundary_f": 0.0,
                "delta_trimap_iou": 0.0,
                "sam3_accept_rate": 0.0,
            }
            continue
        summary[label] = {
            "n": len(rows),
            "initial_miou": float(np.mean([float(row.get("initial_iou", 0.0)) for row in rows])),
            "miou": float(np.mean([float(row.get("iou", 0.0)) for row in rows])),
            "delta_miou": float(np.mean([float(row.get("delta_iou", 0.0)) for row in rows])),
            "delta_boundary_f": float(
                np.mean([float(row.get("delta_boundary_f", 0.0)) for row in rows])
            ),
            "delta_trimap_iou": float(
                np.mean([float(row.get("delta_trimap_iou", 0.0)) for row in rows])
            ),
            "sam3_accept_rate": float(
                np.mean([1.0 if bool(row.get("sam3_accepted", False)) else 0.0 for row in rows])
            ),
        }
    return summary


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


def compute_selection_ranking_scores(scores: torch.Tensor, *, mode: str) -> torch.Tensor:
    """Return the score surface used by a GT-free primitive selector.

    ``score_margin`` and ``score_ratio`` suppress ambiguous primitives by
    comparing each query score against the strongest competing text query for
    the same Gaussian. ``entropy_score`` additionally downweights primitives
    whose scene-softmax distribution is high entropy.
    """
    if scores.ndim != 2:
        raise ValueError(f"Expected score matrix [N,K], got {tuple(scores.shape)}")
    scores_f = scores.float()
    if mode in {"top_ratio", "score_threshold", "mean_std"}:
        return scores_f
    if scores_f.shape[1] <= 1:
        return scores_f
    if mode in {"score_margin", "score_ratio"}:
        top2 = torch.topk(scores_f, k=2, dim=1, largest=True).values
        top1 = top2[:, :1]
        top2_val = top2[:, 1:2]
        competitors = torch.where(scores_f == top1, top2_val, top1)
        if mode == "score_margin":
            return scores_f - competitors
        return scores_f / competitors.clamp_min(1e-8)
    if mode == "entropy_score":
        probs = scores_f.clamp_min(1e-8)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=1, keepdim=True)
        max_entropy = math.log(max(int(scores_f.shape[1]), 2))
        confidence = (1.0 - entropy / max_entropy).clamp_min(0.0)
        return scores_f * confidence
    raise ValueError(f"Unsupported selection ranking mode: {mode}")


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

    scores_for_selection = compute_selection_ranking_scores(scores, mode=spec.mode)
    selected = torch.zeros_like(scores_for_selection, dtype=torch.float32)
    if spec.mode == "top_ratio":
        ratio = min(max(float(spec.value), 0.0), 1.0)
        k = max(int(round(n_gaussians * ratio)), int(min_select))
        k = min(k, n_gaussians)
        if k <= 0:
            return selected
        _, idx = torch.topk(scores_for_selection.float(), k=k, dim=0, largest=True)
        selected.scatter_(0, idx, 1.0)
        return selected

    if spec.mode in {"score_threshold", "score_margin", "score_ratio", "entropy_score"}:
        return (scores_for_selection.float() > float(spec.value)).float()

    if spec.mode == "mean_std":
        mean = scores_for_selection.float().mean(dim=0, keepdim=True)
        std = scores_for_selection.float().std(dim=0, keepdim=True, unbiased=False)
        return (scores_for_selection.float() > mean + float(spec.value) * std).float()

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


def select_gaussians_by_proposal_components(
    scores: torch.Tensor,
    xyz: torch.Tensor,
    *,
    support_ratio: float,
    resolution: int,
    keep_components: int,
    min_component_size: int,
    rank_by: str,
) -> torch.Tensor:
    """Select object-like connected proposals from a broad score support set.

    Unlike seed expansion, this branch does not start from a strict primitive
    mask. It first builds a GT-free support pool from text scores, groups that
    support into 3D connected components, and then selects the highest-ranked
    components as object proposals. This is the formal OPR-style readout used
    for direct 3D object selection.
    """
    if rank_by not in {"mean_score", "score_sum", "size"}:
        raise ValueError(f"Unsupported component rank_by: {rank_by}")
    if scores.ndim != 2:
        raise ValueError(f"Expected scores [N,K], got {tuple(scores.shape)}")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Expected xyz [N,3] aligned with scores [N,K], got "
            f"{tuple(xyz.shape)} and {tuple(scores.shape)}"
        )
    if resolution <= 1 or support_ratio <= 0:
        return torch.zeros_like(scores, dtype=torch.float32)

    try:
        from scipy import ndimage
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("proposal component selection requires scipy") from exc

    support = select_gaussians_from_scores(
        scores,
        SelectionSpec("top_ratio", float(support_ratio)),
        min_select=1,
    )
    device = scores.device
    support_cpu = support.detach().float().cpu()
    scores_cpu = scores.detach().float().cpu()
    xyz_cpu = xyz.detach().float().cpu()
    lo = xyz_cpu.min(dim=0).values
    hi = xyz_cpu.max(dim=0).values
    extent = (hi - lo).clamp_min(1e-6)
    coords = ((xyz_cpu - lo) / extent * float(resolution)).floor().long()
    coords = coords.clamp_(0, resolution - 1).numpy()

    selected = torch.zeros_like(scores_cpu)
    structure = ndimage.generate_binary_structure(3, 1)
    grid_shape = (int(resolution), int(resolution), int(resolution))
    keep_count = max(int(keep_components), 1)
    min_size = max(int(min_component_size), 1)

    for query_idx in range(scores_cpu.shape[1]):
        support_idx = torch.nonzero(support_cpu[:, query_idx] > 0, as_tuple=False).flatten()
        if support_idx.numel() == 0:
            continue

        support_np = support_idx.numpy()
        support_coords = coords[support_np]
        occupancy = np.zeros(grid_shape, dtype=bool)
        occupancy[support_coords[:, 0], support_coords[:, 1], support_coords[:, 2]] = True
        labels, num_components = ndimage.label(occupancy, structure=structure)
        if num_components <= 0:
            continue

        component_labels = labels[
            support_coords[:, 0],
            support_coords[:, 1],
            support_coords[:, 2],
        ].astype(np.int64)
        sizes = np.bincount(component_labels, minlength=num_components + 1).astype(np.float64)
        score_values = scores_cpu[support_idx, query_idx].numpy().astype(np.float64)
        score_sums = np.bincount(
            component_labels,
            weights=score_values,
            minlength=num_components + 1,
        )
        valid_labels = np.arange(1, num_components + 1, dtype=np.int64)
        valid_labels = valid_labels[sizes[valid_labels] >= min_size]
        if valid_labels.size == 0:
            valid_labels = np.arange(1, num_components + 1, dtype=np.int64)

        if rank_by == "size":
            ranks = sizes[valid_labels]
        elif rank_by == "mean_score":
            ranks = score_sums[valid_labels] / np.maximum(sizes[valid_labels], 1.0)
        else:
            ranks = score_sums[valid_labels]
        order = np.argsort(-ranks, kind="stable")
        kept_labels = set(int(label) for label in valid_labels[order[:keep_count]])
        keep_mask = np.array([int(label) in kept_labels for label in component_labels], dtype=bool)
        if keep_mask.any():
            selected[support_idx[torch.from_numpy(keep_mask)], query_idx] = 1.0

    return selected.to(device=device, dtype=torch.float32)


def smooth_scores_with_voxel_proposals(
    scores: torch.Tensor,
    xyz: torch.Tensor,
    *,
    voxel_size: float,
    alpha: float,
    min_count: int = 2,
    gate: str = "all",
    margin_threshold: float = 0.0,
    confidence_threshold: float = 0.0,
    proposal_consensus_threshold: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Blend primitive text scores with scores pooled over 3D voxel proposals."""
    if scores.ndim != 2:
        raise ValueError(f"Expected scores [N,K], got {tuple(scores.shape)}")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Expected xyz [N,3] aligned with scores [N,K], got "
            f"{tuple(xyz.shape)} and {tuple(scores.shape)}"
        )
    if scores.shape[0] == 0 or alpha <= 0.0:
        return scores, {
            "enabled": False,
            "mode": "voxel",
            "voxel_size": float(voxel_size),
            "alpha": float(alpha),
            "min_count": int(min_count),
            "gate": gate,
            "margin_threshold": float(margin_threshold),
            "confidence_threshold": float(confidence_threshold),
            "proposal_consensus_threshold": float(proposal_consensus_threshold),
            "num_proposals": 0,
            "num_assigned": 0,
        }

    labels = build_voxel_proposal_labels(
        xyz.detach().float().cpu(),
        voxel_size=voxel_size,
    ).to(scores.device)
    smoothed, stats = propagate_logits_with_proposals(
        scores,
        labels,
        alpha=alpha,
        min_count=min_count,
        gate=gate,
        margin_threshold=margin_threshold,
        confidence_threshold=confidence_threshold,
        proposal_consensus_threshold=proposal_consensus_threshold,
    )
    stats = dict(stats)
    stats.update(
        {
            "mode": "voxel",
            "voxel_size": float(voxel_size),
        }
    )
    return smoothed, stats


def _sam3_cache_frame_id(path: str | Path) -> int:
    match = re.search(r"frame_(\d+)", Path(path).stem)
    if not match:
        raise ValueError(f"Cannot parse SAM3 cache frame id from {path}")
    return int(match.group(1))


def _resolve_sam3_proposal_cache_paths(cache_root: str | Path, scene: str) -> list[Path]:
    root = Path(cache_root)
    if not root.exists():
        return []
    scene_root = root / scene
    base = scene_root if scene_root.exists() else root
    return sorted(base.glob("frame_*.pt"), key=_sam3_cache_frame_id)


def _camera_intrinsics_from_dataset(
    dataset: LERFDataset,
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    params = getattr(dataset, "camera_params", {}) or {}
    width = float(params.get("w", image_width) or image_width)
    height = float(params.get("h", image_height) or image_height)
    fx = params.get("fl_x")
    fy = params.get("fl_y")
    if fx is None:
        angle_x = float(params.get("camera_angle_x", 0.0) or 0.0)
        fx = 0.5 * width / math.tan(0.5 * angle_x) if angle_x > 0 else width
    if fy is None:
        fy = fx
    cx = float(params.get("cx", width * 0.5))
    cy = float(params.get("cy", height * 0.5))
    return float(fx), float(fy), cx, cy


def _project_points_to_image(
    xyz: torch.Tensor,
    pose_w2c: torch.Tensor,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world-space points to image pixels using a LERF world-to-camera pose."""
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N,3], got {tuple(xyz.shape)}")
    if pose_w2c.shape != (4, 4):
        raise ValueError(f"pose_w2c must have shape [4,4], got {tuple(pose_w2c.shape)}")
    device = xyz.device
    pose = pose_w2c.to(device=device, dtype=torch.float32)
    ones = torch.ones((xyz.shape[0], 1), dtype=torch.float32, device=device)
    xyz_h = torch.cat([xyz.float(), ones], dim=-1)
    cam = xyz_h @ pose.t()
    z = cam[:, 2]
    x = float(fx) * cam[:, 0] / z.clamp_min(1e-6) + float(cx)
    y = float(fy) * cam[:, 1] / z.clamp_min(1e-6) + float(cy)
    pixels = torch.stack([x, y], dim=-1)
    visible = (
        torch.isfinite(pixels).all(dim=-1)
        & (z > 1e-6)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= int(image_width) - 1)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= int(image_height) - 1)
    )
    return pixels, visible.float()


def smooth_scores_with_sam3_training_view_proposals(
    scores: torch.Tensor,
    xyz: torch.Tensor,
    *,
    cache_root: str | Path,
    scene: str,
    scene_categories: Sequence[str],
    dataset: LERFDataset,
    image_width: int,
    image_height: int,
    alpha: float,
    min_probability: float,
    max_masks_per_frame: int = 0,
    gate: str = "low_margin",
    margin_threshold: float = 0.05,
    query_conditioned: bool = False,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Fuse primitive scores with official SAM3 training-view proposal memory."""
    stats: Dict[str, Any] = {
        "enabled": False,
        "mode": "sam3_trainview",
        "cache_root": str(cache_root),
        "alpha": float(alpha),
        "min_probability": float(min_probability),
        "max_masks_per_frame": int(max_masks_per_frame),
        "gate": gate,
        "margin_threshold": float(margin_threshold),
        "query_conditioned": bool(query_conditioned),
        "num_cache_frames": 0,
        "num_used_frames": 0,
        "num_proposals": 0,
        "num_memberships": 0,
        "num_assigned": 0,
        "skipped_frames": 0,
    }
    if scores.ndim != 2:
        raise ValueError(f"Expected scores [N,K], got {tuple(scores.shape)}")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Expected xyz [N,3] aligned with scores [N,K], got "
            f"{tuple(xyz.shape)} and {tuple(scores.shape)}"
        )
    if not cache_root or float(alpha) <= 0.0:
        return scores, stats

    cache_paths = _resolve_sam3_proposal_cache_paths(cache_root, scene)
    stats["num_cache_frames"] = len(cache_paths)
    if not cache_paths:
        return scores, stats

    fx, fy, cx, cy = _camera_intrinsics_from_dataset(
        dataset,
        image_width=image_width,
        image_height=image_height,
    )
    xyz_cpu = xyz.detach().float().cpu()
    row_chunks: list[torch.Tensor] = []
    proposal_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    proposal_query_chunks: list[torch.Tensor] = []
    proposal_offset = 0
    used_frame_ids: list[int] = []
    scene_query_lookup = {
        str(query).strip().lower(): idx
        for idx, query in enumerate(scene_categories)
    }

    for cache_path in cache_paths:
        frame_id = _sam3_cache_frame_id(cache_path)
        pose_np = getattr(dataset, "pose_by_frame_idx", {}).get(frame_id)
        if pose_np is None and 0 <= frame_id < len(getattr(dataset, "poses_w2c", [])):
            pose_np = dataset.poses_w2c[frame_id]
        if pose_np is None and 1 <= frame_id <= len(getattr(dataset, "poses_w2c", [])):
            pose_np = dataset.poses_w2c[frame_id - 1]
        if pose_np is None:
            stats["skipped_frames"] = int(stats["skipped_frames"]) + 1
            continue

        cache = load_foundation_cache(cache_path, require_official=True)
        head = cache.heads.get("sam3")
        if head is None or head.mask_logits is None or head.mask_logits.shape[0] == 0:
            stats["skipped_frames"] = int(stats["skipped_frames"]) + 1
            continue

        mask_logits = head.mask_logits.detach().float().cpu()
        mask_query_indices: torch.Tensor | None = None
        if query_conditioned:
            if head.mask_query_indices is None or not head.queries:
                stats["skipped_frames"] = int(stats["skipped_frames"]) + 1
                continue
            cache_to_scene = torch.full(
                (len(head.queries),),
                -1,
                dtype=torch.long,
            )
            for cache_idx, query in enumerate(head.queries):
                cache_to_scene[cache_idx] = scene_query_lookup.get(
                    str(query).strip().lower(),
                    -1,
                )
            local_query = head.mask_query_indices.detach().long().cpu()
            valid_local = (local_query >= 0) & (local_query < cache_to_scene.numel())
            mask_query_indices = torch.full_like(local_query, -1)
            mask_query_indices[valid_local] = cache_to_scene[local_query[valid_local]]
        mask_h, mask_w = int(mask_logits.shape[-2]), int(mask_logits.shape[-1])
        pixels, visibility = _project_points_to_image(
            xyz_cpu,
            torch.as_tensor(pose_np, dtype=torch.float32),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            image_width=image_width,
            image_height=image_height,
        )
        pixels_mask = pixels.clone()
        pixels_mask[:, 0] *= (mask_w - 1) / max(int(image_width) - 1, 1)
        pixels_mask[:, 1] *= (mask_h - 1) / max(int(image_height) - 1, 1)
        memberships = build_sam3_mask_memberships(
            mask_logits,
            pixels_mask,
            scores=head.scores.detach().float().cpu() if head.scores is not None else None,
            mask_query_indices=mask_query_indices,
            visibility=visibility,
            min_probability=min_probability,
            max_masks=max_masks_per_frame if max_masks_per_frame > 0 else None,
            proposal_offset=proposal_offset,
        )
        proposal_offset += memberships.num_proposals
        if memberships.row_indices.numel() == 0:
            continue
        row_chunks.append(memberships.row_indices)
        proposal_chunks.append(memberships.proposal_indices)
        weight_chunks.append(memberships.weights)
        if query_conditioned:
            if memberships.proposal_query_indices is None:
                raise RuntimeError("query-conditioned SAM3 proposal registration missing query ids")
            proposal_query_chunks.append(memberships.proposal_query_indices)
        used_frame_ids.append(frame_id)

    stats["num_used_frames"] = len(used_frame_ids)
    stats["used_frame_ids"] = used_frame_ids
    stats["num_proposals"] = int(proposal_offset)
    if not row_chunks:
        return scores, stats

    if query_conditioned:
        fused, fusion_stats = fuse_scores_with_query_sam3_proposals(
            scores,
            torch.cat(row_chunks).to(scores.device),
            torch.cat(proposal_chunks).to(scores.device),
            torch.cat(weight_chunks).to(scores.device),
            torch.cat(proposal_query_chunks).to(scores.device),
            alpha=alpha,
            gate=gate,  # type: ignore[arg-type]
            margin_threshold=margin_threshold,
        )
    else:
        fused, fusion_stats = fuse_scores_with_sam3_proposals(
            scores,
            torch.cat(row_chunks).to(scores.device),
            torch.cat(proposal_chunks).to(scores.device),
            torch.cat(weight_chunks).to(scores.device),
            alpha=alpha,
            gate=gate,  # type: ignore[arg-type]
            margin_threshold=margin_threshold,
        )
    stats.update(fusion_stats)
    stats["mode"] = "sam3_trainview"
    stats["cache_root"] = str(cache_root)
    stats["num_cache_frames"] = len(cache_paths)
    stats["num_used_frames"] = len(used_frame_ids)
    stats["used_frame_ids"] = used_frame_ids
    stats["skipped_frames"] = int(stats.get("skipped_frames", 0))
    return fused, stats


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


def build_opacity_primitive_confidence(
    opacities: torch.Tensor,
    *,
    mode: str,
    threshold: float,
) -> torch.Tensor:
    """Map per-Gaussian opacity to a GT-free primitive confidence in [0, 1]."""
    if mode not in {"none", "opacity", "opacity_log"}:
        raise ValueError(f"Unsupported direct primitive confidence mode: {mode}")
    opacity = opacities.detach().float()
    if opacity.dim() == 2:
        if opacity.shape[1] == 1:
            opacity = opacity[:, 0]
        elif opacity.shape[0] == 1:
            opacity = opacity[0]
    opacity = opacity.reshape(-1).clamp_min(0.0)
    if mode == "none":
        return torch.ones_like(opacity)

    threshold_f = max(float(threshold), 0.0)
    if mode == "opacity":
        confidence = opacity
    else:
        confidence = torch.log1p(opacity)
    if threshold_f > 0:
        confidence = torch.where(opacity >= threshold_f, confidence, torch.zeros_like(confidence))
    return (confidence / confidence.max().clamp_min(1e-8)).clamp(0.0, 1.0)


def apply_direct_primitive_confidence(
    scores: torch.Tensor,
    confidence: Optional[torch.Tensor],
    *,
    blend: float,
) -> torch.Tensor:
    """Downweight low-confidence direct primitive rows before threshold selection."""
    if confidence is None:
        return scores
    blend_f = min(max(float(blend), 0.0), 1.0)
    if blend_f <= 0.0:
        return scores
    if scores.ndim != 2:
        raise ValueError(f"Expected scores [N,K], got {tuple(scores.shape)}")
    conf = confidence.to(device=scores.device, dtype=scores.dtype).reshape(-1)
    if conf.shape != (scores.shape[0],):
        raise ValueError(
            f"Expected direct primitive confidence [{scores.shape[0]}], got {tuple(conf.shape)}"
        )
    scale = (1.0 - blend_f) + blend_f * conf.clamp(0.0, 1.0)
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


def _normalize_single_view_image(
    image: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Return a single-channel image as [H,W], resized if needed."""
    tensor = image.float()
    if tensor.dim() == 4 and tensor.shape[0] == 1 and tensor.shape[1] == 1:
        tensor = tensor[0, 0]
    elif tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    elif tensor.dim() == 2:
        pass
    else:
        raise ValueError(f"Expected single-view image, got {tuple(image.shape)}")
    if tuple(tensor.shape) != (int(image_height), int(image_width)):
        tensor = F.interpolate(
            tensor[None, None],
            size=(int(image_height), int(image_width)),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
    return tensor


def compute_raster_contribution_weights(
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    depths: Optional[torch.Tensor] = None,
    depth_map: Optional[torch.Tensor] = None,
    alpha_map: Optional[torch.Tensor] = None,
    depth_tolerance: float = 0.08,
    relative_depth_tolerance: float = 0.02,
    alpha_threshold: float = 0.0,
    mode: str = "uniform",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Approximate per-Gaussian raster contribution weights for VPR.

    The input indices are rasterizer-level Gaussian/pixel intersections from
    gsplat. ``alpha`` uses the projected Gaussian footprint and opacity as the
    contribution weight; ``alpha_depth`` additionally gates non-surface hits by
    the rendered expected-depth map. This replaces the earlier center-only VPR
    assignment path while keeping the protocol GT-free.
    """
    if mode not in {"uniform", "alpha", "alpha_depth"}:
        raise ValueError(f"Unsupported registration weight mode: {mode}")
    if gaussian_ids.ndim != 1 or pixel_ids.ndim != 1 or gaussian_ids.shape != pixel_ids.shape:
        raise ValueError(
            f"Expected gaussian_ids/pixel_ids [M], got "
            f"{tuple(gaussian_ids.shape)} and {tuple(pixel_ids.shape)}"
        )
    device = gaussian_ids.device
    n_hits = int(gaussian_ids.numel())
    if n_hits == 0:
        empty = torch.empty(0, dtype=torch.float32, device=device)
        return torch.empty(0, dtype=torch.bool, device=device), empty

    height = int(image_height)
    width = int(image_width)
    gids = gaussian_ids.long()
    pids = pixel_ids.long()

    means = means2d.float()
    if means.dim() == 3:
        if means.shape[0] != 1:
            raise ValueError("compute_raster_contribution_weights expects one view")
        means = means[0]
    conic = conics.float()
    if conic.dim() == 3:
        if conic.shape[0] != 1:
            raise ValueError("compute_raster_contribution_weights expects one view")
        conic = conic[0]
    opacity = opacities.float()
    if opacity.dim() == 2:
        if opacity.shape[0] == 1:
            opacity = opacity[0]
        elif opacity.shape[1] == 1:
            opacity = opacity[:, 0]
    opacity = opacity.reshape(-1)

    valid = (
        (gids >= 0)
        & (gids < means.shape[0])
        & (pids >= 0)
        & (pids < height * width)
    )
    x = (pids % width).to(dtype=torch.float32)
    y = torch.div(pids, width, rounding_mode="floor").to(dtype=torch.float32)
    mu = means[gids.clamp(0, means.shape[0] - 1)]
    q = conic[gids.clamp(0, conic.shape[0] - 1)]
    dx = x - mu[:, 0]
    dy = y - mu[:, 1]
    power = -0.5 * (q[:, 0] * dx.square() + 2.0 * q[:, 1] * dx * dy + q[:, 2] * dy.square())
    footprint_alpha = opacity[gids.clamp(0, opacity.shape[0] - 1)] * torch.exp(power.clamp(max=0.0))
    footprint_alpha = footprint_alpha.clamp(min=0.0, max=0.999)

    if mode == "uniform":
        weights = torch.ones(n_hits, dtype=torch.float32, device=device)
    else:
        weights = footprint_alpha.clamp_min(1e-8)

    sampled_alpha = None
    if alpha_map is not None:
        alpha = _normalize_single_view_image(
            alpha_map.to(device=device),
            image_height=height,
            image_width=width,
        )
        sampled_alpha = alpha.reshape(-1)[pids.clamp(0, height * width - 1)].clamp_min(0.0)
        if alpha_threshold > 0:
            valid = valid & (sampled_alpha >= float(alpha_threshold))

    if mode == "alpha_depth" and depth_map is not None and depths is not None:
        rendered_depth = _normalize_single_view_image(
            depth_map.to(device=device),
            image_height=height,
            image_width=width,
        )
        sampled_depth = rendered_depth.reshape(-1)[pids.clamp(0, height * width - 1)]
        gaussian_depths = depths.float()
        if gaussian_depths.dim() == 3 and gaussian_depths.shape[-1] == 1:
            gaussian_depths = gaussian_depths[..., 0]
        if gaussian_depths.dim() == 2:
            if gaussian_depths.shape[0] != 1:
                raise ValueError("compute_raster_contribution_weights expects one view")
            gaussian_depths = gaussian_depths[0]
        gaussian_depths = gaussian_depths.reshape(-1).to(device=device)
        z = gaussian_depths[gids.clamp(0, gaussian_depths.shape[0] - 1)]
        tolerance = torch.maximum(
            torch.full_like(z, float(depth_tolerance)),
            z.abs() * float(relative_depth_tolerance),
        ).clamp_min(1e-6)
        depth_error = (sampled_depth - z).abs()
        valid = valid & (sampled_depth > 0.0) & (depth_error <= tolerance)
        weights = weights * torch.exp(-depth_error / tolerance)

    weights = torch.where(valid, weights, torch.zeros_like(weights))
    return valid, weights.float()


def accumulate_raster_contribution_features(
    feature_map: torch.Tensor,
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    n_gaussians: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scatter weighted rendered-view features from raster hits to Gaussians."""
    if feature_map.dim() == 4:
        if feature_map.shape[0] != 1:
            raise ValueError("accumulate_raster_contribution_features expects one view")
        features = feature_map[0]
    elif feature_map.dim() == 3:
        features = feature_map
    else:
        raise ValueError(f"Expected feature map [C,H,W] or [1,C,H,W], got {tuple(feature_map.shape)}")
    if gaussian_ids.ndim != 1 or pixel_ids.ndim != 1 or weights.ndim != 1:
        raise ValueError("gaussian_ids, pixel_ids, and weights must be 1D")
    if gaussian_ids.shape != pixel_ids.shape or gaussian_ids.shape != weights.shape:
        raise ValueError("gaussian_ids, pixel_ids, and weights must have matching shapes")

    device = features.device
    channels, height, width = features.shape
    gids = gaussian_ids.to(device=device, dtype=torch.long)
    pids = pixel_ids.to(device=device, dtype=torch.long)
    w = weights.to(device=device, dtype=torch.float32)
    valid = (
        (gids >= 0)
        & (gids < int(n_gaussians))
        & (pids >= 0)
        & (pids < height * width)
        & (w > 0)
    )
    sums = torch.zeros(int(n_gaussians), channels, dtype=torch.float32, device=device)
    counts = torch.zeros(int(n_gaussians), dtype=torch.float32, device=device)
    if not bool(valid.any()):
        return sums, counts

    gids = gids[valid]
    pids = pids[valid]
    w = w[valid]
    flat = features.float().reshape(channels, height * width).t()
    sampled = flat[pids] * w.unsqueeze(1)
    sums.index_add_(0, gids, sampled)
    counts.index_add_(0, gids, w)
    return sums, counts


def normalize_registered_feature_sums(
    registered_sum: torch.Tensor,
    registered_counts: torch.Tensor,
) -> torch.Tensor:
    """Return L2-normalized weighted primitive features from accumulated VPR sums."""
    sums = torch.as_tensor(registered_sum).float()
    counts = torch.as_tensor(registered_counts).float().to(device=sums.device)
    if sums.ndim != 2 or counts.ndim != 1 or int(sums.shape[0]) != int(counts.shape[0]):
        raise ValueError(
            "registered_sum must be [N,D] and registered_counts must be [N] with matching N"
        )
    normalized = torch.zeros_like(sums)
    valid = counts > 0
    if bool(valid.any()):
        mean = sums[valid] / counts[valid].clamp_min(1e-8).unsqueeze(1)
        normalized[valid] = F.normalize(mean, dim=-1)
    return normalized


def select_dominant_raster_hits(
    pixel_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_pixels: int,
) -> torch.Tensor:
    """Keep the strongest Gaussian hit for each rendered pixel."""
    if pixel_ids.ndim != 1 or weights.ndim != 1 or pixel_ids.shape != weights.shape:
        raise ValueError("pixel_ids and weights must be matching 1D tensors")
    device = pixel_ids.device
    if pixel_ids.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=device)
    pids = pixel_ids.long()
    w = weights.float()
    valid = (pids >= 0) & (pids < int(num_pixels)) & (w > 0)
    max_weights = torch.full(
        (int(num_pixels),),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    if bool(valid.any()):
        max_weights.scatter_reduce_(
            0,
            pids[valid],
            w[valid],
            reduce="amax",
            include_self=True,
        )
    return valid & (w >= max_weights[pids.clamp(0, int(num_pixels) - 1)] - 1e-8)


def select_top_raster_hits_per_gaussian(
    gaussian_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    n_gaussians: int,
) -> torch.Tensor:
    """Keep the strongest raster hit for each Gaussian footprint."""
    if gaussian_ids.ndim != 1 or weights.ndim != 1 or gaussian_ids.shape != weights.shape:
        raise ValueError("gaussian_ids and weights must be matching 1D tensors")
    device = gaussian_ids.device
    if gaussian_ids.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=device)
    gids = gaussian_ids.long()
    w = weights.float()
    valid = (gids >= 0) & (gids < int(n_gaussians)) & (w > 0)
    max_weights = torch.full(
        (int(n_gaussians),),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    if bool(valid.any()):
        max_weights.scatter_reduce_(
            0,
            gids[valid],
            w[valid],
            reduce="amax",
            include_self=True,
        )
    return valid & (w >= max_weights[gids.clamp(0, int(n_gaussians) - 1)] - 1e-8)


@torch.no_grad()
def rasterize_registered_view_features(
    *,
    model: torch.nn.Module,
    renderer: FeatureFieldRenderer,
    viewmat: torch.Tensor,
    siglip_feat: torch.Tensor,
    depth_map: torch.Tensor,
    alpha_map: torch.Tensor,
    registration_depth_tolerance: float,
    registration_relative_depth_tolerance: float,
    registration_alpha_threshold: float,
    registration_weight_mode: str,
    dominant_only: bool = False,
    gaussian_top1: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Register a rendered feature map to primitives using rasterizer hits."""
    if getattr(renderer, "use_2dgs", False):
        raise RuntimeError("raster_contrib registration currently supports 3DGS rasterization only")

    from gsplat import rasterization
    from gsplat.cuda._wrapper import rasterize_to_indices_in_range

    device = siglip_feat.device
    if viewmat.dim() == 2:
        viewmats = viewmat.to(device=device, dtype=torch.float32).unsqueeze(0)
    elif viewmat.dim() == 3 and viewmat.shape[0] == 1:
        viewmats = viewmat.to(device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Expected single view matrix, got {tuple(viewmat.shape)}")

    n_gaussians = int(model.get_xyz().shape[0])
    if n_gaussians == 0:
        return (
            torch.empty(0, int(siglip_feat.shape[1]), dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

    height = int(siglip_feat.shape[-2])
    width = int(siglip_feat.shape[-1])
    means = model.get_xyz().to(device=device, dtype=torch.float32)
    quats = model.get_rotation().to(device=device, dtype=torch.float32)
    scales = model.get_scaling().to(device=device, dtype=torch.float32)
    opacities = model.get_opacity().to(device=device, dtype=torch.float32)
    if opacities.dim() == 2 and opacities.shape[1] == 1:
        opacities = opacities[:, 0]
    colors = torch.zeros(n_gaussians, 3, dtype=torch.float32, device=device)
    backgrounds = torch.zeros(1, 3, dtype=torch.float32, device=device)
    Ks = renderer.K.to(device=device, dtype=torch.float32).unsqueeze(0)

    _renders, _alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=renderer.near_plane,
        far_plane=renderer.far_plane,
        backgrounds=backgrounds,
        render_mode="RGB+ED",
        packed=False,
    )
    total_intersections = int(info.get("flatten_ids", torch.empty(0, device=device)).numel())
    if total_intersections <= 0:
        return (
            torch.zeros(n_gaussians, int(siglip_feat.shape[1]), dtype=torch.float32, device=device),
            torch.zeros(n_gaussians, dtype=torch.float32, device=device),
        )

    transmittances = torch.ones(1, height, width, dtype=torch.float32, device=device)
    gaussian_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range(
        0,
        total_intersections,
        transmittances,
        info["means2d"],
        info["conics"],
        info["opacities"],
        width,
        height,
        info["tile_size"],
        info["isect_offsets"],
        info["flatten_ids"],
    )
    if camera_ids.numel() > 0:
        keep = camera_ids == 0
        gaussian_ids = gaussian_ids[keep]
        pixel_ids = pixel_ids[keep]

    valid, weights = compute_raster_contribution_weights(
        gaussian_ids,
        pixel_ids,
        info["means2d"],
        info["conics"],
        info["opacities"],
        image_height=height,
        image_width=width,
        depths=info.get("depths"),
        depth_map=depth_map,
        alpha_map=alpha_map,
        depth_tolerance=registration_depth_tolerance,
        relative_depth_tolerance=registration_relative_depth_tolerance,
        alpha_threshold=registration_alpha_threshold,
        mode=registration_weight_mode,
    )
    if dominant_only:
        dominant = select_dominant_raster_hits(
            pixel_ids,
            weights,
            num_pixels=height * width,
        )
        valid = valid & dominant
    if gaussian_top1:
        top1 = select_top_raster_hits_per_gaussian(
            gaussian_ids,
            weights,
            n_gaussians=n_gaussians,
        )
        valid = valid & top1
    sums, counts = accumulate_raster_contribution_features(
        siglip_feat,
        gaussian_ids[valid],
        pixel_ids[valid],
        weights[valid],
        n_gaussians=n_gaussians,
    )
    return sums, counts


def raster_adjoint_registered_view_features(
    *,
    model: torch.nn.Module,
    renderer: FeatureFieldRenderer,
    viewmat: torch.Tensor,
    siglip_feat: torch.Tensor,
    alpha_map: Optional[torch.Tensor] = None,
    alpha_threshold: float = 0.0,
    channel_chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lift rendered features to primitives with the rasterizer's color adjoint.

    For each rendered pixel feature ``F(u)``, the gradient of
    ``sum_u rendered_color(u) * F(u)`` with respect to per-Gaussian colors is the
    compositing-weighted primitive contribution ``sum_u w_ui F(u)``. A second
    one-channel adjoint pass yields ``sum_u w_ui`` for normalization. This keeps
    the VPR registration label-free while using true alpha-compositing
    contributions instead of center or footprint proxy assignments.
    """
    if getattr(renderer, "use_2dgs", False):
        raise RuntimeError("raster_adjoint registration currently supports 3DGS rasterization only")
    if siglip_feat.dim() != 4 or int(siglip_feat.shape[0]) != 1:
        raise ValueError(f"Expected siglip_feat [1,C,H,W], got {tuple(siglip_feat.shape)}")

    from gsplat import rasterization

    device = siglip_feat.device
    target = siglip_feat.detach().float()
    _, channels, height, width = target.shape
    n_gaussians = int(model.get_xyz().shape[0])
    if n_gaussians == 0:
        return (
            torch.empty(0, int(channels), dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

    if viewmat.dim() == 2:
        viewmats = viewmat.to(device=device, dtype=torch.float32).unsqueeze(0)
    elif viewmat.dim() == 3 and int(viewmat.shape[0]) == 1:
        viewmats = viewmat.to(device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Expected single view matrix, got {tuple(viewmat.shape)}")

    means = model.get_xyz().detach().to(device=device, dtype=torch.float32)
    quats = model.get_rotation().detach().to(device=device, dtype=torch.float32)
    scales = model.get_scaling().detach().to(device=device, dtype=torch.float32)
    opacities = model.get_opacity().detach().to(device=device, dtype=torch.float32)
    if opacities.dim() == 2 and int(opacities.shape[1]) == 1:
        opacities = opacities[:, 0]
    Ks = renderer.K.detach().to(device=device, dtype=torch.float32).unsqueeze(0)

    pixel_weight = None
    if alpha_threshold > 0.0 and alpha_map is not None:
        alpha = _normalize_single_view_image(
            alpha_map.to(device=device),
            image_height=int(height),
            image_width=int(width),
        )
        pixel_weight = (alpha >= float(alpha_threshold)).to(dtype=torch.float32)

    def _render_color_adjoint(color_dim: int, pixel_target: torch.Tensor) -> torch.Tensor:
        colors = torch.zeros(
            n_gaussians,
            int(color_dim),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        backgrounds = torch.zeros(1, int(color_dim), dtype=torch.float32, device=device)
        renders, _alphas, _info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=int(width),
            height=int(height),
            near_plane=renderer.near_plane,
            far_plane=renderer.far_plane,
            backgrounds=backgrounds,
            render_mode="RGB",
            packed=False,
        )
        loss = (renders[0].float() * pixel_target.float()).sum()
        grad = torch.autograd.grad(loss, colors, retain_graph=False, create_graph=False)[0]
        return grad.detach().float()

    chunk = int(channel_chunk_size or getattr(renderer, "max_channels_per_chunk", 32) or 32)
    chunk = max(1, chunk)
    sums = torch.zeros(n_gaussians, int(channels), dtype=torch.float32, device=device)

    with torch.enable_grad():
        denom_target = torch.ones(int(height), int(width), 1, dtype=torch.float32, device=device)
        if pixel_weight is not None:
            denom_target = denom_target * pixel_weight.unsqueeze(-1)
        counts = _render_color_adjoint(1, denom_target).squeeze(-1).clamp_min(0.0)

        for start in range(0, int(channels), chunk):
            end = min(start + chunk, int(channels))
            chunk_target = target[0, start:end].permute(1, 2, 0).contiguous()
            if pixel_weight is not None:
                chunk_target = chunk_target * pixel_weight.unsqueeze(-1)
            sums[:, start:end] = _render_color_adjoint(end - start, chunk_target)

    invalid = counts <= 0
    if bool(invalid.any()):
        sums[invalid] = 0.0
    return sums, counts


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


def _build_point_summary_adapter(
    config: object,
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    ckpt = load_trusted_checkpoint(checkpoint_path, map_location=device)
    metadata = ckpt.get("point_summary_adapter_metadata") or {}
    contract = metadata.get("direct_head_contract") if isinstance(metadata, dict) else {}
    context_features = str(
        (contract or {}).get(
            "point_summary_adapter_context_features",
            metadata.get(
                "point_summary_adapter_context_features",
                getattr(config, "point_summary_adapter_context_features", ""),
            ),
        )
        or ""
    )
    adapter = CompactToSummaryAdapter(
        input_dim=getattr(config, "bottleneck_dim", getattr(config, "hybrid_output_dim", 128))
        + point_summary_context_dim(context_features),
        output_dim=1536,
        hidden_dim=getattr(config, "point_summary_adapter_hidden_dim", 512),
        num_layers=getattr(config, "point_summary_adapter_num_layers", 2),
        dropout=getattr(config, "point_summary_adapter_dropout", 0.0),
    ).to(device)
    state = ckpt.get("point_summary_adapter_state_dict")
    if state is None:
        raise KeyError(
            "Checkpoint has no point_summary_adapter_state_dict; train a "
            "VPR-to-primitive point summary adapter first"
        )
    adapter.load_state_dict(state, strict=True)
    return adapter.eval()


def build_direct_head_eval_status(
    checkpoint: Dict[str, Any],
    *,
    score_source: str,
    use_point_summary_adapter: bool,
    adapter_loaded: bool,
    compact_feature_key: str = "features",
    direct_readout_mode: str = "gaussian",
    point_summary_adapter_blend_alpha: float = 1.0,
    point_summary_adapter_valid_mask_mode: str = "none",
    point_summary_adapter_context_features: str = "",
    teacher_feature_space: str = "",
    teacher_cache_feature_key: str = "",
    direct_point_query_mode: str = "",
    direct_point_gaussian_position_mode: str = "",
) -> Dict[str, Any]:
    """Describe direct-head eval settings that can silently change direct-field rows."""
    has_adapter = "point_summary_adapter_state_dict" in checkpoint
    metadata = checkpoint.get("point_summary_adapter_metadata") or {}
    contract = metadata.get("direct_head_contract") if isinstance(metadata, dict) else None
    if not isinstance(contract, dict):
        contract = {}
    warnings: list[str] = []
    if score_source == "direct":
        if has_adapter and not use_point_summary_adapter:
            warnings.append("checkpoint_has_point_summary_adapter_but_eval_disabled")
        if use_point_summary_adapter and not adapter_loaded:
            warnings.append("use_point_summary_adapter_true_but_adapter_not_loaded")
        if has_adapter and use_point_summary_adapter and not contract:
            warnings.append("checkpoint_has_point_summary_adapter_but_missing_direct_head_contract")
        if contract:
            expected_feature_key = str(contract.get("compact_feature_key", "features"))
            if str(compact_feature_key) != expected_feature_key:
                warnings.append(
                    f"compact_feature_key_mismatch: expected={expected_feature_key} "
                    f"got={compact_feature_key}"
                )
            expected_readout = str(contract.get("direct_readout_mode", "gaussian"))
            if str(direct_readout_mode) != expected_readout:
                warnings.append(
                    f"direct_readout_mode_mismatch: expected={expected_readout} "
                    f"got={direct_readout_mode}"
                )
            expected_alpha = float(contract.get("point_summary_adapter_blend_alpha", 1.0))
            if abs(float(point_summary_adapter_blend_alpha) - expected_alpha) > 1e-6:
                warnings.append(
                    "point_summary_adapter_blend_alpha_mismatch: "
                    f"expected={expected_alpha:g} got={float(point_summary_adapter_blend_alpha):g}"
                )
            expected_mask = str(contract.get("point_summary_adapter_valid_mask_mode", "none"))
            if str(point_summary_adapter_valid_mask_mode) != expected_mask:
                warnings.append(
                    f"point_summary_adapter_valid_mask_mode_mismatch: "
                    f"expected={expected_mask} got={point_summary_adapter_valid_mask_mode}"
                )
            expected_context = str(contract.get("point_summary_adapter_context_features", ""))
            if str(point_summary_adapter_context_features or "") != expected_context:
                warnings.append(
                    "point_summary_adapter_context_features_mismatch: "
                    f"expected={expected_context!r} got={point_summary_adapter_context_features!r}"
                )
            contract_checks = {
                "teacher_feature_space": str(teacher_feature_space or ""),
                "teacher_cache_feature_key": str(teacher_cache_feature_key or ""),
                "direct_point_query_mode": str(direct_point_query_mode or ""),
                "direct_point_gaussian_position_mode": str(direct_point_gaussian_position_mode or ""),
            }
            for name, actual in contract_checks.items():
                expected = str(contract.get(name, ""))
                if expected and actual and actual != expected:
                    warnings.append(f"{name}_mismatch: expected={expected} got={actual}")
    return {
        "score_source": score_source,
        "checkpoint_has_point_summary_adapter": bool(has_adapter),
        "use_point_summary_adapter": bool(use_point_summary_adapter),
        "adapter_loaded": bool(adapter_loaded),
        "compact_feature_key": str(compact_feature_key),
        "direct_readout_mode": str(direct_readout_mode),
        "point_summary_adapter_blend_alpha": float(point_summary_adapter_blend_alpha),
        "point_summary_adapter_valid_mask_mode": str(point_summary_adapter_valid_mask_mode),
        "point_summary_adapter_context_features": str(point_summary_adapter_context_features or ""),
        "teacher_feature_space": str(teacher_feature_space or ""),
        "teacher_cache_feature_key": str(teacher_cache_feature_key or ""),
        "direct_point_query_mode": str(direct_point_query_mode or ""),
        "direct_point_gaussian_position_mode": str(direct_point_gaussian_position_mode or ""),
        "point_summary_adapter_metadata": metadata,
        "direct_head_contract": contract,
        "warnings": warnings,
    }


def enforce_direct_head_eval_consistency(status: Dict[str, Any], *, strict: bool) -> None:
    warnings = list(status.get("warnings") or [])
    if strict and warnings:
        raise ValueError(
            "direct head consistency check failed: "
            + ", ".join(str(warning) for warning in warnings)
        )


def _validate_teacher_cache_geometry(
    payload: Dict[str, Any],
    *,
    expected_xyz: Optional[torch.Tensor],
    cache_path: Path,
) -> None:
    if expected_xyz is None:
        return
    fingerprint = payload.get("geometry_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError(f"teacher cache missing geometry_fingerprint: {cache_path}")
    expected_fingerprint = xyz_geometry_fingerprint(expected_xyz)
    cached_hash = str(fingerprint.get("xyz_sha256", ""))
    expected_hash = str(expected_fingerprint.get("xyz_sha256", ""))
    cached_count = int(fingerprint.get("num_gaussians", -1))
    expected_count = int(expected_fingerprint["num_gaussians"])
    if cached_hash != expected_hash or cached_count != expected_count:
        raise ValueError(
            "teacher cache geometry mismatch: "
            f"{cache_path} cached_count={cached_count} expected_count={expected_count} "
            f"cached_xyz_sha256={cached_hash} expected_xyz_sha256={expected_hash}"
        )


def _load_point_summary_adapter_valid_mask(
    checkpoint_path: str,
    *,
    expected_count: int,
    device: torch.device,
    fallback_teacher_cache: str = "",
    expected_xyz: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    ckpt = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    metadata = ckpt.get("point_summary_adapter_metadata") or {}
    teacher_cache = metadata.get("teacher_cache") or fallback_teacher_cache
    if not teacher_cache:
        return None
    cache_path = Path(str(teacher_cache))
    if not cache_path.exists():
        logger.warning("Point summary adapter teacher cache not found: %s", cache_path)
        return None
    payload = torch.load(cache_path, map_location="cpu")
    _validate_teacher_cache_geometry(payload, expected_xyz=expected_xyz, cache_path=cache_path)
    valid = payload.get("valid")
    if not isinstance(valid, torch.Tensor):
        logger.warning("Point summary adapter teacher cache has no valid mask: %s", cache_path)
        return None
    if int(valid.numel()) != int(expected_count):
        logger.warning(
            "Point summary adapter valid mask size mismatch: got %d, expected %d",
            int(valid.numel()),
            int(expected_count),
        )
        return None
    return valid.bool().to(device=device)


def _load_point_summary_adapter_view_counts(
    checkpoint_path: str,
    *,
    expected_count: int,
    device: torch.device,
    fallback_teacher_cache: str = "",
    expected_xyz: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    ckpt = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    metadata = ckpt.get("point_summary_adapter_metadata") or {}
    teacher_cache = metadata.get("teacher_cache") or fallback_teacher_cache
    if not teacher_cache:
        return None
    cache_path = Path(str(teacher_cache))
    if not cache_path.exists():
        logger.warning("Point summary adapter teacher cache not found: %s", cache_path)
        return None
    payload = torch.load(cache_path, map_location="cpu")
    _validate_teacher_cache_geometry(payload, expected_xyz=expected_xyz, cache_path=cache_path)
    counts = payload.get("view_counts")
    if not isinstance(counts, torch.Tensor):
        logger.warning("Point summary adapter teacher cache has no view_counts: %s", cache_path)
        return None
    if int(counts.numel()) != int(expected_count):
        logger.warning(
            "Point summary adapter view_counts size mismatch: got %d, expected %d",
            int(counts.numel()),
            int(expected_count),
        )
        return None
    return counts.float().to(device=device)


def _blend_point_summary_adapter_features(
    base_summary: torch.Tensor,
    adapter_summary: torch.Tensor,
    *,
    alpha: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Blend base decoded summaries with adapter summaries, gated by VPR support."""
    if base_summary.shape != adapter_summary.shape:
        raise ValueError(
            f"base/adapter summary shape mismatch: "
            f"{tuple(base_summary.shape)} vs {tuple(adapter_summary.shape)}"
        )
    blend = torch.full(
        (base_summary.shape[0], 1),
        min(max(float(alpha), 0.0), 1.0),
        dtype=base_summary.dtype,
        device=base_summary.device,
    )
    if valid_mask is not None:
        if valid_mask.shape != (base_summary.shape[0],):
            raise ValueError(
                f"valid_mask shape mismatch: got {tuple(valid_mask.shape)}, "
                f"expected {(base_summary.shape[0],)}"
            )
        blend = blend * valid_mask.to(device=base_summary.device, dtype=base_summary.dtype).unsqueeze(1)
    return F.normalize(
        base_summary.float() * (1.0 - blend.float()) + adapter_summary.float() * blend.float(),
        dim=-1,
    )


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
    point_summary_adapter: Optional[torch.nn.Module] = None,
    point_summary_adapter_blend_alpha: float = 1.0,
    point_summary_adapter_valid_mask: Optional[torch.Tensor] = None,
    point_summary_adapter_context_features: str = "",
    point_summary_adapter_view_counts: Optional[torch.Tensor] = None,
    point_summary_adapter_view_count_max: Optional[float] = None,
) -> torch.Tensor:
    """Decode Gaussian-center features and score them against scene text queries."""
    if not 0.0 <= float(point_summary_adapter_blend_alpha) <= 1.0:
        raise ValueError("point_summary_adapter_blend_alpha must be in [0, 1]")
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
        adapter_siglip = None
        if point_summary_adapter is not None:
            opacity = None
            scales = None
            if "opacity" in point_summary_adapter_context_features:
                opacity = model.get_opacity()[idx].to(device=device)
            if "scale_log" in point_summary_adapter_context_features:
                scales = model.get_scaling()[idx].to(device=device)
            view_counts = (
                point_summary_adapter_view_counts[idx].to(device=device)
                if point_summary_adapter_view_counts is not None
                else None
            )
            adapter_input = append_point_summary_context(
                compact.float(),
                context_features=point_summary_adapter_context_features,
                opacity=opacity,
                scales=scales,
                view_counts=view_counts,
                view_count_max=point_summary_adapter_view_count_max,
            )
            adapter_siglip = F.normalize(point_summary_adapter(adapter_input).float(), dim=-1)

        valid_mask_chunk = None
        if point_summary_adapter_valid_mask is not None:
            valid_mask_chunk = point_summary_adapter_valid_mask[idx].to(device=device)

        if (
            point_summary_adapter is not None
            and point_summary_adapter_blend_alpha >= 1.0
            and valid_mask_chunk is None
        ):
            siglip = adapter_siglip
        else:
            radio = codec.decode_points(compact.float())
            radio_tokens = radio.unsqueeze(0)
            head_param = next(summary_head.parameters(), None)
            if head_param is not None:
                radio_tokens = radio_tokens.to(dtype=head_param.dtype)
            siglip = summary_head(radio_tokens).squeeze(0)
            siglip = F.normalize(siglip.float(), dim=-1)
            if adapter_siglip is not None:
                siglip = _blend_point_summary_adapter_features(
                    siglip,
                    adapter_siglip,
                    alpha=float(point_summary_adapter_blend_alpha),
                    valid_mask=valid_mask_chunk,
                )

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

        del compact, siglip, scores
        if "radio" in locals():
            del radio
        if "radio_tokens" in locals():
            del radio_tokens
        if adapter_siglip is not None:
            del adapter_siglip
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
    registration_assignment_mode: str,
    registration_weight_mode: str,
    registration_confidence_blend: float,
    registration_confidence_mode: str,
    fallback_scores: Optional[torch.Tensor],
    device: torch.device,
    registered_feature_cache_path: Optional[str] = None,
    registered_feature_cache_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Register rendered SigLIP2 features back to Gaussian primitives and score text.

    This is the no-training VPR primitive readout: query still happens in 3D
    on Gaussian primitives, while rendered views provide the language-aligned
    registration signal without using LERF labels or masks for training/scoring.
    """
    if registration_assignment_mode not in {"center", "raster_contrib", "raster_dominant", "raster_gaussian_top1", "raster_adjoint"}:
        raise ValueError(f"Unsupported registration assignment mode: {registration_assignment_mode}")
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

        if registration_assignment_mode == "raster_adjoint":
            frame_sum, frame_counts = raster_adjoint_registered_view_features(
                model=model,
                renderer=renderer,
                viewmat=viewmat,
                siglip_feat=siglip_feat,
                alpha_map=alpha_map,
                alpha_threshold=registration_alpha_threshold,
            )
            frame_counts_cpu = frame_counts.detach().float().cpu()
            valid_cpu = frame_counts_cpu > 0
            if valid_cpu.any():
                registered_sum[valid_cpu] += frame_sum.detach().float().cpu()[valid_cpu]
                registered_counts[valid_cpu] += frame_counts_cpu[valid_cpu]
        elif registration_assignment_mode in {"raster_contrib", "raster_dominant", "raster_gaussian_top1"}:
            frame_sum, frame_counts = rasterize_registered_view_features(
                model=model,
                renderer=renderer,
                viewmat=viewmat,
                siglip_feat=siglip_feat,
                depth_map=depth_map,
                alpha_map=alpha_map,
                registration_depth_tolerance=registration_depth_tolerance,
                registration_relative_depth_tolerance=registration_relative_depth_tolerance,
                registration_alpha_threshold=registration_alpha_threshold,
                registration_weight_mode=registration_weight_mode,
                dominant_only=registration_assignment_mode == "raster_dominant",
                gaussian_top1=registration_assignment_mode == "raster_gaussian_top1",
            )
            frame_counts_cpu = frame_counts.detach().float().cpu()
            valid_cpu = frame_counts_cpu > 0
            if valid_cpu.any():
                registered_sum[valid_cpu] += frame_sum.detach().float().cpu()[valid_cpu]
                registered_counts[valid_cpu] += frame_counts_cpu[valid_cpu]
        else:
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
    if registered_feature_cache_path:
        summary_features = normalize_registered_feature_sums(
            registered_sum,
            registered_counts,
        )
        save_registered_feature_cache(
            registered_feature_cache_path,
            xyz=xyz_cpu,
            summary_features=summary_features,
            valid=valid_all,
            view_counts=registered_counts,
            metadata=registered_feature_cache_metadata or {},
            scales=model.get_scaling().detach().cpu().float(),
            rotations=model.get_rotation().detach().cpu().float(),
            opacities=model.get_opacity().detach().cpu().float(),
        )

    all_scores: List[torch.Tensor] = []
    for start in tqdm(range(0, n_gaussians, chunk_size), desc="  score registered", leave=False):
        end = min(start + chunk_size, n_gaussians)
        registered = normalize_registered_feature_sums(
            registered_sum[start:end],
            registered_counts[start:end],
        ).to(device)
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
        "assignment_mode": registration_assignment_mode,
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


def save_float_heatmap(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    heat = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    u8 = (heat * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(path), cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO))


def save_geometry_alignment_overlay(
    path: Path,
    discontinuity: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    heat = np.clip(_resize_float_map(discontinuity, gt.shape), 0.0, 1.0)
    overlay = cv2.applyColorMap((heat * 255.0).round().astype(np.uint8), cv2.COLORMAP_INFERNO)
    pred_u8, gt_u8 = _resize_binary_mask(pred, gt)
    gt_boundary = _binary_boundary(gt_u8) > 0
    pred_boundary = _binary_boundary(pred_u8) > 0
    overlay[gt_boundary] = (0, 255, 0)
    overlay[pred_boundary] = (0, 0, 255)
    overlay[np.logical_and(gt_boundary, pred_boundary)] = (255, 255, 255)
    cv2.imwrite(str(path), overlay)


def normalize_score_heatmap_features(scores: torch.Tensor) -> torch.Tensor:
    """Normalize primitive query scores to [0, 1] per query for score rendering."""
    if scores.ndim != 2:
        raise ValueError(f"Expected scores [N,Q], got {tuple(scores.shape)}")
    values = scores.detach().float()
    min_values = values.min(dim=0, keepdim=True).values
    max_values = values.max(dim=0, keepdim=True).values
    denom = (max_values - min_values).clamp_min(1e-8)
    return ((values - min_values) / denom).clamp(0.0, 1.0)


def build_direct3d_prompt_initial_mask(
    coarse_mask: np.ndarray,
    heatmap: np.ndarray | torch.Tensor,
    *,
    initial_refinement: str = "none",
) -> np.ndarray:
    """Build the feature-only SAM prompt mask for direct-3D projected masks."""
    pred = np.asarray(coarse_mask).astype(bool)
    if pred.ndim != 2:
        raise ValueError(f"Expected 2D coarse mask, got {pred.shape}")
    if initial_refinement == "none":
        return pred.copy()
    if initial_refinement == "peak_component":
        heat = heatmap.detach().float().cpu() if isinstance(heatmap, torch.Tensor) else torch.as_tensor(heatmap)
        if heat.ndim == 3:
            heat = heat[0]
        if heat.ndim != 2:
            raise ValueError(f"Expected 2D heatmap, got {tuple(heat.shape)}")
        return keep_peak_connected_component(
            pred,
            heatmap_peak_in_shape(heat, tuple(pred.shape)),
        ).astype(bool)
    raise ValueError(f"Unsupported direct-3D prompt initial refinement: {initial_refinement}")


def build_direct3d_oracle_prompt_initial_mask(
    coarse_mask: np.ndarray,
    gt_mask: np.ndarray,
    *,
    mode: str = "none",
) -> np.ndarray:
    """Diagnostic-only oracle prompt for measuring feature-only SAM head ceiling."""
    coarse = np.asarray(coarse_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    if coarse.ndim != 2:
        raise ValueError(f"Expected 2D coarse mask, got {coarse.shape}")
    if gt.shape != coarse.shape:
        raise ValueError(f"gt_mask shape {gt.shape} does not match coarse mask {coarse.shape}")
    if mode == "none":
        return coarse.copy()
    if not bool(gt.any()):
        return coarse.copy()
    if mode == "gt_mask":
        return gt.copy()
    if mode == "gt_box":
        ys, xs = np.where(gt)
        box = np.zeros_like(coarse, dtype=bool)
        box[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1] = True
        return box
    raise ValueError(f"Unsupported direct-3D oracle prompt mode: {mode}")


def apply_sam3_prompt_heatmap_guard(
    initial_mask: np.ndarray,
    refined_mask: np.ndarray,
    heatmap: np.ndarray | torch.Tensor,
    *,
    min_mean_ratio: float = 0.0,
    min_mass_ratio: float = 0.0,
    require_peak_in_refined: bool = False,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """GT-free guard for feature-only SAM prompt refinement in direct 3D."""
    return filter_refined_mask_by_heatmap_support(
        initial_mask,
        refined_mask,
        heatmap,
        min_mean_ratio=min_mean_ratio,
        min_mass_ratio=min_mass_ratio,
        require_peak_in_refined=require_peak_in_refined,
    )


def finalize_prompt_conditioned_sam3_mask(
    coarse_mask: np.ndarray,
    prompt_initial_mask: np.ndarray,
    candidate_mask: np.ndarray,
    sam3_report: Dict[str, Any],
    *,
    heatmap_guard_report: Optional[Dict[str, Any]] = None,
    geometry_gate_report: Optional[Dict[str, Any]] = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Choose the final prompt-SAM mask while preserving the coarse fallback.

    The prompt mask can be a cleaned support such as a peak component, but it is
    not the semantic fallback. If feature-only SAM or its GT-free guards reject
    a candidate, the evaluator must keep the original direct-3D coarse mask.
    """
    coarse = np.asarray(coarse_mask).astype(bool)
    prompt_initial = np.asarray(prompt_initial_mask).astype(bool)
    candidate = np.asarray(candidate_mask).astype(bool)
    report = dict(sam3_report)
    accepted = bool(report.get("accepted", False))
    fallback_reason = str(report.get("fallback_reason", "") or "")

    if heatmap_guard_report is not None:
        report["heatmap_support"] = dict(heatmap_guard_report)
        if not bool(heatmap_guard_report.get("accepted", False)):
            accepted = False
            fallback_reason = str(
                heatmap_guard_report.get("fallback_reason", fallback_reason)
            )

    if geometry_gate_report is not None:
        report.update(geometry_gate_report)
        if not bool(geometry_gate_report.get("geometry_gate_accepted", False)):
            accepted = False
            fallback_reason = str(
                geometry_gate_report.get("geometry_gate_reason", fallback_reason)
            )

    report["accepted"] = bool(accepted)
    report["prompt_initial_area"] = int(prompt_initial.sum())
    report["coarse_fallback_area"] = int(coarse.sum())
    if not accepted:
        report["fallback_reason"] = fallback_reason or "candidate_rejected"
        return coarse.copy(), report
    return candidate.copy(), report


def keep_largest_mask_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 8-connected component in a predicted binary mask.

    This is a GT-free projection cleanup for direct 3D selection. It removes
    isolated rendered fragments while preserving the dominant predicted object
    support.
    """
    pred = np.asarray(mask).astype(bool)
    if pred.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {pred.shape}")
    if not pred.any():
        return pred.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        pred.astype(np.uint8),
        connectivity=8,
    )
    if num_labels <= 2:
        return pred.copy()
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(component_areas.argmax()) + 1
    return labels == largest_label


def keep_largest_mask_component_if_dominant(
    mask: np.ndarray,
    *,
    min_largest_fraction: float = 0.65,
    min_total_pixels_for_multicomponent: int = 0,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Keep the largest component only when it dominates the support.

    Multi-part or occluded objects can be split into several projected
    components. A strict largest-component cleanup can then erase valid support
    and hurt Acc@0.25. This GT-free guard preserves the full support unless one
    component clearly dominates.
    """
    pred = np.asarray(mask).astype(bool)
    if pred.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {pred.shape}")
    total = int(pred.sum())
    report: dict[str, float | int | bool] = {
        "component_guard_total_pixels": total,
        "component_guard_largest_pixels": 0,
        "component_guard_largest_fraction": 0.0,
        "component_guard_component_count": 0,
        "component_guard_kept_largest": False,
        "component_guard_kept_largest_due_to_small_support": False,
        "component_guard_min_total_pixels_for_multicomponent": int(min_total_pixels_for_multicomponent),
    }
    if total == 0:
        return pred.copy(), report
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        pred.astype(np.uint8),
        connectivity=8,
    )
    component_count = max(int(num_labels) - 1, 0)
    report["component_guard_component_count"] = component_count
    if num_labels <= 2:
        return pred.copy(), report
    component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    largest_idx = int(component_areas.argmax())
    largest_area = int(component_areas[largest_idx])
    largest_fraction = float(largest_area / max(total, 1))
    report["component_guard_largest_pixels"] = largest_area
    report["component_guard_largest_fraction"] = largest_fraction
    if total < int(min_total_pixels_for_multicomponent):
        report["component_guard_kept_largest"] = True
        report["component_guard_kept_largest_due_to_small_support"] = True
        return labels == (largest_idx + 1), report
    if largest_fraction >= float(min_largest_fraction):
        report["component_guard_kept_largest"] = True
        return labels == (largest_idx + 1), report
    return pred.copy(), report


def keep_mask_components_by_heatmap_score(
    mask: np.ndarray,
    heatmap: np.ndarray,
    *,
    min_mass_fraction: float = 0.20,
    min_mean_fraction: float = 0.0,
    max_components: int = 0,
    min_total_pixels_for_multicomponent: int = 0,
    min_recovery_pixels: int = 0,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Keep connected support components that carry compact-field score mass.

    The RGB/GrabCut guard can produce multiple components after snapping
    rendered primitive support to image edges. Area-only cleanup is brittle for
    small or multipart objects: the largest component may be clutter, while a
    smaller component can contain the strongest text-aligned score response.
    This GT-free guard ranks components by the rendered compact score heatmap and
    keeps the high-mass components under a fixed global policy.
    """
    pred = np.asarray(mask).astype(bool)
    if pred.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {pred.shape}")
    heat = np.asarray(heatmap, dtype=np.float32)
    if heat.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got {heat.shape}")
    if heat.shape != pred.shape:
        heat = cv2.resize(heat, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    heat = heat - float(heat.min())
    heat_max = float(heat.max())
    if heat_max > 1e-8:
        heat = heat / heat_max
    else:
        heat = pred.astype(np.float32)

    total = int(pred.sum())
    report: dict[str, float | int | bool] = {
        "score_component_guard_total_pixels": total,
        "score_component_guard_component_count": 0,
        "score_component_guard_kept_components": 0,
        "score_component_guard_min_mass_fraction": float(min_mass_fraction),
        "score_component_guard_min_mean_fraction": float(min_mean_fraction),
        "score_component_guard_max_components": int(max_components),
        "score_component_guard_min_total_pixels_for_multicomponent": int(
            min_total_pixels_for_multicomponent
        ),
        "score_component_guard_min_recovery_pixels": int(min_recovery_pixels),
        "score_component_guard_heatmap_recovered": False,
        "score_component_guard_top_mass": 0.0,
        "score_component_guard_top_mean": 0.0,
        "score_component_guard_kept_top_only_due_to_small_support": False,
    }
    recovery_pixels = max(int(min_recovery_pixels), 0)
    if recovery_pixels > 0 and total < recovery_pixels:
        flat = heat.reshape(-1)
        if flat.size > 0 and float(flat.max()) > 1e-8:
            k = min(recovery_pixels, flat.size)
            chosen = np.argpartition(flat, -k)[-k:]
            recovered = np.zeros(flat.size, dtype=bool)
            recovered[chosen] = True
            pred = np.logical_or(pred, recovered.reshape(pred.shape))
            total = int(pred.sum())
            report["score_component_guard_total_pixels"] = total
            report["score_component_guard_heatmap_recovered"] = True
    if total == 0:
        return pred.copy(), report

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        pred.astype(np.uint8),
        connectivity=8,
    )
    component_count = max(int(num_labels) - 1, 0)
    report["score_component_guard_component_count"] = component_count
    if num_labels <= 2:
        report["score_component_guard_kept_components"] = component_count
        return pred.copy(), report

    components: list[tuple[int, float, float, int]] = []
    for label in range(1, int(num_labels)):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        mass = float(heat[component].sum())
        mean = float(mass / max(area, 1))
        components.append((label, mass, mean, area))
    components.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
    top_label, top_mass, top_mean, _ = components[0]
    report["score_component_guard_top_mass"] = float(top_mass)
    report["score_component_guard_top_mean"] = float(top_mean)

    if total < int(min_total_pixels_for_multicomponent):
        kept_labels = [int(top_label)]
        report["score_component_guard_kept_top_only_due_to_small_support"] = True
    else:
        mass_floor = float(top_mass) * max(float(min_mass_fraction), 0.0)
        mean_floor = float(top_mean) * max(float(min_mean_fraction), 0.0)
        kept_labels = [
            int(label)
            for label, mass, mean, _area in components
            if float(mass) >= mass_floor and float(mean) >= mean_floor
        ]
        if not kept_labels:
            kept_labels = [int(top_label)]
        if max_components > 0:
            kept_labels = kept_labels[: int(max_components)]

    kept = np.isin(labels, np.asarray(kept_labels, dtype=np.int32))
    report["score_component_guard_kept_components"] = int(len(kept_labels))
    return kept, report


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


def _resize_bool_mask(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(shape_hw[0]), int(shape_hw[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target mask shape: {shape_hw}")
    mask_u8 = np.asarray(mask).astype(np.uint8)
    if mask_u8.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask_u8.shape}")
    if mask_u8.shape == (target_h, target_w):
        return mask_u8.astype(bool)
    resized = cv2.resize(
        mask_u8,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def _scaled_morph_radius(radius_pixels: int, src_shape: Tuple[int, int], dst_shape: Tuple[int, int]) -> int:
    radius = int(radius_pixels)
    if radius <= 0:
        return 0
    src_h, src_w = max(int(src_shape[0]), 1), max(int(src_shape[1]), 1)
    dst_h, dst_w = max(int(dst_shape[0]), 1), max(int(dst_shape[1]), 1)
    scale = 0.5 * (float(dst_h) / float(src_h) + float(dst_w) / float(src_w))
    return max(1, int(round(float(radius) * scale)))


def _morph_bool_mask(mask: np.ndarray, op: str, radius: int) -> np.ndarray:
    pred = np.asarray(mask).astype(bool)
    if radius <= 0 or not pred.any():
        return pred.copy()
    kernel_size = max(1, int(radius) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = pred.astype(np.uint8)
    if op == "dilate":
        return cv2.dilate(mask_u8, kernel, iterations=1).astype(bool)
    if op == "erode":
        return cv2.erode(mask_u8, kernel, iterations=1).astype(bool)
    raise ValueError(f"Unsupported morphology op: {op}")


def _keep_component_with_seed_overlap(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    pred = np.asarray(mask).astype(bool)
    seed_bool = np.asarray(seed).astype(bool)
    if pred.ndim != 2 or seed_bool.ndim != 2 or pred.shape != seed_bool.shape:
        raise ValueError(f"Mask/seed shape mismatch: {pred.shape} vs {seed_bool.shape}")
    if not pred.any():
        return pred.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        pred.astype(np.uint8),
        connectivity=8,
    )
    if num_labels <= 2:
        return pred.copy()
    best_label = -1
    best_overlap = -1
    best_area = -1
    for label in range(1, num_labels):
        component = labels == label
        overlap = int(np.logical_and(component, seed_bool).sum())
        area = int(stats[label, cv2.CC_STAT_AREA])
        if overlap > best_overlap or (overlap == best_overlap and area > best_area):
            best_label = int(label)
            best_overlap = overlap
            best_area = area
    if best_label <= 0:
        return keep_largest_mask_component(pred)
    return labels == best_label


def _box_support_from_mask(mask: np.ndarray, padding: int) -> np.ndarray:
    pred = np.asarray(mask).astype(bool)
    if pred.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {pred.shape}")
    support = np.zeros_like(pred, dtype=bool)
    if not pred.any():
        return support
    ys, xs = np.nonzero(pred)
    pad = max(int(padding), 0)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, pred.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, pred.shape[1])
    support[y0:y1, x0:x1] = True
    return support


def refine_mask_with_sam3_adaptor_features(
    feature_map: torch.Tensor,
    initial_mask: np.ndarray,
    *,
    support_mode: str = "mask_dilate",
    prototype_mode: str = "mask_inner",
    support_dilate_pixels: int = 8,
    inner_erode_pixels: int = 2,
    score_std_scale: float = 0.0,
    min_area_scale: float = 0.25,
    max_area_scale: float = 1.25,
    max_initial_area_fraction: float = 1.0,
    background_weight: float = 0.20,
    min_initial_iou: float = 0.03,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Refine a projected 3D mask using only CTF/RADIO features in SAM3-adaptor space.

    The initial direct-3D mask acts as a prompt: eroded high-confidence pixels
    define a foreground prototype, a dilated local band defines the admissible
    support, and normalized adaptor-token similarity reselects the object area.
    No RGB image or official SAM3 image encoder/decoder is used here.
    """
    initial = np.asarray(initial_mask).astype(bool)
    report: Dict[str, Any] = {
        "backend": "radio_sam3_adaptor_feature_prototype",
        "attempted": True,
        "accepted": False,
        "fallback_reason": "",
        "candidate_count": 1,
        "selected_index": 0,
        "best_initial_overlap": 0.0,
        "selected_score": 0.0,
        "box_prompt_format": "",
        "box_prompt_cxcywh_norm": None,
        "box_prompt_xyxy_pixels": None,
        "support_mode": support_mode,
        "prototype_mode": prototype_mode,
        "support_dilate_pixels": int(support_dilate_pixels),
        "inner_erode_pixels": int(inner_erode_pixels),
        "score_std_scale": float(score_std_scale),
        "min_area_scale": float(min_area_scale),
        "max_area_scale": float(max_area_scale),
        "max_initial_area_fraction": float(max_initial_area_fraction),
        "background_weight": float(background_weight),
        "min_initial_iou": float(min_initial_iou),
    }
    if initial.ndim != 2:
        raise ValueError(f"Expected 2D initial mask, got {initial.shape}")
    if not initial.any():
        report["fallback_reason"] = "empty_initial_mask"
        return initial.copy(), report
    initial_area_fraction = float(initial.sum()) / float(max(initial.size, 1))
    report["initial_area_fraction"] = initial_area_fraction
    if initial_area_fraction > float(max_initial_area_fraction):
        report["fallback_reason"] = "initial_mask_too_large"
        return initial.copy(), report

    feat = feature_map.detach()
    if feat.ndim == 4:
        if feat.shape[0] != 1:
            raise ValueError(f"Expected singleton batch feature map, got {tuple(feat.shape)}")
        feat = feat[0]
    if feat.ndim != 3:
        raise ValueError(f"Expected feature_map [C,H,W] or [1,C,H,W], got {tuple(feature_map.shape)}")
    feat = F.normalize(feat.float(), dim=0)
    channels, feat_h, feat_w = int(feat.shape[0]), int(feat.shape[1]), int(feat.shape[2])
    report["feature_shape"] = [channels, feat_h, feat_w]

    low_initial = _resize_bool_mask(initial, (feat_h, feat_w))
    if not low_initial.any():
        report["fallback_reason"] = "empty_resized_initial_mask"
        return initial.copy(), report

    support_radius = _scaled_morph_radius(
        support_dilate_pixels,
        initial.shape,
        (feat_h, feat_w),
    )
    erode_radius = _scaled_morph_radius(
        inner_erode_pixels,
        initial.shape,
        (feat_h, feat_w),
    )
    if support_mode == "mask_dilate":
        support = _morph_bool_mask(low_initial, "dilate", support_radius)
    elif support_mode == "box":
        support = _box_support_from_mask(low_initial, support_radius)
    else:
        raise ValueError(f"Unsupported SAM3-adaptor support mode: {support_mode}")
    inner = _morph_bool_mask(low_initial, "erode", erode_radius)
    if not inner.any():
        inner = low_initial
    if not support.any():
        support = low_initial
    if prototype_mode == "mask_inner":
        prototype_mask = inner
    elif prototype_mode == "box":
        prototype_mask = support
    else:
        raise ValueError(f"Unsupported SAM3-adaptor prototype mode: {prototype_mode}")

    inner_idx = torch.from_numpy(prototype_mask.reshape(-1)).to(device=feat.device)
    support_idx = torch.from_numpy(support.reshape(-1)).to(device=feat.device)
    tokens = feat.reshape(channels, feat_h * feat_w).transpose(0, 1)
    fg_tokens = tokens[inner_idx]
    if fg_tokens.numel() == 0:
        report["fallback_reason"] = "empty_foreground_tokens"
        return initial.copy(), report
    fg_proto = F.normalize(fg_tokens.mean(dim=0, keepdim=True), dim=-1)
    scores = (tokens @ fg_proto.t()).flatten()

    bg_mask = np.logical_and(support, ~_morph_bool_mask(low_initial, "dilate", 1))
    bg_idx = torch.from_numpy(bg_mask.reshape(-1)).to(device=feat.device)
    if bg_idx.any() and float(background_weight) > 0.0:
        bg_proto = F.normalize(tokens[bg_idx].mean(dim=0, keepdim=True), dim=-1)
        scores = scores - float(background_weight) * (tokens @ bg_proto.t()).flatten()

    support_scores = scores[support_idx]
    if support_scores.numel() == 0:
        report["fallback_reason"] = "empty_support_scores"
        return initial.copy(), report
    threshold = support_scores.mean() + float(score_std_scale) * support_scores.std(unbiased=False)
    candidate_flat = torch.zeros((feat_h * feat_w,), dtype=torch.bool, device=feat.device)
    candidate_flat[support_idx] = scores[support_idx] >= threshold

    initial_area = max(int(low_initial.sum()), 1)
    min_area = max(1, int(round(float(min_area_scale) * float(initial_area))))
    max_area = max(min_area, int(round(float(max_area_scale) * float(initial_area))))
    max_area = min(max_area, int(support_idx.sum().item()))
    support_order = torch.argsort(scores[support_idx], descending=True)
    support_linear = torch.nonzero(support_idx, as_tuple=False).flatten()
    if int(candidate_flat.sum().item()) < min_area:
        chosen = support_linear[support_order[:min_area]]
        candidate_flat.zero_()
        candidate_flat[chosen] = True
    elif int(candidate_flat.sum().item()) > max_area:
        chosen = support_linear[support_order[:max_area]]
        candidate_flat.zero_()
        candidate_flat[chosen] = True
    candidate_low = candidate_flat.reshape(feat_h, feat_w).detach().cpu().numpy().astype(bool)
    candidate_low = _keep_component_with_seed_overlap(candidate_low, inner)
    if not candidate_low.any():
        report["fallback_reason"] = "empty_candidate_mask"
        return initial.copy(), report

    refined = _resize_bool_mask(candidate_low, initial.shape)
    initial_overlap = mask_overlap_stats(refined, initial)
    report["best_initial_overlap"] = float(initial_overlap["iou"])
    report["selected_score"] = float(scores[candidate_flat].mean().detach().cpu().item()) if candidate_flat.any() else 0.0
    report["low_initial_pixels"] = int(low_initial.sum())
    report["low_support_pixels"] = int(support.sum())
    report["low_refined_pixels"] = int(candidate_low.sum())
    report["pred_pixels"] = int(refined.sum())
    if float(initial_overlap["iou"]) < float(min_initial_iou):
        report["fallback_reason"] = "low_initial_overlap"
        return initial.copy(), report
    report["accepted"] = True
    report["fallback_reason"] = "accepted"
    return refined, report


def _feature_map_to_grabcut_image(feature_map: torch.Tensor) -> np.ndarray:
    feat = feature_map.detach()
    if feat.ndim == 4:
        if feat.shape[0] != 1:
            raise ValueError(f"Expected singleton batch feature map, got {tuple(feat.shape)}")
        feat = feat[0]
    if feat.ndim != 3:
        raise ValueError(f"Expected feature map [C,H,W] or [1,C,H,W], got {tuple(feature_map.shape)}")
    feat = feat.float()
    channels, height, width = int(feat.shape[0]), int(feat.shape[1]), int(feat.shape[2])
    if channels >= 3:
        tokens = feat.reshape(channels, height * width).transpose(0, 1)
        tokens = tokens - tokens.mean(dim=0, keepdim=True)
        try:
            _, _, vh = torch.linalg.svd(tokens, full_matrices=False)
            projected = tokens @ vh[:3].transpose(0, 1)
        except RuntimeError:
            projected = tokens[:, :3]
        image = projected.reshape(height, width, 3).detach().cpu().numpy()
    else:
        image = feat.detach().cpu().permute(1, 2, 0).numpy()
        image = np.repeat(image, repeats=3, axis=2)[:, :, :3]
    image = image.astype(np.float32)
    lo = np.percentile(image.reshape(-1, 3), 1, axis=0)
    hi = np.percentile(image.reshape(-1, 3), 99, axis=0)
    image = (image - lo.reshape(1, 1, 3)) / np.maximum(
        hi.reshape(1, 1, 3) - lo.reshape(1, 1, 3),
        1e-6,
    )
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def refine_mask_with_sam3_feature_grabcut(
    feature_map: torch.Tensor,
    initial_mask: np.ndarray,
    *,
    iterations: int = 1,
    dilate_pixels: int = 5,
    erode_pixels: int = 2,
    min_initial_iou: float = 0.05,
    min_refined_area_ratio: float = 0.0,
    max_refined_area_ratio: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Snap a coarse mask to SAM3-adaptor feature boundaries without RGB input."""
    initial = np.asarray(initial_mask).astype(bool)
    report: Dict[str, Any] = {
        "backend": "radio_sam3_adaptor_feature_grabcut",
        "attempted": True,
        "accepted": False,
        "fallback_reason": "",
        "candidate_count": 1,
        "selected_index": 0,
        "best_initial_overlap": 0.0,
        "selected_score": 0.0,
        "box_prompt_format": "",
        "box_prompt_cxcywh_norm": None,
        "box_prompt_xyxy_pixels": None,
        "iterations": int(iterations),
        "dilate_pixels": int(dilate_pixels),
        "erode_pixels": int(erode_pixels),
        "min_initial_iou": float(min_initial_iou),
        "min_refined_area_ratio": float(min_refined_area_ratio),
        "max_refined_area_ratio": float(max_refined_area_ratio),
        "refined_area_ratio": 0.0,
    }
    if initial.ndim != 2:
        raise ValueError(f"Expected 2D initial mask, got {initial.shape}")
    if not initial.any():
        report["fallback_reason"] = "empty_initial_mask"
        return initial.copy(), report
    if initial.all():
        report["fallback_reason"] = "full_initial_mask"
        return initial.copy(), report
    try:
        feature_image = _feature_map_to_grabcut_image(feature_map)
    except ValueError as exc:
        report["fallback_reason"] = "feature_shape_mismatch"
        report["error"] = str(exc)
        return initial.copy(), report
    feat_h, feat_w = feature_image.shape[:2]
    low_initial = _resize_bool_mask(initial, (feat_h, feat_w))
    if not low_initial.any() or low_initial.all():
        report["fallback_reason"] = "degenerate_resized_initial_mask"
        return initial.copy(), report
    support_radius = _scaled_morph_radius(dilate_pixels, initial.shape, (feat_h, feat_w))
    erode_radius = _scaled_morph_radius(erode_pixels, initial.shape, (feat_h, feat_w))
    support = _morph_bool_mask(low_initial, "dilate", support_radius)
    sure_fg = _morph_bool_mask(low_initial, "erode", erode_radius)
    if not sure_fg.any():
        sure_fg = low_initial
    init = np.full((feat_h, feat_w), cv2.GC_BGD, dtype=np.uint8)
    init[support] = cv2.GC_PR_FGD
    init[low_initial] = cv2.GC_PR_FGD
    init[sure_fg] = cv2.GC_FGD
    init[~support] = cv2.GC_BGD
    if not np.any(init == cv2.GC_BGD) or not np.any((init == cv2.GC_FGD) | (init == cv2.GC_PR_FGD)):
        report["fallback_reason"] = "invalid_grabcut_initialization"
        return initial.copy(), report
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            feature_image,
            init,
            None,
            bgd_model,
            fgd_model,
            max(1, int(iterations)),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as exc:
        report["fallback_reason"] = "grabcut_failed"
        report["error"] = str(exc)
        return initial.copy(), report
    refined_low = ((init == cv2.GC_FGD) | (init == cv2.GC_PR_FGD)) & support
    if not refined_low.any():
        report["fallback_reason"] = "empty_refined_mask"
        return initial.copy(), report
    refined = _resize_bool_mask(refined_low, initial.shape)
    overlap = mask_overlap_stats(refined, initial)
    overlap_iou = float(overlap["iou"])
    area_ratio = float(refined.sum()) / float(max(initial.sum(), 1))
    report["best_initial_overlap"] = overlap_iou
    report["selected_score"] = overlap_iou
    report["refined_area_ratio"] = area_ratio
    report["low_refined_pixels"] = int(refined_low.sum())
    report["pred_pixels"] = int(refined.sum())
    if overlap_iou < float(min_initial_iou):
        report["fallback_reason"] = "insufficient_initial_overlap"
        return initial.copy(), report
    if float(min_refined_area_ratio) > 0.0 and area_ratio < float(min_refined_area_ratio):
        report["fallback_reason"] = "refined_area_too_small"
        return initial.copy(), report
    if float(max_refined_area_ratio) > 0.0 and area_ratio > float(max_refined_area_ratio):
        report["fallback_reason"] = "refined_area_too_large"
        return initial.copy(), report
    report["accepted"] = True
    report["fallback_reason"] = "accepted"
    return refined, report


def mask_to_sam3_box_prompt(
    mask: np.ndarray,
    *,
    padding_pixels: int = 0,
) -> Optional[List[float]]:
    """Convert a binary mask to SAM3's normalized [cx, cy, w, h] box prompt."""
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
    cx = (float(x0) + box_w * 0.5) / float(max(width, 1))
    cy = (float(y0) + box_h * 0.5) / float(max(height, 1))
    return [
        float(np.clip(cx, 0.0, 1.0)),
        float(np.clip(cy, 0.0, 1.0)),
        float(np.clip(box_w / float(max(width, 1)), 0.0, 1.0)),
        float(np.clip(box_h / float(max(height, 1)), 0.0, 1.0)),
    ]


def choose_sam3_box_refined_mask(
    initial_mask: np.ndarray,
    candidate_masks: np.ndarray,
    *,
    scores: Optional[np.ndarray] = None,
    min_initial_iou: float = 0.05,
) -> np.ndarray:
    refined, _report = choose_sam3_box_refined_mask_with_report(
        initial_mask,
        candidate_masks,
        scores=scores,
        min_initial_iou=min_initial_iou,
    )
    return refined


def choose_sam3_box_refined_mask_with_report(
    initial_mask: np.ndarray,
    candidate_masks: np.ndarray,
    *,
    scores: Optional[np.ndarray] = None,
    min_initial_iou: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Select a SAM3 candidate by overlap with the initial predicted mask.

    This deliberately uses the rendered prediction as the only selector, not GT.
    The SAM3 score only breaks ties between candidates with similar prompt-mask
    overlap.
    """
    initial = np.asarray(initial_mask).astype(bool)
    report: Dict[str, Any] = {
        "accepted": False,
        "fallback_reason": "",
        "candidate_count": 0,
        "selected_index": -1,
        "best_initial_overlap": 0.0,
        "selected_score": 0.0,
    }
    if initial.ndim != 2:
        raise ValueError(f"Expected 2D initial mask, got {initial.shape}")
    if not initial.any():
        report["fallback_reason"] = "empty_initial_mask"
        return initial.copy(), report
    masks = np.asarray(candidate_masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3 or masks.shape[-2:] != initial.shape:
        report["fallback_reason"] = "candidate_shape_mismatch"
        report["candidate_shape"] = list(masks.shape)
        return initial.copy(), report
    report["candidate_count"] = int(masks.shape[0])
    if masks.shape[0] == 0:
        report["fallback_reason"] = "empty_candidate_set"
        return initial.copy(), report

    score_arr = np.asarray(scores if scores is not None else np.zeros((masks.shape[0],)), dtype=np.float32)
    if score_arr.ndim != 1 or score_arr.shape[0] != masks.shape[0]:
        score_arr = np.zeros((masks.shape[0],), dtype=np.float32)
    best_idx = -1
    best_overlap = -1.0
    best_score = -float("inf")
    for idx, candidate in enumerate(masks):
        cand = np.asarray(candidate) > 0
        inter = float(np.logical_and(initial, cand).sum())
        union = float(np.logical_or(initial, cand).sum())
        overlap = inter / union if union > 0 else 0.0
        candidate_score = float(score_arr[idx])
        if overlap > best_overlap + 1e-8 or (
            abs(overlap - best_overlap) <= 1e-8 and candidate_score > best_score
        ):
            best_idx = idx
            best_overlap = overlap
            best_score = candidate_score
    report["selected_index"] = int(best_idx)
    report["best_initial_overlap"] = float(max(best_overlap, 0.0))
    report["selected_score"] = float(best_score if np.isfinite(best_score) else 0.0)
    if best_idx < 0:
        report["fallback_reason"] = "no_valid_candidate"
        return initial.copy(), report
    if best_overlap < float(min_initial_iou):
        report["fallback_reason"] = "low_initial_overlap"
        return initial.copy(), report
    report["accepted"] = True
    report["fallback_reason"] = "accepted"
    return (np.asarray(masks[best_idx]) > 0).astype(bool), report


def _sam3_mask_head_logits_to_candidates(
    logits: torch.Tensor,
    *,
    target_shape: Tuple[int, int],
    logit_threshold: float,
) -> np.ndarray:
    pred_logits = logits.detach().float()
    if pred_logits.ndim == 4:
        if pred_logits.shape[0] != 1:
            raise ValueError(f"Expected singleton batch logits, got {tuple(pred_logits.shape)}")
        pred_logits = pred_logits[0]
    if pred_logits.ndim != 3:
        raise ValueError(f"Expected logits [M,H,W] or [1,M,H,W], got {tuple(logits.shape)}")
    masks = pred_logits.unsqueeze(0)
    if tuple(masks.shape[-2:]) != (int(target_shape[0]), int(target_shape[1])):
        masks = F.interpolate(
            masks,
            size=(int(target_shape[0]), int(target_shape[1])),
            mode="bilinear",
            align_corners=False,
        )
    return (masks[0] > float(logit_threshold)).detach().cpu().numpy().astype(bool)


def choose_sam3_mask_head_refined_mask_with_report(
    initial_mask: np.ndarray,
    mask_logits: torch.Tensor,
    *,
    logit_threshold: float = 0.0,
    min_initial_iou: float = 0.05,
    max_initial_area_fraction: float = 1.0,
    min_refined_area_ratio: float = 0.0,
    max_refined_area_ratio: float = 0.0,
    support_dilate: int = -1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Select a trained feature-only SAM3 mask-logit candidate by coarse-mask overlap."""
    initial = np.asarray(initial_mask).astype(bool)
    report: Dict[str, Any] = {
        "backend": "ctf_sam3_mask_logit_projector",
        "attempted": True,
        "accepted": False,
        "fallback_reason": "",
        "candidate_count": 0,
        "selected_index": -1,
        "best_initial_overlap": 0.0,
        "selected_score": 0.0,
        "box_prompt_format": "",
        "box_prompt_cxcywh_norm": None,
        "box_prompt_xyxy_pixels": None,
        "logit_threshold": float(logit_threshold),
        "min_initial_iou": float(min_initial_iou),
        "max_initial_area_fraction": float(max_initial_area_fraction),
        "min_refined_area_ratio": float(min_refined_area_ratio),
        "max_refined_area_ratio": float(max_refined_area_ratio),
        "support_dilate": int(support_dilate),
    }
    if initial.ndim != 2:
        raise ValueError(f"Expected 2D initial mask, got {initial.shape}")
    if not initial.any():
        report["fallback_reason"] = "empty_initial_mask"
        return initial.copy(), report
    initial_area_fraction = float(initial.sum()) / float(max(initial.size, 1))
    report["initial_area_fraction"] = initial_area_fraction
    if initial_area_fraction > float(max_initial_area_fraction):
        report["fallback_reason"] = "initial_mask_too_large"
        return initial.copy(), report
    try:
        candidates = _sam3_mask_head_logits_to_candidates(
            mask_logits,
            target_shape=initial.shape,
            logit_threshold=logit_threshold,
        )
    except ValueError as exc:
        report["fallback_reason"] = "candidate_shape_mismatch"
        report["error"] = str(exc)
        return initial.copy(), report
    if candidates.shape[0] == 0:
        report["fallback_reason"] = "empty_candidate_set"
        return initial.copy(), report
    scores_np = (
        torch.sigmoid(mask_logits.detach().float())
        .flatten(-2)
        .mean(dim=-1)
        .reshape(-1)
        .cpu()
        .numpy()
    )
    refined, choose_report = choose_mask_candidate_by_initial_overlap(
        initial,
        candidates,
        scores=scores_np,
        min_initial_iou=min_initial_iou,
        min_refined_area_ratio=min_refined_area_ratio,
        max_refined_area_ratio=max_refined_area_ratio,
        support_dilate=support_dilate,
    )
    report.update(choose_report)
    report["backend"] = "ctf_sam3_mask_logit_projector"
    report["attempted"] = True
    report["logit_threshold"] = float(logit_threshold)
    report["max_initial_area_fraction"] = float(max_initial_area_fraction)
    report["min_refined_area_ratio"] = float(min_refined_area_ratio)
    report["max_refined_area_ratio"] = float(max_refined_area_ratio)
    report["support_dilate"] = int(support_dilate)
    return refined, report


def refine_mask_with_prompt_conditioned_sam3_head(
    *,
    feature_map: torch.Tensor,
    prompt_embedding: torch.Tensor,
    coarse_mask: np.ndarray,
    head: torch.nn.Module,
    logit_threshold: float = 0.0,
    min_initial_iou: float = 0.05,
    max_initial_area_fraction: float = 1.0,
    min_refined_area_ratio: float = 0.0,
    max_refined_area_ratio: float = 0.0,
    support_dilate: int = -1,
    coarse_dilate: int = 0,
    coarse_threshold: float = 0.5,
    min_quality: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Refine a coarse mask using rendered features and a text-conditioned mask head."""

    initial = np.asarray(coarse_mask).astype(bool)
    report: Dict[str, Any] = {
        "backend": "prompt_conditioned_ctf_sam3_mask_head_no_rgb",
        "attempted": True,
        "accepted": False,
        "fallback_reason": "",
        "candidate_count": 0,
        "selected_index": -1,
        "best_initial_overlap": 0.0,
        "selected_score": 0.0,
        "box_prompt_format": "",
        "box_prompt_cxcywh_norm": None,
        "box_prompt_xyxy_pixels": None,
        "logit_threshold": float(logit_threshold),
        "min_initial_iou": float(min_initial_iou),
        "max_initial_area_fraction": float(max_initial_area_fraction),
        "min_refined_area_ratio": float(min_refined_area_ratio),
        "max_refined_area_ratio": float(max_refined_area_ratio),
        "support_dilate": int(support_dilate),
        "coarse_dilate": int(coarse_dilate),
        "min_quality": float(min_quality),
        "predicted_quality": None,
    }
    if initial.ndim != 2:
        raise ValueError(f"Expected 2D coarse mask, got {initial.shape}")
    if not initial.any():
        report["fallback_reason"] = "empty_initial_mask"
        return initial.copy(), report
    initial_area_fraction = float(initial.sum()) / float(max(initial.size, 1))
    report["initial_area_fraction"] = initial_area_fraction
    if initial_area_fraction > float(max_initial_area_fraction):
        report["fallback_reason"] = "initial_mask_too_large"
        return initial.copy(), report

    if feature_map.ndim == 3:
        feature_map = feature_map.unsqueeze(0)
    if feature_map.ndim != 4:
        raise ValueError(f"Expected feature_map [B,C,H,W] or [C,H,W], got {tuple(feature_map.shape)}")
    prompt_shape = (int(feature_map.shape[-2]), int(feature_map.shape[-1]))
    prompt = prompt_embedding.detach().float()
    if prompt.ndim != 1:
        raise ValueError(f"Expected prompt embedding [D], got {tuple(prompt.shape)}")
    prompt_initial = initial
    report["coarse_prompt_input_shape"] = [int(initial.shape[0]), int(initial.shape[1])]
    report["coarse_prompt_shape"] = [prompt_shape[0], prompt_shape[1]]
    report["coarse_prompt_resized"] = tuple(initial.shape) != prompt_shape
    if tuple(initial.shape) != prompt_shape:
        prompt_tensor = torch.from_numpy(initial.astype(np.float32)).view(1, 1, *initial.shape)
        prompt_tensor = F.interpolate(
            prompt_tensor,
            size=prompt_shape,
            mode="nearest",
        )
        prompt_initial = (prompt_tensor[0, 0].cpu().numpy() > float(coarse_threshold))
        if not prompt_initial.any():
            report["fallback_reason"] = "empty_prompt_after_resize"
            return initial.copy(), report
    coarse_tensor = torch.from_numpy(prompt_initial.astype(np.float32)).unsqueeze(0)
    coarse_tensor = build_coarse_prompt_from_target(
        coarse_tensor,
        dilate=int(coarse_dilate),
        threshold=float(coarse_threshold),
    )
    try:
        device = next(head.parameters()).device
    except StopIteration:
        device = feature_map.device
    with torch.no_grad():
        if hasattr(head, "forward_with_quality"):
            logits, quality_logits = head.forward_with_quality(
                feature_map.to(device=device),
                prompt.to(device=device).view(1, 1, -1),
                coarse_tensor.to(device=device).unsqueeze(0),
            )
        else:
            logits = head(
                feature_map.to(device=device),
                prompt.to(device=device).view(1, 1, -1),
                coarse_tensor.to(device=device).unsqueeze(0),
            )
            quality_logits = None
    if quality_logits is not None:
        quality = float(torch.sigmoid(quality_logits.detach().float()).reshape(-1)[0].cpu())
        report["predicted_quality"] = quality
        if float(min_quality) > 0.0 and quality < float(min_quality):
            report["fallback_reason"] = "low_predicted_quality"
            return initial.copy(), report
    refined, choose_report = choose_sam3_mask_head_refined_mask_with_report(
        initial,
        logits,
        logit_threshold=logit_threshold,
        min_initial_iou=min_initial_iou,
        max_initial_area_fraction=max_initial_area_fraction,
        min_refined_area_ratio=min_refined_area_ratio,
        max_refined_area_ratio=max_refined_area_ratio,
        support_dilate=support_dilate,
    )
    report.update(choose_report)
    report["backend"] = "prompt_conditioned_ctf_sam3_mask_head_no_rgb"
    report["attempted"] = True
    report["coarse_dilate"] = int(coarse_dilate)
    report["min_quality"] = float(min_quality)
    return refined, report


class Sam3BoxMaskRefiner:
    """Official SAM3 box-prompt mask refinement for projected 3D selections."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        device: str,
        confidence_threshold: float,
        resolution: int,
        amp_dtype: str,
        box_padding_pixels: int,
        min_initial_iou: float,
    ) -> None:
        from radio_gs.scripts.build_sam3_foundation_cache import (
            _load_sam3_model,
            resolve_sam3_amp_dtype,
            validate_sam3_resolution,
        )

        resolved_resolution = validate_sam3_resolution(
            resolution,
            allow_unsafe=False,
        )
        self.processor = _load_sam3_model(
            checkpoint_path=checkpoint_path,
            device=device,
            confidence_threshold=confidence_threshold,
            dtype="auto",
            resolution=resolved_resolution,
        )
        self.amp_dtype = resolve_sam3_amp_dtype(device, amp_dtype)
        self.box_padding_pixels = int(box_padding_pixels)
        self.min_initial_iou = float(min_initial_iou)

    def set_image(self, rgb_bgr: np.ndarray) -> Dict[str, Any]:
        from radio_gs.scripts.build_sam3_foundation_cache import sam3_autocast_context

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(np.ascontiguousarray(rgb))
        with torch.no_grad(), sam3_autocast_context(str(self.processor.device), self.amp_dtype):
            return self.processor.set_image(image)

    def refine_from_state(self, state: Dict[str, Any], mask: np.ndarray) -> np.ndarray:
        refined, _report = self.refine_from_state_with_report(state, mask)
        return refined

    def refine_from_state_with_report(
        self,
        state: Dict[str, Any],
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        from radio_gs.scripts.build_sam3_foundation_cache import sam3_autocast_context

        pred = np.asarray(mask).astype(bool)
        if pred.ndim != 2:
            raise ValueError(f"Expected 2D mask, got {pred.shape}")
        height, width = pred.shape
        report: Dict[str, Any] = {
            "attempted": True,
            "accepted": False,
            "fallback_reason": "",
            "candidate_count": 0,
            "selected_index": -1,
            "best_initial_overlap": 0.0,
            "selected_score": 0.0,
            "box_prompt_format": "normalized_cxcywh",
            "box_prompt_cxcywh_norm": None,
            "box_prompt_xyxy_pixels": None,
            "min_initial_iou": float(self.min_initial_iou),
        }
        box = mask_to_sam3_box_prompt(
            pred,
            padding_pixels=self.box_padding_pixels,
        )
        report["box_prompt_cxcywh_norm"] = box
        if box is None:
            report["fallback_reason"] = "empty_initial_mask"
            return pred.copy(), report
        cx, cy, bw, bh = [float(v) for v in box]
        abs_w = bw * float(width)
        abs_h = bh * float(height)
        x0 = cx * float(width) - abs_w * 0.5
        y0 = cy * float(height) - abs_h * 0.5
        report["box_prompt_xyxy_pixels"] = [float(x0), float(y0), float(x0 + abs_w), float(y0 + abs_h)]
        query_state = dict(state)
        with torch.no_grad(), sam3_autocast_context(str(self.processor.device), self.amp_dtype):
            output = self.processor.add_geometric_prompt(box, True, query_state)
        masks = output.get("masks")
        if masks is None:
            logits = output.get("masks_logits")
            if logits is None:
                report["fallback_reason"] = "missing_masks_and_logits"
                return pred.copy(), report
            masks = logits.float() > 0.0
        if torch.is_tensor(masks):
            masks_np = masks.detach().cpu().numpy()
        else:
            masks_np = np.asarray(masks)
        scores = output.get("scores")
        scores_np = (
            scores.detach().float().cpu().numpy()
            if torch.is_tensor(scores)
            else np.asarray(scores if scores is not None else [], dtype=np.float32)
        )
        refined, choose_report = choose_sam3_box_refined_mask_with_report(
            pred,
            masks_np,
            scores=scores_np,
            min_initial_iou=self.min_initial_iou,
        )
        report.update(choose_report)
        return refined, report


class Sam3AdaptorMaskRefiner:
    """Feature-only SAM3-adaptor boundary refinement for direct-3D masks."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        codec: torch.nn.Module,
        renderer: FeatureFieldRenderer,
        sharpener: torch.nn.Module,
        refiner: Optional[torch.nn.Module],
        config: object,
        is_hybrid: bool,
        checkpoint_path: str,
        device: torch.device,
        support_mode: str,
        prototype_mode: str,
        support_dilate_pixels: int,
        inner_erode_pixels: int,
        score_std_scale: float,
        min_area_scale: float,
        max_area_scale: float,
        max_initial_area_fraction: float,
        background_weight: float,
        min_initial_iou: float,
    ) -> None:
        self.model = model
        self.codec = codec
        self.renderer = renderer
        self.sharpener = sharpener
        self.refiner = refiner
        self.config = config
        self.is_hybrid = bool(is_hybrid)
        self.device = device
        self.support_mode = support_mode
        self.prototype_mode = prototype_mode
        self.support_dilate_pixels = int(support_dilate_pixels)
        self.inner_erode_pixels = int(inner_erode_pixels)
        self.score_std_scale = float(score_std_scale)
        self.min_area_scale = float(min_area_scale)
        self.max_area_scale = float(max_area_scale)
        self.max_initial_area_fraction = float(max_initial_area_fraction)
        self.background_weight = float(background_weight)
        self.min_initial_iou = float(min_initial_iou)
        self.adaptor = load_radio_adaptor_from_checkpoint(
            checkpoint_path,
            "sam3",
            kind="feature_projection",
        ).to(device)
        self.adaptor = self.adaptor.half() if device.type == "cuda" else self.adaptor.float()
        self.adaptor.eval()

    def set_frame(self, viewmat: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            decoded = render_1280d(
                self.model,
                self.codec,
                self.renderer,
                self.sharpener,
                self.refiner,
                viewmat.unsqueeze(0),
                is_hybrid=self.is_hybrid,
                config=self.config,
                device=self.device,
                rgb_image=None,
            )
            projected = project_feature_map_with_adaptor(decoded, self.adaptor, normalize=True)
        return {"feature_map": projected.squeeze(0).detach()}

    def refine_from_state_with_report(
        self,
        state: Dict[str, Any],
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        feature_map = state.get("feature_map")
        if feature_map is None:
            pred = np.asarray(mask).astype(bool)
            return pred.copy(), {
                "backend": "radio_sam3_adaptor_feature_prototype",
                "attempted": False,
                "accepted": False,
                "fallback_reason": "missing_feature_map",
                "candidate_count": 0,
                "selected_index": -1,
                "best_initial_overlap": 0.0,
                "selected_score": 0.0,
                "box_prompt_format": "",
                "box_prompt_cxcywh_norm": None,
                "box_prompt_xyxy_pixels": None,
            }
        return refine_mask_with_sam3_adaptor_features(
            feature_map,
            mask,
            support_mode=self.support_mode,
            prototype_mode=self.prototype_mode,
            support_dilate_pixels=self.support_dilate_pixels,
            inner_erode_pixels=self.inner_erode_pixels,
            score_std_scale=self.score_std_scale,
            min_area_scale=self.min_area_scale,
            max_area_scale=self.max_area_scale,
            max_initial_area_fraction=self.max_initial_area_fraction,
            background_weight=self.background_weight,
            min_initial_iou=self.min_initial_iou,
        )

    def refine_grabcut_from_state_with_report(
        self,
        state: Dict[str, Any],
        mask: np.ndarray,
        *,
        iterations: int,
        dilate_pixels: int,
        erode_pixels: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        feature_map = state.get("feature_map")
        if feature_map is None:
            pred = np.asarray(mask).astype(bool)
            return pred.copy(), {
                "backend": "radio_sam3_adaptor_feature_grabcut",
                "attempted": False,
                "accepted": False,
                "fallback_reason": "missing_feature_map",
                "candidate_count": 0,
                "selected_index": -1,
                "best_initial_overlap": 0.0,
                "selected_score": 0.0,
                "box_prompt_format": "",
                "box_prompt_cxcywh_norm": None,
                "box_prompt_xyxy_pixels": None,
            }
        return refine_mask_with_sam3_feature_grabcut(
            feature_map,
            mask,
            iterations=iterations,
            dilate_pixels=dilate_pixels,
            erode_pixels=erode_pixels,
            min_initial_iou=self.min_initial_iou,
            min_refined_area_ratio=self.min_area_scale,
            max_refined_area_ratio=self.max_area_scale,
        )


def _load_sam3_mask_logit_projector_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> FoundationMaskLogitProjector:
    ckpt = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    state = ckpt.get("foundation_cache_projectors_state_dict")
    if not isinstance(state, dict):
        raise KeyError(
            f"Checkpoint does not contain foundation_cache_projectors_state_dict: {checkpoint_path}"
        )
    prefix = "sam3."
    sam3_state = {
        key[len(prefix):]: value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    if not sam3_state:
        raise KeyError(f"Checkpoint does not contain a sam3 projector: {checkpoint_path}")
    conv0 = sam3_state.get("net.0.weight")
    conv2 = sam3_state.get("net.2.weight")
    if conv0 is None or conv2 is None:
        raise KeyError("SAM3 projector state must contain net.0.weight and net.2.weight")
    projector = FoundationMaskLogitProjector(
        input_dim=int(conv0.shape[1]),
        hidden_dim=int(conv0.shape[0]),
        output_masks=int(conv2.shape[0]),
    ).to(device)
    projector.load_state_dict(sam3_state, strict=True)
    projector = projector.half() if device.type == "cuda" else projector.float()
    return projector.eval()


class Sam3MaskHeadRefiner:
    """Feature-only trained SAM3 mask-logit head readout for direct-3D masks."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        codec: torch.nn.Module,
        renderer: FeatureFieldRenderer,
        sharpener: torch.nn.Module,
        refiner: Optional[torch.nn.Module],
        config: object,
        is_hybrid: bool,
        checkpoint_path: str,
        device: torch.device,
        logit_threshold: float,
        min_initial_iou: float,
        max_initial_area_fraction: float,
    ) -> None:
        self.model = model
        self.codec = codec
        self.renderer = renderer
        self.sharpener = sharpener
        self.refiner = refiner
        self.config = config
        self.is_hybrid = bool(is_hybrid)
        self.device = device
        self.projector = _load_sam3_mask_logit_projector_from_checkpoint(
            checkpoint_path,
            device=device,
        )
        self.logit_threshold = float(logit_threshold)
        self.min_initial_iou = float(min_initial_iou)
        self.max_initial_area_fraction = float(max_initial_area_fraction)

    def set_frame(self, viewmat: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            decoded = render_1280d(
                self.model,
                self.codec,
                self.renderer,
                self.sharpener,
                self.refiner,
                viewmat.unsqueeze(0),
                is_hybrid=self.is_hybrid,
                config=self.config,
                device=self.device,
                rgb_image=None,
            )
            try:
                first_param = next(self.projector.parameters())
            except StopIteration:
                first_param = None
            if first_param is not None:
                decoded = decoded.to(device=first_param.device, dtype=first_param.dtype)
            logits = self.projector(decoded)
        return {"mask_logits": logits.detach()}

    def refine_from_state_with_report(
        self,
        state: Dict[str, Any],
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        logits = state.get("mask_logits")
        if logits is None:
            pred = np.asarray(mask).astype(bool)
            return pred.copy(), {
                "backend": "ctf_sam3_mask_logit_projector",
                "attempted": False,
                "accepted": False,
                "fallback_reason": "missing_mask_logits",
                "candidate_count": 0,
                "selected_index": -1,
                "best_initial_overlap": 0.0,
                "selected_score": 0.0,
                "box_prompt_format": "",
                "box_prompt_cxcywh_norm": None,
                "box_prompt_xyxy_pixels": None,
            }
        return choose_sam3_mask_head_refined_mask_with_report(
            mask,
            logits,
            logit_threshold=self.logit_threshold,
            min_initial_iou=self.min_initial_iou,
            max_initial_area_fraction=self.max_initial_area_fraction,
        )


class PromptConditionedSam3MaskHeadRefiner:
    """Feature-only prompt-conditioned SAM3 pseudo-mask head readout."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        codec: torch.nn.Module,
        renderer: FeatureFieldRenderer,
        sharpener: torch.nn.Module,
        refiner: Optional[torch.nn.Module],
        config: object,
        is_hybrid: bool,
        checkpoint_path: str,
        text_embedding_cache: str,
        device: torch.device,
        logit_threshold: float,
        min_initial_iou: float,
        max_initial_area_fraction: float,
        min_refined_area_ratio: float,
        max_refined_area_ratio: float,
        support_dilate: int,
        coarse_dilate: int,
        coarse_threshold: float,
        min_quality: float,
    ) -> None:
        self.model = model
        self.codec = codec
        self.renderer = renderer
        self.sharpener = sharpener
        self.refiner = refiner
        self.config = config
        self.is_hybrid = bool(is_hybrid)
        self.device = device
        ckpt = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
        state = ckpt.get("prompt_mask_head_state_dict")
        if not isinstance(state, dict):
            raise KeyError(
                f"Checkpoint does not contain prompt_mask_head_state_dict: {checkpoint_path}"
            )
        self.text_embeddings = _load_text_embedding_map(text_embedding_cache)
        if not self.text_embeddings:
            raise ValueError(f"No text embeddings found in {text_embedding_cache}")
        prompt_dim = int(ckpt.get("prompt_dim", next(iter(self.text_embeddings.values())).numel()))
        feature_dim = int(ckpt.get("feature_dim", 1280))
        hidden_dim = int(ckpt.get("hidden_dim", 128))
        self.target_size = tuple(int(v) for v in ckpt.get("target_size", (240, 320)))
        self.head = PromptConditionedMaskHead(
            feature_dim=feature_dim,
            prompt_dim=prompt_dim,
            hidden_dim=hidden_dim,
            predict_quality=bool(ckpt.get("predict_quality", False)),
        ).to(device)
        self.head.load_state_dict(state, strict=True)
        self.head = self.head.float()
        self.head.eval()
        self.logit_threshold = float(logit_threshold)
        self.min_initial_iou = float(min_initial_iou)
        self.max_initial_area_fraction = float(max_initial_area_fraction)
        self.min_refined_area_ratio = float(min_refined_area_ratio)
        self.max_refined_area_ratio = float(max_refined_area_ratio)
        self.support_dilate = int(support_dilate)
        self.coarse_dilate = int(coarse_dilate)
        self.coarse_threshold = float(coarse_threshold)
        self.min_quality = float(min_quality)
        self._normalised_text = {
            re.sub(r"\s+", " ", key.strip().lower()): value
            for key, value in self.text_embeddings.items()
        }

    def _prompt_for_category(self, category: str) -> Optional[torch.Tensor]:
        if category in self.text_embeddings:
            return self.text_embeddings[category]
        return self._normalised_text.get(re.sub(r"\s+", " ", str(category).strip().lower()))

    def missing_categories(self, categories: Iterable[str]) -> List[str]:
        return [
            str(category)
            for category in categories
            if self._prompt_for_category(str(category)) is None
        ]

    def set_frame(self, viewmat: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            decoded = render_1280d(
                self.model,
                self.codec,
                self.renderer,
                self.sharpener,
                self.refiner,
                viewmat.unsqueeze(0),
                is_hybrid=self.is_hybrid,
                config=self.config,
                device=self.device,
                rgb_image=None,
            )
            decoded = F.interpolate(
                decoded.float(),
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )
            try:
                first_param = next(self.head.parameters())
                decoded = decoded.to(device=first_param.device, dtype=first_param.dtype)
            except StopIteration:
                pass
        return {"feature_map": decoded.detach()}

    def refine_from_state_with_report(
        self,
        state: Dict[str, Any],
        mask: np.ndarray,
        category: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        feature_map = state.get("feature_map")
        prompt = self._prompt_for_category(category)
        if feature_map is None or prompt is None:
            pred = np.asarray(mask).astype(bool)
            return pred.copy(), {
                "backend": "prompt_conditioned_ctf_sam3_mask_head_no_rgb",
                "attempted": False,
                "accepted": False,
                "fallback_reason": "missing_feature_map" if feature_map is None else "missing_text_prompt",
                "candidate_count": 0,
                "selected_index": -1,
                "best_initial_overlap": 0.0,
                "selected_score": 0.0,
                "box_prompt_format": "",
                "box_prompt_cxcywh_norm": None,
                "box_prompt_xyxy_pixels": None,
            }
        return refine_mask_with_prompt_conditioned_sam3_head(
            feature_map=feature_map,
            prompt_embedding=prompt,
            coarse_mask=mask,
            head=self.head,
            logit_threshold=self.logit_threshold,
            min_initial_iou=self.min_initial_iou,
            max_initial_area_fraction=self.max_initial_area_fraction,
            min_refined_area_ratio=self.min_refined_area_ratio,
            max_refined_area_ratio=self.max_refined_area_ratio,
            support_dilate=self.support_dilate,
            coarse_dilate=self.coarse_dilate,
            coarse_threshold=self.coarse_threshold,
            min_quality=self.min_quality,
        )


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
    component_guard_min_largest_fraction: float,
    component_guard_min_total_pixels_for_multicomponent: int,
    score_component_guard_min_mass_fraction: float,
    score_component_guard_min_mean_fraction: float,
    score_component_guard_max_components: int,
    score_component_guard_min_recovery_pixels: int,
    sam3_refinement_geometry_gate: bool,
    sam3_refinement_gate_min_area_ratio: float,
    sam3_refinement_gate_max_area_ratio: float,
    sam3_refinement_gate_min_boundary_gain: float,
    sam3_box_refiner: Optional[Sam3BoxMaskRefiner],
    sam3_adaptor_refiner: Optional[Sam3AdaptorMaskRefiner],
    sam3_mask_head_refiner: Optional[Sam3MaskHeadRefiner],
    sam3_prompt_mask_head_refiner: Optional[PromptConditionedSam3MaskHeadRefiner],
    sam3_prompt_mask_head_initial_refinement: str,
    sam3_prompt_mask_head_oracle_prompt: str,
    sam3_prompt_mask_head_min_heatmap_mean_ratio: float,
    sam3_prompt_mask_head_min_heatmap_mass_ratio: float,
    sam3_prompt_mask_head_require_peak_in_refined: bool,
    min_select: int,
    output_dir: Path,
    save_masks: bool,
    save_geometry_maps: bool,
    device: torch.device,
) -> Dict:
    selected = select_gaussians_from_scores(
        scores,
        spec,
        min_select=min_select,
    )
    ranking_scores = compute_selection_ranking_scores(scores, mode=spec.mode)
    selected = apply_selection_ratio_bounds(
        selected,
        ranking_scores,
        min_ratio=selection_min_ratio,
        max_ratio=selection_max_ratio,
        min_select=min_select,
    )
    if selection_refinement == "proposal_components":
        selected = select_gaussians_by_proposal_components(
            scores,
            model.get_xyz().detach().cpu(),
            support_ratio=component_support_ratio,
            resolution=component_resolution,
            keep_components=component_keep,
            min_component_size=component_min_size,
            rank_by=component_rank_by,
        )
    elif selection_refinement == "seed_expand_components":
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
    needs_prompt_heatmap = mask_refinement == "rgb_grabcut_score_component_guard" or (
        mask_refinement == "sam3_prompt_mask_head"
        and (
            sam3_prompt_mask_head_initial_refinement != "none"
            or float(sam3_prompt_mask_head_min_heatmap_mean_ratio) > 0.0
            or float(sam3_prompt_mask_head_min_heatmap_mass_ratio) > 0.0
            or bool(sam3_prompt_mask_head_require_peak_in_refined)
        )
    )
    score_heatmap_proxy = (
        GaussianSelectionProxy(
            model,
            normalize_score_heatmap_features(ranking_scores).to(device=device, dtype=torch.float32),
        )
        if needs_prompt_heatmap
        else None
    )

    ious: List[float] = []
    boundary_fs: List[float] = []
    trimap_ious: List[float] = []
    initial_ious: List[float] = []
    initial_boundary_fs: List[float] = []
    initial_trimap_ious: List[float] = []
    delta_ious: List[float] = []
    delta_boundary_fs: List[float] = []
    delta_trimap_ious: List[float] = []
    sam3_reports: List[Dict[str, Any]] = []
    per_category: Dict[str, List[float]] = {cat: [] for cat in scene_categories}
    per_category_boundary: Dict[str, List[float]] = {cat: [] for cat in scene_categories}
    per_category_trimap: Dict[str, List[float]] = {cat: [] for cat in scene_categories}
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
            alpha_np = rendered["alpha_map"].detach().float().cpu().numpy()
            depth_np = rendered["depth_map"].detach().float().cpu().numpy()
            score_heatmaps = (
                renderer.render_features(score_heatmap_proxy, viewmat)["feature_map"]
                .detach()
                .float()
                .cpu()
                .numpy()
                if score_heatmap_proxy is not None
                else None
            )
        geometry_maps = geometry_discontinuity_maps(alpha_np, depth_np) if save_geometry_maps else None
        if save_geometry_maps and geometry_maps is not None:
            for map_name, map_value in geometry_maps.items():
                save_float_heatmap(
                    output_dir
                    / "geometry_maps"
                    / spec.tag
                    / scene
                    / f"frame_{frame_id:05d}_{map_name}.png",
                    map_value,
                )
        gt_masks = build_gt_masks(frame_objects, scene_categories, img_h, img_w)
        rgb_for_refinement = None
        sam3_state = None
        sam3_adaptor_state = None
        sam3_mask_head_state = None
        needs_rgb_refinement = mask_refinement in {
            "rgb_grabcut",
            "largest_component_rgb_grabcut",
            "rgb_grabcut_largest_component",
            "rgb_grabcut_component_guard",
            "rgb_grabcut_score_component_guard",
            "sam3_box",
        }
        if needs_rgb_refinement:
            rgb_for_refinement = load_lerf_rgb_frame(scene, frame_id, getattr(dataset, "scene_root", ""))
            if mask_refinement == "sam3_box" and rgb_for_refinement is not None and sam3_box_refiner is not None:
                sam3_state = sam3_box_refiner.set_image(rgb_for_refinement)
        if mask_refinement in {"sam3_adaptor_boundary", "sam3_adaptor_grabcut"} and sam3_adaptor_refiner is not None:
            sam3_adaptor_state = sam3_adaptor_refiner.set_frame(viewmat)
        if mask_refinement == "sam3_mask_head" and sam3_mask_head_refiner is not None:
            sam3_mask_head_state = sam3_mask_head_refiner.set_frame(viewmat)
        if (
            mask_refinement == "sam3_prompt_mask_head"
            and sam3_prompt_mask_head_refiner is not None
        ):
            sam3_mask_head_state = sam3_prompt_mask_head_refiner.set_frame(viewmat)

        active_cats = sorted({obj["category"] for obj in frame_objects})
        frame_scores: Dict[str, float] = {}
        for cat in active_cats:
            if cat not in per_category:
                continue
            cat_idx = scene_categories.index(cat)
            gt = gt_masks[cat]
            if gt.sum() == 0:
                continue
            pred = silhouette[cat_idx] > float(silhouette_threshold)
            initial_pred = pred.copy()
            prompt_initial_pred = pred.copy()
            prompt_heatmap = score_heatmaps[cat_idx] if score_heatmaps is not None else silhouette[cat_idx]
            component_guard_report: Optional[Dict[str, float | int | bool]] = None
            if mask_refinement == "sam3_prompt_mask_head":
                prompt_initial_pred = build_direct3d_prompt_initial_mask(
                    pred,
                    prompt_heatmap,
                    initial_refinement=sam3_prompt_mask_head_initial_refinement,
                )
                prompt_initial_pred = build_direct3d_oracle_prompt_initial_mask(
                    prompt_initial_pred,
                    gt,
                    mode=sam3_prompt_mask_head_oracle_prompt,
                )
            sam3_report: Optional[Dict[str, Any]] = None
            if mask_refinement in {"largest_component", "largest_component_rgb_grabcut"}:
                pred = keep_largest_mask_component(pred)
            if (
                mask_refinement
                in {
                    "rgb_grabcut",
                    "largest_component_rgb_grabcut",
                    "rgb_grabcut_largest_component",
                    "rgb_grabcut_component_guard",
                    "rgb_grabcut_score_component_guard",
                }
                and rgb_for_refinement is not None
            ):
                pred = refine_mask_with_rgb_edges(
                    rgb_for_refinement,
                    pred,
                    iterations=mask_refinement_iters,
                    dilate_pixels=mask_refinement_dilate,
                    erode_pixels=mask_refinement_erode,
                )
            if mask_refinement == "rgb_grabcut_largest_component":
                pred = keep_largest_mask_component(pred)
            if mask_refinement == "rgb_grabcut_component_guard":
                pred, component_guard_report = keep_largest_mask_component_if_dominant(
                    pred,
                    min_largest_fraction=component_guard_min_largest_fraction,
                    min_total_pixels_for_multicomponent=component_guard_min_total_pixels_for_multicomponent,
                )
            if mask_refinement == "rgb_grabcut_score_component_guard":
                pred, component_guard_report = keep_mask_components_by_heatmap_score(
                    pred,
                    prompt_heatmap,
                    min_mass_fraction=score_component_guard_min_mass_fraction,
                    min_mean_fraction=score_component_guard_min_mean_fraction,
                    max_components=score_component_guard_max_components,
                    min_total_pixels_for_multicomponent=component_guard_min_total_pixels_for_multicomponent,
                    min_recovery_pixels=score_component_guard_min_recovery_pixels,
                )
            if mask_refinement == "sam3_box" and sam3_state is not None and sam3_box_refiner is not None:
                pred, sam3_report = sam3_box_refiner.refine_from_state_with_report(sam3_state, pred)
            elif mask_refinement == "sam3_box":
                if rgb_for_refinement is None:
                    fallback_reason = "missing_rgb_frame"
                elif sam3_box_refiner is None:
                    fallback_reason = "missing_sam3_refiner"
                else:
                    fallback_reason = "missing_sam3_state"
                sam3_report = {
                    "attempted": False,
                    "accepted": False,
                    "fallback_reason": fallback_reason,
                    "candidate_count": 0,
                    "selected_index": -1,
                    "best_initial_overlap": 0.0,
                    "selected_score": 0.0,
                    "box_prompt_format": "normalized_cxcywh",
                    "box_prompt_cxcywh_norm": None,
                    "box_prompt_xyxy_pixels": None,
                }
            if (
                mask_refinement == "sam3_adaptor_boundary"
                and sam3_adaptor_state is not None
                and sam3_adaptor_refiner is not None
            ):
                candidate_pred, sam3_report = sam3_adaptor_refiner.refine_from_state_with_report(
                    sam3_adaptor_state,
                    pred,
                )
                if sam3_refinement_geometry_gate:
                    pred, sam3_report = apply_geometry_gate_to_sam3_report(
                        initial_pred,
                        candidate_pred,
                        sam3_report,
                        alpha_np,
                        depth_np,
                        min_area_ratio=sam3_refinement_gate_min_area_ratio,
                        max_area_ratio=sam3_refinement_gate_max_area_ratio,
                        min_boundary_gain=sam3_refinement_gate_min_boundary_gain,
                    )
                else:
                    pred = candidate_pred
            elif mask_refinement == "sam3_adaptor_boundary":
                sam3_report = {
                    "backend": "radio_sam3_adaptor_feature_prototype",
                    "attempted": False,
                    "accepted": False,
                    "fallback_reason": "missing_sam3_adaptor_refiner",
                    "candidate_count": 0,
                    "selected_index": -1,
                    "best_initial_overlap": 0.0,
                    "selected_score": 0.0,
                    "box_prompt_format": "",
                    "box_prompt_cxcywh_norm": None,
                    "box_prompt_xyxy_pixels": None,
                }
            if (
                mask_refinement == "sam3_adaptor_grabcut"
                and sam3_adaptor_state is not None
                and sam3_adaptor_refiner is not None
            ):
                candidate_pred, sam3_report = sam3_adaptor_refiner.refine_grabcut_from_state_with_report(
                    sam3_adaptor_state,
                    pred,
                    iterations=mask_refinement_iters,
                    dilate_pixels=mask_refinement_dilate,
                    erode_pixels=mask_refinement_erode,
                )
                if sam3_refinement_geometry_gate:
                    pred, sam3_report = apply_geometry_gate_to_sam3_report(
                        initial_pred,
                        candidate_pred,
                        sam3_report,
                        alpha_np,
                        depth_np,
                        min_area_ratio=sam3_refinement_gate_min_area_ratio,
                        max_area_ratio=sam3_refinement_gate_max_area_ratio,
                        min_boundary_gain=sam3_refinement_gate_min_boundary_gain,
                    )
                else:
                    pred = candidate_pred
            elif mask_refinement == "sam3_adaptor_grabcut":
                sam3_report = {
                    "backend": "radio_sam3_adaptor_feature_grabcut",
                    "attempted": False,
                    "accepted": False,
                    "fallback_reason": "missing_sam3_adaptor_refiner",
                    "candidate_count": 0,
                    "selected_index": -1,
                    "best_initial_overlap": 0.0,
                    "selected_score": 0.0,
                    "box_prompt_format": "",
                    "box_prompt_cxcywh_norm": None,
                    "box_prompt_xyxy_pixels": None,
                }
            if (
                mask_refinement == "sam3_mask_head"
                and sam3_mask_head_state is not None
                and sam3_mask_head_refiner is not None
            ):
                candidate_pred, sam3_report = sam3_mask_head_refiner.refine_from_state_with_report(
                    sam3_mask_head_state,
                    pred,
                )
                if sam3_refinement_geometry_gate:
                    pred, sam3_report = apply_geometry_gate_to_sam3_report(
                        initial_pred,
                        candidate_pred,
                        sam3_report,
                        alpha_np,
                        depth_np,
                        min_area_ratio=sam3_refinement_gate_min_area_ratio,
                        max_area_ratio=sam3_refinement_gate_max_area_ratio,
                        min_boundary_gain=sam3_refinement_gate_min_boundary_gain,
                    )
                else:
                    pred = candidate_pred
            elif mask_refinement == "sam3_mask_head":
                sam3_report = {
                    "backend": "ctf_sam3_mask_logit_projector",
                    "attempted": False,
                    "accepted": False,
                    "fallback_reason": "missing_sam3_mask_head_refiner",
                    "candidate_count": 0,
                    "selected_index": -1,
                    "best_initial_overlap": 0.0,
                    "selected_score": 0.0,
                    "box_prompt_format": "",
                    "box_prompt_cxcywh_norm": None,
                    "box_prompt_xyxy_pixels": None,
                }
            if (
                mask_refinement == "sam3_prompt_mask_head"
                and sam3_mask_head_state is not None
                and sam3_prompt_mask_head_refiner is not None
            ):
                candidate_pred, sam3_report = sam3_prompt_mask_head_refiner.refine_from_state_with_report(
                    sam3_mask_head_state,
                    prompt_initial_pred,
                    cat,
                )
                heatmap_guard_report: Optional[Dict[str, Any]] = None
                geometry_gate_report: Optional[Dict[str, Any]] = None
                if (
                    float(sam3_prompt_mask_head_min_heatmap_mean_ratio) > 0.0
                    or float(sam3_prompt_mask_head_min_heatmap_mass_ratio) > 0.0
                    or bool(sam3_prompt_mask_head_require_peak_in_refined)
                ):
                    candidate_pred, heatmap_guard_report = apply_sam3_prompt_heatmap_guard(
                        prompt_initial_pred,
                        candidate_pred,
                        prompt_heatmap,
                        min_mean_ratio=sam3_prompt_mask_head_min_heatmap_mean_ratio,
                        min_mass_ratio=sam3_prompt_mask_head_min_heatmap_mass_ratio,
                        require_peak_in_refined=sam3_prompt_mask_head_require_peak_in_refined,
                    )
                    if not bool(heatmap_guard_report.get("accepted", False)):
                        sam3_report["fallback_reason"] = str(
                            heatmap_guard_report.get("fallback_reason", "")
                        )
                        sam3_report["accepted"] = False
                if sam3_refinement_geometry_gate and bool(sam3_report.get("accepted", False)):
                    candidate_pred, geometry_gate_report = choose_refined_mask_by_geometry_with_report(
                        initial_pred,
                        candidate_pred,
                        alpha_np,
                        depth_np,
                        min_area_ratio=sam3_refinement_gate_min_area_ratio,
                        max_area_ratio=sam3_refinement_gate_max_area_ratio,
                        min_boundary_gain=sam3_refinement_gate_min_boundary_gain,
                    )
                pred, sam3_report = finalize_prompt_conditioned_sam3_mask(
                    initial_pred,
                    prompt_initial_pred,
                    candidate_pred,
                    sam3_report,
                    heatmap_guard_report=heatmap_guard_report,
                    geometry_gate_report=geometry_gate_report,
                )
                sam3_report["initial_refinement"] = sam3_prompt_mask_head_initial_refinement
                sam3_report["oracle_prompt_mode"] = sam3_prompt_mask_head_oracle_prompt
            elif mask_refinement == "sam3_prompt_mask_head":
                sam3_report = {
                    "backend": "prompt_conditioned_ctf_sam3_mask_head_no_rgb",
                    "attempted": False,
                    "accepted": False,
                    "fallback_reason": "missing_sam3_prompt_mask_head_refiner",
                    "candidate_count": 0,
                    "selected_index": -1,
                    "best_initial_overlap": 0.0,
                    "selected_score": 0.0,
                    "box_prompt_format": "",
                    "box_prompt_cxcywh_norm": None,
                    "box_prompt_xyxy_pixels": None,
                    "oracle_prompt_mode": sam3_prompt_mask_head_oracle_prompt,
                }
            initial_overlap = mask_overlap_stats(initial_pred, gt)
            initial_iou = float(initial_overlap["iou"])
            initial_boundary_f = boundary_f_score(initial_pred, gt)
            initial_trimap_score = trimap_iou(initial_pred, gt)
            overlap = mask_overlap_stats(pred, gt)
            iou = float(overlap["iou"])
            boundary_f = boundary_f_score(pred, gt)
            trimap_score = trimap_iou(pred, gt)
            initial_ious.append(initial_iou)
            initial_boundary_fs.append(initial_boundary_f)
            initial_trimap_ious.append(initial_trimap_score)
            delta_ious.append(iou - initial_iou)
            delta_boundary_fs.append(boundary_f - initial_boundary_f)
            delta_trimap_ious.append(trimap_score - initial_trimap_score)
            ious.append(iou)
            boundary_fs.append(boundary_f)
            trimap_ious.append(trimap_score)
            per_category[cat].append(iou)
            per_category_boundary[cat].append(boundary_f)
            per_category_trimap[cat].append(trimap_score)
            frame_scores[cat] = iou
            query_details.append(
                query_detail := {
                    "frame": f"frame_{frame_id:05d}",
                    "frame_id": int(frame_id),
                    "category": cat,
                    "iou": iou,
                    "boundary_f": boundary_f,
                    "trimap_iou": trimap_score,
                    "initial_iou": initial_iou,
                    "initial_boundary_f": initial_boundary_f,
                    "initial_trimap_iou": initial_trimap_score,
                    "delta_iou": iou - initial_iou,
                    "delta_boundary_f": boundary_f - initial_boundary_f,
                    "delta_trimap_iou": trimap_score - initial_trimap_score,
                    "initial_pred_pixels": int(initial_overlap["pred_pixels"]),
                    "pred_pixels": int(overlap["pred_pixels"]),
                    "gt_pixels": int(overlap["gt_pixels"]),
                    "intersection_pixels": int(overlap["intersection_pixels"]),
                    "union_pixels": int(overlap["union_pixels"]),
                    "overselect_ratio": float(overlap["overselect_ratio"]),
                    "selected_gaussians": int(selected[:, cat_idx].sum().item()),
                }
            )
            if sam3_report is not None:
                sam3_reports.append(sam3_report)
                query_detail.update(
                    {
                        "sam3_accepted": bool(sam3_report.get("accepted", False)),
                        "sam3_attempted": bool(sam3_report.get("attempted", False)),
                        "sam3_fallback_reason": str(sam3_report.get("fallback_reason", "")),
                        "sam3_candidate_count": int(sam3_report.get("candidate_count", 0)),
                        "sam3_selected_index": int(sam3_report.get("selected_index", -1)),
                        "sam3_best_initial_overlap": float(
                            sam3_report.get("best_initial_overlap", 0.0)
                        ),
                        "sam3_selected_score": float(sam3_report.get("selected_score", 0.0)),
                        "sam3_oracle_prompt_mode": str(
                            sam3_report.get("oracle_prompt_mode", "")
                        ),
                        "sam3_box_prompt_format": str(
                            sam3_report.get("box_prompt_format", "")
                        ),
                        "sam3_box_prompt_cxcywh_norm": sam3_report.get(
                            "box_prompt_cxcywh_norm"
                        ),
                        "sam3_box_prompt_xyxy_pixels": sam3_report.get(
                            "box_prompt_xyxy_pixels"
                        ),
                    }
                )
                for report_key in (
                    "geometry_gate_enabled",
                    "geometry_gate_accepted",
                    "geometry_gate_reason",
                    "geometry_gate_initial_score",
                    "geometry_gate_refined_score",
                    "geometry_gate_boundary_gain",
                    "geometry_gate_area_ratio",
                ):
                    if report_key in sam3_report:
                        value = sam3_report[report_key]
                        if isinstance(value, (bool, str)):
                            query_detail[report_key] = value
                        else:
                            query_detail[report_key] = float(value)
            if geometry_maps is not None:
                geom_metrics = compute_geometry_boundary_alignment(pred, gt, alpha_np, depth_np)
                query_detail.update(
                    {
                        key: float(value) if isinstance(value, float) else int(value)
                        for key, value in geom_metrics.items()
                    }
                )
                for map_name in geometry_maps:
                    query_detail[f"geometry_{map_name}_path"] = str(
                        (
                            output_dir
                            / "geometry_maps"
                            / spec.tag
                            / scene
                            / f"frame_{frame_id:05d}_{map_name}.png"
                        ).relative_to(output_dir)
                    )
                overlay_path = (
                    output_dir
                    / "geometry_overlays"
                    / spec.tag
                    / scene
                    / f"frame_{frame_id:05d}_{cat.replace('/', '_')}.png"
                )
                save_geometry_alignment_overlay(
                    overlay_path,
                    geometry_maps["discontinuity"],
                    pred,
                    gt,
                )
                query_detail["geometry_overlay_path"] = str(overlay_path.relative_to(output_dir))
            if component_guard_report is not None:
                query_detail.update(
                    {
                        key: bool(value) if isinstance(value, bool) else float(value)
                        for key, value in component_guard_report.items()
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
            "boundary_f": float(np.asarray(per_category_boundary[cat], dtype=np.float32).mean())
            if per_category_boundary[cat]
            else 0.0,
            "trimap_iou": float(np.asarray(per_category_trimap[cat], dtype=np.float32).mean())
            if per_category_trimap[cat]
            else 0.0,
            "selected_gaussians": int(selected[:, ci].sum().item()),
        }
        for ci, (cat, vals) in enumerate(per_category.items())
    }
    summary = summarize_ious(ious)
    initial_summary = summarize_ious(initial_ious)
    boundary_summary = bootstrap_mean_ci(boundary_fs)
    trimap_summary = bootstrap_mean_ci(trimap_ious)
    initial_boundary_summary = bootstrap_mean_ci(initial_boundary_fs)
    initial_trimap_summary = bootstrap_mean_ci(initial_trimap_ious)
    delta_iou_summary = bootstrap_mean_ci(delta_ious)
    delta_boundary_summary = bootstrap_mean_ci(delta_boundary_fs)
    delta_trimap_summary = bootstrap_mean_ci(delta_trimap_ious)
    sam3_attempt_count = sum(1 for report in sam3_reports if bool(report.get("attempted", True)))
    sam3_skip_count = len(sam3_reports) - sam3_attempt_count
    sam3_accept_count = sum(1 for report in sam3_reports if bool(report.get("accepted", False)))
    sam3_fallback_reasons: Dict[str, int] = {}
    for report in sam3_reports:
        reason = str(report.get("fallback_reason", ""))
        sam3_fallback_reasons[reason] = sam3_fallback_reasons.get(reason, 0) + 1
    sam3_rejection_reasons = {
        reason: count
        for reason, count in sam3_fallback_reasons.items()
        if reason and reason != "accepted"
    }
    summary.update(
        {
            "boundary_f": float(boundary_summary["mean"]),
            "trimap_iou": float(trimap_summary["mean"]),
            "initial_miou": float(initial_summary["miou"]),
            "initial_acc025": float(initial_summary["acc025"]),
            "initial_acc050": float(initial_summary["acc050"]),
            "initial_boundary_f": float(initial_boundary_summary["mean"]),
            "initial_trimap_iou": float(initial_trimap_summary["mean"]),
            "delta_miou": float(delta_iou_summary["mean"]),
            "delta_boundary_f": float(delta_boundary_summary["mean"]),
            "delta_trimap_iou": float(delta_trimap_summary["mean"]),
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
            "component_guard_min_largest_fraction": component_guard_min_largest_fraction,
            "component_guard_min_total_pixels_for_multicomponent": int(
                component_guard_min_total_pixels_for_multicomponent
            ),
            "score_component_guard_min_mass_fraction": float(score_component_guard_min_mass_fraction),
            "score_component_guard_min_mean_fraction": float(score_component_guard_min_mean_fraction),
            "score_component_guard_max_components": int(score_component_guard_max_components),
            "score_component_guard_min_recovery_pixels": int(
                score_component_guard_min_recovery_pixels
            ),
            "sam3_refinement_count": int(len(sam3_reports)),
            "sam3_attempt_count": int(sam3_attempt_count),
            "sam3_skip_count": int(sam3_skip_count),
            "sam3_accept_count": int(sam3_accept_count),
            "sam3_accept_rate": float(sam3_accept_count / max(sam3_attempt_count, 1)),
            "sam3_fallback_reasons": sam3_fallback_reasons,
            "sam3_rejection_reasons": sam3_rejection_reasons,
            "per_category": per_cat_summary,
            "per_frame": per_frame,
            "query_details": query_details,
            "initial_iou_buckets": summarize_initial_iou_buckets(query_details),
            "bootstrap_miou": bootstrap_mean_ci(ious),
            "bootstrap_boundary_f": boundary_summary,
            "bootstrap_trimap_iou": trimap_summary,
            "bootstrap_initial_miou": bootstrap_mean_ci(initial_ious),
            "bootstrap_initial_boundary_f": initial_boundary_summary,
            "bootstrap_initial_trimap_iou": initial_trimap_summary,
            "bootstrap_delta_miou": delta_iou_summary,
            "bootstrap_delta_boundary_f": delta_boundary_summary,
            "bootstrap_delta_trimap_iou": delta_trimap_summary,
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
    registered_feature_cache_path: Optional[str],
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
    proposal_smoothing: str,
    proposal_voxel_size: float,
    proposal_smoothing_alpha: float,
    proposal_min_count: int,
    proposal_smoothing_gate: str,
    proposal_margin_threshold: float,
    proposal_confidence_threshold: float,
    proposal_consensus_threshold: float,
    sam3_proposal_registration_dir: str,
    sam3_proposal_registration_alpha: float,
    sam3_proposal_registration_min_probability: float,
    sam3_proposal_registration_max_masks_per_frame: int,
    sam3_proposal_registration_gate: str,
    sam3_proposal_registration_margin_threshold: float,
    sam3_proposal_registration_query_conditioned: bool,
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
    registration_assignment_mode: str,
    registration_weight_mode: str,
    registration_confidence_blend: float,
    registration_confidence_mode: str,
    disable_registered_refiner: bool,
    use_point_summary_adapter: bool,
    point_summary_adapter_blend_alpha: float,
    point_summary_adapter_valid_mask_mode: str,
    strict_direct_head_consistency: bool,
    direct_primitive_confidence_mode: str,
    direct_primitive_confidence_blend: float,
    direct_primitive_opacity_threshold: float,
    silhouette_threshold: float,
    mask_refinement: str,
    mask_refinement_iters: int,
    mask_refinement_dilate: int,
    mask_refinement_erode: int,
    component_guard_min_largest_fraction: float,
    component_guard_min_total_pixels_for_multicomponent: int,
    score_component_guard_min_mass_fraction: float,
    score_component_guard_min_mean_fraction: float,
    score_component_guard_max_components: int,
    score_component_guard_min_recovery_pixels: int,
    sam3_refinement_geometry_gate: bool,
    sam3_refinement_gate_min_area_ratio: float,
    sam3_refinement_gate_max_area_ratio: float,
    sam3_refinement_gate_min_boundary_gain: float,
    sam3_checkpoint_path: str,
    sam3_confidence_threshold: float,
    sam3_resolution: int,
    sam3_amp_dtype: str,
    sam3_box_padding: int,
    sam3_min_initial_iou: float,
    sam3_adaptor_checkpoint: str,
    sam3_adaptor_support_mode: str,
    sam3_adaptor_prototype_mode: str,
    sam3_adaptor_support_dilate: int,
    sam3_adaptor_inner_erode: int,
    sam3_adaptor_score_std_scale: float,
    sam3_adaptor_min_area_scale: float,
    sam3_adaptor_max_area_scale: float,
    sam3_adaptor_max_initial_area_fraction: float,
    sam3_adaptor_background_weight: float,
    sam3_adaptor_min_initial_iou: float,
    sam3_mask_head_checkpoint: str,
    sam3_mask_head_logit_threshold: float,
    sam3_mask_head_min_initial_iou: float,
    sam3_mask_head_max_initial_area_fraction: float,
    sam3_prompt_mask_head_checkpoint: str,
    sam3_prompt_mask_head_text_embedding_cache: str,
    sam3_prompt_mask_head_logit_threshold: float,
    sam3_prompt_mask_head_min_initial_iou: float,
    sam3_prompt_mask_head_max_initial_area_fraction: float,
    sam3_prompt_mask_head_min_refined_area_ratio: float,
    sam3_prompt_mask_head_max_refined_area_ratio: float,
    sam3_prompt_mask_head_support_dilate: int,
    sam3_prompt_mask_head_coarse_dilate: int,
    sam3_prompt_mask_head_coarse_threshold: float,
    sam3_prompt_mask_head_min_quality: float,
    sam3_prompt_mask_head_initial_refinement: str,
    sam3_prompt_mask_head_oracle_prompt: str,
    sam3_prompt_mask_head_min_heatmap_mean_ratio: float,
    sam3_prompt_mask_head_min_heatmap_mass_ratio: float,
    sam3_prompt_mask_head_require_peak_in_refined: bool,
    allow_missing_sam3_prompt_text_embeddings: bool,
    min_select: int,
    chunk_size: int,
    official_frames_only: bool,
    save_masks: bool,
    save_geometry_maps: bool,
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
    checkpoint_for_status = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    ps_metadata = checkpoint_for_status.get("point_summary_adapter_metadata") or {}
    ps_contract = ps_metadata.get("direct_head_contract") if isinstance(ps_metadata, dict) else {}
    point_summary_adapter_context_features = str(
        (ps_contract or {}).get(
            "point_summary_adapter_context_features",
            ps_metadata.get(
                "point_summary_adapter_context_features",
                getattr(config, "point_summary_adapter_context_features", ""),
            ),
        )
        or ""
    )
    point_summary_adapter = (
        _build_point_summary_adapter(config, checkpoint_path, device)
        if use_point_summary_adapter
        else None
    )
    direct_head_eval_status = build_direct_head_eval_status(
        checkpoint_for_status,
        score_source=score_source,
        use_point_summary_adapter=use_point_summary_adapter,
        adapter_loaded=point_summary_adapter is not None,
        compact_feature_key=compact_feature_key,
        direct_readout_mode=direct_readout_mode,
        point_summary_adapter_blend_alpha=point_summary_adapter_blend_alpha,
        point_summary_adapter_valid_mask_mode=point_summary_adapter_valid_mask_mode,
        point_summary_adapter_context_features=point_summary_adapter_context_features,
        teacher_feature_space=str(getattr(config, "direct_point_teacher_cache_feature_space", "") or ""),
        teacher_cache_feature_key=str(getattr(config, "direct_point_teacher_cache_feature_key", "") or ""),
        direct_point_query_mode=str(getattr(config, "direct_point_query_mode", "") or ""),
        direct_point_gaussian_position_mode=str(
            getattr(config, "direct_point_gaussian_position_mode", "") or ""
        ),
    )
    if direct_head_eval_status.get("warnings"):
        logger.warning(
            "Direct-head eval consistency warnings: %s",
            ", ".join(str(w) for w in direct_head_eval_status["warnings"]),
        )
    enforce_direct_head_eval_consistency(
        direct_head_eval_status,
        strict=strict_direct_head_consistency,
    )
    needs_point_summary_valid_mask = (
        use_point_summary_adapter and point_summary_adapter_valid_mask_mode == "teacher_cache"
    ) or direct_primitive_confidence_mode == "teacher_cache_valid"
    if point_summary_adapter_valid_mask_mode not in {"teacher_cache", "opacity", "none"}:
        raise ValueError("point_summary_adapter_valid_mask_mode must be 'teacher_cache', 'opacity', or 'none'")
    fallback_point_summary_teacher_cache = ""
    if needs_point_summary_valid_mask:
        cache_raw = str(getattr(config, "direct_point_teacher_cache", "") or "")
        if cache_raw:
            try:
                cache_raw = cache_raw.format(scene=scene)
            except Exception:
                pass
            cache_path = Path(cache_raw)
            if not cache_path.is_absolute():
                cache_path = Path.cwd() / cache_path
            fallback_point_summary_teacher_cache = str(cache_path)
    point_summary_adapter_valid_mask = (
        _load_point_summary_adapter_valid_mask(
            checkpoint_path,
            expected_count=int(model.get_xyz().shape[0]),
            device=device,
            fallback_teacher_cache=fallback_point_summary_teacher_cache,
            expected_xyz=model.get_xyz().detach().cpu(),
        )
        if needs_point_summary_valid_mask
        else None
    )
    point_summary_adapter_view_counts = (
        _load_point_summary_adapter_view_counts(
            checkpoint_path,
            expected_count=int(model.get_xyz().shape[0]),
            device=device,
            fallback_teacher_cache=fallback_point_summary_teacher_cache,
            expected_xyz=model.get_xyz().detach().cpu(),
        )
        if use_point_summary_adapter and "view_count" in point_summary_adapter_context_features
        else None
    )
    if (
        use_point_summary_adapter
        and point_summary_adapter_valid_mask_mode == "teacher_cache"
        and point_summary_adapter_valid_mask is None
    ):
        raise RuntimeError(
            "point_summary_adapter_valid_mask_mode=teacher_cache requires an "
            "aligned teacher-cache valid mask"
        )
    if (
        use_point_summary_adapter
        and "view_count" in point_summary_adapter_context_features
        and point_summary_adapter_view_counts is None
    ):
        raise RuntimeError(
            "point_summary_adapter_context_features includes view_count but no "
            "teacher-cache view_counts could be loaded"
        )
    raw_view_count_max = ps_metadata.get("point_summary_adapter_view_count_max")
    point_summary_adapter_view_count_max = (
        float(raw_view_count_max)
        if raw_view_count_max is not None
        else (
            float(point_summary_adapter_view_counts.max().detach().cpu())
            if point_summary_adapter_view_counts is not None
            else None
        )
    )
    direct_primitive_confidence: Optional[torch.Tensor] = None
    if direct_primitive_confidence_mode == "teacher_cache_valid":
        if point_summary_adapter_valid_mask is None:
            raise RuntimeError(
                "direct_primitive_confidence_mode=teacher_cache_valid requires "
                "a valid direct_point_teacher_cache mask"
            )
        direct_primitive_confidence = point_summary_adapter_valid_mask.float()
    elif direct_primitive_confidence_mode != "none" or (
        use_point_summary_adapter and point_summary_adapter_valid_mask_mode == "opacity"
    ):
        opacity_confidence = build_opacity_primitive_confidence(
            model.get_opacity(),
            mode=(
                direct_primitive_confidence_mode
                if direct_primitive_confidence_mode != "none"
                else "opacity"
            ),
            threshold=direct_primitive_opacity_threshold,
        ).to(device=device)
        if direct_primitive_confidence_mode != "none":
            direct_primitive_confidence = opacity_confidence
        if use_point_summary_adapter and point_summary_adapter_valid_mask_mode == "opacity":
            point_summary_adapter_valid_mask = opacity_confidence > 0

    dataset = build_lerf_dataset_for_scene(
        scene,
        config,
        label_dir,
        feature_height=img_h,
        feature_width=img_w,
    )
    renderer = build_mask_renderer(config, height=img_h, width=img_w, device=device)
    sam3_box_refiner: Optional[Sam3BoxMaskRefiner] = None
    sam3_adaptor_refiner: Optional[Sam3AdaptorMaskRefiner] = None
    sam3_mask_head_refiner: Optional[Sam3MaskHeadRefiner] = None
    sam3_prompt_mask_head_refiner: Optional[PromptConditionedSam3MaskHeadRefiner] = None
    if mask_refinement == "sam3_box":
        if not sam3_checkpoint_path:
            raise ValueError("--sam3_checkpoint_path is required for --mask_refinement sam3_box")
        print(
            "  loading official SAM3 box refiner "
            f"(checkpoint={sam3_checkpoint_path}, resolution={sam3_resolution})"
        )
        sam3_box_refiner = Sam3BoxMaskRefiner(
            checkpoint_path=sam3_checkpoint_path,
            device="cuda" if device.type == "cuda" else "cpu",
            confidence_threshold=sam3_confidence_threshold,
            resolution=sam3_resolution,
            amp_dtype=sam3_amp_dtype,
            box_padding_pixels=sam3_box_padding,
            min_initial_iou=sam3_min_initial_iou,
        )
    elif mask_refinement == "sam3_adaptor_boundary":
        if not sam3_adaptor_checkpoint:
            raise ValueError("--sam3_adaptor_checkpoint is required for --mask_refinement sam3_adaptor_boundary")
        print(
            "  loading feature-only SAM3-adaptor boundary refiner "
            f"(checkpoint={sam3_adaptor_checkpoint})"
        )
        sam3_adaptor_refiner = Sam3AdaptorMaskRefiner(
            model=model,
            codec=codec,
            renderer=_renderer,
            sharpener=_sharpener,
            refiner=_refiner,
            config=config,
            is_hybrid=is_hybrid,
            checkpoint_path=sam3_adaptor_checkpoint,
            device=device,
            support_mode=sam3_adaptor_support_mode,
            prototype_mode=sam3_adaptor_prototype_mode,
            support_dilate_pixels=sam3_adaptor_support_dilate,
            inner_erode_pixels=sam3_adaptor_inner_erode,
            score_std_scale=sam3_adaptor_score_std_scale,
            min_area_scale=sam3_adaptor_min_area_scale,
            max_area_scale=sam3_adaptor_max_area_scale,
            max_initial_area_fraction=sam3_adaptor_max_initial_area_fraction,
            background_weight=sam3_adaptor_background_weight,
            min_initial_iou=sam3_adaptor_min_initial_iou,
        )
    elif mask_refinement == "sam3_adaptor_grabcut":
        if not sam3_adaptor_checkpoint:
            raise ValueError("--sam3_adaptor_checkpoint is required for --mask_refinement sam3_adaptor_grabcut")
        print(
            "  loading feature-only SAM3-adaptor GrabCut refiner "
            f"(checkpoint={sam3_adaptor_checkpoint})"
        )
        sam3_adaptor_refiner = Sam3AdaptorMaskRefiner(
            model=model,
            codec=codec,
            renderer=_renderer,
            sharpener=_sharpener,
            refiner=_refiner,
            config=config,
            is_hybrid=is_hybrid,
            checkpoint_path=sam3_adaptor_checkpoint,
            device=device,
            support_mode=sam3_adaptor_support_mode,
            prototype_mode=sam3_adaptor_prototype_mode,
            support_dilate_pixels=sam3_adaptor_support_dilate,
            inner_erode_pixels=sam3_adaptor_inner_erode,
            score_std_scale=sam3_adaptor_score_std_scale,
            min_area_scale=sam3_adaptor_min_area_scale,
            max_area_scale=sam3_adaptor_max_area_scale,
            max_initial_area_fraction=sam3_adaptor_max_initial_area_fraction,
            background_weight=sam3_adaptor_background_weight,
            min_initial_iou=sam3_adaptor_min_initial_iou,
        )
    elif mask_refinement == "sam3_mask_head":
        mask_head_checkpoint = sam3_mask_head_checkpoint or checkpoint_path
        print(
            "  loading feature-only SAM3 mask-logit head refiner "
            f"(checkpoint={mask_head_checkpoint})"
        )
        sam3_mask_head_refiner = Sam3MaskHeadRefiner(
            model=model,
            codec=codec,
            renderer=_renderer,
            sharpener=_sharpener,
            refiner=_refiner,
            config=config,
            is_hybrid=is_hybrid,
            checkpoint_path=mask_head_checkpoint,
            device=device,
            logit_threshold=sam3_mask_head_logit_threshold,
            min_initial_iou=sam3_mask_head_min_initial_iou,
            max_initial_area_fraction=sam3_mask_head_max_initial_area_fraction,
        )
    elif mask_refinement == "sam3_prompt_mask_head":
        if not sam3_prompt_mask_head_checkpoint:
            raise ValueError(
                "--sam3_prompt_mask_head_checkpoint is required for "
                "--mask_refinement sam3_prompt_mask_head"
            )
        prompt_text_cache = sam3_prompt_mask_head_text_embedding_cache or str(text_embedding_cache or "")
        if not prompt_text_cache:
            raise ValueError("--sam3_prompt_mask_head_text_embedding_cache or --text_embedding_cache is required")
        print(
            "  loading prompt-conditioned feature-only SAM3 mask head refiner "
            f"(checkpoint={sam3_prompt_mask_head_checkpoint})"
        )
        sam3_prompt_mask_head_refiner = PromptConditionedSam3MaskHeadRefiner(
            model=model,
            codec=codec,
            renderer=_renderer,
            sharpener=_sharpener,
            refiner=_refiner,
            config=config,
            is_hybrid=is_hybrid,
            checkpoint_path=sam3_prompt_mask_head_checkpoint,
            text_embedding_cache=prompt_text_cache,
            device=device,
            logit_threshold=sam3_prompt_mask_head_logit_threshold,
            min_initial_iou=sam3_prompt_mask_head_min_initial_iou,
            max_initial_area_fraction=sam3_prompt_mask_head_max_initial_area_fraction,
            min_refined_area_ratio=sam3_prompt_mask_head_min_refined_area_ratio,
            max_refined_area_ratio=sam3_prompt_mask_head_max_refined_area_ratio,
            support_dilate=sam3_prompt_mask_head_support_dilate,
            coarse_dilate=sam3_prompt_mask_head_coarse_dilate,
            coarse_threshold=sam3_prompt_mask_head_coarse_threshold,
            min_quality=sam3_prompt_mask_head_min_quality,
        )
        missing_prompts = sam3_prompt_mask_head_refiner.missing_categories(scene_categories)
        if missing_prompts:
            message = (
                "Prompt-conditioned SAM3 mask head is missing text embeddings for "
                f"{len(missing_prompts)} categories: {missing_prompts[:12]}"
            )
            if allow_missing_sam3_prompt_text_embeddings:
                logger.warning(message)
            else:
                raise ValueError(
                    message
                    + "; pass --allow_missing_sam3_prompt_text_embeddings to permit per-query fallback"
                )

    scene_text: Optional[torch.Tensor] = None
    canonical_text: Optional[torch.Tensor] = None

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
        "registration_assignment_mode": registration_assignment_mode,
        "registration_weight_mode": registration_weight_mode,
        "registration_confidence_blend": float(registration_confidence_blend),
        "registration_confidence_mode": registration_confidence_mode,
        "disable_registered_refiner": bool(disable_registered_refiner),
        "direct_primitive_confidence_mode": direct_primitive_confidence_mode,
        "direct_primitive_confidence_blend": float(direct_primitive_confidence_blend),
        "direct_primitive_opacity_threshold": float(direct_primitive_opacity_threshold),
    }
    if use_point_summary_adapter:
        score_cache_metadata.update(
            {
                "use_point_summary_adapter": True,
                "point_summary_adapter_blend_alpha": float(point_summary_adapter_blend_alpha),
                "point_summary_adapter_valid_mask": point_summary_adapter_valid_mask_mode,
            }
        )
    scores: Optional[torch.Tensor] = None
    cache_path = Path(score_cache_path) if score_cache_path else None
    if cache_path is not None:
        score_cache_info["path"] = str(cache_path)
        if cache_path.exists():
            print(f"  loading primitive score cache: {cache_path}")
            scores, registration_stats = load_score_cache(
                cache_path,
                expected_metadata=score_cache_metadata,
                expected_xyz=model.get_xyz().detach().cpu(),
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

    if scores is None:
        print("  loading SigLIP2 text embeddings")
        scene_text = load_or_generate_prompt_ensemble_embeddings(
            scene_categories,
            device,
            cache_path=text_embedding_cache,
            prompt_templates=prompt_templates,
        )
        scene_text = F.normalize(scene_text.float(), dim=-1)
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

    direct_scores: Optional[torch.Tensor] = None
    needs_direct_scores = score_source == "direct" or (
        score_source == "registered_view" and registered_view_fallback == "direct"
    )
    if scores is None and needs_direct_scores:
        if scene_text is None:
            raise RuntimeError("Internal error: text embeddings were not loaded for direct scoring")
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
            point_summary_adapter=point_summary_adapter,
            point_summary_adapter_blend_alpha=point_summary_adapter_blend_alpha,
            point_summary_adapter_valid_mask=point_summary_adapter_valid_mask,
            point_summary_adapter_context_features=point_summary_adapter_context_features,
            point_summary_adapter_view_counts=point_summary_adapter_view_counts,
            point_summary_adapter_view_count_max=point_summary_adapter_view_count_max,
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
        scores = apply_direct_primitive_confidence(
            scores,
            direct_primitive_confidence,
            blend=direct_primitive_confidence_blend,
        )
    elif score_source == "registered_view":
        if scene_text is None:
            raise RuntimeError("Internal error: text embeddings were not loaded for VPR scoring")
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
            registration_assignment_mode=registration_assignment_mode,
            registration_weight_mode=registration_weight_mode,
            registration_confidence_blend=registration_confidence_blend,
            registration_confidence_mode=registration_confidence_mode,
            fallback_scores=direct_scores if registered_view_fallback == "direct" else None,
            device=device,
            registered_feature_cache_path=registered_feature_cache_path,
            registered_feature_cache_metadata=score_cache_metadata,
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
            xyz=model.get_xyz().detach().cpu(),
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
    proposal_smoothing_stats: Dict[str, Any] = {
        "enabled": False,
        "mode": proposal_smoothing,
        "voxel_size": float(proposal_voxel_size),
        "alpha": float(proposal_smoothing_alpha),
        "min_count": int(proposal_min_count),
        "gate": proposal_smoothing_gate,
        "margin_threshold": float(proposal_margin_threshold),
        "confidence_threshold": float(proposal_confidence_threshold),
        "proposal_consensus_threshold": float(proposal_consensus_threshold),
        "num_proposals": 0,
        "num_assigned": 0,
    }
    if proposal_smoothing == "voxel" and proposal_smoothing_alpha > 0:
        print(
            "  smoothing primitive scores with proposal memory "
            f"(voxel={proposal_voxel_size:g}, alpha={proposal_smoothing_alpha:g}, "
            f"min_count={proposal_min_count}, gate={proposal_smoothing_gate}, "
            f"margin={proposal_margin_threshold:g}, "
            f"confidence={proposal_confidence_threshold:g}, "
            f"consensus={proposal_consensus_threshold:g})"
        )
        scores, proposal_smoothing_stats = smooth_scores_with_voxel_proposals(
            scores,
            model.get_xyz().detach().cpu(),
            voxel_size=proposal_voxel_size,
            alpha=proposal_smoothing_alpha,
            min_count=proposal_min_count,
            gate=proposal_smoothing_gate,
            margin_threshold=proposal_margin_threshold,
            confidence_threshold=proposal_confidence_threshold,
            proposal_consensus_threshold=proposal_consensus_threshold,
        )
    elif proposal_smoothing != "none":
        raise ValueError(f"Unsupported proposal_smoothing: {proposal_smoothing}")
    sam3_proposal_registration_stats: Dict[str, Any] = {
        "enabled": False,
        "mode": "sam3_trainview",
        "cache_root": str(sam3_proposal_registration_dir),
        "alpha": float(sam3_proposal_registration_alpha),
        "min_probability": float(sam3_proposal_registration_min_probability),
        "max_masks_per_frame": int(sam3_proposal_registration_max_masks_per_frame),
        "gate": sam3_proposal_registration_gate,
        "margin_threshold": float(sam3_proposal_registration_margin_threshold),
        "query_conditioned": bool(sam3_proposal_registration_query_conditioned),
        "num_cache_frames": 0,
        "num_used_frames": 0,
        "num_proposals": 0,
        "num_memberships": 0,
        "num_assigned": 0,
    }
    if sam3_proposal_registration_dir and sam3_proposal_registration_alpha > 0:
        print(
            "  fusing primitive scores with SAM3 training-view proposal memory "
            f"(alpha={sam3_proposal_registration_alpha:g}, "
            f"prob>={sam3_proposal_registration_min_probability:g}, "
            f"gate={sam3_proposal_registration_gate}, "
            f"margin={sam3_proposal_registration_margin_threshold:g}, "
            f"query_conditioned={sam3_proposal_registration_query_conditioned})"
        )
        scores, sam3_proposal_registration_stats = smooth_scores_with_sam3_training_view_proposals(
            scores,
            model.get_xyz().detach().cpu(),
            cache_root=sam3_proposal_registration_dir,
            scene=scene,
            scene_categories=scene_categories,
            dataset=dataset,
            image_width=img_w,
            image_height=img_h,
            alpha=sam3_proposal_registration_alpha,
            min_probability=sam3_proposal_registration_min_probability,
            max_masks_per_frame=sam3_proposal_registration_max_masks_per_frame,
            gate=sam3_proposal_registration_gate,
            margin_threshold=sam3_proposal_registration_margin_threshold,
            query_conditioned=sam3_proposal_registration_query_conditioned,
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
            component_guard_min_largest_fraction=component_guard_min_largest_fraction,
            component_guard_min_total_pixels_for_multicomponent=component_guard_min_total_pixels_for_multicomponent,
            score_component_guard_min_mass_fraction=score_component_guard_min_mass_fraction,
            score_component_guard_min_mean_fraction=score_component_guard_min_mean_fraction,
            score_component_guard_max_components=score_component_guard_max_components,
            score_component_guard_min_recovery_pixels=score_component_guard_min_recovery_pixels,
            sam3_refinement_geometry_gate=sam3_refinement_geometry_gate,
            sam3_refinement_gate_min_area_ratio=sam3_refinement_gate_min_area_ratio,
            sam3_refinement_gate_max_area_ratio=sam3_refinement_gate_max_area_ratio,
            sam3_refinement_gate_min_boundary_gain=sam3_refinement_gate_min_boundary_gain,
            sam3_box_refiner=sam3_box_refiner,
            sam3_adaptor_refiner=sam3_adaptor_refiner,
            sam3_mask_head_refiner=sam3_mask_head_refiner,
            sam3_prompt_mask_head_refiner=sam3_prompt_mask_head_refiner,
            sam3_prompt_mask_head_initial_refinement=sam3_prompt_mask_head_initial_refinement,
            sam3_prompt_mask_head_oracle_prompt=sam3_prompt_mask_head_oracle_prompt,
            sam3_prompt_mask_head_min_heatmap_mean_ratio=sam3_prompt_mask_head_min_heatmap_mean_ratio,
            sam3_prompt_mask_head_min_heatmap_mass_ratio=sam3_prompt_mask_head_min_heatmap_mass_ratio,
            sam3_prompt_mask_head_require_peak_in_refined=sam3_prompt_mask_head_require_peak_in_refined,
            min_select=min_select,
            output_dir=output_dir,
            save_masks=save_masks,
            save_geometry_maps=save_geometry_maps,
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
        "proposal_smoothing": proposal_smoothing_stats,
        "sam3_proposal_registration": sam3_proposal_registration_stats,
        "score_cache": score_cache_info,
        "direct_head_eval": direct_head_eval_status,
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
    elif args.selection_mode in {"score_margin", "score_ratio", "entropy_score"}:
        values = parse_float_list(args.confidence_sweep) or [float(args.confidence_threshold)]
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
            f"({float(registration.get('registered_fraction', 0.0)):.3f}); "
            f"assignment={registration.get('assignment_mode', 'center')}, "
            f"weight={registration.get('weight_mode', 'uniform')}."
        )
    if protocol.get("score_aggregation", "none") != "none":
        rows.append(
            "- Score aggregation: "
            f"{protocol.get('score_aggregation')} "
            f"(res={protocol.get('score_aggregation_resolution')}, "
            f"blend={protocol.get('score_aggregation_blend')})."
        )
    if protocol.get("proposal_smoothing", "none") != "none":
        rows.append(
            "- Proposal memory: "
            f"{protocol.get('proposal_smoothing')} "
            f"(voxel={protocol.get('proposal_voxel_size')}, "
            f"alpha={protocol.get('proposal_smoothing_alpha')}, "
            f"min_count={protocol.get('proposal_min_count')}, "
            f"gate={protocol.get('proposal_smoothing_gate')}, "
            f"margin={protocol.get('proposal_margin_threshold')})."
        )
    if protocol.get("sam3_proposal_registration_dir") and float(
        protocol.get("sam3_proposal_registration_alpha", 0.0) or 0.0
    ) > 0:
        sam3_stats = report.get("scene", {}).get("sam3_proposal_registration", {})
        rows.append(
            "- SAM3 training-view proposal registration: "
            f"alpha={protocol.get('sam3_proposal_registration_alpha')}, "
            f"prob>={protocol.get('sam3_proposal_registration_min_probability')}, "
            f"gate={protocol.get('sam3_proposal_registration_gate')}, "
            f"query_conditioned={protocol.get('sam3_proposal_registration_query_conditioned')}, "
            f"frames={sam3_stats.get('num_used_frames', 0)}/"
            f"{sam3_stats.get('num_cache_frames', 0)}, "
            f"memberships={sam3_stats.get('num_memberships', 0)}."
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
    parser.add_argument("--registered_feature_cache", default="", help="Optional VPR-registered primitive summary-feature cache path written when registered-view scores are computed")
    parser.add_argument("--prompt_templates", default=DEFAULT_PROMPT_TEMPLATES, help="Prompt templates separated by '|'; use {query}")
    parser.add_argument(
        "--selection_mode",
        choices=["top_ratio", "score_threshold", "mean_std", "score_margin", "score_ratio", "entropy_score"],
        default="top_ratio",
    )
    parser.add_argument("--top_ratio", type=float, default=0.02, help="Main fixed Gaussian top-ratio for top_ratio mode")
    parser.add_argument("--ratio_sweep", default="", help="Comma/space separated top-ratio sweep values")
    parser.add_argument("--score_threshold", type=float, default=0.25, help="Main score threshold for score_threshold mode")
    parser.add_argument("--threshold_sweep", default="", help="Comma/space separated score thresholds")
    parser.add_argument("--mean_std", type=float, default=1.0, help="Main mean+std multiplier for mean_std mode")
    parser.add_argument("--mean_std_sweep", default="", help="Comma/space separated mean_std multipliers")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Main confidence threshold for score_margin/score_ratio/entropy_score modes")
    parser.add_argument("--confidence_sweep", default="", help="Comma/space separated confidence thresholds")
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
    parser.add_argument("--proposal_smoothing", choices=["none", "voxel"], default="none", help="GT-free proposal-memory smoothing applied to primitive text scores before selection")
    parser.add_argument("--proposal_voxel_size", type=float, default=0.08, help="Scene-space voxel size for --proposal_smoothing voxel")
    parser.add_argument("--proposal_smoothing_alpha", type=float, default=0.0, help="Residual blend for proposal-memory score smoothing; 0 disables")
    parser.add_argument("--proposal_min_count", type=int, default=2, help="Minimum primitives in a proposal before score smoothing is applied")
    parser.add_argument(
        "--proposal_smoothing_gate",
        choices=[
            "all",
            "low_margin",
            "low_confidence",
            "low_margin_or_low_confidence",
            "proposal_consensus",
            "low_margin_and_proposal_consensus",
            "low_confidence_and_proposal_consensus",
        ],
        default="all",
        help=(
            "Apply proposal-memory smoothing to all, low-margin, low-confidence, "
            "or high-consensus proposal primitives"
        ),
    )
    parser.add_argument("--proposal_margin_threshold", type=float, default=0.0, help="Top1-top2 score margin threshold for --proposal_smoothing_gate low_margin")
    parser.add_argument("--proposal_confidence_threshold", type=float, default=0.0, help="Softmax top1 confidence threshold for low-confidence proposal smoothing gates")
    parser.add_argument("--proposal_consensus_threshold", type=float, default=0.0, help="Minimum proposal top-1 agreement for proposal-consensus smoothing gates")
    parser.add_argument("--sam3_proposal_registration_dir", default="", help="Optional official SAM3 training-view cache root used as label-free object proposal memory")
    parser.add_argument("--sam3_proposal_registration_alpha", type=float, default=0.0, help="Residual blend for SAM3 training-view proposal score fusion; 0 disables")
    parser.add_argument("--sam3_proposal_registration_min_probability", type=float, default=0.55, help="Minimum sampled SAM3 proposal probability required for Gaussian membership")
    parser.add_argument("--sam3_proposal_registration_max_masks_per_frame", type=int, default=0, help="Keep only top-N SAM3 masks per training view; 0 keeps all masks")
    parser.add_argument("--sam3_proposal_registration_gate", choices=["all", "low_margin"], default="low_margin", help="Apply SAM3 proposal fusion to all assigned primitives or only low-margin primitives")
    parser.add_argument("--sam3_proposal_registration_margin_threshold", type=float, default=0.05, help="Top1-top2 primitive score margin threshold for low-margin SAM3 proposal fusion")
    parser.add_argument("--sam3_proposal_registration_query_conditioned", action="store_true", help="Use only SAM3 training-view masks generated by the same text query for each query score")
    parser.add_argument("--selection_refinement", choices=["none", "top_score_components", "largest_components", "seed_expand_components", "proposal_components"], default="none", help="GT-free connected-component filtering after score-based primitive selection")
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
    parser.add_argument("--registration_assignment_mode", choices=["center", "raster_contrib", "raster_dominant", "raster_gaussian_top1", "raster_adjoint"], default="center", help="Primitive assignment for registered-view scoring; raster_contrib uses gsplat Gaussian-pixel intersections, raster_dominant keeps the strongest hit per pixel, raster_gaussian_top1 keeps the strongest hit per Gaussian, raster_adjoint uses the rasterizer color adjoint as true compositing contribution")
    parser.add_argument("--registration_weight_mode", choices=["uniform", "alpha", "alpha_depth"], default="uniform", help="Contribution-style weighting for VPR registered samples")
    parser.add_argument("--registration_confidence_blend", type=float, default=0.0, help="Blend weight for GT-free registration-count confidence calibration")
    parser.add_argument("--registration_confidence_mode", choices=["log", "linear"], default="log", help="How registration counts are mapped to confidence")
    parser.add_argument("--disable_registered_refiner", action="store_true", help="Disable VFA/screen refiner only for registered-view primitive scoring")
    parser.add_argument(
        "--direct_primitive_confidence_mode",
        choices=["none", "opacity", "opacity_log", "teacher_cache_valid"],
        default="none",
        help=(
            "GT-free confidence used to downweight weak direct primitive rows before "
            "selection. teacher_cache_valid uses only the row support mask from the "
            "training VPR/raster teacher cache, not RADIO reference features."
        ),
    )
    parser.add_argument("--direct_primitive_confidence_blend", type=float, default=0.0, help="Blend weight for direct primitive confidence calibration")
    parser.add_argument("--direct_primitive_opacity_threshold", type=float, default=0.02, help="Opacity threshold used by direct primitive confidence and opacity adapter gating")
    parser.add_argument("--use_point_summary_adapter", action="store_true", help="Use checkpoint point_summary_adapter_state_dict for direct Gaussian text scoring")
    parser.add_argument("--point_summary_adapter_blend_alpha", type=float, default=1.0, help="Blend decoded base summary with point adapter summary; 1 uses adapter only")
    parser.add_argument("--strict_direct_head_consistency", action="store_true", help="Fail if direct eval settings do not use the checkpoint's trained primitive head")
    parser.add_argument(
        "--point_summary_adapter_valid_mask_mode",
        choices=["teacher_cache", "opacity", "none"],
        default="teacher_cache",
        help=(
            "Which GT-free support mask gates point-adapter blending. "
            "Use 'opacity' for deployed compact-map evaluation with no VPR-cache read."
        ),
    )
    parser.add_argument("--silhouette_threshold", type=float, default=0.7, help="OpenGaussian-style rendered silhouette threshold")
    parser.add_argument(
        "--mask_refinement",
        choices=[
            "none",
            "rgb_grabcut",
            "largest_component",
            "largest_component_rgb_grabcut",
            "rgb_grabcut_largest_component",
            "rgb_grabcut_component_guard",
            "rgb_grabcut_score_component_guard",
            "sam3_box",
            "sam3_adaptor_boundary",
            "sam3_adaptor_grabcut",
            "sam3_mask_head",
            "sam3_prompt_mask_head",
        ],
        default="none",
        help="Optional GT-free projection cleanup after rendering selected primitives",
    )
    parser.add_argument("--mask_refinement_iters", type=int, default=1, help="GrabCut iterations for rgb_grabcut mask refinement")
    parser.add_argument("--mask_refinement_dilate", type=int, default=5, help="Pixel dilation radius defining the rgb_grabcut support band")
    parser.add_argument("--mask_refinement_erode", type=int, default=2, help="Pixel erosion radius defining sure foreground for rgb_grabcut")
    parser.add_argument("--component_guard_min_largest_fraction", type=float, default=0.65, help="For rgb_grabcut_component_guard, keep only the largest component when it covers at least this fraction of the refined support")
    parser.add_argument(
        "--component_guard_min_total_pixels_for_multicomponent",
        type=int,
        default=0,
        help="For rgb_grabcut_component_guard, preserve multi-component refined support only when the total support has at least this many pixels",
    )
    parser.add_argument("--score_component_guard_min_mass_fraction", type=float, default=0.20, help="For rgb_grabcut_score_component_guard, keep components whose compact score-heatmap mass is at least this fraction of the top component")
    parser.add_argument("--score_component_guard_min_mean_fraction", type=float, default=0.0, help="For rgb_grabcut_score_component_guard, optional mean score floor as a fraction of the top component mean")
    parser.add_argument("--score_component_guard_max_components", type=int, default=0, help="For rgb_grabcut_score_component_guard, maximum kept components per mask; 0 keeps all components passing score gates")
    parser.add_argument(
        "--score_component_guard_min_recovery_pixels",
        type=int,
        default=0,
        help=(
            "For rgb_grabcut_score_component_guard, recover a tiny/empty support "
            "from the rendered compact score heatmap before component ranking"
        ),
    )
    parser.add_argument("--sam3_refinement_geometry_gate", action="store_true", help="Accept feature-only SAM3 refinement only when alpha/depth boundary alignment improves without GT")
    parser.add_argument("--sam3_refinement_gate_min_area_ratio", type=float, default=0.5, help="Minimum refined/coarse area ratio for geometry-gated SAM3 feature refinement")
    parser.add_argument("--sam3_refinement_gate_max_area_ratio", type=float, default=1.5, help="Maximum refined/coarse area ratio for geometry-gated SAM3 feature refinement")
    parser.add_argument("--sam3_refinement_gate_min_boundary_gain", type=float, default=0.0, help="Minimum alpha/depth boundary-score gain required by geometry-gated SAM3 feature refinement")
    parser.add_argument("--sam3_checkpoint_path", default="checkpoints/sam3_modelscope/sam3.pt", help="Official SAM3 checkpoint for sam3_box refinement")
    parser.add_argument("--sam3_confidence_threshold", type=float, default=0.0, help="SAM3 confidence threshold for sam3_box refinement")
    parser.add_argument("--sam3_resolution", type=int, default=1008, help="SAM3 processor resolution")
    parser.add_argument("--sam3_amp_dtype", choices=["auto", "off", "bfloat16"], default="auto", help="SAM3 CUDA autocast dtype")
    parser.add_argument("--sam3_box_padding", type=int, default=8, help="Pixel padding around rendered mask box before SAM3 prompt")
    parser.add_argument("--sam3_min_initial_iou", type=float, default=0.05, help="Minimum initial-mask overlap required to accept a SAM3 box-refined mask")
    parser.add_argument("--sam3_adaptor_checkpoint", default=DEFAULT_RADIO_ADAPTOR_CHECKPOINT, help="RADIO checkpoint containing the frozen sam3 feature_projection adaptor")
    parser.add_argument("--sam3_adaptor_support_mode", choices=["mask_dilate", "box"], default="mask_dilate", help="Support region for feature-only SAM3-adaptor refinement")
    parser.add_argument("--sam3_adaptor_prototype_mode", choices=["mask_inner", "box"], default="mask_inner", help="Feature prototype source for feature-only SAM3-adaptor refinement")
    parser.add_argument("--sam3_adaptor_support_dilate", type=int, default=8, help="Pixel dilation radius around the rendered direct-3D mask used as feature-only SAM3-adaptor support")
    parser.add_argument("--sam3_adaptor_inner_erode", type=int, default=2, help="Pixel erosion radius used to form confident foreground tokens for feature-only SAM3-adaptor refinement")
    parser.add_argument("--sam3_adaptor_score_std_scale", type=float, default=0.0, help="Mean+std threshold scale inside the adaptor support band")
    parser.add_argument("--sam3_adaptor_min_area_scale", type=float, default=0.25, help="Lower bound on refined feature-mask area relative to the coarse mask at adaptor resolution")
    parser.add_argument("--sam3_adaptor_max_area_scale", type=float, default=1.25, help="Upper bound on refined feature-mask area relative to the coarse mask at adaptor resolution")
    parser.add_argument("--sam3_adaptor_max_initial_area_fraction", type=float, default=1.0, help="Skip feature-only SAM3-adaptor refinement when the coarse mask covers more than this image fraction")
    parser.add_argument("--sam3_adaptor_background_weight", type=float, default=0.20, help="Local background prototype subtraction weight for feature-only SAM3-adaptor refinement")
    parser.add_argument("--sam3_adaptor_min_initial_iou", type=float, default=0.03, help="Minimum coarse-mask overlap required to accept a feature-only SAM3-adaptor refined mask")
    parser.add_argument("--sam3_mask_head_checkpoint", default="", help="Checkpoint containing foundation_cache_projectors_state_dict/sam3 for feature-only SAM3 mask-logit readout; defaults to --checkpoint")
    parser.add_argument("--sam3_mask_head_logit_threshold", type=float, default=0.5, help="Threshold for feature-only SAM3 mask-head candidate masks; foundation cache targets are mask probabilities")
    parser.add_argument("--sam3_mask_head_min_initial_iou", type=float, default=0.05, help="Minimum coarse-mask overlap required to accept a SAM3 mask-head candidate")
    parser.add_argument("--sam3_mask_head_max_initial_area_fraction", type=float, default=1.0, help="Skip SAM3 mask-head refinement when the coarse mask covers more than this image fraction")
    parser.add_argument("--sam3_prompt_mask_head_checkpoint", default="", help="Checkpoint from train_prompt_conditioned_sam3_mask_head for prompt-conditioned feature-only mask refinement")
    parser.add_argument("--sam3_prompt_mask_head_text_embedding_cache", default="", help="Text embedding cache for prompt-conditioned SAM3 mask head; defaults to --text_embedding_cache")
    parser.add_argument("--sam3_prompt_mask_head_logit_threshold", type=float, default=0.0, help="Logit threshold for prompt-conditioned SAM3 mask-head output")
    parser.add_argument("--sam3_prompt_mask_head_min_initial_iou", type=float, default=0.05, help="Minimum coarse-mask overlap required to accept prompt-conditioned SAM3 mask-head output")
    parser.add_argument("--sam3_prompt_mask_head_max_initial_area_fraction", type=float, default=1.0, help="Skip prompt-conditioned SAM3 mask-head refinement when the coarse mask covers more than this image fraction")
    parser.add_argument("--sam3_prompt_mask_head_min_refined_area_ratio", type=float, default=0.0, help="Reject prompt-conditioned SAM3 mask-head output when refined/coarse area is below this ratio")
    parser.add_argument("--sam3_prompt_mask_head_max_refined_area_ratio", type=float, default=0.0, help="Reject prompt-conditioned SAM3 mask-head output when refined/coarse area is above this ratio; <=0 disables")
    parser.add_argument("--sam3_prompt_mask_head_support_dilate", type=int, default=-1, help="Clip prompt-conditioned SAM3 mask-head candidates to the coarse mask dilated by this pixel radius; <0 disables")
    parser.add_argument("--sam3_prompt_mask_head_coarse_dilate", type=int, default=0, help="Dilation radius applied to the rendered coarse mask before prompt-conditioned mask-head inference")
    parser.add_argument("--sam3_prompt_mask_head_coarse_threshold", type=float, default=0.5, help="Threshold used to binarize the coarse mask prompt")
    parser.add_argument("--sam3_prompt_mask_head_min_quality", type=float, default=0.0, help="Fallback to the coarse direct-3D mask when the learned prompt-mask quality score is below this threshold; <=0 disables")
    parser.add_argument("--sam3_prompt_mask_head_initial_refinement", choices=["none", "peak_component"], default="none", help="GT-free cleanup for the direct-3D coarse prompt before feature-only SAM3 mask-head inference")
    parser.add_argument("--sam3_prompt_mask_head_oracle_prompt", choices=["none", "gt_mask", "gt_box"], default="none", help="Diagnostic-only oracle prompt for feature-only SAM3 mask-head ceiling; not a valid main-protocol setting")
    parser.add_argument("--allow_sam3_prompt_mask_head_oracle_diagnostic", action="store_true", help="Explicitly allow GT oracle prompts for diagnostic-only SAM3 prompt-mask-head ceiling runs")
    parser.add_argument("--sam3_prompt_mask_head_min_heatmap_mean_ratio", type=float, default=0.0, help="Reject prompt-mask-head refinements whose mean score-heatmap support falls below this ratio")
    parser.add_argument("--sam3_prompt_mask_head_min_heatmap_mass_ratio", type=float, default=0.0, help="Reject prompt-mask-head refinements whose score-heatmap mass falls below this ratio")
    parser.add_argument("--sam3_prompt_mask_head_require_peak_in_refined", action="store_true", help="Require feature-only SAM3 prompt-mask-head output to retain the direct score-heatmap peak")
    parser.add_argument("--allow_missing_sam3_prompt_text_embeddings", action="store_true", help="Permit prompt-conditioned SAM3 mask-head fallback when a query category has no text embedding")
    parser.add_argument("--min_select", type=int, default=1, help="Minimum selected Gaussians per query")
    parser.add_argument("--chunk_size", type=int, default=8192, help="Gaussian decode/projection chunk size")
    parser.add_argument("--all_labeled_frames", action="store_true", help="Use all local labels instead of OpenGaussian official frames")
    parser.add_argument("--save_masks", action="store_true", help="Save rendered binary prediction masks")
    parser.add_argument(
        "--save_geometry_maps",
        action="store_true",
        help="Save alpha/depth discontinuity maps and per-query boundary-alignment overlays",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    args = parser.parse_args()
    if (
        args.sam3_prompt_mask_head_oracle_prompt != "none"
        and not args.allow_sam3_prompt_mask_head_oracle_diagnostic
    ):
        parser.error(
            "--sam3_prompt_mask_head_oracle_prompt is diagnostic-only; pass "
            "--allow_sam3_prompt_mask_head_oracle_diagnostic to run a GT oracle prompt ceiling check"
        )

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
    if args.score_source == "direct" and args.direct_primitive_confidence_mode != "none":
        print(
            "Direct conf: "
            f"{args.direct_primitive_confidence_mode}/"
            f"{args.direct_primitive_confidence_blend:g} "
            f"(opacity>{args.direct_primitive_opacity_threshold:g})"
        )
    if args.score_source == "registered_view":
        print(f"VPR assign: {args.registration_assignment_mode}/{args.registration_weight_mode}")
    if args.proposal_smoothing != "none":
        print(
            "Proposal:   "
            f"{args.proposal_smoothing} "
            f"(voxel={args.proposal_voxel_size:g}, "
            f"alpha={args.proposal_smoothing_alpha:g}, "
            f"min_count={args.proposal_min_count}, "
            f"gate={args.proposal_smoothing_gate}, "
            f"margin={args.proposal_margin_threshold:g}, "
            f"confidence={args.proposal_confidence_threshold:g}, "
            f"consensus={args.proposal_consensus_threshold:g})"
        )
    if args.sam3_proposal_registration_dir and args.sam3_proposal_registration_alpha > 0:
        print(
            "SAM3 prop.: "
            f"train-view memory alpha={args.sam3_proposal_registration_alpha:g}, "
            f"prob>={args.sam3_proposal_registration_min_probability:g}, "
            f"gate={args.sam3_proposal_registration_gate}, "
            f"margin={args.sam3_proposal_registration_margin_threshold:g}, "
            f"query_conditioned={args.sam3_proposal_registration_query_conditioned}"
        )
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
        registered_feature_cache_path=args.registered_feature_cache or None,
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
        proposal_smoothing=args.proposal_smoothing,
        proposal_voxel_size=args.proposal_voxel_size,
        proposal_smoothing_alpha=args.proposal_smoothing_alpha,
        proposal_min_count=args.proposal_min_count,
        proposal_smoothing_gate=args.proposal_smoothing_gate,
        proposal_margin_threshold=args.proposal_margin_threshold,
        proposal_confidence_threshold=args.proposal_confidence_threshold,
        proposal_consensus_threshold=args.proposal_consensus_threshold,
        sam3_proposal_registration_dir=args.sam3_proposal_registration_dir,
        sam3_proposal_registration_alpha=args.sam3_proposal_registration_alpha,
        sam3_proposal_registration_min_probability=args.sam3_proposal_registration_min_probability,
        sam3_proposal_registration_max_masks_per_frame=args.sam3_proposal_registration_max_masks_per_frame,
        sam3_proposal_registration_gate=args.sam3_proposal_registration_gate,
        sam3_proposal_registration_margin_threshold=args.sam3_proposal_registration_margin_threshold,
        sam3_proposal_registration_query_conditioned=args.sam3_proposal_registration_query_conditioned,
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
        registration_assignment_mode=args.registration_assignment_mode,
        registration_weight_mode=args.registration_weight_mode,
        registration_confidence_blend=args.registration_confidence_blend,
        registration_confidence_mode=args.registration_confidence_mode,
        disable_registered_refiner=args.disable_registered_refiner,
        use_point_summary_adapter=args.use_point_summary_adapter,
        point_summary_adapter_blend_alpha=args.point_summary_adapter_blend_alpha,
        point_summary_adapter_valid_mask_mode=args.point_summary_adapter_valid_mask_mode,
        strict_direct_head_consistency=args.strict_direct_head_consistency,
        direct_primitive_confidence_mode=args.direct_primitive_confidence_mode,
        direct_primitive_confidence_blend=args.direct_primitive_confidence_blend,
        direct_primitive_opacity_threshold=args.direct_primitive_opacity_threshold,
        silhouette_threshold=args.silhouette_threshold,
        mask_refinement=args.mask_refinement,
        mask_refinement_iters=args.mask_refinement_iters,
        mask_refinement_dilate=args.mask_refinement_dilate,
        mask_refinement_erode=args.mask_refinement_erode,
        component_guard_min_largest_fraction=args.component_guard_min_largest_fraction,
        component_guard_min_total_pixels_for_multicomponent=args.component_guard_min_total_pixels_for_multicomponent,
        score_component_guard_min_mass_fraction=args.score_component_guard_min_mass_fraction,
        score_component_guard_min_mean_fraction=args.score_component_guard_min_mean_fraction,
        score_component_guard_max_components=args.score_component_guard_max_components,
        score_component_guard_min_recovery_pixels=args.score_component_guard_min_recovery_pixels,
        sam3_refinement_geometry_gate=args.sam3_refinement_geometry_gate,
        sam3_refinement_gate_min_area_ratio=args.sam3_refinement_gate_min_area_ratio,
        sam3_refinement_gate_max_area_ratio=args.sam3_refinement_gate_max_area_ratio,
        sam3_refinement_gate_min_boundary_gain=args.sam3_refinement_gate_min_boundary_gain,
        sam3_checkpoint_path=args.sam3_checkpoint_path,
        sam3_confidence_threshold=args.sam3_confidence_threshold,
        sam3_resolution=args.sam3_resolution,
        sam3_amp_dtype=args.sam3_amp_dtype,
        sam3_box_padding=args.sam3_box_padding,
        sam3_min_initial_iou=args.sam3_min_initial_iou,
        sam3_adaptor_checkpoint=args.sam3_adaptor_checkpoint,
        sam3_adaptor_support_mode=args.sam3_adaptor_support_mode,
        sam3_adaptor_prototype_mode=args.sam3_adaptor_prototype_mode,
        sam3_adaptor_support_dilate=args.sam3_adaptor_support_dilate,
        sam3_adaptor_inner_erode=args.sam3_adaptor_inner_erode,
        sam3_adaptor_score_std_scale=args.sam3_adaptor_score_std_scale,
        sam3_adaptor_min_area_scale=args.sam3_adaptor_min_area_scale,
        sam3_adaptor_max_area_scale=args.sam3_adaptor_max_area_scale,
        sam3_adaptor_max_initial_area_fraction=args.sam3_adaptor_max_initial_area_fraction,
        sam3_adaptor_background_weight=args.sam3_adaptor_background_weight,
        sam3_adaptor_min_initial_iou=args.sam3_adaptor_min_initial_iou,
        sam3_mask_head_checkpoint=args.sam3_mask_head_checkpoint,
        sam3_mask_head_logit_threshold=args.sam3_mask_head_logit_threshold,
        sam3_mask_head_min_initial_iou=args.sam3_mask_head_min_initial_iou,
        sam3_mask_head_max_initial_area_fraction=args.sam3_mask_head_max_initial_area_fraction,
        sam3_prompt_mask_head_checkpoint=args.sam3_prompt_mask_head_checkpoint,
        sam3_prompt_mask_head_text_embedding_cache=args.sam3_prompt_mask_head_text_embedding_cache,
        sam3_prompt_mask_head_logit_threshold=args.sam3_prompt_mask_head_logit_threshold,
        sam3_prompt_mask_head_min_initial_iou=args.sam3_prompt_mask_head_min_initial_iou,
        sam3_prompt_mask_head_max_initial_area_fraction=args.sam3_prompt_mask_head_max_initial_area_fraction,
        sam3_prompt_mask_head_min_refined_area_ratio=args.sam3_prompt_mask_head_min_refined_area_ratio,
        sam3_prompt_mask_head_max_refined_area_ratio=args.sam3_prompt_mask_head_max_refined_area_ratio,
        sam3_prompt_mask_head_support_dilate=args.sam3_prompt_mask_head_support_dilate,
        sam3_prompt_mask_head_coarse_dilate=args.sam3_prompt_mask_head_coarse_dilate,
        sam3_prompt_mask_head_coarse_threshold=args.sam3_prompt_mask_head_coarse_threshold,
        sam3_prompt_mask_head_min_quality=args.sam3_prompt_mask_head_min_quality,
        sam3_prompt_mask_head_initial_refinement=args.sam3_prompt_mask_head_initial_refinement,
        sam3_prompt_mask_head_oracle_prompt=args.sam3_prompt_mask_head_oracle_prompt,
        sam3_prompt_mask_head_min_heatmap_mean_ratio=args.sam3_prompt_mask_head_min_heatmap_mean_ratio,
        sam3_prompt_mask_head_min_heatmap_mass_ratio=args.sam3_prompt_mask_head_min_heatmap_mass_ratio,
        sam3_prompt_mask_head_require_peak_in_refined=args.sam3_prompt_mask_head_require_peak_in_refined,
        allow_missing_sam3_prompt_text_embeddings=args.allow_missing_sam3_prompt_text_embeddings,
        min_select=args.min_select,
        chunk_size=args.chunk_size,
        official_frames_only=not args.all_labeled_frames,
        save_masks=args.save_masks,
        save_geometry_maps=args.save_geometry_maps,
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
                else (
                    "pre-refiner Gaussian-center compact features projected by VPR-trained point summary adapter"
                    if args.use_point_summary_adapter
                    else "pre-refiner Gaussian-center decoded RADIO-compatible features"
                )
            ),
            "score_source": args.score_source,
            "score_cache": args.score_cache,
            "registered_feature_cache": args.registered_feature_cache,
            "compact_feature_key": args.compact_feature_key,
            "direct_readout_mode": args.direct_readout_mode,
            "direct_readout_k": args.direct_readout_k,
            "direct_readout_candidate_k": args.direct_readout_candidate_k,
            "text_head": "SigLIP2 summary/text-aligned head",
            "canonical_embedding_cache": args.canonical_embedding_cache if args.scoring == "relevancy" else "",
            "score_aggregation": args.score_aggregation,
            "score_aggregation_resolution": args.score_aggregation_resolution,
            "score_aggregation_blend": args.score_aggregation_blend,
            "proposal_smoothing": args.proposal_smoothing,
            "proposal_voxel_size": float(args.proposal_voxel_size),
            "proposal_smoothing_alpha": float(args.proposal_smoothing_alpha),
            "proposal_min_count": int(args.proposal_min_count),
            "proposal_smoothing_gate": args.proposal_smoothing_gate,
            "proposal_margin_threshold": float(args.proposal_margin_threshold),
            "proposal_confidence_threshold": float(args.proposal_confidence_threshold),
            "proposal_consensus_threshold": float(args.proposal_consensus_threshold),
            "sam3_proposal_registration_dir": args.sam3_proposal_registration_dir,
            "sam3_proposal_registration_alpha": float(args.sam3_proposal_registration_alpha),
            "sam3_proposal_registration_min_probability": float(args.sam3_proposal_registration_min_probability),
            "sam3_proposal_registration_max_masks_per_frame": int(args.sam3_proposal_registration_max_masks_per_frame),
            "sam3_proposal_registration_gate": args.sam3_proposal_registration_gate,
            "sam3_proposal_registration_margin_threshold": float(args.sam3_proposal_registration_margin_threshold),
            "sam3_proposal_registration_query_conditioned": bool(args.sam3_proposal_registration_query_conditioned),
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
            "registration_assignment_mode": args.registration_assignment_mode,
            "registration_weight_mode": args.registration_weight_mode,
            "registration_confidence_blend": args.registration_confidence_blend,
            "registration_confidence_mode": args.registration_confidence_mode,
            "disable_registered_refiner": bool(args.disable_registered_refiner),
            "direct_primitive_confidence_mode": args.direct_primitive_confidence_mode,
            "direct_primitive_confidence_blend": float(args.direct_primitive_confidence_blend),
            "direct_primitive_opacity_threshold": float(args.direct_primitive_opacity_threshold),
            "use_point_summary_adapter": bool(args.use_point_summary_adapter),
            "strict_direct_head_consistency": bool(args.strict_direct_head_consistency),
            "point_summary_adapter_blend_alpha": float(args.point_summary_adapter_blend_alpha),
            "point_summary_adapter_valid_mask": (
                args.point_summary_adapter_valid_mask_mode if args.use_point_summary_adapter else ""
            ),
            "render_role": "render selected primitives only for mask evaluation",
            "metrics": ["mIoU", "Acc@0.25", "Acc@0.50", "boundary_f", "trimap_iou"],
            "geometry_alignment_maps": bool(args.save_geometry_maps),
            "silhouette_threshold": args.silhouette_threshold,
            "mask_refinement": args.mask_refinement,
            "mask_refinement_iters": args.mask_refinement_iters,
            "mask_refinement_dilate": args.mask_refinement_dilate,
            "mask_refinement_erode": args.mask_refinement_erode,
            "component_guard_min_largest_fraction": args.component_guard_min_largest_fraction,
            "component_guard_min_total_pixels_for_multicomponent": int(
                args.component_guard_min_total_pixels_for_multicomponent
            ),
            "score_component_guard_min_mass_fraction": float(
                args.score_component_guard_min_mass_fraction
            ),
            "score_component_guard_min_mean_fraction": float(
                args.score_component_guard_min_mean_fraction
            ),
            "score_component_guard_max_components": int(args.score_component_guard_max_components),
            "score_component_guard_min_recovery_pixels": int(
                args.score_component_guard_min_recovery_pixels
            ),
            "sam3_refinement_geometry_gate": bool(args.sam3_refinement_geometry_gate),
            "sam3_refinement_gate_min_area_ratio": float(args.sam3_refinement_gate_min_area_ratio),
            "sam3_refinement_gate_max_area_ratio": float(args.sam3_refinement_gate_max_area_ratio),
            "sam3_refinement_gate_min_boundary_gain": float(args.sam3_refinement_gate_min_boundary_gain),
            "sam3_checkpoint_path": args.sam3_checkpoint_path if args.mask_refinement == "sam3_box" else "",
            "sam3_checkpoint_sha256": (
                sha256_file_if_exists(args.sam3_checkpoint_path)
                if args.mask_refinement == "sam3_box"
                else ""
            ),
            "sam3_adaptor_checkpoint": (
                args.sam3_adaptor_checkpoint
                if args.mask_refinement in {"sam3_adaptor_boundary", "sam3_adaptor_grabcut"}
                else ""
            ),
            "sam3_adaptor_checkpoint_sha256": (
                sha256_file_if_exists(args.sam3_adaptor_checkpoint)
                if args.mask_refinement in {"sam3_adaptor_boundary", "sam3_adaptor_grabcut"}
                else ""
            ),
            "sam3_mask_head_checkpoint": (
                (args.sam3_mask_head_checkpoint or args.checkpoint)
                if args.mask_refinement == "sam3_mask_head"
                else ""
            ),
            "sam3_mask_head_checkpoint_sha256": (
                sha256_file_if_exists(args.sam3_mask_head_checkpoint or args.checkpoint)
                if args.mask_refinement == "sam3_mask_head"
                else ""
            ),
            "sam3_prompt_mask_head_checkpoint": (
                args.sam3_prompt_mask_head_checkpoint
                if args.mask_refinement == "sam3_prompt_mask_head"
                else ""
            ),
            "sam3_prompt_mask_head_checkpoint_sha256": (
                sha256_file_if_exists(args.sam3_prompt_mask_head_checkpoint)
                if args.mask_refinement == "sam3_prompt_mask_head"
                else ""
            ),
            "diagnostic_oracle_prompt": bool(
                args.mask_refinement == "sam3_prompt_mask_head"
                and args.sam3_prompt_mask_head_oracle_prompt != "none"
                and args.allow_sam3_prompt_mask_head_oracle_diagnostic
            ),
            "config_sha256": sha256_file_if_exists(args.config),
            "checkpoint_sha256": sha256_file_if_exists(args.checkpoint),
            "score_cache_sha256": sha256_file_if_exists(args.score_cache) if args.score_cache else "",
            "summary_head_weights_sha256": sha256_file_if_exists(args.summary_head_weights),
            "text_embedding_cache_sha256": sha256_file_if_exists(args.text_embedding_cache)
            if args.text_embedding_cache
            else "",
            "repo_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
            ).strip(),
            "sam3_backend": (
                "facebookresearch/sam3"
                if args.mask_refinement == "sam3_box"
                else (
                    "frozen_RADIO_sam3_feature_projection_no_RGB_decoder"
                    if args.mask_refinement == "sam3_adaptor_boundary"
                    else "frozen_RADIO_sam3_feature_projection_feature_grabcut_no_RGB_decoder"
                    if args.mask_refinement == "sam3_adaptor_grabcut"
                    else "trained_CTF_SAM3_mask_logit_projector_no_RGB_decoder"
                    if args.mask_refinement == "sam3_mask_head"
                    else "prompt_conditioned_CTF_SAM3_pseudo_mask_head_no_RGB_decoder"
                    if args.mask_refinement == "sam3_prompt_mask_head"
                    else ""
                )
            ),
            "sam3_eval_mode": bool(args.mask_refinement == "sam3_box"),
            "sam3_confidence_threshold": args.sam3_confidence_threshold if args.mask_refinement == "sam3_box" else 0.0,
            "sam3_resolution": args.sam3_resolution if args.mask_refinement == "sam3_box" else 0,
            "sam3_amp_dtype": args.sam3_amp_dtype if args.mask_refinement == "sam3_box" else "",
            "sam3_box_padding": args.sam3_box_padding if args.mask_refinement == "sam3_box" else 0,
            "sam3_min_initial_iou": args.sam3_min_initial_iou if args.mask_refinement == "sam3_box" else 0.0,
            "sam3_adaptor_support_dilate": (
                args.sam3_adaptor_support_dilate
                if args.mask_refinement in {"sam3_adaptor_boundary", "sam3_adaptor_grabcut"}
                else 0
            ),
            "sam3_adaptor_support_mode": (
                args.sam3_adaptor_support_mode
                if args.mask_refinement in {"sam3_adaptor_boundary", "sam3_adaptor_grabcut"}
                else ""
            ),
            "sam3_adaptor_prototype_mode": (
                args.sam3_adaptor_prototype_mode
                if args.mask_refinement in {"sam3_adaptor_boundary", "sam3_adaptor_grabcut"}
                else ""
            ),
            "sam3_adaptor_inner_erode": (
                args.sam3_adaptor_inner_erode
                if args.mask_refinement in {"sam3_adaptor_boundary", "sam3_adaptor_grabcut"}
                else 0
            ),
            "sam3_adaptor_score_std_scale": (
                args.sam3_adaptor_score_std_scale
                if args.mask_refinement in {"sam3_adaptor_boundary", "sam3_adaptor_grabcut"}
                else 0.0
            ),
            "sam3_adaptor_min_area_scale": (
                args.sam3_adaptor_min_area_scale
                if args.mask_refinement in {"sam3_adaptor_boundary", "sam3_adaptor_grabcut"}
                else 0.0
            ),
            "sam3_adaptor_max_area_scale": (
                args.sam3_adaptor_max_area_scale
                if args.mask_refinement in {"sam3_adaptor_boundary", "sam3_adaptor_grabcut"}
                else 0.0
            ),
            "sam3_adaptor_max_initial_area_fraction": (
                args.sam3_adaptor_max_initial_area_fraction
                if args.mask_refinement in {"sam3_adaptor_boundary", "sam3_adaptor_grabcut"}
                else 0.0
            ),
            "sam3_adaptor_background_weight": (
                args.sam3_adaptor_background_weight
                if args.mask_refinement in {"sam3_adaptor_boundary", "sam3_adaptor_grabcut"}
                else 0.0
            ),
            "sam3_adaptor_min_initial_iou": (
                args.sam3_adaptor_min_initial_iou
                if args.mask_refinement in {"sam3_adaptor_boundary", "sam3_adaptor_grabcut"}
                else 0.0
            ),
            "sam3_mask_head_logit_threshold": (
                args.sam3_mask_head_logit_threshold
                if args.mask_refinement == "sam3_mask_head"
                else 0.0
            ),
            "sam3_mask_head_min_initial_iou": (
                args.sam3_mask_head_min_initial_iou
                if args.mask_refinement == "sam3_mask_head"
                else 0.0
            ),
            "sam3_mask_head_max_initial_area_fraction": (
                args.sam3_mask_head_max_initial_area_fraction
                if args.mask_refinement == "sam3_mask_head"
                else 0.0
            ),
            "sam3_prompt_mask_head_logit_threshold": (
                args.sam3_prompt_mask_head_logit_threshold
                if args.mask_refinement == "sam3_prompt_mask_head"
                else 0.0
            ),
            "sam3_prompt_mask_head_min_initial_iou": (
                args.sam3_prompt_mask_head_min_initial_iou
                if args.mask_refinement == "sam3_prompt_mask_head"
                else 0.0
            ),
            "sam3_prompt_mask_head_max_initial_area_fraction": (
                args.sam3_prompt_mask_head_max_initial_area_fraction
                if args.mask_refinement == "sam3_prompt_mask_head"
                else 0.0
            ),
            "sam3_prompt_mask_head_min_refined_area_ratio": (
                args.sam3_prompt_mask_head_min_refined_area_ratio
                if args.mask_refinement == "sam3_prompt_mask_head"
                else 0.0
            ),
            "sam3_prompt_mask_head_max_refined_area_ratio": (
                args.sam3_prompt_mask_head_max_refined_area_ratio
                if args.mask_refinement == "sam3_prompt_mask_head"
                else 0.0
            ),
            "sam3_prompt_mask_head_support_dilate": (
                args.sam3_prompt_mask_head_support_dilate
                if args.mask_refinement == "sam3_prompt_mask_head"
                else -1
            ),
            "sam3_prompt_mask_head_coarse_dilate": (
                args.sam3_prompt_mask_head_coarse_dilate
                if args.mask_refinement == "sam3_prompt_mask_head"
                else 0
            ),
            "sam3_prompt_mask_head_min_quality": (
                args.sam3_prompt_mask_head_min_quality
                if args.mask_refinement == "sam3_prompt_mask_head"
                else 0.0
            ),
            "sam3_prompt_mask_head_initial_refinement": (
                args.sam3_prompt_mask_head_initial_refinement
                if args.mask_refinement == "sam3_prompt_mask_head"
                else "none"
            ),
            "sam3_prompt_mask_head_oracle_prompt": (
                args.sam3_prompt_mask_head_oracle_prompt
                if args.mask_refinement == "sam3_prompt_mask_head"
                else "none"
            ),
            "sam3_prompt_mask_head_min_heatmap_mean_ratio": (
                args.sam3_prompt_mask_head_min_heatmap_mean_ratio
                if args.mask_refinement == "sam3_prompt_mask_head"
                else 0.0
            ),
            "sam3_prompt_mask_head_min_heatmap_mass_ratio": (
                args.sam3_prompt_mask_head_min_heatmap_mass_ratio
                if args.mask_refinement == "sam3_prompt_mask_head"
                else 0.0
            ),
            "sam3_prompt_mask_head_require_peak_in_refined": (
                bool(args.sam3_prompt_mask_head_require_peak_in_refined)
                if args.mask_refinement == "sam3_prompt_mask_head"
                else False
            ),
            "sam3_prompt_format": "normalized_cxcywh" if args.mask_refinement == "sam3_box" else "",
            "sam3_candidate_selection": (
                "max_initial_mask_iou_score_tiebreak_no_gt"
                if args.mask_refinement == "sam3_box"
                else "coarse_mask_seeded_feature_similarity_no_gt"
                if args.mask_refinement == "sam3_adaptor_boundary"
                else "feature_space_grabcut_from_coarse_mask_no_gt"
                if args.mask_refinement == "sam3_adaptor_grabcut"
                else "max_initial_mask_iou_over_trained_mask_logit_bank_no_gt"
                if args.mask_refinement == "sam3_mask_head"
                else "single_prompt_conditioned_mask_from_rendered_features_and_coarse_mask_no_gt"
                if args.mask_refinement == "sam3_prompt_mask_head"
                else ""
            ),
            "sam3_fallback_policy": (
                "return_initial_mask_on_empty_prompt_missing_output_shape_mismatch_or_low_initial_overlap"
                if args.mask_refinement == "sam3_box"
                else "return_initial_mask_on_empty_feature_or_low_initial_overlap"
                if args.mask_refinement == "sam3_adaptor_boundary"
                else "return_initial_mask_on_empty_feature_or_grabcut_failure"
                if args.mask_refinement == "sam3_adaptor_grabcut"
                else "return_initial_mask_on_empty_logits_large_initial_mask_or_low_initial_overlap"
                if args.mask_refinement == "sam3_mask_head"
                else "return_initial_mask_on_missing_feature_prompt_large_initial_mask_low_initial_overlap_or_unbounded_area_change"
                if args.mask_refinement == "sam3_prompt_mask_head"
                else ""
            ),
            "sam3_uses_official_rgb_readout": bool(args.mask_refinement == "sam3_box"),
        },
        "prompt_templates": prompt_templates,
        "elapsed_seconds": time.time() - t0,
        "scene": scene_report,
    }
    write_scene_report(out_root, args.scene, report)


if __name__ == "__main__":
    main()
