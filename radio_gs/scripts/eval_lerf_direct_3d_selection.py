#!/usr/bin/env python3
"""Evaluate LERF-OVS direct 3D object selection.

This script implements an OpenGaussian-style protocol for RADIO-GS:

1. Decode pre-refiner Gaussian/primitive features at 3D Gaussian centers.
2. Project decoded RADIO-compatible features into SigLIP2 text space.
3. Select 3D primitives from text-Gaussian similarity scores.
4. Render selected primitives as binary masks on the official LERF-OVS views.
5. Report mIoU, Acc@0.25, and Acc@0.50 against LERF-OVS masks.

The view-space refiner is intentionally not used. It is a rendered-view module,
while this evaluator queries Gaussian-level features directly in 3D.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, ".")

from radio_gs.data.lerf_dataset import LERFDataset
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

    if mode == "official":
        frames = sorted(available & annotated & official)
        if not frames:
            frames = sorted(available & annotated)
    elif mode == "annotated":
        frames = sorted(available & annotated)
    elif mode == "all_poses":
        frames = sorted(available)
    else:
        raise ValueError(f"Unsupported registration frame mode: {mode}")

    return _evenly_subsample_frames(frames, int(max_frames))


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


def aggregate_scores_by_voxel(
    scores: torch.Tensor,
    xyz: torch.Tensor,
    *,
    mode: str,
    resolution: int,
    blend: float,
) -> torch.Tensor:
    """Blend per-Gaussian scores with same-voxel spatial context.

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
    if mode not in {"voxel_mean", "voxel_max"}:
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
    fallback_scores: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Register rendered SigLIP2 features back to Gaussian centers and score text.

    This is the P0, no-training object-aware primitive readout: query still
    happens in 3D on Gaussian primitives, while rendered views provide a
    Dr. Splat-style language-aligned registration signal without using LERF
    labels or masks for training/scoring.
    """
    frame_ids = select_registration_frame_ids(
        available_pose_ids=dataset.pose_by_frame_idx.keys(),
        annotated_frame_ids=frame_annotations.keys(),
        official_frame_ids=OPEN_GAUSSIAN_LERF_FRAMES.get(scene, []),
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
            valid_cpu = valid.detach().cpu()
            if valid_cpu.any():
                counts_valid = counts[valid].detach().float().cpu()
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
    silhouette_threshold: float,
    min_select: int,
    output_dir: Path,
    save_masks: bool,
    device: torch.device,
) -> Dict:
    selected = select_gaussians_from_scores(
        scores,
        spec,
        min_select=min_select,
    ).to(device=device, dtype=torch.float32)
    proxy = GaussianSelectionProxy(model, selected)

    ious: List[float] = []
    per_category: Dict[str, List[float]] = {cat: [] for cat in scene_categories}
    per_frame: Dict[str, Dict[str, float]] = {}

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

        active_cats = sorted({obj["category"] for obj in frame_objects})
        frame_scores: Dict[str, float] = {}
        for cat in active_cats:
            if cat not in per_category:
                continue
            cat_idx = scene_categories.index(cat)
            pred = silhouette[cat_idx] > float(silhouette_threshold)
            gt = gt_masks[cat]
            if gt.sum() == 0:
                continue
            iou = mask_iou(pred, gt)
            ious.append(iou)
            per_category[cat].append(iou)
            frame_scores[cat] = iou
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
            "silhouette_threshold": silhouette_threshold,
            "per_category": per_cat_summary,
            "per_frame": per_frame,
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
    text_embedding_cache: Optional[str],
    canonical_embedding_cache: Optional[str],
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
    registered_view_fallback: str,
    registration_frame_mode: str,
    registration_max_frames: int,
    registration_chunk_size: int,
    registration_depth_tolerance: float,
    registration_relative_depth_tolerance: float,
    registration_alpha_threshold: float,
    silhouette_threshold: float,
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
    direct_scores: Optional[torch.Tensor] = None
    needs_direct_scores = score_source == "direct" or (
        score_source == "registered_view" and registered_view_fallback == "direct"
    )
    if needs_direct_scores:
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

    if score_source == "direct":
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
            refiner=_refiner,
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
            silhouette_threshold=silhouette_threshold,
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
    rows.append("- Text head: SigLIP2 summary/text space.")
    rows.append("")
    rows.append("| Selection | mIoU | Acc@0.25 | Acc@0.50 | N |")
    rows.append("|---|---:|---:|---:|---:|")
    for tag, metrics in report["scene"]["results"].items():
        rows.append(
            f"| {tag} | {metrics['miou']:.4f} | {metrics['acc025']:.4f} | "
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
    parser.add_argument("--prompt_templates", default=DEFAULT_PROMPT_TEMPLATES, help="Prompt templates separated by '|'; use {query}")
    parser.add_argument("--selection_mode", choices=["top_ratio", "score_threshold", "mean_std"], default="top_ratio")
    parser.add_argument("--top_ratio", type=float, default=0.02, help="Main fixed Gaussian top-ratio for top_ratio mode")
    parser.add_argument("--ratio_sweep", default="", help="Comma/space separated top-ratio sweep values")
    parser.add_argument("--score_threshold", type=float, default=0.25, help="Main score threshold for score_threshold mode")
    parser.add_argument("--threshold_sweep", default="", help="Comma/space separated score thresholds")
    parser.add_argument("--mean_std", type=float, default=1.0, help="Main mean+std multiplier for mean_std mode")
    parser.add_argument("--mean_std_sweep", default="", help="Comma/space separated mean_std multipliers")
    parser.add_argument("--score_source", choices=["direct", "registered_view"], default="direct", help="Primitive score source for direct 3D selection")
    parser.add_argument("--scoring", choices=["cosine", "softmax_scene", "relevancy"], default="cosine", help="Text-Gaussian score")
    parser.add_argument("--canonical_embedding_cache", default="checkpoints/siglip2_canonical_embeddings.pt", help="Canonical text embeddings for relevancy scoring")
    parser.add_argument("--direct_readout_mode", choices=["gaussian", "knn"], default="gaussian", help="Direct 3D compact readout mode before HCD decoding")
    parser.add_argument("--direct_readout_k", type=int, default=8, help="Neighbour count for knn direct_readout_mode")
    parser.add_argument("--direct_readout_candidate_k", type=int, default=0, help="Optional candidate count before scale-aware KNN pruning")
    parser.add_argument("--compact_feature_key", choices=["features", "fused", "semantic", "geometry"], default="features", help="Hybrid compact readout before HCD decoding")
    parser.add_argument("--softmax_temperature", type=float, default=50.0, help="Logit scale for softmax_scene")
    parser.add_argument("--score_aggregation", choices=["none", "voxel_mean", "voxel_max"], default="none", help="GT-free spatial aggregation applied to Gaussian text scores")
    parser.add_argument("--score_aggregation_resolution", type=int, default=64, help="Voxel resolution per scene axis for score aggregation")
    parser.add_argument("--score_aggregation_blend", type=float, default=0.0, help="Blend weight for aggregated scores; 0 disables aggregation")
    parser.add_argument("--registered_view_fallback", choices=["direct", "low"], default="direct", help="Fallback score for Gaussians not visible in registration views")
    parser.add_argument("--registration_frame_mode", choices=["official", "annotated", "all_poses"], default="official", help="Views used to register rendered features back to primitives")
    parser.add_argument("--registration_max_frames", type=int, default=0, help="Evenly subsample registration views; 0 uses all selected views")
    parser.add_argument("--registration_chunk_size", type=int, default=32768, help="Gaussian chunk size for rendered-view registration sampling")
    parser.add_argument("--registration_depth_tolerance", type=float, default=0.08, help="Absolute depth tolerance for rendered-view primitive registration")
    parser.add_argument("--registration_relative_depth_tolerance", type=float, default=0.02, help="Relative depth tolerance for rendered-view primitive registration")
    parser.add_argument("--registration_alpha_threshold", type=float, default=0.02, help="Minimum rendered alpha for rendered-view primitive registration")
    parser.add_argument("--silhouette_threshold", type=float, default=0.7, help="OpenGaussian-style rendered silhouette threshold")
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
        text_embedding_cache=args.text_embedding_cache,
        canonical_embedding_cache=args.canonical_embedding_cache,
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
        registered_view_fallback=args.registered_view_fallback,
        registration_frame_mode=args.registration_frame_mode,
        registration_max_frames=args.registration_max_frames,
        registration_chunk_size=args.registration_chunk_size,
        registration_depth_tolerance=args.registration_depth_tolerance,
        registration_relative_depth_tolerance=args.registration_relative_depth_tolerance,
        registration_alpha_threshold=args.registration_alpha_threshold,
        silhouette_threshold=args.silhouette_threshold,
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
                "post-refiner rendered SigLIP2 features registered back to Gaussian primitives"
                if args.score_source == "registered_view"
                else "pre-refiner Gaussian-center decoded RADIO-compatible features"
            ),
            "score_source": args.score_source,
            "compact_feature_key": args.compact_feature_key,
            "direct_readout_mode": args.direct_readout_mode,
            "direct_readout_k": args.direct_readout_k,
            "direct_readout_candidate_k": args.direct_readout_candidate_k,
            "text_head": "SigLIP2 summary/text-aligned head",
            "canonical_embedding_cache": args.canonical_embedding_cache if args.scoring == "relevancy" else "",
            "score_aggregation": args.score_aggregation,
            "score_aggregation_resolution": args.score_aggregation_resolution,
            "score_aggregation_blend": args.score_aggregation_blend,
            "registered_view_fallback": args.registered_view_fallback,
            "registration_frame_mode": args.registration_frame_mode,
            "registration_max_frames": args.registration_max_frames,
            "registration_depth_tolerance": args.registration_depth_tolerance,
            "registration_relative_depth_tolerance": args.registration_relative_depth_tolerance,
            "registration_alpha_threshold": args.registration_alpha_threshold,
            "render_role": "render selected primitives only for mask evaluation",
            "metrics": ["mIoU", "Acc@0.25", "Acc@0.50"],
            "silhouette_threshold": args.silhouette_threshold,
        },
        "prompt_templates": prompt_templates,
        "elapsed_seconds": time.time() - t0,
        "scene": scene_report,
    }
    write_scene_report(out_root, args.scene, report)


if __name__ == "__main__":
    main()
