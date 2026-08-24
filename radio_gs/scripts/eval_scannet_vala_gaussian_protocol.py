#!/usr/bin/env python3
"""Evaluate RADIO-GS on ScanNet in the Gaussian domain used by VALA.

This is intentionally separate from ``eval_scannet_pointcloud_radio_gs``.
The latter queries the learned field at annotated mesh vertices and optionally
calibrates/propagates logits on that mesh.  VALA instead:

1. predicts one open-vocabulary class per optimized Gaussian center;
2. propagates mesh labels to those centers with anisotropic Gaussian-density
   voting; and
3. computes per-scene mIoU/mAcc with opacity-times-volume weights.

The released VALA evaluator forms a radius-culled Euclidean top-k candidate
set and then performs Mahalanobis-density voting.  The implementation below is
an equivalent KD-tree acceleration of that operation; it also records the
row-aligned OpenGaussian metric as an audit-only comparator because the local
ScanNet RGB Gaussians happen to preserve the input mesh vertex count/order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from tqdm import tqdm

from radio_gs.artifact_paths import DEFAULT_SIGLIP2_PROJECTION_WEIGHTS
from radio_gs.config import load_config
from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
)
from radio_gs.scripts.eval_lerf_grounding import parse_prompt_templates
from radio_gs.querying.unified_query import cosine_bank_torch
from radio_gs.querying.sam_categorical_instance_posterior import (
    propagate_categorical_identity_over_proposals,
)
from radio_gs.scripts.build_lerf_sam3_exact_mpr_memberships import (
    _float32_rows_sha256,
)
from radio_gs.utils.checkpoint_io import load_trusted_checkpoint
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import (
    _build_hybrid_model,
    _decode_gaussian_indices_1280,
    _default_label_ply,
    _format_scene_path,
    _load_or_generate_class_text_embeddings,
    _load_projection,
    _parse_scene_list,
    _parse_splits,
    _read_label_ply,
)


DEFAULT_SCENES = (
    "scene0000_00",
    "scene0062_00",
    "scene0070_00",
    "scene0097_00",
    "scene0140_00",
    "scene0347_00",
    "scene0400_00",
    "scene0590_00",
)


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert GraphDECO/VALA ``(w,x,y,z)`` quaternions to rotation matrices."""
    q = np.asarray(quaternion, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = (q[:, idx] for idx in range(4))
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


def assign_vala_pseudo_labels(
    gaussian_xyz: np.ndarray,
    gaussian_scales: np.ndarray,
    gaussian_rotations: np.ndarray,
    point_xyz: np.ndarray,
    point_labels: np.ndarray,
    *,
    radius_factor: float = 5.0,
    candidate_k: int = 1000,
    fallback_k: int = 1,
    class_balance: bool = True,
    chunk_size: int = 512,
) -> tuple[np.ndarray, dict[str, object]]:
    """Assign raw NYU40 labels with VALA's anisotropic density vote.

    KD-tree top-k is equivalent to the released evaluator's dense ``cdist`` +
    radius mask + ``topk`` operation.  ``fallback_k`` implements the fallback
    described in the paper for an empty radius set.  In the prepared eight
    scenes the optimized Gaussian closest point is always inside the radius,
    so this setting does not affect the result.
    """
    gaussian_xyz = np.asarray(gaussian_xyz, dtype=np.float32)
    gaussian_scales = np.asarray(gaussian_scales, dtype=np.float32)
    gaussian_rotations = np.asarray(gaussian_rotations, dtype=np.float32)
    point_xyz = np.asarray(point_xyz, dtype=np.float32)
    point_labels = np.asarray(point_labels, dtype=np.int32)
    if gaussian_xyz.ndim != 2 or gaussian_xyz.shape[1] != 3:
        raise ValueError(f"gaussian_xyz must be [N,3], got {gaussian_xyz.shape}")
    if gaussian_scales.shape != gaussian_xyz.shape:
        raise ValueError("gaussian_scales must match gaussian_xyz")
    if gaussian_rotations.shape != (gaussian_xyz.shape[0], 4):
        raise ValueError("gaussian_rotations must be [N,4]")
    if point_xyz.ndim != 2 or point_xyz.shape[1] != 3:
        raise ValueError(f"point_xyz must be [Q,3], got {point_xyz.shape}")
    if point_labels.shape != (point_xyz.shape[0],):
        raise ValueError("point_labels must align with point_xyz")
    if candidate_k <= 0 or chunk_size <= 0:
        raise ValueError("candidate_k and chunk_size must be positive")

    num_gaussians = gaussian_xyz.shape[0]
    query_k = min(int(candidate_k), point_xyz.shape[0])
    num_labels = int(max(int(point_labels.max(initial=0)) + 1, 1))
    radii = float(radius_factor) * gaussian_scales.max(axis=1)
    rotations = _quaternion_to_rotation_matrix(gaussian_rotations)
    inv_scale = 1.0 / np.maximum(gaussian_scales, 1e-6)
    inv_sqrt_cov = np.einsum(
        "nij,njk,nlk->nil",
        rotations,
        np.eye(3, dtype=np.float32)[None, ...] * inv_scale[:, :, None],
        rotations,
        optimize=True,
    )
    tree = cKDTree(point_xyz)
    labels_out = np.zeros(num_gaussians, dtype=np.int32)
    candidate_counts = np.zeros(num_gaussians, dtype=np.int32)
    empty_before_fallback = np.zeros(num_gaussians, dtype=bool)

    for start in tqdm(
        range(0, num_gaussians, int(chunk_size)),
        desc="VALA pseudo-GT",
    ):
        end = min(start + int(chunk_size), num_gaussians)
        centers = gaussian_xyz[start:end]
        try:
            distances, indices = tree.query(centers, k=query_k, workers=-1)
        except TypeError:  # scipy<1.6
            distances, indices = tree.query(centers, k=query_k)
        distances = np.asarray(distances)
        indices = np.asarray(indices, dtype=np.int64)
        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        valid = distances <= radii[start:end, None]
        empty = ~valid.any(axis=1)
        empty_before_fallback[start:end] = empty
        if fallback_k > 0 and empty.any():
            valid[empty, : min(int(fallback_k), query_k)] = True
        candidate_counts[start:end] = valid.sum(axis=1)

        selected_points = point_xyz[indices]
        selected_labels = point_labels[indices]
        delta = selected_points - centers[:, None, :]
        whitened = np.einsum(
            "gij,gkj->gki", inv_sqrt_cov[start:end], delta, optimize=True
        )
        distance_sq = np.einsum("gki,gki->gk", whitened, whitened, optimize=True)
        # Subtracting one per-Gaussian constant preserves the class argmax but
        # prevents every fallback density from underflowing to zero.
        min_valid_distance_sq = np.where(valid, distance_sq, np.inf).min(axis=1)
        min_valid_distance_sq[~valid.any(axis=1)] = 0.0
        stable_distance_sq = distance_sq - min_valid_distance_sq[:, None]
        weights = np.zeros_like(distance_sq, dtype=np.float32)
        weights[valid] = np.exp(-0.5 * stable_distance_sq[valid]).astype(np.float32)

        rows = np.broadcast_to(
            np.arange(end - start, dtype=np.int64)[:, None], selected_labels.shape
        )
        scores = np.zeros((end - start, num_labels), dtype=np.float32)
        np.add.at(scores, (rows.ravel(), selected_labels.ravel()), weights.ravel())
        if class_balance:
            counts = np.zeros_like(scores)
            np.add.at(
                counts,
                (rows.ravel(), selected_labels.ravel()),
                valid.astype(np.float32).ravel(),
            )
            scores /= np.maximum(counts, 1.0)
        labels_out[start:end] = scores.argmax(axis=1).astype(np.int32)

    quantiles = np.quantile(candidate_counts, [0.0, 0.1, 0.5, 0.9, 0.99, 1.0])
    stats: dict[str, object] = {
        "num_gaussians": int(num_gaussians),
        "num_label_points": int(point_xyz.shape[0]),
        "radius_factor": float(radius_factor),
        "candidate_k": int(candidate_k),
        "fallback_k": int(fallback_k),
        "class_balance": bool(class_balance),
        "empty_before_fallback": int(empty_before_fallback.sum()),
        "candidate_count_quantiles": {
            key: float(value)
            for key, value in zip(("min", "p10", "p50", "p90", "p99", "max"), quantiles)
        },
        "num_at_candidate_cap": int((candidate_counts == query_k).sum()),
    }
    return labels_out, stats


def volume_weighted_split_metrics(
    pseudo_gt_raw: np.ndarray,
    pred_raw: np.ndarray,
    significance: np.ndarray,
    split_ids: Iterable[int],
) -> dict[str, object]:
    """VALA mIoU/mAcc for one scene and one OpenGaussian class split."""
    gt = np.asarray(pseudo_gt_raw, dtype=np.int32)
    pred = np.asarray(pred_raw, dtype=np.int32).copy()
    weights = np.asarray(significance, dtype=np.float64).reshape(-1)
    if gt.shape != pred.shape or gt.shape != weights.shape:
        raise ValueError("gt, pred, and significance must have the same length")
    split_ids = [int(value) for value in split_ids]
    valid = np.isin(gt, split_ids)
    pred[~valid] = 0
    per_class: dict[str, object] = {}
    ious: list[float] = []
    accuracies: list[float] = []
    for class_id in split_ids:
        gt_mask = gt == class_id
        if not gt_mask.any():
            continue
        pred_mask = pred == class_id
        intersection = float(weights[gt_mask & pred_mask].sum())
        union = float(weights[gt_mask | pred_mask].sum())
        total = float(weights[gt_mask].sum())
        iou = intersection / union if union > 0 else 0.0
        accuracy = intersection / total if total > 0 else 0.0
        per_class[str(class_id)] = {
            "name": NYU40_ID_TO_NAME.get(class_id, str(class_id)),
            "iou": iou,
            "acc": accuracy,
            "weighted_gt": total,
        }
        ious.append(iou)
        accuracies.append(accuracy)
    return {
        "miou": float(np.mean(ious)) if ious else 0.0,
        "macc": float(np.mean(accuracies)) if accuracies else 0.0,
        "num_valid_gaussians": int(valid.sum()),
        "num_present_classes": len(ious),
        "per_class": per_class,
    }


def _load_or_build_pseudo_gt(
    cache_path: Path,
    gaussian_xyz: np.ndarray,
    gaussian_scales: np.ndarray,
    gaussian_rotations: np.ndarray,
    point_xyz: np.ndarray,
    point_labels: np.ndarray,
    *,
    radius_factor: float,
    candidate_k: int,
    fallback_k: int,
    class_balance: bool,
    chunk_size: int,
    force: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    geometry_hash = _array_sha256(
        np.concatenate([gaussian_xyz, gaussian_scales, gaussian_rotations], axis=1)
    )
    label_hash = _array_sha256(np.concatenate([point_xyz, point_labels[:, None]], axis=1))
    settings = {
        "radius_factor": float(radius_factor),
        "candidate_k": int(candidate_k),
        "fallback_k": int(fallback_k),
        "class_balance": bool(class_balance),
        "geometry_sha256": geometry_hash,
        "label_cloud_sha256": label_hash,
    }
    if cache_path.exists() and not force:
        cached = np.load(cache_path, allow_pickle=False)
        cached_settings = json.loads(str(cached["settings_json"].item()))
        if cached_settings == settings:
            return np.asarray(cached["pseudo_labels"], dtype=np.int32), json.loads(
                str(cached["stats_json"].item())
            )
    pseudo_labels, stats = assign_vala_pseudo_labels(
        gaussian_xyz,
        gaussian_scales,
        gaussian_rotations,
        point_xyz,
        point_labels,
        radius_factor=radius_factor,
        candidate_k=candidate_k,
        fallback_k=fallback_k,
        class_balance=class_balance,
        chunk_size=chunk_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        pseudo_labels=pseudo_labels,
        settings_json=np.asarray(json.dumps(settings, sort_keys=True)),
        stats_json=np.asarray(json.dumps(stats, sort_keys=True)),
    )
    return pseudo_labels, stats


def _predict_gaussian_labels(
    model: torch.nn.Module,
    codec: torch.nn.Module,
    projection: torch.nn.Module,
    split_text_embeddings: dict[str, torch.Tensor],
    split_names: list[str],
    *,
    device: torch.device,
    chunk_size: int,
    compact_feature_key: str,
    external_query_features: torch.Tensor | None = None,
    external_score_banks: dict[str, torch.Tensor] | None = None,
    sam_region_graph: tuple[torch.Tensor, torch.Tensor] | None = None,
    sam_proposal_memberships: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor
    ] | None = None,
    sam_region_alpha: float = 0.0,
    sam_region_margin_threshold: float = 0.03,
    instance_topology_settings: dict[str, float | int] | None = None,
    instance_topology_stats_out: dict[str, object] | None = None,
    raw_score_banks_out: dict[str, torch.Tensor] | None = None,
) -> dict[str, np.ndarray]:
    if external_score_banks is not None:
        if external_query_features is not None or set(external_score_banks) != set(split_names):
            raise ValueError("external score and feature caches are mutually exclusive")
        score_banks = {
            split: torch.as_tensor(external_score_banks[split]).detach().cpu().float().clone()
            for split in split_names
        }
        for split, scores in score_banks.items():
            expected = (
                model.num_gaussians,
                len(OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]),
            )
            if scores.shape != expected or not bool(torch.isfinite(scores).all()):
                raise ValueError(f"external split{split} score bank differs")
    else:
        score_banks = {
            split: torch.empty(
                (model.num_gaussians, len(OPENGAUSSIAN_NYU40_CLASS_SPLITS[split])),
                dtype=torch.float32,
            )
            for split in split_names
        }
        for start in tqdm(
            range(0, model.num_gaussians, int(chunk_size)),
            desc="Gaussian text classification",
        ):
            end = min(start + int(chunk_size), model.num_gaussians)
            indices = torch.arange(start, end, device=device, dtype=torch.long)
            with torch.no_grad():
                if external_query_features is None:
                    decoded = _decode_gaussian_indices_1280(
                        model,
                        codec,
                        indices,
                        points_xyz=None,
                        compact_feature_key=compact_feature_key,
                    )
                    visual = F.normalize(
                        projection(decoded.unsqueeze(0)).squeeze(0).float(), dim=-1
                    )
                else:
                    visual = external_query_features[start:end].to(
                        device=device, dtype=torch.float32
                    )
                for split in split_names:
                    logits = cosine_bank_torch(
                        visual, split_text_embeddings[split].to(device)
                    )
                    score_banks[split][start:end] = logits.detach().float().cpu()
    predictions: dict[str, np.ndarray] = {}
    for split in split_names:
        scores = score_banks[split]
        if raw_score_banks_out is not None:
            raw_score_banks_out[split] = scores.clone()
        if sam_proposal_memberships is not None:
            if instance_topology_settings is None:
                raise ValueError("SAM proposal memberships require instance topology settings")
            scores, topology_stats = propagate_categorical_identity_over_proposals(
                scores,
                sam_proposal_memberships[0],
                sam_proposal_memberships[1],
                sam_proposal_memberships[2],
                num_proposals=sam_proposal_memberships[3],
                proposal_view_indices=sam_proposal_memberships[4],
                seed_margin_threshold=float(
                    instance_topology_settings["seed_margin_threshold"]
                ),
                update_margin_threshold=float(
                    instance_topology_settings["update_margin_threshold"]
                ),
                semantic_tolerance=float(
                    instance_topology_settings["semantic_tolerance"]
                ),
                consensus_threshold=float(
                    instance_topology_settings["consensus_threshold"]
                ),
                minimum_supporting_proposals=int(
                    instance_topology_settings.get("minimum_supporting_proposals", 2)
                ),
                minimum_supporting_views=int(
                    instance_topology_settings.get("minimum_supporting_views", 1)
                ),
                iterations=int(instance_topology_settings["iterations"]),
            )
            if instance_topology_stats_out is not None:
                instance_topology_stats_out[split] = topology_stats
        elif sam_region_graph is not None:
            if instance_topology_settings is not None:
                scores, topology_stats = (
                    propagate_categorical_identity_over_instance_topology(
                        scores,
                        sam_region_graph[0],
                        sam_region_graph[1],
                        seed_margin_threshold=float(
                            instance_topology_settings["seed_margin_threshold"]
                        ),
                        update_margin_threshold=float(
                            instance_topology_settings["update_margin_threshold"]
                        ),
                        semantic_tolerance=float(
                            instance_topology_settings["semantic_tolerance"]
                        ),
                        consensus_threshold=float(
                            instance_topology_settings["consensus_threshold"]
                        ),
                        iterations=int(instance_topology_settings["iterations"]),
                    )
                )
                if instance_topology_stats_out is not None:
                    instance_topology_stats_out[split] = topology_stats
            elif float(sam_region_alpha) > 0:
                scores, _ = smooth_categorical_scores_with_region_graph(
                    scores,
                    sam_region_graph[0],
                    sam_region_graph[1],
                    alpha=sam_region_alpha,
                    margin_threshold=sam_region_margin_threshold,
                    device=device,
                    chunk_size=chunk_size,
                )
        class_ids = np.asarray(
            OPENGAUSSIAN_NYU40_CLASS_SPLITS[split], dtype=np.int32
        )
        predictions[split] = class_ids[scores.argmax(dim=-1).numpy()]
    return predictions


def load_official_sam_proposal_memberships(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_xyz: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor], dict[str, object]
]:
    """Load query-free official-SAM instances lifted by exact MPR."""

    source = Path(path).expanduser().resolve()
    actual_sha256 = sha256_file(source)
    if len(expected_sha256) != 64 or actual_sha256 != expected_sha256:
        raise ValueError("official-SAM proposal membership SHA-256 mismatch")
    payload = load_trusted_checkpoint(source, map_location="cpu")
    metadata = dict(payload.get("metadata", {}))
    rows = torch.as_tensor(payload.get("row_indices")).long().cpu()
    proposals = torch.as_tensor(payload.get("proposal_indices")).long().cpu()
    weights = torch.as_tensor(payload.get("weights")).float().cpu()
    proposal_views = torch.as_tensor(payload.get("proposal_view_indices")).long().cpu()
    num_rows = int(payload.get("num_rows", -1))
    num_proposals = int(payload.get("num_proposals", -1))
    if (
        payload.get("schema")
        != "radio_gs.scannet_official_sam3_exact_mpr_memberships.v1"
        or num_rows != int(expected_xyz.shape[0])
        or num_proposals <= 0
        or metadata.get("query_independent_proposal_set") is not True
        or metadata.get("official_sam3_decoder") is not True
        or metadata.get("membership_lifting")
        != "exact_front_to_back_marginal_target_weight"
        or metadata.get("xyz_sha256")
        != _float32_rows_sha256(expected_xyz.detach().cpu().float())
        or any(
            bool(metadata.get(key, True))
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "evaluation_rgb_opened",
                "text_queries_opened",
            )
        )
    ):
        raise ValueError("official-SAM proposal membership contract differs")
    if not (rows.shape == proposals.shape == weights.shape):
        raise ValueError("official-SAM sparse membership axes differ")
    if proposal_views.shape != (num_proposals,) or bool((proposal_views < 0).any()):
        raise ValueError("official-SAM proposal view indices differ")
    if rows.numel() and (
        bool(((rows < 0) | (rows >= num_rows)).any())
        or bool(((proposals < 0) | (proposals >= num_proposals)).any())
        or not bool(torch.isfinite(weights).all())
        or bool((weights <= 0).any())
        or bool((weights > 1).any())
    ):
        raise ValueError("official-SAM sparse membership values differ")
    return (rows, proposals, weights, num_proposals, proposal_views), {
        "path": str(source),
        "sha256": actual_sha256,
        "construction": "official_sam3_source_masks_exact_mpr_lift",
        "source_view_count": int(metadata.get("source_view_count", 0)),
        "proposal_count": num_proposals,
        "membership_count": int(rows.numel()),
        "min_membership": float(metadata.get("min_membership", 0.0)),
    }


def load_official_sam_region_features(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_xyz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Load a query-free, exact-MPR lifted official RADIO-SAM feature bank."""

    source = Path(path).expanduser().resolve()
    if len(expected_sha256) != 64 or sha256_file(source) != expected_sha256:
        raise ValueError("official SAM region feature cache SHA-256 mismatch")
    payload = load_trusted_checkpoint(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("official SAM region feature cache must be a mapping")
    metadata = payload.get("metadata")
    features = payload.get("features")
    xyz = payload.get("xyz")
    valid = payload.get("valid")
    if (
        not isinstance(metadata, dict)
        or metadata.get("feature_space") != "sam3"
        or bool(metadata.get("benchmark_masks_opened", True))
        or bool(metadata.get("benchmark_images_opened", True))
        or bool(metadata.get("text_queries_opened", True))
    ):
        raise ValueError("official SAM region cache access contract differs")
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 2
        or int(features.shape[0]) != int(expected_xyz.shape[0])
        or not isinstance(xyz, torch.Tensor)
        or xyz.shape != expected_xyz.shape
        or not isinstance(valid, torch.Tensor)
        or valid.shape != (int(expected_xyz.shape[0]),)
    ):
        raise ValueError("official SAM region cache row shape differs")
    max_xyz_error = float(
        (xyz.float() - expected_xyz.detach().cpu().float()).norm(dim=-1).max()
    )
    if max_xyz_error > 1e-6:
        raise ValueError(f"official SAM region cache xyz mismatch: {max_xyz_error:.3e}")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("official SAM region features contain NaN or infinity")
    return features.detach().cpu(), valid.detach().cpu().bool(), {
        "path": str(source),
        "sha256": expected_sha256,
        "feature_dim": int(features.shape[1]),
        "valid_rows": int(valid.bool().sum()),
        "construction": metadata.get("construction", ""),
        "official_adaptor_checkpoint_sha256": metadata.get(
            "official_adaptor_checkpoint_sha256", ""
        ),
    }


def build_official_sam_region_graph(
    xyz: np.ndarray,
    sam_features: torch.Tensor,
    sam_valid: torch.Tensor,
    *,
    k: int,
    radius: float,
    similarity_threshold: float,
    device: torch.device,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build a local graph whose edges require spatial and SAM agreement."""

    points = np.asarray(xyz, dtype=np.float32)
    count = int(points.shape[0])
    if points.shape != (count, 3) or int(k) <= 0 or float(radius) <= 0:
        raise ValueError("SAM region graph requires [N,3] xyz, positive k and radius")
    if not -1.0 <= float(similarity_threshold) < 1.0:
        raise ValueError("similarity_threshold must be in [-1,1)")
    if sam_features.ndim != 2 or sam_features.shape[0] != count:
        raise ValueError("SAM region features must align with xyz")
    tree = cKDTree(points)
    query_k = min(int(k) + 1, count)
    try:
        distances, indices = tree.query(points, k=query_k, workers=-1)
    except TypeError:
        distances, indices = tree.query(points, k=query_k)
    distances = np.asarray(distances, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    row_ids = np.arange(count, dtype=np.int64)[:, None]
    geometric_valid = (
        np.isfinite(distances)
        & (distances <= float(radius))
        & (indices != row_ids)
    )
    indices = np.clip(indices, 0, max(count - 1, 0))
    feature_bank = F.normalize(
        sam_features.to(device=device, dtype=torch.float32), dim=-1, eps=1e-8
    )
    neighbor_indices = torch.from_numpy(indices).long()
    geometric = torch.from_numpy(geometric_valid)
    distances_t = torch.from_numpy(distances)
    valid_rows = sam_valid.detach().cpu().bool()
    weights = torch.zeros_like(distances_t, dtype=torch.float32)
    for start in range(0, count, int(chunk_size)):
        end = min(start + int(chunk_size), count)
        idx = neighbor_indices[start:end].to(device)
        similarities = (
            feature_bank[start:end, None, :] * feature_bank[idx]
        ).sum(dim=-1)
        affinity = (
            (similarities - float(similarity_threshold))
            / (1.0 - float(similarity_threshold))
        ).clamp(0.0, 1.0)
        spatial = torch.exp(
            -0.5
            * (
                distances_t[start:end].to(device=device, dtype=torch.float32)
                / float(radius)
            ).square()
        )
        legal = geometric[start:end].to(device)
        legal &= valid_rows[start:end].to(device)[:, None]
        legal &= valid_rows[neighbor_indices[start:end]].to(device)
        weights[start:end] = (affinity * spatial * legal.float()).cpu()
    edge_count = int((weights > 0).sum())
    stats: dict[str, object] = {
        "k": int(k),
        "radius": float(radius),
        "similarity_threshold": float(similarity_threshold),
        "edge_count": edge_count,
        "rows_with_neighbors": int((weights.sum(dim=1) > 0).sum()),
        "mean_positive_edge_weight": (
            float(weights[weights > 0].mean()) if edge_count else 0.0
        ),
    }
    del feature_bank
    return neighbor_indices, weights, stats


def smooth_categorical_scores_with_region_graph(
    scores: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_weights: torch.Tensor,
    *,
    alpha: float,
    margin_threshold: float,
    device: torch.device,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Apply a SAM-boundary-aware residual only to uncertain categorical rows."""

    values = torch.as_tensor(scores).detach().cpu().float()
    if values.ndim != 2 or neighbor_indices.shape != neighbor_weights.shape:
        raise ValueError("scores or SAM region graph shape differs")
    if neighbor_indices.ndim != 2 or neighbor_indices.shape[0] != values.shape[0]:
        raise ValueError("SAM region graph rows must align with scores")
    if not 0.0 <= float(alpha) <= 1.0 or float(margin_threshold) < 0:
        raise ValueError("alpha must be in [0,1] and margin_threshold non-negative")
    output = values.clone()
    score_bank = values.to(device)
    changed = 0
    supported = 0
    for start in range(0, int(values.shape[0]), int(chunk_size)):
        end = min(start + int(chunk_size), int(values.shape[0]))
        local = score_bank[start:end].clone()
        idx = neighbor_indices[start:end].to(device=device, dtype=torch.long)
        weights = neighbor_weights[start:end].to(device=device, dtype=torch.float32)
        mass = weights.sum(dim=1)
        supported_mask = mass > 1e-8
        supported += int(supported_mask.sum())
        neighbor_mean = (
            score_bank[idx] * weights[:, :, None]
        ).sum(dim=1) / mass.clamp_min(1e-8)[:, None]
        if local.shape[1] > 1:
            top2 = torch.topk(local, k=2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = local[:, 0].abs()
        apply = supported_mask & (margin <= float(margin_threshold))
        candidate = (1.0 - float(alpha)) * local + float(alpha) * neighbor_mean
        local[apply] = candidate[apply]
        changed += int(apply.sum())
        output[start:end] = local.cpu()
    del score_bank
    return output, {
        "alpha": float(alpha),
        "margin_threshold": float(margin_threshold),
        "supported_rows": supported,
        "changed_rows": changed,
    }


def propagate_categorical_identity_over_instance_topology(
    scores: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_weights: torch.Tensor,
    *,
    seed_margin_threshold: float,
    update_margin_threshold: float,
    semantic_tolerance: float,
    consensus_threshold: float,
    iterations: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Marker-controlled categorical propagation over a SAM topology graph.

    Categorical logits remain the sole identity authority.  Rows whose
    categorical margin exceeds ``seed_margin_threshold`` become immutable
    markers.  SAM/spatial edges may only extend those identities into
    low-margin rows whose own categorical score still considers the proposed
    class plausible.  Conflicting markers therefore stop at a watershed-like
    boundary instead of being averaged across it.
    """

    values = torch.as_tensor(scores).detach().cpu().float()
    indices = torch.as_tensor(neighbor_indices).detach().cpu().long()
    weights = torch.as_tensor(neighbor_weights).detach().cpu().float()
    if values.ndim != 2 or indices.shape != weights.shape:
        raise ValueError("scores or SAM instance topology graph shape differs")
    if indices.ndim != 2 or indices.shape[0] != values.shape[0]:
        raise ValueError("SAM instance topology rows must align with scores")
    if values.shape[1] < 2:
        return values.clone(), {"changed_rows": 0, "seed_rows": 0}
    if not 0.0 <= float(consensus_threshold) <= 1.0:
        raise ValueError("consensus_threshold must be in [0,1]")
    if min(
        float(seed_margin_threshold),
        float(update_margin_threshold),
        float(semantic_tolerance),
    ) < 0 or int(iterations) <= 0:
        raise ValueError("margin/tolerance must be non-negative and iterations positive")

    top2 = torch.topk(values, k=2, dim=-1)
    original_labels = top2.indices[:, 0]
    original_margin = top2.values[:, 0] - top2.values[:, 1]
    immutable = original_margin >= float(seed_margin_threshold)
    eligible = (~immutable) & (
        original_margin <= float(update_margin_threshold)
    )
    owners = torch.full_like(original_labels, -1)
    owners[immutable] = original_labels[immutable]
    assignment_round = torch.full_like(original_labels, -1)
    changed_per_iteration: list[int] = []

    for round_index in range(int(iterations)):
        neighbor_owners = owners[indices]
        valid = (neighbor_owners >= 0) & (weights > 0)
        candidate_rows = eligible & (owners < 0) & valid.any(dim=1)
        row_ids = torch.nonzero(candidate_rows, as_tuple=False).flatten()
        if row_ids.numel() == 0:
            changed_per_iteration.append(0)
            break
        local_owners = neighbor_owners[row_ids]
        local_weights = weights[row_ids] * valid[row_ids].float()
        votes = torch.zeros(
            (row_ids.numel(), values.shape[1]), dtype=torch.float32
        )
        votes.scatter_add_(1, local_owners.clamp_min(0), local_weights)
        support, proposed = votes.max(dim=1)
        mass = votes.sum(dim=1).clamp_min(1e-8)
        consensus = support / mass
        original_best = values[row_ids].max(dim=1).values
        proposed_score = values[row_ids, proposed]
        plausible = (
            original_best - proposed_score
        ) <= float(semantic_tolerance)
        accept = (
            (consensus >= float(consensus_threshold))
            & plausible
            & (support > 0)
        )
        accepted_rows = row_ids[accept]
        owners[accepted_rows] = proposed[accept]
        assignment_round[accepted_rows] = round_index
        count = int(accept.sum())
        changed_per_iteration.append(count)
        if count == 0:
            break

    output = values.clone()
    assigned = eligible & (owners >= 0) & (owners != original_labels)
    assigned_rows = torch.nonzero(assigned, as_tuple=False).flatten()
    if assigned_rows.numel():
        # Only the hard categorical decision is changed.  Preserve all source
        # logits and add the minimum deterministic epsilon required for the
        # marker identity to win; SAM never invents a new class score.
        winner = values[assigned_rows].max(dim=1).values
        output[assigned_rows, owners[assigned_rows]] = winner + 1e-6
    return output, {
        "construction": "marker_controlled_sam_topology_with_categorical_plausibility",
        "seed_margin_threshold": float(seed_margin_threshold),
        "update_margin_threshold": float(update_margin_threshold),
        "semantic_tolerance": float(semantic_tolerance),
        "consensus_threshold": float(consensus_threshold),
        "iterations": int(iterations),
        "seed_rows": int(immutable.sum()),
        "eligible_rows": int(eligible.sum()),
        "owned_rows": int((owners >= 0).sum()),
        "changed_rows": int(assigned.sum()),
        "changed_per_iteration": changed_per_iteration,
    }


def load_method_v1_external_query_features(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_xyz: torch.Tensor,
    expected_dim: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Load one SHA-bound, row-aligned Method-v1 primitive descriptor cache."""

    source = Path(path).expanduser().resolve()
    if len(expected_sha256) != 64:
        raise ValueError(
            "--expected_external_query_feature_cache_sha256 is required"
        )
    actual_sha256 = sha256_file(source)
    if actual_sha256 != expected_sha256:
        raise ValueError("external query feature cache SHA-256 mismatch")
    payload = load_trusted_checkpoint(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("external query feature cache must be a mapping")
    metadata = payload.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("artifact_type")
        != "radio_gs_method_v1_primitive_query_cache"
        or metadata.get("method_id") != "radio-gs-method-v1"
        or metadata.get("query_independent") is not True
        or metadata.get("postprocessing") != "none"
    ):
        raise ValueError("external query feature cache is not a Method-v1 cache")
    if (
        metadata.get("feature_space")
        == "global_decoder_restored_direct_siglip_descriptor"
        and metadata.get("source_gate_passed") is not True
    ):
        raise ValueError("restored capability cache did not pass its source gate")
    features = payload.get("summary_features", payload.get("features"))
    xyz = payload.get("xyz")
    valid = payload.get("valid")
    expected_shape = (int(expected_xyz.shape[0]), int(expected_dim))
    if not isinstance(features, torch.Tensor) or tuple(features.shape) != expected_shape:
        raise ValueError(f"external query features must be {expected_shape}")
    if not isinstance(xyz, torch.Tensor) or xyz.shape != expected_xyz.shape:
        raise ValueError("external query feature cache xyz shape mismatch")
    if not isinstance(valid, torch.Tensor) or valid.shape != (expected_shape[0],):
        raise ValueError("external query feature cache valid mask mismatch")
    if not bool(valid.bool().all()):
        raise ValueError("ScanNet Method-v1 query cache requires all rows valid")
    max_xyz_error = float(
        (xyz.float() - expected_xyz.detach().cpu().float()).norm(dim=-1).max()
    )
    if max_xyz_error > 1e-6:
        raise ValueError(
            f"external query feature cache xyz mismatch: {max_xyz_error:.3e}"
        )
    features = features.float()
    if not bool(torch.isfinite(features).all()):
        raise ValueError("external query features contain NaN or infinity")
    features = F.normalize(features, dim=-1, eps=1e-8)
    return features, {
        "path": str(source),
        "sha256": actual_sha256,
        "feature_dim": int(features.shape[1]),
        "num_gaussians": int(features.shape[0]),
        "method_id": metadata["method_id"],
        "construction": metadata.get("construction", ""),
    }


def load_direct_language_score_cache(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_xyz: torch.Tensor,
    split_names: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Load a label-open direct-language attribution control, never a field."""

    source = Path(path).expanduser().resolve()
    if len(expected_sha256) != 64 or sha256_file(source) != expected_sha256:
        raise ValueError("external direct-language score cache SHA-256 mismatch")
    payload = load_trusted_checkpoint(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("direct-language score cache must be a mapping")
    metadata = payload.get("metadata")
    if (
        payload.get("schema") != "radio_gs.scannet_direct_language_score_cache.v1"
        or not isinstance(metadata, dict)
        or metadata.get("artifact_type")
        != "radio_gs_scannet_direct_language_score_cache"
        or metadata.get("query_independent") is not False
        or metadata.get("evaluation_diagnostic_only") is not True
        or metadata.get("benchmark_masks_opened") is not False
        or metadata.get("benchmark_labels_opened") is not True
        or metadata.get("text_queries_opened") is not True
        or metadata.get("postprocessing") != "none"
    ):
        raise ValueError("direct-language score-cache access contract differs")
    xyz = payload.get("xyz")
    valid = payload.get("valid")
    if not isinstance(xyz, torch.Tensor) or xyz.shape != expected_xyz.shape:
        raise ValueError("direct-language score-cache xyz shape differs")
    if not isinstance(valid, torch.Tensor) or valid.shape != (expected_xyz.shape[0],):
        raise ValueError("direct-language score-cache valid mask differs")
    if not bool(valid.bool().all()):
        raise ValueError("direct-language totality cache must define every row")
    max_xyz_error = float(
        (xyz.float() - expected_xyz.detach().cpu().float()).norm(dim=-1).max()
    )
    if max_xyz_error > 1e-6:
        raise ValueError(f"direct-language score-cache xyz mismatch: {max_xyz_error:.3e}")
    scores: dict[str, torch.Tensor] = {}
    for split in split_names:
        value = payload.get(f"scores_split_{split}")
        expected_shape = (
            int(expected_xyz.shape[0]),
            len(OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]),
        )
        if not isinstance(value, torch.Tensor) or value.shape != expected_shape:
            raise ValueError(f"direct-language split{split} score shape differs")
        value = value.detach().cpu().float()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"direct-language split{split} scores are non-finite")
        scores[split] = value
    return scores, {
        "path": str(source),
        "sha256": expected_sha256,
        "construction": metadata.get("construction", ""),
        "evaluation_diagnostic_only": True,
        "direct_observed_rows": int(
            torch.as_tensor(payload.get("direct_observed")).bool().sum()
        ),
    }


def _scene_macro(scene_results: dict[str, dict[str, object]], protocol: str) -> dict[str, dict[str, float]]:
    macro: dict[str, dict[str, float]] = {}
    for split in ("19", "15", "10"):
        rows = [result[protocol][split] for result in scene_results.values()]
        macro[split] = {
            "miou": float(np.mean([float(row["miou"]) for row in rows])),
            "macc": float(np.mean([float(row["macc"]) for row in rows])),
        }
    return macro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_list", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--prepared_root", default="dataset/scannet_og")
    parser.add_argument("--config", required=True, help="May contain {scene}")
    parser.add_argument("--checkpoint", required=True, help="May contain {scene}")
    parser.add_argument("--label_ply", default=None, help="May contain {scene}")
    parser.add_argument(
        "--output_dir", default="output/scannet_pointcloud_eval/vala_gaussian_protocol"
    )
    parser.add_argument("--class_splits", default="19,15,10")
    parser.add_argument("--feature_chunk_size", type=int, default=8192)
    parser.add_argument("--pseudo_chunk_size", type=int, default=512)
    parser.add_argument("--radius_factor", type=float, default=5.0)
    parser.add_argument("--candidate_k", type=int, default=1000)
    parser.add_argument("--fallback_k", type=int, default=1)
    parser.add_argument("--no_class_balance", action="store_true")
    parser.add_argument("--force_pseudo_gt", action="store_true")
    parser.add_argument(
        "--pseudo_gt_cache_dir",
        default="",
        help="Optional existing pseudo-GT cache directory, independent of output_dir.",
    )
    parser.add_argument("--row_opacity_threshold", type=float, default=0.1)
    parser.add_argument("--compact_feature_key", default="features")
    parser.add_argument("--prompt_templates", default="{query}")
    parser.add_argument("--text_embedding_cache", default=None)
    parser.add_argument("--text_encoder", choices=("siglip2", "openclip"), default="siglip2")
    parser.add_argument("--openclip_model", default="ViT-B-16")
    parser.add_argument("--openclip_pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--class_aliases", default="none")
    parser.add_argument("--projection_weights", default=DEFAULT_SIGLIP2_PROJECTION_WEIGHTS)
    parser.add_argument("--summary_head_weights", default="checkpoints/siglip2_summary_head.pth")
    parser.add_argument(
        "--external_query_feature_cache",
        default="",
        help="May contain {scene}; row-aligned Method-v1 primitive descriptors.",
    )
    parser.add_argument(
        "--expected_external_query_feature_cache_sha256",
        default="",
        help="Required SHA-256 when one external cache is evaluated.",
    )
    parser.add_argument(
        "--external_query_score_cache",
        default="",
        help=(
            "One-scene, SHA-bound direct-language score attribution cache; "
            "diagnostic only and mutually exclusive with a feature cache."
        ),
    )
    parser.add_argument(
        "--expected_external_query_score_cache_sha256",
        default="",
        help="Required SHA-256 for --external_query_score_cache.",
    )
    parser.add_argument(
        "--sam_region_feature_cache",
        default="",
        help="May contain {scene}; row-aligned exact-MPR official RADIO-SAM features.",
    )
    parser.add_argument("--expected_sam_region_feature_cache_sha256", default="")
    parser.add_argument("--sam_region_k", type=int, default=8)
    parser.add_argument("--sam_region_radius", type=float, default=0.10)
    parser.add_argument("--sam_region_similarity_threshold", type=float, default=0.50)
    parser.add_argument("--sam_region_alpha", type=float, default=0.25)
    parser.add_argument("--sam_region_margin_threshold", type=float, default=0.03)
    parser.add_argument(
        "--sam_proposal_membership_cache",
        default="",
        help="May contain {scene}; exact-MPR lifted official-SAM proposal memberships.",
    )
    parser.add_argument("--expected_sam_proposal_membership_cache_sha256", default="")
    parser.add_argument("--sam_instance_topology", action="store_true")
    parser.add_argument("--sam_instance_seed_margin", type=float, default=0.04)
    parser.add_argument("--sam_instance_update_margin", type=float, default=0.04)
    parser.add_argument("--sam_instance_semantic_tolerance", type=float, default=0.025)
    parser.add_argument("--sam_instance_consensus", type=float, default=0.70)
    parser.add_argument("--sam_instance_minimum_supporting_proposals", type=int, default=2)
    parser.add_argument("--sam_instance_minimum_supporting_views", type=int, default=1)
    parser.add_argument("--sam_instance_iterations", type=int, default=6)
    parser.add_argument(
        "--save_development_score_cache",
        action="store_true",
        help="Save raw categorical logits and SAM graph for development sweeps.",
    )
    parser.add_argument(
        "--score_cache_only",
        action="store_true",
        help=(
            "Materialize raw logits plus a validated SAM topology without applying "
            "a proposal/category postprocessor. Requires --save_development_score_cache."
        ),
    )
    parser.add_argument(
        "--allow_topology_free_score_cache",
        action="store_true",
        help=(
            "Save only raw category scores, pseudo labels, and significance when "
            "an independently SHA-bound topology is consumed by a downstream "
            "evaluator. Requires --score_cache_only."
        ),
    )
    parser.add_argument(
        "--radio_checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--use_summary_head", action="store_true", default=True)
    parser.add_argument("--no_summary_head", dest="use_summary_head", action="store_false")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scenes = _parse_scene_list(args.scene_list) or []
    split_names = _parse_splits(args.class_splits)
    if split_names != ["19", "15", "10"]:
        raise ValueError("This audit currently requires class_splits=19,15,10")
    if args.score_cache_only and not args.save_development_score_cache:
        raise ValueError("--score_cache_only requires --save_development_score_cache")
    if args.allow_topology_free_score_cache and not args.score_cache_only:
        raise ValueError("topology-free score cache requires --score_cache_only")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    projection = _load_projection(args, device)
    prompt_templates = parse_prompt_templates(args.prompt_templates)
    split_text_embeddings: dict[str, torch.Tensor] = {}
    for split in split_names:
        class_names = [
            NYU40_ID_TO_NAME[class_id]
            for class_id in OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
        ]
        cache_path = None
        if args.text_embedding_cache:
            base = Path(args.text_embedding_cache)
            cache_path = str(base.with_name(f"{base.stem}_split{split}.pt"))
        split_text_embeddings[split] = _load_or_generate_class_text_embeddings(
            class_names,
            device,
            cache_path=cache_path,
            prompt_templates=prompt_templates,
            class_aliases=args.class_aliases,
            text_encoder=args.text_encoder,
            openclip_model=args.openclip_model,
            openclip_pretrained=args.openclip_pretrained,
        )

    scene_results: dict[str, dict[str, object]] = {}
    prepared_root = Path(args.prepared_root)
    for scene in scenes:
        print(f"\n=== {scene} ===", flush=True)
        config_path = _format_scene_path(args.config, scene)
        checkpoint_path = _format_scene_path(args.checkpoint, scene)
        config = load_config(config_path)
        model, codec = _build_hybrid_model(config, checkpoint_path, device)
        gaussian_xyz = model.get_xyz().detach().float().cpu().numpy()
        gaussian_scales = model.get_scaling().detach().float().cpu().numpy()
        gaussian_rotations = model.get_rotation().detach().float().cpu().numpy()
        gaussian_opacity = model.get_opacity().detach().float().cpu().numpy().reshape(-1)
        external_query_features = None
        external_query_feature_record = None
        external_score_banks = None
        external_score_record = None
        sam_region_graph = None
        sam_region_record = None
        sam_proposal_memberships = None
        sam_proposal_record = None
        if args.sam_region_feature_cache and args.sam_proposal_membership_cache:
            raise ValueError("select either SAM feature graph or exact proposal memberships")
        if args.external_query_feature_cache and args.external_query_score_cache:
            raise ValueError(
                "external query feature and direct-language score caches are mutually exclusive"
            )
        if args.external_query_feature_cache:
            if len(scenes) != 1:
                raise ValueError(
                    "SHA-bound external query features currently require one scene per run"
                )
            external_path = _format_scene_path(
                args.external_query_feature_cache, scene
            )
            external_query_features, external_query_feature_record = (
                load_method_v1_external_query_features(
                    external_path,
                    expected_sha256=args.expected_external_query_feature_cache_sha256,
                    expected_xyz=model.get_xyz(),
                    expected_dim=int(split_text_embeddings[split_names[0]].shape[1]),
                )
            )
        if args.external_query_score_cache:
            if len(scenes) != 1:
                raise ValueError(
                    "SHA-bound direct-language scores currently require one scene per run"
                )
            score_path = _format_scene_path(args.external_query_score_cache, scene)
            external_score_banks, external_score_record = (
                load_direct_language_score_cache(
                    score_path,
                    expected_sha256=args.expected_external_query_score_cache_sha256,
                    expected_xyz=model.get_xyz(),
                    split_names=split_names,
                )
            )
        if args.sam_region_feature_cache:
            if len(scenes) != 1:
                raise ValueError("SHA-bound SAM region features require one scene per run")
            sam_path = _format_scene_path(args.sam_region_feature_cache, scene)
            sam_features, sam_valid, sam_region_record = load_official_sam_region_features(
                sam_path,
                expected_sha256=args.expected_sam_region_feature_cache_sha256,
                expected_xyz=model.get_xyz(),
            )
            graph_indices, graph_weights, graph_stats = build_official_sam_region_graph(
                gaussian_xyz,
                sam_features,
                sam_valid,
                k=args.sam_region_k,
                radius=args.sam_region_radius,
                similarity_threshold=args.sam_region_similarity_threshold,
                device=device,
                chunk_size=args.feature_chunk_size,
            )
            sam_region_graph = (graph_indices, graph_weights)
            sam_region_record.update(graph_stats)
            sam_region_record.update({
                "alpha": float(args.sam_region_alpha),
                "margin_threshold": float(args.sam_region_margin_threshold),
            })
        if args.sam_proposal_membership_cache:
            if len(scenes) != 1:
                raise ValueError("SHA-bound SAM proposal memberships require one scene")
            proposal_path = _format_scene_path(
                args.sam_proposal_membership_cache, scene
            )
            sam_proposal_memberships, sam_proposal_record = (
                load_official_sam_proposal_memberships(
                    proposal_path,
                    expected_sha256=args.expected_sam_proposal_membership_cache_sha256,
                    expected_xyz=model.get_xyz(),
                )
            )
        label_ply = _format_scene_path(args.label_ply, scene) or _default_label_ply(
            prepared_root, scene
        )
        point_xyz, point_labels = _read_label_ply(label_ply)
        pseudo_cache_root = (
            Path(args.pseudo_gt_cache_dir)
            if args.pseudo_gt_cache_dir
            else output_dir / "pseudo_gt"
        )
        pseudo_labels, pseudo_stats = _load_or_build_pseudo_gt(
            pseudo_cache_root / f"{scene}.npz",
            gaussian_xyz,
            gaussian_scales,
            gaussian_rotations,
            point_xyz,
            point_labels,
            radius_factor=args.radius_factor,
            candidate_k=args.candidate_k,
            fallback_k=args.fallback_k,
            class_balance=not args.no_class_balance,
            chunk_size=args.pseudo_chunk_size,
            force=args.force_pseudo_gt,
        )
        topology_stats: dict[str, object] = {}
        topology_settings = (
            {
                "seed_margin_threshold": args.sam_instance_seed_margin,
                "update_margin_threshold": args.sam_instance_update_margin,
                "semantic_tolerance": args.sam_instance_semantic_tolerance,
                "consensus_threshold": args.sam_instance_consensus,
                "minimum_supporting_proposals": args.sam_instance_minimum_supporting_proposals,
                "minimum_supporting_views": args.sam_instance_minimum_supporting_views,
                "iterations": args.sam_instance_iterations,
            }
            if args.sam_instance_topology
            else None
        )
        raw_score_banks: dict[str, torch.Tensor] = {}
        predictions = _predict_gaussian_labels(
            model,
            codec,
            projection,
            split_text_embeddings,
            split_names,
            device=device,
            chunk_size=args.feature_chunk_size,
            compact_feature_key=args.compact_feature_key,
            external_query_features=external_query_features,
            external_score_banks=external_score_banks,
            sam_region_graph=sam_region_graph,
            sam_proposal_memberships=(
                None if args.score_cache_only else sam_proposal_memberships
            ),
            sam_region_alpha=args.sam_region_alpha,
            sam_region_margin_threshold=args.sam_region_margin_threshold,
            instance_topology_settings=topology_settings,
            instance_topology_stats_out=topology_stats,
            raw_score_banks_out=(
                raw_score_banks if args.save_development_score_cache else None
            ),
        )
        significance = gaussian_scales.prod(axis=1) * gaussian_opacity
        if point_labels.shape[0] != gaussian_xyz.shape[0]:
            row_labels = np.zeros_like(pseudo_labels)
            row_available = False
            row_distance = None
        else:
            row_labels = point_labels
            row_available = True
            row_distance = np.linalg.norm(gaussian_xyz - point_xyz, axis=1)
        row_weights = np.ones_like(significance)
        row_labels_filtered = row_labels.copy()
        row_labels_filtered[gaussian_opacity < float(args.row_opacity_threshold)] = 0
        vala_metrics = {}
        row_metrics = {}
        for split in split_names:
            split_ids = OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
            vala_metrics[split] = volume_weighted_split_metrics(
                pseudo_labels, predictions[split], significance, split_ids
            )
            row_metrics[split] = volume_weighted_split_metrics(
                row_labels_filtered, predictions[split], row_weights, split_ids
            )
            print(
                f"split{split}: VALA-pseudo {vala_metrics[split]['miou']:.4f}/"
                f"{vala_metrics[split]['macc']:.4f}; row {row_metrics[split]['miou']:.4f}/"
                f"{row_metrics[split]['macc']:.4f}",
                flush=True,
            )
        prediction_path = output_dir / "predictions" / f"{scene}.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            gaussian_xyz=gaussian_xyz,
            pseudo_labels=pseudo_labels,
            row_labels=row_labels,
            significance=significance,
            **{f"pred_split_{split}": predictions[split] for split in split_names},
        )
        if args.save_development_score_cache:
            if (
                sam_region_graph is None
                and sam_proposal_memberships is None
                and not args.allow_topology_free_score_cache
            ):
                raise ValueError("development score cache requires a SAM topology")
            score_cache_path = output_dir / "development" / f"{scene}_scores.npz"
            score_cache_path.parent.mkdir(parents=True, exist_ok=True)
            topology_arrays = (
                {
                    "sam_neighbor_indices": sam_region_graph[0].numpy(),
                    "sam_neighbor_weights": sam_region_graph[1].numpy(),
                }
                if sam_region_graph is not None
                else {
                    "sam_membership_rows": sam_proposal_memberships[0].numpy(),
                    "sam_membership_proposals": sam_proposal_memberships[1].numpy(),
                    "sam_membership_weights": sam_proposal_memberships[2].numpy(),
                    "sam_num_proposals": np.asarray(sam_proposal_memberships[3]),
                    "sam_proposal_view_indices": sam_proposal_memberships[4].numpy(),
                }
                if sam_proposal_memberships is not None
                else {}
            )
            np.savez_compressed(
                score_cache_path,
                gaussian_xyz=gaussian_xyz,
                pseudo_labels=pseudo_labels,
                significance=significance,
                **topology_arrays,
                **{
                    f"scores_split_{split}": raw_score_banks[split].numpy()
                    for split in split_names
                },
            )
        scene_results[scene] = {
            "num_gaussians": int(gaussian_xyz.shape[0]),
            "num_label_points": int(point_xyz.shape[0]),
            "label_ply": str(label_ply),
            "pseudo_gt": pseudo_stats,
            "row_alignment": {
                "available": row_available,
                "mean_center_displacement": (
                    float(row_distance.mean()) if row_distance is not None else None
                ),
                "p95_center_displacement": (
                    float(np.quantile(row_distance, 0.95)) if row_distance is not None else None
                ),
                "pseudo_row_label_agreement": (
                    float((pseudo_labels == row_labels).mean()) if row_available else None
                ),
            },
            "vala_pseudo_volume": vala_metrics,
            "opengaussian_row_unweighted": row_metrics,
            "prediction_npz": str(prediction_path),
            "query_feature_source": (
                external_score_record
                if external_score_record is not None
                else (
                    external_query_feature_record
                    if external_query_feature_record is not None
                    else {"source": "decoded_geometry_checkpoint"}
                )
            ),
            "sam_region_readout": (
                {
                    **(
                        sam_proposal_record
                        if sam_proposal_record is not None
                        else sam_region_record
                    ),
                    "instance_topology": topology_stats,
                }
                if sam_region_record is not None or sam_proposal_record is not None
                else {"enabled": False}
            ),
        }
        del model, codec
        torch.cuda.empty_cache()

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "prediction_domain": "optimized Gaussian centers",
            "prediction_postprocess": (
                (
                    "marker_controlled_official_sam_instance_topology"
                    if args.sam_instance_topology
                    else "official_sam_local_low_margin_residual"
                )
                if args.sam_region_feature_cache or args.sam_proposal_membership_cache
                else "none"
            ),
            "test_scene_calibration": "none",
            "pseudo_gt": "VALA Mahalanobis-density voting",
            "metric_weights": "opacity * sx * sy * sz",
            "scene_aggregation": "unweighted scene macro",
            "text_encoder": args.text_encoder,
            "prompt_templates": prompt_templates,
            "primitive_readout": (
                "external_direct_language_score_attribution_cache"
                if args.external_query_score_cache
                else (
                    "external_method_v1_query_feature_cache"
                    if args.external_query_feature_cache
                    else "decoded_geometry_checkpoint"
                )
            ),
        },
        "args": {key: str(value) for key, value in vars(args).items()},
        "macro": {
            "vala_pseudo_volume": _scene_macro(scene_results, "vala_pseudo_volume"),
            "opengaussian_row_unweighted": _scene_macro(
                scene_results, "opengaussian_row_unweighted"
            ),
        },
        "scenes": scene_results,
    }
    report_path = output_dir / "scannet_vala_gaussian_protocol_results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nMacro:")
    print(json.dumps(report["macro"], indent=2))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
