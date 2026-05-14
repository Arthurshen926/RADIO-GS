"""SAM3 prompt-style segmentation and DINOv3 matching/propagation probes.

The script uses frozen RADIO adaptor projections on either original RADIO RGB
features (teacher) or RADIO-GS rendered features.  It does not call an external
SAM or DINO model; the goal is to test whether reconstructed RADIO features
remain useful in the official SAM3/DINOv3 adaptor spaces. DINO mask propagation
optionally contrasts source-mask features against source-background features.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from radio_gs.data.lerf_dataset import LERFDataset
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint
from radio_gs.scripts.eval_lerf_adaptor_downstream import (
    DEFAULT_RADIO_ADAPTOR_CHECKPOINT,
    _build_frame_masks,
    _load_projected_features,
    _overlay_heatmap,
    _overlay_mask,
    _resolve_gt_feature_dir,
    build_masked_prototype,
    compute_prototype_heatmap,
    select_source_target_pairs,
)
from radio_gs.scripts.eval_lerf_grounding import (
    DEFAULT_LABEL_DIR,
    LERF_OVS_SCENES,
    load_lerf_ovs_labels,
    load_lerf_rgb_frame,
    load_render_pipeline,
    localization_accuracy,
    resolve_lerf_label_dir,
    resolve_lerf_scene_root,
)

logger = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower()).strip("_")
    return slug or "item"


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return an inclusive ``(x0, y0, x1, y1)`` box around a binary mask."""
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        raise ValueError("Cannot compute a box for an empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_centroid_token(mask: np.ndarray, feature_height: int, feature_width: int) -> Tuple[int, int]:
    """Map the binary-mask centroid to the nearest foreground feature token."""
    full_ys, full_xs = np.nonzero(mask > 0)
    if len(full_xs) == 0:
        raise ValueError("Cannot compute a centroid for an empty mask")
    full_centroid_y = float(full_ys.mean())
    full_centroid_x = float(full_xs.mean())

    mask_feat = cv2.resize(
        (mask > 0).astype(np.uint8),
        (feature_width, feature_height),
        interpolation=cv2.INTER_NEAREST,
    )
    ys, xs = np.nonzero(mask_feat > 0)
    if len(xs) == 0:
        y = int(np.clip(round((full_centroid_y + 0.5) * feature_height / mask.shape[0] - 0.5), 0, feature_height - 1))
        x = int(np.clip(round((full_centroid_x + 0.5) * feature_width / mask.shape[1] - 0.5), 0, feature_width - 1))
        return y, x
    centroid = np.array(
        [
            (full_centroid_y + 0.5) * feature_height / mask.shape[0] - 0.5,
            (full_centroid_x + 0.5) * feature_width / mask.shape[1] - 0.5,
        ]
    )
    coords = np.stack([ys, xs], axis=1)
    idx = int(np.argmin(((coords - centroid[None, :]) ** 2).sum(axis=1)))
    y, x = coords[idx]
    return int(y), int(x)


def _box_mask_at_feature_resolution(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask_feat = cv2.resize(
        (mask > 0).astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    x0, y0, x1, y1 = bbox_from_mask(mask_feat)
    box = np.zeros((height, width), dtype=np.uint8)
    box[y0 : y1 + 1, x0 : x1 + 1] = 1
    return box


def dense_match_points(
    source_feature: torch.Tensor,
    target_feature: torch.Tensor,
    source_points: Iterable[Tuple[int, int]],
    *,
    mutual_check: bool = False,
    cycle_max_distance: float = 1.5,
    min_score: Optional[float] = None,
) -> List[Dict[str, float]]:
    """Nearest-neighbor dense matches from source tokens to target tokens."""
    if source_feature.ndim != 3 or target_feature.ndim != 3:
        raise ValueError("Expected source/target features with shape [C,H,W]")
    src = F.normalize(source_feature.float(), dim=0)
    tgt = F.normalize(target_feature.float(), dim=0)
    _, src_h, src_w = src.shape
    channels, tgt_h, tgt_w = tgt.shape
    if channels != src.shape[0]:
        raise ValueError(f"Channel mismatch: {src.shape[0]} vs {channels}")
    src_tokens = src.reshape(channels, src_h * src_w)
    tgt_tokens = tgt.reshape(channels, tgt_h * tgt_w)

    matches: List[Dict[str, float]] = []
    for y, x in source_points:
        sy = int(y)
        sx = int(x)
        if sy < 0 or sy >= src_h or sx < 0 or sx >= src_w:
            continue
        vector = src[:, sy, sx]
        scores = vector @ tgt_tokens
        flat = int(scores.argmax().item())
        ty, tx = divmod(flat, tgt_w)
        score = float(scores[flat].item())
        if min_score is not None and score < float(min_score):
            continue
        match = {
            "src_y": sy,
            "src_x": sx,
            "tgt_y": int(ty),
            "tgt_x": int(tx),
            "score": score,
        }
        if mutual_check:
            reverse_scores = tgt[:, int(ty), int(tx)] @ src_tokens
            reverse_flat = int(reverse_scores.argmax().item())
            reverse_y, reverse_x = divmod(reverse_flat, src_w)
            cycle_distance = float(((reverse_y - sy) ** 2 + (reverse_x - sx) ** 2) ** 0.5)
            if cycle_distance > float(cycle_max_distance):
                continue
            match["reverse_src_y"] = int(reverse_y)
            match["reverse_src_x"] = int(reverse_x)
            match["cycle_distance"] = cycle_distance
        matches.append(match)
    return matches


def filter_matches_by_ransac(
    matches: List[Dict[str, float]],
    *,
    model: str = "none",
    reproj_threshold: float = 1.5,
    min_inliers: int = 4,
) -> List[Dict[str, float]]:
    """Filter dense matches with a GT-free geometric RANSAC model."""
    if model == "none" or not matches:
        return matches
    if model not in {"homography", "fundamental"}:
        raise ValueError(f"Unsupported RANSAC model: {model}")
    min_required = 4 if model == "homography" else 8
    if len(matches) < max(min_required, int(min_inliers)):
        return matches

    src_pts = np.asarray(
        [[float(match["src_x"]), float(match["src_y"])] for match in matches],
        dtype=np.float32,
    )
    tgt_pts = np.asarray(
        [[float(match["tgt_x"]), float(match["tgt_y"])] for match in matches],
        dtype=np.float32,
    )
    cv2.setRNGSeed(0)
    if model == "homography":
        _, inlier_mask = cv2.findHomography(
            src_pts,
            tgt_pts,
            cv2.RANSAC,
            ransacReprojThreshold=float(reproj_threshold),
        )
    else:
        _, inlier_mask = cv2.findFundamentalMat(
            src_pts,
            tgt_pts,
            cv2.FM_RANSAC,
            ransacReprojThreshold=float(reproj_threshold),
            confidence=0.99,
        )
    if inlier_mask is None:
        return matches
    inliers = inlier_mask.reshape(-1).astype(bool)
    if int(inliers.sum()) < int(min_inliers):
        return matches
    filtered: List[Dict[str, float]] = []
    for match, keep in zip(matches, inliers):
        if not keep:
            continue
        out = dict(match)
        out["ransac_inlier"] = 1.0
        filtered.append(out)
    return filtered


def binary_iou(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """IoU between two binary masks at the same resolution."""
    pred = pred_mask.astype(bool)
    target = target_mask.astype(bool)
    if pred.shape != target.shape:
        raise ValueError(f"Mask shape mismatch: {pred.shape} vs {target.shape}")
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(pred, target).sum()
    return float(intersection) / float(union)


def topk_mask_from_scores(
    scores: torch.Tensor | np.ndarray,
    k: int,
    *,
    allowed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a binary mask containing the top-k score locations.

    ``allowed_mask`` is a prompt constraint, e.g. a SAM box prompt. It does not
    use the target evaluation mask.
    """
    score_np = (
        scores.detach().cpu().numpy().astype(np.float32)
        if isinstance(scores, torch.Tensor)
        else np.asarray(scores, dtype=np.float32)
    )
    if score_np.ndim != 2:
        raise ValueError(f"Expected 2D scores, got {score_np.shape}")
    valid = np.ones(score_np.shape, dtype=bool)
    if allowed_mask is not None:
        if allowed_mask.shape != score_np.shape:
            raise ValueError(
                f"allowed_mask shape {allowed_mask.shape} does not match scores {score_np.shape}"
            )
        valid &= allowed_mask.astype(bool)
    valid_indices = np.flatnonzero(valid.reshape(-1))
    if valid_indices.size == 0 or k <= 0:
        return np.zeros(score_np.shape, dtype=np.uint8)
    k_eff = min(int(k), int(valid_indices.size))
    flat_scores = score_np.reshape(-1)
    valid_scores = flat_scores[valid_indices]
    chosen_local = np.argpartition(valid_scores, -k_eff)[-k_eff:]
    chosen = valid_indices[chosen_local]
    out = np.zeros(score_np.size, dtype=np.uint8)
    out[chosen] = 1
    return out.reshape(score_np.shape)


def pooled_token_similarity(
    target_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    *,
    mode: str = "max",
    topk_ratio: float = 0.1,
) -> torch.Tensor:
    """Compute pooled target-to-reference dot-product similarity."""
    if target_tokens.ndim != 2 or reference_tokens.ndim != 2:
        raise ValueError("Expected target/reference tokens with shape [N,C]")
    if target_tokens.shape[1] != reference_tokens.shape[1]:
        raise ValueError(
            f"Token channel mismatch: {target_tokens.shape[1]} vs {reference_tokens.shape[1]}"
        )
    if reference_tokens.shape[0] == 0:
        return torch.zeros(target_tokens.shape[0], dtype=target_tokens.dtype, device=target_tokens.device)
    scores = target_tokens @ reference_tokens.transpose(0, 1)
    if mode == "max":
        return scores.max(dim=1).values
    if mode == "mean":
        return scores.mean(dim=1)
    if mode == "topk_mean":
        k = max(1, int(round(reference_tokens.shape[0] * float(topk_ratio))))
        k = min(k, int(reference_tokens.shape[0]))
        return torch.topk(scores, k=k, dim=1, largest=True).values.mean(dim=1)
    raise ValueError(f"Unsupported token pooling mode: {mode}")


def scaled_bounded_area(
    *,
    source_area: int,
    total_area: int,
    scale: float = 1.0,
    min_area_ratio: float = 0.0,
    max_area_ratio: float = 0.0,
) -> int:
    """Scale a source-support area with optional GT-free feature-grid bounds."""
    total = max(int(total_area), 1)
    area = max(1, int(round(int(source_area) * float(scale))))
    if min_area_ratio > 0:
        area = max(area, int(round(total * float(min_area_ratio))))
    if max_area_ratio > 0:
        area = min(area, max(1, int(round(total * float(max_area_ratio)))))
    return min(max(area, 1), total)


def keep_component_by_score(
    mask: np.ndarray,
    scores: np.ndarray,
    *,
    mode: str = "none",
) -> np.ndarray:
    """Apply GT-free connected-component cleanup to a binary mask."""
    binary = (mask > 0).astype(np.uint8)
    if mode == "none" or not binary.any():
        return binary
    if scores.shape != binary.shape:
        raise ValueError(f"scores shape {scores.shape} does not match mask {binary.shape}")
    num_labels, labels = cv2.connectedComponents(binary, connectivity=4)
    if num_labels <= 1:
        return binary
    score_np = np.asarray(scores, dtype=np.float32)
    if mode == "peak":
        flat = int(np.argmax(np.where(binary > 0, score_np, -np.inf).reshape(-1)))
        label = int(labels.reshape(-1)[flat])
    elif mode == "largest":
        component_ids, counts = np.unique(labels[binary > 0], return_counts=True)
        label = int(component_ids[int(np.argmax(counts))])
    elif mode == "score_sum":
        best_label = 0
        best_score = -float("inf")
        for component_id in range(1, num_labels):
            component = labels == component_id
            component_score = float(score_np[component].sum())
            if component_score > best_score:
                best_score = component_score
                best_label = component_id
        label = best_label
    else:
        raise ValueError(f"Unsupported component cleanup mode: {mode}")
    if label <= 0:
        return np.zeros_like(binary)
    return (labels == label).astype(np.uint8)


def mask_heatmap_outside_prompt(heatmap: torch.Tensor, allowed_mask: np.ndarray) -> torch.Tensor:
    """Suppress heatmap tokens outside a prompt mask before peak-based metrics."""
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got {tuple(heatmap.shape)}")
    if allowed_mask.shape != tuple(heatmap.shape):
        raise ValueError(
            f"allowed_mask shape {allowed_mask.shape} does not match heatmap {tuple(heatmap.shape)}"
        )
    valid = torch.as_tensor(allowed_mask.astype(bool), device=heatmap.device)
    if not bool(valid.any().item()):
        return heatmap
    floor = heatmap.detach().min() - heatmap.detach().abs().max().clamp_min(1.0) - 1.0
    return torch.where(valid, heatmap, floor.to(dtype=heatmap.dtype, device=heatmap.device))


def connected_component_from_seed(mask: np.ndarray, seed_y: int, seed_x: int) -> np.ndarray:
    """Keep only the connected component containing the prompt seed."""
    binary = (mask > 0).astype(np.uint8)
    if binary.size == 0:
        return binary
    seed_y = int(np.clip(seed_y, 0, binary.shape[0] - 1))
    seed_x = int(np.clip(seed_x, 0, binary.shape[1] - 1))
    num_labels, labels = cv2.connectedComponents(binary, connectivity=4)
    if num_labels <= 1:
        return binary
    label = int(labels[seed_y, seed_x])
    if label == 0:
        return np.zeros_like(binary)
    return (labels == label).astype(np.uint8)


def threshold_mask_from_heatmap(
    heatmap: torch.Tensor,
    threshold_ratio: float,
    *,
    allowed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Binarize a heatmap with the same relative threshold used by LERF mIoU."""
    values = heatmap.detach().cpu().float().numpy()
    hmin = float(values.min())
    hmax = float(values.max())
    if hmax - hmin <= 1e-8:
        binary = np.zeros(values.shape, dtype=np.uint8)
    else:
        threshold = hmin + float(threshold_ratio) * (hmax - hmin)
        binary = (values >= threshold).astype(np.uint8)
    if allowed_mask is not None:
        if allowed_mask.shape != binary.shape:
            raise ValueError(
                f"allowed_mask shape {allowed_mask.shape} does not match heatmap {binary.shape}"
            )
        binary = (binary & allowed_mask.astype(np.uint8)).astype(np.uint8)
    return binary


def propagate_mask_by_dense_matches(
    source_feature: torch.Tensor,
    target_feature: torch.Tensor,
    source_mask: np.ndarray,
    *,
    target_area: Optional[int] = None,
    background_contrast: float = 0.0,
    foreground_pool: str = "max",
    foreground_topk_ratio: float = 0.1,
    background_pool: str = "max",
    background_topk_ratio: float = 0.1,
    area_scale: float = 1.0,
    min_area_ratio: float = 0.0,
    max_area_ratio: float = 0.0,
    component_cleanup: str = "none",
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate a source mask to the target view by dense DINO-style matching."""
    if source_feature.ndim != 3 or target_feature.ndim != 3:
        raise ValueError("Expected source/target features with shape [C,H,W]")
    src = F.normalize(source_feature.float(), dim=0)
    tgt = F.normalize(target_feature.float(), dim=0)
    _, src_h, src_w = src.shape
    channels, tgt_h, tgt_w = tgt.shape
    mask = cv2.resize(
        (source_mask > 0).astype(np.uint8),
        (src_w, src_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    if not mask.any():
        return np.zeros((tgt_h, tgt_w), dtype=np.uint8), np.zeros((tgt_h, tgt_w), dtype=np.float32)
    source_tokens = src[:, mask].transpose(0, 1)  # [Ns,C]
    target_tokens = tgt.reshape(channels, tgt_h * tgt_w).transpose(0, 1)  # [Nt,C]
    score_flat = pooled_token_similarity(
        target_tokens,
        source_tokens,
        mode=foreground_pool,
        topk_ratio=foreground_topk_ratio,
    )
    if background_contrast > 0:
        background = ~mask
        if bool(background.any()):
            background_tokens = src[:, background].transpose(0, 1)
            background_scores = pooled_token_similarity(
                target_tokens,
                background_tokens,
                mode=background_pool,
                topk_ratio=background_topk_ratio,
            )
            score_flat = score_flat - float(background_contrast) * background_scores
    score_map = score_flat.reshape(tgt_h, tgt_w)
    raw_area = int(target_area if target_area is not None else mask.sum())
    area = scaled_bounded_area(
        source_area=raw_area,
        total_area=tgt_h * tgt_w,
        scale=area_scale,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
    )
    pred = topk_mask_from_scores(score_map, k=max(area, 1))
    pred = keep_component_by_score(
        pred,
        score_map.detach().cpu().numpy().astype(np.float32),
        mode=component_cleanup,
    )
    return pred, score_map.detach().cpu().numpy().astype(np.float32)


def feature_token_to_image_xy(
    token_y: int,
    token_x: int,
    *,
    feature_height: int,
    feature_width: int,
    image_height: int,
    image_width: int,
) -> Tuple[int, int]:
    """Map a feature-token center to an image pixel coordinate."""
    px = int((float(token_x) + 0.5) * image_width / max(feature_width, 1))
    py = int((float(token_y) + 0.5) * image_height / max(feature_height, 1))
    return min(max(px, 0), image_width - 1), min(max(py, 0), image_height - 1)


def _sample_source_points(mask_feat: np.ndarray, max_points: int) -> List[Tuple[int, int]]:
    ys, xs = np.nonzero(mask_feat > 0)
    if len(xs) == 0:
        return []
    order = np.lexsort((xs, ys))
    coords = np.stack([ys[order], xs[order]], axis=1)
    if len(coords) > max_points:
        take = np.linspace(0, len(coords) - 1, max_points).round().astype(np.int64)
        coords = coords[take]
    return [(int(y), int(x)) for y, x in coords]


def _empty_acc() -> Dict[str, float]:
    return {"correct": 0.0, "total": 0.0, "iou_sum": 0.0}


def _update_seg(acc: Dict[str, float], loc: bool, iou: float) -> None:
    acc["correct"] += float(loc)
    acc["total"] += 1.0
    acc["iou_sum"] += float(iou)


def _finalize_seg(acc: Mapping[str, float]) -> Dict[str, float]:
    total = max(float(acc["total"]), 1.0)
    return {
        "loc_acc": float(acc["correct"]) / total,
        "miou": float(acc["iou_sum"]) / total,
        "n_samples": int(acc["total"]),
    }


def _empty_match_acc() -> Dict[str, float]:
    return {"hits": 0.0, "total": 0.0, "score_sum": 0.0}


def _match_hit(match: Mapping[str, float], target_mask_full: np.ndarray, feature_height: int, feature_width: int) -> bool:
    mh, mw = target_mask_full.shape[:2]
    py = min(int((float(match["tgt_y"]) + 0.5) * mh / feature_height), mh - 1)
    px = min(int((float(match["tgt_x"]) + 0.5) * mw / feature_width), mw - 1)
    return bool(target_mask_full[py, px] > 0)


def _update_match(acc: Dict[str, float], matches: List[Dict[str, float]], target_mask_full: np.ndarray, height: int, width: int) -> None:
    for match in matches:
        acc["hits"] += float(_match_hit(match, target_mask_full, height, width))
        acc["total"] += 1.0
        acc["score_sum"] += float(match["score"])


def _finalize_match(acc: Mapping[str, float]) -> Dict[str, float]:
    total = max(float(acc["total"]), 1.0)
    return {
        "hit_rate": float(acc["hits"]) / total,
        "mean_score": float(acc["score_sum"]) / total,
        "n_matches": int(acc["total"]),
    }


def _draw_matches(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    source_points: List[Tuple[int, int]],
    matches: List[Dict[str, float]],
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    title: str,
    feature_height: int,
    feature_width: int,
) -> np.ndarray:
    src = _overlay_mask(source_rgb, source_mask)
    tgt = _overlay_mask(target_rgb, target_mask)
    height = min(src.shape[0], tgt.shape[0], 420)
    src = cv2.resize(src, (int(src.shape[1] * height / src.shape[0]), height), interpolation=cv2.INTER_AREA)
    tgt = cv2.resize(tgt, (int(tgt.shape[1] * height / tgt.shape[0]), height), interpolation=cv2.INTER_AREA)
    canvas = np.concatenate([src, tgt], axis=1)
    offset_x = src.shape[1]

    colors = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
    ]
    for idx, match in enumerate(matches):
        color = colors[idx % len(colors)]
        src_px, src_py = feature_token_to_image_xy(
            int(match["src_y"]),
            int(match["src_x"]),
            feature_height=feature_height,
            feature_width=feature_width,
            image_height=src.shape[0],
            image_width=src.shape[1],
        )
        tgt_px_local, tgt_py = feature_token_to_image_xy(
            int(match["tgt_y"]),
            int(match["tgt_x"]),
            feature_height=feature_height,
            feature_width=feature_width,
            image_height=tgt.shape[0],
            image_width=tgt.shape[1],
        )
        tgt_px = offset_x + tgt_px_local
        cv2.circle(canvas, (src_px, src_py), 5, color, -1)
        cv2.circle(canvas, (tgt_px, tgt_py), 5, color, -1)
        cv2.line(canvas, (src_px, src_py), (tgt_px, tgt_py), color, 1, cv2.LINE_AA)

    header = np.zeros((34, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title[:120], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([header, canvas], axis=0)


def _save_sam_visual(
    out_dir: Path,
    scene: str,
    task: str,
    category: str,
    frame_id: int,
    rgb: np.ndarray,
    mask_full: np.ndarray,
    heatmaps: Mapping[str, torch.Tensor],
    *,
    family: str = "sam3",
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = [rgb, _overlay_mask(rgb, mask_full)]
    labels = ["RGB", "GT"]
    for mode in ("teacher", "rendered"):
        if mode not in heatmaps:
            continue
        parts.append(_overlay_heatmap(rgb, heatmaps[mode]))
        labels.append(mode)
    height = min(part.shape[0] for part in parts)
    resized = [
        cv2.resize(part, (int(part.shape[1] * height / part.shape[0]), height), interpolation=cv2.INTER_AREA)
        for part in parts
    ]
    grid = np.concatenate(resized, axis=1)
    header = np.zeros((52, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"{scene} | {family} | {task} | {category} | frame {frame_id}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    x = 0
    for label, part in zip(labels, resized):
        cv2.putText(header, label, (x + 8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        x += part.shape[1]
    image = np.concatenate([header, grid], axis=0)
    path = out_dir / (
        f"{_slugify(scene)}_{_slugify(family)}_{_slugify(task)}_"
        f"{frame_id:05d}_{_slugify(category)}.png"
    )
    cv2.imwrite(str(path), image)
    return str(path)


def evaluate_scene_tasks(
    scene: str,
    label_dir: str,
    adaptors: Mapping[str, torch.nn.Module],
    device: torch.device,
    *,
    gt_feature_dir: Optional[Path],
    render_pipeline: Optional[tuple],
    lerf_dataset: Optional[LERFDataset],
    output_dir: Path,
    iou_threshold: float = 0.5,
    max_visuals: int = 10,
    max_match_points: int = 24,
    dino_background_contrast: float = 0.0,
    dino_foreground_pool: str = "max",
    dino_foreground_topk_ratio: float = 0.1,
    dino_background_pool: str = "max",
    dino_background_topk_ratio: float = 0.1,
    dino_area_scale: float = 1.0,
    dino_min_area_ratio: float = 0.0,
    dino_max_area_ratio: float = 0.0,
    dino_component_cleanup: str = "none",
    dino_match_mutual: bool = False,
    dino_match_cycle_max_distance: float = 1.5,
    dino_match_min_score: Optional[float] = None,
    dino_match_ransac_model: str = "none",
    dino_match_ransac_threshold: float = 1.5,
    dino_match_ransac_min_inliers: int = 4,
) -> Dict:
    frame_annotations, _, img_h, img_w = load_lerf_ovs_labels(label_dir, scene)
    frame_ids = sorted(frame_annotations)
    projected = _load_projected_features(
        scene,
        frame_ids,
        adaptors,
        device,
        gt_feature_dir=gt_feature_dir,
        render_pipeline=render_pipeline,
        lerf_dataset=lerf_dataset,
    )
    first_feature = None
    for mode_features in projected.values():
        for adaptor_features in mode_features.values():
            if adaptor_features:
                first_feature = next(iter(adaptor_features.values()))
                break
        if first_feature is not None:
            break
    if first_feature is None:
        raise RuntimeError(f"No projected features available for {scene}")
    feat_h, feat_w = first_feature.shape[-2:]
    full_masks, feat_masks, frames_by_category = _build_frame_masks(
        frame_annotations,
        img_h,
        img_w,
        int(feat_h),
        int(feat_w),
    )

    sam_accs = {
        task: defaultdict(_empty_acc)
        for task in ("point_prompt_segmentation", "box_prompt_segmentation", "mask_prompt_propagation")
    }
    dino_accs = defaultdict(_empty_match_acc)
    dino_mask_accs = defaultdict(_empty_acc)
    visual_paths: List[str] = []
    visual_counts: Dict[str, int] = defaultdict(int)
    scene_root_hint = getattr(render_pipeline[5], "scene_root", "") if render_pipeline is not None else ""

    for task in ("point_prompt_segmentation", "box_prompt_segmentation"):
        for frame_id in frame_ids:
            rgb = load_lerf_rgb_frame(scene, frame_id, scene_root_hint)
            if rgb is None:
                rgb = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            for category, mask_full in full_masks[frame_id].items():
                heatmaps_for_vis = {}
                for mode in ("teacher", "rendered"):
                    feature = projected.get(mode, {}).get("sam3", {}).get(frame_id)
                    if feature is None:
                        continue
                    if task == "point_prompt_segmentation":
                        y, x = mask_centroid_token(mask_full, feature.shape[-2], feature.shape[-1])
                        prototype = F.normalize(feature[:, y, x].float(), dim=0)
                        prompt_mask = None
                        seed = (y, x)
                    else:
                        box_mask = _box_mask_at_feature_resolution(mask_full, feature.shape[-2], feature.shape[-1])
                        prototype = build_masked_prototype(feature, box_mask)
                        prompt_mask = box_mask
                        seed = None
                    heatmap = compute_prototype_heatmap(feature, prototype)
                    metric_heatmap = (
                        mask_heatmap_outside_prompt(heatmap, prompt_mask)
                        if prompt_mask is not None
                        else heatmap
                    )
                    loc = localization_accuracy(metric_heatmap, mask_full)
                    pred_mask = threshold_mask_from_heatmap(
                        heatmap,
                        iou_threshold,
                        allowed_mask=prompt_mask,
                    )
                    if seed is not None:
                        component = connected_component_from_seed(pred_mask, seed_y=seed[0], seed_x=seed[1])
                        if component.any():
                            pred_mask = component
                    iou = binary_iou(pred_mask, feat_masks[frame_id][category])
                    _update_seg(sam_accs[task][mode], loc, iou)
                    heatmaps_for_vis[mode] = heatmap.detach().cpu()
                if (
                    heatmaps_for_vis
                    and visual_counts[task] < max_visuals
                    and category in frames_by_category
                ):
                    visual_paths.append(
                        _save_sam_visual(
                            output_dir / "visualizations" / scene,
                            scene,
                            task,
                            category,
                            frame_id,
                            rgb,
                            mask_full,
                            heatmaps_for_vis,
                        )
                    )
                    visual_counts[task] += 1

    for category, source_frame, target_frame in select_source_target_pairs(frames_by_category):
        source_rgb = load_lerf_rgb_frame(scene, source_frame, scene_root_hint)
        target_rgb = load_lerf_rgb_frame(scene, target_frame, scene_root_hint)
        if source_rgb is None or target_rgb is None:
            continue
        mask_heatmaps = {}
        for mode in ("teacher", "rendered"):
            sam_features = projected.get(mode, {}).get("sam3", {})
            source_feature = sam_features.get(source_frame)
            target_feature = sam_features.get(target_frame)
            if source_feature is None or target_feature is None:
                continue
            prototype = build_masked_prototype(source_feature, feat_masks[source_frame][category])
            heatmap = compute_prototype_heatmap(target_feature, prototype)
            source_area = int(feat_masks[source_frame][category].sum())
            pred_mask = topk_mask_from_scores(
                heatmap,
                k=max(source_area, 1),
            )
            loc = localization_accuracy(heatmap, full_masks[target_frame][category])
            iou = binary_iou(pred_mask, feat_masks[target_frame][category])
            _update_seg(sam_accs["mask_prompt_propagation"][mode], loc, iou)
            mask_heatmaps[mode] = heatmap.detach().cpu()
        if mask_heatmaps and visual_counts["mask_prompt_propagation"] < max_visuals:
            visual_paths.append(
                _save_sam_visual(
                    output_dir / "visualizations" / scene,
                    scene,
                    "mask_prompt_propagation",
                    category,
                    target_frame,
                    target_rgb,
                    full_masks[target_frame][category],
                    mask_heatmaps,
                )
            )
            visual_counts["mask_prompt_propagation"] += 1

        dino_propagation_heatmaps = {}
        for mode in ("teacher", "rendered"):
            dino_features = projected.get(mode, {}).get("dino_v3", {})
            source_feature = dino_features.get(source_frame)
            target_feature = dino_features.get(target_frame)
            if source_feature is None or target_feature is None:
                continue
            source_mask_feat = feat_masks[source_frame][category]
            points = _sample_source_points(source_mask_feat, max_match_points)
            matches = dense_match_points(
                source_feature,
                target_feature,
                points,
                mutual_check=dino_match_mutual,
                cycle_max_distance=dino_match_cycle_max_distance,
                min_score=dino_match_min_score,
            )
            matches = filter_matches_by_ransac(
                matches,
                model=dino_match_ransac_model,
                reproj_threshold=dino_match_ransac_threshold,
                min_inliers=dino_match_ransac_min_inliers,
            )
            _update_match(
                dino_accs[mode],
                matches,
                full_masks[target_frame][category],
                source_feature.shape[-2],
                source_feature.shape[-1],
            )
            propagated_mask, propagation_scores = propagate_mask_by_dense_matches(
                source_feature,
                target_feature,
                source_mask_feat,
                background_contrast=dino_background_contrast,
                foreground_pool=dino_foreground_pool,
                foreground_topk_ratio=dino_foreground_topk_ratio,
                background_pool=dino_background_pool,
                background_topk_ratio=dino_background_topk_ratio,
                area_scale=dino_area_scale,
                min_area_ratio=dino_min_area_ratio,
                max_area_ratio=dino_max_area_ratio,
                component_cleanup=dino_component_cleanup,
            )
            propagation_heatmap = torch.from_numpy(propagation_scores)
            loc = localization_accuracy(
                propagation_heatmap,
                full_masks[target_frame][category],
            )
            iou = binary_iou(propagated_mask, feat_masks[target_frame][category])
            _update_seg(dino_mask_accs[mode], loc, iou)
            dino_propagation_heatmaps[mode] = propagation_heatmap
            if visual_counts[f"dino_{mode}"] < max_visuals and matches:
                drawn = _draw_matches(
                    source_rgb,
                    target_rgb,
                    points,
                    matches,
                    full_masks[source_frame][category],
                    full_masks[target_frame][category],
                    f"{scene} | dino_v3 dense matching | {mode} | {category} | {source_frame}->{target_frame}",
                    source_feature.shape[-2],
                    source_feature.shape[-1],
                )
                vis_dir = output_dir / "visualizations" / scene
                vis_dir.mkdir(parents=True, exist_ok=True)
                path = vis_dir / (
                    f"{_slugify(scene)}_dino_v3_dense_matching_{mode}_"
                    f"{source_frame:05d}_{target_frame:05d}_{_slugify(category)}.png"
                )
                cv2.imwrite(str(path), drawn)
                visual_paths.append(str(path))
                visual_counts[f"dino_{mode}"] += 1
        if dino_propagation_heatmaps and visual_counts["dino_mask_propagation"] < max_visuals:
            visual_paths.append(
                _save_sam_visual(
                    output_dir / "visualizations" / scene,
                    scene,
                    "mask_propagation",
                    category,
                    target_frame,
                    target_rgb,
                    full_masks[target_frame][category],
                    dino_propagation_heatmaps,
                    family="dino_v3",
                )
            )
            visual_counts["dino_mask_propagation"] += 1

    return {
        "sam3": {
            task: {mode: _finalize_seg(acc) for mode, acc in sorted(modes.items())}
            for task, modes in sam_accs.items()
        },
        "dino_v3": {
            "dense_matching": {
                mode: _finalize_match(acc) for mode, acc in sorted(dino_accs.items())
            },
            "mask_propagation": {
                mode: _finalize_seg(acc) for mode, acc in sorted(dino_mask_accs.items())
            },
        },
        "visualizations": visual_paths,
    }


def _print_summary(report: Mapping[str, object]) -> None:
    macro = report.get("macro", {})
    print("\n" + "=" * 80)
    print("  LERF SAM3/DINOv3 TASK SUMMARY")
    print("=" * 80)
    for task, modes in macro.get("sam3", {}).items():
        print(f"\n[SAM3] {task}")
        for mode, metrics in modes.items():
            print(
                f"  {mode:<8} loc={metrics['loc_acc']:.4f} "
                f"miou={metrics['miou']:.4f} n={metrics['n_samples']}"
            )
    dino = macro.get("dino_v3", {}).get("dense_matching", {})
    if dino:
        print("\n[DINOv3] dense_matching")
        for mode, metrics in dino.items():
            print(
                f"  {mode:<8} hit={metrics['hit_rate']:.4f} "
                f"score={metrics['mean_score']:.4f} n={metrics['n_matches']}"
            )
    dino_mask = macro.get("dino_v3", {}).get("mask_propagation", {})
    if dino_mask:
        print("\n[DINOv3] mask_propagation")
        for mode, metrics in dino_mask.items():
            print(
                f"  {mode:<8} loc={metrics['loc_acc']:.4f} "
                f"miou={metrics['miou']:.4f} n={metrics['n_samples']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--scene", default="all")
    parser.add_argument("--label_dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--gt_feature_dir", default=None)
    parser.add_argument("--output_dir", default="output/lerf_sam_dino_tasks")
    parser.add_argument("--radio_adaptor_checkpoint", default=DEFAULT_RADIO_ADAPTOR_CHECKPOINT)
    parser.add_argument("--adaptor_kind", default="feature_projection")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_visuals", type=int, default=10)
    parser.add_argument("--max_match_points", type=int, default=24)
    parser.add_argument("--dino_background_contrast", type=float, default=0.5)
    parser.add_argument("--dino_foreground_pool", default="max", choices=["max", "mean", "topk_mean"])
    parser.add_argument("--dino_foreground_topk_ratio", type=float, default=0.1)
    parser.add_argument("--dino_background_pool", default="max", choices=["max", "mean", "topk_mean"])
    parser.add_argument("--dino_background_topk_ratio", type=float, default=0.1)
    parser.add_argument("--dino_area_scale", type=float, default=1.0)
    parser.add_argument("--dino_min_area_ratio", type=float, default=0.0)
    parser.add_argument("--dino_max_area_ratio", type=float, default=0.0)
    parser.add_argument("--dino_component_cleanup", default="none", choices=["none", "peak", "largest", "score_sum"])
    parser.add_argument("--dino_match_mutual", action="store_true", help="Keep only source-target matches whose reverse nearest neighbor cycles back to the source token")
    parser.add_argument("--dino_match_cycle_max_distance", type=float, default=1.5, help="Maximum feature-grid cycle distance for --dino_match_mutual")
    parser.add_argument("--dino_match_min_score", type=float, default=None, help="Optional cosine-score floor for DINO dense-match visualization/metric")
    parser.add_argument("--dino_match_ransac_model", default="none", choices=["none", "homography", "fundamental"], help="Optional GT-free RANSAC outlier filtering for DINO dense-match diagnostics")
    parser.add_argument("--dino_match_ransac_threshold", type=float, default=1.5, help="Feature-grid reprojection threshold for DINO RANSAC match filtering")
    parser.add_argument("--dino_match_ransac_min_inliers", type=int, default=4, help="Minimum inliers required to accept DINO RANSAC filtering")
    parser.add_argument("--gt_only", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    label_dir = resolve_lerf_label_dir(args.label_dir)
    scenes = LERF_OVS_SCENES if args.scene == "all" else (args.scene,)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if not args.gt_only and (not args.config or not args.checkpoint):
        parser.error("Provide --config and --checkpoint for rendered mode, or pass --gt_only")

    adaptors = {}
    for name in ("sam3", "dino_v3"):
        adaptor = load_radio_adaptor_from_checkpoint(
            args.radio_adaptor_checkpoint,
            name,
            kind=args.adaptor_kind,
        ).to(device)
        adaptor.eval()
        for param in adaptor.parameters():
            param.requires_grad_(False)
        adaptors[name] = adaptor

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    render_pipeline = None
    datasets = {}
    if not args.gt_only:
        render_pipeline = load_render_pipeline(args.config, args.checkpoint, device)
        config = render_pipeline[5]
        for scene in scenes:
            gt_dir = _resolve_gt_feature_dir(args.gt_feature_dir, scene)
            scene_root = resolve_lerf_scene_root(scene, getattr(config, "scene_root", ""))
            datasets[scene] = LERFDataset(
                scene_root=str(scene_root),
                feature_dir=str(gt_dir),
                annotation_dir=str(Path(label_dir) / scene),
                feature_height=getattr(config, "feature_height", 30),
                feature_width=getattr(config, "feature_width", 40),
            )

    scene_reports = {}
    sam_macro = {
        task: defaultdict(_empty_acc)
        for task in ("point_prompt_segmentation", "box_prompt_segmentation", "mask_prompt_propagation")
    }
    dino_macro = defaultdict(_empty_match_acc)
    dino_mask_macro = defaultdict(_empty_acc)

    for scene in scenes:
        print(f"\nScene: {scene}")
        gt_dir = _resolve_gt_feature_dir(args.gt_feature_dir, scene)
        if not gt_dir.exists():
            gt_dir = None
        scene_report = evaluate_scene_tasks(
            scene,
            label_dir,
            adaptors,
            device,
            gt_feature_dir=gt_dir,
            render_pipeline=None if args.gt_only else render_pipeline,
            lerf_dataset=None if args.gt_only else datasets.get(scene),
            output_dir=output_dir,
            iou_threshold=args.iou_threshold,
            max_visuals=args.max_visuals,
            max_match_points=args.max_match_points,
            dino_background_contrast=args.dino_background_contrast,
            dino_foreground_pool=args.dino_foreground_pool,
            dino_foreground_topk_ratio=args.dino_foreground_topk_ratio,
            dino_background_pool=args.dino_background_pool,
            dino_background_topk_ratio=args.dino_background_topk_ratio,
            dino_area_scale=args.dino_area_scale,
            dino_min_area_ratio=args.dino_min_area_ratio,
            dino_max_area_ratio=args.dino_max_area_ratio,
            dino_component_cleanup=args.dino_component_cleanup,
            dino_match_mutual=args.dino_match_mutual,
            dino_match_cycle_max_distance=args.dino_match_cycle_max_distance,
            dino_match_min_score=args.dino_match_min_score,
            dino_match_ransac_model=args.dino_match_ransac_model,
            dino_match_ransac_threshold=args.dino_match_ransac_threshold,
            dino_match_ransac_min_inliers=args.dino_match_ransac_min_inliers,
        )
        scene_reports[scene] = scene_report
        for task, modes in scene_report["sam3"].items():
            for mode, metrics in modes.items():
                sam_macro[task][mode]["correct"] += metrics["loc_acc"] * metrics["n_samples"]
                sam_macro[task][mode]["total"] += metrics["n_samples"]
                sam_macro[task][mode]["iou_sum"] += metrics["miou"] * metrics["n_samples"]
        for mode, metrics in scene_report["dino_v3"]["dense_matching"].items():
            dino_macro[mode]["hits"] += metrics["hit_rate"] * metrics["n_matches"]
            dino_macro[mode]["total"] += metrics["n_matches"]
            dino_macro[mode]["score_sum"] += metrics["mean_score"] * metrics["n_matches"]
        for mode, metrics in scene_report["dino_v3"]["mask_propagation"].items():
            dino_mask_macro[mode]["correct"] += metrics["loc_acc"] * metrics["n_samples"]
            dino_mask_macro[mode]["total"] += metrics["n_samples"]
            dino_mask_macro[mode]["iou_sum"] += metrics["miou"] * metrics["n_samples"]

    macro = {
        "sam3": {
            task: {mode: _finalize_seg(acc) for mode, acc in sorted(modes.items())}
            for task, modes in sam_macro.items()
        },
        "dino_v3": {
            "dense_matching": {
                mode: _finalize_match(acc) for mode, acc in sorted(dino_macro.items())
            },
            "mask_propagation": {
                mode: _finalize_seg(acc) for mode, acc in sorted(dino_mask_macro.items())
            },
        },
    }
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {k: str(v) for k, v in vars(args).items()},
        "protocol": {
            "sam3": (
                "SAM3-adaptor prompted region segmentation: point/box prompts are "
                "derived from each target annotation, and mask propagation uses a "
                "source-view support mask. The frozen RADIO SAM3 adaptor is used; "
                "no external SAM3 mask decoder is called."
            ),
            "dino_v3": (
                "DINOv3-adaptor source-target dense matching and mask propagation: "
                "source masks are propagated with frozen adaptor-space nearest-neighbor "
                "similarity and evaluated on target LERF masks. Optional source-background "
                f"contrast weight is {args.dino_background_contrast:g}; foreground pool is "
                f"{args.dino_foreground_pool}, background pool is {args.dino_background_pool}, "
                f"area_scale={args.dino_area_scale:g}, component_cleanup={args.dino_component_cleanup}; "
                f"dense_match_mutual={bool(args.dino_match_mutual)}, "
                f"ransac_model={args.dino_match_ransac_model}."
            ),
            "gt_usage": "GT masks are used to form prompts/support masks and for final evaluation only.",
        },
        "scenes": scene_reports,
        "macro": macro,
    }
    report_path = output_dir / "lerf_sam_dino_task_results.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    _print_summary(report)
    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
