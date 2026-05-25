"""Feature-only mask refinement utilities for prompt-conditioned SAM-style heads."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def resize_bool_mask(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(shape_hw[0]), int(shape_hw[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target mask shape: {shape_hw}")
    mask_u8 = np.asarray(mask).astype(np.uint8)
    if mask_u8.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask_u8.shape}")
    if mask_u8.shape == (target_h, target_w):
        return mask_u8.astype(bool)
    resized = cv2.resize(mask_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


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


def build_coarse_prompt_from_mask(
    mask: torch.Tensor,
    *,
    dilate: int = 0,
    threshold: float = 0.5,
) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError(f"mask must be [Q,H,W], got {tuple(mask.shape)}")
    coarse = (mask.float() > float(threshold)).float()
    radius = int(dilate)
    if radius > 0 and coarse.numel() > 0:
        kernel = radius * 2 + 1
        coarse = F.max_pool2d(
            coarse.unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=radius,
        ).squeeze(1)
    return coarse


def mask_head_logits_to_candidates(
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


def choose_mask_candidate_by_initial_overlap(
    initial_mask: np.ndarray,
    candidate_masks: np.ndarray,
    *,
    scores: np.ndarray | None = None,
    min_initial_iou: float = 0.05,
    min_refined_area_ratio: float = 0.0,
    max_refined_area_ratio: float = 0.0,
    support_dilate: int = -1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    initial = np.asarray(initial_mask).astype(bool)
    report: Dict[str, Any] = {
        "accepted": False,
        "fallback_reason": "",
        "candidate_count": 0,
        "selected_index": -1,
        "best_initial_overlap": 0.0,
        "selected_score": 0.0,
        "min_refined_area_ratio": float(min_refined_area_ratio),
        "max_refined_area_ratio": float(max_refined_area_ratio),
        "support_dilate": int(support_dilate),
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
    support: np.ndarray | None = None
    if int(support_dilate) >= 0:
        support = initial.astype(np.uint8)
        radius = int(support_dilate)
        if radius > 0:
            kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
            support = cv2.dilate(support, kernel, iterations=1)
        support = support.astype(bool)
    best_idx = -1
    best_overlap = -1.0
    best_score = -float("inf")
    best_mask: np.ndarray | None = None
    for idx, candidate in enumerate(masks):
        cand = np.asarray(candidate) > 0
        if support is not None:
            cand = np.logical_and(cand, support)
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
            best_mask = cand
    report["selected_index"] = int(best_idx)
    report["best_initial_overlap"] = float(max(best_overlap, 0.0))
    report["selected_score"] = float(best_score if np.isfinite(best_score) else 0.0)
    if best_idx < 0:
        report["fallback_reason"] = "no_valid_candidate"
        return initial.copy(), report
    if best_overlap < float(min_initial_iou):
        report["fallback_reason"] = "low_initial_overlap"
        return initial.copy(), report

    refined = np.asarray(best_mask if best_mask is not None else masks[best_idx]).astype(bool)
    refined_area_ratio = float(refined.sum()) / float(max(int(initial.sum()), 1))
    report["refined_area_ratio"] = refined_area_ratio
    if float(min_refined_area_ratio) > 0 and refined_area_ratio < float(min_refined_area_ratio):
        report["fallback_reason"] = "refined_mask_too_small"
        return initial.copy(), report
    if float(max_refined_area_ratio) > 0 and refined_area_ratio > float(max_refined_area_ratio):
        report["fallback_reason"] = "refined_mask_too_large"
        return initial.copy(), report
    report["accepted"] = True
    report["fallback_reason"] = "accepted"
    return refined, report


def choose_prompt_head_refined_mask_with_report(
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
    initial = np.asarray(initial_mask).astype(bool)
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
        candidates = mask_head_logits_to_candidates(
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
    report["backend"] = "prompt_conditioned_ctf_sam3_mask_head_no_rgb"
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
) -> Tuple[np.ndarray, Dict[str, Any]]:
    initial = np.asarray(coarse_mask).astype(bool)
    if initial.ndim != 2:
        raise ValueError(f"Expected 2D coarse mask, got {initial.shape}")
    if feature_map.ndim == 3:
        feature_map = feature_map.unsqueeze(0)
    if feature_map.ndim != 4:
        raise ValueError(f"Expected feature_map [B,C,H,W] or [C,H,W], got {tuple(feature_map.shape)}")
    prompt = prompt_embedding.detach().float()
    if prompt.ndim != 1:
        raise ValueError(f"Expected prompt embedding [D], got {tuple(prompt.shape)}")
    coarse_tensor = torch.from_numpy(initial.astype(np.float32)).unsqueeze(0)
    coarse_tensor = build_coarse_prompt_from_mask(
        coarse_tensor,
        dilate=int(coarse_dilate),
        threshold=float(coarse_threshold),
    )
    try:
        device = next(head.parameters()).device
    except StopIteration:
        device = feature_map.device
    with torch.no_grad():
        logits = head(
            feature_map.to(device=device),
            prompt.to(device=device).view(1, 1, -1),
            coarse_tensor.to(device=device).unsqueeze(0),
        )
    refined, report = choose_prompt_head_refined_mask_with_report(
        initial,
        logits,
        logit_threshold=logit_threshold,
        min_initial_iou=min_initial_iou,
        max_initial_area_fraction=max_initial_area_fraction,
        min_refined_area_ratio=min_refined_area_ratio,
        max_refined_area_ratio=max_refined_area_ratio,
        support_dilate=support_dilate,
    )
    report["coarse_dilate"] = int(coarse_dilate)
    return refined, report


def filter_refined_mask_by_heatmap_support(
    initial_mask: np.ndarray,
    refined_mask: np.ndarray,
    heatmap: np.ndarray | torch.Tensor,
    *,
    min_mean_ratio: float = 0.0,
    min_mass_ratio: float = 0.0,
    require_peak_in_refined: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reject SAM-style refinements that discard the query heatmap evidence.

    This is a GT-free guardrail: the refined mask may sharpen boundaries, but it
    should not move away from the CTF/RADIO query response that produced the
    coarse prompt.
    """
    initial = np.asarray(initial_mask).astype(bool)
    refined = np.asarray(refined_mask).astype(bool)
    if initial.ndim != 2 or refined.ndim != 2:
        raise ValueError(
            f"Expected 2D masks, got initial={initial.shape}, refined={refined.shape}"
        )
    if refined.shape != initial.shape:
        refined = resize_bool_mask(refined, initial.shape)

    heat = heatmap.detach().float().cpu().numpy() if isinstance(heatmap, torch.Tensor) else np.asarray(heatmap)
    if heat.ndim == 3:
        heat = heat[0]
    if heat.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got {heat.shape}")
    heat = heat.astype(np.float32)
    if heat.shape != initial.shape:
        heat = cv2.resize(
            heat,
            (initial.shape[1], initial.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    heat = heat - float(np.nanmin(heat))
    heat_max = float(np.nanmax(heat))
    if heat_max > 1e-8:
        heat = heat / heat_max

    initial_mass = float(heat[initial].sum()) if initial.any() else 0.0
    refined_mass = float(heat[refined].sum()) if refined.any() else 0.0
    initial_mean = float(heat[initial].mean()) if initial.any() else 0.0
    refined_mean = float(heat[refined].mean()) if refined.any() else 0.0
    mass_ratio = refined_mass / max(initial_mass, 1e-8)
    mean_ratio = refined_mean / max(initial_mean, 1e-8)
    peak_y, peak_x = np.unravel_index(int(np.nanargmax(heat)), heat.shape)
    peak_in_refined = bool(refined[peak_y, peak_x])

    report: Dict[str, Any] = {
        "accepted": True,
        "fallback_reason": "accepted",
        "heatmap_initial_mass": initial_mass,
        "heatmap_refined_mass": refined_mass,
        "heatmap_initial_mean": initial_mean,
        "heatmap_refined_mean": refined_mean,
        "heatmap_mass_ratio": float(mass_ratio),
        "heatmap_mean_ratio": float(mean_ratio),
        "heatmap_peak_in_refined": peak_in_refined,
        "min_heatmap_mean_ratio": float(min_mean_ratio),
        "min_heatmap_mass_ratio": float(min_mass_ratio),
        "require_peak_in_refined": bool(require_peak_in_refined),
    }
    if require_peak_in_refined and not peak_in_refined:
        report["accepted"] = False
        report["fallback_reason"] = "heatmap_peak_outside_refined"
        return initial.copy(), report
    if float(min_mean_ratio) > 0 and mean_ratio < float(min_mean_ratio):
        report["accepted"] = False
        report["fallback_reason"] = "low_heatmap_mean_ratio"
        return initial.copy(), report
    if float(min_mass_ratio) > 0 and mass_ratio < float(min_mass_ratio):
        report["accepted"] = False
        report["fallback_reason"] = "low_heatmap_mass_ratio"
        return initial.copy(), report
    return refined.copy(), report
