"""Shared query-transient RGB/SAM interface for NVOS and SPIn-NeRF.

The persistent method ends at a rendered signed prompt.  This module defines
the benchmark adapter after that seam: deterministic signed point prompts,
reference-only calibration where a full reference mask is public, and exact
observation clamping after a frozen SAM proposal is lifted back to the field.
No graph, connected component, target mask, or target metric is part of the
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
import torch


class PromptMode(str, Enum):
    SIGNED_SCRIBBLE = "signed_scribble"
    FULL_REFERENCE_MASK = "full_reference_mask"


@dataclass(frozen=True)
class TransientRgbSamPolicy:
    trials: int = 10
    positive_points_per_trial: int = 3
    negative_points_per_trial: int = 3
    prompt_pool_mass_fraction: float = 0.40
    maximum_prompt_pool_rows: int = 8192
    signed_vote_threshold: float = 0.50
    sam_fusion_weight: float = 1.0
    full_mask_threshold_candidates: tuple[float, ...] = tuple(
        value / 100.0 for value in range(5, 100, 5)
    )

    def __post_init__(self) -> None:
        integer_values = (
            self.trials,
            self.positive_points_per_trial,
            self.negative_points_per_trial,
            self.maximum_prompt_pool_rows,
        )
        if any(int(value) <= 0 for value in integer_values):
            raise ValueError("transient RGB/SAM integer policy values must be positive")
        scalars = (
            self.prompt_pool_mass_fraction,
            self.signed_vote_threshold,
            self.sam_fusion_weight,
            *self.full_mask_threshold_candidates,
        )
        if any(not math.isfinite(float(value)) for value in scalars):
            raise ValueError("transient RGB/SAM policy values must be finite")
        if not 0.0 < self.prompt_pool_mass_fraction <= 1.0:
            raise ValueError("prompt_pool_mass_fraction must lie in (0,1]")
        if not 0.0 <= self.signed_vote_threshold <= 1.0:
            raise ValueError("signed_vote_threshold must lie in [0,1]")
        if not 0.0 <= self.sam_fusion_weight <= 1.0:
            raise ValueError("sam_fusion_weight must lie in [0,1]")
        if (
            not self.full_mask_threshold_candidates
            or tuple(sorted(set(self.full_mask_threshold_candidates)))
            != self.full_mask_threshold_candidates
            or any(
                not 0.0 <= float(value) <= 1.0
                for value in self.full_mask_threshold_candidates
            )
        ):
            raise ValueError("full-mask threshold candidates must be unique and sorted")


FROZEN_POLICY = TransientRgbSamPolicy()


@dataclass(frozen=True)
class ReferenceCalibration:
    branch: str
    candidate_index: int
    threshold: float
    reference_iou: float
    canonical_reference_iou: float | None


def _finite_nonnegative_map(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if (
        array.ndim != 2
        or min(array.shape) <= 0
        or not bool(np.isfinite(array).all())
        or bool((array < 0).any())
        or float(array.sum()) <= 0.0
    ):
        raise ValueError(f"{label} must be a finite nonnegative nonempty [H,W] map")
    return array


def _mass_pool(
    values: np.ndarray,
    *,
    needed: int,
    policy: TransientRgbSamPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(-1)
    support = np.flatnonzero(flat > 0.0)
    if support.size <= 0:
        raise ValueError("signed prompt evidence has no positive support")
    order = support[np.argsort(-flat[support], kind="stable")]
    sorted_values = flat[order]
    boundary = float(sorted_values.sum()) * float(policy.prompt_pool_mass_fraction)
    count = int(np.searchsorted(np.cumsum(sorted_values), boundary, side="left")) + 1
    count = max(int(needed), min(count, int(policy.maximum_prompt_pool_rows)))
    count = min(count, order.size)
    rows = order[:count]
    weights = flat[rows] / max(float(flat[rows[0]]), 1e-12)
    return rows.astype(np.int64, copy=False), weights.astype(np.float32, copy=False)


def _weighted_farthest_rows(
    values: np.ndarray,
    *,
    needed: int,
    policy: TransientRgbSamPolicy,
) -> np.ndarray:
    rows, weights = _mass_pool(values, needed=needed, policy=policy)
    height, width = values.shape
    coordinates = np.stack((rows % width, rows // width), axis=-1).astype(np.float32)
    selected = np.empty(needed, dtype=np.int64)
    available = np.ones(rows.size, dtype=bool)
    distance = np.full(rows.size, np.inf, dtype=np.float32)
    diagonal = max(float(height * height + width * width), 1.0)
    for index in range(needed):
        if index == 0:
            choice = 0
        else:
            utility = distance / diagonal + 0.05 * weights
            utility[~available] = -np.inf
            choice = int(np.argmax(utility))
        selected[index] = rows[choice]
        available[choice] = False
        squared = ((coordinates - coordinates[choice]) ** 2).sum(axis=-1)
        distance = np.minimum(distance, squared)
        if not bool(available.any()) and index + 1 < needed:
            available[:] = True
            available[choice] = False
    return selected


def deterministic_signed_point_trials(
    positive_prompt: np.ndarray,
    negative_prompt: np.ndarray,
    *,
    image_shape: tuple[int, int],
    policy: TransientRgbSamPolicy = FROZEN_POLICY,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert rendered signed prompt mass into deterministic SAM points."""

    positive = _finite_nonnegative_map(positive_prompt, label="positive prompt")
    negative = _finite_nonnegative_map(negative_prompt, label="negative prompt")
    if negative.shape != positive.shape:
        raise ValueError("positive and negative prompt maps must match")
    image_height, image_width = map(int, image_shape)
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_shape must contain positive height and width")
    # Prompt locations should express signed evidence, not merely high unsigned
    # response. Remove common-mode mass before sampling and fail closed if one
    # sign has no exclusive evidence; synthesizing the missing sign from an
    # overlapping unsigned response would create contradictory SAM prompts.
    positive_evidence = np.maximum(positive - negative, 0.0)
    negative_evidence = np.maximum(negative - positive, 0.0)
    if min(float(positive_evidence.sum()), float(negative_evidence.sum())) <= 0.0:
        raise ValueError("signed prompt requires exclusive positive and negative evidence")
    positive_count = int(policy.trials * policy.positive_points_per_trial)
    negative_count = int(policy.trials * policy.negative_points_per_trial)
    positive_rows = _weighted_farthest_rows(
        positive_evidence, needed=positive_count, policy=policy
    )
    negative_rows = _weighted_farthest_rows(
        negative_evidence, needed=negative_count, policy=policy
    )
    prompt_height, prompt_width = positive.shape

    def coordinates(rows: np.ndarray) -> np.ndarray:
        x = (rows % prompt_width + 0.5) * image_width / prompt_width - 0.5
        y = (rows // prompt_width + 0.5) * image_height / prompt_height - 0.5
        return np.stack(
            (
                np.clip(x, 0.0, image_width - 1.0),
                np.clip(y, 0.0, image_height - 1.0),
            ),
            axis=-1,
        ).astype(np.float32)

    positive_xy = coordinates(positive_rows).reshape(
        policy.trials, policy.positive_points_per_trial, 2
    )
    negative_xy = coordinates(negative_rows).reshape(
        policy.trials, policy.negative_points_per_trial, 2
    )
    points = np.concatenate((positive_xy, negative_xy), axis=1)
    labels = np.concatenate(
        (
            np.ones(policy.positive_points_per_trial, dtype=np.int32),
            np.zeros(policy.negative_points_per_trial, dtype=np.int32),
        )
    )
    labels = np.broadcast_to(labels, points.shape[:2]).copy()
    return points, labels


def aggregate_sam_trials(
    candidate_masks: np.ndarray,
    *,
    policy: TransientRgbSamPolicy = FROZEN_POLICY,
) -> np.ndarray:
    """Average ten SAM trials without target-dependent candidate selection."""

    masks = np.asarray(candidate_masks)
    if (
        masks.ndim != 4
        or min(masks.shape) <= 0
        or masks.shape[0] != int(policy.trials)
        or not (
            np.issubdtype(masks.dtype, np.bool_)
            or np.issubdtype(masks.dtype, np.number)
        )
    ):
        raise ValueError(
            "SAM candidate masks must be [policy.trials,candidates,H,W]"
        )
    values = masks.astype(np.float32, copy=False)
    if not bool(np.isfinite(values).all()) or bool(((values < 0) | (values > 1)).any()):
        raise ValueError("SAM candidate masks must lie in [0,1]")
    return values.mean(axis=0, dtype=np.float32)


def _iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return 1.0 if union == 0 else float(intersection / union)


def calibrate_full_reference_interface(
    sam_candidates: np.ndarray,
    reference_mask: np.ndarray,
    *,
    canonical_probability: np.ndarray | None = None,
    allow_canonical_fallback: bool = False,
    policy: TransientRgbSamPolicy = FROZEN_POLICY,
) -> ReferenceCalibration:
    """Choose a single candidate/threshold using only the permitted reference mask."""

    candidates = np.asarray(sam_candidates, dtype=np.float32)
    target = np.asarray(reference_mask).astype(bool, copy=False)
    if candidates.ndim != 3 or candidates.shape[1:] != target.shape:
        raise ValueError("SAM candidates and reference mask must align")
    if not bool(np.isfinite(candidates).all()) or bool(
        ((candidates < 0) | (candidates > 1)).any()
    ):
        raise ValueError("SAM reference candidates must lie in [0,1]")
    best = (-1.0, 0, float(policy.full_mask_threshold_candidates[0]))
    for candidate in range(candidates.shape[0]):
        for threshold in policy.full_mask_threshold_candidates:
            score = _iou(candidates[candidate] >= float(threshold), target)
            proposal = (score, -candidate, -float(threshold))
            incumbent = (best[0], -best[1], -best[2])
            if proposal > incumbent:
                best = (score, candidate, float(threshold))
    canonical_iou = None
    branch = "sam"
    candidate_index = best[1]
    threshold = best[2]
    reference_iou = best[0]
    if canonical_probability is not None:
        canonical = np.asarray(canonical_probability, dtype=np.float32)
        if canonical.shape != target.shape or not bool(np.isfinite(canonical).all()):
            raise ValueError("canonical reference probability must align and be finite")
        canonical_best = max(
            (
                _iou(canonical >= float(candidate), target),
                -float(candidate),
            )
            for candidate in policy.full_mask_threshold_candidates
        )
        canonical_iou = float(canonical_best[0])
        if allow_canonical_fallback and canonical_iou >= reference_iou:
            branch = "canonical"
            candidate_index = -1
            threshold = -float(canonical_best[1])
            reference_iou = canonical_iou
    return ReferenceCalibration(
        branch=branch,
        candidate_index=int(candidate_index),
        threshold=float(threshold),
        reference_iou=float(reference_iou),
        canonical_reference_iou=canonical_iou,
    )


def observation_clamped_fusion(
    base_probability: torch.Tensor,
    sam_probability: torch.Tensor,
    *,
    positive_observed: torch.Tensor,
    negative_observed: torch.Tensor,
    policy: TransientRgbSamPolicy = FROZEN_POLICY,
) -> tuple[torch.Tensor, dict[str, int | float | bool]]:
    """Fuse a transient SAM observation while preserving exact signed evidence."""

    base = torch.as_tensor(base_probability)
    sam = torch.as_tensor(sam_probability, device=base.device, dtype=base.dtype)
    positive = torch.as_tensor(positive_observed, device=base.device).bool()
    negative = torch.as_tensor(negative_observed, device=base.device).bool()
    if (
        not base.is_floating_point()
        or sam.shape != base.shape
        or positive.shape != base.shape
        or negative.shape != base.shape
        or not bool(torch.isfinite(base).all())
        or not bool(torch.isfinite(sam).all())
        or bool(((base < 0) | (base > 1)).any())
        or bool(((sam < 0) | (sam > 1)).any())
    ):
        raise ValueError("observation-clamped inputs must be aligned probabilities")
    positive_exclusive = positive & ~negative
    negative_exclusive = negative & ~positive
    conflict = positive & negative
    unknown = ~(positive | negative)
    fused = base.clone()
    weight = float(policy.sam_fusion_weight)
    fused[unknown] = (1.0 - weight) * base[unknown] + weight * sam[unknown]
    fused[positive_exclusive] = 1.0
    fused[negative_exclusive] = 0.0
    fused[conflict] = base[conflict]
    return fused, {
        "positive_exclusive": int(positive_exclusive.sum()),
        "negative_exclusive": int(negative_exclusive.sum()),
        "conflict": int(conflict.sum()),
        "unknown": int(unknown.sum()),
        "sam_fusion_weight": weight,
        "observed_values_preserved_exactly": True,
        "conflicts_preserve_base": True,
    }


def transient_adapter_contract(prompt_mode: PromptMode) -> dict[str, object]:
    """Return the paper-facing persistent/transient boundary."""

    return {
        "prompt_mode": PromptMode(prompt_mode).value,
        "persistent_scene_state": False,
        "query_transient": True,
        "target_rgb_opened": True,
        "target_mask_opened": False,
        "target_metric_used_for_selection": False,
        "signed_positive_negative_points": True,
        "trials": FROZEN_POLICY.trials,
        "points_per_trial": {
            "positive": FROZEN_POLICY.positive_points_per_trial,
            "negative": FROZEN_POLICY.negative_points_per_trial,
        },
        "graph": False,
        "connected_component": False,
        "observation_clamped_fusion": True,
        "full_reference_calibration_only": (
            PromptMode(prompt_mode) is PromptMode.FULL_REFERENCE_MASK
        ),
    }


__all__ = [
    "FROZEN_POLICY",
    "PromptMode",
    "ReferenceCalibration",
    "TransientRgbSamPolicy",
    "aggregate_sam_trials",
    "calibrate_full_reference_interface",
    "deterministic_signed_point_trials",
    "observation_clamped_fusion",
    "transient_adapter_contract",
]
