"""Source-only local positive evidence from repeated NVOS completion trials."""

from __future__ import annotations

import math
from typing import Mapping

import torch


METHOD = "local_majority_positive_proposal_v2"
TRIAL_COUNT = 10
MAJORITY_THRESHOLD = 0.5


def method_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "method": METHOD,
        "input": "immutable_official_sam3_binary_source_trial_masks",
        "completion_probability": "q=arithmetic_mean_of_binary_trials",
        "proposal_region": "q>0.5",
        "proposal_probability": "p=1",
        "proposal_reliability": "c=2*q-1",
        "nonproposal_policy": "c=0_no_negative_evidence",
        "raw_positive_override": "p=1,c=1",
        "raw_negative_override": "p=0,c=1",
        "scribble_conflict_policy": "fail_closed",
        "majority_threshold": MAJORITY_THRESHOLD,
        "parameter_sweep": False,
        "learned_or_scene_specific_constants": False,
        "uses_target_rgb_mask_or_metric": False,
    }


def _bool_mask(value: torch.Tensor, shape: torch.Size, label: str) -> torch.Tensor:
    mask = torch.as_tensor(value)
    if mask.device.type != "cpu" or mask.dtype != torch.bool or mask.shape != shape:
        raise ValueError(f"{label} must be a CPU bool tensor with shape {tuple(shape)}")
    return mask.contiguous()


def local_majority_positive_evidence(
    completion_probability: torch.Tensor,
    *,
    positive_scribble: torch.Tensor,
    negative_scribble: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Bernoulli proposal probability and separate reliability.

    Non-proposal pixels use a neutral probability only for a well-defined
    tensor value.  Their reliability is exactly zero, so they contribute no
    foreground or background observation mass.
    """

    q = torch.as_tensor(completion_probability)
    if q.device.type != "cpu" or not q.dtype.is_floating_point or q.ndim != 2:
        raise ValueError("completion_probability must be a CPU floating [H,W] tensor")
    q = q.float().contiguous()
    if not torch.isfinite(q).all() or bool((q < 0).any()) or bool((q > 1).any()):
        raise ValueError("completion_probability must be finite and in [0,1]")
    positive = _bool_mask(positive_scribble, q.shape, "positive_scribble")
    negative = _bool_mask(negative_scribble, q.shape, "negative_scribble")
    if bool((positive & negative).any()):
        raise ValueError("positive and negative scribbles overlap")

    proposal = q > MAJORITY_THRESHOLD
    probability = torch.full_like(q, 0.5)
    probability[proposal] = 1.0
    reliability = torch.where(proposal, 2.0 * q - 1.0, torch.zeros_like(q))
    probability[positive] = 1.0
    reliability[positive] = 1.0
    probability[negative] = 0.0
    reliability[negative] = 1.0
    return probability.contiguous(), reliability.contiguous()


def _safe_ratio(numerator: float, denominator: float, *, empty: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float(empty)


def source_only_loo_diagnostic(
    trial_masks: torch.Tensor,
    *,
    positive_scribble: torch.Tensor,
    negative_scribble: torch.Tensor,
) -> dict[str, object]:
    """Evaluate fixed local-positive evidence against held-out source trials."""

    masks = torch.as_tensor(trial_masks)
    if (
        masks.device.type != "cpu"
        or masks.dtype != torch.bool
        or masks.ndim != 3
        or masks.shape[0] != TRIAL_COUNT
        or masks.shape[1] <= 0
        or masks.shape[2] <= 0
    ):
        raise ValueError(
            f"trial_masks must be CPU bool [{TRIAL_COUNT},H,W] with nonzero H,W"
        )
    masks = masks.contiguous()
    shape = masks.shape[1:]
    positive = _bool_mask(positive_scribble, shape, "positive_scribble")
    negative = _bool_mask(negative_scribble, shape, "negative_scribble")
    if bool((positive & negative).any()):
        raise ValueError("positive and negative scribbles overlap")
    raw = positive | negative
    evaluable = ~raw
    evaluable_pixels = int(evaluable.sum())
    if evaluable_pixels <= 0:
        raise ValueError("source diagnostic requires non-scribble pixels")

    full_q = masks.float().mean(dim=0)
    full_proposal = (full_q > MAJORITY_THRESHOLD) & evaluable
    full_confidence = torch.where(
        full_proposal, 2.0 * full_q - 1.0, torch.zeros_like(full_q)
    )

    total = masks.to(torch.int16).sum(dim=0)
    records: list[dict[str, object]] = []
    pooled_proposal = 0
    pooled_intersection = 0
    pooled_heldout_positive = 0
    pooled_union = 0
    pooled_confidence = 0.0
    pooled_true_confidence = 0.0
    for index in range(TRIAL_COUNT):
        heldout = masks[index] & evaluable
        q = (total - masks[index].to(torch.int16)).float() / float(
            TRIAL_COUNT - 1
        )
        proposal = (q > MAJORITY_THRESHOLD) & evaluable
        confidence = torch.where(
            proposal, 2.0 * q - 1.0, torch.zeros_like(q)
        )
        proposal_pixels = int(proposal.sum())
        heldout_pixels = int(heldout.sum())
        intersection = int((proposal & heldout).sum())
        union = int((proposal | heldout).sum())
        confidence_mass = float(confidence.double().sum())
        true_confidence_mass = float(confidence[heldout].double().sum())
        hard_precision = _safe_ratio(intersection, proposal_pixels, empty=1.0)
        weighted_precision = _safe_ratio(
            true_confidence_mass, confidence_mass, empty=1.0
        )
        record = {
            "trial_index": index,
            "proposal_pixels": proposal_pixels,
            "heldout_positive_pixels": heldout_pixels,
            "intersection_pixels": intersection,
            "union_pixels": union,
            "proposal_confidence_mass": confidence_mass,
            "heldout_true_confidence_mass": true_confidence_mass,
            "hard_majority_precision": hard_precision,
            "confidence_weighted_precision": weighted_precision,
            "precision_gain_from_margin_weighting": (
                weighted_precision - hard_precision
            ),
            "hard_majority_recall": _safe_ratio(
                intersection, heldout_pixels, empty=1.0
            ),
            "hard_majority_iou": _safe_ratio(intersection, union, empty=1.0),
            "confidence_weighted_false_positive_fraction": (
                1.0 - weighted_precision if confidence_mass > 0 else 0.0
            ),
        }
        if not all(
            math.isfinite(float(value))
            for value in record.values()
            if isinstance(value, float)
        ):
            raise RuntimeError("source-only LOO diagnostic produced a nonfinite metric")
        records.append(record)
        pooled_proposal += proposal_pixels
        pooled_intersection += intersection
        pooled_heldout_positive += heldout_pixels
        pooled_union += union
        pooled_confidence += confidence_mass
        pooled_true_confidence += true_confidence_mass

    hard_precisions = [float(row["hard_majority_precision"]) for row in records]
    weighted_precisions = [
        float(row["confidence_weighted_precision"]) for row in records
    ]
    mean_hard = math.fsum(hard_precisions) / TRIAL_COUNT
    mean_weighted = math.fsum(weighted_precisions) / TRIAL_COUNT
    return {
        "method_contract": method_contract(),
        "full_fit": {
            "evaluable_non_scribble_pixels": evaluable_pixels,
            "proposal_pixels": int(full_proposal.sum()),
            "proposal_fraction": float(full_proposal.sum()) / evaluable_pixels,
            "proposal_confidence_mass": float(full_confidence.double().sum()),
            "confidence_mass_fraction": float(full_confidence.double().sum())
            / evaluable_pixels,
            "minimum_positive_proposal_reliability": (
                float(full_confidence[full_proposal].min())
                if bool(full_proposal.any())
                else None
            ),
            "maximum_proposal_reliability": (
                float(full_confidence[full_proposal].max())
                if bool(full_proposal.any())
                else None
            ),
            "nonproposal_completion_confidence_mass": float(
                full_confidence[(full_q <= MAJORITY_THRESHOLD) & evaluable]
                .double()
                .sum()
            ),
        },
        "per_trial": records,
        "summary": {
            "mean_loo_hard_majority_precision": mean_hard,
            "mean_loo_confidence_weighted_precision": mean_weighted,
            "mean_loo_precision_gain_from_margin_weighting": (
                mean_weighted - mean_hard
            ),
            "minimum_loo_confidence_weighted_precision": min(
                weighted_precisions
            ),
            "pooled_hard_majority_precision": _safe_ratio(
                pooled_intersection, pooled_proposal, empty=1.0
            ),
            "pooled_confidence_weighted_precision": _safe_ratio(
                pooled_true_confidence, pooled_confidence, empty=1.0
            ),
            "pooled_hard_majority_recall": _safe_ratio(
                pooled_intersection, pooled_heldout_positive, empty=1.0
            ),
            "pooled_hard_majority_iou": _safe_ratio(
                pooled_intersection, pooled_union, empty=1.0
            ),
            "pooled_proposal_pixels": pooled_proposal,
            "pooled_intersection_pixels": pooled_intersection,
            "pooled_confidence_mass": pooled_confidence,
            "pooled_true_confidence_mass": pooled_true_confidence,
        },
        "safety": {
            "scribble_pixels_excluded_from_loo_metrics": True,
            "absence_used_as_negative_evidence": False,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
        },
    }
