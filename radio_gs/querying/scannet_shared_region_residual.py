"""Class-shared categorical residual over query-independent SAM proposals."""

from __future__ import annotations

import torch


def class_count_aware_unknown_alpha(
    num_classes: int,
    *,
    minimum_class_count: int = 15,
    enabled_alpha: float = 1.0,
) -> float:
    """Return one class-count policy shared by every category and scene."""

    count = int(num_classes)
    minimum = int(minimum_class_count)
    strength = float(enabled_alpha)
    if count < 2 or minimum < 2:
        raise ValueError("class counts must be at least two")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("enabled_alpha must be in [0,1]")
    return strength if count >= minimum else 0.0


def shared_uncertainty_background_rejection(
    scores: torch.Tensor,
    *,
    alpha: float = 0.0,
    normalized_margin_threshold: float = 0.2,
    normalized_entropy_threshold: float = 0.85,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Append a class-symmetric background probability for uncertain rows.

    Confidence is measured after row-wise standardization, so the same two
    thresholds apply to different class-set sizes.  The returned last column
    is explicit background/abstention.  ``alpha=0`` cannot win and therefore
    exactly replays the primitive categorical decision.
    """

    values = torch.as_tensor(scores).detach().cpu().float()
    strength = float(alpha)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("scores must be [N,C] with C>=2")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if normalized_margin_threshold < 0 or not 0.0 <= normalized_entropy_threshold <= 1.0:
        raise ValueError("uncertainty thresholds are outside their domains")
    centered = values - values.mean(dim=-1, keepdim=True)
    standardized = centered / values.std(dim=-1, keepdim=True).clamp_min(1.0e-6)
    probability = torch.softmax(standardized, dim=-1)
    top2 = torch.topk(standardized, 2, dim=-1).values
    normalized_margin = top2[:, 0] - top2[:, 1]
    normalized_entropy = -(
        probability * probability.clamp_min(1.0e-12).log()
    ).sum(dim=-1) / torch.log(values.new_tensor(float(values.shape[1])))
    uncertain = (
        (normalized_margin <= float(normalized_margin_threshold))
        & (normalized_entropy >= float(normalized_entropy_threshold))
    )
    background = values.new_zeros((values.shape[0], 1))
    background[uncertain] = strength
    foreground = probability * (1.0 - background)
    posterior = torch.cat((foreground, background), dim=-1)
    prediction = posterior.argmax(dim=-1)
    if strength == 0.0 and not torch.equal(prediction, values.argmax(dim=-1)):
        raise RuntimeError("zero rejection strength changed the primitive decision")
    return posterior, {
        "enabled": bool(strength > 0 and uncertain.any()),
        "alpha": strength,
        "uncertain_rows": int(uncertain.sum()),
        "rejected_rows": int((prediction == values.shape[1]).sum()),
        "normalized_margin_threshold": float(normalized_margin_threshold),
        "normalized_entropy_threshold": float(normalized_entropy_threshold),
    }


def shared_proposal_consensus_residual(
    scores: torch.Tensor,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    membership_weights: torch.Tensor,
    *,
    num_proposals: int,
    alpha: float = 0.0,
    row_margin_threshold: float = 0.04,
    proposal_margin_scale: float = 0.02,
    minimum_row_mass: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Pool class probabilities over proposals and return a bounded residual.

    The operator has no class-indexed parameter.  Permuting class columns or
    proposal identifiers therefore commutes with it.  High-margin primitive
    rows are immutable and ``alpha=0`` is a bitwise identity, making the
    residual safe to initialize at Primitive Readout-v0.
    """

    values = torch.as_tensor(scores).detach().cpu().float()
    rows = torch.as_tensor(row_indices).detach().cpu().long().reshape(-1)
    proposals = torch.as_tensor(proposal_indices).detach().cpu().long().reshape(-1)
    weights = torch.as_tensor(membership_weights).detach().cpu().float().reshape(-1)
    proposal_count = int(num_proposals)
    strength = float(alpha)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("scores must be [N,C] with C>=2")
    if not (rows.shape == proposals.shape == weights.shape):
        raise ValueError("sparse proposal membership axes differ")
    if proposal_count <= 0:
        raise ValueError("num_proposals must be positive")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if min(row_margin_threshold, proposal_margin_scale, minimum_row_mass) < 0:
        raise ValueError("margin scales and row mass must be non-negative")
    if strength == 0.0:
        return values.clone(), {
            "enabled": False,
            "alpha": 0.0,
            "changed_rows": 0,
            "eligible_rows": 0,
            "supported_rows": 0,
        }

    count, classes = values.shape
    valid_edges = (
        (rows >= 0)
        & (rows < count)
        & (proposals >= 0)
        & (proposals < proposal_count)
        & torch.isfinite(weights)
        & (weights > 0)
    )
    rows, proposals, weights = (
        rows[valid_edges],
        proposals[valid_edges],
        weights[valid_edges],
    )
    if rows.numel() == 0:
        return values.clone(), {
            "enabled": False,
            "alpha": strength,
            "changed_rows": 0,
            "eligible_rows": 0,
            "supported_rows": 0,
        }

    primitive_probability = torch.softmax(values, dim=-1)
    proposal_sum = values.new_zeros((proposal_count, classes))
    proposal_mass = values.new_zeros((proposal_count,))
    proposal_sum.index_add_(
        0, proposals, primitive_probability[rows] * weights[:, None]
    )
    proposal_mass.index_add_(0, proposals, weights)
    proposal_probability = proposal_sum / proposal_mass.clamp_min(1.0e-8)[:, None]
    proposal_top2 = torch.topk(proposal_probability, 2, dim=-1).values
    proposal_margin = proposal_top2[:, 0] - proposal_top2[:, 1]
    if float(proposal_margin_scale) == 0.0:
        proposal_precision = torch.ones_like(proposal_margin)
    else:
        proposal_precision = (
            proposal_margin / float(proposal_margin_scale)
        ).clamp(0.0, 1.0)

    edge_precision = weights * proposal_precision[proposals]
    row_sum = torch.zeros_like(primitive_probability)
    row_mass = values.new_zeros((count,))
    row_sum.index_add_(
        0, rows, proposal_probability[proposals] * edge_precision[:, None]
    )
    row_mass.index_add_(0, rows, edge_precision)
    region_probability = row_sum / row_mass.clamp_min(1.0e-8)[:, None]

    primitive_top2 = torch.topk(values, 2, dim=-1).values
    primitive_margin = primitive_top2[:, 0] - primitive_top2[:, 1]
    supported = row_mass >= float(minimum_row_mass)
    eligible = supported & (primitive_margin <= float(row_margin_threshold))
    output_probability = primitive_probability.clone()
    output_probability[eligible] = (
        (1.0 - strength) * primitive_probability[eligible]
        + strength * region_probability[eligible]
    )
    # Log-probabilities preserve the categorical decision while exposing a
    # normalized posterior to downstream proper-score diagnostics.
    output = output_probability.clamp_min(1.0e-12).log()
    changed = output.argmax(dim=-1) != values.argmax(dim=-1)
    return output, {
        "enabled": bool(eligible.any()),
        "alpha": strength,
        "changed_rows": int(changed.sum()),
        "eligible_rows": int(eligible.sum()),
        "supported_rows": int(supported.sum()),
        "proposal_count": proposal_count,
        "membership_count": int(rows.numel()),
    }


__all__ = [
    "class_count_aware_unknown_alpha",
    "shared_proposal_consensus_residual",
    "shared_uncertainty_background_rejection",
]
