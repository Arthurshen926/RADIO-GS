"""Label-free proposal-memory utilities for object-aware readouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProposalMemory:
    """Values pooled over non-negative proposal ids."""

    proposal_ids: torch.Tensor
    pooled_values: torch.Tensor
    counts: torch.Tensor
    weight_sums: torch.Tensor
    row_to_proposal: torch.Tensor


def _validate_values_and_labels(values: torch.Tensor, proposal_labels: torch.Tensor) -> None:
    if values.ndim != 2:
        raise ValueError(f"values must have shape [N,D], got {tuple(values.shape)}")
    if proposal_labels.ndim != 1:
        raise ValueError(
            f"proposal_labels must have shape [N], got {tuple(proposal_labels.shape)}"
        )
    if values.shape[0] != proposal_labels.shape[0]:
        raise ValueError(
            "values and proposal_labels must have the same number of rows: "
            f"{values.shape[0]} vs {proposal_labels.shape[0]}"
        )


def build_proposal_memory_from_labels(
    values: torch.Tensor,
    proposal_labels: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
) -> ProposalMemory:
    """Pool row values by integer proposal ids.

    Negative proposal ids are treated as unassigned and ignored.  Optional
    confidence weights are label-free reliability weights, not class labels.
    """

    _validate_values_and_labels(values, proposal_labels)
    device = values.device
    labels = proposal_labels.to(device=device, dtype=torch.long)
    valid = labels >= 0
    row_to_proposal = torch.full(
        (values.shape[0],),
        -1,
        dtype=torch.long,
        device=device,
    )
    if not bool(valid.any()):
        empty_values = values.new_empty((0, values.shape[1]))
        empty_ids = labels.new_empty((0,))
        empty_counts = labels.new_empty((0,))
        empty_weights = values.new_empty((0,))
        return ProposalMemory(
            proposal_ids=empty_ids,
            pooled_values=empty_values,
            counts=empty_counts,
            weight_sums=empty_weights,
            row_to_proposal=row_to_proposal,
        )

    valid_labels = labels[valid]
    proposal_ids, inverse = torch.unique(valid_labels, sorted=True, return_inverse=True)
    row_to_proposal[valid] = inverse
    num_proposals = int(proposal_ids.numel())

    valid_values = values[valid].float()
    if confidence is None:
        weights = valid_values.new_ones((valid_values.shape[0],))
    else:
        if confidence.ndim != 1 or confidence.shape[0] != values.shape[0]:
            raise ValueError(
                "confidence must have shape [N] aligned with values; got "
                f"{tuple(confidence.shape)} for N={values.shape[0]}"
            )
        weights = confidence.to(device=device, dtype=valid_values.dtype)[valid].clamp_min(0.0)

    pooled_sum = valid_values.new_zeros((num_proposals, valid_values.shape[1]))
    pooled_sum.index_add_(0, inverse, valid_values * weights[:, None])
    weight_sums = valid_values.new_zeros((num_proposals,))
    weight_sums.index_add_(0, inverse, weights)
    pooled_values = pooled_sum / weight_sums.clamp_min(1e-12)[:, None]

    counts = torch.bincount(inverse, minlength=num_proposals).to(device=device, dtype=torch.long)
    return ProposalMemory(
        proposal_ids=proposal_ids,
        pooled_values=pooled_values.to(dtype=values.dtype),
        counts=counts,
        weight_sums=weight_sums,
        row_to_proposal=row_to_proposal,
    )


def propagate_logits_with_proposals(
    logits: torch.Tensor,
    proposal_labels: torch.Tensor,
    *,
    alpha: float,
    confidence: torch.Tensor | None = None,
    min_count: int = 1,
    gate: str = "all",
    margin_threshold: float = 0.0,
    confidence_threshold: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Blend point/primitive logits with logits pooled over proposals."""

    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [N,C], got {tuple(logits.shape)}")
    if gate not in {"all", "low_margin", "low_confidence", "low_margin_or_low_confidence"}:
        raise ValueError(
            "gate must be one of: all, low_margin, low_confidence, "
            "low_margin_or_low_confidence"
        )
    alpha_f = float(alpha)
    if not 0.0 <= alpha_f <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if alpha_f <= 0.0 or logits.shape[0] == 0:
        return logits, {
            "enabled": False,
            "num_proposals": 0,
            "num_assigned": int((proposal_labels >= 0).sum().item())
            if proposal_labels.ndim == 1
            else 0,
            "alpha": alpha_f,
            "min_count": int(min_count),
            "gate": gate,
            "margin_threshold": float(margin_threshold),
            "confidence_threshold": float(confidence_threshold),
        }

    memory = build_proposal_memory_from_labels(
        logits,
        proposal_labels,
        confidence=confidence,
    )
    if memory.proposal_ids.numel() == 0:
        return logits, {
            "enabled": False,
            "num_proposals": 0,
            "num_assigned": 0,
            "alpha": alpha_f,
            "min_count": int(min_count),
            "gate": gate,
            "margin_threshold": float(margin_threshold),
            "confidence_threshold": float(confidence_threshold),
        }

    mapped = logits.clone()
    assigned = memory.row_to_proposal >= 0
    if min_count > 1:
        valid_prop = torch.zeros_like(assigned)
        valid_indices = memory.row_to_proposal[assigned]
        valid_prop[assigned] = memory.counts[valid_indices] >= int(min_count)
        assigned = valid_prop
    if gate in {"low_margin", "low_margin_or_low_confidence"}:
        if logits.shape[1] <= 1:
            margins = logits[:, 0].abs()
        else:
            top2 = torch.topk(logits.float(), k=2, dim=-1).values
            margins = top2[:, 0] - top2[:, 1]
        low_margin = margins <= float(margin_threshold)
    else:
        low_margin = torch.zeros_like(assigned)
    if gate in {"low_confidence", "low_margin_or_low_confidence"}:
        if confidence is not None:
            row_confidence = confidence.to(device=logits.device, dtype=torch.float32).reshape(-1)
        elif logits.shape[1] <= 1:
            row_confidence = torch.sigmoid(logits.float().reshape(-1))
        else:
            row_confidence = torch.softmax(logits.float(), dim=-1).max(dim=-1).values
        low_confidence = row_confidence <= float(confidence_threshold)
    else:
        low_confidence = torch.zeros_like(assigned)
    if gate == "low_margin":
        assigned &= low_margin
    elif gate == "low_confidence":
        assigned &= low_confidence
    elif gate == "low_margin_or_low_confidence":
        assigned &= low_margin | low_confidence
    if bool(assigned.any()):
        pooled_rows = memory.pooled_values[memory.row_to_proposal[assigned]]
        mapped[assigned] = (1.0 - alpha_f) * logits[assigned] + alpha_f * pooled_rows

    return mapped, {
        "enabled": bool(assigned.any()),
        "num_proposals": int(memory.proposal_ids.numel()),
        "num_assigned": int(assigned.sum().item()),
        "alpha": alpha_f,
        "min_count": int(min_count),
        "gate": gate,
        "margin_threshold": float(margin_threshold),
        "confidence_threshold": float(confidence_threshold),
    }


def build_voxel_proposal_labels(
    xyz: torch.Tensor,
    *,
    voxel_size: float,
) -> torch.Tensor:
    """Assign each point to a deterministic 3D voxel proposal id."""

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N,3], got {tuple(xyz.shape)}")
    voxel_size_f = float(voxel_size)
    if voxel_size_f <= 0.0:
        raise ValueError("voxel_size must be positive")
    if xyz.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=xyz.device)

    coords = torch.floor(xyz.float() / voxel_size_f).to(dtype=torch.long)
    _, inverse = torch.unique(coords, sorted=True, return_inverse=True, dim=0)
    return inverse.to(device=xyz.device, dtype=torch.long)
