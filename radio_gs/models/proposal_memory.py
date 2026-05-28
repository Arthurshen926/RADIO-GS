"""Label-free proposal-memory utilities for object-aware readouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


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


def compute_region_prototype_contrast_loss(
    values: torch.Tensor,
    proposal_labels: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    min_count: int = 1,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Contrast rows against label-free region prototypes.

    Rows inside the same proposal use their pooled prototype as the positive;
    other proposal prototypes are negatives. This turns proposal memory from a
    readout smoother into a training-time feature-topology objective.
    """

    _validate_values_and_labels(values, proposal_labels)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    zero = values.sum() * 0.0
    memory = build_proposal_memory_from_labels(
        values,
        proposal_labels,
        confidence=confidence,
    )
    num_rows = values.shape[0]
    num_proposals = int(memory.proposal_ids.numel())
    assigned = memory.row_to_proposal >= 0
    if int(min_count) > 1 and bool(assigned.any()):
        assigned_rows = memory.row_to_proposal[assigned]
        keep = memory.counts[assigned_rows] >= int(min_count)
        full = torch.zeros_like(assigned)
        full[assigned] = keep
        assigned = full
    valid_ratio = assigned.float().mean() if num_rows else values.new_tensor(0.0)
    if num_proposals < 2 or not bool(assigned.any()):
        active_valid_ratio = values.new_tensor(0.0)
        return zero, {
            "valid_ratio": active_valid_ratio.detach(),
            "num_proposals": values.new_tensor(float(num_proposals)),
            "num_valid": values.new_tensor(float(int(assigned.sum().item()))),
        }

    row_values = F.normalize(values[assigned].float(), dim=-1)
    prototypes = F.normalize(memory.pooled_values.float().detach(), dim=-1)
    logits = row_values @ prototypes.T / float(temperature)
    targets = memory.row_to_proposal[assigned].to(device=logits.device, dtype=torch.long)
    per_row = F.cross_entropy(logits, targets, reduction="none")
    if confidence is not None:
        weights = confidence.to(device=values.device, dtype=per_row.dtype).reshape(-1)[assigned]
        loss = (per_row * weights).sum() / weights.sum().clamp_min(1e-6)
    else:
        loss = per_row.mean()
    return loss, {
        "valid_ratio": valid_ratio.detach(),
        "num_proposals": values.new_tensor(float(num_proposals)),
        "num_valid": values.new_tensor(float(int(assigned.sum().item()))),
    }


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
    proposal_consensus_threshold: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Blend point/primitive logits with logits pooled over proposals."""

    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [N,C], got {tuple(logits.shape)}")
    if gate not in {
        "all",
        "low_margin",
        "low_confidence",
        "low_margin_or_low_confidence",
        "proposal_consensus",
        "low_margin_and_proposal_consensus",
        "low_confidence_and_proposal_consensus",
    }:
        raise ValueError(
            "gate must be one of: all, low_margin, low_confidence, "
            "low_margin_or_low_confidence, proposal_consensus, "
            "low_margin_and_proposal_consensus, low_confidence_and_proposal_consensus"
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
            "proposal_consensus_threshold": float(proposal_consensus_threshold),
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
            "proposal_consensus_threshold": float(proposal_consensus_threshold),
        }

    mapped = logits.clone()
    assigned = memory.row_to_proposal >= 0
    if min_count > 1:
        valid_prop = torch.zeros_like(assigned)
        valid_indices = memory.row_to_proposal[assigned]
        valid_prop[assigned] = memory.counts[valid_indices] >= int(min_count)
        assigned = valid_prop
    if gate in {
        "low_margin",
        "low_margin_or_low_confidence",
        "low_margin_and_proposal_consensus",
    }:
        if logits.shape[1] <= 1:
            margins = logits[:, 0].abs()
        else:
            top2 = torch.topk(logits.float(), k=2, dim=-1).values
            margins = top2[:, 0] - top2[:, 1]
        low_margin = margins <= float(margin_threshold)
    else:
        low_margin = torch.zeros_like(assigned)
    if gate in {
        "low_confidence",
        "low_margin_or_low_confidence",
        "low_confidence_and_proposal_consensus",
    }:
        if confidence is not None:
            row_confidence = confidence.to(device=logits.device, dtype=torch.float32).reshape(-1)
        elif logits.shape[1] <= 1:
            row_confidence = torch.sigmoid(logits.float().reshape(-1))
        else:
            row_confidence = torch.softmax(logits.float(), dim=-1).max(dim=-1).values
        low_confidence = row_confidence <= float(confidence_threshold)
    else:
        low_confidence = torch.zeros_like(assigned)
    if "proposal_consensus" in gate:
        if logits.shape[1] <= 1:
            row_top1 = (logits.float().reshape(-1) >= 0).long()
            proposal_top1 = (memory.pooled_values.float().reshape(-1) >= 0).long()
        else:
            row_top1 = logits.float().argmax(dim=-1)
            proposal_top1 = memory.pooled_values.float().argmax(dim=-1)
        proposal_consensus = logits.new_zeros((memory.proposal_ids.numel(),), dtype=torch.float32)
        assigned_rows = memory.row_to_proposal >= 0
        assigned_prop = memory.row_to_proposal[assigned_rows]
        row_agrees = (
            row_top1[assigned_rows]
            == proposal_top1[assigned_prop].to(device=row_top1.device)
        ).to(dtype=torch.float32)
        proposal_consensus.index_add_(0, assigned_prop, row_agrees)
        proposal_consensus = proposal_consensus / memory.counts.to(
            device=proposal_consensus.device,
            dtype=proposal_consensus.dtype,
        ).clamp_min(1)
        high_consensus = torch.zeros_like(assigned)
        high_consensus[assigned_rows] = (
            proposal_consensus[assigned_prop] >= float(proposal_consensus_threshold)
        )
        num_consensus_proposals = int(
            (proposal_consensus >= float(proposal_consensus_threshold)).sum().item()
        )
    else:
        high_consensus = torch.zeros_like(assigned)
        num_consensus_proposals = 0
    if gate == "low_margin":
        assigned &= low_margin
    elif gate == "low_confidence":
        assigned &= low_confidence
    elif gate == "low_margin_or_low_confidence":
        assigned &= low_margin | low_confidence
    elif gate == "proposal_consensus":
        assigned &= high_consensus
    elif gate == "low_margin_and_proposal_consensus":
        assigned &= low_margin & high_consensus
    elif gate == "low_confidence_and_proposal_consensus":
        assigned &= low_confidence & high_consensus
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
        "proposal_consensus_threshold": float(proposal_consensus_threshold),
        "num_consensus_proposals": int(num_consensus_proposals),
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
