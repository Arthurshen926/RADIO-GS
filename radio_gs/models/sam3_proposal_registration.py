"""SAM3 training-view proposal registration utilities.

The helpers in this module are label-free: SAM3 masks are treated as object
proposals observed in training views, and no LERF query masks are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class Sam3ProposalMemberships:
    """Soft memberships between rows and SAM3 object proposals."""

    row_indices: torch.Tensor
    proposal_indices: torch.Tensor
    weights: torch.Tensor
    num_rows: int
    num_proposals: int
    proposal_query_indices: torch.Tensor | None = None


def _empty_memberships(
    *,
    num_rows: int,
    num_proposals: int,
    device: torch.device | None = None,
) -> Sam3ProposalMemberships:
    dev = device or torch.device("cpu")
    return Sam3ProposalMemberships(
        row_indices=torch.empty(0, dtype=torch.long, device=dev),
        proposal_indices=torch.empty(0, dtype=torch.long, device=dev),
        weights=torch.empty(0, dtype=torch.float32, device=dev),
        num_rows=int(num_rows),
        num_proposals=int(num_proposals),
        proposal_query_indices=None,
    )


def build_sam3_mask_memberships(
    mask_logits: torch.Tensor,
    pixels_xy: torch.Tensor,
    *,
    scores: torch.Tensor | None = None,
    mask_query_indices: torch.Tensor | None = None,
    visibility: torch.Tensor | None = None,
    min_probability: float = 0.5,
    max_masks: int | None = None,
    proposal_offset: int = 0,
) -> Sam3ProposalMemberships:
    """Sample SAM3 masks at projected row pixels and return confident pairs.

    Args:
        mask_logits: SAM3 proposal logits with shape ``[M,H,W]``.
        pixels_xy: Row projections in the same pixel grid, shape ``[N,2]``.
        scores: Optional per-mask confidence, shape ``[M]``.
        mask_query_indices: Optional per-mask text-query ids, shape ``[M]``.
        visibility: Optional per-row visibility confidence, shape ``[N]``.
        min_probability: Minimum sigmoid(mask_logit) required for membership.
        max_masks: Keep only the highest-confidence masks before sampling.
        proposal_offset: Added to returned proposal ids when concatenating views.
    """

    if mask_logits.ndim != 3:
        raise ValueError(f"mask_logits must have shape [M,H,W], got {tuple(mask_logits.shape)}")
    if pixels_xy.ndim != 2 or pixels_xy.shape[1] != 2:
        raise ValueError(f"pixels_xy must have shape [N,2], got {tuple(pixels_xy.shape)}")
    num_rows = int(pixels_xy.shape[0])
    num_masks = int(mask_logits.shape[0])
    device = mask_logits.device
    if num_rows == 0 or num_masks == 0:
        return _empty_memberships(
            num_rows=num_rows,
            num_proposals=num_masks,
            device=device,
        )

    logits = mask_logits.float()
    mask_scores: torch.Tensor | None = None
    if scores is not None:
        if scores.ndim != 1 or scores.shape[0] != num_masks:
            raise ValueError(
                f"scores must have shape [M] aligned with mask_logits; got {tuple(scores.shape)}"
            )
        mask_scores = scores.to(device=device, dtype=torch.float32).clamp_min(0.0)
    query_indices: torch.Tensor | None = None
    if mask_query_indices is not None:
        if mask_query_indices.ndim != 1 or mask_query_indices.shape[0] != num_masks:
            raise ValueError(
                "mask_query_indices must have shape [M] aligned with mask_logits; got "
                f"{tuple(mask_query_indices.shape)}"
            )
        query_indices = mask_query_indices.to(device=device, dtype=torch.long)
    if max_masks is not None and int(max_masks) > 0 and int(max_masks) < num_masks:
        if mask_scores is None:
            keep = torch.arange(int(max_masks), device=device)
        else:
            keep = torch.topk(mask_scores, k=int(max_masks), largest=True).indices
        logits = logits[keep]
        if mask_scores is not None:
            mask_scores = mask_scores[keep]
        if query_indices is not None:
            query_indices = query_indices[keep]
        proposal_ids = (
            torch.arange(int(keep.numel()), dtype=torch.long, device=device)
            + int(proposal_offset)
        )
    else:
        proposal_ids = (
            torch.arange(num_masks, dtype=torch.long, device=device) + int(proposal_offset)
        )

    if visibility is not None:
        if visibility.ndim != 1 or visibility.shape[0] != num_rows:
            raise ValueError(
                "visibility must have shape [N] aligned with pixels_xy; got "
                f"{tuple(visibility.shape)}"
            )
        row_visibility = visibility.to(device=device, dtype=torch.float32).clamp_min(0.0)
    else:
        row_visibility = torch.ones(num_rows, dtype=torch.float32, device=device)

    height, width = int(logits.shape[1]), int(logits.shape[2])
    xy = pixels_xy.to(device=device, dtype=torch.float32)
    valid_xy = (
        torch.isfinite(xy).all(dim=-1)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] <= height - 1)
        & (row_visibility > 0)
    )
    if not bool(valid_xy.any()):
        return _empty_memberships(
            num_rows=num_rows,
            num_proposals=int(proposal_ids.numel()),
            device=device,
        )

    rows = torch.nonzero(valid_xy, as_tuple=False).flatten()
    x = xy[rows, 0].round().long().clamp_(0, width - 1)
    y = xy[rows, 1].round().long().clamp_(0, height - 1)
    sampled = torch.sigmoid(logits[:, y, x]).transpose(0, 1)
    weights = sampled * row_visibility[rows, None]
    if mask_scores is not None:
        weights = weights * mask_scores[None, :]
    keep_pairs = weights >= float(min_probability)
    if not bool(keep_pairs.any()):
        return _empty_memberships(
            num_rows=num_rows,
            num_proposals=int(proposal_ids.numel()),
            device=device,
        )

    local_row, local_prop = torch.nonzero(keep_pairs, as_tuple=True)
    return Sam3ProposalMemberships(
        row_indices=rows[local_row].to(dtype=torch.long),
        proposal_indices=proposal_ids[local_prop].to(dtype=torch.long),
        weights=weights[local_row, local_prop].to(dtype=torch.float32),
        num_rows=num_rows,
        num_proposals=int(proposal_ids.numel()),
        proposal_query_indices=query_indices.detach().clone() if query_indices is not None else None,
    )


def fuse_scores_with_sam3_proposals(
    scores: torch.Tensor,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    weights: torch.Tensor,
    *,
    alpha: float,
    gate: Literal["all", "low_margin"] = "low_margin",
    margin_threshold: float = 0.05,
    min_weight_sum: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float | int | bool | str]]:
    """Blend primitive scores with scores pooled over SAM3 proposals."""

    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [N,K], got {tuple(scores.shape)}")
    if gate not in {"all", "low_margin"}:
        raise ValueError("gate must be one of: all, low_margin")
    if row_indices.ndim != 1 or proposal_indices.ndim != 1 or weights.ndim != 1:
        raise ValueError("row_indices, proposal_indices, and weights must be 1D")
    if row_indices.shape != proposal_indices.shape or row_indices.shape != weights.shape:
        raise ValueError("row_indices, proposal_indices, and weights must have matching shapes")
    alpha_f = float(alpha)
    if not 0.0 <= alpha_f <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    base_stats: dict[str, float | int | bool | str] = {
        "enabled": False,
        "mode": "sam3_trainview",
        "alpha": alpha_f,
        "gate": gate,
        "margin_threshold": float(margin_threshold),
        "num_memberships": int(row_indices.numel()),
        "num_proposals": 0,
        "num_assigned": 0,
    }
    if alpha_f <= 0.0 or scores.shape[0] == 0 or row_indices.numel() == 0:
        return scores, base_stats

    device = scores.device
    rows = row_indices.to(device=device, dtype=torch.long)
    props_raw = proposal_indices.to(device=device, dtype=torch.long)
    w = weights.to(device=device, dtype=torch.float32).clamp_min(0.0)
    valid = (
        (rows >= 0)
        & (rows < scores.shape[0])
        & (props_raw >= 0)
        & (w > 0)
    )
    if not bool(valid.any()):
        return scores, base_stats

    rows = rows[valid]
    props_raw = props_raw[valid]
    w = w[valid]
    prop_ids, props = torch.unique(props_raw, sorted=True, return_inverse=True)
    num_props = int(prop_ids.numel())

    score_sums = scores.new_zeros((num_props, scores.shape[1]), dtype=torch.float32)
    weight_sums = scores.new_zeros((num_props,), dtype=torch.float32)
    score_sums.index_add_(0, props, scores.float()[rows] * w[:, None])
    weight_sums.index_add_(0, props, w)
    pooled_scores = score_sums / weight_sums.clamp_min(float(min_weight_sum))[:, None]

    row_score_sums = scores.new_zeros(scores.shape, dtype=torch.float32)
    row_weight_sums = scores.new_zeros((scores.shape[0],), dtype=torch.float32)
    row_score_sums.index_add_(0, rows, pooled_scores[props] * w[:, None])
    row_weight_sums.index_add_(0, rows, w)
    assigned = row_weight_sums > float(min_weight_sum)

    if gate == "low_margin":
        if scores.shape[1] <= 1:
            margins = scores.float()[:, 0].abs()
        else:
            top2 = torch.topk(scores.float(), k=2, dim=-1).values
            margins = top2[:, 0] - top2[:, 1]
        assigned = assigned & (margins <= float(margin_threshold))

    fused = scores.clone()
    if bool(assigned.any()):
        proposal_rows = row_score_sums[assigned] / row_weight_sums[assigned, None].clamp_min(
            float(min_weight_sum)
        )
        fused[assigned] = (1.0 - alpha_f) * scores[assigned] + alpha_f * proposal_rows.to(
            dtype=scores.dtype
        )

    stats = dict(base_stats)
    stats.update(
        {
            "enabled": bool(assigned.any()),
            "num_memberships": int(valid.sum().item()),
            "num_proposals": num_props,
            "num_assigned": int(assigned.sum().item()),
            "mean_membership_weight": float(w.mean().item()) if w.numel() else 0.0,
            "mean_proposal_weight_sum": float(weight_sums.mean().item()) if weight_sums.numel() else 0.0,
        }
    )
    return fused, stats


def fuse_scores_with_query_sam3_proposals(
    scores: torch.Tensor,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    weights: torch.Tensor,
    proposal_query_indices: torch.Tensor,
    *,
    alpha: float,
    gate: Literal["all", "low_margin"] = "low_margin",
    margin_threshold: float = 0.05,
    min_weight_sum: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float | int | bool | str]]:
    """Blend each query score only with SAM3 proposals generated by that query."""

    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [N,K], got {tuple(scores.shape)}")
    if gate not in {"all", "low_margin"}:
        raise ValueError("gate must be one of: all, low_margin")
    if row_indices.ndim != 1 or proposal_indices.ndim != 1 or weights.ndim != 1:
        raise ValueError("row_indices, proposal_indices, and weights must be 1D")
    if proposal_query_indices.ndim != 1:
        raise ValueError("proposal_query_indices must be 1D")
    if row_indices.shape != proposal_indices.shape or row_indices.shape != weights.shape:
        raise ValueError("row_indices, proposal_indices, and weights must have matching shapes")
    alpha_f = float(alpha)
    if not 0.0 <= alpha_f <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    stats: dict[str, float | int | bool | str] = {
        "enabled": False,
        "mode": "sam3_trainview",
        "query_conditioned": True,
        "alpha": alpha_f,
        "gate": gate,
        "margin_threshold": float(margin_threshold),
        "num_memberships": int(row_indices.numel()),
        "num_proposals": int(proposal_query_indices.numel()),
        "num_assigned": 0,
    }
    if alpha_f <= 0.0 or scores.shape[0] == 0 or row_indices.numel() == 0:
        return scores, stats

    device = scores.device
    rows = row_indices.to(device=device, dtype=torch.long)
    props = proposal_indices.to(device=device, dtype=torch.long)
    w = weights.to(device=device, dtype=torch.float32).clamp_min(0.0)
    prop_queries = proposal_query_indices.to(device=device, dtype=torch.long)
    valid = (
        (rows >= 0)
        & (rows < scores.shape[0])
        & (props >= 0)
        & (props < prop_queries.numel())
        & (w > 0)
    )
    if not bool(valid.any()):
        return scores, stats

    rows = rows[valid]
    props = props[valid]
    w = w[valid]
    queries = prop_queries[props]
    valid_query = (queries >= 0) & (queries < scores.shape[1])
    if not bool(valid_query.any()):
        return scores, stats
    rows = rows[valid_query]
    props = props[valid_query]
    w = w[valid_query]
    queries = queries[valid_query]

    num_props = int(prop_queries.numel())
    prop_score_sums = scores.new_zeros((num_props,), dtype=torch.float32)
    prop_weight_sums = scores.new_zeros((num_props,), dtype=torch.float32)
    prop_score_sums.index_add_(0, props, scores.float()[rows, queries] * w)
    prop_weight_sums.index_add_(0, props, w)
    prop_scores = prop_score_sums / prop_weight_sums.clamp_min(float(min_weight_sum))

    row_query_score_sums = scores.new_zeros(scores.shape, dtype=torch.float32)
    row_query_weight_sums = scores.new_zeros(scores.shape, dtype=torch.float32)
    row_query_score_sums.index_put_(
        (rows, queries),
        prop_scores[props] * w,
        accumulate=True,
    )
    row_query_weight_sums.index_put_((rows, queries), w, accumulate=True)
    assigned = row_query_weight_sums > float(min_weight_sum)

    if gate == "low_margin":
        if scores.shape[1] <= 1:
            margins = scores.float()[:, 0].abs()
        else:
            top2 = torch.topk(scores.float(), k=2, dim=-1).values
            margins = top2[:, 0] - top2[:, 1]
        assigned = assigned & (margins[:, None] <= float(margin_threshold))

    fused = scores.clone()
    if bool(assigned.any()):
        proposal_values = row_query_score_sums[assigned] / row_query_weight_sums[
            assigned
        ].clamp_min(float(min_weight_sum))
        fused[assigned] = (1.0 - alpha_f) * scores[assigned] + alpha_f * proposal_values.to(
            dtype=scores.dtype
        )

    stats.update(
        {
            "enabled": bool(assigned.any()),
            "num_memberships": int(valid_query.sum().item()),
            "num_proposals": num_props,
            "num_assigned": int(assigned.sum().item()),
            "mean_membership_weight": float(w.mean().item()) if w.numel() else 0.0,
            "mean_proposal_weight_sum": float(prop_weight_sums.mean().item())
            if prop_weight_sums.numel()
            else 0.0,
        }
    )
    return fused, stats
