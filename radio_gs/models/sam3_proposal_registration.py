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


def fuse_scores_with_seeded_sam3_extent(
    scores: torch.Tensor,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    weights: torch.Tensor,
    proposal_view_indices: torch.Tensor,
    *,
    proposal_query_indices: torch.Tensor | None = None,
    alpha: float = 1.0,
    proposal_mean_ratio: float = 0.50,
    seed_support_ratio: float = 0.80,
    minimum_views: int = 2,
    query_conditioned: bool = False,
    min_weight_sum: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Use text peaks for identity and source-SAM proposals only for extent.

    For every text query, the highest-scoring primitive is the immutable
    identity seed.  In each source view, at most one proposal containing that
    seed is selected by its mean primitive text score.  The union of those
    proposals becomes a soft extent map.  No proposal is allowed to move the
    seed or create an object when fewer than ``minimum_views`` support it.
    """

    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [N,K], got {tuple(scores.shape)}")
    for name, value in (
        ("row_indices", row_indices),
        ("proposal_indices", proposal_indices),
        ("weights", weights),
    ):
        if value.ndim != 1:
            raise ValueError(f"{name} must be 1D")
    if row_indices.shape != proposal_indices.shape or row_indices.shape != weights.shape:
        raise ValueError("sparse membership tensors must have matching shapes")
    if proposal_view_indices.ndim != 1:
        raise ValueError("proposal_view_indices must be 1D")
    if query_conditioned and proposal_query_indices is None:
        raise ValueError("query-conditioned extent requires proposal_query_indices")
    if proposal_query_indices is not None and proposal_query_indices.shape != proposal_view_indices.shape:
        raise ValueError("proposal query/view vectors must have matching shapes")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if not 0.0 <= float(proposal_mean_ratio) <= 1.0:
        raise ValueError("proposal_mean_ratio must be in [0,1]")
    if not 0.0 < float(seed_support_ratio) <= 1.0:
        raise ValueError("seed_support_ratio must be in (0,1]")
    if int(minimum_views) <= 0:
        raise ValueError("minimum_views must be positive")

    num_rows, num_queries = int(scores.shape[0]), int(scores.shape[1])
    num_proposals = int(proposal_view_indices.numel())
    stats: dict[str, object] = {
        "enabled": False,
        "mode": "seeded_exact_mpr_extent",
        "alpha": float(alpha),
        "proposal_mean_ratio": float(proposal_mean_ratio),
        "seed_support_ratio": float(seed_support_ratio),
        "minimum_views": int(minimum_views),
        "query_conditioned": bool(query_conditioned),
        "num_memberships": int(row_indices.numel()),
        "num_proposals": num_proposals,
        "num_queries": num_queries,
        "num_queries_with_extent": 0,
        "selected_proposals_per_query": [],
        "selected_views_per_query": [],
    }
    if (
        float(alpha) <= 0.0
        or num_rows == 0
        or num_queries == 0
        or num_proposals == 0
        or row_indices.numel() == 0
    ):
        return scores, stats

    device = scores.device
    rows = row_indices.to(device=device, dtype=torch.long)
    props = proposal_indices.to(device=device, dtype=torch.long)
    membership_weights = weights.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    views = proposal_view_indices.to(device=device, dtype=torch.long)
    proposal_queries = (
        proposal_query_indices.to(device=device, dtype=torch.long)
        if proposal_query_indices is not None
        else None
    )
    valid = (
        (rows >= 0)
        & (rows < num_rows)
        & (props >= 0)
        & (props < num_proposals)
        & torch.isfinite(membership_weights)
        & (membership_weights > 0)
    )
    if not bool(valid.any()):
        return scores, stats
    rows = rows[valid]
    props = props[valid]
    membership_weights = membership_weights[valid]

    proposal_weight_sums = scores.new_zeros((num_proposals,), dtype=torch.float32)
    proposal_score_sums = scores.new_zeros((num_proposals, num_queries), dtype=torch.float32)
    proposal_weight_sums.index_add_(0, props, membership_weights)
    proposal_score_sums.index_add_(
        0,
        props,
        scores.float()[rows] * membership_weights[:, None],
    )
    proposal_means = proposal_score_sums / proposal_weight_sums.clamp_min(
        float(min_weight_sum)
    )[:, None]
    seeds = torch.argmax(scores.float(), dim=0)
    peaks = scores.float()[seeds, torch.arange(num_queries, device=device)]
    fused = scores.clone()
    selected_counts: list[int] = []
    selected_view_counts: list[int] = []
    support_proposal_counts: list[int] = []
    semantic_proposal_counts: list[int] = []
    support_view_counts: list[int] = []
    semantic_view_counts: list[int] = []
    maximum_proposal_mean_ratios: list[float | None] = []
    queries_with_extent = 0

    for query_index in range(num_queries):
        seed = seeds[query_index]
        seed_support = scores.float()[:, query_index] >= (
            peaks[query_index] * float(seed_support_ratio)
        )
        support_pair_weights = membership_weights * seed_support[rows].float()
        proposal_support_sums = scores.new_zeros((num_proposals,), dtype=torch.float32)
        proposal_support_sums.index_add_(0, props, support_pair_weights)
        proposal_support_fraction = proposal_support_sums / proposal_weight_sums.clamp_min(
            float(min_weight_sum)
        )
        exact_seed_sums = scores.new_zeros((num_proposals,), dtype=torch.float32)
        exact_seed_pairs = rows == seed
        if bool(exact_seed_pairs.any()):
            exact_seed_sums.index_add_(
                0,
                props[exact_seed_pairs],
                membership_weights[exact_seed_pairs],
            )
        seed_props = torch.nonzero(
            proposal_support_fraction > 0,
            as_tuple=False,
        ).flatten()
        support_proposal_counts.append(int(seed_props.numel()))
        support_view_counts.append(
            int(torch.unique(views[seed_props]).numel()) if seed_props.numel() else 0
        )
        if seed_props.numel() == 0:
            semantic_proposal_counts.append(0)
            semantic_view_counts.append(0)
            maximum_proposal_mean_ratios.append(None)
            selected_counts.append(0)
            selected_view_counts.append(0)
            continue
        seed_weights = proposal_support_fraction[seed_props]
        candidate_scores = proposal_means[seed_props, query_index]
        peak_denominator = peaks[query_index].abs().clamp_min(float(min_weight_sum))
        maximum_proposal_mean_ratios.append(
            float((candidate_scores.max() / peak_denominator).item())
        )
        candidate_keep = candidate_scores >= (
            peaks[query_index] * float(proposal_mean_ratio)
        )
        if query_conditioned:
            assert proposal_queries is not None
            candidate_keep &= proposal_queries[seed_props] == query_index
        seed_props = seed_props[candidate_keep]
        seed_weights = seed_weights[candidate_keep]
        candidate_scores = candidate_scores[candidate_keep]
        semantic_proposal_counts.append(int(seed_props.numel()))
        semantic_view_counts.append(
            int(torch.unique(views[seed_props]).numel()) if seed_props.numel() else 0
        )
        if seed_props.numel() == 0:
            selected_counts.append(0)
            selected_view_counts.append(0)
            continue

        selected: list[torch.Tensor] = []
        candidate_views = views[seed_props]
        for view_index in torch.unique(candidate_views, sorted=True):
            in_view = candidate_views == view_index
            view_props = seed_props[in_view]
            exact_in_view = exact_seed_sums[view_props] > 0
            if bool(exact_in_view.any()):
                in_view_indices = torch.nonzero(in_view, as_tuple=False).flatten()
                keep_indices = in_view_indices[exact_in_view]
                local_quality = (
                    candidate_scores[keep_indices]
                    * seed_weights[keep_indices]
                    * exact_seed_sums[seed_props[keep_indices]]
                )
                local_props = seed_props[keep_indices]
                selected.append(local_props[torch.argmax(local_quality)])
                continue
            local_quality = candidate_scores[in_view] * seed_weights[in_view]
            local_props = view_props
            selected.append(local_props[torch.argmax(local_quality)])
        selected_props = torch.stack(selected)
        selected_view_count = int(selected_props.numel())
        if selected_view_count < int(minimum_views):
            selected_counts.append(0)
            selected_view_counts.append(selected_view_count)
            continue

        extent = scores.new_zeros((num_rows,), dtype=torch.float32)
        for proposal_index in selected_props:
            in_proposal = props == proposal_index
            proposal_rows = rows[in_proposal]
            proposal_weights = membership_weights[in_proposal]
            # Proposal ids are unique per view, so a short explicit maximum is
            # deterministic and avoids a dense proposal-by-row allocation.
            extent[proposal_rows] = torch.maximum(extent[proposal_rows], proposal_weights)
        if not bool((extent > 0).any()):
            selected_counts.append(0)
            selected_view_counts.append(selected_view_count)
            continue
        extent[seed] = 1.0
        proposal_extent_scores = peaks[query_index] * extent
        fused[:, query_index] = (
            (1.0 - float(alpha)) * scores[:, query_index]
            + float(alpha) * proposal_extent_scores.to(dtype=scores.dtype)
        )
        # Identity is invariant by construction, even with low-confidence SAM
        # memberships or a partial source-view proposal hierarchy.
        fused[seed, query_index] = scores[seed, query_index]
        selected_counts.append(int(selected_props.numel()))
        selected_view_counts.append(selected_view_count)
        queries_with_extent += 1

    stats.update(
        {
            "enabled": queries_with_extent > 0,
            "num_memberships": int(valid.sum().item()),
            "num_queries_with_extent": queries_with_extent,
            "selected_proposals_per_query": selected_counts,
            "selected_views_per_query": selected_view_counts,
            "support_proposals_per_query": support_proposal_counts,
            "semantic_proposals_per_query": semantic_proposal_counts,
            "support_views_per_query": support_view_counts,
            "semantic_views_per_query": semantic_view_counts,
            "maximum_proposal_mean_ratio_per_query": maximum_proposal_mean_ratios,
            "mean_selected_views": (
                float(sum(selected_view_counts)) / len(selected_view_counts)
                if selected_view_counts
                else 0.0
            ),
            "seed_support_ratio": float(seed_support_ratio),
        }
    )
    return fused, stats
