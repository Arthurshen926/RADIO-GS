"""Identity-seeded, multi-view object-topology posterior.

Text relevance and source-view segmentation solve different conditional
problems.  Text determines which object is queried, while a proposal mask
describes membership conditional on that object.  This module therefore never
averages text scores over a proposal.  It uses the immutable text maximum to
select one query-matched proposal hierarchy node per source view, then composes
the selected nodes into an object-membership posterior.

The sparse cache weights include a proposal-wide SAM quality factor.  That
factor is useful for ranking proposals but must not shrink every pixel of an
accepted mask below the downstream Bayes threshold.  We consequently retain
the original weight for proposal selection and divide by the proposal maximum
only when interpreting within-proposal membership.
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch


def compile_view_exclusive_physical_tracks(
    edge_left: torch.Tensor,
    edge_right: torch.Tensor,
    edge_probability: torch.Tensor,
    proposal_view_indices: torch.Tensor,
    *,
    minimum_probability: float = 0.5,
    minimum_views: int = 2,
) -> torch.Tensor:
    """Compile a maximum-confidence forest with at most one node per view.

    Plain connected components percolate through scale-conflicting SAM nodes.
    A physical object cannot own two different proposal nodes in one registered
    view, so an edge is accepted only when its two components have disjoint
    view sets.  Sorting by calibrated same-object probability makes this a
    deterministic, view-partition-constrained maximum spanning forest.
    """
    if (
        edge_left.ndim != 1 or edge_right.shape != edge_left.shape
        or edge_probability.shape != edge_left.shape
        or proposal_view_indices.ndim != 1
    ):
        raise ValueError("physical-track edge/view domains differ")
    if not 0.0 <= float(minimum_probability) <= 1.0 or int(minimum_views) < 2:
        raise ValueError("physical-track probability/view contract differs")
    count = int(proposal_view_indices.numel())
    if edge_left.numel() and (
        int(torch.minimum(edge_left, edge_right).min()) < 0
        or int(torch.maximum(edge_left, edge_right).max()) >= count
    ):
        raise ValueError("physical-track edge index exceeds proposal domain")
    parent = list(range(count))
    component_views = [{int(proposal_view_indices[index])} for index in range(count)]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    order = torch.argsort(edge_probability.float(), descending=True, stable=True)
    for edge in order.tolist():
        if float(edge_probability[edge]) < float(minimum_probability):
            break
        left = find(int(edge_left[edge])); right = find(int(edge_right[edge]))
        if left == right or component_views[left] & component_views[right]:
            continue
        if len(component_views[left]) < len(component_views[right]):
            left, right = right, left
        parent[right] = left
        component_views[left].update(component_views[right])
    components: dict[int, list[int]] = {}
    for proposal in range(count):
        components.setdefault(find(proposal), []).append(proposal)
    tracks = torch.full((count,), -1, dtype=torch.long)
    accepted = [
        rows for rows in components.values()
        if len({int(proposal_view_indices[row]) for row in rows}) >= int(minimum_views)
    ]
    accepted.sort(key=lambda rows: (min(rows), tuple(rows)))
    for track, rows in enumerate(accepted):
        tracks[torch.tensor(rows, dtype=torch.long)] = track
    return tracks


def _scatter_amax(
    values: torch.Tensor,
    indices: torch.Tensor,
    size: int,
) -> torch.Tensor:
    result = values.new_zeros((int(size),))
    if values.numel():
        result.scatter_reduce_(0, indices, values, reduce="amax", include_self=True)
    return result


def identity_seeded_object_topology_posterior(
    scores: torch.Tensor,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    weights: torch.Tensor,
    proposal_view_indices: torch.Tensor,
    proposal_query_indices: torch.Tensor,
    *,
    proposal_track_indices: torch.Tensor | None = None,
    proposal_scores: torch.Tensor | None = None,
    seed_support_ratio: float = 0.80,
    identity_core_ratio: float = 0.80,
    minimum_object_views: int = 2,
    minimum_row_views: int = 1,
    extent_membership_floor: float = 0.50,
    sibling_exclusion_strength: float = 0.0,
    unknown_policy: Literal["preserve_text_prior", "negative_outside_topology"] = (
        "preserve_text_prior"
    ),
    membership_calibration: Literal["proposal_max", "pure_probability"] = (
        "proposal_max"
    ),
    use_proposal_quality: bool = False,
    min_weight_sum: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Factor text identity from source-SAM object extent.

    The returned values are calibrated to the same ``[0,1]`` decision domain
    as ``scores``.  For a query with source evidence in fewer than
    ``minimum_object_views`` the operation is the exact identity fallback.
    Otherwise the output is the union of:

    * the high-response text identity core, but only inside the selected
      object topology; and
    * proposal-normalized mask membership composed across views by noisy-or.

    Rows outside both sets are negative, not weak positives.  This prevents the
    disconnected full-scene text islands that a residual blend cannot remove.
    The text argmax is restored bit-for-bit after every topology operation.
    """

    if scores.ndim != 2:
        raise ValueError("scores must have shape [N,Q]")
    for name, value in (
        ("row_indices", row_indices),
        ("proposal_indices", proposal_indices),
        ("weights", weights),
    ):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    if row_indices.shape != proposal_indices.shape or row_indices.shape != weights.shape:
        raise ValueError("sparse membership vectors must have matching shapes")
    if proposal_view_indices.ndim != 1 or proposal_query_indices.ndim != 1:
        raise ValueError("proposal view/query indices must be one-dimensional")
    if proposal_view_indices.shape != proposal_query_indices.shape:
        raise ValueError("proposal view/query indices must have matching shapes")
    if proposal_scores is not None and proposal_scores.shape != proposal_view_indices.shape:
        raise ValueError("proposal scores must align with proposal rows")
    if proposal_track_indices is not None and proposal_track_indices.shape != proposal_view_indices.shape:
        raise ValueError("proposal tracks must align with proposal rows")
    if not 0.0 < float(seed_support_ratio) <= 1.0:
        raise ValueError("seed_support_ratio must be in (0,1]")
    if not 0.0 < float(identity_core_ratio) <= 1.0:
        raise ValueError("identity_core_ratio must be in (0,1]")
    if int(minimum_object_views) <= 0 or int(minimum_row_views) <= 0:
        raise ValueError("view minima must be positive")
    if not 0.0 <= float(extent_membership_floor) <= 1.0:
        raise ValueError("extent_membership_floor must be in [0,1]")
    if not 0.0 <= float(sibling_exclusion_strength) <= 1.0:
        raise ValueError("sibling_exclusion_strength must be in [0,1]")
    if unknown_policy not in {"preserve_text_prior", "negative_outside_topology"}:
        raise ValueError("unsupported unknown_policy")
    if membership_calibration not in {"proposal_max", "pure_probability"}:
        raise ValueError("unsupported membership_calibration")
    if use_proposal_quality and proposal_scores is None:
        raise ValueError("proposal quality was requested but is absent")

    num_rows, num_queries = (int(scores.shape[0]), int(scores.shape[1]))
    num_proposals = int(proposal_view_indices.numel())
    base = scores.float()
    stats: dict[str, object] = {
        "enabled": False,
        "mode": "identity_seeded_multiview_object_topology_v1",
        "capability_track": "resolved_after_proposal_query_contract_validation",
        "identity_authority": "immutable_per_query_text_argmax_and_high_response_core",
        "extent_authority": "query_matched_source_sam_exact_mpr_membership",
        "query_independent_mask_hierarchy": False,
        "proposal_selection": "one_hierarchy_node_per_view_without_proposal_mean_text",
        "membership_calibration": str(membership_calibration),
        "proposal_quality_role": (
            "proposal_ranking_and_multiview_observation_confidence"
            if use_proposal_quality
            else "unused"
        ),
        "unknown_policy": str(unknown_policy),
        "seed_support_ratio": float(seed_support_ratio),
        "identity_core_ratio": float(identity_core_ratio),
        "minimum_object_views": int(minimum_object_views),
        "minimum_row_views": int(minimum_row_views),
        "extent_membership_floor": float(extent_membership_floor),
        "sibling_exclusion_strength": float(sibling_exclusion_strength),
        "num_rows": num_rows,
        "num_queries": num_queries,
        "num_proposals": num_proposals,
        "num_memberships": int(row_indices.numel()),
    }
    if num_rows == 0 or num_queries == 0 or num_proposals == 0 or row_indices.numel() == 0:
        return scores, stats

    device = scores.device
    rows = row_indices.to(device=device, dtype=torch.long)
    props = proposal_indices.to(device=device, dtype=torch.long)
    membership = weights.to(device=device, dtype=torch.float32)
    views = proposal_view_indices.to(device=device, dtype=torch.long)
    prop_queries = proposal_query_indices.to(device=device, dtype=torch.long)
    prop_tracks = (
        proposal_track_indices.to(device=device, dtype=torch.long)
        if proposal_track_indices is not None else None
    )
    query_independent_hierarchy = bool((prop_queries < 0).all())
    stats["query_independent_mask_hierarchy"] = query_independent_hierarchy
    stats["capability_track"] = (
        "query_free_source_sam_exact_mpr_object_topology"
        if query_independent_hierarchy
        else "query_conditioned_source_sam_diagnostic_not_p0"
    )
    prop_quality = (
        proposal_scores.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        if proposal_scores is not None
        else torch.ones(num_proposals, device=device, dtype=torch.float32)
    )
    valid = (
        (rows >= 0)
        & (rows < num_rows)
        & (props >= 0)
        & (props < num_proposals)
        & torch.isfinite(membership)
        & (membership > 0)
    )
    if not bool(valid.any()):
        return scores, stats
    rows, props, membership = rows[valid], props[valid], membership[valid]

    proposal_weight_sum = base.new_zeros((num_proposals,))
    proposal_weight_sum.index_add_(0, props, membership)
    proposal_weight_max = _scatter_amax(membership, props, num_proposals)
    if membership_calibration == "proposal_max":
        conditional_membership = membership / proposal_weight_max[props].clamp_min(
            float(min_weight_sum)
        )
    else:
        conditional_membership = membership.clamp(0.0, 1.0)

    seeds = torch.argmax(base, dim=0)
    query_axis = torch.arange(num_queries, device=device)
    peaks = base[seeds, query_axis]
    selected_by_query: list[torch.Tensor | None] = []
    selected_view_counts: list[int] = []
    exact_seed_view_counts: list[int] = []
    selected_track_indices: list[int] = []

    # Stage 1: immutable identity seed selects a single hierarchy node in each
    # source view.  Weighted peak support, not proposal-wide mean relevance, is
    # the semantic statistic, so object size cannot dilute identity.
    for query_index in range(num_queries):
        query_scores = base[:, query_index]
        support = query_scores >= peaks[query_index] * float(seed_support_ratio)
        support_sum = base.new_zeros((num_proposals,))
        support_sum.index_add_(0, props, membership * support[rows].float())
        support_fraction = support_sum / proposal_weight_sum.clamp_min(float(min_weight_sum))

        pair_score = query_scores[rows]
        proposal_tail = _scatter_amax(pair_score, props, num_proposals)
        exact_seed = base.new_zeros((num_proposals,))
        exact_pairs = rows == seeds[query_index]
        if bool(exact_pairs.any()):
            exact_seed.index_add_(0, props[exact_pairs], membership[exact_pairs])

        candidates = torch.nonzero(
            ((prop_queries == query_index) | (prop_queries < 0)) & (support_sum > 0),
            as_tuple=False,
        ).flatten()
        selected_track = -1
        if prop_tracks is not None and candidates.numel():
            candidates = candidates[prop_tracks[candidates] >= 0]
            if candidates.numel():
                exact = exact_seed[candidates] > 0
                anchor_candidates = candidates[exact] if bool(exact.any()) else candidates
                anchor_quality = (
                    proposal_tail[anchor_candidates].clamp_min(0)
                    * support_fraction[anchor_candidates].clamp_min(float(min_weight_sum)).sqrt()
                )
                if bool(exact.any()):
                    anchor_quality = anchor_quality * exact_seed[anchor_candidates].clamp_min(
                        float(min_weight_sum)
                    )
                anchor = anchor_candidates[torch.argmax(anchor_quality)]
                selected_track = int(prop_tracks[anchor])
                candidates = candidates[prop_tracks[candidates] == selected_track]
        if candidates.numel() == 0:
            selected_by_query.append(None)
            selected_view_counts.append(0)
            exact_seed_view_counts.append(0)
            selected_track_indices.append(-1)
            continue
        chosen: list[torch.Tensor] = []
        exact_views = 0
        for view in torch.unique(views[candidates], sorted=True):
            in_view = candidates[views[candidates] == view]
            exact = exact_seed[in_view] > 0
            if bool(exact.any()):
                in_view = in_view[exact]
                exact_views += 1
                quality = (
                    exact_seed[in_view]
                    * proposal_tail[in_view].clamp_min(0)
                    * support_fraction[in_view].clamp_min(float(min_weight_sum)).sqrt()
                )
            else:
                quality = (
                    proposal_tail[in_view].clamp_min(0)
                    * support_fraction[in_view].clamp_min(float(min_weight_sum)).sqrt()
                )
            if use_proposal_quality:
                quality = quality * prop_quality[in_view]
            chosen.append(in_view[torch.argmax(quality)])
        selected = torch.stack(chosen) if chosen else None
        selected_by_query.append(selected)
        selected_view_counts.append(0 if selected is None else int(selected.numel()))
        exact_seed_view_counts.append(int(exact_views))
        selected_track_indices.append(selected_track)

    # Stage 2: source views are conditionally independent observations of mask
    # membership.  Noisy-or preserves a confident single visible surface while
    # strengthening repeated evidence.  The count remains available as a
    # stricter safety gate when requested.
    topology = base.new_zeros((num_rows, num_queries))
    row_view_counts = torch.zeros(
        (num_rows, num_queries), dtype=torch.int16, device=device
    )
    accepted_queries: list[bool] = []
    selected_proposal_counts: list[int] = []
    for query_index, selected in enumerate(selected_by_query):
        if selected is None or int(selected.numel()) < int(minimum_object_views):
            accepted_queries.append(False)
            selected_proposal_counts.append(0 if selected is None else int(selected.numel()))
            continue
        selected_mask = torch.zeros(num_proposals, dtype=torch.bool, device=device)
        selected_mask[selected] = True
        selected_pairs = selected_mask[props]
        selected_rows = rows[selected_pairs]
        selected_membership = conditional_membership[selected_pairs].clamp(0.0, 1.0)
        if use_proposal_quality:
            selected_membership = (
                selected_membership * prop_quality[props[selected_pairs]]
            )
        log_survival = base.new_zeros((num_rows,))
        log_survival.index_add_(
            0,
            selected_rows,
            torch.log1p(-selected_membership.clamp_max(1.0 - 1e-6)),
        )
        view_count = torch.zeros(num_rows, dtype=torch.int16, device=device)
        view_count.index_add_(
            0,
            selected_rows,
            torch.ones_like(selected_rows, dtype=torch.int16),
        )
        posterior = 1.0 - torch.exp(log_survival)
        posterior[view_count < int(minimum_row_views)] = 0.0
        posterior[posterior < float(extent_membership_floor)] = 0.0
        topology[:, query_index] = posterior
        row_view_counts[:, query_index] = view_count
        accepted_queries.append(True)
        selected_proposal_counts.append(int(selected.numel()))

    result = base.clone()
    changed_queries = 0
    for query_index, accepted in enumerate(accepted_queries):
        if not accepted:
            continue
        peak = peaks[query_index]
        # A second high text island is not identity evidence merely because it
        # is high.  Text-core retention is therefore conditional on membership
        # in the selected object topology; the unique argmax is restored below.
        identity_core = (
            (base[:, query_index] >= peak * float(identity_core_ratio))
            & (topology[:, query_index] > 0)
        )
        identity_score = torch.where(
            identity_core,
            base[:, query_index],
            torch.zeros_like(base[:, query_index]),
        )
        extent_score = topology[:, query_index]
        if float(sibling_exclusion_strength) > 0 and num_queries > 1:
            sibling = torch.cat(
                (topology[:, :query_index], topology[:, query_index + 1 :]), dim=1
            ).amax(dim=1)
            sibling_text = torch.cat(
                (base[:, :query_index], base[:, query_index + 1 :]), dim=1
            ).amax(dim=1)
            conflict = (sibling_text > base[:, query_index]).float() * sibling
            extent_score = extent_score * (
                1.0 - float(sibling_exclusion_strength) * conflict
            )
        if unknown_policy == "preserve_text_prior":
            # Absence from a sparse source-view mask is unknown: the primitive
            # may be occluded or outside that view's observation support.  Only
            # positive mask evidence is therefore allowed to override the text
            # prior.  This is a monotone, safe correction.
            result[:, query_index] = torch.maximum(base[:, query_index], extent_score)
        else:
            # Kept solely as a named coverage/purity diagnostic.  It requires
            # complete negative observation support, which the current cache
            # does not contain and must not be used as the main candidate.
            result[:, query_index] = torch.maximum(identity_score, extent_score)
        result[seeds[query_index], query_index] = scores[
            seeds[query_index], query_index
        ]
        changed_queries += 1

    stats.update(
        {
            "enabled": changed_queries > 0,
            "num_valid_memberships": int(valid.sum().item()),
            "num_queries_with_object_consensus": changed_queries,
            "selected_proposals_per_query": selected_proposal_counts,
            "selected_views_per_query": selected_view_counts,
            "exact_seed_views_per_query": exact_seed_view_counts,
            "selected_physical_track_per_query": selected_track_indices,
            "mean_extent_rows_per_accepted_query": (
                float((topology > 0).sum().item()) / changed_queries
                if changed_queries
                else 0.0
            ),
            "mean_multiview_extent_rows_per_accepted_query": (
                float((row_view_counts >= 2).sum().item()) / changed_queries
                if changed_queries
                else 0.0
            ),
            "identity_peaks_preserved_exactly": bool(
                torch.equal(result[seeds, query_axis], scores[seeds, query_axis])
            ),
        }
    )
    return result.to(dtype=scores.dtype), stats


def proposal_query_indices_from_names(
    proposal_query_names: Sequence[str],
    query_names: Sequence[str],
) -> torch.Tensor:
    """Map proposal prompt strings to the declared query interface."""

    lookup = {
        str(name).strip().casefold(): index for index, name in enumerate(query_names)
    }
    return torch.tensor(
        [lookup.get(str(name).strip().casefold(), -1) for name in proposal_query_names],
        dtype=torch.long,
    )
