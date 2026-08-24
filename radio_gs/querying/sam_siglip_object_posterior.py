"""Typed text-object posterior from SAM extent and SigLIP identity evidence."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from radio_gs.querying.latent_proposal_posterior import (
    latent_proposal_null_posterior,
)


def _scatter_amax(values: torch.Tensor, indices: torch.Tensor, size: int) -> torch.Tensor:
    result = values.new_zeros((int(size),))
    if values.numel():
        result.scatter_reduce_(0, indices, values, reduce="amax", include_self=True)
    return result


def sam_siglip_object_posterior(
    base_scores: torch.Tensor,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    membership_weights: torch.Tensor,
    proposal_view_indices: torch.Tensor,
    proposal_parent_index: torch.Tensor,
    proposal_descriptor_scores: torch.Tensor,
    proposal_context_scores: torch.Tensor,
    *,
    proposal_appearance_features: torch.Tensor | None = None,
    proposal_quality: torch.Tensor | None = None,
    proposal_area_fraction: torch.Tensor | None = None,
    positive_core_ratio: float = 0.80,
    minimum_object_views: int = 2,
    maximum_object_views: int = 12,
    view_identity_margin: float = 0.12,
    minimum_descriptor_score: float = 0.55,
    descriptor_gate: str = "absolute",
    descriptor_listwise_margin: float = 0.12,
    parent_identity_tolerance: float = 0.05,
    parent_field_peak_ratio: float = 0.75,
    extent_membership_floor: float = 0.50,
    association_membership_floor: float | None = None,
    composition: str = "maximum",
    association_mode: str = "weighted_jaccard_components",
    candidates_per_view: int = 3,
    maximum_proposal_area_fraction: float = 0.25,
    minimum_cross_view_jaccard: float = 0.02,
    minimum_cross_view_overlap: float = 0.15,
    minimum_cross_view_appearance_cosine: float = -1.0,
    require_field_peak_anchor: bool = True,
    latent_logit_temperature: float = 8.0,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Factor proposal identity from object extent with per-query safe fallback.

    Identity is a convex evidence score from official mask-aligned SigLIP2,
    Gaussian field peak/core support, and immutable field-peak containment.
    The best identity node in each source view ascends the SAM containment tree
    while identity remains stable, explicitly changing scale from discriminant
    part to enclosing object.  Only views within a fixed margin of the strongest
    source observation participate.  Insufficient multi-view evidence returns
    the base query column bit-for-bit.
    """

    base = torch.as_tensor(base_scores).float()
    rows = torch.as_tensor(row_indices).long()
    props = torch.as_tensor(proposal_indices).long()
    weights = torch.as_tensor(membership_weights).float()
    views = torch.as_tensor(proposal_view_indices).long()
    parents = torch.as_tensor(proposal_parent_index).long()
    descriptor = torch.as_tensor(proposal_descriptor_scores).float()
    context = torch.as_tensor(proposal_context_scores).float()
    if base.ndim != 2:
        raise ValueError("base_scores must be [N,Q]")
    if not (rows.ndim == props.ndim == weights.ndim == 1) or not (
        rows.shape == props.shape == weights.shape
    ):
        raise ValueError("sparse membership vectors must align")
    num_rows, num_queries = map(int, base.shape)
    num_proposals = int(views.numel())
    if (
        parents.shape != (num_proposals,)
        or descriptor.shape != (num_proposals, num_queries)
        or context.shape != descriptor.shape
    ):
        raise ValueError("proposal topology/identity axes differ")
    if proposal_quality is not None and torch.as_tensor(proposal_quality).shape != (num_proposals,):
        raise ValueError("proposal quality must align with proposals")
    if proposal_area_fraction is not None and torch.as_tensor(proposal_area_fraction).shape != (num_proposals,):
        raise ValueError("proposal area must align with proposals")
    appearance = None
    if proposal_appearance_features is not None:
        appearance = F.normalize(
            torch.as_tensor(proposal_appearance_features).float(), dim=-1, eps=1e-8
        )
        if appearance.ndim != 2 or appearance.shape[0] != num_proposals:
            raise ValueError("proposal appearance must align with proposals")
    if not 0.0 < float(positive_core_ratio) <= 1.0:
        raise ValueError("positive_core_ratio must lie in (0,1]")
    if int(minimum_object_views) <= 0 or int(maximum_object_views) < int(minimum_object_views):
        raise ValueError("object view limits differ")
    if not 0.0 <= float(view_identity_margin) <= 1.0:
        raise ValueError("view_identity_margin must lie in [0,1]")
    if descriptor_gate not in {"absolute", "query_listwise"}:
        raise ValueError("unsupported descriptor gate")
    if not 0.0 <= float(descriptor_listwise_margin) <= 1.0:
        raise ValueError("descriptor_listwise_margin must lie in [0,1]")
    if composition not in {"maximum", "noisy_or"}:
        raise ValueError("unsupported topology composition")
    association_floor = (
        float(extent_membership_floor)
        if association_membership_floor is None
        else float(association_membership_floor)
    )
    if not 0.0 < association_floor <= 1.0:
        raise ValueError("association_membership_floor must lie in (0,1]")
    if association_mode not in {
        "none",
        "weighted_jaccard_components",
        "field_peak_anchored_star",
        "latent_proposal_marginal",
    }:
        raise ValueError("unsupported proposal association mode")
    if int(candidates_per_view) <= 0:
        raise ValueError("candidates_per_view must be positive")
    if not 0.0 < float(maximum_proposal_area_fraction) <= 1.0:
        raise ValueError("maximum proposal area must lie in (0,1]")
    if not 0.0 <= float(minimum_cross_view_jaccard) <= 1.0:
        raise ValueError("minimum cross-view Jaccard must lie in [0,1]")
    if not 0.0 <= float(minimum_cross_view_overlap) <= 1.0:
        raise ValueError("minimum cross-view overlap must lie in [0,1]")
    if not -1.0 <= float(minimum_cross_view_appearance_cosine) <= 1.0:
        raise ValueError("minimum appearance cosine must lie in [-1,1]")
    if appearance is None and float(minimum_cross_view_appearance_cosine) > -1.0:
        raise ValueError("appearance gate requested without proposal appearance teacher")
    if float(latent_logit_temperature) <= 0:
        raise ValueError("latent logit temperature must be positive")
    valid = (
        (rows >= 0)
        & (rows < num_rows)
        & (props >= 0)
        & (props < num_proposals)
        & torch.isfinite(weights)
        & (weights > 0)
    )
    if not bool(valid.any()):
        return base_scores, {"enabled": False, "fallback_reason": "empty_membership"}
    rows, props, weights = rows[valid], props[valid], weights[valid]
    proposal_mass = base.new_zeros((num_proposals,))
    proposal_mass.index_add_(0, props, weights)
    proposal_max = _scatter_amax(weights, props, num_proposals)
    conditional = weights / proposal_max[props].clamp_min(1e-8)
    quality = (
        torch.as_tensor(proposal_quality).float().clamp(0.0, 1.0)
        if proposal_quality is not None
        else torch.ones(num_proposals)
    )
    area = (
        torch.as_tensor(proposal_area_fraction).float()
        if proposal_area_fraction is not None
        else torch.full((num_proposals,), float("nan"))
    )

    peaks, peak_rows = base.max(dim=0)
    field_tail = base.new_zeros((num_proposals, num_queries))
    core_fraction = base.new_zeros((num_proposals, num_queries))
    peak_membership = base.new_zeros((num_proposals, num_queries))
    for query_index in range(num_queries):
        values = base[rows, query_index]
        field_tail[:, query_index] = _scatter_amax(values, props, num_proposals)
        core = values >= peaks[query_index] * float(positive_core_ratio)
        numerator = base.new_zeros((num_proposals,))
        numerator.index_add_(0, props, weights * core.float())
        core_fraction[:, query_index] = numerator / proposal_mass.clamp_min(1e-8)
        exact = rows == peak_rows[query_index]
        if bool(exact.any()):
            peak_membership[:, query_index].scatter_reduce_(
                0,
                props[exact],
                conditional[exact],
                reduce="amax",
                include_self=True,
            )
    contrast = (descriptor - context).clamp_min(0.0)
    identity = (
        0.55 * descriptor
        + 0.20 * field_tail
        + 0.15 * core_fraction.sqrt()
        + 0.10 * peak_membership
        + 0.05 * contrast
    ) * (0.75 + 0.25 * quality[:, None])

    result = base.clone()
    selected_counts: list[int] = []
    accepted_views: list[int] = []
    parent_ascent_steps: list[int] = []
    fallback: list[bool] = []
    selected_proposal_indices: list[list[int]] = []
    selected_proposal_area_fraction: list[list[float]] = []
    selected_proposal_descriptor_scores: list[list[float]] = []
    selected_proposal_identity_scores: list[list[float]] = []
    selected_component_view_counts: list[int] = []
    selected_component_edge_counts: list[int] = []
    pre_ascent_proposal_indices: list[list[int]] = []
    parent_appearance_rejections: list[int] = []
    unique_views = torch.unique(views, sorted=True)

    # A proposal is an object observation only if another source view sees a
    # compatible set of Gaussian rows.  Precompute thresholded row supports;
    # pair similarities are evaluated lazily because a query uses only the top
    # few identity candidates from each view.
    support_sets: list[set[int]] = [set() for _ in range(num_proposals)]
    # Association uses a separately controllable high-purity core.  Extent
    # composition may still use the broader proposal membership after an
    # identity track has been accepted.
    support_keep = conditional >= association_floor
    for row, proposal in zip(rows[support_keep].tolist(), props[support_keep].tolist()):
        support_sets[int(proposal)].add(int(row))

    def associated(left: int, right: int) -> bool:
        if int(views[left]) == int(views[right]):
            return False
        a, b = support_sets[left], support_sets[right]
        if not a or not b:
            return False
        intersection = len(a.intersection(b))
        if intersection == 0:
            return False
        if appearance is not None and float(
            torch.dot(appearance[left], appearance[right])
        ) < float(minimum_cross_view_appearance_cosine):
            return False
        jaccard = intersection / max(len(a) + len(b) - intersection, 1)
        overlap = intersection / max(min(len(a), len(b)), 1)
        return (
            jaccard >= float(minimum_cross_view_jaccard)
            or overlap >= float(minimum_cross_view_overlap)
        )

    def association_strength(left: int, right: int) -> float:
        """Continuous source-only evidence for a cross-view identity edge."""

        if int(views[left]) == int(views[right]):
            return 0.0
        a, b = support_sets[left], support_sets[right]
        if not a or not b:
            return 0.0
        intersection = len(a.intersection(b))
        if intersection == 0:
            return 0.0
        jaccard = intersection / max(len(a) + len(b) - intersection, 1)
        overlap = intersection / max(min(len(a), len(b)), 1)
        return max(float(jaccard), float(overlap))

    if association_mode == "latent_proposal_marginal":
        proposal_valid = torch.zeros(
            (num_proposals, num_queries), dtype=torch.bool, device=base.device
        )
        proposal_logits = base.new_zeros((num_proposals, num_queries))
        association = base.new_zeros((num_proposals,))
        for left in range(num_proposals):
            association[left] = max(
                (
                    association_strength(left, right)
                    for right in range(num_proposals)
                    if right != left
                ),
                default=0.0,
            )
        for query_index in range(num_queries):
            eligible = torch.ones(num_proposals, dtype=torch.bool, device=base.device)
            if proposal_area_fraction is not None:
                eligible &= area.to(base.device) <= float(maximum_proposal_area_fraction)
            descriptor_floor = float(minimum_descriptor_score)
            if descriptor_gate == "query_listwise" and bool(eligible.any()):
                descriptor_floor = float(
                    descriptor[eligible.cpu(), query_index].max()
                ) - float(descriptor_listwise_margin)
            eligible &= descriptor[:, query_index].to(base.device) >= descriptor_floor
            eligible &= association > 0
            if require_field_peak_anchor:
                eligible &= peak_membership[:, query_index].to(base.device) > 0
            proposal_valid[:, query_index] = eligible
            count = int(eligible.sum())
            if count:
                # A uniform prior over the complete proposal cohort prevents
                # proposal multiplicity from defeating the explicit null.
                evidence = (
                    float(latent_logit_temperature)
                    * (identity[:, query_index].to(base.device) - descriptor_floor)
                    + torch.log(association.clamp_min(1e-8))
                    - torch.log(base.new_tensor(float(count)))
                )
                proposal_logits[:, query_index] = evidence
        marginal = latent_proposal_null_posterior(
            base.clamp(0.0, 1.0),
            rows,
            props,
            conditional.clamp(0.0, 1.0),
            proposal_logits,
            base.new_zeros((num_queries,)),
            proposal_valid=proposal_valid,
        )
        probability = marginal.probability
        probability[base < 0] = base[base < 0]
        return probability, {
            "enabled": True,
            "mode": "official_sam3_siglip2_latent_proposal_null_marginal_v1",
            "association_mode": association_mode,
            "identity_authority": "official_mask_aligned_siglip2_plus_field_peak_core",
            "extent_authority": "official_sam3_exact_mpr_probability",
            "latent_logit_temperature": float(latent_logit_temperature),
            "null_logit": 0.0,
            "proposal_prior": "uniform_over_valid_complete_source_proposal_cohort",
            "valid_proposal_counts": proposal_valid.sum(dim=0).tolist(),
            "fallback_queries": (~proposal_valid.any(dim=0)).tolist(),
            "fallback_query_count": int((~proposal_valid.any(dim=0)).sum()),
            "null_probability": marginal.null_probability.tolist(),
            "maximum_proposal_area_fraction": float(maximum_proposal_area_fraction),
            "descriptor_gate": str(descriptor_gate),
            "descriptor_listwise_margin": float(descriptor_listwise_margin),
            "require_field_peak_anchor": bool(require_field_peak_anchor),
        }

    for query_index in range(num_queries):
        eligible_area = torch.ones(num_proposals, dtype=torch.bool)
        if proposal_area_fraction is not None:
            eligible_area = area <= float(maximum_proposal_area_fraction)
        descriptor_floor = float(minimum_descriptor_score)
        if descriptor_gate == "query_listwise" and bool(eligible_area.any()):
            # Frozen SigLIP relevancy is comparable within one query but its
            # offset varies across text strings.  A query-relative margin is
            # therefore the calibrated identity test; cross-view association
            # and immutable field-peak anchoring remain the object test.
            descriptor_floor = float(descriptor[eligible_area, query_index].max()) - float(
                descriptor_listwise_margin
            )
        candidate_nodes: list[int] = []
        for view in unique_views.tolist():
            candidates = torch.where(views == int(view))[0]
            if not candidates.numel():
                continue
            if proposal_area_fraction is not None:
                candidates = candidates[
                    area[candidates] <= float(maximum_proposal_area_fraction)
                ]
            candidates = candidates[
                descriptor[candidates, query_index] >= descriptor_floor
            ]
            if not candidates.numel():
                continue
            ranking = identity[candidates, query_index]
            order = torch.argsort(ranking, descending=True, stable=True)
            candidate_nodes.extend(
                int(value)
                for value in candidates[order[: int(candidates_per_view)]].tolist()
            )
        strongest = max(
            (float(identity[index, query_index]) for index in candidate_nodes),
            default=0.0,
        )
        descriptor_best = max(
            (float(descriptor[index, query_index]) for index in candidate_nodes),
            default=0.0,
        )
        candidate_nodes = [
            index
            for index in candidate_nodes
            if float(identity[index, query_index])
            >= strongest - float(view_identity_margin)
            and float(descriptor[index, query_index])
            >= max(
                descriptor_floor,
                descriptor_best - float(view_identity_margin),
            )
        ]

        component_edge_count = 0
        association_anchor: int | None = None
        if association_mode == "field_peak_anchored_star" and candidate_nodes:
            # Connected-component transitivity is not a physical identity
            # relation: A may overlap B and B may overlap C even when A and C
            # are different adjacent instances.  A conservative star admits a
            # source observation only when it is directly associated with the
            # immutable field-peak observation for this query.
            anchors = [
                proposal
                for proposal in candidate_nodes
                if float(peak_membership[proposal, query_index]) > 0.0
            ]
            if anchors:
                anchor = max(
                    anchors,
                    key=lambda proposal: (
                        float(identity[proposal, query_index]),
                        float(peak_membership[proposal, query_index]),
                        -int(proposal),
                    ),
                )
                association_anchor = int(anchor)
                directly_linked = [
                    proposal
                    for proposal in candidate_nodes
                    if proposal == anchor or associated(anchor, proposal)
                ]
                component_edge_count = len(directly_linked) - 1
                by_view: dict[int, int] = {}
                for proposal in directly_linked:
                    view = int(views[proposal])
                    proposal_rank = (
                        float(torch.dot(appearance[anchor], appearance[proposal]))
                        if appearance is not None
                        else float(identity[proposal, query_index])
                    )
                    current = by_view.get(view)
                    current_rank = (
                        float(torch.dot(appearance[anchor], appearance[current]))
                        if appearance is not None and current is not None
                        else (
                            float(identity[current, query_index])
                            if current is not None
                            else float("-inf")
                        )
                    )
                    if (
                        current is None
                        or proposal_rank > current_rank
                        or (
                            proposal_rank == current_rank
                            and float(identity[proposal, query_index])
                            > float(identity[current, query_index])
                        )
                    ):
                        by_view[view] = proposal
                component = list(by_view.values())
                if len(component) < int(minimum_object_views):
                    component = []
            else:
                component = []
        elif association_mode == "weighted_jaccard_components" and candidate_nodes:
            roots = list(range(len(candidate_nodes)))

            def find(index: int) -> int:
                while roots[index] != index:
                    roots[index] = roots[roots[index]]
                    index = roots[index]
                return index

            def union(left: int, right: int) -> None:
                a, b = find(left), find(right)
                if a != b:
                    roots[b] = a

            for left in range(len(candidate_nodes)):
                for right in range(left + 1, len(candidate_nodes)):
                    if associated(candidate_nodes[left], candidate_nodes[right]):
                        union(left, right)
                        component_edge_count += 1
            components: dict[int, list[int]] = {}
            for position, proposal in enumerate(candidate_nodes):
                components.setdefault(find(position), []).append(proposal)
            ranked_components: list[
                tuple[tuple[float, float, float, int], list[int], bool]
            ] = []
            for component in components.values():
                by_view: dict[int, int] = {}
                for proposal in component:
                    view = int(views[proposal])
                    if (
                        view not in by_view
                        or float(identity[proposal, query_index])
                        > float(identity[by_view[view], query_index])
                    ):
                        by_view[view] = proposal
                chosen = list(by_view.values())
                view_count = len(chosen)
                mean_identity = sum(
                    float(identity[index, query_index]) for index in chosen
                ) / max(view_count, 1)
                peak_anchor = max(
                    (float(peak_membership[index, query_index]) for index in chosen),
                    default=0.0,
                )
                strongest_identity = max(
                    (float(identity[index, query_index]) for index in chosen),
                    default=0.0,
                )
                rank = (
                    1.0 if peak_anchor > 0 else 0.0,
                    strongest_identity + 0.10 * mean_identity,
                    peak_anchor,
                    view_count,
                )
                ranked_components.append((rank, chosen, peak_anchor > 0))
            eligible = [
                value
                for value in ranked_components
                if len(value[1]) >= int(minimum_object_views)
                and (not require_field_peak_anchor or value[2])
            ]
            component = max(eligible, key=lambda value: value[0])[1] if eligible else []
        else:
            by_view: dict[int, int] = {}
            for proposal in candidate_nodes:
                view = int(views[proposal])
                if view not in by_view or float(identity[proposal, query_index]) > float(
                    identity[by_view[view], query_index]
                ):
                    by_view[view] = proposal
            component = list(by_view.values())

        pre_ascent_proposal_indices.append([int(value) for value in component])
        appearance_rejections = 0
        per_view: list[tuple[float, int, int]] = []
        for original in component:
            selected = int(original)
            ascent = 0
            seen: set[int] = set()
            while int(parents[selected]) >= 0:
                parent = int(parents[selected])
                if parent in seen or int(views[parent]) != int(views[original]):
                    break
                seen.add(parent)
                parent_changes_native_identity = False
                if (
                    appearance is not None
                    and association_anchor is not None
                    and float(minimum_cross_view_appearance_cosine) > -1.0
                ):
                    # Association is established on the discriminant child
                    # proposal.  SAM parent ascent changes the actual extent
                    # used by the posterior, so it must preserve that same
                    # native-DINO physical-identity authority.  Otherwise a
                    # valid child edge can silently expand into an unrelated
                    # enclosing object/background parent.
                    parent_changes_native_identity = float(
                        torch.dot(appearance[association_anchor], appearance[parent])
                    ) < float(minimum_cross_view_appearance_cosine)
                if (
                    (
                        proposal_area_fraction is not None
                        and float(area[parent]) > float(maximum_proposal_area_fraction)
                    )
                    or
                    float(descriptor[parent, query_index])
                    < descriptor_floor
                    or
                    float(identity[parent, query_index])
                    < float(identity[selected, query_index]) - float(parent_identity_tolerance)
                    or float(descriptor[parent, query_index])
                    < float(descriptor[selected, query_index]) - float(parent_identity_tolerance)
                    or float(field_tail[parent, query_index])
                    < float(field_tail[selected, query_index]) * float(parent_field_peak_ratio)
                    or parent_changes_native_identity
                ):
                    appearance_rejections += int(parent_changes_native_identity)
                    break
                selected = parent
                ascent += 1
            per_view.append((float(identity[selected, query_index]), selected, ascent))
        per_view.sort(key=lambda value: (-value[0], value[1]))
        selected = per_view[: int(maximum_object_views)]
        parent_appearance_rejections.append(appearance_rejections)
        if len(selected) < int(minimum_object_views):
            selected_counts.append(len(selected))
            accepted_views.append(0)
            parent_ascent_steps.append(sum(value[2] for value in selected))
            fallback.append(True)
            selected_proposal_indices.append([value[1] for value in selected])
            selected_proposal_area_fraction.append([float(area[value[1]]) for value in selected])
            selected_proposal_descriptor_scores.append([float(descriptor[value[1], query_index]) for value in selected])
            selected_proposal_identity_scores.append([float(value[0]) for value in selected])
            selected_component_view_counts.append(len(component))
            selected_component_edge_counts.append(component_edge_count)
            continue
        selected_ids = torch.tensor([value[1] for value in selected], dtype=torch.long)
        pair_keep = torch.isin(props, selected_ids)
        selected_rows = rows[pair_keep]
        selected_membership = conditional[pair_keep].clamp(0.0, 1.0)
        topology = base.new_zeros((num_rows,))
        if composition == "maximum":
            topology.scatter_reduce_(
                0,
                selected_rows,
                selected_membership,
                reduce="amax",
                include_self=True,
            )
        else:
            log_survival = base.new_zeros((num_rows,))
            log_survival.index_add_(
                0,
                selected_rows,
                torch.log1p(-selected_membership.clamp_max(1.0 - 1e-6)),
            )
            topology = 1.0 - torch.exp(log_survival)
        topology[topology < float(extent_membership_floor)] = 0.0
        # The field peak remains the immutable localization authority even if
        # the selected source surfaces do not observe that exact Gaussian.
        topology[peak_rows[query_index]] = peaks[query_index]
        result[:, query_index] = topology
        selected_counts.append(len(selected))
        accepted_views.append(len(selected))
        parent_ascent_steps.append(sum(value[2] for value in selected))
        fallback.append(False)
        selected_proposal_indices.append([value[1] for value in selected])
        selected_proposal_area_fraction.append([float(area[value[1]]) for value in selected])
        selected_proposal_descriptor_scores.append([float(descriptor[value[1], query_index]) for value in selected])
        selected_proposal_identity_scores.append([float(value[0]) for value in selected])
        selected_component_view_counts.append(len(component))
        selected_component_edge_counts.append(component_edge_count)

    return result, {
        "enabled": True,
        "mode": "official_sam3_siglip2_identity_extent_factorization_v3",
        "identity_authority": "official_mask_aligned_siglip2_plus_field_peak_core",
        "extent_authority": "official_sam3_multiscale_parent_ascent_exact_mpr",
        "composition": str(composition),
        "positive_core_ratio": float(positive_core_ratio),
        "minimum_object_views": int(minimum_object_views),
        "maximum_object_views": int(maximum_object_views),
        "view_identity_margin": float(view_identity_margin),
        "minimum_descriptor_score": float(minimum_descriptor_score),
        "descriptor_gate": str(descriptor_gate),
        "descriptor_listwise_margin": float(descriptor_listwise_margin),
        "parent_identity_tolerance": float(parent_identity_tolerance),
        "parent_field_peak_ratio": float(parent_field_peak_ratio),
        "extent_membership_floor": float(extent_membership_floor),
        "association_membership_floor": association_floor,
        "association_mode": str(association_mode),
        "candidates_per_view": int(candidates_per_view),
        "maximum_proposal_area_fraction": float(maximum_proposal_area_fraction),
        "minimum_cross_view_jaccard": float(minimum_cross_view_jaccard),
        "minimum_cross_view_overlap": float(minimum_cross_view_overlap),
        "minimum_cross_view_appearance_cosine": float(
            minimum_cross_view_appearance_cosine
        ),
        "proposal_appearance_authority": (
            "independent_native_dinov2_mask_pool"
            if appearance is not None
            else "none"
        ),
        "require_field_peak_anchor": bool(require_field_peak_anchor),
        "selected_proposal_counts": selected_counts,
        "accepted_view_counts": accepted_views,
        "parent_ascent_steps": parent_ascent_steps,
        "fallback_queries": fallback,
        "fallback_query_count": int(sum(fallback)),
        "selected_proposal_indices": selected_proposal_indices,
        "selected_proposal_area_fraction": selected_proposal_area_fraction,
        "selected_proposal_descriptor_scores": selected_proposal_descriptor_scores,
        "selected_proposal_identity_scores": selected_proposal_identity_scores,
        "selected_component_view_counts": selected_component_view_counts,
        "selected_component_edge_counts": selected_component_edge_counts,
        "pre_ascent_proposal_indices": pre_ascent_proposal_indices,
        "parent_appearance_rejections": parent_appearance_rejections,
    }
