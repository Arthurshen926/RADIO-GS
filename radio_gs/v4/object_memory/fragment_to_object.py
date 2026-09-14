"""Offline soft fragment-to-object hypothesis construction.

The construction is deliberately independent of the historical online token
birth/update path.  Query-free source fragments are compared globally, a
same-view cannot-link constrained agglomeration provides only an
initialization, and every fragment keeps a top-2 soft assignment plus explicit
null mass.  Object surface support is the probabilistic union of its assigned
observed fragments; completion is a separate later stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .learned_fragment_affinity import fragment_pair_features


@dataclass(frozen=True)
class FragmentAssociationConfiguration:
    evidence_threshold: float = 0.05
    merge_affinity: float = 0.55
    geometry_weight: float = 0.55
    appearance_weight: float = 0.25
    spatial_weight: float = 0.20
    conflict_weight: float = 0.0
    assignment_temperature: float = 0.10
    null_affinity: float = 0.45
    maximum_fragment_hypotheses: int = 2
    seed_policy: str = "all_fragments"
    minimum_seed_support_views: int = 3
    assignment_normalization: str = "full_then_top2"
    fragment_evidence_pooling: str = "probabilistic_union"
    cannot_link_policy: str = "exact_same_view"

    def validate(self) -> None:
        weights = self.geometry_weight + self.appearance_weight + self.spatial_weight
        if not 0 < self.evidence_threshold < 1:
            raise ValueError("fragment evidence threshold must lie in (0,1)")
        if not 0 <= self.merge_affinity <= 1 or not 0 <= self.null_affinity <= 1:
            raise ValueError("association affinities must lie in [0,1]")
        if min(self.geometry_weight, self.appearance_weight, self.spatial_weight) < 0:
            raise ValueError("association weights must be non-negative")
        if not 0 <= self.conflict_weight <= 1:
            raise ValueError("association conflict weight must lie in [0,1]")
        if abs(weights - 1.0) > 1e-6:
            raise ValueError("association weights must sum to one")
        if self.assignment_temperature <= 0:
            raise ValueError("assignment temperature must be positive")
        if self.maximum_fragment_hypotheses != 2:
            raise ValueError("v4 deployment association is fixed to top-2")
        if self.seed_policy not in {
            "all_fragments", "hierarchy_roots", "cross_view_supported"
        }:
            raise ValueError("fragment seed policy is unsupported")
        if self.minimum_seed_support_views <= 0:
            raise ValueError("minimum seed support views must be positive")
        if self.assignment_normalization not in {
            "full_then_top2", "top2_then_normalize"
        }:
            raise ValueError("fragment assignment normalization is unsupported")
        if self.fragment_evidence_pooling not in {
            "probabilistic_union", "same_view_max_then_union", "seed_medoid"
        }:
            raise ValueError("fragment evidence pooling is unsupported")
        if self.cannot_link_policy not in {
            "exact_same_view",
            "disjoint_same_view",
            "exact_group_disjoint_assignment",
        }:
            raise ValueError("fragment cannot-link policy is unsupported")


def _disjoint_same_view_cannot_link(
    geometry: torch.Tensor, view_index: torch.Tensor
) -> torch.Tensor:
    """Disjoint *observed supports*, not proof of different object identity.

    Geometry is half (IoU + containment), so only zero denotes disjointness.
    A missing/empty support is unknown, and must not create a hard negative.
    """
    geometry = torch.as_tensor(geometry, dtype=torch.float32).cpu()
    views = torch.as_tensor(view_index, dtype=torch.long).cpu()
    if geometry.shape != (views.numel(), views.numel()):
        raise ValueError("geometry and fragment views do not align")
    valid = geometry.diagonal() > 0
    return (
        (views[:, None] == views[None, :])
        & ~torch.eye(views.numel(), dtype=torch.bool)
        & valid[:, None] & valid[None, :]
        & (geometry == 0)
    )


def _pairwise_geometry(
    positive: torch.Tensor,
    element_centres: torch.Tensor,
    *,
    evidence_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(positive, dtype=torch.float32).cpu()
    centres = torch.as_tensor(element_centres, dtype=torch.float32).cpu()
    if values.ndim != 2 or centres.shape != (values.shape[1], 3):
        raise ValueError("fragment evidence and surface centres do not align")
    binary = values >= evidence_threshold
    mass = binary.sum(-1).float()
    sparse = binary.float().to_sparse()
    intersection = torch.sparse.mm(sparse, sparse.T).to_dense()
    union = mass[:, None] + mass[None] - intersection
    iou = intersection / union.clamp_min(1)
    containment = intersection / torch.minimum(mass[:, None], mass[None]).clamp_min(1)
    geometry = 0.5 * (iou + containment)
    weight = values / values.sum(-1, keepdim=True).clamp_min(1e-8)
    fragment_centres = weight @ centres
    second_moment = weight @ centres.square().sum(-1)
    radius = (
        second_moment - fragment_centres.square().sum(-1)
    ).clamp_min(0).sqrt().clamp_min(1e-4)
    distance = torch.cdist(fragment_centres, fragment_centres)
    spatial = torch.exp(-distance / (radius[:, None] + radius[None]).clamp_min(1e-4))
    valid_pair = (mass > 0)[:, None] & (mass > 0)[None]
    return (
        geometry.clamp(0, 1) * valid_pair,
        spatial.clamp(0, 1) * valid_pair,
        mass,
    )


def _pairwise_appearance(descriptors: torch.Tensor | None, count: int) -> torch.Tensor:
    if descriptors is None:
        return torch.zeros((count, count), dtype=torch.float32)
    values = torch.as_tensor(descriptors, dtype=torch.float32).cpu()
    if values.ndim == 2:
        values = values[:, None]
    if values.ndim != 3 or values.shape[0] != count or values.shape[1] == 0:
        raise ValueError("fragment appearance prototypes must have shape [F,P,D]")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("fragment appearance prototypes must be finite")
    values = F.normalize(values, dim=-1, eps=1e-8)
    similarity = torch.full((count, count), -1.0)
    for first in range(values.shape[1]):
        for second in range(values.shape[1]):
            similarity = torch.maximum(similarity, values[:, first] @ values[:, second].T)
    return ((similarity + 1.0) * 0.5).clamp(0, 1)


def _pairwise_visibility_conflict(
    positive: torch.Tensor,
    view_index: torch.Tensor,
    view_visibility: torch.Tensor,
    *,
    evidence_threshold: float,
) -> torch.Tensor:
    """Return conservative symmetric explicit-negative conflict.

    ``conflict[f,g]`` is high only when each fragment's surface is visible as
    explicit negative evidence in the other fragment's source view.  Taking
    the minimum of the two directional conflicts avoids declaring occluded or
    nested part/whole observations incompatible from a one-sided absence.
    """

    evidence = torch.as_tensor(positive, dtype=torch.float32).cpu()
    views = torch.as_tensor(view_index, dtype=torch.long).cpu()
    visibility = torch.as_tensor(view_visibility, dtype=torch.bool).cpu()
    if evidence.ndim != 2 or views.shape != (evidence.shape[0],):
        raise ValueError("fragment evidence and view indices do not align")
    if visibility.ndim != 2 or visibility.shape[1] != evidence.shape[1]:
        raise ValueError("view visibility and fragment surface axes do not align")
    if views.numel() and (int(views.min()) < 0 or int(views.max()) >= visibility.shape[0]):
        raise ValueError("fragment view index is outside the visibility inventory")
    binary = evidence >= evidence_threshold
    intersection = torch.sparse.mm(
        binary.float().to_sparse(), binary.float().to_sparse().T
    ).to_dense()
    visible_mass_by_view = torch.stack(
        [(binary & row).sum(-1).float() for row in visibility], dim=0
    )
    visible_mass = visible_mass_by_view[views]
    directional = torch.where(
        visible_mass > 0,
        (visible_mass - intersection).clamp_min(0) / visible_mass.clamp_min(1),
        torch.zeros_like(visible_mass),
    )
    conflict = torch.minimum(directional, directional.T).clamp(0, 1)
    conflict.fill_diagonal_(0)
    return conflict


def _constrained_groups(
    affinity: torch.Tensor,
    root_ids: torch.Tensor,
    view_index: torch.Tensor,
    *,
    minimum_affinity: float,
    cannot_link: torch.Tensor | None = None,
) -> list[list[int]]:
    """Globally merge root seeds while enforcing exact same-view cannot-link."""

    roots = [int(value) for value in torch.as_tensor(root_ids, dtype=torch.long).tolist()]
    if not roots:
        raise ValueError("fragment association has no root proposals")
    views = torch.as_tensor(view_index, dtype=torch.long).cpu()
    forbidden = None
    if cannot_link is not None:
        forbidden = torch.as_tensor(cannot_link, dtype=torch.bool).cpu()
        if forbidden.shape != affinity.shape or not torch.equal(forbidden, forbidden.T):
            raise ValueError("fragment cannot-link matrix must be aligned and symmetric")
    parent = list(range(len(roots)))
    members = {index: [fragment] for index, fragment in enumerate(roots)}
    member_views = {index: {int(views[fragment])} for index, fragment in enumerate(roots)}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    root_affinity = affinity[root_ids][:, root_ids]
    first, second = torch.where(torch.triu(root_affinity, diagonal=1) >= minimum_affinity)
    order = torch.argsort(root_affinity[first, second], descending=True, stable=True)
    for edge_index in order.tolist():
        left, right = find(int(first[edge_index])), find(int(second[edge_index]))
        if left == right:
            continue
        left_members = torch.tensor(members[left], dtype=torch.long)
        right_members = torch.tensor(members[right], dtype=torch.long)
        if forbidden is None:
            if member_views[left] & member_views[right]:
                continue
        elif bool(forbidden[left_members][:, right_members].any()):
            continue
        cross = affinity[left_members][:, right_members]
        support_count = min(cross.numel(), max(len(members[left]), len(members[right])))
        cluster_affinity = float(cross.flatten().topk(support_count).values.mean())
        if cluster_affinity < minimum_affinity:
            continue
        keep, remove = min(left, right), max(left, right)
        parent[remove] = keep
        members[keep].extend(members.pop(remove))
        member_views[keep] |= member_views.pop(remove)
    groups: dict[int, list[int]] = {}
    for index, root in enumerate(roots):
        groups.setdefault(find(index), []).append(root)
    return [sorted(groups[key]) for key in sorted(groups)]


def _global_parent_indices(
    view_index: torch.Tensor,
    local_index: torch.Tensor,
    parent_index: torch.Tensor,
) -> torch.Tensor:
    view = torch.as_tensor(view_index, dtype=torch.long).cpu()
    local = torch.as_tensor(local_index, dtype=torch.long).cpu()
    parent = torch.as_tensor(parent_index, dtype=torch.long).cpu()
    if not (view.shape == local.shape == parent.shape):
        raise ValueError("fragment hierarchy vectors must align")
    lookup = {(int(v), int(i)): row for row, (v, i) in enumerate(zip(view, local))}
    output = torch.full_like(parent, -1)
    for row in torch.where(parent >= 0)[0].tolist():
        key = (int(view[row]), int(parent[row]))
        if key not in lookup:
            raise ValueError("fragment parent is absent from its source view")
        output[row] = lookup[key]
    return output


def _cross_view_support_count(
    affinity: torch.Tensor,
    view_index: torch.Tensor,
    *,
    minimum_affinity: float,
) -> torch.Tensor:
    """Count distinct other views that independently support each fragment."""

    values = torch.as_tensor(affinity, dtype=torch.float32).cpu()
    views = torch.as_tensor(view_index, dtype=torch.long).cpu()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("fragment affinity must be square")
    if views.shape != (values.shape[0],):
        raise ValueError("fragment affinity and views do not align")
    support = torch.zeros(values.shape[0], dtype=torch.long)
    for view in torch.unique(views, sorted=True):
        candidates = views == view
        supported = values[:, candidates].max(-1).values >= minimum_affinity
        support += supported & (views != view)
    return support


def _soft_assignments(
    affinity: torch.Tensor,
    groups: list[list[int]],
    view_index: torch.Tensor,
    global_parent: torch.Tensor,
    *,
    temperature: float,
    null_affinity: float,
    top_k: int,
    inherit_hierarchy_root: bool,
    normalization: str = "full_then_top2",
    cannot_link: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    fragment_count = affinity.shape[0]
    logits = affinity.new_empty((fragment_count, len(groups)))
    group_by_root: dict[int, int] = {}
    mandatory_group = torch.full((fragment_count,), -1, dtype=torch.long)
    for group_id, members in enumerate(groups):
        member_ids = torch.tensor(members, dtype=torch.long)
        logits[:, group_id] = affinity[:, member_ids].max(-1).values
        group_by_root.update({member: group_id for member in members})
        mandatory_group[member_ids] = group_id

    if inherit_hierarchy_root:
        # Legacy diagnostic only.  SAM containment does not in general imply
        # object identity: a child can be the object while its parent is a
        # table, crop, or background region.  The corrected all-fragment seed
        # policy therefore does not apply this prior.
        for fragment_id in range(fragment_count):
            ancestor = fragment_id
            seen: set[int] = set()
            while int(global_parent[ancestor]) >= 0:
                if ancestor in seen:
                    raise ValueError("fragment hierarchy contains a cycle")
                seen.add(ancestor)
                ancestor = int(global_parent[ancestor])
            if ancestor in group_by_root:
                logits[fragment_id, group_by_root[ancestor]] = torch.maximum(
                    logits[fragment_id, group_by_root[ancestor]], logits.new_tensor(0.95)
                )

    # A root cannot join a hypothesis that already contains another root from
    # the same view.  Its own initialized group remains legal.
    views = torch.as_tensor(view_index, dtype=torch.long).cpu()
    forbidden = None
    if cannot_link is not None:
        forbidden = torch.as_tensor(cannot_link, dtype=torch.bool).cpu()
        if forbidden.shape != affinity.shape:
            raise ValueError("fragment cannot-link matrix does not align")
        # Constraints also apply to non-seed fragments. Previously only roots
        # were checked, allowing children/unsupported seeds to enter an
        # explicitly incompatible group in the restricted-seed policies.
        for group_id, members in enumerate(groups):
            blocked = forbidden[:, torch.as_tensor(members)].any(-1)
            blocked[torch.as_tensor(members)] = False
            logits[blocked, group_id] = -torch.inf
    for root, own_group in group_by_root.items():
        for group_id, members in enumerate(groups):
            if group_id == own_group:
                continue
            if forbidden is None:
                blocked = any(
                    int(views[item]) == int(views[root]) for item in members
                )
            else:
                blocked = bool(forbidden[root, torch.as_tensor(members)].any())
            if blocked:
                logits[root, group_id] = -torch.inf
        logits[root, own_group] = 1.0

    null = logits.new_full((fragment_count, 1), null_affinity)

    def topk_with_mandatory(values: torch.Tensor, count: int) -> torch.Tensor:
        indices = values.topk(count, dim=-1).indices
        rows = torch.where(mandatory_group >= 0)[0]
        present = (
            indices[rows] == mandatory_group[rows, None]
        ).any(-1)
        missing_rows = rows[~present]
        if missing_rows.numel():
            # Exact score ties between nested proposals can otherwise evict a
            # seed's own hypothesis by column order and create a zero column.
            indices[missing_rows, -1] = mandatory_group[missing_rows]
        return indices

    if normalization == "top2_then_normalize":
        # Structural ablation: remove cardinality dependence by selecting the
        # bounded support before normalization.
        selected_count = min(top_k, logits.shape[1])
        selected_indices = topk_with_mandatory(logits, selected_count)
        selected_logits = logits.gather(1, selected_indices)
        posterior = torch.softmax(
            torch.cat([selected_logits, null], dim=-1) / temperature, dim=-1
        )
        known = torch.zeros_like(logits).scatter(
            1, selected_indices, posterior[:, :selected_count]
        )
        null_probability = posterior[:, -1]
    elif normalization == "full_then_top2":
        posterior = torch.softmax(
            torch.cat([logits, null], dim=-1) / temperature, dim=-1
        )
        known, null_probability = posterior[:, :-1], posterior[:, -1]
        if known.shape[1] > top_k:
            indices = topk_with_mandatory(known, top_k)
            values = known.gather(1, indices)
            retained = torch.zeros_like(known).scatter(1, indices, values)
            null_probability = (
                null_probability + known.sum(-1) - retained.sum(-1)
            )
            known = retained
    else:
        raise ValueError("fragment assignment normalization is unsupported")
    return known, null_probability


def _group_medoid_indices(
    groups: list[list[int]],
    affinity: torch.Tensor,
    quality: torch.Tensor,
) -> torch.Tensor:
    """Select one auditable source fragment per global hypothesis."""

    values = torch.as_tensor(affinity, dtype=torch.float32).cpu()
    quality = torch.as_tensor(quality, dtype=torch.float32).cpu()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("medoid affinity must be square")
    if quality.shape != (values.shape[0],):
        raise ValueError("medoid quality and affinity do not align")
    medoids = []
    for members in groups:
        ids = torch.as_tensor(members, dtype=torch.long)
        if ids.ndim != 1 or not ids.numel():
            raise ValueError("every object hypothesis needs seed fragments")
        centrality = values[ids][:, ids].mean(-1)
        score = centrality * quality[ids].clamp(0, 1)
        medoids.append(int(ids[torch.argmax(score)]))
    return torch.tensor(medoids, dtype=torch.long)


def _compose_observed_membership(
    positive: torch.Tensor,
    quality: torch.Tensor,
    fragment_assignment: torch.Tensor,
    fragment_view_index: torch.Tensor,
    pooling: str = "probabilistic_union",
) -> torch.Tensor:
    evidence = torch.as_tensor(positive, dtype=torch.float32).cpu()
    quality = torch.as_tensor(quality, dtype=torch.float32).cpu().clamp(0, 1)
    assignment = torch.as_tensor(fragment_assignment, dtype=torch.float32).cpu()
    view_index = torch.as_tensor(fragment_view_index, dtype=torch.long).cpu()
    if assignment.shape[0] != evidence.shape[0] or quality.shape != (evidence.shape[0],):
        raise ValueError("fragment evidence, quality, and assignments do not align")
    if view_index.shape != (evidence.shape[0],):
        raise ValueError("fragment evidence and view indices do not align")
    membership = torch.zeros((evidence.shape[1], assignment.shape[1]), dtype=torch.float32)
    for hypothesis_id in range(assignment.shape[1]):
        selected = torch.where(assignment[:, hypothesis_id] > 1e-6)[0]
        if not selected.numel():
            continue
        if pooling == "same_view_max_then_union":
            # Structural ablation for correlated hierarchy proposals.
            view_contributions = []
            for view in torch.unique(view_index[selected], sorted=True):
                rows = selected[view_index[selected] == view]
                view_contributions.append(
                    (
                        evidence[rows]
                        * quality[rows, None]
                        * assignment[rows, hypothesis_id, None]
                    ).max(0).values
                )
            contribution = torch.stack(view_contributions).clamp(0, 1 - 1e-6)
        elif pooling == "probabilistic_union":
            contribution = (
                evidence[selected]
                * quality[selected, None]
                * assignment[selected, hypothesis_id, None]
            ).clamp(0, 1 - 1e-6)
        else:
            raise ValueError("fragment evidence pooling is unsupported")
        membership[:, hypothesis_id] = 1.0 - torch.exp(
            torch.log1p(-contribution).sum(0)
        )
    return membership.clamp(0, 1)


@torch.no_grad()
def build_fragment_object_hypotheses(
    *,
    fragment_positive: torch.Tensor,
    element_centres: torch.Tensor,
    fragment_view_index: torch.Tensor,
    fragment_local_index: torch.Tensor,
    parent_index: torch.Tensor,
    quality: torch.Tensor,
    appearance_prototypes: torch.Tensor | None,
    view_visibility: torch.Tensor | None = None,
    learned_affinity_model: torch.nn.Module | None = None,
    configuration: FragmentAssociationConfiguration,
) -> dict[str, Any]:
    configuration.validate()
    positive = torch.as_tensor(fragment_positive, dtype=torch.float32).cpu()
    view_index = torch.as_tensor(fragment_view_index, dtype=torch.long).cpu()
    parent = torch.as_tensor(parent_index, dtype=torch.long).cpu()
    if positive.ndim != 2 or view_index.shape != (positive.shape[0],):
        raise ValueError("fragment evidence and view metadata do not align")
    if appearance_prototypes is None and configuration.appearance_weight > 0:
        raise ValueError("positive appearance weight requires fragment prototypes")
    geometry, spatial, binary_mass = _pairwise_geometry(
        positive,
        element_centres,
        evidence_threshold=configuration.evidence_threshold,
    )
    raw_appearance = _pairwise_appearance(appearance_prototypes, positive.shape[0])
    if configuration.conflict_weight > 0 or learned_affinity_model is not None:
        if view_visibility is None:
            raise ValueError("positive conflict weight requires view visibility")
        conflict = _pairwise_visibility_conflict(
            positive,
            view_index,
            view_visibility,
            evidence_threshold=configuration.evidence_threshold,
        )
    else:
        conflict = torch.zeros_like(geometry)
    appearance_floor = 0.0
    appearance_ceiling = 1.0
    if appearance_prototypes is not None:
        cross_view = view_index[:, None] != view_index[None]
        off_diagonal = ~torch.eye(positive.shape[0], dtype=torch.bool)
        calibration_values = raw_appearance[cross_view & off_diagonal]
        if not calibration_values.numel():
            raise ValueError("appearance calibration requires cross-view fragment pairs")
        appearance_floor = float(torch.quantile(calibration_values, 0.50))
        appearance_ceiling = float(torch.quantile(calibration_values, 0.95))
        appearance = (
            (raw_appearance - appearance_floor)
            / max(appearance_ceiling - appearance_floor, 1e-6)
        ).clamp(0, 1)
    else:
        appearance = raw_appearance
    fixed_affinity = (
        configuration.geometry_weight * geometry
        + configuration.appearance_weight * appearance
        + configuration.spatial_weight * spatial
        - configuration.conflict_weight * conflict
    ).clamp(0, 1)
    if learned_affinity_model is None:
        grouping_affinity = fixed_affinity
        assignment_affinity = fixed_affinity
        affinity_source = "fixed_weighted_pair_components"
        assignment_affinity_source = affinity_source
    else:
        learned_affinity_model = learned_affinity_model.cpu().eval()
        pair_features = fragment_pair_features(
            geometry, appearance, spatial, conflict
        )
        with torch.no_grad():
            if getattr(learned_affinity_model, "set_conditioned", False):
                learned_logits = learned_affinity_model(
                    pair_features, view_index
                )
            else:
                learned_logits = learned_affinity_model(
                    pair_features.reshape(-1, pair_features.shape[-1])
                ).reshape(geometry.shape)
            grouping_affinity = torch.sigmoid(learned_logits)
        grouping_affinity = (
            0.5 * (grouping_affinity + grouping_affinity.T)
        ).clamp(0, 1)
        # The learned value is a calibrated binary merge-edge probability, not
        # a fragment-to-hypothesis categorical logit.  Reusing it for q_fk
        # makes hundreds of high-but-subthreshold edges steal assignment mass.
        assignment_affinity = fixed_affinity
        affinity_source = str(getattr(
            learned_affinity_model,
            "affinity_source",
            "scene_disjoint_scannet_fragment_affinity_mlp",
        ))
        assignment_affinity_source = "fixed_weighted_pair_components"
    grouping_affinity.fill_diagonal_(1.0)
    assignment_affinity.fill_diagonal_(1.0)
    same_view = view_index[:, None] == view_index[None, :]
    off_diagonal = ~torch.eye(positive.shape[0], dtype=torch.bool)
    exact_cannot_link = same_view & off_diagonal
    if configuration.cannot_link_policy == "exact_same_view":
        grouping_cannot_link = exact_cannot_link
        assignment_cannot_link = exact_cannot_link
    else:
        # Partial overlap is not disjointness.  Keep this conservative support
        # heuristic separate from evidence of different physical identities.
        disjoint_cannot_link = _disjoint_same_view_cannot_link(geometry, view_index)
        assignment_cannot_link = disjoint_cannot_link
        grouping_cannot_link = (
            exact_cannot_link
            if configuration.cannot_link_policy == "exact_group_disjoint_assignment"
            else disjoint_cannot_link
        )
    hierarchy_roots = torch.where((parent < 0) & (binary_mass > 0))[0]
    cross_view_support = _cross_view_support_count(
        grouping_affinity,
        view_index,
        minimum_affinity=configuration.merge_affinity,
    )
    if configuration.seed_policy == "all_fragments":
        seeds = torch.where(binary_mass > 0)[0]
    elif configuration.seed_policy == "hierarchy_roots":
        seeds = hierarchy_roots
    else:
        seeds = torch.where(
            (binary_mass > 0)
            & (cross_view_support >= configuration.minimum_seed_support_views)
        )[0]
        if not seeds.numel():
            raise ValueError("no fragment has the required cross-view seed support")
    groups = _constrained_groups(
        grouping_affinity,
        seeds,
        view_index,
        minimum_affinity=configuration.merge_affinity,
        cannot_link=grouping_cannot_link,
    )
    global_parent = _global_parent_indices(
        view_index, fragment_local_index, parent
    )
    assignment, null_probability = _soft_assignments(
        assignment_affinity,
        groups,
        view_index,
        global_parent,
        temperature=configuration.assignment_temperature,
        null_affinity=configuration.null_affinity,
        top_k=configuration.maximum_fragment_hypotheses,
        inherit_hierarchy_root=configuration.seed_policy == "hierarchy_roots",
        normalization=configuration.assignment_normalization,
        cannot_link=assignment_cannot_link,
    )
    unsupported = binary_mass <= 0
    assignment[unsupported] = 0
    null_probability[unsupported] = 1
    medoid_indices = _group_medoid_indices(groups, grouping_affinity, quality)
    if configuration.fragment_evidence_pooling == "seed_medoid":
        membership = (
            positive[medoid_indices].T
            * torch.as_tensor(quality, dtype=torch.float32).cpu()[medoid_indices]
        ).clamp(0, 1)
    else:
        membership = _compose_observed_membership(
            positive,
            quality,
            assignment,
            view_index,
            pooling=configuration.fragment_evidence_pooling,
        )
    active = (assignment > 0).sum(-1)
    return {
        "fragment_assignment": assignment,
        "fragment_null_probability": null_probability,
        "observed_membership": membership,
        "hypothesis_seed_fragments": groups,
        "hypothesis_representative_fragments": medoid_indices,
        "pairwise_audit": {
            "seed_policy": configuration.seed_policy,
            "pair_affinity_source": affinity_source,
            "fragment_assignment_affinity_source": assignment_affinity_source,
            "seed_fragment_count": int(seeds.numel()),
            "hierarchy_root_fragment_count": int(hierarchy_roots.numel()),
            "minimum_seed_support_views": (
                configuration.minimum_seed_support_views
                if configuration.seed_policy == "cross_view_supported" else None
            ),
            "mean_cross_view_support_count": float(
                cross_view_support.float().mean()
            ),
            "maximum_cross_view_support_count": int(cross_view_support.max()),
            "object_hypothesis_count": len(groups),
            "singleton_hypothesis_count": sum(len(group) == 1 for group in groups),
            "maximum_roots_per_hypothesis": max(map(len, groups)),
            "mean_roots_per_hypothesis": float(sum(map(len, groups)) / len(groups)),
            "top2_active_fragment_fraction": float((active > 1).float().mean()),
            "mean_null_probability": float(null_probability.mean()),
            "known_surface_fraction": float((membership.max(-1).values > 0).float().mean()),
            "membership_uses_single_seed_medoid": (
                configuration.fragment_evidence_pooling == "seed_medoid"
            ),
            "cannot_link_policy": configuration.cannot_link_policy,
            "cannot_link_implementation": "nonempty_zero_overlap_v2",
            "same_view_pair_count": int((same_view & off_diagonal).sum() // 2),
            "same_view_group_cannot_link_pair_count": int(
                grouping_cannot_link.sum() // 2
            ),
            "same_view_assignment_cannot_link_pair_count": int(
                assignment_cannot_link.sum() // 2
            ),
            "appearance_calibration_median": appearance_floor,
            "appearance_calibration_q95": appearance_ceiling,
            "mean_cross_view_visibility_conflict": float(
                conflict[view_index[:, None] != view_index[None, :]].mean()
            ),
        },
    }
