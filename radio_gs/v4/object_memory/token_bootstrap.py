"""Deterministic source-only initialization of surface object tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .mask_matching import partial_soft_match
from .observed_evidence import ObservedObjectEvidence


@dataclass(frozen=True)
class BootstrapViewResult:
    token_ids: torch.Tensor
    token_probability: torch.Tensor
    null_probability: torch.Tensor
    granularity: torch.Tensor
    created: torch.Tensor


class SurfaceTokenBootstrap:
    """Build tokens online from query-free masks and their surface evidence.

    Only hierarchy roots may create a new token.  Any proposal, including a
    part proposal, may update an existing token.  The implementation consumes
    no class, instance, text, or benchmark-query signal.
    """

    def __init__(
        self,
        element_centres: torch.Tensor,
        *,
        minimum_overlap: float = 0.20,
        null_logit: float = 0.50,
        temperature: float = 0.10,
        geometry_weight: float = 0.25,
        appearance_weight: float = 0.0,
        membership_epsilon: float = 1e-6,
        minimum_scale: float = 0.04,
        batch_birth_overlap: float | None = None,
    ) -> None:
        centres = torch.as_tensor(element_centres, dtype=torch.float32)
        if centres.ndim != 2 or centres.shape[1] != 3:
            raise ValueError("element centres must have shape [E, 3]")
        if not 0 <= minimum_overlap <= 1:
            raise ValueError("minimum overlap must lie in [0, 1]")
        self.element_centres = centres
        self.membership = centres.new_zeros((centres.shape[0], 0))
        self.evidence_mass = centres.new_zeros((0,))
        self.birth_view = torch.empty(0, dtype=torch.long, device=centres.device)
        self.descriptor_sum: torch.Tensor | None = None
        self.descriptor_mass = centres.new_zeros((0,))
        self.minimum_overlap = float(minimum_overlap)
        self.null_logit = float(null_logit)
        self.temperature = float(temperature)
        self.geometry_weight = float(geometry_weight)
        self.appearance_weight = float(appearance_weight)
        self.membership_epsilon = float(membership_epsilon)
        self.minimum_scale = float(minimum_scale)
        self.batch_birth_overlap = float(
            minimum_overlap if batch_birth_overlap is None else batch_birth_overlap
        )
        if not 0 <= self.batch_birth_overlap <= 1:
            raise ValueError("batch birth overlap must lie in [0, 1]")

    @property
    def num_tokens(self) -> int:
        return int(self.membership.shape[1])

    def _geometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_tokens == 0:
            empty = self.element_centres.new_zeros((0, 3))
            return empty, empty
        weights = self.membership
        mass = weights.sum(0).clamp_min(self.membership_epsilon)
        centres = weights.T @ self.element_centres / mass[:, None]
        residual = self.element_centres[:, None, :] - centres[None, :, :]
        variance = (weights[..., None] * residual.square()).sum(0) / mass[:, None]
        scales = variance.sqrt().clamp_min(self.minimum_scale)
        return centres, scales

    def _mask_centre(self, positive: torch.Tensor) -> torch.Tensor:
        mass = positive.sum().clamp_min(self.membership_epsilon)
        return positive @ self.element_centres / mass

    def _create(
        self,
        positive: torch.Tensor,
        quality: torch.Tensor,
        view_id: int,
        descriptor: torch.Tensor | None,
    ) -> int:
        value = (positive * quality.clamp(0, 1)).clamp(0, 1)
        if float(value.sum()) <= self.membership_epsilon:
            return -1
        self.membership = torch.cat([self.membership, value[:, None]], dim=1)
        self.evidence_mass = torch.cat([self.evidence_mass, value.sum()[None]])
        self.birth_view = torch.cat([
            self.birth_view,
            torch.tensor([view_id], dtype=torch.long, device=self.birth_view.device),
        ])
        if descriptor is not None:
            weight = quality.clamp(0, 1) * positive.sum().clamp_min(self.membership_epsilon)
            value = descriptor * weight
            if self.descriptor_sum is None:
                self.descriptor_sum = value[None]
            else:
                self.descriptor_sum = torch.cat([self.descriptor_sum, value[None]], dim=0)
            self.descriptor_mass = torch.cat([self.descriptor_mass, weight[None]])
        return self.num_tokens - 1

    def _update(
        self,
        token_id: int,
        positive: torch.Tensor,
        quality: torch.Tensor,
        descriptor: torch.Tensor | None,
    ) -> None:
        value = (positive * quality.clamp(0, 1)).clamp(0, 1)
        self.membership[:, token_id] = torch.maximum(self.membership[:, token_id], value)
        self.evidence_mass[token_id] += value.sum()
        if descriptor is not None:
            if self.descriptor_sum is None:
                raise RuntimeError("descriptor state was not initialized at token birth")
            weight = quality.clamp(0, 1) * positive.sum().clamp_min(self.membership_epsilon)
            self.descriptor_sum[token_id] += descriptor * weight
            self.descriptor_mass[token_id] += weight

    @torch.no_grad()
    def process_view(
        self,
        evidence: ObservedObjectEvidence,
        *,
        element_visibility: torch.Tensor,
        parent_index: torch.Tensor,
        proposal_descriptors: torch.Tensor | None = None,
    ) -> BootstrapViewResult:
        if evidence.positive.device != self.element_centres.device:
            raise ValueError("evidence and token bootstrap must use the same device")
        visibility = torch.as_tensor(
            element_visibility, dtype=torch.float32, device=self.element_centres.device
        )
        parents = torch.as_tensor(parent_index, dtype=torch.long, device=self.element_centres.device)
        masks = evidence.positive.shape[0]
        if visibility.shape != evidence.positive.shape or parents.shape != (masks,):
            raise ValueError("visibility or parent index does not align with proposals")
        descriptors = None
        if proposal_descriptors is not None:
            descriptors = torch.as_tensor(
                proposal_descriptors, dtype=torch.float32, device=self.element_centres.device
            )
            if descriptors.ndim != 2 or descriptors.shape[0] != masks:
                raise ValueError("proposal descriptors must have shape [M, D]")
            descriptors = torch.nn.functional.normalize(descriptors, dim=-1, eps=1e-8)
            if self.descriptor_sum is not None and descriptors.shape[1] != self.descriptor_sum.shape[1]:
                raise ValueError("proposal descriptor dimension changed across views")
        elif self.descriptor_sum is not None:
            raise ValueError("proposal descriptors cannot be omitted after descriptor token birth")

        token_ids = torch.full((masks,), -1, dtype=torch.long, device=self.element_centres.device)
        token_probability = torch.zeros(masks, device=self.element_centres.device)
        null_probability = torch.ones(masks, device=self.element_centres.device)
        granularity = torch.full((masks,), 2, dtype=torch.long, device=self.element_centres.device)
        created = torch.zeros(masks, dtype=torch.bool, device=self.element_centres.device)

        # Establish object-scale hypotheses before allowing auxiliary proposals
        # to add evidence.  Quality and mass make the order deterministic.
        mass = evidence.positive.sum(-1)
        priority = evidence.quality * mass
        roots = torch.where(parents < 0)[0]
        parts = torch.where(parents >= 0)[0]
        root_order = roots[torch.argsort(priority[roots], descending=True, stable=True)]
        part_order = parts[torch.argsort(priority[parts], descending=True, stable=True)]

        for mask_id in torch.cat([root_order, part_order]).tolist():
            positive = evidence.positive[mask_id]
            if float(positive.sum()) <= self.membership_epsilon:
                continue
            matched_id = -1
            if self.num_tokens:
                token_centres, token_scales = self._geometry()
                single = ObservedObjectEvidence(
                    evidence.positive[mask_id : mask_id + 1],
                    evidence.negative[mask_id : mask_id + 1],
                    evidence.unknown[mask_id : mask_id + 1],
                    evidence.view_ids[mask_id : mask_id + 1],
                    evidence.quality[mask_id : mask_id + 1],
                )
                match = partial_soft_match(
                    single,
                    self.membership,
                    element_visibility=visibility[mask_id : mask_id + 1],
                    mask_centres=self._mask_centre(positive)[None],
                    token_centres=token_centres,
                    token_scales=token_scales,
                    geometry_weight=self.geometry_weight,
                    appearance_score=(
                        (descriptors[mask_id : mask_id + 1] @ torch.nn.functional.normalize(
                            self.descriptor_sum / self.descriptor_mass[:, None].clamp_min(1e-8),
                            dim=-1,
                            eps=1e-8,
                        ).T + 1.0) * 0.5
                        if descriptors is not None else None
                    ),
                    appearance_weight=self.appearance_weight,
                    null_logit=self.null_logit,
                    temperature=self.temperature,
                )
                best_probability, best_id = match.token_probability[0].max(0)
                overlap = torch.maximum(
                    match.visible_overlap[0, best_id], match.mask_containment[0, best_id]
                )
                null_probability[mask_id] = match.null_probability[0]
                token_probability[mask_id] = best_probability
                if (
                    best_probability > match.null_probability[0]
                    and overlap >= self.minimum_overlap
                ):
                    matched_id = int(best_id)
                    granularity[mask_id] = match.granularity[0]

            if matched_id >= 0:
                token_ids[mask_id] = matched_id
                self._update(
                    matched_id,
                    positive,
                    evidence.quality[mask_id],
                    None if descriptors is None else descriptors[mask_id],
                )
            elif int(parents[mask_id]) < 0:
                new_id = self._create(
                    positive,
                    evidence.quality[mask_id],
                    int(evidence.view_ids[mask_id]),
                    None if descriptors is None else descriptors[mask_id],
                )
                if new_id >= 0:
                    token_ids[mask_id] = new_id
                    token_probability[mask_id] = 1.0
                    null_probability[mask_id] = 0.0
                    granularity[mask_id] = 0
                    created[mask_id] = True

        return BootstrapViewResult(
            token_ids=token_ids,
            token_probability=token_probability,
            null_probability=null_probability,
            granularity=granularity,
            created=created,
        )

    @torch.no_grad()
    def process_batch(
        self,
        evidences: list[ObservedObjectEvidence],
        *,
        element_visibilities: list[torch.Tensor],
        parent_indices: list[torch.Tensor],
        proposal_descriptors: list[torch.Tensor | None] | None = None,
        proposal_identity_ids: list[torch.Tensor] | None = None,
    ) -> list[BootstrapViewResult]:
        """Associate a frozen view batch and commit token updates only once."""

        if not evidences or not (
            len(evidences) == len(element_visibilities) == len(parent_indices)
        ):
            raise ValueError("batch evidence, visibility, and hierarchy lists must align")
        if proposal_descriptors is None:
            proposal_descriptors = [None] * len(evidences)
        if len(proposal_descriptors) != len(evidences):
            raise ValueError("batch descriptor list must align with evidence")
        if proposal_identity_ids is None:
            proposal_identity_ids = [
                torch.full((count,), -1, dtype=torch.long, device=self.element_centres.device)
                for count in [int(value.positive.shape[0]) for value in evidences]
            ]
        if len(proposal_identity_ids) != len(evidences):
            raise ValueError("batch identity list must align with evidence")
        uses_descriptors = [value is not None for value in proposal_descriptors]
        if any(uses_descriptors) and not all(uses_descriptors):
            raise ValueError("a batch must either provide all proposal descriptors or none")

        counts = [int(value.positive.shape[0]) for value in evidences]
        offsets = torch.tensor(
            [0, *torch.tensor(counts).cumsum(0).tolist()],
            dtype=torch.long,
            device=self.element_centres.device,
        )
        positive = torch.cat([value.positive for value in evidences], dim=0)
        negative = torch.cat([value.negative for value in evidences], dim=0)
        unknown = torch.cat([value.unknown for value in evidences], dim=0)
        view_ids = torch.cat([value.view_ids for value in evidences], dim=0)
        quality = torch.cat([value.quality for value in evidences], dim=0)
        visibility = torch.cat([
            torch.as_tensor(value, dtype=torch.float32, device=self.element_centres.device)
            for value in element_visibilities
        ], dim=0)
        local_parents = [
            torch.as_tensor(value, dtype=torch.long, device=self.element_centres.device)
            for value in parent_indices
        ]
        parents = torch.cat(local_parents, dim=0)
        identity_ids = torch.cat([
            torch.as_tensor(value, dtype=torch.long, device=self.element_centres.device)
            for value in proposal_identity_ids
        ], dim=0)
        if identity_ids.shape != (positive.shape[0],):
            raise ValueError("proposal identity IDs must have one value per proposal")
        descriptors = None
        if all(uses_descriptors):
            descriptors = torch.nn.functional.normalize(
                torch.cat([
                    torch.as_tensor(value, dtype=torch.float32, device=self.element_centres.device)
                    for value in proposal_descriptors
                    if value is not None
                ], dim=0),
                dim=-1,
                eps=1e-8,
            )
            if self.descriptor_sum is not None and descriptors.shape[1] != self.descriptor_sum.shape[1]:
                raise ValueError("proposal descriptor dimension changed across batches")
        elif self.descriptor_sum is not None:
            raise ValueError("proposal descriptors cannot be omitted after descriptor token birth")

        total = int(positive.shape[0])
        token_ids = torch.full((total,), -1, dtype=torch.long, device=self.element_centres.device)
        token_probability = torch.zeros(total, device=self.element_centres.device)
        null_probability = torch.ones(total, device=self.element_centres.device)
        granularity = torch.full((total,), 2, dtype=torch.long, device=self.element_centres.device)
        created = torch.zeros(total, dtype=torch.bool, device=self.element_centres.device)
        mass = positive.sum(-1)

        # Existing prototypes remain immutable while every proposal is scored.
        frozen_token_count = self.num_tokens
        if frozen_token_count and total:
            token_centres, token_scales = self._geometry()
            mask_centres = positive @ self.element_centres / mass[:, None].clamp_min(
                self.membership_epsilon
            )
            combined = ObservedObjectEvidence(
                positive, negative, unknown, view_ids, quality
            )
            match = partial_soft_match(
                combined,
                self.membership,
                element_visibility=visibility,
                mask_centres=mask_centres,
                token_centres=token_centres,
                token_scales=token_scales,
                geometry_weight=self.geometry_weight,
                appearance_score=(
                    (descriptors @ torch.nn.functional.normalize(
                        self.descriptor_sum / self.descriptor_mass[:, None].clamp_min(1e-8),
                        dim=-1,
                        eps=1e-8,
                    ).T + 1.0) * 0.5
                    if descriptors is not None else None
                ),
                appearance_weight=self.appearance_weight,
                null_logit=self.null_logit,
                temperature=self.temperature,
            )
            best_probability, best_id = match.token_probability.max(-1)
            rows = torch.arange(total, device=self.element_centres.device)
            best_overlap = torch.maximum(
                match.visible_overlap[rows, best_id], match.mask_containment[rows, best_id]
            )
            accepted = (
                (best_probability > match.null_probability)
                & (best_overlap >= self.minimum_overlap)
                & (mass > self.membership_epsilon)
            )
            token_ids[accepted] = best_id[accepted]
            token_probability[:] = best_probability
            null_probability[:] = match.null_probability
            granularity[accepted] = match.granularity[accepted]

            # Root proposals are object-scale alternatives: one existing token
            # may accept at most one root from a given view.  Auxiliary parts
            # retain the intended many-to-one behavior.
            accepted_roots = torch.where(accepted & (parents < 0))[0]
            for view_id in torch.unique(view_ids[accepted_roots]).tolist():
                view_rows = accepted_roots[view_ids[accepted_roots] == view_id]
                for token_id in torch.unique(token_ids[view_rows]).tolist():
                    token_rows = view_rows[token_ids[view_rows] == token_id]
                    if token_rows.numel() <= 1:
                        continue
                    confidence = token_probability[token_rows]
                    keep = int(token_rows[confidence.argmax()])
                    reject = token_rows[token_rows != keep]
                    token_ids[reject] = -1
                    granularity[reject] = 2

        # Unmatched roots establish batch-local hypotheses.  Each hypothesis is
        # represented by one frozen seed; at most one root from another view may
        # join it, so same-view alternatives can never collapse together.
        root_mask = parents < 0
        candidates = torch.where(
            root_mask & (token_ids < 0) & (mass > self.membership_epsilon)
        )[0]
        groups: list[list[int]] = []
        if candidates.numel():
            candidate_positive = positive[candidates]
            candidate_mass = mass[candidates]
            intersection = candidate_positive @ candidate_positive.T
            union = candidate_mass[:, None] + candidate_mass[None] - intersection
            pair_score = torch.maximum(
                intersection / union.clamp_min(self.membership_epsilon),
                intersection
                / torch.minimum(candidate_mass[:, None], candidate_mass[None]).clamp_min(
                    self.membership_epsilon
                ),
            )
            priority = quality[candidates] * candidate_mass
            order = torch.argsort(priority, descending=True, stable=True)
            available = torch.ones(len(candidates), dtype=torch.bool, device=positive.device)
            for seed_position in order.tolist():
                if not bool(available[seed_position]):
                    continue
                seed = int(candidates[seed_position])
                group = [seed]
                available[seed_position] = False
                ranked = torch.argsort(
                    pair_score[seed_position], descending=True, stable=True
                ).tolist()
                used_views = {int(view_ids[seed])}
                for candidate_position in ranked:
                    if not bool(available[candidate_position]):
                        continue
                    candidate = int(candidates[candidate_position])
                    candidate_view = int(view_ids[candidate])
                    if candidate_view in used_views:
                        continue
                    if float(pair_score[seed_position, candidate_position]) < self.batch_birth_overlap:
                        break
                    seed_view_positions = torch.where(
                        view_ids[candidates] == int(view_ids[seed])
                    )[0]
                    reciprocal = seed_view_positions[
                        pair_score[candidate_position, seed_view_positions].argmax()
                    ]
                    if int(reciprocal) != seed_position:
                        continue
                    group.append(candidate)
                    used_views.add(candidate_view)
                    available[candidate_position] = False
                groups.append(group)

            # Preserve the complete geometry-only partition, then merge only
            # tokens supported by sealed identity edges.  A merge is rejected
            # when it would place two root alternatives from one view together.
            row_to_group = {
                row: group_index
                for group_index, group in enumerate(groups)
                for row in group
            }
            group_parent = list(range(len(groups)))
            group_views = [{int(view_ids[row]) for row in group} for group in groups]

            def find_group(value: int) -> int:
                while group_parent[value] != value:
                    group_parent[value] = group_parent[group_parent[value]]
                    value = group_parent[value]
                return value

            for identity_id in sorted(set(
                int(identity_ids[row]) for row in candidates.tolist()
                if int(identity_ids[row]) >= 0
            )):
                identity_groups = sorted({
                    find_group(row_to_group[row])
                    for row in candidates.tolist()
                    if int(identity_ids[row]) == identity_id
                })
                if not identity_groups:
                    continue
                anchor = identity_groups[0]
                for other in identity_groups[1:]:
                    anchor, other = find_group(anchor), find_group(other)
                    if anchor == other or group_views[anchor] & group_views[other]:
                        continue
                    keep, remove = min(anchor, other), max(anchor, other)
                    group_parent[remove] = keep
                    group_views[keep] |= group_views[remove]
                    anchor = keep

            merged: dict[int, list[int]] = {}
            for group_index, group in enumerate(groups):
                merged.setdefault(find_group(group_index), []).extend(group)
            groups = [merged[key] for key in sorted(merged)]
            for group_index, group in enumerate(groups):
                group_tensor = torch.tensor(group, dtype=torch.long, device=positive.device)
                token_ids[group_tensor] = frozen_token_count + group_index
                token_probability[group_tensor] = 1.0
                null_probability[group_tensor] = 0.0
                granularity[group_tensor] = 0
                created[group[0]] = True

        # A null part may inherit a token only through its explicit SAM parent
        # chain within the same frozen view.
        for view_index, count in enumerate(counts):
            start = int(offsets[view_index])
            local_parent = local_parents[view_index]
            for local_id in range(count):
                global_id = start + local_id
                if token_ids[global_id] >= 0 or int(local_parent[local_id]) < 0:
                    continue
                parent = int(local_parent[local_id])
                visited = set()
                while parent >= 0 and parent not in visited:
                    visited.add(parent)
                    parent_global = start + parent
                    if token_ids[parent_global] >= 0:
                        token_ids[global_id] = token_ids[parent_global]
                        token_probability[global_id] = token_probability[parent_global]
                        null_probability[global_id] = 0.0
                        granularity[global_id] = 1
                        break
                    parent = int(local_parent[parent])

        # Only now may proposal evidence alter prototypes.
        final_token_count = frozen_token_count + len(groups)
        for token_id in range(final_token_count):
            rows = torch.where(token_ids == token_id)[0]
            if not rows.numel():
                continue
            weighted = positive[rows] * quality[rows, None].clamp(0, 1)
            aggregate = weighted.amax(0)
            descriptor = None
            if descriptors is not None:
                descriptor_weight = quality[rows] * mass[rows]
                descriptor = (
                    descriptors[rows] * descriptor_weight[:, None]
                ).sum(0) / descriptor_weight.sum().clamp_min(self.membership_epsilon)
            if token_id < frozen_token_count:
                self._update(token_id, aggregate, torch.tensor(1.0, device=positive.device), descriptor)
            else:
                group = groups[token_id - frozen_token_count]
                created_id = self._create(
                    aggregate,
                    torch.tensor(1.0, device=positive.device),
                    int(view_ids[group[0]]),
                    descriptor,
                )
                if created_id != token_id:
                    raise RuntimeError("batch token creation order changed")

        results = []
        for view_index, count in enumerate(counts):
            start, stop = int(offsets[view_index]), int(offsets[view_index + 1])
            results.append(BootstrapViewResult(
                token_ids=token_ids[start:stop],
                token_probability=token_probability[start:stop],
                null_probability=null_probability[start:stop],
                granularity=granularity[start:stop],
                created=created[start:stop],
            ))
        return results
