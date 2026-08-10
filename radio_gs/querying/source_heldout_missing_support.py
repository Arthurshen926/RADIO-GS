"""Source-heldout labels and calibration-free ranking for missing support.

The target label is deliberately computed only from instance responsibility in
held-out source views.  Proposal scores, graph evidence, and the seed instance
are supplied independently; the seed instance must have been inferred from the
complementary (non-heldout) views by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


SEMANTIC_BOUNDARY = 0.6
FEATURE_NAMES = (
    "edge_comembership_reliability",
    "source_observation_count",
    "source_observation_agreement",
    "target_selected_scale_anchor",
    "target_selected_scale_maximum",
    "target_selected_scale_mean",
    "target_selected_scale_median",
    "target_selected_scale_covered_fraction",
    "seed_to_target_median_ratio",
    "log1p_target_region_size",
    "target_visibility",
    "coverage_deficit",
)
RANK_RELIABILITY_FEATURES = (0, 1, 2, 3, 10)
RANK_POTENTIAL_FEATURES = (7, 11)


@dataclass(frozen=True)
class HeldoutMissingSupportLabel:
    evaluable: bool
    seed_instance_id: int
    missing_primitive_count: int
    heldout_visible_mass: float
    heldout_target_mass: float
    heldout_target_mass_fraction: float
    hard_positive: bool
    signed_utility: float


def infer_seed_instance_from_training_views(
    *,
    seed_rows: torch.Tensor,
    training_primitive_instance_mass: torch.Tensor,
) -> int:
    """Return the positive instance dominating a seed in non-heldout views."""

    rows = torch.as_tensor(seed_rows).detach().long().cpu().reshape(-1)
    mass = (
        torch.as_tensor(training_primitive_instance_mass)
        .detach()
        .double()
        .cpu()
    )
    if (
        rows.numel() <= 0
        or mass.ndim != 2
        or mass.shape[1] < 2
        or bool((rows < 0).any())
        or bool((rows >= mass.shape[0]).any())
    ):
        raise ValueError("source-heldout seed-instance inputs differ")
    selected = mass[rows]
    if not bool(torch.isfinite(selected).all()) or bool((selected < 0.0).any()):
        raise ValueError("source-heldout seed-instance mass differs")
    positive = selected[:, 1:].sum(dim=0)
    if float(positive.sum()) <= 0.0:
        return -1
    return int(positive.argmax()) + 1


def heldout_missing_support_label(
    *,
    target_selected_scale_scores: torch.Tensor,
    target_rows: torch.Tensor,
    seed_rows: torch.Tensor,
    training_primitive_instance_mass: torch.Tensor,
    heldout_primitive_instance_mass: torch.Tensor,
    semantic_boundary: float = SEMANTIC_BOUNDARY,
) -> HeldoutMissingSupportLabel:
    """Measure whether filling a target's missing core recovers the seed object.

    The proposal is evaluable only when the seed object is identifiable without
    held-out labels and the held-out views observe nonzero responsibility on at
    least one below-boundary target primitive.  Ties are negative, which keeps
    ambiguous held-out support from becoming a positive completion label.
    """

    scores = (
        torch.as_tensor(target_selected_scale_scores)
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )
    rows = torch.as_tensor(target_rows).detach().long().cpu().reshape(-1)
    heldout = (
        torch.as_tensor(heldout_primitive_instance_mass)
        .detach()
        .double()
        .cpu()
    )
    training = (
        torch.as_tensor(training_primitive_instance_mass)
        .detach()
        .double()
        .cpu()
    )
    boundary = float(semantic_boundary)
    if (
        scores.shape != rows.shape
        or rows.numel() <= 0
        or training.shape != heldout.shape
        or training.ndim != 2
        or training.shape[1] < 2
        or bool((rows < 0).any())
        or bool((rows >= training.shape[0]).any())
        or not bool(torch.isfinite(scores).all())
        or bool((scores < 0.0).any())
        or bool((scores > 1.0).any())
        or not 0.0 < boundary < 1.0
    ):
        raise ValueError("source-heldout missing-support inputs differ")
    seed_instance = infer_seed_instance_from_training_views(
        seed_rows=seed_rows,
        training_primitive_instance_mass=training,
    )
    return heldout_missing_support_label_for_seed_instance(
        target_selected_scale_scores=scores,
        target_rows=rows,
        seed_instance_id=seed_instance,
        heldout_primitive_instance_mass=heldout,
        semantic_boundary=boundary,
    )


def heldout_missing_support_label_for_seed_instance(
    *,
    target_selected_scale_scores: torch.Tensor,
    target_rows: torch.Tensor,
    seed_instance_id: int,
    heldout_primitive_instance_mass: torch.Tensor,
    semantic_boundary: float = SEMANTIC_BOUNDARY,
) -> HeldoutMissingSupportLabel:
    """Label one proposal after its seed identity was frozen from training views."""

    scores = torch.as_tensor(target_selected_scale_scores).detach().float().cpu().reshape(-1)
    rows = torch.as_tensor(target_rows).detach().long().cpu().reshape(-1)
    heldout = torch.as_tensor(heldout_primitive_instance_mass).detach().double().cpu()
    seed_instance = int(seed_instance_id)
    boundary = float(semantic_boundary)
    if (
        scores.shape != rows.shape
        or rows.numel() <= 0
        or heldout.ndim != 2
        or heldout.shape[1] < 2
        or bool((rows < 0).any())
        or bool((rows >= heldout.shape[0]).any())
        or not bool(torch.isfinite(scores).all())
        or bool((scores < 0.0).any())
        or bool((scores > 1.0).any())
        or seed_instance >= heldout.shape[1]
        or not 0.0 < boundary < 1.0
    ):
        raise ValueError("source-heldout frozen-seed label inputs differ")
    missing = scores <= boundary
    missing_count = int(missing.sum())
    if seed_instance < 1 or missing_count <= 0:
        return HeldoutMissingSupportLabel(
            evaluable=False,
            seed_instance_id=seed_instance,
            missing_primitive_count=missing_count,
            heldout_visible_mass=0.0,
            heldout_target_mass=0.0,
            heldout_target_mass_fraction=0.0,
            hard_positive=False,
            signed_utility=0.0,
        )
    selected_rows = heldout[rows[missing]]
    if not bool(torch.isfinite(selected_rows).all()) or bool((selected_rows < 0.0).any()):
        raise ValueError("source-heldout target mass differs")
    selected = selected_rows.sum(dim=0)
    visible = float(selected.sum())
    target_mass = float(selected[seed_instance])
    if visible <= 0.0:
        return HeldoutMissingSupportLabel(
            evaluable=False,
            seed_instance_id=seed_instance,
            missing_primitive_count=missing_count,
            heldout_visible_mass=0.0,
            heldout_target_mass=0.0,
            heldout_target_mass_fraction=0.0,
            hard_positive=False,
            signed_utility=0.0,
        )
    fraction = target_mass / visible
    other = selected.clone()
    other[seed_instance] = 0.0
    hard = target_mass > float(other.max())
    return HeldoutMissingSupportLabel(
        evaluable=True,
        seed_instance_id=seed_instance,
        missing_primitive_count=missing_count,
        heldout_visible_mass=visible,
        heldout_target_mass=target_mass,
        heldout_target_mass_fraction=fraction,
        hard_positive=hard,
        signed_utility=2.0 * fraction - 1.0,
    )


def proposal_feature_vector(
    *,
    edge_comembership_reliability: float,
    source_observation_count: float,
    source_observation_agreement: float,
    target_selected_scale_scores: torch.Tensor,
    target_anchor_score: float,
    seed_median_score: float,
    target_visibility: float,
    semantic_boundary: float = SEMANTIC_BOUNDARY,
) -> torch.Tensor:
    """Build the fixed target feature vector without consulting its label."""

    scores = (
        torch.as_tensor(target_selected_scale_scores)
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )
    scalar = torch.tensor(
        [
            edge_comembership_reliability,
            source_observation_count,
            source_observation_agreement,
            target_anchor_score,
            seed_median_score,
            target_visibility,
            semantic_boundary,
        ],
        dtype=torch.float64,
    )
    if (
        scores.numel() <= 0
        or not bool(torch.isfinite(scores).all())
        or bool((scores < 0.0).any())
        or bool((scores > 1.0).any())
        or not bool(torch.isfinite(scalar).all())
        or not 0.0 <= float(edge_comembership_reliability) <= 1.0
        or float(source_observation_count) < 0.0
        or not 0.0 <= float(source_observation_agreement) <= 1.0
        or not 0.0 <= float(target_anchor_score) <= 1.0
        or not 0.0 <= float(seed_median_score) <= 1.0
        or not 0.0 <= float(target_visibility) <= 1.0
        or not 0.0 < float(semantic_boundary) < 1.0
    ):
        raise ValueError("source-heldout proposal feature inputs differ")
    maximum = float(scores.max())
    mean = float(scores.mean())
    median = float(scores.median())
    covered = float((scores > float(semantic_boundary)).float().mean())
    seed_ratio = median / max(float(seed_median_score), 1e-6)
    excess = ((scores - float(semantic_boundary)) / (1.0 - float(semantic_boundary))).clamp(0.0, 1.0)
    deficit = float((1.0 - excess.mean()) * (1.0 - excess.median()))
    return torch.tensor(
        [
            edge_comembership_reliability,
            source_observation_count,
            source_observation_agreement,
            target_anchor_score,
            maximum,
            mean,
            median,
            covered,
            seed_ratio,
            torch.log1p(torch.tensor(float(scores.numel()))).item(),
            target_visibility,
            deficit,
        ],
        dtype=torch.float32,
    ).contiguous()


def _inclusive_empirical_rank(values: torch.Tensor) -> torch.Tensor:
    """Tie-invariant inclusive empirical CDF in (0, 1]."""

    vector = torch.as_tensor(values).detach().double().cpu().reshape(-1)
    if vector.numel() <= 0 or not bool(torch.isfinite(vector).all()):
        raise ValueError("source-heldout rank vector differs")
    return (vector[:, None] >= vector[None, :]).double().mean(dim=1)


def calibration_free_maximin_selection(
    features: torch.Tensor,
    group_ids: torch.Tensor,
    target_region_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one Pareto-safe candidate per group using only within-group ranks.

    The primary score is the worst percentile across reliability axes.  Mean
    reliability percentile and then missing-support potential break ties.  No
    source or target label chooses a numeric probability threshold.
    """

    values = torch.as_tensor(features).detach().float().cpu()
    groups = torch.as_tensor(group_ids).detach().long().cpu().reshape(-1)
    regions = torch.as_tensor(target_region_indices).detach().long().cpu().reshape(-1)
    if (
        values.ndim != 2
        or values.shape[1] != len(FEATURE_NAMES)
        or groups.shape != (values.shape[0],)
        or regions.shape != groups.shape
        or values.shape[0] <= 0
        or not bool(torch.isfinite(values).all())
        or bool((groups < 0).any())
        or bool((regions < 0).any())
    ):
        raise ValueError("source-heldout ranking inputs differ")
    selected = torch.zeros(values.shape[0], dtype=torch.bool)
    score = torch.zeros(values.shape[0], dtype=torch.float32)
    for group in torch.unique(groups, sorted=True).tolist():
        rows = torch.where(groups == int(group))[0]
        reliability = torch.stack(
            [_inclusive_empirical_rank(values[rows, index]) for index in RANK_RELIABILITY_FEATURES],
            dim=1,
        )
        potential = torch.stack(
            [_inclusive_empirical_rank(values[rows, index]) for index in RANK_POTENTIAL_FEATURES],
            dim=1,
        ).mean(dim=1)
        worst = reliability.amin(dim=1)
        mean = reliability.mean(dim=1)
        # The exported scalar is the primary maximin rank only.  Mean and
        # potential are used lexicographically below, never mixed by arbitrary
        # numerical weights.
        score[rows] = worst.float()
        ordered = sorted(
            range(rows.numel()),
            key=lambda local: (
                -float(worst[local]),
                -float(mean[local]),
                -float(potential[local]),
                int(regions[rows[local]]),
            ),
        )
        selected[rows[ordered[0]]] = True
    return selected.contiguous(), score.contiguous()


__all__ = [
    "FEATURE_NAMES",
    "HeldoutMissingSupportLabel",
    "SEMANTIC_BOUNDARY",
    "calibration_free_maximin_selection",
    "heldout_missing_support_label",
    "heldout_missing_support_label_for_seed_instance",
    "infer_seed_instance_from_training_views",
    "proposal_feature_vector",
]
