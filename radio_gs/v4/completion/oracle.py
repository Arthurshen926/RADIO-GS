"""Scene-agnostic object completion with an oracle token identity.

The model scores an element against every already-observed object token.  It
never receives an integer instance identity: the identity is represented by a
prototype computed only from observed positive elements.  Observed oracle
membership is an explicit input; only unobserved 3-D membership is target-only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PartialObjectMembership:
    """Mutually exclusive positive/negative/unknown evidence for ``E x K``.

    Every carrier element-token pair has exactly one state.  Surfaces outside
    the retained token set are explicit negatives when observed and remain
    unknown when unobserved, allowing a learned null decision.
    """

    positive: torch.Tensor
    negative: torch.Tensor
    unknown: torch.Tensor
    eligible_elements: torch.Tensor

    def __post_init__(self) -> None:
        positive = torch.as_tensor(self.positive, dtype=torch.bool).cpu()
        negative = torch.as_tensor(self.negative, dtype=torch.bool).cpu()
        unknown = torch.as_tensor(self.unknown, dtype=torch.bool).cpu()
        eligible = torch.as_tensor(self.eligible_elements, dtype=torch.bool).cpu()
        if positive.ndim != 2 or positive.shape[1] == 0:
            raise ValueError("partial membership must have shape [E, K] with K > 0")
        if negative.shape != positive.shape or unknown.shape != positive.shape:
            raise ValueError("positive, negative, and unknown states must align")
        if eligible.shape != (positive.shape[0],):
            raise ValueError("eligible_elements must have shape [E]")
        state_count = positive.to(torch.int8) + negative.to(torch.int8) + unknown.to(torch.int8)
        expected = eligible[:, None].expand_as(state_count).to(torch.int8)
        if not torch.equal(state_count, expected):
            raise ValueError("every eligible pair must be exactly positive, negative, or unknown")
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "unknown", unknown)
        object.__setattr__(self, "eligible_elements", eligible)

    @classmethod
    def from_oracle_visibility(
        cls,
        token_index: torch.Tensor,
        visible: torch.Tensor,
        *,
        token_count: int,
        eligible_elements: torch.Tensor | None = None,
    ) -> "PartialObjectMembership":
        """Create partial facts while using labels only to simulate observations."""

        labels = torch.as_tensor(token_index, dtype=torch.long).cpu()
        visible = torch.as_tensor(visible, dtype=torch.bool).cpu()
        if labels.ndim != 1 or visible.shape != labels.shape:
            raise ValueError("token_index and visible must be aligned vectors")
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        eligible = (
            torch.ones_like(labels, dtype=torch.bool)
            if eligible_elements is None
            else torch.as_tensor(eligible_elements, dtype=torch.bool).cpu()
        )
        if eligible.shape != labels.shape:
            raise ValueError("eligible_elements must align with token_index")
        if bool((labels[eligible] >= token_count).any()):
            raise ValueError("token_index exceeds token_count")
        token_ids = torch.arange(token_count)[None, :]
        belongs = labels[:, None] == token_ids
        observed = visible & eligible
        positive = observed[:, None] & belongs
        negative = observed[:, None] & ~belongs
        unknown = (eligible & ~visible)[:, None].expand_as(positive)
        return cls(positive, negative, unknown, eligible)

    @property
    def element_is_observed(self) -> torch.Tensor:
        return self.eligible_elements & ~self.unknown.any(-1)


@dataclass(frozen=True)
class TokenContext:
    centre: torch.Tensor
    scale: torch.Tensor
    feature_prototype: torch.Tensor
    observed_mass: torch.Tensor
    local_positive: torch.Tensor
    local_negative: torch.Tensor

    def __post_init__(self) -> None:
        centre = torch.as_tensor(self.centre, dtype=torch.float32).cpu()
        scale = torch.as_tensor(self.scale, dtype=torch.float32).cpu()
        prototype = torch.as_tensor(self.feature_prototype, dtype=torch.float32).cpu()
        mass = torch.as_tensor(self.observed_mass, dtype=torch.float32).cpu()
        local_positive = torch.as_tensor(self.local_positive, dtype=torch.float32).cpu()
        local_negative = torch.as_tensor(self.local_negative, dtype=torch.float32).cpu()
        if centre.ndim != 2 or centre.shape[1] != 3 or scale.shape != centre.shape:
            raise ValueError("token centre and scale must have shape [K, 3]")
        if prototype.ndim != 2 or prototype.shape[0] != centre.shape[0]:
            raise ValueError("feature prototypes must have shape [K, F]")
        if mass.shape != (centre.shape[0],):
            raise ValueError("observed mass must have shape [K]")
        expected_local = (local_positive.shape[0], centre.shape[0])
        if local_positive.ndim != 2 or local_positive.shape != expected_local:
            raise ValueError("local positive evidence must have shape [E, K]")
        if local_negative.shape != local_positive.shape:
            raise ValueError("local negative evidence must align")
        values = (centre, scale, prototype, mass, local_positive, local_negative)
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError("token context must be finite")
        if bool((scale <= 0).any()) or bool((mass <= 0).any()):
            raise ValueError("every token needs positive scale and observed mass")
        object.__setattr__(self, "centre", centre)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "feature_prototype", prototype)
        object.__setattr__(self, "observed_mass", mass)
        object.__setattr__(self, "local_positive", local_positive.clamp(0, 1))
        object.__setattr__(self, "local_negative", local_negative.clamp(0, 1))


def _neighbor_fraction(values: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    element_count = values.shape[0]
    edges = torch.as_tensor(edge_index, dtype=torch.long).cpu()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, A]")
    result = torch.zeros_like(values, dtype=torch.float32)
    degree = torch.zeros(element_count, dtype=torch.float32)
    if edges.shape[1]:
        source, destination = edges
        result.index_add_(0, destination, values[source].float())
        degree.index_add_(0, destination, torch.ones_like(destination, dtype=torch.float32))
    return result / degree.clamp_min(1)[:, None]


def build_token_context(
    centres: torch.Tensor,
    local_features: torch.Tensor,
    partial: PartialObjectMembership,
    edge_index: torch.Tensor,
    *,
    minimum_scale: float,
) -> TokenContext:
    centres = torch.as_tensor(centres, dtype=torch.float32).cpu()
    local_features = torch.as_tensor(local_features, dtype=torch.float32).cpu()
    if centres.ndim != 2 or centres.shape[1] != 3:
        raise ValueError("centres must have shape [E, 3]")
    if local_features.ndim != 2 or local_features.shape[0] != centres.shape[0]:
        raise ValueError("local_features must have shape [E, F]")
    if partial.positive.shape[0] != centres.shape[0]:
        raise ValueError("partial membership and elements do not align")
    if not torch.isfinite(centres).all() or not torch.isfinite(local_features).all():
        raise ValueError("completion inputs must be finite")
    if minimum_scale <= 0:
        raise ValueError("minimum_scale must be positive")
    weight = partial.positive.float()
    mass = weight.sum(0)
    if bool((mass <= 0).any()):
        raise ValueError("oracle completion requires every token to have observed positives")
    centre = weight.T @ centres / mass[:, None]
    feature = weight.T @ local_features / mass[:, None]
    centred = centres[:, None, :] - centre[None, :, :]
    variance = (weight[..., None] * centred.square()).sum(0) / mass[:, None]
    scale = variance.sqrt().clamp_min(float(minimum_scale))
    return TokenContext(
        centre=centre,
        scale=scale,
        feature_prototype=feature,
        observed_mass=mass,
        local_positive=_neighbor_fraction(partial.positive, edge_index),
        local_negative=_neighbor_fraction(partial.negative, edge_index),
    )


def build_pair_features(
    centres: torch.Tensor,
    local_features: torch.Tensor,
    context: TokenContext,
    element_indices: torch.Tensor,
    *,
    minimum_scale: float,
) -> torch.Tensor:
    """Build target-free element/token features with shape ``[N, K, D]``."""

    centres = torch.as_tensor(centres, dtype=torch.float32).cpu()
    local_features = torch.as_tensor(local_features, dtype=torch.float32).cpu()
    indices = torch.as_tensor(element_indices, dtype=torch.long).cpu()
    if indices.ndim != 1 or (indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= centres.shape[0])):
        raise ValueError("element_indices are outside the carrier domain")
    if local_features.shape[0] != centres.shape[0]:
        raise ValueError("local features and centres do not align")
    element = centres[indices, None, :]
    relative = element - context.centre[None, :, :]
    normalized = (
        relative / context.scale[None, :, :].clamp_min(minimum_scale)
    ).clamp(-16, 16)
    feature = local_features[indices, None, :].expand(-1, context.centre.shape[0], -1)
    prototype = context.feature_prototype[None, :, :].expand_as(feature)
    log_scale = torch.log(context.scale.clamp_min(minimum_scale) / minimum_scale)[None].expand(indices.numel(), -1, -1)
    log_mass = torch.log1p(context.observed_mass)[None, :, None].expand(indices.numel(), -1, -1)
    return torch.cat(
        (
            normalized,
            normalized.abs(),
            normalized.square().sum(-1, keepdim=True).sqrt(),
            log_scale,
            feature,
            prototype,
            feature - prototype,
            feature * prototype,
            context.local_positive[indices, :, None],
            context.local_negative[indices, :, None],
            log_mass,
        ),
        dim=-1,
    )


def build_feature_cosine_similarity(
    local_features: torch.Tensor,
    context: TokenContext,
    element_indices: torch.Tensor,
    *,
    feature_start: int,
    feature_stop: int,
) -> torch.Tensor:
    """Return an explicit element/token cosine without reading target membership.

    A zero vector has similarity zero.  The sealed local-feature contract already
    carries source availability separately, so this convention makes the
    residual exactly inactive for elements without a source descriptor.
    """

    local = torch.as_tensor(local_features, dtype=torch.float32).cpu()
    indices = torch.as_tensor(element_indices, dtype=torch.long).cpu()
    if local.ndim != 2 or local.shape[1] != context.feature_prototype.shape[1]:
        raise ValueError("local features and token prototypes must align")
    if indices.ndim != 1 or (
        indices.numel()
        and (int(indices.min()) < 0 or int(indices.max()) >= local.shape[0])
    ):
        raise ValueError("element_indices are outside the local-feature domain")
    if not 0 <= feature_start < feature_stop <= local.shape[1]:
        raise ValueError("cosine feature slice is outside the local-feature layout")
    element = local[indices, feature_start:feature_stop]
    prototype = context.feature_prototype[:, feature_start:feature_stop]
    numerator = element @ prototype.T
    denominator = (
        torch.linalg.vector_norm(element, dim=-1, keepdim=True)
        * torch.linalg.vector_norm(prototype, dim=-1)[None, :]
    )
    similarity = torch.where(
        denominator > 1e-12,
        numerator / denominator.clamp_min(1e-12),
        torch.zeros_like(numerator),
    ).clamp(-1, 1)
    if not torch.isfinite(similarity).all():
        raise ValueError("explicit feature cosine must be finite")
    return similarity


class OracleIdentityCompletionMLP(nn.Module):
    """Shared pair scorer; token count and identity are scene-specific inputs."""

    token_cardinality_normalization = "none_raw_learned_null_logit"

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int = 128,
        dropout: float = 0.1,
        *,
        explicit_similarity_residual: bool = False,
        availability_conditioned_experts: bool = False,
    ) -> None:
        super().__init__()
        if input_dimension <= 0 or hidden_dimension <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid completion MLP dimensions/dropout")
        self.input_dimension = int(input_dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.dropout = float(dropout)
        self.explicit_similarity_residual = bool(explicit_similarity_residual)
        self.availability_conditioned_experts = bool(availability_conditioned_experts)
        if self.explicit_similarity_residual and self.availability_conditioned_experts:
            raise ValueError("similarity residual and dual experts are separate ablations")

        def scorer() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dimension, hidden_dimension),
                nn.LayerNorm(hidden_dimension),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dimension, hidden_dimension),
                nn.GELU(),
                nn.Linear(hidden_dimension, 1),
            )

        # ``network`` is the only scorer for the baseline and the never-visible
        # scorer for the deterministic dual-expert ablation.
        self.network = scorer()
        self.visible_network = (
            scorer() if self.availability_conditioned_experts else None
        )
        self.null_logit = nn.Parameter(torch.zeros(()))
        if self.explicit_similarity_residual:
            # softplus(0.5413...) == 1.0: start as a unit, positive-only
            # evidence residual while leaving the learned MLP unrestricted.
            self.similarity_log_scale = nn.Parameter(torch.tensor(0.5413248546))
        else:
            self.register_parameter("similarity_log_scale", None)

    def forward(
        self,
        pair_features: torch.Tensor,
        *,
        source_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pair_features.ndim < 2 or pair_features.shape[-1] != self.input_dimension:
            raise ValueError("pair features have the wrong final dimension")
        unseen = self.network(pair_features).squeeze(-1)
        if not self.availability_conditioned_experts:
            if source_available is not None:
                raise ValueError("source availability was provided to a single scorer")
            return unseen
        if source_available is None:
            raise ValueError("dual experts require sealed source availability")
        available = torch.as_tensor(
            source_available, dtype=torch.bool, device=unseen.device
        )
        if available.shape != unseen.shape[:-1]:
            raise ValueError("source availability must have one value per element")
        visible = self.visible_network(pair_features).squeeze(-1)
        return torch.where(available[..., None], visible, unseen)

    def categorical_logits(
        self,
        pair_features: torch.Tensor,
        *,
        explicit_similarity: torch.Tensor | None = None,
        source_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_logits = self(pair_features, source_available=source_available)
        if self.explicit_similarity_residual:
            if explicit_similarity is None or explicit_similarity.shape != token_logits.shape:
                raise ValueError(
                    "explicit similarity residual must align with token logits"
                )
            if not torch.isfinite(explicit_similarity).all():
                raise ValueError("explicit similarity residual must be finite")
            token_logits = token_logits + torch.nn.functional.softplus(
                self.similarity_log_scale
            ) * explicit_similarity
        elif explicit_similarity is not None:
            raise ValueError("explicit similarity was provided to a plain MLP scorer")
        token_count = int(token_logits.shape[-1])
        if token_count <= 0:
            raise ValueError("categorical completion requires at least one token")
        # The cardinality-normalized null ablation was rejected scene-macro.  Keep
        # the accepted v3-fixed contract: one learned raw null logit, shared across
        # elements and token cardinalities.
        null = self.null_logit.expand(*token_logits.shape[:-1], 1)
        return torch.cat((token_logits, null), dim=-1)


def complete_unknown_only(
    partial: PartialObjectMembership,
    unknown_probability: torch.Tensor,
    *,
    unknown_null_probability: torch.Tensor | None = None,
    completion_confidence_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp factual evidence and reserve null mass for every completion."""

    probability = torch.as_tensor(unknown_probability, dtype=torch.float32).cpu()
    if probability.shape != partial.positive.shape:
        raise ValueError("unknown_probability must have shape [E, K]")
    if not torch.isfinite(probability).all() or bool((probability < 0).any()):
        raise ValueError("unknown completion probabilities must be finite and non-negative")
    if not 0 < completion_confidence_cap < 1:
        raise ValueError("completion confidence cap must be strictly between zero and one")
    null_probability = (
        torch.zeros(probability.shape[0], dtype=torch.float32)
        if unknown_null_probability is None
        else torch.as_tensor(unknown_null_probability, dtype=torch.float32).cpu()
    )
    if null_probability.shape != (probability.shape[0],):
        raise ValueError("unknown_null_probability must have shape [E]")
    if not torch.isfinite(null_probability).all() or bool((null_probability < 0).any()):
        raise ValueError("unknown null probabilities must be finite and non-negative")
    unknown_rows = partial.unknown.any(-1)
    if bool(unknown_rows.any()):
        row_sum = probability[unknown_rows].sum(-1) + null_probability[unknown_rows]
        if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4):
            raise ValueError("unknown completion probabilities must form a token-plus-null simplex")
    membership = torch.zeros_like(probability)
    membership[partial.positive] = 1.0
    membership = torch.where(
        partial.unknown,
        probability.clamp(0, 1) * completion_confidence_cap,
        membership,
    )
    membership = torch.where(partial.negative, torch.zeros_like(membership), membership)
    null = torch.zeros(membership.shape[0])
    observed_null = partial.element_is_observed & ~partial.positive.any(-1)
    null[observed_null] = 1.0
    null[unknown_rows] = 1.0 - membership[unknown_rows].sum(-1)
    null[~partial.eligible_elements] = 1.0
    return membership, null


def _masked_soft_iou(
    membership: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    prediction = membership * mask[:, None]
    masked_target = target * mask[:, None]
    intersection = (prediction * masked_target).sum(0)
    union = prediction.sum(0) + masked_target.sum(0) - intersection
    # Include false-positive-only tokens, but do not invent a score for a token
    # that is absent from both prediction and target in this cohort.
    valid = union > 0
    return (
        float((intersection[valid] / union[valid].clamp_min(1e-12)).mean())
        if bool(valid.any()) else 0.0
    )


def _categorical_subset_metrics(
    membership: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    categorical: torch.Tensor,
    *,
    null_index: int,
    subset: torch.Tensor,
) -> dict[str, float | int]:
    target_object = subset & (labels >= 0)
    target_null = subset & (labels < 0)
    predicted_token = subset & (categorical != null_index)
    predicted_null = subset & (categorical == null_index)
    correct_token = predicted_token & target_object & (categorical == labels)
    wrong_token_on_object = predicted_token & target_object & (categorical != labels)
    token_on_null = predicted_token & target_null
    null_on_object = predicted_null & target_object
    correct_null = predicted_null & target_null

    element_count = int(subset.sum())
    target_object_count = int(target_object.sum())
    target_null_count = int(target_null.sum())
    predicted_token_count = int(predicted_token.sum())
    predicted_null_count = int(predicted_null.sum())
    correct_token_count = int(correct_token.sum())
    assigned_object_count = int((predicted_token & target_object).sum())
    correct_null_count = int(correct_null.sum())
    if (
        correct_token_count
        + int(wrong_token_on_object.sum())
        + int(token_on_null.sum())
        + int(null_on_object.sum())
        + correct_null_count
        != element_count
    ):
        raise RuntimeError("categorical completion confusion does not partition the cohort")
    return {
        "soft_3d_miou": _masked_soft_iou(membership, target, subset),
        "assignment_precision": (
            correct_token_count / predicted_token_count if predicted_token_count else 0.0
        ),
        "retained_object_coverage": (
            assigned_object_count / target_object_count if target_object_count else 1.0
        ),
        "correct_assignment_recall": (
            correct_token_count / target_object_count if target_object_count else 1.0
        ),
        "assigned_object_top1_accuracy": (
            correct_token_count / assigned_object_count if assigned_object_count else 0.0
        ),
        "retained_set_null_recall": (
            correct_null_count / target_null_count if target_null_count else 1.0
        ),
        "element_count": element_count,
        "retained_object_count": target_object_count,
        "retained_set_null_count": target_null_count,
        "predicted_token_count": predicted_token_count,
        "predicted_null_count": predicted_null_count,
        "assigned_object_count": assigned_object_count,
        "correct_token_count": correct_token_count,
        "wrong_token_on_object_count": int(wrong_token_on_object.sum()),
        "token_on_null_count": int(token_on_null.sum()),
        "null_on_object_count": int(null_on_object.sum()),
        "correct_null_count": correct_null_count,
    }


def completion_metrics(
    membership: torch.Tensor,
    partial: PartialObjectMembership,
    target_token_index: torch.Tensor,
    *,
    null_probability: torch.Tensor | None = None,
    assignment_threshold: float = 0.5,
    unknown_strata: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, float | int]:
    membership = torch.as_tensor(membership, dtype=torch.float32).cpu()
    labels = torch.as_tensor(target_token_index, dtype=torch.long).cpu()
    if membership.shape != partial.positive.shape or labels.shape != (membership.shape[0],):
        raise ValueError("completion metric inputs do not align")
    if abs(float(assignment_threshold) - 0.5) > 1e-12:
        raise ValueError("the legacy assignment diagnostic is frozen at 0.5")
    eligible = partial.eligible_elements
    eligible_membership = membership[eligible]
    if not torch.isfinite(eligible_membership).all() or bool(
        ((eligible_membership < 0) | (eligible_membership > 1)).any()
    ):
        raise ValueError("eligible completion memberships must be finite probabilities")
    object_surface = (labels >= 0) & eligible
    if not bool(object_surface.any()):
        raise ValueError("completion metrics require retained object targets")
    target = torch.zeros_like(membership)
    target[object_surface, labels[object_surface]] = 1.0
    evaluated_membership = torch.where(
        eligible[:, None], membership, torch.zeros_like(membership)
    )
    intersection = (evaluated_membership * target).sum(0)
    union = evaluated_membership.sum(0) + target.sum(0) - intersection
    valid_token = target.sum(0) > 0
    soft_iou = intersection[valid_token] / union[valid_token].clamp_min(1e-12)
    object_prediction = evaluated_membership[object_surface]
    top1 = object_prediction.argmax(-1)
    top1_accuracy = (top1 == labels[object_surface]).float().mean()
    row_mass = evaluated_membership.sum(-1)
    nonzero = row_mass > 0
    concentration_values = evaluated_membership.max(-1).values / row_mass.clamp_min(1e-12)
    observed = partial.element_is_observed[object_surface]
    background = (labels < 0) & eligible
    background_score = evaluated_membership[background].max(-1).values if bool(background.any()) else torch.zeros(1)
    if null_probability is None:
        null = 1.0 - membership.sum(-1).clamp(0, 1)
    else:
        null = torch.as_tensor(null_probability, dtype=torch.float32).cpu()
        if null.shape != labels.shape:
            raise ValueError("null_probability must have shape [E]")
    eligible_null = null[eligible]
    if not torch.isfinite(eligible_null).all() or bool(
        ((eligible_null < 0) | (eligible_null > 1)).any()
    ):
        raise ValueError("eligible completion null values must be finite probabilities")
    eligible_simplex_mass = evaluated_membership[eligible].sum(-1) + eligible_null
    if not torch.allclose(
        eligible_simplex_mass, torch.ones_like(eligible_simplex_mass), atol=1e-4
    ):
        raise ValueError("eligible completion rows must form one K-plus-null simplex")
    null = torch.where(eligible, null, torch.ones_like(null))
    categorical_scores = torch.cat((evaluated_membership, null[:, None]), -1)
    categorical = categorical_scores.argmax(-1)
    null_index = membership.shape[1]
    target_categorical = labels.clone()
    target_categorical[background] = null_index
    unknown_all = partial.unknown.any(-1) & eligible
    unknown_object = unknown_all & object_surface
    maximum_token = evaluated_membership.max(-1).values
    unknown_summary = _categorical_subset_metrics(
        evaluated_membership,
        target,
        labels,
        categorical,
        null_index=null_index,
        subset=unknown_all,
    )

    # Preserve the historical thresholded numbers as explicitly named
    # diagnostics.  They are not the primary K+null decision and must not gate.
    predicted_token = categorical != null_index
    assigned_unknown_at_0p5 = unknown_all & predicted_token & (maximum_token >= 0.5)
    correct_unknown_at_0p5 = (
        assigned_unknown_at_0p5 & object_surface & (categorical == labels)
    )
    assigned_unknown_object_at_0p5 = assigned_unknown_at_0p5 & object_surface
    assigned_count_at_0p5 = int(assigned_unknown_at_0p5.sum())
    assigned_object_count_at_0p5 = int(assigned_unknown_object_at_0p5.sum())
    correct_count_at_0p5 = int(correct_unknown_at_0p5.sum())
    full_categorical_accuracy = float(
        (categorical[eligible] == target_categorical[eligible]).float().mean()
    )
    correct_mass = evaluated_membership[object_surface, labels[object_surface]].sum()
    total_token_mass = evaluated_membership[eligible].sum()
    target_mass_precision = float(correct_mass / total_token_mass.clamp_min(1e-12))
    unknown_correct_mass = (
        evaluated_membership[unknown_object, labels[unknown_object]].sum()
        if bool(unknown_object.any()) else torch.tensor(0.0)
    )
    unknown_token_mass = evaluated_membership[unknown_all].sum()
    unknown_target_mass_precision = (
        float(unknown_correct_mass / unknown_token_mass.clamp_min(1e-12))
        if float(unknown_token_mass) > 0 else 0.0
    )
    positive_error = (
        (membership[partial.positive] - 1).abs().max()
        if bool(partial.positive.any()) else torch.tensor(0.0)
    )
    negative_error = (
        membership[partial.negative].abs().max()
        if bool(partial.negative.any()) else torch.tensor(0.0)
    )
    result: dict[str, float | int] = {
        "soft_3d_miou": float(soft_iou.mean()),
        "unknown_only_soft_3d_miou": float(unknown_summary["soft_3d_miou"]),
        "full_object_token_top1_accuracy": float(top1_accuracy),
        "full_k_plus_null_categorical_accuracy": full_categorical_accuracy,
        "token_probability_concentration": (
            float(concentration_values[nonzero].mean()) if bool(nonzero.any()) else 0.0
        ),
        "target_aware_token_mass_precision": target_mass_precision,
        "unknown_target_aware_token_mass_precision": unknown_target_mass_precision,
        "unknown_assignment_precision": float(unknown_summary["assignment_precision"]),
        "unknown_retained_object_coverage": float(unknown_summary["retained_object_coverage"]),
        "unknown_correct_assignment_recall": float(unknown_summary["correct_assignment_recall"]),
        "assigned_unknown_object_top1_accuracy": float(unknown_summary["assigned_object_top1_accuracy"]),
        "unknown_retained_set_null_recall": float(unknown_summary["retained_set_null_recall"]),
        "unknown_assignment_precision_at_0p5": (
            correct_count_at_0p5 / assigned_count_at_0p5
            if assigned_count_at_0p5 else 0.0
        ),
        "unknown_retained_object_coverage_at_0p5": (
            assigned_object_count_at_0p5 / int(unknown_object.sum())
            if bool(unknown_object.any()) else 1.0
        ),
        "unknown_correct_assignment_recall_at_0p5": (
            correct_count_at_0p5 / int(unknown_object.sum())
            if bool(unknown_object.any()) else 1.0
        ),
        "assigned_unknown_object_top1_accuracy_at_0p5": (
            correct_count_at_0p5 / assigned_object_count_at_0p5
            if assigned_object_count_at_0p5 else 0.0
        ),
        "known_element_fraction": float(observed.float().mean()),
        "unknown_element_count": int(unknown_summary["element_count"]),
        "unknown_retained_object_count": int(unknown_summary["retained_object_count"]),
        "unknown_retained_set_null_count": int(unknown_summary["retained_set_null_count"]),
        "assigned_unknown_count": int(unknown_summary["predicted_token_count"]),
        "assigned_unknown_count_at_0p5": assigned_count_at_0p5,
        "unknown_predicted_token_count": int(unknown_summary["predicted_token_count"]),
        "unknown_predicted_null_count": int(unknown_summary["predicted_null_count"]),
        "unknown_assigned_object_count": int(unknown_summary["assigned_object_count"]),
        "unknown_correct_token_count": int(unknown_summary["correct_token_count"]),
        "unknown_wrong_token_on_object_count": int(unknown_summary["wrong_token_on_object_count"]),
        "unknown_token_on_null_count": int(unknown_summary["token_on_null_count"]),
        "unknown_null_on_object_count": int(unknown_summary["null_on_object_count"]),
        "unknown_correct_null_count": int(unknown_summary["correct_null_count"]),
        "retained_set_false_positive_mean": float(background_score.mean()),
        "retained_set_false_positive_max": float(background_score.max()),
        "positive_clamp_max_error": float(positive_error),
        "negative_clamp_max_error": float(negative_error),
    }
    if unknown_strata is not None:
        allowed = {"visible_but_unmasked", "never_visible"}
        provided = set(unknown_strata)
        if provided != allowed:
            raise ValueError(
                "unknown completion strata must be exactly visible_but_unmasked and never_visible"
            )
        occupied = torch.zeros_like(unknown_all)
        for name, values in unknown_strata.items():
            subset = torch.as_tensor(values, dtype=torch.bool).cpu()
            if subset.shape != labels.shape:
                raise ValueError(f"unknown stratum {name!r} must have shape [E]")
            if bool((subset & ~unknown_all).any()):
                raise ValueError(f"unknown stratum {name!r} contains a non-unknown element")
            if bool((subset & occupied).any()):
                raise ValueError("unknown completion strata must be disjoint")
            occupied |= subset
            summary = _categorical_subset_metrics(
                evaluated_membership,
                target,
                labels,
                categorical,
                null_index=null_index,
                subset=subset,
            )
            for key, value in summary.items():
                result[f"{name}_{key}"] = value
        if not torch.equal(occupied, unknown_all):
            raise ValueError("unknown completion strata must partition every unknown element")
    return result
