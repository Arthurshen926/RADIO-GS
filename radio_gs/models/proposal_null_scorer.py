"""Source-trainable, set-equivariant proposal/null probability scorer."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ProposalNullScores:
    proposal_logits: torch.Tensor
    null_logits: torch.Tensor
    joint_probability: torch.Tensor


class ProposalNullScorer(nn.Module):
    """Score a variable proposal set without proposal-count prior inflation.

    The same proposal network is applied to every proposal/query pair.  A
    DeepSets summary supplies query-level evidence to the explicit null head.
    Both final heads are zero initialized.  At epoch zero, null has probability
    1/2 and the complete valid proposal cohort shares the remaining 1/2,
    independent of proposal count.
    """

    def __init__(
        self,
        feature_median: torch.Tensor,
        feature_robust_scale: torch.Tensor,
        *,
        scene_feature_dimension: int,
        hidden_dimension: int = 64,
    ) -> None:
        super().__init__()
        median = torch.as_tensor(feature_median).detach().float().reshape(-1)
        scale = torch.as_tensor(feature_robust_scale).detach().float().reshape(-1)
        if (
            median.numel() <= 0
            or scale.shape != median.shape
            or not bool(torch.isfinite(median).all())
            or not bool(torch.isfinite(scale).all())
            or bool((scale <= 0).any())
            or int(scene_feature_dimension) < 0
            or int(hidden_dimension) <= 0
        ):
            raise ValueError("proposal/null scorer dimensions differ")
        self.register_buffer("feature_median", median.contiguous())
        self.register_buffer("feature_robust_scale", scale.contiguous())
        self.scene_feature_dimension = int(scene_feature_dimension)
        self.encoder = nn.Sequential(
            nn.Linear(int(median.numel()), int(hidden_dimension)),
            nn.GELU(),
            nn.Linear(int(hidden_dimension), int(hidden_dimension)),
            nn.GELU(),
        )
        self.proposal_head = nn.Linear(int(hidden_dimension), 1)
        self.null_head = nn.Linear(
            2 * int(hidden_dimension) + self.scene_feature_dimension, 1
        )
        nn.init.zeros_(self.proposal_head.weight)
        nn.init.zeros_(self.proposal_head.bias)
        nn.init.zeros_(self.null_head.weight)
        nn.init.zeros_(self.null_head.bias)

    def forward(
        self,
        proposal_features: torch.Tensor,
        proposal_valid: torch.Tensor,
        scene_features: torch.Tensor,
    ) -> ProposalNullScores:
        values = torch.as_tensor(proposal_features)
        valid = torch.as_tensor(proposal_valid, device=values.device).bool()
        scene = torch.as_tensor(
            scene_features, device=values.device, dtype=values.dtype
        )
        if (
            values.ndim != 3
            or int(values.shape[2]) != int(self.feature_median.numel())
            or valid.shape != values.shape[:2]
            or scene.shape
            != (values.shape[1], self.scene_feature_dimension)
            or not values.is_floating_point()
            or not bool(torch.isfinite(values).all())
            or not bool(torch.isfinite(scene).all())
            or not bool(valid.any(dim=0).all())
        ):
            raise ValueError("proposal/null scorer input axes differ")
        normalized = (
            values - self.feature_median.to(values)[None, None]
        ) / self.feature_robust_scale.to(values)[None, None]
        encoded = self.encoder(normalized)
        masked = encoded.masked_fill(~valid[..., None], 0.0)
        counts = valid.sum(dim=0).clamp_min(1)
        mean = masked.sum(dim=0) / counts[:, None]
        maximum = encoded.masked_fill(~valid[..., None], -torch.inf).amax(dim=0)
        null_logits = self.null_head(torch.cat((mean, maximum, scene), dim=-1)).squeeze(-1)

        raw_proposal = self.proposal_head(encoded).squeeze(-1)
        # A count-corrected exchangeable prior: duplicating an uninformative
        # proposal cannot steal probability mass from the null branch.
        proposal_logits = raw_proposal - counts.float().log()[None]
        proposal_logits = proposal_logits.masked_fill(~valid, -torch.inf)
        joint_logits = torch.cat((null_logits[None], proposal_logits), dim=0)
        probability = torch.softmax(joint_logits, dim=0)
        return ProposalNullScores(
            proposal_logits=proposal_logits,
            null_logits=null_logits,
            joint_probability=probability,
        )


def proposal_null_proper_loss(
    joint_probability: torch.Tensor,
    target_distribution: torch.Tensor,
    evaluable: torch.Tensor,
    *,
    brier_weight: float = 1.0,
) -> torch.Tensor:
    """Categorical log score plus Brier score; unknown queries are ignored."""

    probability = torch.as_tensor(joint_probability)
    target = torch.as_tensor(
        target_distribution, device=probability.device, dtype=probability.dtype
    )
    known = torch.as_tensor(evaluable, device=probability.device).bool()
    if (
        probability.ndim != 2
        or target.shape != probability.shape
        or known.shape != (probability.shape[1],)
        or not bool(known.any())
        or not bool(torch.isfinite(probability).all())
        or not bool(torch.isfinite(target).all())
        or bool((probability < 0).any())
        or bool((target < 0).any())
        or not torch.allclose(probability.sum(dim=0), torch.ones_like(known, dtype=probability.dtype))
        or not torch.allclose(target[:, known].sum(dim=0), torch.ones(int(known.sum()), device=target.device, dtype=target.dtype))
        or not math.isfinite(float(brier_weight))
        or float(brier_weight) < 0
    ):
        raise ValueError("proposal/null proper-loss inputs differ")
    selected_probability = probability[:, known].clamp_min(1e-8)
    selected_target = target[:, known]
    log_score = -(selected_target * selected_probability.log()).sum(dim=0).mean()
    brier = (selected_probability - selected_target).square().sum(dim=0).mean()
    return log_score + float(brier_weight) * brier


def proposal_null_set_proper_loss(
    joint_probability: torch.Tensor,
    acceptable_outcomes: torch.Tensor,
    evaluable: torch.Tensor,
    *,
    brier_weight: float = 1.0,
) -> torch.Tensor:
    """Proper score for a coarsened null-or-equivalent-proposal outcome.

    Multiple source proposals can be equally valid observations of one object.
    Their internal identity is unobserved, so the supervised event probability
    is their summed mass.  Treating one arbitrary member as the sole class
    would introduce false competition between same-object masks.
    """

    probability = torch.as_tensor(joint_probability)
    acceptable = torch.as_tensor(
        acceptable_outcomes, device=probability.device
    ).bool()
    known = torch.as_tensor(evaluable, device=probability.device).bool()
    if (
        probability.ndim != 2
        or acceptable.shape != probability.shape
        or known.shape != (probability.shape[1],)
        or not bool(known.any())
        or not bool(acceptable[:, known].any(dim=0).all())
        or not bool(torch.isfinite(probability).all())
        or bool((probability < 0).any())
        or not torch.allclose(
            probability.sum(dim=0),
            torch.ones_like(known, dtype=probability.dtype),
        )
        or not math.isfinite(float(brier_weight))
        or float(brier_weight) < 0
    ):
        raise ValueError("proposal/null set proper-loss inputs differ")
    event_probability = (
        probability[:, known] * acceptable[:, known].to(probability.dtype)
    ).sum(dim=0)
    log_score = -event_probability.clamp_min(1e-8).log().mean()
    brier = (1.0 - event_probability).square().mean()
    return log_score + float(brier_weight) * brier


__all__ = [
    "ProposalNullScorer",
    "ProposalNullScores",
    "proposal_null_proper_loss",
    "proposal_null_set_proper_loss",
]
