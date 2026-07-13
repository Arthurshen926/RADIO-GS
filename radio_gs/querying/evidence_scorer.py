"""Shared primitive-domain unary scorer for every query modality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .query_spec import PrototypeSet, QuerySpec


@dataclass(frozen=True)
class EvidenceScoringConfig:
    semantic_weight: float = 1.0
    appearance_weight: float = 1.0
    boundary_weight: float = 0.35
    prototype_temperature: float = 0.07

    def __post_init__(self) -> None:
        if min(self.semantic_weight, self.appearance_weight, self.boundary_weight) < 0:
            raise ValueError("evidence weights must be non-negative")
        if self.prototype_temperature <= 0:
            raise ValueError("prototype_temperature must be positive")


def _score_bank(
    field_features: torch.Tensor,
    evidence: PrototypeSet,
    *,
    temperature: float,
) -> torch.Tensor:
    field = F.normalize(torch.as_tensor(field_features).float(), dim=-1, eps=1e-8)
    prototypes = evidence.features.to(field.device)
    logits = field @ prototypes.T
    weights = evidence.weights.to(field.device).clamp_min(1e-8)
    positive = temperature * torch.logsumexp(
        logits / temperature + weights.log()[None], dim=1
    )
    if evidence.negatives is None:
        return positive
    negative = (field @ evidence.negatives.to(field.device).T).amax(dim=1)
    return positive - negative


def score_query_evidence(
    query: QuerySpec,
    feature_banks: Mapping[str, torch.Tensor],
    *,
    config: EvidenceScoringConfig = EvidenceScoringConfig(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components: dict[str, torch.Tensor] = {}
    weighted: list[torch.Tensor] = []
    for name, evidence, weight in (
        ("semantic", query.semantic_evidence, config.semantic_weight),
        ("appearance", query.appearance_evidence, config.appearance_weight),
        ("boundary", query.boundary_evidence, config.boundary_weight),
    ):
        if evidence is None:
            continue
        if name not in feature_banks:
            raise KeyError(f"query requires missing {name} feature bank")
        score = _score_bank(
            feature_banks[name], evidence, temperature=config.prototype_temperature
        )
        components[name] = score
        weighted.append(score * float(weight))
    if not weighted:
        count = (
            query.positive_seeds.weights.numel()
            if query.positive_seeds is not None
            else 0
        )
        if count == 0:
            raise ValueError("query has neither prototypes nor seeds")
        # A seed-only query has no global foreground evidence.  Use the
        # minimum cosine prior instead of an ambiguous 0-logit (p=0.5), then
        # let the same graph diffuse the registered/world-space seeds.
        unary = torch.full((count,), -1.0, dtype=torch.float32)
    else:
        unary = torch.stack(weighted).sum(dim=0) / max(
            1e-8,
            sum(
                weight
                for evidence, weight in (
                    (query.semantic_evidence, config.semantic_weight),
                    (query.appearance_evidence, config.appearance_weight),
                    (query.boundary_evidence, config.boundary_weight),
                )
                if evidence is not None
            ),
        )
    return unary, components
