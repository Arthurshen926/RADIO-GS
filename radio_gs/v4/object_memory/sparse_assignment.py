"""Top-k sparse element-to-object memberships with explicit unknown mass."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ElementQueryPosterior:
    foreground: torch.Tensor
    assignment_unknown: torch.Tensor
    query_null_probability: float


@dataclass(frozen=True)
class SparseObjectAssignments:
    token_ids: torch.Tensor
    weights: torch.Tensor
    unknown_weight: torch.Tensor
    num_tokens: int

    def __post_init__(self) -> None:
        token_ids = torch.as_tensor(self.token_ids, dtype=torch.long).cpu()
        weights = torch.as_tensor(self.weights, dtype=torch.float32).cpu()
        unknown = torch.as_tensor(self.unknown_weight, dtype=torch.float32).cpu()
        if token_ids.ndim != 2 or token_ids.shape != weights.shape or token_ids.shape[1] not in (1, 2):
            raise ValueError("token_ids and weights must have shape [E, 1|2]")
        if unknown.shape != (token_ids.shape[0],) or self.num_tokens <= 0:
            raise ValueError("unknown_weight must have shape [E] and num_tokens must be positive")
        if not torch.isfinite(weights).all() or not torch.isfinite(unknown).all():
            raise ValueError("assignment probabilities must be finite")
        if bool((weights < 0).any()) or bool((unknown < 0).any()):
            raise ValueError("assignment probabilities must be non-negative")
        occupied = weights > 0
        if bool((token_ids[occupied] < 0).any()) or bool((token_ids[occupied] >= self.num_tokens).any()):
            raise ValueError("occupied token id outside codebook")
        if bool((token_ids[~occupied] != -1).any()):
            raise ValueError("zero-weight slots must use token id -1")
        if bool((weights.sum(-1) + unknown > 1.0 + 1e-5).any()):
            raise ValueError("known and unknown assignment mass exceeds one")
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "unknown_weight", unknown)

    @property
    def num_elements(self) -> int:
        return int(self.token_ids.shape[0])

    @classmethod
    def from_dense(
        cls,
        probabilities: torch.Tensor,
        *,
        unknown_weight: torch.Tensor | None = None,
        top_k: int = 2,
    ) -> "SparseObjectAssignments":
        dense = torch.as_tensor(probabilities, dtype=torch.float32).cpu()
        if dense.ndim != 2 or dense.shape[1] == 0 or top_k not in (1, 2):
            raise ValueError("probabilities must have shape [E, K] and top_k must be 1 or 2")
        if not torch.isfinite(dense).all() or bool((dense < 0).any()):
            raise ValueError("dense probabilities must be finite and non-negative")
        unknown = (
            (1.0 - dense.sum(-1)).clamp_min(0)
            if unknown_weight is None
            else torch.as_tensor(unknown_weight, dtype=torch.float32).cpu()
        )
        if unknown.shape != (dense.shape[0],) or bool((unknown < 0).any()):
            raise ValueError("unknown_weight must be a non-negative [E] vector")
        total = dense.sum(-1) + unknown
        normalized = dense / total.clamp_min(1e-12)[:, None]
        normalized_unknown = torch.where(total > 0, unknown / total.clamp_min(1e-12), torch.ones_like(total))
        count = min(top_k, dense.shape[1])
        values, ids = normalized.topk(count, dim=-1)
        if count < top_k:
            values = torch.cat([values, torch.zeros(dense.shape[0], top_k - count)], dim=-1)
            ids = torch.cat([ids, torch.full((dense.shape[0], top_k - count), -1)], dim=-1)
        retained = values.sum(-1)
        normalized_unknown = normalized_unknown + (1.0 - normalized_unknown - retained).clamp_min(0)
        ids[values <= 0] = -1
        return cls(ids, values, normalized_unknown, dense.shape[1])

    def to_dense(self) -> torch.Tensor:
        result = torch.zeros(self.num_elements, self.num_tokens)
        rows = torch.arange(self.num_elements)[:, None].expand_as(self.token_ids)
        occupied = self.token_ids >= 0
        result.index_put_((rows[occupied], self.token_ids[occupied]), self.weights[occupied], accumulate=True)
        return result

    def element_posterior(
        self,
        token_posterior: torch.Tensor,
        *,
        null_probability: float,
    ) -> ElementQueryPosterior:
        posterior = torch.as_tensor(token_posterior, dtype=torch.float32).cpu()
        null = float(null_probability)
        if (
            posterior.shape != (self.num_tokens,)
            or not torch.isfinite(posterior).all()
            or bool((posterior < 0).any())
            or bool((posterior > 1).any())
            or not 0 <= null <= 1
        ):
            raise ValueError("token and null probabilities must lie in [0, 1]")
        if abs(float(posterior.sum()) + null - 1.0) > 1e-5:
            raise ValueError("token probabilities and explicit null must sum to one")
        safe_ids = self.token_ids.clamp_min(0)
        selected = posterior[safe_ids] * self.weights
        selected[self.token_ids < 0] = 0
        return ElementQueryPosterior(
            foreground=selected.sum(-1),
            assignment_unknown=(1.0 - null) * self.unknown_weight,
            query_null_probability=null,
        )
