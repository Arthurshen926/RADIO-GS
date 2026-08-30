"""Training-time dense assignments kept separate from deployment sparsity."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .sparse_assignment import SparseObjectAssignments


@dataclass(frozen=True)
class DenseObjectAssignments:
    token_probability: torch.Tensor
    unknown_probability: torch.Tensor

    def __post_init__(self) -> None:
        token = torch.as_tensor(self.token_probability, dtype=torch.float32)
        unknown = torch.as_tensor(self.unknown_probability, dtype=torch.float32)
        if token.ndim != 2 or token.shape[1] == 0 or unknown.shape != (token.shape[0],):
            raise ValueError("dense assignments require [E, K] token and [E] unknown probabilities")
        if not torch.isfinite(token).all() or not torch.isfinite(unknown).all():
            raise ValueError("dense assignment probabilities must be finite")
        if bool((token < 0).any()) or bool((unknown < 0).any()):
            raise ValueError("dense assignment probabilities must be non-negative")
        if not torch.allclose(token.sum(-1) + unknown, torch.ones_like(unknown), atol=1e-5):
            raise ValueError("dense token and unknown probabilities must form a simplex")
        object.__setattr__(self, "token_probability", token)
        object.__setattr__(self, "unknown_probability", unknown)

    @classmethod
    def from_logits(
        cls,
        token_logits: torch.Tensor,
        unknown_logit: torch.Tensor,
    ) -> "DenseObjectAssignments":
        token_logits = torch.as_tensor(token_logits, dtype=torch.float32)
        unknown_logit = torch.as_tensor(unknown_logit, dtype=torch.float32)
        if token_logits.ndim != 2 or unknown_logit.shape != (token_logits.shape[0],):
            raise ValueError("logits must have shape [E, K] and [E]")
        probability = torch.softmax(torch.cat([token_logits, unknown_logit[:, None]], -1), -1)
        return cls(probability[:, :-1], probability[:, -1])

    def compress(self, top_k: int = 2) -> SparseObjectAssignments:
        """Compress only for a frozen deployment/export candidate."""

        return SparseObjectAssignments.from_dense(
            self.token_probability.detach().cpu(),
            unknown_weight=self.unknown_probability.detach().cpu(),
            top_k=top_k,
        )
