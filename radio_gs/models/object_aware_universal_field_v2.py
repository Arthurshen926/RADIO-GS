"""Compact query-independent object structure for Universal Field v2 pilots.

The module deliberately contains no text query or benchmark label.  Source
SAM masks supervise a small per-Gaussian code through sparse exact-MPR pooling;
one scene-independent decoder maps pooled object codes to the frozen teacher
language space.  Unknown pair relations are excluded from the proper loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def sparse_proposal_pool(
    codes: torch.Tensor,
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    weights: torch.Tensor,
    num_proposals: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact-MPR weighted proposal means and their evidence mass."""

    if codes.ndim != 2:
        raise ValueError("object codes must be a matrix")
    rows = torch.as_tensor(row_indices, device=codes.device, dtype=torch.long)
    proposals = torch.as_tensor(
        proposal_indices, device=codes.device, dtype=torch.long
    )
    values = torch.as_tensor(weights, device=codes.device, dtype=codes.dtype)
    if rows.ndim != 1 or rows.shape != proposals.shape or rows.shape != values.shape:
        raise ValueError("sparse membership axes differ")
    if rows.numel() and (
        bool(((rows < 0) | (rows >= codes.shape[0])).any())
        or bool(((proposals < 0) | (proposals >= int(num_proposals))).any())
    ):
        raise ValueError("sparse membership index is outside its domain")
    if bool((~torch.isfinite(values) | (values < 0)).any()):
        raise ValueError("membership weights must be finite and non-negative")
    pooled = codes.new_zeros((int(num_proposals), codes.shape[1]))
    mass = codes.new_zeros((int(num_proposals),))
    if rows.numel():
        pooled.index_add_(0, proposals, codes[rows] * values[:, None])
        mass.index_add_(0, proposals, values)
    pooled = pooled / mass.clamp_min(torch.finfo(codes.dtype).eps)[:, None]
    return pooled, mass


def scale_bins(area_fraction: torch.Tensor, bins: int = 4) -> torch.Tensor:
    """Map mask area to fixed log-area bins without dataset quantiles."""

    area = torch.as_tensor(area_fraction).float()
    if bins != 4:
        raise ValueError("the v2 pilot freezes four scale bins")
    if bool((~torch.isfinite(area) | (area < 0) | (area > 1)).any()):
        raise ValueError("proposal area must be finite in [0,1]")
    boundaries = area.new_tensor([1.0 / 256.0, 1.0 / 64.0, 1.0 / 16.0])
    return torch.bucketize(area, boundaries)


class ObjectAwareFieldHead(nn.Module):
    """A compact object code plus global scale and language decoders."""

    def __init__(
        self,
        num_gaussians: int,
        *,
        object_dim: int = 16,
        language_dim: int = 1536,
        seed: int = 20260821,
    ) -> None:
        super().__init__()
        if num_gaussians <= 0 or object_dim <= 0 or language_dim <= 0:
            raise ValueError("object-aware head dimensions must be positive")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        initial = torch.randn(
            num_gaussians, object_dim, generator=generator, dtype=torch.float32
        ) / object_dim**0.5
        self.object_codes = nn.Parameter(initial)
        self.scale_log_gates = nn.Parameter(torch.zeros(4, object_dim))
        self.language_decoder = nn.Linear(object_dim, language_dim, bias=False)
        nn.init.normal_(self.language_decoder.weight, std=object_dim**-0.5)

    def proposal_embeddings(
        self,
        row_indices: torch.Tensor,
        proposal_indices: torch.Tensor,
        weights: torch.Tensor,
        area_fraction: torch.Tensor,
        num_proposals: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, mass = sparse_proposal_pool(
            self.object_codes,
            row_indices,
            proposal_indices,
            weights,
            num_proposals,
        )
        bins = scale_bins(area_fraction).to(pooled.device)
        gates = F.softplus(self.scale_log_gates[bins])
        return F.normalize(pooled * gates, dim=-1), mass

    def decode_language(self, proposal_embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.language_decoder(proposal_embeddings), dim=-1)


@dataclass(frozen=True)
class ObjectAwareLoss:
    total: torch.Tensor
    same: torch.Tensor
    different: torch.Tensor
    language: torch.Tensor


def object_aware_proper_loss(
    proposal_embeddings: torch.Tensor,
    decoded_language: torch.Tensor,
    language_teacher: torch.Tensor,
    edge_left: torch.Tensor,
    edge_right: torch.Tensor,
    edge_relation: torch.Tensor,
    *,
    language_weight: float = 0.20,
    relation_logit_scale: float = 8.0,
) -> ObjectAwareLoss:
    """Balanced same/different loss plus mask-aligned language distillation.

    Relation values are ``1`` (same), ``0`` (different), and ``-1``
    (unknown).  Unknown edges contribute exactly zero.
    """

    left = torch.as_tensor(edge_left, device=proposal_embeddings.device).long()
    right = torch.as_tensor(edge_right, device=proposal_embeddings.device).long()
    relation = torch.as_tensor(
        edge_relation, device=proposal_embeddings.device
    ).to(torch.int8)
    if left.shape != right.shape or left.shape != relation.shape or left.ndim != 1:
        raise ValueError("object relation axes differ")
    if not float(relation_logit_scale) > 0:
        raise ValueError("relation logit scale must be positive")
    cosine = (proposal_embeddings[left] * proposal_embeddings[right]).sum(-1)
    logits = float(relation_logit_scale) * cosine
    same_mask = relation == 1
    different_mask = relation == 0
    zero = cosine.sum() * 0.0
    same = (
        F.binary_cross_entropy_with_logits(
            logits[same_mask], torch.ones_like(logits[same_mask])
        )
        if bool(same_mask.any())
        else zero
    )
    different = (
        F.binary_cross_entropy_with_logits(
            logits[different_mask], torch.zeros_like(logits[different_mask])
        )
        if bool(different_mask.any())
        else zero
    )
    teacher = F.normalize(language_teacher.to(decoded_language), dim=-1)
    if teacher.shape != decoded_language.shape:
        raise ValueError("decoded and teacher language axes differ")
    language = (1.0 - (decoded_language * teacher).sum(-1)).mean()
    total = same + different + float(language_weight) * language
    return ObjectAwareLoss(total=total, same=same, different=different, language=language)


__all__ = [
    "ObjectAwareFieldHead",
    "ObjectAwareLoss",
    "object_aware_proper_loss",
    "scale_bins",
    "sparse_proposal_pool",
]
