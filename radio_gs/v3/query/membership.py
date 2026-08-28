"""The single Gaussian-domain posterior shared by all query modalities."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def pool_prototype(instance: torch.Tensor, support_weight: torch.Tensor) -> torch.Tensor:
    embedding = torch.as_tensor(instance).float()
    weight = torch.as_tensor(support_weight, device=embedding.device).float().reshape(-1)
    if embedding.ndim != 2 or weight.shape != (embedding.shape[0],):
        raise ValueError("prototype support axes differ")
    if bool((weight < 0).any()) or not bool(torch.isfinite(weight).all()) or float(weight.sum()) <= 0:
        raise ValueError("prototype requires positive finite support")
    return F.normalize((embedding * weight[:, None]).sum(0) / weight.sum(), dim=0, eps=1e-8)


def membership_from_prototype(
    instance: torch.Tensor,
    prototype: torch.Tensor,
    *,
    temperature: float,
    margin: float = 0.0,
    geometric_logit: torch.Tensor | None = None,
) -> torch.Tensor:
    embedding = F.normalize(torch.as_tensor(instance).float(), dim=-1, eps=1e-8)
    proto = F.normalize(torch.as_tensor(prototype, device=embedding.device).float().reshape(-1), dim=0, eps=1e-8)
    if proto.shape != (embedding.shape[1],) or temperature <= 0:
        raise ValueError("prototype dimension or temperature differs")
    if not torch.isfinite(torch.tensor(float(margin))):
        raise ValueError("prototype margin differs")
    logit = (embedding @ proto - float(margin)) / float(temperature)
    if geometric_logit is not None:
        geometry = torch.as_tensor(geometric_logit, device=logit.device).float().reshape(-1)
        if geometry.shape != logit.shape or not bool(torch.isfinite(geometry).all()):
            raise ValueError("geometric logit axes differ")
        logit = logit + geometry
    return logit.sigmoid()


def relative_membership_from_prototypes(
    instance: torch.Tensor,
    positive_prototype: torch.Tensor,
    negative_prototypes: torch.Tensor,
    *,
    temperature: float,
    margin: float = 0.0,
) -> torch.Tensor:
    """Binary posterior from positive evidence versus the hardest negative.

    Unlike ``membership_from_prototype``, a cosine value of zero is not treated
    as positive evidence with probability 0.5.  The posterior is determined by
    the positive-minus-negative similarity gap.  Multiple negatives are
    intentionally reduced with a maximum so that an easy negative cannot hide
    a competing object prototype.
    """

    embedding = F.normalize(torch.as_tensor(instance).float(), dim=-1, eps=1e-8)
    positive = F.normalize(
        torch.as_tensor(positive_prototype, device=embedding.device).float().reshape(-1),
        dim=0,
        eps=1e-8,
    )
    negatives = F.normalize(
        torch.as_tensor(negative_prototypes, device=embedding.device).float(),
        dim=-1,
        eps=1e-8,
    )
    if negatives.ndim == 1:
        negatives = negatives[None]
    if (
        embedding.ndim != 2
        or positive.shape != (embedding.shape[1],)
        or negatives.ndim != 2
        or negatives.shape[1] != embedding.shape[1]
        or not negatives.shape[0]
        or temperature <= 0
    ):
        raise ValueError("relative prototype axes or temperature differ")
    if not torch.isfinite(torch.tensor(float(margin))):
        raise ValueError("relative prototype margin differs")
    positive_similarity = embedding @ positive
    hardest_negative = (embedding @ negatives.T).max(dim=1).values
    return ((positive_similarity - hardest_negative - float(margin)) / float(temperature)).sigmoid()


__all__ = [
    "membership_from_prototype",
    "pool_prototype",
    "relative_membership_from_prototypes",
]
