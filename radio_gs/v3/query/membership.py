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
    geometric_logit: torch.Tensor | None = None,
) -> torch.Tensor:
    embedding = F.normalize(torch.as_tensor(instance).float(), dim=-1, eps=1e-8)
    proto = F.normalize(torch.as_tensor(prototype, device=embedding.device).float().reshape(-1), dim=0, eps=1e-8)
    if proto.shape != (embedding.shape[1],) or temperature <= 0:
        raise ValueError("prototype dimension or temperature differs")
    logit = embedding @ proto / float(temperature)
    if geometric_logit is not None:
        geometry = torch.as_tensor(geometric_logit, device=logit.device).float().reshape(-1)
        if geometry.shape != logit.shape or not bool(torch.isfinite(geometry).all()):
            raise ValueError("geometric logit axes differ")
        logit = logit + geometry
    return logit.sigmoid()


__all__ = ["membership_from_prototype", "pool_prototype"]
