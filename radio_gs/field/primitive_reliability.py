"""Query-independent confidence for reconstructed Gaussian descriptors."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrimitiveReliability:
    """Auditable components of one canonical reliability estimate."""

    confidence: torch.Tensor
    observation_evidence: torch.Tensor
    multiview_agreement: torch.Tensor
    reconstruction_fidelity: torch.Tensor


def canonical_primitive_reliability(
    view_counts: torch.Tensor,
    multiview_reliability: torch.Tensor,
    reconstruction_cosine: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> PrimitiveReliability:
    """Estimate descriptor precision without a query, label, or tuned threshold.

    Observation evidence uses the one-pseudocount saturation ``n / (n + 1)``
    so its meaning does not depend on how many cameras a dataset happens to
    contain.  It is combined by an equal-weight geometric mean with MPR
    agreement and the compact-field/teacher cosine.  The third historical MPR
    reliability channel is deliberately not consumed: completed caches used
    it as a provenance bit, whereas robust-consensus caches used it as
    stability.  Restricting the contract to the common first two channels
    keeps this confidence definition valid for both cache families.
    """

    counts = torch.as_tensor(view_counts).float().reshape(-1)
    reliability = torch.as_tensor(multiview_reliability).float()
    cosine = torch.as_tensor(reconstruction_cosine).float().reshape(-1)
    count = int(counts.numel())
    if reliability.ndim != 2 or reliability.shape[0] != count:
        raise ValueError("multiview_reliability must align as [N,R]")
    if reliability.shape[1] < 2:
        raise ValueError("multiview_reliability requires count/agreement channels")
    if cosine.shape != (count,):
        raise ValueError("reconstruction_cosine must align as [N]")
    if bool((counts < 0).any()) or not bool(torch.isfinite(counts).all()):
        raise ValueError("view_counts must be finite and non-negative")
    if not bool(torch.isfinite(reliability[:, :2]).all()):
        raise ValueError("multiview reliability contains NaN or infinity")
    if not bool(torch.isfinite(cosine).all()):
        raise ValueError("reconstruction cosine contains NaN or infinity")

    if valid is None:
        valid_rows = counts > 0
    else:
        valid_rows = torch.as_tensor(valid).bool().reshape(-1)
        if valid_rows.shape != (count,):
            raise ValueError("valid must align as [N]")
        if bool((valid_rows & (counts <= 0)).any()):
            raise ValueError("valid primitives must have at least one observation")

    observation = counts / (counts + 1.0)
    agreement = reliability[:, 1].clamp(0.0, 1.0)
    reconstruction = cosine.clamp(0.0, 1.0)
    confidence = (observation * agreement * reconstruction).clamp_min(0.0).pow(
        1.0 / 3.0
    )
    for values in (confidence, observation, agreement, reconstruction):
        values.masked_fill_(~valid_rows, 0.0)
    return PrimitiveReliability(
        confidence=confidence,
        observation_evidence=observation,
        multiview_agreement=agreement,
        reconstruction_fidelity=reconstruction,
    )
