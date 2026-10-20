"""Query-transient response refinement over frozen, explicitly supplied evidence.

The common core never interprets a missing mask as background or estimates
identity from a benchmark name. Adapters supply positive/exclusion permissions.
Evidence admission remains fallible; this module does not guarantee accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class PartialResponseEvidence:
    operator: torch.Tensor
    target: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor
    query_compatible: torch.Tensor
    confidence: torch.Tensor
    normalization_domain: torch.Tensor | None = None

    def weights(self, prior):
        op = self.operator
        if op.layout != torch.sparse_coo or op.ndim != 2 or op.shape[1] != len(prior):
            raise ValueError("evidence requires an observation-by-element COO operator")
        if op.device != prior.device or op.dtype != prior.dtype:
            raise ValueError("operator and response device/dtype differ")
        values = op.coalesce().values()
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("invalid observation contributions")
        shape = (op.shape[0], prior.shape[1])
        if self.target.shape != shape or self.target.device != prior.device or self.target.dtype != prior.dtype:
            raise ValueError("target and response columns/device/dtype differ")
        if not bool(torch.isfinite(self.target).all()) or bool(((self.target < 0) | (self.target > 1)).any()):
            raise ValueError("target must be a bounded response, not raw similarity")
        masks = []
        for value in (self.positive, self.negative, self.query_compatible):
            if value.dtype != torch.bool or value.device != prior.device:
                raise ValueError("evidence permissions must be boolean on the response device")
            try:
                masks.append(torch.broadcast_to(value, shape))
            except RuntimeError as error:
                raise ValueError("permission shape differs from query columns") from error
        positive, negative, compatible = masks
        if bool((positive & negative).any()):
            raise ValueError("one observation cannot assert both signs at the same position")
        confidence = self.confidence
        if confidence.device != prior.device or confidence.dtype != prior.dtype:
            raise ValueError("confidence and response device/dtype differ")
        if not bool(torch.isfinite(confidence).all()) or bool((confidence < 0).any()):
            raise ValueError("invalid evidence confidence")
        try:
            confidence = torch.broadcast_to(confidence, shape)
        except RuntimeError as error:
            raise ValueError("confidence shape differs from query columns") from error
        return positive & compatible, negative & compatible, confidence


def _validate_prior(prior):
    if prior.ndim != 2 or not prior.is_floating_point():
        raise ValueError("prior must be an element-by-query floating response")
    if not bool(torch.isfinite(prior).all()) or bool(((prior < 0) | (prior > 1)).any()):
        raise ValueError("prior must be a bounded response, not raw similarity")


@torch.no_grad()
def fuse_admitted_evidence(initial, observations):
    """Extracted legacy visibility fusion, retaining its exact operation order.

Observations are (operator, target, admitted_weight). This compatibility path
preserves existing admission semantics; it does not certify them as correct.
"""
    mass = torch.zeros_like(initial)
    numerator = torch.zeros_like(initial)
    for operator, target, known in observations:
        mass += torch.sparse.mm(operator.T, known)
        numerator += torch.sparse.mm(operator.T, known * target)
    value = torch.where(mass > 1e-8, numerator / mass.clamp_min(1e-8), initial)
    return value, mass


class ResponseRefiner:
    """Accumulate once over a fixed prior; supports matched streaming controls."""

    def __init__(self, prior):
        _validate_prior(prior)
        self.prior = prior
        self.numerator = torch.zeros_like(prior)
        self.mass = torch.zeros_like(prior)
        self.positive_entries = self.negative_entries = self.observations = 0

    @torch.no_grad()
    def add(self, item):
        positive, negative, confidence = item.weights(self.prior)
        residual = item.target - torch.sparse.mm(item.operator, self.prior)
        admitted = (positive & (residual > 0)) | (negative & (residual < 0))
        weight = admitted.to(self.prior.dtype) * confidence
        self.numerator += torch.sparse.mm(item.operator.T, weight * residual)
        self.mass += torch.sparse.mm(item.operator.T, weight)
        self.positive_entries += int((positive & (residual > 0) & (confidence > 0)).sum())
        self.negative_entries += int((negative & (residual < 0) & (confidence > 0)).sum())
        self.observations += 1

    @torch.no_grad()
    def finish(self):
        delta = self.numerator / self.mass.clamp_min(1e-8)
        result = torch.where(self.mass > 1e-8, (self.prior + delta).clamp(0, 1), self.prior)
        return result, dict(observations=self.observations, positive_entries=self.positive_entries,
            negative_entries=self.negative_entries, updated_entries=int((result != self.prior).sum()),
            supported_entries=int((self.mass > 1e-8).sum()), no_feature_update=True,
            no_evidence_retains_prior=True, accuracy_non_degradation_guaranteed=False)


@torch.no_grad()
def refine_response(prior: torch.Tensor, evidence: Iterable[PartialResponseEvidence]):
    """One directional residual step. Unknown evidence preserves the prior.

    Positive-only evidence cannot lower the response; explicit exclusion can.
    All residuals use the original prior, without iterative self-confirmation.
    """
    refiner = ResponseRefiner(prior)
    for item in evidence:
        refiner.add(item)
    return refiner.finish()
