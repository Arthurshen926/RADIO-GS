"""Shared no-harm contract for prompt-conditioned primitive completion.

Prompt interfaces first produce an analytic primitive probability (the
``anchor``).  A learned unary head or graph solver may then propose a logit
residual, but its intervention budget is the product of

``(1 - observation_confidence) * completion_confidence``.

Consequently a fully observed primitive is an exact identity, a partially
observed primitive receives only a proportional residual, and an unobserved
primitive may use the complete globally bounded proposal.  Keeping this
operation in one module prevents full-mask, scribble, and graph readouts from
quietly adopting incompatible definitions of "observation preserving".
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class AnchorPreservingTransportOutput:
    """Result and auditable intervention terms for one transport step."""

    probability: torch.Tensor
    residual_gate: torch.Tensor
    applied_logit_residual: torch.Tensor
    anchor_probability: torch.Tensor


def method_contract() -> dict[str, object]:
    """Return the query- and benchmark-independent scientific contract."""

    return {
        "schema_version": 1,
        "method": "anchor_preserving_confidence_gated_logit_residual_v1",
        "anchor": "analytic_prompt_probability_before_learned_or_graph_completion",
        "residual_gate": "(1-observation_confidence)*completion_confidence",
        "active_domain_policy": "inactive_rows_are_exact_anchor_identity",
        "fully_observed_policy": "exact_identity",
        "partially_observed_policy": "proportional_logit_residual_budget",
        "unobserved_policy": "globally_bounded_completion_residual",
        "row_coupling": "permitted_only_inside_the_completion_proposal",
        "connected_selection": False,
        "uses_target_rgb_mask_or_metric": False,
        "scene_specific_parameter": False,
    }


def _probability_vector(value: torch.Tensor, *, label: str) -> torch.Tensor:
    result = torch.as_tensor(value).reshape(-1)
    if not result.is_floating_point():
        result = result.float()
    if not bool(torch.isfinite(result).all()) or bool(
        ((result < 0) | (result > 1)).any()
    ):
        raise ValueError(f"{label} must be a finite probability vector")
    return result


def residual_budget(
    observation_confidence: torch.Tensor,
    *,
    completion_confidence: torch.Tensor | None = None,
    active_domain: torch.Tensor | None = None,
    fully_observed_tolerance: float = 1e-5,
) -> torch.Tensor:
    """Compute the intervention budget without a learned numeric threshold."""

    observed = _probability_vector(
        observation_confidence, label="observation_confidence"
    )
    if not math.isfinite(float(fully_observed_tolerance)) or not (
        0 <= float(fully_observed_tolerance) < 1
    ):
        raise ValueError("fully_observed_tolerance must be finite in [0,1)")
    if completion_confidence is None:
        completion = torch.ones_like(observed)
    else:
        completion = _probability_vector(
            completion_confidence, label="completion_confidence"
        ).to(device=observed.device, dtype=observed.dtype)
        if completion.shape != observed.shape:
            raise ValueError("completion and observation confidence differ")
    unknown_fraction = (1.0 - observed).clamp(0.0, 1.0)
    # Exact zeros make the no-harm property bitwise auditable even when an
    # upstream confidence is within serialization tolerance of one.
    unknown_fraction = torch.where(
        observed >= 1.0 - float(fully_observed_tolerance),
        torch.zeros_like(unknown_fraction),
        unknown_fraction,
    )
    budget = unknown_fraction * completion
    if active_domain is not None:
        active = torch.as_tensor(active_domain, device=observed.device).reshape(-1).bool()
        if active.shape != observed.shape:
            raise ValueError("active_domain and observation confidence differ")
        budget = torch.where(active, budget, torch.zeros_like(budget))
    return budget.contiguous()


def apply_anchor_preserving_logit_residual(
    anchor_probability: torch.Tensor,
    proposed_logit_residual: torch.Tensor,
    observation_confidence: torch.Tensor,
    *,
    completion_confidence: torch.Tensor | None = None,
    active_domain: torch.Tensor | None = None,
    max_abs_logit_residual: float,
    fully_observed_tolerance: float = 1e-5,
    eps: float = 1e-6,
) -> AnchorPreservingTransportOutput:
    """Apply a bounded proposal while preserving fully observed anchors.

    ``proposed_logit_residual`` is clamped before gating.  This makes the
    global bound part of the shared contract instead of relying on every head
    or solver to implement it independently.
    """

    anchor = _probability_vector(anchor_probability, label="anchor_probability")
    proposal = torch.as_tensor(
        proposed_logit_residual, device=anchor.device, dtype=anchor.dtype
    ).reshape(-1)
    observation = _probability_vector(
        observation_confidence, label="observation_confidence"
    ).to(device=anchor.device, dtype=anchor.dtype)
    if proposal.shape != anchor.shape or observation.shape != anchor.shape:
        raise ValueError("anchor, residual, and observation confidence differ")
    if not bool(torch.isfinite(proposal).all()):
        raise ValueError("proposed_logit_residual must be finite")
    maximum = float(max_abs_logit_residual)
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("max_abs_logit_residual must be finite and positive")
    if not math.isfinite(float(eps)) or not 0 < float(eps) < 0.5:
        raise ValueError("eps must be finite in (0,0.5)")
    gate = residual_budget(
        observation,
        completion_confidence=completion_confidence,
        active_domain=active_domain,
        fully_observed_tolerance=fully_observed_tolerance,
    )
    applied = gate * proposal.clamp(-maximum, maximum)
    safe_anchor = anchor.clamp(float(eps), 1.0 - float(eps))
    anchor_logit = torch.logit(safe_anchor)
    # Difference form guarantees exact identity wherever ``applied`` is zero,
    # including anchors serialized at the probability endpoints.
    probability = anchor + (
        torch.sigmoid(anchor_logit + applied) - torch.sigmoid(anchor_logit)
    )
    probability = probability.clamp(0.0, 1.0)
    if bool((gate == 0).any()) and not torch.equal(
        probability[gate == 0], anchor[gate == 0]
    ):
        raise RuntimeError("anchor-preserving identity invariant failed")
    return AnchorPreservingTransportOutput(
        probability=probability.contiguous(),
        residual_gate=gate.contiguous(),
        applied_logit_residual=applied.contiguous(),
        anchor_probability=anchor.contiguous(),
    )


def apply_anchor_preserving_probability_proposal(
    anchor_probability: torch.Tensor,
    proposal_probability: torch.Tensor,
    observation_confidence: torch.Tensor,
    *,
    completion_confidence: torch.Tensor | None = None,
    active_domain: torch.Tensor | None = None,
    max_abs_logit_residual: float,
    fully_observed_tolerance: float = 1e-5,
    eps: float = 1e-6,
) -> AnchorPreservingTransportOutput:
    """Convert a graph probability proposal to the shared logit residual."""

    anchor = _probability_vector(anchor_probability, label="anchor_probability")
    proposal = _probability_vector(
        proposal_probability, label="proposal_probability"
    ).to(device=anchor.device, dtype=anchor.dtype)
    if proposal.shape != anchor.shape:
        raise ValueError("anchor and proposal probability differ")
    safe_anchor = anchor.clamp(float(eps), 1.0 - float(eps))
    safe_proposal = proposal.clamp(float(eps), 1.0 - float(eps))
    proposed_residual = torch.logit(safe_proposal) - torch.logit(safe_anchor)
    return apply_anchor_preserving_logit_residual(
        anchor,
        proposed_residual,
        observation_confidence,
        completion_confidence=completion_confidence,
        active_domain=active_domain,
        max_abs_logit_residual=max_abs_logit_residual,
        fully_observed_tolerance=fully_observed_tolerance,
        eps=eps,
    )


__all__ = [
    "AnchorPreservingTransportOutput",
    "apply_anchor_preserving_logit_residual",
    "apply_anchor_preserving_probability_proposal",
    "method_contract",
    "residual_budget",
]
