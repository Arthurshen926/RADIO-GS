"""Disjoint composition of observed likelihood and abstention completion.

The operator is deliberately a partitioned assignment, not a probabilistic
fusion.  Rows observed by the original registered prompt retain the learned
query-likelihood unary bitwise.  Region memory may write only rows on which the
same original observation had exactly zero confidence.  Remaining rows keep
the learned branch's abstaining field prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256


MODE = "registered_likelihood_object_memory_disjoint_v1"


@dataclass(frozen=True)
class DisjointDomainComposition:
    unary: torch.Tensor
    observed_rows: torch.Tensor
    memory_rows: torch.Tensor
    abstained_rows: torch.Tensor
    hard_anchor_rows: torch.Tensor
    diagnostics: dict[str, object]


def compose_disjoint_domain_unary(
    learned_unary: torch.Tensor,
    memory_unary: torch.Tensor,
    *,
    original_observation_confidence: torch.Tensor,
    memory_confidence: torch.Tensor,
    hard_anchor_mask: torch.Tensor,
) -> DisjointDomainComposition:
    """Compose two unary branches on an exhaustive, mutually exclusive domain."""

    learned = torch.as_tensor(learned_unary)
    memory = torch.as_tensor(memory_unary)
    base_confidence = (
        torch.as_tensor(original_observation_confidence).detach().float().cpu().reshape(-1)
    )
    memory_reliability = (
        torch.as_tensor(memory_confidence).detach().float().cpu().reshape(-1)
    )
    anchors = torch.as_tensor(hard_anchor_mask).detach().bool().cpu().reshape(-1)
    if (
        learned.ndim != 1
        or memory.shape != learned.shape
        or base_confidence.shape != learned.shape
        or memory_reliability.shape != learned.shape
        or anchors.shape != learned.shape
        or not bool(torch.isfinite(learned).all())
        or not bool(torch.isfinite(memory).all())
        or not bool(torch.isfinite(base_confidence).all())
        or not bool(torch.isfinite(memory_reliability).all())
        or bool((base_confidence < 0).any())
        or bool((memory_reliability < 0).any())
    ):
        raise ValueError("disjoint-domain unary inputs do not align")

    observed = base_confidence > 0
    memory_domain = (base_confidence == 0) & (memory_reliability > 0)
    abstained = (base_confidence == 0) & (memory_reliability == 0)
    if not torch.equal(observed | memory_domain | abstained, torch.ones_like(observed)):
        raise RuntimeError("disjoint-domain partition is not exhaustive")
    if bool(
        (observed & memory_domain).any()
        or (observed & abstained).any()
        or (memory_domain & abstained).any()
    ):
        raise RuntimeError("disjoint-domain partition overlaps")
    if bool((anchors & ~observed).any()):
        raise ValueError("hard anchors must belong to the original observed domain")
    if not bool(memory_domain.any()):
        raise ValueError("region memory does not complete an original abstention")

    observed_device = observed.to(learned.device)
    memory_device = memory_domain.to(learned.device)
    anchors_device = anchors.to(learned.device)
    forward = learned.clone()
    forward[observed_device] = learned[observed_device]
    forward[memory_device] = memory[memory_device]
    reverse = learned.clone()
    reverse[memory_device] = memory[memory_device]
    reverse[observed_device] = learned[observed_device]
    if not torch.equal(forward, reverse):
        raise RuntimeError("disjoint-domain assignments are not commutative")
    if not (
        torch.equal(forward[observed_device], learned[observed_device])
        and torch.equal(forward[memory_device], memory[memory_device])
        and torch.equal(
            forward[abstained.to(learned.device)],
            learned[abstained.to(learned.device)],
        )
        and torch.equal(forward[anchors_device], learned[anchors_device])
    ):
        raise RuntimeError("disjoint-domain bitwise preservation failed")

    return DisjointDomainComposition(
        unary=forward,
        observed_rows=observed,
        memory_rows=memory_domain,
        abstained_rows=abstained,
        hard_anchor_rows=anchors,
        diagnostics={
            "mode": MODE,
            "rows": int(observed.numel()),
            "observed_rows": int(observed.sum()),
            "memory_rows": int(memory_domain.sum()),
            "abstained_rows": int(abstained.sum()),
            "hard_anchor_rows": int(anchors.sum()),
            "raw_memory_observed_rows_ignored": int(
                ((base_confidence > 0) & (memory_reliability > 0)).sum()
            ),
            "partition_exhaustive": True,
            "partition_pairwise_disjoint": True,
            "assignment_commutative": True,
            "same_row_double_counted": False,
            "probability_average_or_product_of_experts_used": False,
            "observed_unary_bitwise_equal_to_learned": True,
            "memory_unary_bitwise_equal_to_memory_branch": True,
            "abstained_unary_bitwise_equal_to_learned_field_prior": True,
            "hard_anchor_unary_bitwise_equal_to_learned": True,
            "observed_mask_sha256": tensor_sha256(observed.contiguous()),
            "memory_mask_sha256": tensor_sha256(memory_domain.contiguous()),
            "abstained_mask_sha256": tensor_sha256(abstained.contiguous()),
            "hard_anchor_mask_sha256": tensor_sha256(anchors.contiguous()),
        },
    )


__all__ = [
    "DisjointDomainComposition",
    "MODE",
    "compose_disjoint_domain_unary",
]
