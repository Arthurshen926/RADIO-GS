"""Raw-canonical unary specificity for query-agnostic LERF region graphs.

Canonical-negative probabilities use one negative bank and one logit scale
before the independent per-query VALA min/max remap.  They can therefore be
used to ask a narrow semantic question that the graph cannot answer: which
query is dominant in a region?  This module intentionally does not consume
the remapped O0 scores, target labels, masks, or metrics.

The deployable statistic is the query-order-equivariant argmax (ties retained)
of the valid-core mean raw canonical probability.  A primitive-wise top-query
fraction is returned as a diagnostic only; it is not a calibrated threshold or
a gate in this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import canonical_json_sha256


SCHEMA = "radio_gs.lerf_raw_unary_region_specificity.v1"


def specificity_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "input": (
            "raw_canonical_negative_probability_at_each_query_frozen_o0_scale"
        ),
        "required_common_scale": "same_negative_bank_and_logit_scale",
        "forbidden_input": "independently_per_query_vala_minmax_remapped_score",
        "region_statistic": "valid_core_mean_raw_probability_per_query",
        "specificity": "region_mean_argmax_with_all_exact_ties_retained",
        "primitive_top1_fraction": "diagnostic_only_not_a_gate",
        "primitive_majority_threshold": None,
        "candidate_gate": "existing_graph_candidate_and_region_mean_dominant_query",
        "anchor_gate": "existing_o0_anchor_and_region_mean_dominant_query",
        "graph_order": "apply_anchor_gate_before_direct_support_propagation",
        "query_order_equivariant": True,
        "query_conditioned_parameters": False,
        "scene_conditioned_parameters": False,
        "target_metrics_used": False,
    }


CONTRACT_SHA256 = canonical_json_sha256(specificity_contract())


@dataclass(frozen=True)
class RawUnaryRegionSpecificity:
    mean_raw_probability: torch.Tensor
    dominant_query_mask: torch.Tensor
    primitive_top1_fraction: torch.Tensor
    valid_core_counts: torch.Tensor


@dataclass(frozen=True)
class RawDominantGraphGate:
    specific_anchor_region: torch.Tensor
    direct_specific_anchor_support: torch.Tensor
    specific_candidate_region: torch.Tensor


def symmetric_raw_dominant_graph_gate(
    *,
    base_anchor_region: torch.Tensor,
    dominant_query_mask: torch.Tensor,
    pair_indices: torch.Tensor,
    edge_eligible_mask: torch.Tensor,
    region_eligible_mask: torch.Tensor,
    anchor_quorum: int,
) -> RawDominantGraphGate:
    """Recompute graph support after symmetric raw-dominant unary filtering."""

    anchor = torch.as_tensor(base_anchor_region).detach()
    dominant = torch.as_tensor(dominant_query_mask).detach()
    pairs = torch.as_tensor(pair_indices).detach()
    edge = torch.as_tensor(edge_eligible_mask).detach()
    region = torch.as_tensor(region_eligible_mask).detach()
    if (
        anchor.device.type != "cpu"
        or anchor.dtype != torch.bool
        or anchor.ndim != 2
        or min(anchor.shape) <= 0
        or dominant.device.type != "cpu"
        or dominant.dtype != torch.bool
        or dominant.shape != anchor.shape
        or pairs.device.type != "cpu"
        or pairs.dtype not in {torch.int32, torch.int64}
        or pairs.ndim != 2
        or pairs.shape[0] != 2
        or pairs.shape[1] <= 0
        or edge.device.type != "cpu"
        or edge.dtype != torch.bool
        or edge.shape != (pairs.shape[1],)
        or region.device.type != "cpu"
        or region.dtype != torch.bool
        or region.shape != (anchor.shape[0],)
        or isinstance(anchor_quorum, bool)
        or int(anchor_quorum) <= 0
        or bool((pairs < 0).any())
        or bool((pairs >= anchor.shape[0]).any())
    ):
        raise ValueError("raw-dominant graph gate inputs differ")
    if bool((pairs[0] == pairs[1]).any()):
        raise ValueError("raw-dominant graph gate rejects self edges")

    specific_anchor = anchor & dominant & region[:, None]
    support = torch.zeros_like(anchor, dtype=torch.int64)
    for edge_index in torch.nonzero(edge).flatten().tolist():
        left = int(pairs[0, edge_index])
        right = int(pairs[1, edge_index])
        support[left] += specific_anchor[right].long()
        support[right] += specific_anchor[left].long()
    enough_anchor = specific_anchor.sum(dim=0) >= int(anchor_quorum)
    candidate = (
        (support >= 1)
        & (~specific_anchor)
        & dominant
        & region[:, None]
        & enough_anchor[None, :]
    )
    if (
        bool((specific_anchor & ~anchor).any())
        or bool((specific_anchor & ~dominant).any())
        or bool((candidate & ~dominant).any())
        or bool((candidate & specific_anchor).any())
        or bool((candidate & (support < 1)).any())
    ):
        raise RuntimeError("raw-dominant graph gate invariant failed")
    return RawDominantGraphGate(
        specific_anchor_region=specific_anchor.contiguous(),
        direct_specific_anchor_support=support.contiguous(),
        specific_candidate_region=candidate.contiguous(),
    )


def raw_unary_region_specificity(
    *,
    raw_query_probabilities: torch.Tensor,
    region_rows: torch.Tensor,
    core_mask: torch.Tensor,
    primitive_valid_mask: torch.Tensor,
) -> RawUnaryRegionSpecificity:
    """Summarize raw-canonical query specificity on canonical region cores."""

    raw = torch.as_tensor(raw_query_probabilities).detach()
    rows = torch.as_tensor(region_rows).detach()
    core = torch.as_tensor(core_mask).detach()
    valid = torch.as_tensor(primitive_valid_mask).detach()
    if (
        raw.dtype != torch.float32
        or raw.device.type != "cpu"
        or raw.ndim != 2
        or min(raw.shape) <= 0
        or not bool(torch.isfinite(raw).all())
        or bool((raw < 0.0).any())
        or bool((raw > 1.0).any())
        or rows.device.type != "cpu"
        or rows.dtype not in {torch.int32, torch.int64}
        or rows.ndim != 2
        or rows.shape[0] <= 0
        or core.device.type != "cpu"
        or core.dtype != torch.bool
        or core.shape != rows.shape
        or valid.device.type != "cpu"
        or valid.dtype != torch.bool
        or valid.shape != (raw.shape[0],)
        or not bool(core.any(dim=1).all())
    ):
        raise ValueError("raw-unary region specificity inputs differ")
    active_rows = rows[core]
    if (
        active_rows.numel() <= 0
        or bool((active_rows < 0).any())
        or bool((active_rows >= raw.shape[0]).any())
    ):
        raise ValueError("raw-unary region core indices differ")

    safe = rows.long().clamp(min=0, max=raw.shape[0] - 1)
    valid_core = core & valid[safe]
    counts = valid_core.sum(dim=1)
    if not bool((counts > 0).all()):
        raise ValueError("every raw-unary region must have a valid core primitive")
    gathered = raw[safe]
    weight = valid_core[:, :, None].float()
    means = (gathered * weight).sum(dim=1) / counts[:, None].float()
    dominant = means == means.amax(dim=1, keepdim=True)

    primitive_dominant = gathered == gathered.amax(dim=2, keepdim=True)
    fractions = (primitive_dominant & valid_core[:, :, None]).sum(dim=1).float()
    fractions = fractions / counts[:, None].float()
    if (
        not bool(torch.isfinite(means).all())
        or not bool(torch.isfinite(fractions).all())
        or not bool(dominant.any(dim=1).all())
        or bool((fractions < 0.0).any())
        or bool((fractions > 1.0).any())
    ):
        raise RuntimeError("raw-unary region specificity invariant failed")
    return RawUnaryRegionSpecificity(
        mean_raw_probability=means.float().cpu().contiguous(),
        dominant_query_mask=dominant.bool().cpu().contiguous(),
        primitive_top1_fraction=fractions.float().cpu().contiguous(),
        valid_core_counts=counts.long().cpu().contiguous(),
    )


__all__ = [
    "CONTRACT_SHA256",
    "RawDominantGraphGate",
    "RawUnaryRegionSpecificity",
    "SCHEMA",
    "raw_unary_region_specificity",
    "specificity_contract",
    "symmetric_raw_dominant_graph_gate",
]
