"""Strong-unary completion of exact O0-qualified region cores.

The frozen O0 primitive field is the semantic authority.  A query qualifies a
region only when more than 0.6 of that region's valid core scores are positive
on at least 75 percent of the core.  The qualified region then converts that
support fraction into a primitive unary on its own valid core.  Overlapping
regions use a pointwise maximum and the final score is a pointwise maximum
with exact O0.

No graph, relation, cross-query comparison, or query-dependent parameter is
used.  A query with no qualified region is bitwise exact O0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


SCHEMA = "radio_gs.lerf_o0_anchor_self_completion.v1"
EXTERNAL_CACHE_SCHEMA = "radio_gs.lerf_o0_anchor_self_completion_external_scores.v1"
O0_SCORE_MINIMUM = 0.6
O0_CORE_SUPERMAJORITY = 0.75


def completion_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "canonical_capability": "exact_frozen_O0_primitive_probability",
        "fixed_boundaries": {
            "o0_score_minimum": O0_SCORE_MINIMUM,
            "comparator": "strictly_greater_than",
            "valid_core_supermajority": O0_CORE_SUPERMAJORITY,
            "source": "source_fixed_FIX4B_anchor_contract",
        },
        "qualified_anchor": (
            "valid_core_O0_positive_fraction_at_least_0p75_and_"
            "query_independent_region_eligible"
        ),
        "self_completion": (
            "qualified_anchor_positive_fraction_broadcast_to_its_valid_core"
        ),
        "overlap": "pointwise_maximum_completion_probability",
        "fusion": "pointwise_maximum_exact_O0_and_completion_probability",
        "fallback": "no_qualified_anchor_or_invalid_primitive_is_bitwise_O0",
        "graph": False,
        "relation": False,
        "cross_query_argmax": False,
        "query_order_equivariant": True,
        "query_conditioned_parameters": False,
        "scene_conditioned_parameters": False,
        "target_metrics_used": False,
    }


CONTRACT_SHA256 = canonical_json_sha256(completion_contract())


@dataclass(frozen=True)
class O0AnchorSelfCompletionResult:
    valid_core_counts: torch.Tensor
    o0_positive_fraction: torch.Tensor
    qualified_anchor_mask: torch.Tensor
    completion_probability: torch.Tensor
    final_scores: torch.Tensor
    changed_mask: torch.Tensor


def o0_anchor_self_completion(
    *,
    o0_scores: torch.Tensor,
    region_rows: torch.Tensor,
    core_mask: torch.Tensor,
    primitive_valid_mask: torch.Tensor,
    region_eligible_mask: torch.Tensor,
) -> O0AnchorSelfCompletionResult:
    """Convert O0-qualified region cores into strong, monotone unary scores."""

    scores = torch.as_tensor(o0_scores).detach()
    rows = torch.as_tensor(region_rows).detach()
    core = torch.as_tensor(core_mask).detach()
    valid = torch.as_tensor(primitive_valid_mask).detach()
    eligible = torch.as_tensor(region_eligible_mask).detach()
    if (
        scores.device.type != "cpu"
        or scores.dtype != torch.float32
        or scores.ndim != 2
        or min(scores.shape) <= 0
        or not bool(torch.isfinite(scores).all())
        or bool((scores < 0.0).any())
        or bool((scores > 1.0).any())
        or rows.device.type != "cpu"
        or rows.dtype not in {torch.int32, torch.int64}
        or rows.ndim != 2
        or rows.shape[0] <= 0
        or core.device.type != "cpu"
        or core.dtype != torch.bool
        or core.shape != rows.shape
        or valid.device.type != "cpu"
        or valid.dtype != torch.bool
        or valid.shape != (scores.shape[0],)
        or eligible.device.type != "cpu"
        or eligible.dtype != torch.bool
        or eligible.shape != (rows.shape[0],)
        or not bool(core.any(dim=1).all())
        or bool((rows[core] < 0).any())
        or bool((rows[core] >= scores.shape[0]).any())
    ):
        raise ValueError("O0 anchor self-completion inputs differ")

    safe_rows = rows.long().clamp(min=0, max=scores.shape[0] - 1)
    valid_core = core & valid[safe_rows]
    counts = valid_core.sum(dim=1)
    usable = counts > 0
    positive = (scores[safe_rows] > O0_SCORE_MINIMUM) & valid_core[:, :, None]
    fraction = positive.sum(dim=1).float() / counts.clamp_min(1)[:, None].float()
    qualified = (
        (fraction >= O0_CORE_SUPERMAJORITY)
        & eligible[:, None]
        & usable[:, None]
    )

    completion = torch.zeros_like(scores)
    for region in torch.where(qualified.any(dim=1))[0].tolist():
        active = safe_rows[region, valid_core[region]]
        queries = torch.where(qualified[region])[0]
        for query in queries.tolist():
            value = fraction[region, query]
            completion[active, query] = torch.maximum(
                completion[active, query], value.expand(active.numel())
            )
    completion[~valid] = 0.0
    final = torch.maximum(scores, completion)
    changed = final > scores

    no_anchor = ~qualified.any(dim=0)
    if (
        bool(completion[~valid].count_nonzero())
        or bool(changed[~valid].any())
        or not torch.equal(final[~valid], scores[~valid])
        or not torch.equal(final[:, no_anchor], scores[:, no_anchor])
        or bool((final < scores).any())
    ):
        raise RuntimeError("O0 anchor self-completion invariant failed")
    return O0AnchorSelfCompletionResult(
        valid_core_counts=counts.long().contiguous(),
        o0_positive_fraction=fraction.contiguous(),
        qualified_anchor_mask=qualified.contiguous(),
        completion_probability=completion.contiguous(),
        final_scores=final.contiguous(),
        changed_mask=changed.contiguous(),
    )


def external_cache_access_audit() -> dict[str, bool]:
    return {
        "exact_O0_cache_opened": True,
        "query_independent_region_support_opened": True,
        "query_independent_region_reliability_opened": True,
        "query_names_forwarded_without_inspection": True,
        "cross_query_argmax": False,
        "graph_or_relation_applied": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "threshold_scan": False,
        "scene_specific_parameters": False,
    }


def build_external_query_score_cache(
    *,
    result: O0AnchorSelfCompletionResult,
    o0_valid: torch.Tensor,
    o0_xyz: torch.Tensor,
    query_names: Sequence[str],
    scene_id: str,
    input_authority: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    scores = torch.as_tensor(result.final_scores).detach().float().cpu().contiguous()
    valid = torch.as_tensor(o0_valid).detach().bool().cpu().contiguous()
    xyz = torch.as_tensor(o0_xyz).detach().float().cpu().contiguous()
    names = tuple(str(value) for value in query_names)
    records = {
        str(name): {"path": str(value["path"]), "sha256": str(value["sha256"])}
        for name, value in input_authority.items()
    }
    if (
        valid.shape != (scores.shape[0],)
        or xyz.shape != (scores.shape[0], 3)
        or len(names) != scores.shape[1]
        or len(set(names)) != len(names)
        or not scene_id
        or set(records)
        != {
            "exact_o0_cache",
            "positive_o0_cache",
            "negative_o0_cache",
            "region_features",
            "target_descriptor",
            "factorized_primitive_state",
            "renderer_geometry_checkpoint",
        }
    ):
        raise ValueError("O0 self-completion external cache axes differ")
    payload = {
        "schema": EXTERNAL_CACHE_SCHEMA,
        "contract": completion_contract(),
        "contract_sha256": CONTRACT_SHA256,
        "query_scores": scores,
        "valid": valid,
        "xyz": xyz,
        "metadata": {
            "scene_id": str(scene_id),
            "query_names": list(names),
            "score_semantics": (
                "exact_O0_pointwise_max_source_fixed_anchor_self_completion"
            ),
            "input_authority": records,
            "qualified_anchor_counts": result.qualified_anchor_mask.sum(dim=0)
            .long()
            .tolist(),
            "strictly_changed_primitive_query_cells": int(
                result.changed_mask.sum()
            ),
            "graph_or_relation": "none",
            "cross_query_argmax": False,
        },
        "access_audit": external_cache_access_audit(),
    }
    payload["channel_sha256"] = {
        "query_scores": tensor_sha256(payload["query_scores"]),
        "valid": tensor_sha256(payload["valid"]),
        "xyz": tensor_sha256(payload["xyz"]),
        "query_names": canonical_json_sha256(payload["metadata"]["query_names"]),
    }
    return validate_external_query_score_cache(payload)


def validate_external_query_score_cache(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "contract",
        "contract_sha256",
        "query_scores",
        "valid",
        "xyz",
        "metadata",
        "access_audit",
        "channel_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("O0 self-completion external cache fields differ")
    payload = dict(value)
    scores = torch.as_tensor(payload["query_scores"])
    valid = torch.as_tensor(payload["valid"])
    xyz = torch.as_tensor(payload["xyz"])
    metadata = payload["metadata"]
    if (
        payload["schema"] != EXTERNAL_CACHE_SCHEMA
        or payload["contract"] != completion_contract()
        or payload["contract_sha256"] != CONTRACT_SHA256
        or payload["access_audit"] != external_cache_access_audit()
        or not isinstance(metadata, Mapping)
        or scores.dtype != torch.float32
        or scores.device.type != "cpu"
        or scores.ndim != 2
        or min(scores.shape) <= 0
        or valid.dtype != torch.bool
        or valid.shape != (scores.shape[0],)
        or xyz.dtype != torch.float32
        or xyz.shape != (scores.shape[0], 3)
        or not bool(torch.isfinite(scores).all())
        or not bool(torch.isfinite(xyz).all())
        or bool((scores < 0.0).any())
        or bool((scores > 1.0).any())
        or len(metadata.get("query_names", ())) != scores.shape[1]
        or metadata.get("graph_or_relation") != "none"
        or metadata.get("cross_query_argmax") is not False
    ):
        raise ValueError("O0 self-completion external cache differs")
    expected = {
        "query_scores": tensor_sha256(scores),
        "valid": tensor_sha256(valid),
        "xyz": tensor_sha256(xyz),
        "query_names": canonical_json_sha256(metadata["query_names"]),
    }
    if payload["channel_sha256"] != expected:
        raise ValueError("O0 self-completion external cache hashes differ")
    return payload

__all__ = [
    "CONTRACT_SHA256",
    "EXTERNAL_CACHE_SCHEMA",
    "O0AnchorSelfCompletionResult",
    "O0_CORE_SUPERMAJORITY",
    "O0_SCORE_MINIMUM",
    "SCHEMA",
    "completion_contract",
    "build_external_query_score_cache",
    "external_cache_access_audit",
    "o0_anchor_self_completion",
    "validate_external_query_score_cache",
]
