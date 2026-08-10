"""No-GT exact-query premetric contract for a full rank256-to-O0 lift.

The lifted descriptor remains a query-free capability carrier.  This module
opens the frozen text interface only after that carrier has been sealed and
applies the evaluator's exact FP32 cosine, canonical-negative and VALA
readout.  It owns no benchmark label, mask, renderer or metric entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from radio_gs.evaluation.openclip_readout import cosine_logits
from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


SCHEMA = "radio_gs.rank256_o0_full_lift_premetric.v1"
EXTERNAL_CACHE_SCHEMA = "radio_gs.rank256_o0_full_lift_external_scores.v1"
SCORE_BOUNDARY = 0.6
MINIMUM_SUPPORT = 3
MINIMUM_MICRO_O0_SEED_RECALL = 0.8
MINIMUM_MICRO_O0_SEED_PRECISION = 0.75
MAXIMUM_PER_QUERY_EXPANSION = 3.0
MINIMUM_PER_QUERY_PEARSON = 0.85
MAXIMUM_FALLBACK_RAW_DIFFERENCE = 2.0e-6
COSINE_CHUNK_ROWS = 4096
KNN_NEIGHBORS = 10
KNN_CHUNK_ROWS = 65536


def premetric_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "capability_input": "sealed_query_free_full_O0_valid_row_lift",
        "text_interface": {
            "visual_text_score": "independent_normalized_cosine_FP32",
            "positive_text": "frozen_exact_scene_query_subset",
            "negative_text": "frozen_four_canonical_negative_rows",
            "canonical_negative_probability_logit_scale": 10.0,
        },
        "readout": {
            "spatial_smoothing": "VALA_KNN10_on_full_renderer_primitive_axis",
            "scale_selection": "highest_raw_smoothed_peak_per_query",
            "remap": "independent_per_scale_minmax_then_clip_2x_minus_1",
            "external_score_boundary": SCORE_BOUNDARY,
        },
        "fixed_safety_gate": safety_gate_contract(),
        "query_conditioned_parameters": False,
        "scene_conditioned_parameters": False,
        "target_metrics_used": False,
    }


def safety_gate_contract() -> dict[str, Any]:
    return {
        "support": {
            "required_queries": 21,
            "minimum_strictly_above_boundary_per_query": MINIMUM_SUPPORT,
        },
        "O0_seed_agreement": {
            "seed": "exact_O0_valid_and_strictly_above_0p6",
            "candidate": "full_lift_valid_and_strictly_above_0p6",
            "minimum_micro_recall": MINIMUM_MICRO_O0_SEED_RECALL,
            "minimum_micro_precision": MINIMUM_MICRO_O0_SEED_PRECISION,
            "maximum_per_query_expansion": MAXIMUM_PER_QUERY_EXPANSION,
        },
        "continuous_preservation": {
            "minimum_Pearson_per_query_on_O0_valid_rows": (
                MINIMUM_PER_QUERY_PEARSON
            ),
            "maximum_fallback_raw_cosine_difference": (
                MAXIMUM_FALLBACK_RAW_DIFFERENCE
            ),
        },
        "all_finite": True,
        "physical_region_scale_query_axes_exact": True,
        "all_checks_required": True,
    }


CONTRACT_SHA256 = canonical_json_sha256(premetric_contract())


def access_audit() -> dict[str, bool]:
    return {
        "sealed_query_free_full_lift_opened": True,
        "frozen_exact_query_text_opened": True,
        "frozen_O0_positive_negative_scores_opened": True,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "target_metrics_computed": False,
        "metric_execution_authorized": False,
        "threshold_scan": False,
    }


@dataclass(frozen=True)
class RawCosineScores:
    positive: torch.Tensor
    negative: torch.Tensor


def _unit_text(value: object, *, rows: int | None, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach()
    if (
        tensor.dtype != torch.float32
        or tensor.device.type != "cpu"
        or tensor.ndim != 2
        or min(tensor.shape) <= 0
        or (rows is not None and tensor.shape[0] != rows)
        or not bool(torch.isfinite(tensor).all())
    ):
        raise ValueError(f"{label} must be finite CPU FP32 [N,D]")
    tensor = tensor.contiguous()
    norms = torch.linalg.vector_norm(tensor, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=2e-4):
        raise ValueError(f"{label} must be unit L2")
    return tensor


def exact_fp32_cosine_scores(
    lifted_descriptor: torch.Tensor,
    *,
    positive_text: torch.Tensor,
    negative_text: torch.Tensor,
    chunk_rows: int = COSINE_CHUNK_ROWS,
) -> RawCosineScores:
    """Return independent normalized cosine scores without FP16 score storage."""

    descriptor = torch.as_tensor(lifted_descriptor).detach()
    positive = _unit_text(positive_text, rows=None, label="positive text")
    negative = _unit_text(negative_text, rows=4, label="canonical-negative text")
    if (
        descriptor.device.type != "cpu"
        or descriptor.ndim != 3
        or descriptor.shape[1] != 3
        or descriptor.shape[2] != positive.shape[1]
        or descriptor.shape[2] != negative.shape[1]
        or not descriptor.is_floating_point()
        or not bool(torch.isfinite(descriptor).all())
        or isinstance(chunk_rows, bool)
        or int(chunk_rows) <= 0
    ):
        raise ValueError("full-lift descriptor/cosine axes differ")
    primitive_count = int(descriptor.shape[0])
    query_count = int(positive.shape[0])
    positive_scores = torch.empty(
        primitive_count, 3, query_count, dtype=torch.float32
    )
    negative_scores = torch.empty(primitive_count, 3, 4, dtype=torch.float32)
    step = int(chunk_rows)
    for start in range(0, primitive_count, step):
        stop = min(start + step, primitive_count)
        visual = descriptor[start:stop].reshape(-1, descriptor.shape[-1])
        positive_scores[start:stop] = cosine_logits(visual, positive).reshape(
            stop - start, 3, query_count
        )
        negative_scores[start:stop] = cosine_logits(visual, negative).reshape(
            stop - start, 3, 4
        )
    if not bool(torch.isfinite(positive_scores).all()) or not bool(
        torch.isfinite(negative_scores).all()
    ):
        raise FloatingPointError("full-lift FP32 cosine produced non-finite values")
    return RawCosineScores(
        positive=positive_scores.contiguous(),
        negative=negative_scores.contiguous(),
    )


def scatter_sparse_scores(
    sparse: torch.Tensor, *, global_rows: torch.Tensor, total_rows: int
) -> torch.Tensor:
    values = torch.as_tensor(sparse).detach().float().cpu().contiguous()
    rows = torch.as_tensor(global_rows).detach().long().cpu().contiguous()
    count = int(total_rows)
    if (
        values.ndim != 3
        or values.shape[0] != rows.numel()
        or count <= 0
        or bool((rows < 0).any())
        or bool((rows >= count).any())
        or (rows.numel() > 1 and not bool((rows[1:] > rows[:-1]).all()))
    ):
        raise ValueError("sparse full-lift score rows differ")
    result = torch.zeros((count, *values.shape[1:]), dtype=torch.float32)
    result[rows] = values
    return result.contiguous()


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.double()
    y = right.double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) <= 0.0:
        return 1.0 if torch.equal(left, right) else 0.0
    return float((x * y).sum() / denominator)


def build_premetric_audit(
    *,
    candidate_scores: torch.Tensor,
    o0_scores: torch.Tensor,
    valid: torch.Tensor,
    query_ids: Sequence[str],
    candidate_positive_raw_sparse: torch.Tensor,
    candidate_negative_raw_sparse: torch.Tensor,
    o0_positive_raw: torch.Tensor,
    o0_negative_raw: torch.Tensor,
    primitive_global_rows: torch.Tensor,
    fallback_mask: torch.Tensor,
    axes_exact: bool,
) -> dict[str, Any]:
    candidate = torch.as_tensor(candidate_scores).detach().float().cpu().contiguous()
    baseline = torch.as_tensor(o0_scores).detach().float().cpu().contiguous()
    keep = torch.as_tensor(valid).detach().bool().cpu().contiguous()
    names = [str(value) for value in query_ids]
    rows = torch.as_tensor(primitive_global_rows).detach().long().cpu().contiguous()
    fallback = torch.as_tensor(fallback_mask).detach().bool().cpu().contiguous()
    candidate_positive = (
        torch.as_tensor(candidate_positive_raw_sparse)
        .detach()
        .float()
        .cpu()
        .contiguous()
    )
    candidate_negative = (
        torch.as_tensor(candidate_negative_raw_sparse)
        .detach()
        .float()
        .cpu()
        .contiguous()
    )
    o0_positive = torch.as_tensor(o0_positive_raw).detach().float().cpu().contiguous()
    o0_negative = torch.as_tensor(o0_negative_raw).detach().float().cpu().contiguous()
    query_count = len(names)
    all_finite = all(
        bool(torch.isfinite(value).all())
        for value in (
            candidate,
            baseline,
            candidate_positive,
            candidate_negative,
            o0_positive,
            o0_negative,
        )
    )
    if (
        candidate.shape != baseline.shape
        or candidate.ndim != 2
        or candidate.shape[1] != query_count
        or query_count != 21
        or len(set(names)) != query_count
        or keep.shape != (candidate.shape[0],)
        or not bool(keep.any())
        or candidate_positive.shape != (rows.numel(), 3, query_count)
        or candidate_negative.shape != (rows.numel(), 3, 4)
        or o0_positive.shape != (candidate.shape[0], 3, query_count)
        or o0_negative.shape != (candidate.shape[0], 3, 4)
        or fallback.shape != (rows.numel(), 3)
        or rows.numel() != int(keep.sum())
        or not torch.equal(rows, torch.where(keep)[0])
        or type(axes_exact) is not bool
    ):
        raise ValueError("full-lift premetric audit axes differ")

    positive_diff = (
        candidate_positive - o0_positive[rows]
    ).abs()[fallback.unsqueeze(-1).expand_as(candidate_positive)]
    negative_diff = (
        candidate_negative - o0_negative[rows]
    ).abs()[fallback.unsqueeze(-1).expand_as(candidate_negative)]
    fallback_raw_max = max(
        float(positive_diff.max()) if positive_diff.numel() else 0.0,
        float(negative_diff.max()) if negative_diff.numel() else 0.0,
    )

    candidate_membership = (candidate > SCORE_BOUNDARY) & keep[:, None]
    o0_membership = (baseline > SCORE_BOUNDARY) & keep[:, None]
    intersection = candidate_membership & o0_membership
    candidate_counts = candidate_membership.sum(dim=0).long()
    o0_counts = o0_membership.sum(dim=0).long()
    intersection_counts = intersection.sum(dim=0).long()
    total_candidate = int(candidate_counts.sum())
    total_o0 = int(o0_counts.sum())
    total_intersection = int(intersection_counts.sum())
    micro_precision = total_intersection / total_candidate if total_candidate else 0.0
    micro_recall = total_intersection / total_o0 if total_o0 else 0.0
    expansion = [
        (float(candidate_counts[index]) / float(o0_counts[index]))
        if int(o0_counts[index]) > 0
        else (0.0 if int(candidate_counts[index]) == 0 else float("inf"))
        for index in range(query_count)
    ]
    pearson = [
        _pearson(candidate[keep, index], baseline[keep, index])
        for index in range(query_count)
    ]
    support = candidate_counts >= MINIMUM_SUPPORT
    checks = {
        "all_21_queries_supported": bool(support.all()),
        "micro_o0_seed_recall_at_least_0p8": (
            micro_recall >= MINIMUM_MICRO_O0_SEED_RECALL
        ),
        "micro_o0_seed_precision_at_least_0p75": (
            micro_precision >= MINIMUM_MICRO_O0_SEED_PRECISION
        ),
        "every_query_expansion_at_most_3x": all(
            value <= MAXIMUM_PER_QUERY_EXPANSION for value in expansion
        ),
        "minimum_query_Pearson_at_least_0p85": (
            min(pearson) >= MINIMUM_PER_QUERY_PEARSON
        ),
        "fallback_raw_difference_at_most_2e_6": (
            fallback_raw_max <= MAXIMUM_FALLBACK_RAW_DIFFERENCE
        ),
        "all_finite": all_finite,
        "all_axes_exact": axes_exact,
    }
    if any(type(value) is not bool for value in checks.values()):
        raise RuntimeError("full-lift premetric checks are not strict booleans")
    passed = all(checks.values())
    per_query = []
    for index, name in enumerate(names):
        per_query.append(
            {
                "query_id": name,
                "candidate_positive_primitives": int(candidate_counts[index]),
                "o0_seed_primitives": int(o0_counts[index]),
                "intersection_primitives": int(intersection_counts[index]),
                "o0_seed_recall": (
                    float(intersection_counts[index]) / float(o0_counts[index])
                    if int(o0_counts[index])
                    else 0.0
                ),
                "o0_seed_precision": (
                    float(intersection_counts[index]) / float(candidate_counts[index])
                    if int(candidate_counts[index])
                    else 0.0
                ),
                "expansion_ratio": expansion[index],
                "pearson": pearson[index],
                "supported": bool(support[index]),
            }
        )
    return {
        "status": "PASS" if passed else "REJECT",
        "premetric_passed": passed,
        "checks": checks,
        "aggregate": {
            "supported_queries": int(support.sum()),
            "query_count": query_count,
            "candidate_positive_primitive_query_cells": total_candidate,
            "o0_seed_primitive_query_cells": total_o0,
            "intersection_primitive_query_cells": total_intersection,
            "micro_o0_seed_precision": micro_precision,
            "micro_o0_seed_recall": micro_recall,
            "maximum_per_query_expansion": max(expansion),
            "minimum_per_query_pearson": min(pearson),
            "fallback_raw_max_abs_difference": fallback_raw_max,
        },
        "per_query": per_query,
    }


def build_external_query_score_cache(
    *,
    query_scores: torch.Tensor,
    valid: torch.Tensor,
    xyz: torch.Tensor,
    query_ids: Sequence[str],
    scene_id: str,
    physical_space_id: str,
    input_authority: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    scores = torch.as_tensor(query_scores).detach().float().cpu().contiguous()
    keep = torch.as_tensor(valid).detach().bool().cpu().contiguous()
    points = torch.as_tensor(xyz).detach().float().cpu().contiguous()
    names = [str(value) for value in query_ids]
    records = {
        str(name): {"path": str(value["path"]), "sha256": str(value["sha256"])}
        for name, value in input_authority.items()
    }
    payload = {
        "schema": EXTERNAL_CACHE_SCHEMA,
        "contract": premetric_contract(),
        "contract_sha256": CONTRACT_SHA256,
        "query_scores": scores,
        "valid": keep,
        "xyz": points,
        "metadata": {
            "scene_id": str(scene_id),
            "physical_space_id": str(physical_space_id),
            "query_names": names,
            "score_semantics": (
                "rank256_O0_full_lift_exact_FP32_cosine_canonical_negative_"
                "KNN10_peak_scale_per_scale_minmax_clip"
            ),
            "input_authority": records,
            "metric_execution_authorized": False,
        },
        "access_audit": access_audit(),
    }
    payload["channel_sha256"] = {
        "query_scores": tensor_sha256(scores),
        "valid": tensor_sha256(keep),
        "xyz": tensor_sha256(points),
        "query_names": canonical_json_sha256(names),
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
        raise ValueError("full-lift external cache fields differ")
    payload = dict(value)
    scores = torch.as_tensor(payload["query_scores"])
    valid = torch.as_tensor(payload["valid"])
    xyz = torch.as_tensor(payload["xyz"])
    metadata = payload["metadata"]
    if (
        payload["schema"] != EXTERNAL_CACHE_SCHEMA
        or payload["contract"] != premetric_contract()
        or payload["contract_sha256"] != CONTRACT_SHA256
        or payload["access_audit"] != access_audit()
        or not isinstance(metadata, Mapping)
        or set(metadata)
        != {
            "scene_id",
            "physical_space_id",
            "query_names",
            "score_semantics",
            "input_authority",
            "metric_execution_authorized",
        }
        or not metadata["scene_id"]
        or not metadata["physical_space_id"]
        or metadata["metric_execution_authorized"] is not False
        or scores.dtype != torch.float32
        or scores.device.type != "cpu"
        or scores.ndim != 2
        or scores.shape[1] != 21
        or valid.dtype != torch.bool
        or valid.shape != (scores.shape[0],)
        or xyz.dtype != torch.float32
        or xyz.shape != (scores.shape[0], 3)
        or not bool(torch.isfinite(scores).all())
        or not bool(torch.isfinite(xyz).all())
        or bool((scores < 0.0).any())
        or bool((scores > 1.0).any())
        or len(metadata.get("query_names", ())) != scores.shape[1]
        or len(set(metadata.get("query_names", ()))) != scores.shape[1]
    ):
        raise ValueError("full-lift external cache differs")
    expected = {
        "query_scores": tensor_sha256(scores),
        "valid": tensor_sha256(valid),
        "xyz": tensor_sha256(xyz),
        "query_names": canonical_json_sha256(metadata["query_names"]),
    }
    if payload["channel_sha256"] != expected:
        raise ValueError("full-lift external cache hashes differ")
    return payload


__all__ = [
    "CONTRACT_SHA256",
    "EXTERNAL_CACHE_SCHEMA",
    "RawCosineScores",
    "access_audit",
    "build_external_query_score_cache",
    "build_premetric_audit",
    "exact_fp32_cosine_scores",
    "premetric_contract",
    "safety_gate_contract",
    "scatter_sparse_scores",
    "validate_external_query_score_cache",
]
