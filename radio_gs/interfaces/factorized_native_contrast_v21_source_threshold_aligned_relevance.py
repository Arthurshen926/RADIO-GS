"""Source-authorized absolute boundary alignment for contrast-V2.1 relevance.

This schema is deliberately independent of the frozen-relative candidate.  It
only shifts the absolute margin boundary using the one global threshold that
passed the frozen source train/validation envelope.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import factorized_native_contrast_v21_lerf_exact as raw_formal
from radio_gs.interfaces import factorized_native_source_global_hard_threshold_envelope as threshold_formal
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
)


EXECUTION_AUTHORITY_SCHEMA = (
    "radio_gs.factorized_native_contrast_v21_source_threshold_aligned_"
    "relevance_execution_authority.v1"
)
RELEVANCE_SCHEMA = (
    "radio_gs.factorized_native_contrast_v21_source_threshold_aligned_relevance.v1"
)
SCHEMA_VERSION = 1
LOGIT_SCALE = 10.0
ALIGNED_BOUNDARY = 0.5
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path, digest = str(value["path"]), str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def access_audit() -> dict[str, bool]:
    return {
        "source_threshold_envelope_opened_and_promoted": True,
        "raw_exact_relevance_opened": True,
        "raw_query_execution_lineage_validated": True,
        "target_descriptor_lineage_validated": True,
        "health_v4_pass_lineage_validated": True,
        "exact_query_lineage_validated": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "ground_truth_opened": False,
        "target_metrics_computed": False,
        "frozen_relative_candidate_opened": False,
        "candidate_selection_or_mixing_performed": False,
    }


def relevance_contract() -> dict[str, Any]:
    return {
        "schema": RELEVANCE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "candidate_family": "source_authorized_absolute_boundary_alignment",
        "antecedent": {
            "raw_schema": raw_formal.QUERY_RELEVANCE_SCHEMA,
            "source_threshold_schema": threshold_formal.RESULT_SCHEMA,
            "source_threshold_requires_promotion": True,
        },
        "transformation": {
            "raw_probability": "sigmoid_10_times_student_margin",
            "recover_margin": "logit_raw_probability_divided_by_10",
            "aligned_score": (
                "sigmoid_10_times_recovered_margin_minus_frozen_train_threshold"
            ),
            "source_threshold_field": "thresholds.train_selected_candidate",
            "output_boundary": ALIGNED_BOUNDARY,
            "raw_probability_at_output_boundary": (
                "sigmoid_10_times_frozen_train_threshold"
            ),
            "dtype": "float32",
            "rank": "strictly_invariant_per_query",
        },
        "parameters": {
            "one_global_source_train_selected_threshold": True,
            "scene_parameters": False,
            "query_parameters": False,
            "target_derived_parameters": False,
            "threshold_scan": False,
        },
        "audit_only": {
            "query_coverage": True,
            "per_query_max": True,
            "per_query_positive_count_at_0p5": True,
            "ground_truth_or_metric": False,
        },
        "candidate_independence": {
            "frozen_relative_path_opened": False,
            "mixed_with_frozen_relative": False,
            "selected_against_frozen_relative": False,
        },
        "access": access_audit(),
    }


RELEVANCE_CONTRACT_SHA256 = canonical_json_sha256(relevance_contract())


def load_promoted_source_threshold_envelope(
    value: object,
) -> tuple[dict[str, Any], dict[str, str]]:
    shaped = record(value, label="source global hard-threshold envelope")
    raw, digest, source = load_json_object(
        shaped["path"],
        expected_sha256=shaped["sha256"],
        label="promoted source global hard-threshold envelope",
    )
    checked = threshold_formal.validate_result(raw)
    verified = {"path": str(source), "sha256": digest}
    if (
        verified != shaped
        or checked["status"] != "source_only_promoted"
        or checked["global_threshold_authorized"] is not True
    ):
        raise ValueError("source global hard-threshold envelope is not promoted")
    return checked, verified


def boundary_align(
    raw_probability: torch.Tensor,
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.as_tensor(raw_probability).detach().to(device="cpu", dtype=torch.float32).contiguous()
    if (
        raw.ndim != 2
        or raw.shape[0] <= 0
        or raw.shape[1] <= 0
        or not bool(torch.isfinite(raw).all())
        or bool((raw <= 0.0).any())
        or bool((raw >= 1.0).any())
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("source-threshold raw relevance is not finite open-unit probability")
    margin64 = torch.logit(raw.double()) / LOGIT_SCALE
    aligned64 = torch.sigmoid(LOGIT_SCALE * (margin64 - float(threshold)))
    margin = margin64.float().contiguous()
    aligned = aligned64.float().contiguous()
    if (
        not bool(torch.isfinite(margin).all())
        or not bool(torch.isfinite(aligned).all())
        or bool((aligned <= 0.0).any())
        or bool((aligned >= 1.0).any())
    ):
        raise ValueError("source-threshold aligned relevance is invalid")
    raw_sorted = torch.sort(raw, dim=0, stable=True).values
    aligned_sorted = torch.sort(aligned, dim=0, stable=True).values
    raw_strict_step = raw_sorted[1:] > raw_sorted[:-1]
    if bool((raw_strict_step & ~(aligned_sorted[1:] > aligned_sorted[:-1])).any()):
        raise ValueError("source-threshold float32 alignment introduced a ranking tie")
    return margin, aligned


def query_coverage_audit(
    *,
    query_ids: list[str],
    raw_probability: torch.Tensor,
    aligned_probability: torch.Tensor,
) -> dict[str, Any]:
    raw = torch.as_tensor(raw_probability).detach().float().cpu().contiguous()
    aligned = torch.as_tensor(aligned_probability).detach().float().cpu().contiguous()
    if (
        raw.shape != aligned.shape
        or raw.ndim != 2
        or len(query_ids) != raw.shape[1]
        or len(set(query_ids)) != len(query_ids)
    ):
        raise ValueError("source-threshold coverage audit axes differ")
    rows: list[dict[str, Any]] = []
    for index, query_id in enumerate(query_ids):
        raw_count = int((raw[:, index] >= ALIGNED_BOUNDARY).sum())
        aligned_count = int((aligned[:, index] >= ALIGNED_BOUNDARY).sum())
        rows.append(
            {
                "query_id": query_id,
                "raw_max": float(raw[:, index].max()),
                "aligned_max": float(aligned[:, index].max()),
                "raw_identity_positive_count": raw_count,
                "aligned_source_threshold_positive_count": aligned_count,
                "positive_count_gain": aligned_count - raw_count,
                "raw_identity_has_positive": raw_count > 0,
                "aligned_source_threshold_has_positive": aligned_count > 0,
            }
        )
    return {
        "regions": int(raw.shape[0]),
        "queries": int(raw.shape[1]),
        "raw_queries_with_positive": sum(
            int(row["raw_identity_has_positive"]) for row in rows
        ),
        "aligned_queries_with_positive": sum(
            int(row["aligned_source_threshold_has_positive"]) for row in rows
        ),
        "per_query": rows,
        "ground_truth_opened": False,
        "metric_computed": False,
    }


def channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "region_row_ids": canonical_json_sha256(value["region_row_ids"]),
        "canonical_region_indices": tensor_sha256(value["canonical_region_indices"]),
        "region_fingerprints": canonical_json_sha256(value["region_fingerprints"]),
        "query_ids": canonical_json_sha256(value["query_ids"]),
        "recovered_margin": tensor_sha256(value["recovered_margin"]),
        "region_boundary_aligned_relevance": tensor_sha256(
            value["region_boundary_aligned_relevance"]
        ),
    }


def validate_relevance(
    value: object,
    *,
    raw_payload: Mapping[str, Any],
    source_threshold_result: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema", "schema_version", "contract", "contract_sha256", "scene_id",
        "physical_space_id", "producer", "execution_authority", "input_authority",
        "source_global_margin_threshold", "raw_probability_boundary",
        "region_row_ids", "canonical_region_indices", "region_fingerprints",
        "query_ids", "recovered_margin", "region_boundary_aligned_relevance",
        "coverage_audit", "rank_invariance_audit", "channel_sha256", "access_audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("source-threshold aligned relevance fields differ")
    payload = dict(value)
    if (
        payload.get("schema") != RELEVANCE_SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != relevance_contract()
        or payload.get("contract_sha256") != RELEVANCE_CONTRACT_SHA256
        or payload.get("access_audit") != access_audit()
    ):
        raise ValueError("source-threshold aligned relevance header differs")
    payload["producer"] = record(payload["producer"], label="aligned relevance producer")
    payload["execution_authority"] = record(
        payload["execution_authority"], label="aligned relevance execution authority"
    )
    input_names = {
        "source_threshold_envelope", "raw_query_relevance",
        "raw_query_execution_authority", "source_result", "source_checkpoint",
        "target_descriptor", "health_v4_audit", "exact_query_manifest",
        "positive_text_cache", "all_query_text_cache", "canonical_negative_bank",
    }
    inputs = payload.get("input_authority")
    if not isinstance(inputs, Mapping) or set(inputs) != input_names:
        raise ValueError("source-threshold aligned relevance inputs differ")
    payload["input_authority"] = {
        name: record(inputs[name], label=f"aligned relevance input {name}")
        for name in sorted(input_names)
    }
    checked_raw = raw_formal.validate_query_relevance(raw_payload)
    threshold = float(source_threshold_result["thresholds"]["train_selected_candidate"])
    raw_boundary = float(torch.sigmoid(torch.tensor(LOGIT_SCALE * threshold, dtype=torch.float64)))
    if (
        source_threshold_result.get("status") != "source_only_promoted"
        or source_threshold_result.get("global_threshold_authorized") is not True
        or not math.isfinite(float(payload.get("source_global_margin_threshold", math.nan)))
        or abs(float(payload["source_global_margin_threshold"]) - threshold) > 1e-15
        or abs(float(payload.get("raw_probability_boundary", math.nan)) - raw_boundary) > 1e-15
        or payload.get("scene_id") != checked_raw["scene_id"]
        or payload.get("physical_space_id") != checked_raw["physical_space_id"]
    ):
        raise ValueError("source-threshold aligned relevance boundary lineage differs")
    for name in ("region_row_ids", "region_fingerprints", "query_ids"):
        if payload.get(name) != checked_raw[name]:
            raise ValueError(f"source-threshold aligned relevance {name} differs")
    if not torch.equal(payload.get("canonical_region_indices"), checked_raw["canonical_region_indices"]):
        raise ValueError("source-threshold canonical region axis differs")
    expected_margin, expected_aligned = boundary_align(
        checked_raw["region_absolute_relevance"], threshold=threshold
    )
    margin = payload.get("recovered_margin")
    aligned = payload.get("region_boundary_aligned_relevance")
    if (
        not torch.is_tensor(margin)
        or not torch.is_tensor(aligned)
        or margin.dtype != torch.float32
        or aligned.dtype != torch.float32
        or margin.device.type != "cpu"
        or aligned.device.type != "cpu"
        or not torch.equal(margin, expected_margin)
        or not torch.equal(aligned, expected_aligned)
        or payload.get("channel_sha256") != channel_sha256(payload)
    ):
        raise ValueError("source-threshold aligned relevance tensor differs")
    expected_coverage = query_coverage_audit(
        query_ids=list(checked_raw["query_ids"]),
        raw_probability=checked_raw["region_absolute_relevance"],
        aligned_probability=expected_aligned,
    )
    if payload.get("coverage_audit") != expected_coverage:
        raise ValueError("source-threshold aligned relevance coverage audit differs")
    if payload.get("rank_invariance_audit") != {
        "per_query_strict_order_preserved": True,
        "queries_checked": len(checked_raw["query_ids"]),
        "regions_checked": len(checked_raw["region_row_ids"]),
        "ranking_normalization": False,
    }:
        raise ValueError("source-threshold rank invariance audit differs")
    return payload


__all__ = [
    "ALIGNED_BOUNDARY", "EXECUTION_AUTHORITY_SCHEMA", "LOGIT_SCALE",
    "RELEVANCE_CONTRACT_SHA256", "RELEVANCE_SCHEMA", "SCHEMA_VERSION",
    "access_audit", "boundary_align", "channel_sha256",
    "load_promoted_source_threshold_envelope", "query_coverage_audit",
    "record", "relevance_contract", "validate_relevance",
]
