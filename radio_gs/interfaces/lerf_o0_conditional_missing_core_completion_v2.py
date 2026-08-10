"""FIX6c contract and evaluator-facing cache for conditional completion.

The numerical completion kernel is intentionally shared with FIX6b.  This
module owns a separate contract and cache schema so a cache can never claim
the scene0001+scene0002 selector and scene0003 external PASS while carrying
the older scene0001+scene0004 lineage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.lerf_o0_conditional_missing_core_completion import (
    ConditionalMissingCoreCompletionResult,
    MAXIMUM_MEMBERSHIP_EXPANSION,
    O0_CORE_SUPERMAJORITY,
    O0_SCORE_MINIMUM,
    conditional_missing_core_completion,
)
from radio_gs.interfaces.source_monotone_missing_core_selector import (
    SELECTOR_FEATURE_NAMES,
    SOURCE_UNIT_FEATURE_INDICES,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256
from radio_gs.utils.immutable_artifacts import validate_file_record


SCHEMA = "radio_gs.lerf_o0_conditional_missing_core_completion.v2"
EXTERNAL_CACHE_SCHEMA = (
    "radio_gs.lerf_o0_conditional_missing_core_completion_external_scores.v2"
)
SOURCE_INPUT_NAMES = (
    "multisource_selector_authority",
    "multisource_selector_model",
    "multisource_selector_report",
    "scene0003_pass_authority",
    "scene0003_pass_report",
    "scene0003_pass_unit_table",
)
TARGET_INPUT_NAMES = (
    "exact_o0_cache",
    "target_accepted_v2",
    "target_capability_descriptor",
    "factorized_primitive_state",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def completion_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "proposal": (
            "exact_O0_valid_core_score_less_than_or_equal_to_0p6_in_region_"
            "whose_exact_O0_positive_fraction_is_at_least_0p75"
        ),
        "selector": {
            "model": (
                "source_scene0001_scene0002_three_fold_multiscene_"
                "monotone_additive_logistic_v2"
            ),
            "validation": "strict_external_source_scene0003_PASS_required",
            "feature_names": list(SELECTOR_FEATURE_NAMES),
            "source_unit_feature_indices": list(SOURCE_UNIT_FEATURE_INDICES),
            "probability": "minimum_probability_across_three_fold_models",
            "threshold": (
                "exact_validated_multiscene_v2_model_threshold_inclusive"
            ),
            "refit_on_target": False,
        },
        "unit_to_cell_overlap": (
            "a_cell_is_selected_if_any_proposal_for_that_cell_is_selected"
        ),
        "completion_value": (
            "qualified_region_exact_O0_positive_fraction_from_selected_"
            "proposals_only"
        ),
        "overlap": "pointwise_maximum_over_selected_proposals_only",
        "fusion": "pointwise_maximum_exact_O0_and_selected_completion",
        "fallback": "outside_selected_unique_cell_union_is_bitwise_exact_O0",
        "query_order_equivariant": True,
        "graph_or_relation": False,
        "cross_query_argmax": False,
        "scene_or_query_identifier_feature": False,
        "instance_or_target_label_feature": False,
        "target_metrics_used": False,
    }


CONTRACT_SHA256 = canonical_json_sha256(completion_contract())


def access_audit() -> dict[str, bool]:
    return {
        "exact_O0_cache_opened": True,
        "query_independent_region_capability_and_geometry_opened": True,
        "source_scene0001_scene0002_selector_opened": True,
        "source_scene0003_external_PASS_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "threshold_scan_or_refit": False,
        "scene_specific_parameters": False,
    }


def build_external_query_score_cache(
    *,
    result: ConditionalMissingCoreCompletionResult,
    o0_valid: torch.Tensor,
    o0_xyz: torch.Tensor,
    query_names: Sequence[str],
    scene_id: str,
    input_authority: Mapping[str, Mapping[str, str]],
    threshold_inclusive: float,
    threshold_source: Mapping[str, str],
) -> dict[str, Any]:
    scores = result.final_scores.float().cpu().contiguous()
    valid = torch.as_tensor(o0_valid).detach().bool().cpu().contiguous()
    xyz = torch.as_tensor(o0_xyz).detach().float().cpu().contiguous()
    names = tuple(str(value) for value in query_names)
    records = {
        str(name): {"path": str(value["path"]), "sha256": str(value["sha256"])}
        for name, value in input_authority.items()
    }
    threshold_record = {
        "path": str(threshold_source["path"]),
        "sha256": str(threshold_source["sha256"]),
    }
    if (
        set(records) != set((*TARGET_INPUT_NAMES, *SOURCE_INPUT_NAMES))
        or threshold_record != records["multisource_selector_model"]
        or valid.shape != (scores.shape[0],)
        or xyz.shape != (scores.shape[0], 3)
        or len(names) != scores.shape[1]
        or len(set(names)) != len(names)
        or not scene_id
    ):
        raise ValueError("FIX6c external cache axes or lineage differ")
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
                "exact_O0_max_multisource_v2_scene0003_PASS_conditional_completion"
            ),
            "input_authority": records,
            "frozen_threshold_inclusive": float(threshold_inclusive),
            "threshold_source": threshold_record,
            "qualified_anchor_counts": result.qualified_anchor_mask.sum(0)
            .long()
            .tolist(),
            "candidate_units": int(result.unit_region_indices.numel()),
            "selected_units": int(result.selected_unit_mask.sum()),
            "selected_unique_cells": int(result.selected_cell_mask.sum()),
            "strictly_changed_cells": int(result.changed_mask.sum()),
        },
        "access_audit": access_audit(),
    }
    payload["channel_sha256"] = {
        "query_scores": tensor_sha256(scores),
        "valid": tensor_sha256(valid),
        "xyz": tensor_sha256(xyz),
        "query_names": canonical_json_sha256(list(names)),
    }
    return validate_external_query_score_cache(payload)


def validate_external_query_score_cache(value: object) -> dict[str, Any]:
    required = {
        "schema", "contract", "contract_sha256", "query_scores", "valid",
        "xyz", "metadata", "access_audit", "channel_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("FIX6c external cache fields differ")
    payload = dict(value)
    scores = torch.as_tensor(payload["query_scores"])
    valid = torch.as_tensor(payload["valid"])
    xyz = torch.as_tensor(payload["xyz"])
    metadata = payload["metadata"]
    metadata_required = {
        "scene_id", "query_names", "score_semantics", "input_authority",
        "frozen_threshold_inclusive", "threshold_source",
        "qualified_anchor_counts", "candidate_units", "selected_units",
        "selected_unique_cells", "strictly_changed_cells",
    }
    if (
        payload["schema"] != EXTERNAL_CACHE_SCHEMA
        or payload["contract"] != completion_contract()
        or payload["contract_sha256"] != CONTRACT_SHA256
        or payload["access_audit"] != access_audit()
        or not isinstance(metadata, Mapping)
        or set(metadata) != metadata_required
        or scores.device.type != "cpu"
        or scores.dtype != torch.float32
        or scores.ndim != 2
        or min(scores.shape) <= 0
        or valid.device.type != "cpu"
        or valid.dtype != torch.bool
        or valid.shape != (scores.shape[0],)
        or xyz.device.type != "cpu"
        or xyz.dtype != torch.float32
        or xyz.shape != (scores.shape[0], 3)
        or not bool(torch.isfinite(scores).all())
        or bool((scores < 0.0).any())
        or bool((scores > 1.0).any())
        or not bool(torch.isfinite(xyz).all())
    ):
        raise ValueError("FIX6c external cache contract differs")
    names = metadata.get("query_names")
    records = metadata.get("input_authority")
    threshold_source = metadata.get("threshold_source")
    threshold = float(metadata.get("frozen_threshold_inclusive", float("nan")))
    qualified = metadata.get("qualified_anchor_counts")
    counter_names = (
        "candidate_units", "selected_units", "selected_unique_cells",
        "strictly_changed_cells",
    )
    counters = [metadata.get(name) for name in counter_names]
    if (
        metadata.get("score_semantics")
        != "exact_O0_max_multisource_v2_scene0003_PASS_conditional_completion"
        or not isinstance(names, list)
        or len(names) != scores.shape[1]
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or not isinstance(records, Mapping)
        or set(records) != set((*TARGET_INPUT_NAMES, *SOURCE_INPUT_NAMES))
        or threshold_source != records.get("multisource_selector_model")
        or not 0.0 <= threshold <= 1.0
        or not isinstance(metadata.get("scene_id"), str)
        or not metadata["scene_id"]
        or not isinstance(qualified, list)
        or len(qualified) != len(names)
        or any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in qualified)
        or any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counters)
        or counters[1] > counters[0]
        or counters[2] > counters[1]
        or counters[3] != counters[2]
        or counters[2] > scores.numel()
    ):
        raise ValueError("FIX6c external cache metadata differs")
    for name, record in records.items():
        if (
            not isinstance(name, str)
            or not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or not Path(str(record["path"])).is_absolute()
            or str(Path(str(record["path"])).expanduser().resolve())
            != str(record["path"])
            or _SHA256.fullmatch(str(record["sha256"])) is None
        ):
            raise ValueError("FIX6c input authority differs")
        validate_file_record(record, label=f"FIX6c cache input {name}")
    expected_hashes = {
        "query_scores": tensor_sha256(scores),
        "valid": tensor_sha256(valid),
        "xyz": tensor_sha256(xyz),
        "query_names": canonical_json_sha256(names),
    }
    if payload["channel_sha256"] != expected_hashes:
        raise ValueError("FIX6c external cache channel changed")
    return payload


__all__ = [
    "CONTRACT_SHA256", "ConditionalMissingCoreCompletionResult",
    "EXTERNAL_CACHE_SCHEMA", "MAXIMUM_MEMBERSHIP_EXPANSION",
    "O0_CORE_SUPERMAJORITY", "O0_SCORE_MINIMUM", "SCHEMA",
    "SOURCE_INPUT_NAMES", "TARGET_INPUT_NAMES", "access_audit",
    "build_external_query_score_cache", "completion_contract",
    "conditional_missing_core_completion", "validate_external_query_score_cache",
]
