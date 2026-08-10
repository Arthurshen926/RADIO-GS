#!/usr/bin/env python3
"""Seal the FIX4B readout with monotone exact-O0 probability fusion (FIX4C).

FIX4C binds and reuses the frozen FIX4B target execution in full.  Its only
method change is endpoint-safe probability fusion:

    candidate = max(exact_O0_probability, sigmoid(fused_logit))

Actual changes are entries strictly greater than exact O0.  Consequently an
O0 endpoint at probability one remains bitwise one and is not counted as a
change.  No source rule, query gate, evidence, selection, residual amplitude,
target input, metric, or GT access is added here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import lerf_o0_anchored_positive_utility_residual as positive
from radio_gs.scripts import build_lerf_o0_anchored_positive_utility_residual_cache as fix4b
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    validate_file_record,
)


SCHEMA = "radio_gs.lerf_o0_anchored_positive_utility_monotone_external_scores.v1"
EXECUTION_SCHEMA = (
    "radio_gs.lerf_o0_anchored_positive_utility_monotone_execution.v1"
)
EXECUTION_STATUS = (
    "authorized_source_fixed_positive_utility_monotone_target_score_cache_only"
)
IMPLEMENTATION = Path(__file__).resolve()
DEPENDENCIES = {
    "frozen_fix4b_builder": Path(fix4b.__file__).resolve(),
    "positive_utility_interface": Path(positive.__file__).resolve(),
}


def fuse_exact_o0_probabilities_monotone(
    o0_scores: torch.Tensor,
    result: positive.O0AnchoredPositiveUtilityResidualResult,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Fuse without decreasing exact O0, including p=1 endpoints."""

    base = torch.as_tensor(o0_scores).detach()
    residual = torch.as_tensor(result.residual_logits).detach()
    logits = torch.as_tensor(result.fused_logits).detach()
    if (
        base.dtype != torch.float32
        or base.device.type != "cpu"
        or residual.shape != base.shape
        or logits.shape != base.shape
        or not bool(torch.isfinite(base).all())
        or bool((base < 0.0).any())
        or bool((base > 1.0).any())
    ):
        raise ValueError("FIX4C monotone fusion inputs differ")
    residual_mask = residual > 0.0
    fused = base.clone()
    proposed = torch.sigmoid(logits[residual_mask])
    fused[residual_mask] = torch.maximum(base[residual_mask], proposed)
    actual_changed = fused > base
    word_changed = fused.view(torch.int32) != base.view(torch.int32)
    selected_non_decreasing = bool(
        (fused[residual_mask] >= base[residual_mask]).all()
    )
    actual_strictly_increasing = bool(
        (fused[actual_changed] > base[actual_changed]).all()
    )
    if (
        not torch.equal(actual_changed, word_changed)
        or bool((actual_changed & ~residual_mask).any())
        or not selected_non_decreasing
        or not actual_strictly_increasing
        or not torch.equal(fused[~residual_mask], base[~residual_mask])
        or not torch.equal(fused[:, ~result.query_gate], base[:, ~result.query_gate])
    ):
        raise RuntimeError("FIX4C exact-O0 monotone fusion invariant failed")
    quantized_no_change = residual_mask & ~actual_changed
    audit = {
        "actual_changed_definition": "candidate_probability_strictly_greater_than_exact_O0",
        "residual_mask_primitive_counts": residual_mask.sum(dim=0).tolist(),
        "residual_mask_primitive_total": int(residual_mask.sum()),
        "actual_changed_primitive_counts": actual_changed.sum(dim=0).tolist(),
        "actual_changed_primitive_total": int(actual_changed.sum()),
        "quantized_no_change_primitive_counts": quantized_no_change.sum(dim=0).tolist(),
        "quantized_no_change_primitive_total": int(quantized_no_change.sum()),
        "selected_updates_non_decreasing": selected_non_decreasing,
        "actual_changes_strictly_increase_exact_O0": actual_strictly_increasing,
    }
    return fused.contiguous(), actual_changed.contiguous(), audit


def _load_and_validate_execution(
    path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="FIX4C target execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "implementation",
        "dependencies",
        "parent_fix4b_target_execution_authority",
        "supersedes_fix4b_cache",
        "supersedes_fix4b_report",
        "fixed_intervention",
        "output_cache",
        "output_report",
        "target_score_cache_authorized",
        "target_quality_execution_authorized",
        "access_audit",
    }
    authority = dict(raw)
    if (
        set(authority) != required
        or authority["schema"] != EXECUTION_SCHEMA
        or authority["schema_version"] != 1
        or authority["status"] != EXECUTION_STATUS
        or authority["target_score_cache_authorized"] is not True
        or authority["target_quality_execution_authorized"] is not False
        or authority["fixed_intervention"]
        != {
            "scope": "probability_fusion_only",
            "candidate": "maximum(exact_O0_probability,sigmoid(fused_logit))",
            "actual_changed": "candidate_probability_strictly_greater_than_exact_O0",
            "endpoint_policy": "exact_O0_probability_one_remains_bitwise_one",
            "all_other_method_and_selection_logic": "bitwise_frozen_FIX4B",
        }
        or authority["access_audit"]
        != {
            "query_names_opened": True,
            "target_images_opened": False,
            "target_quality_data_opened": False,
            "target_quality_readout_executed": False,
        }
    ):
        raise ValueError("FIX4C target execution header differs")
    if validate_file_record(authority["implementation"], label="implementation") != IMPLEMENTATION:
        raise ValueError("FIX4C target implementation differs")
    dependencies = authority["dependencies"]
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(DEPENDENCIES):
        raise ValueError("FIX4C dependency fields differ")
    for name, dependency in DEPENDENCIES.items():
        if validate_file_record(dependencies[name], label=name) != dependency:
            raise ValueError(f"FIX4C dependency differs: {name}")
    parent_record = fix4b.legacy._record_shape(
        authority["parent_fix4b_target_execution_authority"],
        name="parent FIX4B target execution authority",
    )
    parent_path = validate_file_record(
        parent_record, label="parent FIX4B target execution authority"
    )
    parent = fix4b._load_and_validate_execution(
        parent_path, expected_sha256=parent_record["sha256"]
    )
    superseded_cache = fix4b.legacy._record_shape(
        authority["supersedes_fix4b_cache"], name="superseded FIX4B cache"
    )
    superseded_report = fix4b.legacy._record_shape(
        authority["supersedes_fix4b_report"], name="superseded FIX4B report"
    )
    validate_file_record(superseded_cache, label="superseded FIX4B cache")
    validate_file_record(superseded_report, label="superseded FIX4B report")
    if (
        parent["output_cache"] != superseded_cache["path"]
        or parent["output_report"] != superseded_report["path"]
    ):
        raise ValueError("FIX4C superseded FIX4B output binding differs")
    output = fix4b.legacy._output_path(authority["output_cache"], name="output cache")
    report = fix4b.legacy._output_path(authority["output_report"], name="output report")
    if output == report or output in {parent["output_cache"], parent["output_report"]}:
        raise ValueError("FIX4C outputs must be new and distinct")
    parent.update(
        {
            "output_cache": output,
            "output_report": report,
            "verified_record": {"path": str(source), "sha256": digest},
            "parent_fix4b_target_execution_authority": parent_record,
            "supersedes_fix4b_cache": superseded_cache,
            "supersedes_fix4b_report": superseded_report,
            "access_audit": authority["access_audit"],
        }
    )
    return parent


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Reuse frozen FIX4B execution, replacing only fusion and audit writes."""

    execution = _load_and_validate_execution(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    fusion_audit: dict[str, Any] = {}
    original_loader = fix4b._load_and_validate_execution
    original_fusion = fix4b.fuse_exact_o0_probabilities
    original_schema = fix4b.SCHEMA
    original_implementation = fix4b.IMPLEMENTATION
    original_write_torch = fix4b.write_torch_noclobber
    original_write_json = fix4b.write_frozen_json

    def injected_loader(path: str | Path, *, expected_sha256: str):
        del path, expected_sha256
        return execution

    def injected_fusion(o0_scores, result):
        fused, changed, audit = fuse_exact_o0_probabilities_monotone(
            o0_scores, result
        )
        fusion_audit.update(audit)
        return fused, changed

    def write_cache(path, payload):
        payload["metadata"]["score_semantics"] = (
            "exact_O0_VALA_plus_source_fixed_positive_utility_residual_"
            "with_monotone_exact_O0_probability_fusion"
        )
        payload["metadata"]["supersedes_fix4b_cache"] = execution[
            "supersedes_fix4b_cache"
        ]
        payload["metadata"]["producer"] = file_record(IMPLEMENTATION)
        return original_write_torch(path, payload)

    def write_report(path, payload):
        if not fusion_audit:
            raise RuntimeError("FIX4C fusion audit was not materialized")
        payload["status"] = "o0_anchored_positive_utility_monotone_cache_complete"
        payload["monotone_fusion_audit"] = dict(fusion_audit)
        payload["bitwise_invariants"].update(
            {
                "selected_updates_non_decreasing": fusion_audit[
                    "selected_updates_non_decreasing"
                ],
                "actual_changes_strictly_increase_exact_O0": fusion_audit[
                    "actual_changes_strictly_increase_exact_O0"
                ],
            }
        )
        payload["supersedes_fix4b_cache"] = execution["supersedes_fix4b_cache"]
        payload["supersedes_fix4b_report"] = execution["supersedes_fix4b_report"]
        return original_write_json(path, payload)

    try:
        fix4b._load_and_validate_execution = injected_loader
        fix4b.fuse_exact_o0_probabilities = injected_fusion
        fix4b.SCHEMA = SCHEMA
        fix4b.IMPLEMENTATION = IMPLEMENTATION
        fix4b.write_torch_noclobber = write_cache
        fix4b.write_frozen_json = write_report
        return fix4b.run(args)
    finally:
        fix4b._load_and_validate_execution = original_loader
        fix4b.fuse_exact_o0_probabilities = original_fusion
        fix4b.SCHEMA = original_schema
        fix4b.IMPLEMENTATION = original_implementation
        fix4b.write_torch_noclobber = original_write_torch
        fix4b.write_frozen_json = original_write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
