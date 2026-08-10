#!/usr/bin/env python3
"""Execute the frozen source-only selector through a low-memory lineage bridge.

This executor was created after the source results solely because the
preregistered selector accepts the original dense v2 producer identity while
the completed payloads use a bitwise-equivalent, allocation-only low-memory
producer.  It introduces no candidate, statistic, threshold, or selection
freedom and never represents itself as the original selector.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

from radio_gs.scripts import (
    materialize_lerf_o1_o2_streaming as _core,
)
from radio_gs.scripts import (
    materialize_lerf_o1_o2_teacher_agreement_streaming_v2 as _v2,
)
from radio_gs.scripts import (
    materialize_lerf_o1_o2_teacher_agreement_streaming_v2_lowmem as _lowmem,
)
from radio_gs.scripts import (
    select_lerf_source_only_global_reliability_ceiling as _selector,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
)


OUTPUT_SCHEMA = (
    "radio_gs.source_only_global_reliability_ceiling_lowmem_lineage_compatibility.v1"
)
SCHEMA_VERSION = 1
ORIGINAL_SELECTOR_SHA256 = (
    "8cbe1b962cbf3a2aaa35048d6bc5abde6aa64f1d61a7c8d7a46bcdab8eb1a5e1"
)
ORIGINAL_SELECTOR_METHOD_SHA256 = (
    "d46d118aa90d55921409bafba07c1db63f6ceaa2cea7b6fc346be710a7c4cf07"
)
ORIGINAL_PREREGISTRATION_SHA256 = (
    "4e64d1856c456ee19d23c6e1ea2d9ad852511818c36e3767fe0e80e8d3324389"
)
LOWMEM_IMPLEMENTATION_SHA256 = (
    "7c7a2096bb927ae5b6aef99349e4dd66a62782df9f87c290e47ec13f098484a7"
)
LOWMEM_METHOD_SHA256 = (
    "78e4d78293a863ba962440cccfdc5b62dd0b4c9b14e941722df63b708c44cba0"
)
LOWMEM_TESTS_SHA256 = (
    "a742e62d9f4d37add44102d80169a5a84d5965d040ce5aa40fea787dbd5e23cb"
)
AGREEMENT_V2_IMPLEMENTATION_SHA256 = (
    "9bc3d38ebdeb5c28f11804c698a455a144453acfa9e3d51b324f0c9350baf074"
)
AGREEMENT_V2_METHOD_SHA256 = (
    "538da1d1e40051d034360009d9e775b4604b1ec9e93e9da2dc3110f6f3baa79c"
)
ORIGINAL_PREREGISTRATION_PATH = Path(
    "/root/RADIO-GS/paper/artifacts/"
    "lerf_source_only_global_reliability_ceiling_selector_preregistration_20260807.json"
)
LOWMEM_TESTS_PATH = Path(
    "/root/RADIO-GS/tests/"
    "test_materialize_lerf_o1_o2_teacher_agreement_streaming_v2_lowmem.py"
)


class SourceSpec(NamedTuple):
    scene_id: str
    path: str
    sha256: str


def compatibility_contract() -> dict[str, Any]:
    original = _selector.method_contract()
    return {
        "schema": OUTPUT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "role": "post_result_allocation_lineage_compatibility_executor_only",
        "original_selector_method_contract": copy.deepcopy(original),
        "original_selector_method_contract_sha256": (
            ORIGINAL_SELECTOR_METHOD_SHA256
        ),
        "candidate_grid_radians": list(_selector.CANDIDATE_GRID_RADIANS),
        "minimum_distinct_source_scenes": _selector.MINIMUM_DISTINCT_SCENES,
        "statistic": original["statistic"],
        "eligibility": copy.deepcopy(original["eligibility"]),
        "selection": original["selection"],
        "one_global_ceiling": original["one_global_ceiling"],
        "new_selection_freedom": False,
        "lowmem_difference": "allocation_schedule_and_producer_lineage_only",
        "target_data_or_metric_access": False,
        "metric_execution_authorized": False,
    }


COMPATIBILITY_CONTRACT_SHA256 = canonical_json_sha256(compatibility_contract())


def _local_lineage() -> dict[str, dict[str, str]]:
    return {
        "original_selector_implementation": file_record(
            Path(_selector.__file__).resolve()
        ),
        "original_preregistration": file_record(
            ORIGINAL_PREREGISTRATION_PATH
        ),
        "lowmem_implementation": file_record(Path(_lowmem.__file__).resolve()),
        "lowmem_tests": file_record(LOWMEM_TESTS_PATH),
        "agreement_v2_implementation": file_record(Path(_v2.__file__).resolve()),
    }


def validate_local_lineage() -> dict[str, dict[str, str]]:
    records = _local_lineage()
    expected = {
        "original_selector_implementation": ORIGINAL_SELECTOR_SHA256,
        "original_preregistration": ORIGINAL_PREREGISTRATION_SHA256,
        "lowmem_implementation": LOWMEM_IMPLEMENTATION_SHA256,
        "lowmem_tests": LOWMEM_TESTS_SHA256,
        "agreement_v2_implementation": AGREEMENT_V2_IMPLEMENTATION_SHA256,
    }
    contract = _lowmem.method_contract()
    if (
        any(records[name]["sha256"] != digest for name, digest in expected.items())
        or _selector.METHOD_CONTRACT_SHA256
        != ORIGINAL_SELECTOR_METHOD_SHA256
        or _lowmem.METHOD_CONTRACT_SHA256 != LOWMEM_METHOD_SHA256
        or _v2.METHOD_CONTRACT_SHA256 != AGREEMENT_V2_METHOD_SHA256
        or _selector.method_contract()["candidate_grid_radians"]
        != list(_selector.CANDIDATE_GRID_RADIANS)
        or _selector.method_contract()["minimum_distinct_source_scenes"] != 2
        or contract.get("agreement_and_loo_implementation_reused_without_change")
        is not True
        or contract.get("teacher_mean_chunking_affects_method_numerics")
        is not False
        or contract.get("teacher_mean_chunking_changes_only_allocation_schedule")
        is not True
        or contract.get("teacher_mean_finalization")
        != "row_chunked_fp32_canonical_top4_sum_normalize_fp16_v1"
        or contract.get("teacher_agreement_v2_numerical_implementation")
        != records["agreement_v2_implementation"]
        or compatibility_contract()["original_selector_method_contract"]
        != _selector.method_contract()
        or COMPATIBILITY_CONTRACT_SHA256
        != canonical_json_sha256(compatibility_contract())
    ):
        raise RuntimeError("low-memory allocation-lineage compatibility differs")
    return records


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    result = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    validate_file_record(result, label=label)
    return result


def _validate_lowmem_authority(
    value: object,
    *,
    scene_id: str,
    teacher_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    record = _record(value, label=f"{scene_id} lowmem execution authority")
    authority, _, _ = load_json_object(
        record["path"], expected_sha256=record["sha256"],
        label=f"{scene_id} lowmem execution authority",
    )
    required = {
        "schema", "schema_version", "status", "scene_id", "implementation",
        "method_contract", "method_contract_sha256", "feature_output_bundle_sha256",
        "inputs", "outputs", "execution", "query_free_materialization_authorized",
        "metric_execution_authorized", "access_audit",
    }
    expected_execution = {
        "physical_gpu": 0,
        "cuda_visible_devices": "0",
        "program_device": "cuda:0",
        "projection_batch_candidates": [128, 64],
        "pacing_seconds_per_projection_batch": 0.0,
        "thermal_poll_seconds": 300,
        "soft_pause_temperature_c": 0,
        "maximum_temperature_c": 88,
    }
    if (
        set(authority) != required
        or authority.get("schema") != _v2.AUTHORITY_SCHEMA
        or authority.get("schema_version") != _v2.SCHEMA_VERSION
        or authority.get("status")
        != "authorized_source_only_premetric_o1_o2_streaming"
        or authority.get("scene_id") != scene_id
        or authority.get("implementation") != _lowmem.ENTRYPOINT_IMPLEMENTATION
        or authority.get("method_contract") != _lowmem.method_contract()
        or authority.get("method_contract_sha256") != LOWMEM_METHOD_SHA256
        or authority.get("query_free_materialization_authorized") is not True
        or authority.get("metric_execution_authorized") is not False
        or authority.get("access_audit") != _core.access_audit()
        or authority.get("execution") != expected_execution
        or not isinstance(authority.get("outputs"), Mapping)
        or set(authority["outputs"]) != {
            "teacher_mean", "o1_positive", "o1_negative", "o2_positive",
            "o2_negative", "result",
        }
        or authority["outputs"].get("teacher_mean") != str(teacher_path)
    ):
        raise ValueError("lowmem execution authority contract differs")
    expected_inputs = {
        "base_descriptor", "responsibility_authority", "feature_manifest",
        "scene_config", "renderer_geometry_checkpoint", "official_radio_checkpoint",
        "positive_text", "negative_text", "o0_positive", "o0_negative",
        "frozen_metric_config",
    }
    inputs = authority.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
        raise ValueError("lowmem execution authority inputs differ")
    for name in sorted(expected_inputs):
        _record(inputs[name], label=f"{scene_id} lowmem {name}")
    return authority, record


def load_lowmem_source(spec: SourceSpec) -> dict[str, Any]:
    payload, digest, source = load_torch_mapping(
        spec.path, expected_sha256=spec.sha256, map_location="cpu",
        label=f"{spec.scene_id} lowmem teacher-agreement v2 payload",
    )
    _lowmem.validate_teacher_payload_lowmem(payload)
    if (
        payload.get("scene_id") != spec.scene_id
        or payload.get("producer") != _lowmem.ENTRYPOINT_IMPLEMENTATION
        or payload.get("method_contract_sha256") != LOWMEM_METHOD_SHA256
        or payload.get("access_audit") != _core.access_audit()
    ):
        raise ValueError("lowmem teacher payload lineage differs")
    authority, authority_record = _validate_lowmem_authority(
        payload.get("execution_authority"),
        scene_id=spec.scene_id,
        teacher_path=source,
    )
    if payload.get("input_authority", {}).get("base_descriptor") != authority[
        "inputs"
    ]["base_descriptor"]:
        raise ValueError("lowmem payload/base authority lineage differs")
    audit = payload.get(_v2.LOO_AUDIT_FIELD)
    _v2.validate_source_only_loo_ceiling_audit(audit)
    audit_sha = payload.get(_v2.LOO_AUDIT_SHA256_FIELD)
    if canonical_json_sha256(audit) != audit_sha:
        raise ValueError("lowmem source-only LOO audit SHA-256 differs")
    return {
        "scene_id": spec.scene_id,
        "source_format": "teacher_payload_v2_lowmem_allocation_compatible",
        "source": {"path": str(source), "sha256": digest},
        "teacher_payload": {"path": str(source), "sha256": digest},
        "execution_authority": authority_record,
        "source_only_loo_audit_sha256": audit_sha,
        "source_only_loo_audit": copy.deepcopy(audit),
    }


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def select_compatibility_candidate(
    sources: Sequence[SourceSpec],
    *,
    original_preregistration: Mapping[str, str],
) -> dict[str, Any]:
    lineage = validate_local_lineage()
    prereg = _record(original_preregistration, label="original selector preregistration")
    if prereg != lineage["original_preregistration"]:
        raise ValueError("original selector preregistration lineage differs")
    _selector.validate_preregistration(prereg["path"], prereg["sha256"])
    if len(sources) < _selector.MINIMUM_DISTINCT_SCENES:
        raise ValueError("at least two source scenes are required")
    loaded = sorted((load_lowmem_source(spec) for spec in sources), key=lambda x: x["scene_id"])
    scene_ids = [item["scene_id"] for item in loaded]
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("source scene ids must be distinct")

    candidate_rows = []
    for index, angle in enumerate(_selector.CANDIDATE_GRID_RADIANS):
        per_scene = []
        pooled_delta = 0.0
        pooled_observations = 0
        for item in loaded:
            candidate = item["source_only_loo_audit"]["candidates"][index]
            if candidate.get("maximum_angle_radians") != angle:
                raise ValueError("lowmem source candidate grid differs")
            observations = candidate.get("heldout_scale_observations")
            delta = _finite(
                candidate.get("delta_cosine_sum_vs_o1_0p15"),
                label="source delta cosine sum",
            )
            if not isinstance(observations, int) or isinstance(observations, bool) or observations <= 0:
                raise ValueError("source heldout observations differ")
            mean_delta = delta / observations
            if (
                candidate.get("mean_delta_cosine_vs_o1_0p15") != mean_delta
                or candidate.get("scene_nonregression_vs_o1_0p15")
                is not (delta >= 0.0)
            ):
                raise ValueError("source candidate statistic differs")
            per_scene.append({
                "scene_id": item["scene_id"],
                "heldout_scale_observations": observations,
                "delta_cosine_sum_vs_o1_0p15": delta,
                "mean_delta_cosine_vs_o1_0p15": mean_delta,
                "nonregression": delta >= 0.0,
            })
            pooled_delta += delta
            pooled_observations += observations
        pooled_mean = pooled_delta / pooled_observations
        every_nonregression = all(row["nonregression"] for row in per_scene)
        pooled_improvement = pooled_mean > 0.0
        candidate_rows.append({
            "maximum_angle_radians": angle,
            "pooled_delta_cosine_sum_vs_o1_0p15": pooled_delta,
            "pooled_heldout_scale_observations": pooled_observations,
            "pooled_mean_delta_cosine_vs_o1_0p15": pooled_mean,
            "pooled_improvement_strict": pooled_improvement,
            "every_source_scene_nonregression": every_nonregression,
            "eligible": pooled_improvement and every_nonregression,
            "per_scene": per_scene,
        })
    baseline = candidate_rows[0]
    if (
        baseline["pooled_delta_cosine_sum_vs_o1_0p15"] != 0.0
        or baseline["pooled_mean_delta_cosine_vs_o1_0p15"] != 0.0
        or baseline["eligible"] is not False
    ):
        raise ValueError("0.15-radian source baseline differs")
    eligible = [row["maximum_angle_radians"] for row in candidate_rows if row["eligible"]]
    selected = max(eligible, default=_selector.BASELINE_RADIANS)
    return {
        "schema": OUTPUT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_post_result_allocation_lineage_compatibility_selection",
        "compatibility_executor": file_record(Path(__file__).resolve()),
        "compatibility_contract": compatibility_contract(),
        "compatibility_contract_sha256": COMPATIBILITY_CONTRACT_SHA256,
        "created_after_source_results_for_lineage_compatibility_only": True,
        "selection_rule_preregistered_before_results": True,
        "original_selector": {
            "implementation": lineage["original_selector_implementation"],
            "method_contract_sha256": ORIGINAL_SELECTOR_METHOD_SHA256,
        },
        "original_preregistration": prereg,
        "allocation_lineage": {
            "lowmem_implementation": lineage["lowmem_implementation"],
            "lowmem_method_contract_sha256": LOWMEM_METHOD_SHA256,
            "lowmem_tests": lineage["lowmem_tests"],
            "agreement_v2_implementation": lineage["agreement_v2_implementation"],
            "agreement_v2_method_contract_sha256": AGREEMENT_V2_METHOD_SHA256,
            "agreement_and_loo_reused_without_change": True,
            "teacher_mean_chunking_bitwise_equivalence_test_bound": True,
            "allocation_schedule_only": True,
        },
        "source_scene_ids": scene_ids,
        "source_count": len(scene_ids),
        "inputs": [{key: copy.deepcopy(item[key]) for key in (
            "scene_id", "source_format", "source", "teacher_payload",
            "execution_authority", "source_only_loo_audit_sha256",
        )} for item in loaded],
        "candidate_grid": candidate_rows,
        "selection": {
            "global_maximum_angle_radians": selected,
            "selection_rule": "largest_eligible_angle_else_0.15",
            "baseline_fallback_used": not eligible,
            "one_global_ceiling": True,
            "per_scene_or_per_query_override_authorized": False,
        },
        "access_audit": {
            "lowmem_teacher_agreement_payloads_opened": True,
            "source_only_loo_summaries_opened": True,
            "target_images_opened": False,
            "target_labels_or_masks_opened": False,
            "target_metrics_opened": False,
        },
        "query_independent": True,
        "metric_execution_authorized": False,
        "metric_executed": False,
        "candidate_role": "source_only_global_ceiling_candidate_authority_via_lineage_compatibility",
        "next_gate": "source_only_candidate_execution_authority_before_frozen_target_metric",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-preregistration", required=True)
    parser.add_argument("--original-preregistration-sha256", required=True)
    parser.add_argument(
        "--lowmem-teacher-payload", action="append", nargs=3, required=True,
        metavar=("SCENE_ID", "PATH", "SHA256"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sources = [SourceSpec(*values) for values in args.lowmem_teacher_payload]
    result = select_compatibility_candidate(
        sources,
        original_preregistration={
            "path": str(Path(args.original_preregistration).expanduser().resolve()),
            "sha256": args.original_preregistration_sha256,
        },
    )
    write_frozen_json(args.output, result)
    print(json.dumps({"status": "complete", "result": file_record(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "COMPATIBILITY_CONTRACT_SHA256",
    "OUTPUT_SCHEMA",
    "SourceSpec",
    "compatibility_contract",
    "load_lowmem_source",
    "select_compatibility_candidate",
    "validate_local_lineage",
]
