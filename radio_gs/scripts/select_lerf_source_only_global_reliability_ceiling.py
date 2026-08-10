#!/usr/bin/env python3
"""Select one reliability ceiling from source-only teacher agreement.

The selector consumes at least two distinct scene summaries (or their full
teacher-agreement v2 payloads).  It never opens target data or metrics.  A
candidate is eligible only when its pooled source-only leave-one-view-out
delta is strictly positive and every contributing scene is non-regressing.
The largest eligible angle is selected; otherwise the O1 0.15-radian
baseline is retained.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

from radio_gs.scripts import (
    materialize_lerf_o1_o2_teacher_agreement_streaming_v2 as agreement_v2,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
)


PREREGISTRATION_SCHEMA = (
    "radio_gs.source_only_global_reliability_ceiling_preregistration.v1"
)
SOURCE_SUMMARY_SCHEMA = (
    "radio_gs.lerf_o1_o2_teacher_agreement_source_summary.v1"
)
OUTPUT_SCHEMA = (
    "radio_gs.source_only_global_reliability_ceiling_candidate_authority.v1"
)
SCHEMA_VERSION = 1
CANDIDATE_GRID_RADIANS = (0.15, 0.3, 0.45, 0.6, 0.75)
BASELINE_RADIANS = CANDIDATE_GRID_RADIANS[0]
MINIMUM_DISTINCT_SCENES = 2

EXPECTED_ACCESS_AUDIT = {
    "source_feature_bundle_opened": True,
    "source_responsibility_opened": True,
    "exact_query_axis_opened": True,
    "exact_o0_pair_opened": True,
    "target_images_opened": False,
    "target_ground_truth_opened": False,
    "target_masks_opened": False,
    "target_metrics_opened": False,
    "target_quality_readout_executed": False,
}


def method_contract() -> dict[str, Any]:
    return {
        "schema": OUTPUT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "input_schema": agreement_v2.MEAN_SCHEMA,
        "compact_input_schema": SOURCE_SUMMARY_SCHEMA,
        "teacher_agreement_method_contract_sha256": (
            agreement_v2.METHOD_CONTRACT_SHA256
        ),
        "candidate_grid_radians": list(CANDIDATE_GRID_RADIANS),
        "minimum_distinct_source_scenes": MINIMUM_DISTINCT_SCENES,
        "statistic": (
            "sum_scene(delta_cosine_sum_vs_o1_0p15)/"
            "sum_scene(heldout_scale_observations)"
        ),
        "eligibility": {
            "pooled_mean_delta_strictly_positive": True,
            "every_source_scene_delta_nonnegative": True,
        },
        "selection": "largest_eligible_angle_else_0.15",
        "one_global_ceiling": True,
        "per_scene_ceiling": False,
        "per_query_ceiling": False,
        "query_independent": True,
        "target_data_or_metric_access": False,
        "metric_execution_authorized": False,
        "output_role": "source_only_candidate_authority_next_gate",
    }


METHOD_CONTRACT_SHA256 = canonical_json_sha256(method_contract())


class SourceSpec(NamedTuple):
    scene_id: str
    path: str
    sha256: str
    format: str


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} record differs")
    record = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    validate_file_record(record, label=label)
    return record


def _validate_execution_authority(
    value: object,
    *,
    expected_scene_id: str,
) -> dict[str, str]:
    record = _record(value, label=f"{expected_scene_id} execution authority")
    authority, _, _ = load_json_object(
        record["path"],
        expected_sha256=record["sha256"],
        label=f"{expected_scene_id} teacher-agreement v2 execution authority",
    )
    required = {
        "schema", "schema_version", "status", "scene_id", "implementation",
        "method_contract", "method_contract_sha256",
        "feature_output_bundle_sha256", "inputs", "outputs", "execution",
        "query_free_materialization_authorized", "metric_execution_authorized",
        "access_audit",
    }
    if (
        set(authority) != required
        or authority.get("schema") != agreement_v2.AUTHORITY_SCHEMA
        or authority.get("schema_version") != agreement_v2.SCHEMA_VERSION
        or authority.get("status")
        != "authorized_source_only_premetric_o1_o2_streaming"
        or authority.get("scene_id") != expected_scene_id
        or authority.get("implementation") != agreement_v2.ENTRYPOINT_IMPLEMENTATION
        or authority.get("method_contract") != agreement_v2.method_contract()
        or authority.get("method_contract_sha256")
        != agreement_v2.METHOD_CONTRACT_SHA256
        or authority.get("query_free_materialization_authorized") is not True
        or authority.get("metric_execution_authorized") is not False
        or authority.get("access_audit") != EXPECTED_ACCESS_AUDIT
    ):
        raise ValueError("teacher-agreement v2 execution authority differs")
    return record


def _validate_loo_audit(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source-only LOO audit must be a mapping")
    audit = copy.deepcopy(dict(value))
    agreement_v2.validate_source_only_loo_ceiling_audit(audit)
    candidates = audit.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(
        CANDIDATE_GRID_RADIANS
    ):
        raise ValueError("source-only LOO candidate grid differs")
    for expected_angle, candidate in zip(CANDIDATE_GRID_RADIANS, candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError("source-only LOO candidate differs")
        angle = _finite_number(
            candidate.get("maximum_angle_radians"), label="candidate angle"
        )
        if angle != expected_angle:
            raise ValueError("source-only LOO candidate ordering differs")
    if (
        audit.get("query_independent") is not True
        or audit.get("target_images_labels_masks_metrics_opened") is not False
        or audit.get("candidate_role")
        != "source_only_diagnostic_not_target_authorization"
        or audit.get("target_candidate_authorized") is not False
    ):
        raise ValueError("source-only LOO access contract differs")
    return audit


def _normalized_source(
    *,
    scene_id: str,
    source_record: Mapping[str, str],
    source_format: str,
    teacher_payload_record: Mapping[str, str],
    execution_authority: Mapping[str, str],
    loo_audit: Mapping[str, Any],
    loo_audit_sha256: str,
) -> dict[str, Any]:
    if not scene_id or scene_id.strip() != scene_id:
        raise ValueError("source scene id differs")
    if canonical_json_sha256(loo_audit) != loo_audit_sha256:
        raise ValueError("source-only LOO audit SHA-256 differs")
    return {
        "scene_id": scene_id,
        "source_format": source_format,
        "source": dict(source_record),
        "teacher_payload": dict(teacher_payload_record),
        "execution_authority": dict(execution_authority),
        "source_only_loo_audit_sha256": loo_audit_sha256,
        "source_only_loo_audit": copy.deepcopy(dict(loo_audit)),
    }


def load_teacher_payload_source(spec: SourceSpec) -> dict[str, Any]:
    payload, digest, source = load_torch_mapping(
        spec.path,
        expected_sha256=spec.sha256,
        map_location="cpu",
        label=f"{spec.scene_id} teacher-agreement v2 payload",
    )
    agreement_v2.validate_teacher_payload_v2(payload)
    if (
        payload.get("scene_id") != spec.scene_id
        or payload.get("method_contract_sha256")
        != agreement_v2.METHOD_CONTRACT_SHA256
        or payload.get("access_audit") != EXPECTED_ACCESS_AUDIT
    ):
        raise ValueError("teacher-agreement v2 payload provenance differs")
    execution = _validate_execution_authority(
        payload.get("execution_authority"), expected_scene_id=spec.scene_id
    )
    audit = _validate_loo_audit(payload.get(agreement_v2.LOO_AUDIT_FIELD))
    payload_record = {"path": str(source), "sha256": digest}
    return _normalized_source(
        scene_id=spec.scene_id,
        source_record=payload_record,
        source_format="teacher_payload_v2",
        teacher_payload_record=payload_record,
        execution_authority=execution,
        loo_audit=audit,
        loo_audit_sha256=str(payload.get(agreement_v2.LOO_AUDIT_SHA256_FIELD)),
    )


def load_source_summary(spec: SourceSpec) -> dict[str, Any]:
    summary, digest, source = load_json_object(
        spec.path,
        expected_sha256=spec.sha256,
        label=f"{spec.scene_id} teacher-agreement source summary",
    )
    required = {
        "schema", "schema_version", "status", "scene_id", "summary_producer",
        "teacher_payload", "teacher_payload_schema",
        "teacher_payload_schema_version", "teacher_agreement_method_contract_sha256",
        "execution_authority", agreement_v2.LOO_AUDIT_FIELD,
        agreement_v2.LOO_AUDIT_SHA256_FIELD, "access_audit",
        "query_independent", "target_data_or_metrics_opened",
        "metric_execution_authorized",
    }
    if (
        set(summary) != required
        or summary.get("schema") != SOURCE_SUMMARY_SCHEMA
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("status") != "complete_source_only_compact_summary"
        or summary.get("scene_id") != spec.scene_id
        or summary.get("summary_producer") != file_record(Path(__file__).resolve())
        or summary.get("teacher_payload_schema") != agreement_v2.MEAN_SCHEMA
        or summary.get("teacher_payload_schema_version")
        != agreement_v2.SCHEMA_VERSION
        or summary.get("teacher_agreement_method_contract_sha256")
        != agreement_v2.METHOD_CONTRACT_SHA256
        or summary.get("access_audit") != EXPECTED_ACCESS_AUDIT
        or summary.get("query_independent") is not True
        or summary.get("target_data_or_metrics_opened") is not False
        or summary.get("metric_execution_authorized") is not False
    ):
        raise ValueError("teacher-agreement source summary differs")
    teacher_payload = _record(
        summary.get("teacher_payload"), label=f"{spec.scene_id} teacher payload"
    )
    execution = _validate_execution_authority(
        summary.get("execution_authority"), expected_scene_id=spec.scene_id
    )
    audit = _validate_loo_audit(summary.get(agreement_v2.LOO_AUDIT_FIELD))
    return _normalized_source(
        scene_id=spec.scene_id,
        source_record={"path": str(source), "sha256": digest},
        source_format="compact_source_summary_v1",
        teacher_payload_record=teacher_payload,
        execution_authority=execution,
        loo_audit=audit,
        loo_audit_sha256=str(summary.get(agreement_v2.LOO_AUDIT_SHA256_FIELD)),
    )


def build_compact_summary(spec: SourceSpec, *, output: str | Path) -> dict[str, Any]:
    source = load_teacher_payload_source(spec)
    payload = {
        "schema": SOURCE_SUMMARY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_source_only_compact_summary",
        "scene_id": source["scene_id"],
        "summary_producer": file_record(Path(__file__).resolve()),
        "teacher_payload": source["teacher_payload"],
        "teacher_payload_schema": agreement_v2.MEAN_SCHEMA,
        "teacher_payload_schema_version": agreement_v2.SCHEMA_VERSION,
        "teacher_agreement_method_contract_sha256": (
            agreement_v2.METHOD_CONTRACT_SHA256
        ),
        "execution_authority": source["execution_authority"],
        agreement_v2.LOO_AUDIT_FIELD: source["source_only_loo_audit"],
        agreement_v2.LOO_AUDIT_SHA256_FIELD: source[
            "source_only_loo_audit_sha256"
        ],
        "access_audit": dict(EXPECTED_ACCESS_AUDIT),
        "query_independent": True,
        "target_data_or_metrics_opened": False,
        "metric_execution_authorized": False,
    }
    write_frozen_json(output, payload)
    return {"status": "complete", "summary": file_record(output)}


def validate_preregistration(path: str, sha256: str) -> dict[str, str]:
    prereg, digest, source = load_json_object(
        path,
        expected_sha256=sha256,
        label="source-only global ceiling preregistration",
    )
    required = {
        "schema", "schema_version", "status", "selector_implementation",
        "method_contract", "method_contract_sha256", "registration_scope",
        "actual_teacher_agreement_results_opened", "target_data_or_metrics_opened",
        "metric_execution_authorized", "next_gate",
    }
    if (
        set(prereg) != required
        or prereg.get("schema") != PREREGISTRATION_SCHEMA
        or prereg.get("schema_version") != SCHEMA_VERSION
        or prereg.get("status")
        != "preregistered_before_teacher_agreement_v2_result_inspection"
        or prereg.get("selector_implementation")
        != file_record(Path(__file__).resolve())
        or prereg.get("method_contract") != method_contract()
        or prereg.get("method_contract_sha256") != METHOD_CONTRACT_SHA256
        or prereg.get("registration_scope")
        != "two_or_more_distinct_source_scenes_one_global_ceiling"
        or prereg.get("actual_teacher_agreement_results_opened") is not False
        or prereg.get("target_data_or_metrics_opened") is not False
        or prereg.get("metric_execution_authorized") is not False
        or prereg.get("next_gate")
        != "source_only_candidate_execution_authority_before_frozen_target_metric"
    ):
        raise ValueError("source-only global ceiling preregistration differs")
    return {"path": str(source), "sha256": digest}


def select_global_ceiling(
    sources: Sequence[SourceSpec],
    *,
    preregistration: Mapping[str, str],
) -> dict[str, Any]:
    if (
        not isinstance(preregistration, Mapping)
        or set(preregistration) != {"path", "sha256"}
    ):
        raise ValueError("preregistration record differs")
    preregistration = validate_preregistration(
        str(preregistration["path"]), str(preregistration["sha256"])
    )
    if len(sources) < MINIMUM_DISTINCT_SCENES:
        raise ValueError("at least two source scenes are required")
    loaded = []
    for spec in sources:
        if spec.format == "teacher_payload_v2":
            loaded.append(load_teacher_payload_source(spec))
        elif spec.format == "compact_source_summary_v1":
            loaded.append(load_source_summary(spec))
        else:
            raise ValueError("source format differs")
    loaded.sort(key=lambda item: item["scene_id"])
    scene_ids = [item["scene_id"] for item in loaded]
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("source scene ids must be distinct")

    candidate_rows = []
    for index, angle in enumerate(CANDIDATE_GRID_RADIANS):
        per_scene = []
        pooled_delta_sum = 0.0
        pooled_observations = 0
        for item in loaded:
            candidate = item["source_only_loo_audit"]["candidates"][index]
            observations = candidate.get("heldout_scale_observations")
            if isinstance(observations, bool) or not isinstance(observations, int):
                raise ValueError("heldout source observation count differs")
            if observations <= 0:
                raise ValueError("heldout source observation count must be positive")
            delta = _finite_number(
                candidate.get("delta_cosine_sum_vs_o1_0p15"),
                label="source-scene delta cosine sum",
            )
            declared_mean = _finite_number(
                candidate.get("mean_delta_cosine_vs_o1_0p15"),
                label="source-scene mean delta cosine",
            )
            if not math.isclose(
                declared_mean, delta / observations, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError("source-scene mean delta cosine differs")
            declared_nonregression = candidate.get(
                "scene_nonregression_vs_o1_0p15"
            )
            if (
                not isinstance(declared_nonregression, bool)
                or declared_nonregression is not (delta >= 0.0)
            ):
                raise ValueError("source-scene nonregression flag differs")
            per_scene.append(
                {
                    "scene_id": item["scene_id"],
                    "heldout_scale_observations": observations,
                    "delta_cosine_sum_vs_o1_0p15": delta,
                    "mean_delta_cosine_vs_o1_0p15": declared_mean,
                    "nonregression": declared_nonregression,
                }
            )
            pooled_delta_sum += delta
            pooled_observations += observations
        pooled_mean = pooled_delta_sum / pooled_observations
        every_scene_nonregression = all(row["nonregression"] for row in per_scene)
        pooled_improvement = pooled_mean > 0.0
        candidate_rows.append(
            {
                "maximum_angle_radians": angle,
                "pooled_delta_cosine_sum_vs_o1_0p15": pooled_delta_sum,
                "pooled_heldout_scale_observations": pooled_observations,
                "pooled_mean_delta_cosine_vs_o1_0p15": pooled_mean,
                "pooled_improvement_strict": pooled_improvement,
                "every_source_scene_nonregression": every_scene_nonregression,
                "eligible": pooled_improvement and every_scene_nonregression,
                "per_scene": per_scene,
            }
        )
    baseline = candidate_rows[0]
    if (
        baseline["pooled_delta_cosine_sum_vs_o1_0p15"] != 0.0
        or baseline["pooled_mean_delta_cosine_vs_o1_0p15"] != 0.0
        or baseline["eligible"] is not False
    ):
        raise ValueError("0.15-radian source baseline differs")
    eligible = [row for row in candidate_rows if row["eligible"]]
    selected = max(
        (row["maximum_angle_radians"] for row in eligible),
        default=BASELINE_RADIANS,
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "source_only_candidate_selected_metric_not_authorized",
        "selector_implementation": file_record(Path(__file__).resolve()),
        "preregistration": dict(preregistration),
        "method_contract": method_contract(),
        "method_contract_sha256": METHOD_CONTRACT_SHA256,
        "source_scene_ids": scene_ids,
        "source_count": len(scene_ids),
        "inputs": [
            {
                key: copy.deepcopy(item[key])
                for key in (
                    "scene_id", "source_format", "source", "teacher_payload",
                    "execution_authority", "source_only_loo_audit_sha256",
                )
            }
            for item in loaded
        ],
        "candidate_grid": candidate_rows,
        "selection": {
            "global_maximum_angle_radians": selected,
            "selection_rule": "largest_eligible_angle_else_0.15",
            "baseline_fallback_used": len(eligible) == 0,
            "one_global_ceiling": True,
            "per_scene_or_per_query_override_authorized": False,
        },
        "access_audit": {
            "teacher_agreement_payload_or_summary_opened": True,
            "source_only_loo_summary_opened": True,
            "target_images_opened": False,
            "target_labels_or_masks_opened": False,
            "target_metrics_opened": False,
        },
        "query_independent": True,
        "metric_execution_authorized": False,
        "metric_executed": False,
        "candidate_role": "source_only_global_ceiling_candidate_authority",
        "next_gate": (
            "source_only_candidate_execution_authority_before_frozen_target_metric"
        ),
    }


def _specs(values: Sequence[Sequence[str]], *, format: str) -> list[SourceSpec]:
    return [SourceSpec(str(scene), str(path), str(digest), format) for scene, path, digest in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summarize = commands.add_parser("summarize-payload")
    summarize.add_argument("--scene-id", required=True)
    summarize.add_argument("--teacher-payload", required=True)
    summarize.add_argument("--teacher-payload-sha256", required=True)
    summarize.add_argument("--output", required=True)
    select = commands.add_parser("select")
    select.add_argument("--preregistration", required=True)
    select.add_argument("--preregistration-sha256", required=True)
    select.add_argument(
        "--teacher-payload", action="append", nargs=3, default=[],
        metavar=("SCENE_ID", "PATH", "SHA256"),
    )
    select.add_argument(
        "--source-summary", action="append", nargs=3, default=[],
        metavar=("SCENE_ID", "PATH", "SHA256"),
    )
    select.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "summarize-payload":
        result = build_compact_summary(
            SourceSpec(
                args.scene_id, args.teacher_payload,
                args.teacher_payload_sha256, "teacher_payload_v2",
            ),
            output=args.output,
        )
    else:
        preregistration = validate_preregistration(
            args.preregistration, args.preregistration_sha256
        )
        sources = _specs(args.teacher_payload, format="teacher_payload_v2")
        sources += _specs(
            args.source_summary, format="compact_source_summary_v1"
        )
        payload = select_global_ceiling(
            sources, preregistration=preregistration
        )
        write_frozen_json(args.output, payload)
        result = {"status": "complete", "candidate_authority": file_record(args.output)}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "BASELINE_RADIANS",
    "CANDIDATE_GRID_RADIANS",
    "METHOD_CONTRACT_SHA256",
    "OUTPUT_SCHEMA",
    "PREREGISTRATION_SCHEMA",
    "SOURCE_SUMMARY_SCHEMA",
    "SourceSpec",
    "build_compact_summary",
    "load_source_summary",
    "method_contract",
    "select_global_ceiling",
    "validate_preregistration",
]
