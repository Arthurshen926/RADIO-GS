from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from radio_gs.scripts import (
    materialize_lerf_o1_o2_teacher_agreement_streaming_v2 as agreement_v2,
)
from radio_gs.scripts import (
    select_lerf_source_only_global_reliability_ceiling as selector,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    write_frozen_json,
)


def _authority(tmp_path: Path, scene_id: str) -> dict[str, str]:
    path = tmp_path / f"{scene_id}_execution_authority.json"
    write_frozen_json(
        path,
        {
            "schema": agreement_v2.AUTHORITY_SCHEMA,
            "schema_version": agreement_v2.SCHEMA_VERSION,
            "status": "authorized_source_only_premetric_o1_o2_streaming",
            "scene_id": scene_id,
            "implementation": dict(agreement_v2.ENTRYPOINT_IMPLEMENTATION),
            "method_contract": agreement_v2.method_contract(),
            "method_contract_sha256": agreement_v2.METHOD_CONTRACT_SHA256,
            "feature_output_bundle_sha256": "a" * 64,
            "inputs": {},
            "outputs": {},
            "execution": {},
            "query_free_materialization_authorized": True,
            "metric_execution_authorized": False,
            "access_audit": dict(selector.EXPECTED_ACCESS_AUDIT),
        },
    )
    return file_record(path)


def _audit(deltas: list[float]) -> dict[str, object]:
    assert len(deltas) == len(selector.CANDIDATE_GRID_RADIANS)
    observations = 30
    baseline_sum = 18.0
    candidates = []
    for angle, requested_delta in zip(selector.CANDIDATE_GRID_RADIANS, deltas):
        cosine_sum = baseline_sum + requested_delta
        # Match the producer's declared IEEE-754 operation exactly.
        delta = cosine_sum - baseline_sum
        candidates.append(
            {
                "maximum_angle_radians": angle,
                "heldout_scale_observations": observations,
                "cosine_sum": cosine_sum,
                "mean_cosine": cosine_sum / observations,
                "delta_cosine_sum_vs_o1_0p15": delta,
                "mean_delta_cosine_vs_o1_0p15": delta / observations,
                "scene_nonregression_vs_o1_0p15": delta >= 0.0,
            }
        )
    result: dict[str, object] = {
        "schema": "radio_gs.source_only_loo_reliability_ceiling_audit.v1",
        "schema_version": 1,
        "query_independent": True,
        "target_images_labels_masks_metrics_opened": False,
        "candidate_role": "source_only_diagnostic_not_target_authorization",
        "target_candidate_authorized": False,
        "retained_view_capacity": 4,
        "rows": 10,
        "rows_with_valid_loo_prediction": 10,
        "rows_with_expansion_evidence": 9,
        "heldout_predictions": 10,
        "heldout_scale_observations": observations,
        "loo_direction_mean_cosine": 0.8,
        "loo_direction_mean_angular_error_radians": 0.4,
        "candidate_baseline_maximum_angle_radians": 0.15,
        "candidates": candidates,
        "cross_scene_gate": {
            "pooled_statistic": (
                "sum(delta_cosine_sum_vs_o1_0p15)/"
                "sum(heldout_scale_observations)"
            ),
            "pooled_improvement_required": True,
            "every_source_scene_nonregression_required": True,
            "one_global_ceiling_required": True,
            "preregister_before_target_metric": True,
        },
    }
    agreement_v2.validate_source_only_loo_ceiling_audit(result)
    return result


def _summary(
    tmp_path: Path,
    scene_id: str,
    deltas: list[float],
) -> selector.SourceSpec:
    teacher = tmp_path / f"{scene_id}_teacher.pt"
    teacher.write_bytes(f"immutable-{scene_id}".encode("ascii"))
    audit = _audit(deltas)
    path = tmp_path / f"{scene_id}_summary.json"
    write_frozen_json(
        path,
        {
            "schema": selector.SOURCE_SUMMARY_SCHEMA,
            "schema_version": 1,
            "status": "complete_source_only_compact_summary",
            "scene_id": scene_id,
            "summary_producer": file_record(Path(selector.__file__).resolve()),
            "teacher_payload": file_record(teacher),
            "teacher_payload_schema": agreement_v2.MEAN_SCHEMA,
            "teacher_payload_schema_version": agreement_v2.SCHEMA_VERSION,
            "teacher_agreement_method_contract_sha256": (
                agreement_v2.METHOD_CONTRACT_SHA256
            ),
            "execution_authority": _authority(tmp_path, scene_id),
            agreement_v2.LOO_AUDIT_FIELD: audit,
            agreement_v2.LOO_AUDIT_SHA256_FIELD: canonical_json_sha256(audit),
            "access_audit": dict(selector.EXPECTED_ACCESS_AUDIT),
            "query_independent": True,
            "target_data_or_metrics_opened": False,
            "metric_execution_authorized": False,
        },
    )
    record = file_record(path)
    return selector.SourceSpec(
        scene_id, record["path"], record["sha256"],
        "compact_source_summary_v1",
    )


def _preregistration(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "preregistration.json"
    write_frozen_json(
        path,
        {
            "schema": selector.PREREGISTRATION_SCHEMA,
            "schema_version": 1,
            "status": (
                "preregistered_before_teacher_agreement_v2_result_inspection"
            ),
            "selector_implementation": file_record(
                Path(selector.__file__).resolve()
            ),
            "method_contract": selector.method_contract(),
            "method_contract_sha256": selector.METHOD_CONTRACT_SHA256,
            "registration_scope": (
                "two_or_more_distinct_source_scenes_one_global_ceiling"
            ),
            "actual_teacher_agreement_results_opened": False,
            "target_data_or_metrics_opened": False,
            "metric_execution_authorized": False,
            "next_gate": (
                "source_only_candidate_execution_authority_before_frozen_target_metric"
            ),
        },
    )
    return file_record(path)


def test_largest_globally_eligible_ceiling_is_selected_order_independently(
    tmp_path: Path,
) -> None:
    # 0.75 regresses scene_b; 0.60 is the largest joint-safe improvement.
    a = _summary(tmp_path, "scene_a", [0.0, 1.0, 2.0, 3.0, 4.0])
    b = _summary(tmp_path, "scene_b", [0.0, 0.5, 0.2, 0.1, -0.1])
    prereg = _preregistration(tmp_path)
    result = selector.select_global_ceiling([b, a], preregistration=prereg)
    assert result["source_scene_ids"] == ["scene_a", "scene_b"]
    assert result["selection"] == {
        "global_maximum_angle_radians": 0.6,
        "selection_rule": "largest_eligible_angle_else_0.15",
        "baseline_fallback_used": False,
        "one_global_ceiling": True,
        "per_scene_or_per_query_override_authorized": False,
    }
    assert result["candidate_grid"][-1]["pooled_improvement_strict"] is True
    assert result["candidate_grid"][-1][
        "every_source_scene_nonregression"
    ] is False
    assert result["metric_execution_authorized"] is False
    assert result["metric_executed"] is False


def test_no_joint_improvement_falls_back_to_o1_baseline(tmp_path: Path) -> None:
    a = _summary(tmp_path, "scene_a", [0.0, -0.1, -0.2, -0.3, -0.4])
    b = _summary(tmp_path, "scene_b", [0.0, 0.0, 0.0, 0.0, 0.0])
    result = selector.select_global_ceiling(
        [a, b], preregistration=_preregistration(tmp_path)
    )
    assert result["selection"]["global_maximum_angle_radians"] == 0.15
    assert result["selection"]["baseline_fallback_used"] is True


def test_missing_or_duplicate_scene_fails_closed(tmp_path: Path) -> None:
    a = _summary(tmp_path, "scene_a", [0.0, 1.0, 1.0, 1.0, 1.0])
    prereg = _preregistration(tmp_path)
    with pytest.raises(ValueError, match="at least two"):
        selector.select_global_ceiling([a], preregistration=prereg)
    duplicate = selector.SourceSpec(
        a.scene_id, a.path, a.sha256, a.format
    )
    with pytest.raises(ValueError, match="distinct"):
        selector.select_global_ceiling([a, duplicate], preregistration=prereg)


def test_explicit_input_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    a = _summary(tmp_path, "scene_a", [0.0, 1.0, 1.0, 1.0, 1.0])
    b = _summary(tmp_path, "scene_b", [0.0, 1.0, 1.0, 1.0, 1.0])
    forged = selector.SourceSpec(a.scene_id, a.path, "0" * 64, a.format)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        selector.select_global_ceiling(
            [forged, b], preregistration=_preregistration(tmp_path)
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("target_data_or_metrics_opened", True, "source summary differs"),
        ("metric_execution_authorized", True, "source summary differs"),
        ("query_independent", False, "source summary differs"),
    ],
)
def test_forged_summary_access_claim_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    original = _summary(tmp_path, "scene_a", [0.0, 1.0, 1.0, 1.0, 1.0])
    raw = __import__("json").loads(Path(original.path).read_text())
    raw[field] = value
    forged_path = tmp_path / f"forged_{field}.json"
    write_frozen_json(forged_path, raw)
    forged_record = file_record(forged_path)
    forged = selector.SourceSpec(
        "scene_a", forged_record["path"], forged_record["sha256"],
        "compact_source_summary_v1",
    )
    with pytest.raises(ValueError, match=match):
        selector.load_source_summary(forged)


def test_forged_loo_no_target_flag_fails_even_with_recomputed_hash(
    tmp_path: Path,
) -> None:
    original = _summary(tmp_path, "scene_a", [0.0, 1.0, 1.0, 1.0, 1.0])
    raw = __import__("json").loads(Path(original.path).read_text())
    audit = deepcopy(raw[agreement_v2.LOO_AUDIT_FIELD])
    audit["target_images_labels_masks_metrics_opened"] = True
    raw[agreement_v2.LOO_AUDIT_FIELD] = audit
    raw[agreement_v2.LOO_AUDIT_SHA256_FIELD] = canonical_json_sha256(audit)
    forged_path = tmp_path / "forged_loo.json"
    write_frozen_json(forged_path, raw)
    record = file_record(forged_path)
    with pytest.raises(ValueError, match="LOO ceiling audit contract"):
        selector.load_source_summary(
            selector.SourceSpec(
                "scene_a", record["path"], record["sha256"],
                "compact_source_summary_v1",
            )
        )


def test_preregistration_hash_and_contract_fail_closed(tmp_path: Path) -> None:
    record = _preregistration(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        selector.validate_preregistration(record["path"], "f" * 64)
    raw = __import__("json").loads(Path(record["path"]).read_text())
    raw["actual_teacher_agreement_results_opened"] = True
    forged = tmp_path / "forged_preregistration.json"
    write_frozen_json(forged, raw)
    forged_record = file_record(forged)
    with pytest.raises(ValueError, match="preregistration differs"):
        selector.validate_preregistration(
            forged_record["path"], forged_record["sha256"]
        )


def test_contract_is_global_query_free_and_metric_closed() -> None:
    contract = selector.method_contract()
    assert contract["candidate_grid_radians"] == [0.15, 0.3, 0.45, 0.6, 0.75]
    assert contract["selection"] == "largest_eligible_angle_else_0.15"
    assert contract["minimum_distinct_source_scenes"] == 2
    assert contract["one_global_ceiling"] is True
    assert contract["per_scene_ceiling"] is False
    assert contract["per_query_ceiling"] is False
    assert contract["target_data_or_metric_access"] is False
    assert contract["metric_execution_authorized"] is False
    assert selector.METHOD_CONTRACT_SHA256 == canonical_json_sha256(contract)
