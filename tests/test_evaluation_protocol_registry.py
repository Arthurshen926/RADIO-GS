from copy import deepcopy
from pathlib import Path

import pytest

from radio_gs.scripts.validate_evaluation_protocol_registry import (
    RegistryError,
    load_and_validate,
    validate_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def _valid_registry() -> dict:
    return {
        "schema_version": 1,
        "reporting_policy": {
            "oracle_metrics_are_diagnostic_only": True,
            "incomplete_cohorts_are_strictly_comparable": False,
        },
        "evaluations": {
            "example": {
                "benchmark_family": "example",
                "task": "segmentation",
                "method": "method",
                "completion": "complete",
                "evidence_class": "exact_reproduction",
                "cohort": {"complete": True, "items": 4},
                "protocol": {
                    "aggregation": "scene macro",
                    "metric_domain": "pixels",
                    "calibration": "fixed before test",
                },
                "comparability": {
                    "paper_comparison": "strict",
                    "strict_table_eligible": True,
                    "reasons": [],
                    "protocol_match_to_paper": {
                        "cohort": True,
                        "prompt_or_query": True,
                        "target_visibility": True,
                        "metric_domain": True,
                        "aggregation": True,
                        "calibration": True,
                        "implementation": True,
                    },
                },
                "reported_metrics": [
                    {
                        "name": "miou_percent",
                        "role": "primary",
                        "local": 51.0,
                        "paper": 50.0,
                        "delta_points": 1.0,
                    }
                ],
            }
        },
    }


def test_valid_exact_registry_row_passes():
    validate_registry(_valid_registry())


def test_incomplete_cohort_cannot_be_strict():
    payload = _valid_registry()
    payload["evaluations"]["example"]["cohort"]["complete"] = False
    with pytest.raises(RegistryError, match="complete cohort"):
        validate_registry(payload)


def test_protocol_mismatch_cannot_be_strict():
    payload = _valid_registry()
    payload["evaluations"]["example"]["comparability"][
        "protocol_match_to_paper"
    ]["aggregation"] = False
    with pytest.raises(RegistryError, match="aggregation"):
        validate_registry(payload)


def test_oracle_must_never_select_reported_metric():
    payload = _valid_registry()
    payload["evaluations"]["example"]["oracle_diagnostics"] = {
        "diagnostic_only": True,
        "used_for_reported_metric": True,
        "used_for_model_or_threshold_selection": False,
    }
    with pytest.raises(RegistryError, match="used_for_reported_metric"):
        validate_registry(payload)


def test_diagnostic_row_is_allowed_when_fail_closed():
    payload = _valid_registry()
    row = payload["evaluations"]["example"]
    row["evidence_class"] = "diagnostic"
    row["comparability"]["paper_comparison"] = "diagnostic_only"
    row["comparability"]["strict_table_eligible"] = False
    row["comparability"]["protocol_match_to_paper"]["target_visibility"] = False
    row["comparability"]["reasons"] = ["target RGB is visible at query time"]
    row["oracle_diagnostics"] = {
        "diagnostic_only": True,
        "used_for_reported_metric": False,
        "used_for_model_or_threshold_selection": False,
        "best_iou_percent": 52.0,
    }
    validate_registry(payload)


def test_forbidden_comparison_cannot_hide_paper_delta():
    payload = _valid_registry()
    row = payload["evaluations"]["example"]
    row["evidence_class"] = "diagnostic"
    row["comparability"]["paper_comparison"] = "forbidden"
    row["comparability"]["strict_table_eligible"] = False
    row["comparability"]["reasons"] = ["the benchmark has no matching paper task"]
    with pytest.raises(RegistryError, match="cannot contain paper deltas"):
        validate_registry(payload)


def test_metric_delta_is_verified():
    payload = deepcopy(_valid_registry())
    payload["evaluations"]["example"]["reported_metrics"][0]["delta_points"] = 0.5
    with pytest.raises(RegistryError, match="does not equal"):
        validate_registry(payload)


def test_checked_in_cross_benchmark_registry_is_fail_closed():
    payload = load_and_validate(
        ROOT / "paper" / "artifacts" / "evaluation_protocol_registry_20260731.yaml"
    )
    rows = payload["evaluations"]

    lerf = rows["lerf2d_langsplatv2_exact_camera_20260731"]
    assert lerf["cohort"]["labelled_camera_roles"] == {"train": 15, "test": 7}
    assert lerf["comparability"]["paper_comparison"] == "diagnostic_only"
    assert lerf["comparability"]["strict_table_eligible"] is False

    assert (
        rows["spin_nerf_ludvig_9scene_missing_fork_20260731"]["cohort"][
            "complete"
        ]
        is False
    )
    assert (
        rows["pfpr_ludvig_style_partial6_20260731"]["comparability"][
            "paper_comparison"
        ]
        == "forbidden"
    )
    for row in rows.values():
        if row["comparability"]["paper_comparison"] != "strict":
            assert row["comparability"]["strict_table_eligible"] is False
