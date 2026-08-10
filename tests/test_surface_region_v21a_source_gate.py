from __future__ import annotations

from copy import deepcopy

import pytest

from radio_gs.interfaces import surface_region_v21a_source_gate as gate
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21a_rescue as trainer,
)


def _response_summary(coverage: float = 0.96) -> dict:
    rows = {}
    for scene in trainer.VALIDATION_SCENES:
        rows[scene] = {
            "response_auxiliary_loss": 0.4,
            "response_absolute_relevance_loss": 0.3,
            "response_continuous_pairwise_relevance_loss": 0.2,
            "active_rows": 96,
            "eligible_rows": 100,
            "active_eligible_rows": 96,
            "active_over_eligible_coverage": coverage,
            "angle_radians_p50": 0.01,
            "angle_radians_p95": 0.02,
            "angle_cap_fraction": 0.0,
            "response_authority_hard_negative_pairs": 20,
            "response_pairwise_objective_hard_negative_pairs": 20,
            "response_triplet_objective_hard_negative_pairs": 20,
            "response_pair_trainable_endpoint_coverage": coverage,
        }
    return {
        "scene_count": 2,
        "scene_macro_auxiliary_loss": 0.4,
        "scene_macro_absolute_relevance_loss": 0.3,
        "scene_macro_continuous_pairwise_relevance_loss": 0.2,
        "scene_macro_active_over_eligible_coverage": coverage,
        "minimum_active_over_eligible_coverage": coverage,
        "scene_macro_pair_trainable_endpoint_coverage": coverage,
        "minimum_pair_trainable_endpoint_coverage": coverage,
        "all_authority_pairs_retained": True,
        "per_scene": rows,
    }


def test_response_gate_recomputes_coverage_and_requires_all_validation_pairs() -> None:
    gate._validate_response_summary(_response_summary(), label="valid")
    tampered = _response_summary()
    tampered["minimum_active_over_eligible_coverage"] = 0.97
    with pytest.raises(ValueError, match="minimum coverage"):
        gate._validate_response_summary(tampered, label="tampered")
    tampered = _response_summary()
    tampered["per_scene"][trainer.VALIDATION_SCENES[0]][
        "response_triplet_objective_hard_negative_pairs"
    ] = 19
    with pytest.raises(ValueError, match="denominator or coverage"):
        gate._validate_response_summary(tampered, label="tampered")


def test_response_gate_rejects_impossible_angle_diagnostics() -> None:
    tampered = _response_summary()
    tampered["per_scene"][trainer.VALIDATION_SCENES[0]][
        "angle_radians_p95"
    ] = 0.151
    with pytest.raises(ValueError, match="angle diagnostics"):
        gate._validate_response_summary(tampered, label="tampered")


def test_95_percent_coverage_is_a_checkpoint_selection_constraint() -> None:
    from test_train_surface_region_typed_context_response_listwise_v21a_rescue import (
        _history,
    )

    history = _history()
    for row in history[1:]:
        response = row["validation"]["response_listwise_v21a"]
        response["minimum_active_over_eligible_coverage"] = 0.949
        response["minimum_pair_trainable_endpoint_coverage"] = 0.949
    assert trainer.select_promotion_epoch(history) is None
    assert trainer.select_best_raw_aux_epoch(history) == 30


def test_diagnostic_records_are_two_independent_reconstructable_states() -> None:
    records = {
        "best_raw_aux": {
            "epoch": 17,
            "model_state_dict_sha256": "a" * 64,
            "file": {"path": "/best.pt", "sha256": "b" * 64},
        },
        "final": {
            "epoch": 30,
            "model_state_dict_sha256": "c" * 64,
            "file": {"path": "/final.pt", "sha256": "d" * 64},
        },
    }
    assert gate._diagnostic_records(records) == records
    duplicated = deepcopy(records)
    duplicated.pop("final")
    with pytest.raises(ValueError, match="roles differ"):
        gate._diagnostic_records(duplicated)
