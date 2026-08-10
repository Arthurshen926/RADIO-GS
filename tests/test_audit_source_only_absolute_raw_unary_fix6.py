from __future__ import annotations

import pytest
import torch

from radio_gs.scripts.audit_source_only_absolute_raw_unary_fix6 import (
    _teacher_probability_for_dominant_query,
    _unary_audit,
    source_access,
    validate_execution_authority,
)


def test_teacher_probability_uses_each_regions_exact_dominant_query() -> None:
    mean, count = _teacher_probability_for_dominant_query(
        teacher_descriptors=torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=torch.float32
        ),
        teacher_region_indices=torch.tensor([0, 0, 1]),
        dominant_query_indices=torch.tensor([0, 1]),
        positive_text=torch.eye(2),
        negative_text=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        region_count=2,
    )
    expected_region0 = torch.sigmoid(torch.tensor([5.0, 3.0])).mean()
    expected_region1 = torch.sigmoid(torch.tensor(5.0))
    assert count.tolist() == [2, 1]
    assert mean.tolist() == pytest.approx(
        [float(expected_region0), float(expected_region1)]
    )


def test_unary_audit_exposes_negative_absolute_utility() -> None:
    result = _unary_audit(
        maximum_probability=torch.tensor([0.9, 0.8, 0.7, 0.2]),
        candidate=torch.tensor([True, True, True, False]),
        teacher_true=torch.tensor([True, False, False, True]),
    )
    assert result["candidate_count"] == 3
    assert result["teacher_positive_count"] == 1
    assert result["precision"] == pytest.approx(1.0 / 3.0)
    assert result["teacher_positive_recall"] == pytest.approx(0.5)
    assert result["signed_strength_sum"] == pytest.approx(-0.2)
    assert result["signed_absolute_utility_per_candidate"] < 0.0


def test_authority_forbids_scene_query_argmax_and_target_access() -> None:
    file_record = {"path": __file__, "sha256": "0" * 64}
    authority = {
        "schema": "radio_gs.source_only_absolute_raw_unary_fix6_execution_authority.v1",
        "schema_version": 1,
        "status": "authorized_source_only_absolute_raw_unary_FIX6",
        "implementation": file_record,
        "graph_calibration_authority": file_record,
        "fix4b_result": file_record,
        "fix5_result": file_record,
        "source_v21b_authority": file_record,
        "fixed_candidate": {
            "raw_probability": "sigmoid(10*(query_cosine-hardest_canonical_negative_cosine))",
            "absolute_threshold": 0.5,
            "threshold_source": "semantic_neutral_point_not_fitted",
            "dominance_reference": "same_frozen_target_blind_806_query_bank",
            "dominance": "argmax_all_exact_ties_retained",
            "target_requirement": "runtime_query_must_beat_the_same_806_distractors",
            "scene_query_bank_argmax_used": False,
            "per_scene_parameters": False,
        },
        "promotion_gate": {
            "minimum_every_validation_scene_unary_Wilson95_lower": 0.95,
            "validation_pooled_signed_absolute_utility": "strictly_greater_than_zero",
            "minimum_every_validation_scene_graph_Wilson95_lower": 0.95,
            "every_validation_scene_confirmed_anchor_coverage": "at_least_FIX5_retained_reach",
            "failure_action": "reject_FIX6_keep_target_unopened",
        },
        "source_access": source_access(),
        "benchmark_execution_authorized": False,
    }
    authority["fixed_candidate"]["scene_query_bank_argmax_used"] = True
    with pytest.raises(ValueError, match="header differs"):
        validate_execution_authority(authority)

