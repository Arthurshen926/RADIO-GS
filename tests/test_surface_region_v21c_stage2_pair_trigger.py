from __future__ import annotations

from pathlib import Path

import pytest

from radio_gs.interfaces import surface_region_v21c_stage2_pair_trigger as trigger
from radio_gs.scripts import build_surface_region_v21c_stage2_execution_authority as builder
from radio_gs.scripts import (
    train_surface_region_v21c_two_stage_constrained_adamw as frozen_stage_i,
)
from radio_gs.scripts import (
    train_surface_region_v21c_stage2_pair_constrained_adamw as trainer,
)
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json
from tests.test_train_surface_region_v21c_two_stage_constrained_adamw import (
    _audit_result,
)


def _with_directional_evidence(value: dict, *, pair_conflicts: int, cosine: float) -> dict:
    for row in value["history"]:
        step = int(row["step"])
        conflict = step <= pair_conflicts
        evidence = row["training"]["adamw_candidate_evidence"]
        evidence["gradient_dot_candidate"] = {
            "absolute": 1.0,
            "pairwise": -1e-4 if conflict else 1e-4,
        }
        evidence["gradient_cosine_candidate"] = {
            "absolute": 0.1,
            "pairwise": cosine,
        }
        evidence["constraint_conflict"] = conflict
    conflict_steps = list(range(1, pair_conflicts + 1))
    value["trigger"] = {
        "audited_steps": 30,
        "minimum_conflict_steps": 16,
        "conflict_steps": conflict_steps,
        "conflict_step_count": pair_conflicts,
        "strict_majority_conflict_confirmed": pair_conflicts >= 16,
        "stage_ii_authorized": pair_conflicts >= 16,
    }
    return value


def test_pair_trigger_requires_majority_and_negative_median() -> None:
    parent = {"path": "/parent.json", "sha256": "a" * 64}
    positive = _with_directional_evidence(
        _audit_result(parent, 16), pair_conflicts=16, cosine=-0.1
    )
    evidence = trigger.require_authorized(positive)
    assert evidence["pair_conflict_step_count"] == 16
    assert evidence["pair_candidate_cosine_median"] == pytest.approx(-0.1)

    too_few = _with_directional_evidence(
        _audit_result(parent, 15), pair_conflicts=15, cosine=-0.1
    )
    with pytest.raises(ValueError, match="formal trigger"):
        trigger.require_authorized(too_few)

    nonnegative_median = _with_directional_evidence(
        _audit_result(parent, 16), pair_conflicts=16, cosine=0.0
    )
    with pytest.raises(ValueError, match="requires pair conflict"):
        trigger.require_authorized(nonnegative_median)


def test_absolute_only_formal_conflict_cannot_authorize_stage_ii() -> None:
    parent = {"path": "/parent.json", "sha256": "a" * 64}
    value = _audit_result(parent, 16)
    for row in value["history"]:
        row["training"]["adamw_candidate_evidence"].update(
            {
                "gradient_dot_candidate": {"absolute": -1e-4, "pairwise": 1e-4},
                "gradient_cosine_candidate": {"absolute": -0.1, "pairwise": 0.1},
            }
        )
    evidence = trigger.validate_and_evaluate(value)
    assert evidence["pair_conflict_step_count"] == 0
    assert evidence["stage_ii_authorized"] is False


def test_stage2_builder_binds_positive_audit_and_rejects_absolute_only(
    tmp_path: Path,
) -> None:
    parent_path = tmp_path / "parent.json"
    parent_path.write_text("{}\n")
    stage_i_path = tmp_path / "stage_i_authority.json"
    stage_i_path.write_text("{}\n")
    parent = file_record(parent_path)
    stage_i = file_record(stage_i_path)

    positive = _with_directional_evidence(
        _audit_result(parent, 16), pair_conflicts=16, cosine=-0.2
    )
    positive["execution_authority"] = stage_i
    positive_path = tmp_path / "positive.json"
    write_frozen_json(positive_path, positive)
    spec = {
        "schema": builder.BUILD_SPEC_SCHEMA,
        "schema_version": 1,
        "parent_v21b_execution_authority": parent,
        "stage_i_execution_authority": stage_i,
        "stage_i_audit_result": file_record(positive_path),
    }
    authority = builder.build(spec)
    assert authority["projection_authorized"] is True
    assert authority["pair_trigger_evidence"]["stage_ii_authorized"] is True
    assert authority["source_access"] == trainer.source_access()

    absolute_only = _audit_result(parent, 16)
    absolute_only["execution_authority"] = stage_i
    for row in absolute_only["history"]:
        row["training"]["adamw_candidate_evidence"].update(
            {
                "gradient_dot_candidate": {"absolute": -1e-4, "pairwise": 1e-4},
                "gradient_cosine_candidate": {"absolute": -0.2, "pairwise": 0.2},
            }
        )
    absolute_path = tmp_path / "absolute.json"
    write_frozen_json(absolute_path, absolute_only)
    spec["stage_i_audit_result"] = file_record(absolute_path)
    with pytest.raises(ValueError, match="requires pair conflict"):
        builder.build(spec)
