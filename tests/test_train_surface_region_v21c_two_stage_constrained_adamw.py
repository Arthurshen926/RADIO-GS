from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from radio_gs.scripts import build_surface_region_v21c_execution_authority as builder
from radio_gs.scripts import (
    train_surface_region_v21c_two_stage_constrained_adamw as trainer,
)
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json


def _validation(aux: float, absolute: float, pairwise: float, coverage=0.96):
    per_scene = {
        scene: {
            "response_auxiliary_loss": aux,
            "response_absolute_relevance_loss": absolute,
            "response_continuous_pairwise_relevance_loss": pairwise,
            "active_row_coverage": coverage,
            "response_pair_trainable_endpoint_coverage": coverage,
        }
        for scene in trainer.VALIDATION_SCENES
    }
    return {
        "v1_non_regression": {"non_regression_passed": True},
        "response_listwise_v21b": {
            "scene_macro_auxiliary_loss": aux,
            "scene_macro_absolute_relevance_loss": absolute,
            "scene_macro_continuous_pairwise_relevance_loss": pairwise,
            "per_scene": per_scene,
        },
    }


def test_promotion_requires_pairwise_nonregression_in_every_validation_scene() -> None:
    zero = _validation(1.0, 1.0, 1.0)
    candidate = _validation(0.99, 0.99, 0.99)
    assert trainer.promotion_checks(candidate, zero)["passed"] is True
    candidate["response_listwise_v21b"]["per_scene"][
        trainer.VALIDATION_SCENES[0]
    ]["response_continuous_pairwise_relevance_loss"] = 1.001
    result = trainer.promotion_checks(candidate, zero)
    assert result["passed"] is False
    assert result["checks"]["pairwise_every_scene_non_regression"] is False


def _audit_result(parent_record: dict[str, str], conflict_count: int) -> dict:
    parameter_records = [
        {"name": "p", "shape": [1], "numel": 1, "dtype": "torch.float32"}
    ]
    history = []
    for step in range(1, trainer.OPTIMIZER_STEPS + 1):
        conflict = step <= conflict_count
        history.append(
            {
                "step": step,
                "training": {
                    "scene_count": 4,
                    "equal_scene_weight": 0.25,
                    "per_scene": {},
                    "gradient_evidence": {
                        "gradient_order": ["combined", "absolute", "pairwise"],
                        "gram": [[1.0, 0.0, 0.0]] * 3,
                        "cosine": [[1.0, 0.0, 0.0]] * 3,
                        "norm": {"combined": 1.0, "absolute": 1.0, "pairwise": 1.0},
                    },
                    "adamw_candidate_evidence": {
                        "constraint_conflict": conflict,
                        "candidate_norm": 1.0,
                    },
                },
                "validation": {},
                "model_state_dict_sha256": "a" * 64,
            }
        )
    conflict_steps = list(range(1, conflict_count + 1))
    return {
        "schema": trainer.STAGE_I_RESULT_SCHEMA,
        "schema_version": 1,
        "status": "source_only_stage_i_gradient_conflict_audit_complete",
        "training_contract_sha256": trainer.TRAINING_CONTRACT_SHA256,
        "execution_authority": {"path": "/stage_i.json", "sha256": "b" * 64},
        "parent_v21b_execution_authority": parent_record,
        "parameter_subset": {
            "selection": trainer.projection.PARAMETER_SUBSET_SELECTION,
            "parameter_count": 1,
            "vector_numel": 1,
            "parameter_records_sha256": trainer.canonical_json_sha256(parameter_records),
            "parameter_records": parameter_records,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": trainer.v1_trainer.LEARNING_RATE,
            "weight_decay": trainer.v1_trainer.WEIGHT_DECAY,
            "maximum_gradient_norm": trainer.v1_trainer.MAX_GRADIENT_NORM,
            "ordinary_candidate_applied": True,
            "projection_applied": False,
        },
        "history": history,
        "trigger": {
            "audited_steps": 30,
            "minimum_conflict_steps": 16,
            "conflict_steps": conflict_steps,
            "conflict_step_count": conflict_count,
            "strict_majority_conflict_confirmed": conflict_count >= 16,
            "stage_ii_authorized": conflict_count >= 16,
        },
        "source_access": trainer.source_access(),
        "benchmark_opened": False,
    }


def _spec(parent: dict[str, str], stage: str, audit=None) -> dict:
    return {
        "schema": builder.BUILD_SPEC_SCHEMA,
        "schema_version": 1,
        "stage": stage,
        "parent_v21b_execution_authority": parent,
        "stage_i_audit_result": audit,
    }


def test_stage_i_builder_never_authorizes_projection(tmp_path: Path) -> None:
    parent = tmp_path / "parent.json"
    parent.write_text("{}\n")
    authority = builder.build(_spec(file_record(parent), trainer.STAGE_I))
    assert authority["training_authorized"] is True
    assert authority["projection_authorized"] is False
    assert authority["benchmark_execution_authorized"] is False
    assert authority["stage_i_audit_result"] is None


def test_stage_ii_builder_requires_positive_hash_bound_audit(tmp_path: Path) -> None:
    parent = tmp_path / "parent.json"
    parent.write_text("{}\n")
    parent_record = file_record(parent)
    negative_path = tmp_path / "negative.json"
    write_frozen_json(negative_path, _audit_result(parent_record, 15))
    with pytest.raises(ValueError, match="non-triggering"):
        builder.build(
            _spec(parent_record, trainer.STAGE_II, file_record(negative_path))
        )

    positive_path = tmp_path / "positive.json"
    write_frozen_json(positive_path, _audit_result(parent_record, 16))
    authority = builder.build(
        _spec(parent_record, trainer.STAGE_II, file_record(positive_path))
    )
    assert authority["projection_authorized"] is True
    assert authority["stage_i_audit_result"] == file_record(positive_path)


def test_stage_i_result_trigger_is_recomputed() -> None:
    parent = {"path": "/parent.json", "sha256": "c" * 64}
    value = _audit_result(parent, 16)
    trainer.validate_stage_i_audit_result(value)
    changed = deepcopy(value)
    changed["trigger"]["conflict_step_count"] = 17
    with pytest.raises(ValueError, match="trigger differs"):
        trainer.validate_stage_i_audit_result(changed)
