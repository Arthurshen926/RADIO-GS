from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from radio_gs.scripts import build_surface_region_v21b_execution_authority as builder
from radio_gs.scripts import (
    train_surface_region_v21b_conditioned_rank256_exact4x2 as trainer,
)
from radio_gs.utils.immutable_artifacts import file_record


SHA = "a" * 64


def _record() -> dict[str, str]:
    return {"path": "/frozen", "sha256": SHA}


def _scene(scene_id: str) -> dict:
    return {
        "scene_id": scene_id,
        "training_shard": _record(),
        "adaptive_context": _record(),
        "hard_negative_authority": _record(),
        "hard_negative_content_authority_sha256": SHA,
    }


def _authority() -> dict:
    code = {name: _record() for name in trainer._CODE_RECORD_FIELDS}
    return {
        "schema": trainer.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_source_only_v21b_exact4train_2validation",
        **code,
        "cohort_authority": _record(),
        "pilot_cohort_region_view_registry": _record(),
        "benchmark_exclusion_manifest": _record(),
        "fit_text_bank": _record(),
        "canonical_negative_bank": _record(),
        "compositional_banks": {
            name: {**_record(), "loss_weight": weight}
            for name, weight in trainer.COMPONENT_WEIGHTS.items()
        },
        "typed_relation_authority": {
            **_record(),
            "content_authority_sha256": SHA,
        },
        "source_train": [_scene(scene) for scene in trainer.TRAIN_SCENES],
        "source_validation": [
            _scene(scene) for scene in trainer.VALIDATION_SCENES
        ],
        "training_authorized": True,
        "benchmark_execution_authorized": False,
        "source_access": trainer.source_access(),
    }


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


def test_contract_fixes_steps_denominators_selection_and_state_retention() -> None:
    contract = trainer.training_contract()
    assert contract["optimizer"]["optimizer_steps"] == 30
    assert contract["optimizer"]["early_stopping"] is False
    assert contract["hard_negative_denominators"] == {
        "pairwise": "any_endpoint_trainable",
        "triplet": "anchor_trainable_only",
        "same_policy_in_training_and_validation": True,
    }
    assert contract["state_retention"] == (
        "step_zero_plus_every_promotion_eligible_state"
    )
    dry = trainer.synthetic_dry_run()
    assert dry["pairwise_any_endpoint_pairs"] == 4
    assert dry["triplet_anchor_trainable_pairs"] == 2


def test_promotion_requires_all_relative_coverage_and_scene_checks() -> None:
    epoch_zero = _validation(1.0, 1.0, 1.0)
    candidate = _validation(0.994, 0.994, 0.994)
    gate = trainer.promotion_checks(candidate, epoch_zero)
    assert gate["passed"] is True
    assert all(value >= 0.005 for value in gate["relative_improvement_from_step_zero"].values())

    low_coverage = _validation(0.90, 0.90, 0.90, coverage=0.949)
    assert trainer.promotion_checks(low_coverage, epoch_zero)["passed"] is False
    scene_regression = _validation(0.90, 0.90, 0.90)
    scene_regression["response_listwise_v21b"]["per_scene"][
        trainer.VALIDATION_SCENES[0]
    ]["response_absolute_relevance_loss"] = 1.01
    assert trainer.promotion_checks(scene_regression, epoch_zero)["passed"] is False


def test_selection_is_minimum_auxiliary_then_earliest_and_none_fail_closed() -> None:
    history = [
        {"step": 0, "validation": {"selection_eligible": False}},
        {
            "step": 1,
            "validation": {
                "selection_eligible": True,
                "response_listwise_v21b": {"scene_macro_auxiliary_loss": 0.8},
            },
        },
        {
            "step": 2,
            "validation": {
                "selection_eligible": True,
                "response_listwise_v21b": {"scene_macro_auxiliary_loss": 0.7},
            },
        },
        {
            "step": 3,
            "validation": {
                "selection_eligible": True,
                "response_listwise_v21b": {"scene_macro_auxiliary_loss": 0.7},
            },
        },
    ]
    assert trainer.select_promotion_step(history) == 2
    none = [{"step": 0, "validation": {"selection_eligible": False}}]
    assert trainer.select_promotion_step(none) is None


def _build_spec(record: dict[str, str]) -> dict:
    return {
        "schema": builder.BUILD_SPEC_SCHEMA,
        "schema_version": builder.SCHEMA_VERSION,
        "cohort_authority": dict(record),
        "pilot_cohort_region_view_registry": dict(record),
        "benchmark_exclusion_manifest": dict(record),
        "fit_text_bank": dict(record),
        "canonical_negative_bank": dict(record),
        "compositional_banks": {
            name: {**record, "loss_weight": weight}
            for name, weight in trainer.COMPONENT_WEIGHTS.items()
        },
        "typed_relation_authority": {
            **record,
            "content_authority_sha256": "1" * 64,
        },
        "source_train": [
            {**_scene(scene), **{
                key: dict(record)
                for key in (
                    "training_shard", "adaptive_context", "hard_negative_authority"
                )
            }}
            for scene in trainer.TRAIN_SCENES
        ],
        "source_validation": [
            {**_scene(scene), **{
                key: dict(record)
                for key in (
                    "training_shard", "adaptive_context", "hard_negative_authority"
                )
            }}
            for scene in trainer.VALIDATION_SCENES
        ],
    }


def test_execution_builder_binds_all_code_preregs_and_refuses_clobber(
    tmp_path: Path,
) -> None:
    dependency = tmp_path / "dependency.bin"
    dependency.write_bytes(b"source-only dependency")
    spec = _build_spec(file_record(dependency))
    authority = builder.build(spec)
    expected_code = {
        name: file_record(path)
        for name, path in trainer._resolved_expected_code_paths().items()
    }
    assert {name: authority[name] for name in expected_code} == expected_code
    assert authority["training_authorized"] is True
    assert authority["benchmark_execution_authorized"] is False
    destination = tmp_path / "authority.json"
    builder.write_authority(spec, destination)
    with pytest.raises(FileExistsError, match="refuses to clobber"):
        builder.write_authority(spec, destination)

    changed = deepcopy(spec)
    changed["source_validation"][0]["scene_id"] = "scene0013_00"
    with pytest.raises(ValueError, match="scene records"):
        builder.build(changed)


def test_execution_authority_rejects_any_benchmark_or_scene_change() -> None:
    value = _authority()
    trainer.validate_execution_authority(value)
    value["benchmark_execution_authorized"] = True
    with pytest.raises(ValueError):
        trainer.validate_execution_authority(value)
