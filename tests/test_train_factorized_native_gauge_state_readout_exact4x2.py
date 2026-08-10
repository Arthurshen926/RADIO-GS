from __future__ import annotations

from copy import deepcopy

import pytest

from radio_gs.models.factorized_native_gauge_state_readout import (
    FACTORIZED_NATIVE_READOUT_ARMS,
)
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2 as trainer,
)


SHA = "a" * 64


def _record() -> dict[str, str]:
    return {"path": "/frozen", "sha256": SHA}


def _scene(scene_id: str) -> dict:
    return {
        "scene_id": scene_id,
        "training_shard": _record(),
        "accepted_region_authority": _record(),
        "factorized_state": _record(),
    }


def _authority() -> dict:
    return {
        "schema": trainer.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_source_only_exact4train_2validation",
        **{name: _record() for name in trainer._CODE_RECORD_FIELDS},
        "cohort_authority": _record(),
        "pilot_cohort_region_view_registry": _record(),
        "benchmark_exclusion_manifest": _record(),
        "official_radio_checkpoint": _record(),
        "source_train": [_scene(scene) for scene in trainer.TRAIN_SCENES],
        "source_validation": [
            _scene(scene) for scene in trainer.VALIDATION_SCENES
        ],
        "authorized_arms": list(FACTORIZED_NATIVE_READOUT_ARMS),
        "training_authorized": True,
        "benchmark_execution_authorized": False,
        "source_access": trainer.source_access(),
    }


def _validation(mean: float, p05: float) -> dict:
    return {
        "macro_mean_all_view_cosine": mean,
        "macro_p05_row_mean_all_view_cosine": p05,
        "per_scene": {
            scene: {
                "eligible_rows": 2,
                "mean_all_view_cosine": mean,
                "p05_row_mean_all_view_cosine": p05,
            }
            for scene in trainer.VALIDATION_SCENES
        },
    }


def test_contract_is_exact4x2_query_free_and_does_not_change_legacy() -> None:
    for arm in FACTORIZED_NATIVE_READOUT_ARMS:
        contract = trainer.training_contract(arm)
        assert contract["cohort"]["source_train"] == list(trainer.TRAIN_SCENES)
        assert contract["cohort"]["source_validation"] == list(
            trainer.VALIDATION_SCENES
        )
        assert contract["input"]["raw_radio_vector"] == "prohibited"
        assert contract["input"]["query_or_text"] == "prohibited"
        assert contract["legacy_accepted_v2_default_changed"] is False
        assert contract["benchmark_execution_authorized"] is False
    assert trainer.synthetic_dry_run()["benchmark_opened"] is False


def test_execution_schema_rejects_target_access_scene_drift_and_bad_sha() -> None:
    authority = _authority()
    trainer.validate_execution_authority(authority)

    target = deepcopy(authority)
    target["benchmark_execution_authorized"] = True
    with pytest.raises(ValueError, match="header differs"):
        trainer.validate_execution_authority(target)

    query = deepcopy(authority)
    query["source_access"]["text_queries_opened"] = True
    with pytest.raises(ValueError, match="header differs"):
        trainer.validate_execution_authority(query)

    drift = deepcopy(authority)
    drift["source_validation"][0]["scene_id"] = "figurines"
    with pytest.raises(ValueError, match="scene records differ"):
        trainer.validate_execution_authority(drift)

    bad_sha = deepcopy(authority)
    bad_sha["source_train"][0]["factorized_state"]["sha256"] = "g" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        trainer.validate_execution_authority(bad_sha)

    extra = deepcopy(authority)
    extra["target_scene"] = "teatime"
    with pytest.raises(ValueError, match="header differs"):
        trainer.validate_execution_authority(extra)


def test_selection_requires_macro_and_every_scene_non_regression() -> None:
    baseline = _validation(0.50, 0.30)
    candidate = _validation(0.502, 0.301)
    selected = trainer.attach_selection(candidate, baseline)
    assert selected["selection"]["eligible"] is True

    scene_regression = _validation(0.502, 0.301)
    scene_regression["per_scene"][trainer.VALIDATION_SCENES[0]][
        "p05_row_mean_all_view_cosine"
    ] = 0.29
    assert (
        trainer.attach_selection(scene_regression, baseline)["selection"][
            "eligible"
        ]
        is False
    )


def test_selection_is_best_mean_then_p05_then_earliest() -> None:
    history = [
        {"step": 0, "validation": {"selection": {"eligible": False}}},
        {
            "step": 1,
            "validation": {
                "selection": {"eligible": True},
                "macro_mean_all_view_cosine": 0.6,
                "macro_p05_row_mean_all_view_cosine": 0.4,
            },
        },
        {
            "step": 2,
            "validation": {
                "selection": {"eligible": True},
                "macro_mean_all_view_cosine": 0.6,
                "macro_p05_row_mean_all_view_cosine": 0.41,
            },
        },
        {
            "step": 3,
            "validation": {
                "selection": {"eligible": True},
                "macro_mean_all_view_cosine": 0.6,
                "macro_p05_row_mean_all_view_cosine": 0.41,
            },
        },
    ]
    assert trainer.select_step(history) == 2
