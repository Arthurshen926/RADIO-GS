from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from radio_gs.models.region_comembership_native_v3 import RegionCoMembershipNativeV3
from radio_gs.scripts import train_source_region_comembership_native_v3 as trainer
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


SHA = "a" * 64


def _record() -> dict[str, str]:
    return {"path": "/frozen", "sha256": SHA}


def _execution() -> dict:
    return {
        "schema": trainer.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_source_only_native_v3_exact4train_2validation",
        "implementation": _record(),
        "model_implementation": _record(),
        "source_materializer_implementation": _record(),
        "native_interface_implementation": _record(),
        "preregistration": _record(),
        "parent_v2_source_result": _record(),
        "source_train": [
            {"scene_id": scene, "authority": _record()}
            for scene in trainer.TRAIN_SCENES
        ],
        "source_validation": [
            {"scene_id": scene, "authority": _record()}
            for scene in trainer.VALIDATION_SCENES
        ],
        "training_authorized": True,
        "target_execution_authorized": False,
        "source_access": trainer.source_access(),
    }


def _calibration(brier: float, log_loss: float) -> dict:
    return {
        "scene_macro": {
            "brier": brier,
            "log_loss": log_loss,
            "ece10": 0.0,
        },
        "per_scene": {
            scene: {"brier": brier, "log_loss": log_loss, "ece10": 0.0}
            for scene in trainer.VALIDATION_SCENES
        },
    }


def test_execution_authority_freezes_exact_four_plus_two_and_target_closed() -> None:
    value = trainer.validate_execution_authority(_execution())
    assert [row["scene_id"] for row in value["source_train"]] == list(
        trainer.TRAIN_SCENES
    )
    assert [row["scene_id"] for row in value["source_validation"]] == list(
        trainer.VALIDATION_SCENES
    )
    assert value["target_execution_authorized"] is False
    assert value["source_access"]["benchmark_queries_opened"] is False


@pytest.mark.parametrize("tamper", ["train", "validation", "target"])
def test_execution_authority_fails_closed(tamper: str) -> None:
    value = deepcopy(_execution())
    if tamper == "train":
        value["source_train"][0]["scene_id"] = "scene9999_00"
    elif tamper == "validation":
        value["source_validation"].reverse()
    else:
        value["target_execution_authorized"] = True
    with pytest.raises(ValueError):
        trainer.validate_execution_authority(value)


def test_balanced_calibration_rewards_absolute_probability_quality() -> None:
    target = torch.tensor([False, False, True, True])
    evidence = torch.tensor([100.0, 1.0, 1.0, 100.0])
    epoch_zero = trainer.balanced_probability_calibration(
        torch.full((4,), 0.5), target, evidence
    )
    improved = trainer.balanced_probability_calibration(
        torch.tensor([0.05, 0.10, 0.90, 0.95]), target, evidence
    )
    assert epoch_zero["brier"] == pytest.approx(0.25)
    assert improved["brier"] < epoch_zero["brier"]
    assert improved["log_loss"] < epoch_zero["log_loss"]


def test_calibration_gate_requires_macro_improvement_and_each_scene_safety() -> None:
    baseline = {"pair_calibration": _calibration(0.25, 0.693)}
    candidate = {"pair_calibration": _calibration(0.20, 0.60)}
    assert all(trainer.calibration_gate(candidate, baseline).values())
    candidate["pair_calibration"]["per_scene"][trainer.VALIDATION_SCENES[1]][
        "brier"
    ] = 0.26
    assert (
        trainer.calibration_gate(candidate, baseline)[
            "every_validation_scene_brier_non_regression"
        ]
        is False
    )


def test_training_contract_and_dry_run_keep_legacy_and_target_closed() -> None:
    contract = trainer.training_contract()
    result = trainer.synthetic_dry_run()
    assert contract["cohort"]["identical_to_formal_v2"] is True
    assert contract["selection"]["target_execution_before_promotion"] is False
    assert contract["selection"]["absolute_calibration"][
        "every_validation_scene_brier_non_regression"
    ] is True
    assert result["feature_dimension"] == 30
    assert result["epoch_zero_probability"] == 0.5
    assert result["final_layer_gradient_nonzero"] is True
    assert result["legacy_v2_default_changed"] is False
    assert result["target_metric_computed"] is False


def _checkpoint() -> dict:
    median = torch.zeros(30)
    scale = torch.ones(30)
    model = RegionCoMembershipNativeV3(median, scale)
    state = model.state_dict()
    contract = trainer.training_contract()
    selected = {
        "epoch": 25,
        "method": trainer.v2.METHODS[0],
        "maximum_regions": 2,
        "threshold": trainer.v2.THRESHOLDS[0],
    }
    singleton = {
        "epoch": 0,
        "method": trainer.v2.METHODS[0],
        "maximum_regions": 1,
        "threshold": trainer.v2.THRESHOLDS[0],
    }
    gate = {
        "selected_epoch_positive": True,
        "topology_strictly_exceeds_singleton": True,
        "iou_strictly_exceeds_singleton": True,
        "f1_strictly_exceeds_singleton": True,
        "macro_brier_strictly_improves_epoch_zero": True,
        "macro_log_loss_strictly_improves_epoch_zero": True,
        "every_validation_scene_brier_non_regression": True,
        "passed": True,
    }
    return {
        "schema": trainer.CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "training_contract": contract,
        "training_contract_sha256": canonical_json_sha256(contract),
        "execution_authority": _record(),
        "feature_names": list(trainer.PAIR_FEATURE_NAMES),
        "normalization": {"median": median, "robust_scale": scale},
        "model_state_dict": state,
        "model_state_dict_sha256": trainer._state_sha(state),
        "selected_epoch": 25,
        "selected_rule": {
            "method": selected["method"],
            "maximum_regions": selected["maximum_regions"],
            "threshold": selected["threshold"],
        },
        "selected_validation": selected,
        "singleton_validation": singleton,
        "promotion_gate": gate,
        "source_access": trainer.source_access(),
        "target_execution_performed": False,
    }


def test_checkpoint_validator_binds_model_calibration_gate_and_target_closure() -> None:
    checkpoint = _checkpoint()
    assert trainer.validate_checkpoint(checkpoint)["promotion_gate"]["passed"] is True
    checkpoint["target_execution_performed"] = True
    with pytest.raises(ValueError):
        trainer.validate_checkpoint(checkpoint)
