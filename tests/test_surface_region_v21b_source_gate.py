from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from radio_gs.interfaces import surface_region_v21b_source_gate as gate
from radio_gs.interfaces import (
    surface_region_v21b_reliability_conditioned_residual as model_interface,
)
from radio_gs.scripts import (
    train_surface_region_v21b_conditioned_rank256_exact4x2 as trainer,
)
from tests.test_surface_region_v21_source_gate import _v1_validation


def _history(eligible=(1, 3)):
    return [
        {
            "step": step,
            "validation": {"selection_eligible": step in eligible},
            "model_state_dict_sha256": str(step % 10) * 64,
        }
        for step in range(trainer.OPTIMIZER_STEPS + 1)
    ]


def test_state_archive_requires_step_zero_and_every_eligible_state() -> None:
    normalization = {"median": torch.zeros(30), "robust_scale": torch.ones(30)}
    model = model_interface.build_model_from_source_normalization(normalization)
    state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    digest = trainer._state_sha(state)
    history = _history()
    for step in (0, 1, 3):
        history[step]["model_state_dict_sha256"] = digest
    archive = {
        "schema": trainer.STATE_ARCHIVE_SCHEMA,
        "schema_version": 1,
        "training_contract_sha256": trainer.TRAINING_CONTRACT_SHA256,
        "execution_authority": {"path": "/execution.json", "sha256": "a" * 64},
        "normalization_authority": {"path": "/normal.pt", "sha256": "b" * 64},
        "normalization_content_authority_sha256": "c" * 64,
        "saved_steps": [0, 1, 3],
        "promotion_eligible_steps": [1, 3],
        "model_state_dict_sha256_by_step": {
            str(step): digest for step in (0, 1, 3)
        },
        "model_state_dict_by_step": {
            str(step): deepcopy(state) for step in (0, 1, 3)
        },
        "source_access": trainer.source_access(),
    }
    validated = gate.validate_state_archive(
        archive, normalization=normalization, history=history
    )
    assert validated["saved_steps"] == [0, 1, 3]
    missing = deepcopy(archive)
    missing["saved_steps"] = [0, 1]
    missing["model_state_dict_sha256_by_step"].pop("3")
    missing["model_state_dict_by_step"].pop("3")
    with pytest.raises(ValueError, match="archive identity"):
        gate.validate_state_archive(
            missing, normalization=normalization, history=history
        )


def test_checkpoint_validator_reconstructs_exact_rank256_state() -> None:
    normalization = {"median": torch.zeros(30), "robust_scale": torch.ones(30)}
    model = model_interface.build_model_from_source_normalization(normalization)
    state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload = {
        "schema": trainer.CHECKPOINT_SCHEMA,
        "schema_version": 1,
        "model_class": type(model).__name__,
        "model_architecture": model.architecture(),
        "accepted_v2_authority": gate.accepted_v2_authority(),
        "model_state_dict": state,
        "model_state_dict_sha256": trainer._state_sha(state),
        "normalization_authority": {"path": "/normal.pt", "sha256": "a" * 64},
        "certificate": {"path": "/certificate.json", "sha256": "b" * 64},
        "state_archive": {"path": "/states.pt", "sha256": "c" * 64},
        "selected_step": 1,
        "source_access": trainer.source_access(),
    }
    validated = gate.validate_checkpoint_payload(payload, normalization=normalization)
    assert validated["model_state_dict_sha256"] == trainer._state_sha(state)
    changed = deepcopy(payload)
    changed["model_state_dict"]["residual_projection.bias"][0] = float("nan")
    with pytest.raises(ValueError, match="state residual_projection.bias"):
        gate.validate_checkpoint_payload(changed, normalization=normalization)


def _complete_validation(aux: float, absolute: float, pairwise: float) -> dict:
    per_scene = {
        scene: {
            "combined_objective": aux,
            "response_auxiliary_loss": aux,
            "response_absolute_relevance_loss": absolute,
            "response_continuous_pairwise_relevance_loss": pairwise,
            "complete_canonical_rows": 100,
            "active_rows": 96,
            "active_row_coverage": 0.96,
            "fallback_bitwise_accepted_v2_e0": True,
            "response_authority_hard_negative_pairs": 100,
            "response_objective_hard_negative_pairs": 96,
            "response_pairwise_objective_hard_negative_pairs": 96,
            "response_triplet_objective_hard_negative_pairs": 95,
            "response_pair_trainable_endpoint_coverage": 0.96,
        }
        for scene in trainer.VALIDATION_SCENES
    }
    return {
        "v1_non_regression": _v1_validation(),
        "response_listwise_v21b": {
            "scene_count": 2,
            "scene_macro_auxiliary_loss": aux,
            "scene_macro_absolute_relevance_loss": absolute,
            "scene_macro_continuous_pairwise_relevance_loss": pairwise,
            "scene_macro_active_row_coverage": 0.96,
            "scene_macro_pair_trainable_endpoint_coverage": 0.96,
            "per_scene": per_scene,
        },
        "validation_no_grad": True,
        "benchmark_opened": False,
    }


def test_source_evidence_recomputes_all_thresholds_and_selection() -> None:
    epoch_zero_raw = _complete_validation(1.0, 1.0, 1.0)
    epoch_zero = trainer.attach_promotion(epoch_zero_raw, epoch_zero_raw)
    good = trainer.attach_promotion(
        _complete_validation(0.99, 0.99, 0.99), epoch_zero
    )
    bad = trainer.attach_promotion(
        _complete_validation(1.0, 1.0, 1.0), epoch_zero
    )
    history = []
    for step in range(trainer.OPTIMIZER_STEPS + 1):
        validation = epoch_zero if step == 0 else (good if step == 1 else bad)
        history.append(
            {
                "step": step,
                "training": None if step == 0 else {},
                "validation": deepcopy(validation),
                "model_state_dict_sha256": "a" * 64,
            }
        )
    result = {
        "schema": trainer.RESULT_SCHEMA,
        "schema_version": 1,
        "status": "source_only_v21b_promotion_candidate_complete",
        "training_contract": trainer.training_contract(),
        "training_contract_sha256": trainer.TRAINING_CONTRACT_SHA256,
        "execution_authority": {"path": "/execution.json", "sha256": "1" * 64},
        "normalization_authority": {"path": "/normal.pt", "sha256": "2" * 64},
        "state_archive": {"path": "/states.pt", "sha256": "3" * 64},
        "checkpoint": {"path": "/checkpoint.pt", "sha256": "4" * 64},
        "certificate": {"path": "/certificate.json", "sha256": "5" * 64},
        "selected_step": 1,
        "selected_validation": deepcopy(good),
        "promotion_candidate_available": True,
        "history": history,
        "source_access": trainer.source_access(),
        "benchmark_opened": False,
    }
    evidence = gate.validate_source_promotion_evidence(result)
    assert evidence["passed"] is True
    assert evidence["selected_step"] == 1
    changed = deepcopy(result)
    changed["history"][1]["validation"]["promotion"]["checks"][
        "v1_fidelity_non_regression"
    ] = False
    with pytest.raises(ValueError, match="promotion differs"):
        gate.validate_source_promotion_evidence(changed)
