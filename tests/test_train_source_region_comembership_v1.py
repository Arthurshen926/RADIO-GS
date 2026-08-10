from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from radio_gs.scripts import train_source_region_comembership_v1 as trainer


SHA = "a" * 64


def _record() -> dict[str, str]:
    return {"path": "/frozen", "sha256": SHA}


def _execution() -> dict:
    return {
        "schema": trainer.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_after_complete_4train_2validation_source_preflight",
        "implementation": _record(),
        "preregistration": _record(),
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


def test_execution_authority_fixes_four_plus_two_and_target_closed() -> None:
    value = trainer.validate_execution_authority(_execution())
    assert tuple(row["scene_id"] for row in value["source_train"]) == (
        trainer.TRAIN_SCENES
    )
    assert tuple(row["scene_id"] for row in value["source_validation"]) == (
        trainer.VALIDATION_SCENES
    )
    assert value["target_execution_authorized"] is False


@pytest.mark.parametrize("tamper", ["cohort", "target", "status"])
def test_execution_authority_fails_closed(tamper: str) -> None:
    value = deepcopy(_execution())
    if tamper == "cohort":
        value["source_validation"][0]["scene_id"] = "scene0013_00"
    elif tamper == "target":
        value["target_execution_authorized"] = True
    else:
        value["status"] = "draft"
    with pytest.raises(ValueError):
        trainer.validate_execution_authority(value)


def test_balanced_loss_equalizes_classes_and_has_gradient() -> None:
    logits = torch.tensor([0.1, -0.2, 0.3, -0.4], requires_grad=True)
    target = torch.tensor([True, False, True, False])
    weight = torch.tensor([1.0, 1.0, 0.2, 0.5])
    loss = trainer.balanced_scene_loss(logits, target, weight)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0


def test_epoch_and_threshold_selection_use_source_validation_only() -> None:
    def row(epoch: int, score: float, bce: float) -> dict:
        return {
            "epoch": epoch,
            "validation": {"scene_macro_balanced_weighted_bce": bce},
            "validation_topology": {
                "selected": {
                    "scene_macro_instance_macro": {
                        "topology_score": score,
                        "iou": score,
                        "f1": score,
                        "contamination": 0.0,
                        "giant_excess": 0.0,
                    }
                }
            },
        }

    history = [
        row(0, 0.2, 0.7),
        row(1, 0.4, 0.6),
        row(2, 0.4, 0.5),
    ]
    assert trainer.select_best_epoch(history) == 2
    probability = torch.tensor([0.9, 0.8, 0.2, 0.1])
    target = torch.tensor([True, False, True, False])
    weight = torch.ones(4)
    assert trainer._weighted_f1(probability, target, weight, 0.5) == pytest.approx(0.5)


def test_topology_selection_rejects_pair_f1_favored_false_positive_bridge() -> None:
    pairs = torch.tensor([[0, 1, 3, 4, 2], [1, 2, 4, 5, 3]], dtype=torch.int64)
    targets = torch.tensor([True, True, True, True, False])
    scene = trainer.SceneAuthority(
        scene_id="synthetic_validation",
        split="source_validation",
        record=_record(),
        pair_features=torch.zeros(5, 15),
        pair_indices=pairs,
        targets=targets,
        evidence_weights=torch.ones(5),
        region_count=6,
        dominant_instance_ids=torch.tensor([1, 1, 1, 2, 2, 2]),
        instance_purity=torch.ones(6),
        instance_label_coverage=torch.ones(6),
        instance_observed=torch.ones(6, dtype=torch.bool),
    )
    probability = {scene.scene_id: torch.tensor([0.6, 0.6, 0.6, 0.6, 0.7])}
    result = trainer.select_validation_threshold_from_probabilities(
        (scene,), probability
    )
    assert result["diagnostic_pair_edge_f1_selected_threshold"] <= 0.6
    assert result["selected"]["threshold"] > 0.7
    assert result["selected"]["scene_macro_instance_macro"]["contamination"] == 0
    assert (
        result["candidate_graph_oracle_ceiling"]["scene_macro_instance_macro"]["iou"]
        == 1
    )


def test_synthetic_train_dry_run_has_epoch_zero_and_gradient() -> None:
    result = trainer.synthetic_dry_run()
    assert result["epoch_zero_probability"] == 0.5
    assert result["gradient_nonzero"] is True
    assert result["benchmark_opened"] is False
