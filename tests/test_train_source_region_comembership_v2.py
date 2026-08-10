from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from radio_gs.scripts import train_source_region_comembership_v2 as trainer


SHA = "a" * 64


def _record() -> dict[str, str]:
    return {"path": "/frozen", "sha256": SHA}


def _execution() -> dict:
    return {
        "schema": trainer.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_after_complete_v2_4train_2validation_preflight",
        "implementation": _record(),
        "preregistration": _record(),
        "efficiency_addendum": _record(),
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


def _scene() -> trainer.SceneAuthorityV2:
    primitive = torch.tensor(
        [[0.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]]
    )
    core = (torch.tensor([0]), torch.tensor([1]), torch.tensor([2]))
    region = torch.stack([primitive[value].sum(dim=0) for value in core])
    return trainer.SceneAuthorityV2(
        scene_id="synthetic",
        split="source_validation",
        record={"path": "/x", "sha256": "a" * 64},
        region_count=3,
        pair_indices=torch.tensor([[0, 1], [1, 2]]),
        pair_features=torch.zeros(2, 21),
        targets=torch.tensor([True, False]),
        evidence_weights=torch.ones(2),
        region_rows=torch.arange(3)[:, None],
        token_mask=torch.ones(3, 1, dtype=torch.bool),
        primitive_instance_mass=primitive,
        core_rows=core,
        region_instance_mass=region,
        dominant_instance_ids=torch.tensor([1, 1, 2]),
        dominant_instance_mass=torch.tensor([2.0, 1.0, 3.0]),
        eligible_seeds=torch.ones(3, dtype=torch.bool),
        target_total=torch.tensor([3.0, 3.0], dtype=torch.float64),
        scene_total=6.0,
    )


def test_execution_authority_fixes_four_plus_two_and_target_closed() -> None:
    value = trainer.validate_execution_authority(_execution())
    assert tuple(row["scene_id"] for row in value["source_train"]) == tuple(
        trainer.TRAIN_SCENES
    )
    assert tuple(row["scene_id"] for row in value["source_validation"]) == tuple(
        trainer.VALIDATION_SCENES
    )
    assert value["target_execution_authorized"] is False


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


def test_fixed_proxy_seeds_cover_every_instance_and_are_deterministic() -> None:
    scene = _scene()
    first = trainer.fixed_proxy_seed_indices(scene)
    second = trainer.fixed_proxy_seed_indices(scene)
    assert torch.equal(first, second)
    assert set(scene.dominant_instance_ids[first].tolist()) == {1, 2}
    assert first.tolist() == [0, 1, 2]


def test_exact_mass_metric_scores_clean_same_instance_union() -> None:
    scene = _scene()
    metric = trainer._mass_metrics(
        scene=scene,
        selections={0: (0, 1), 1: (1, 0)},
        maximum_regions=2,
        exact_primitive_union=True,
    )
    assert metric["iou"] == pytest.approx(1.0)
    assert metric["f1"] == pytest.approx(1.0)
    assert metric["contamination"] == pytest.approx(0.0)


def test_proxy_shortlist_keeps_singleton_each_method_k_and_global_supplement() -> None:
    grid = [
        {
            "method": trainer.METHODS[0],
            "maximum_regions": 1,
            "threshold": 0.95,
            "scene_macro": {
                "topology_score": 0.0,
                "iou": 0.0,
                "f1": 0.0,
                "contamination": 0.0,
                "giant_excess": 0.0,
            },
        }
    ]
    for method in trainer.METHODS:
        for maximum in trainer.MAXIMUM_REGIONS[1:]:
            for threshold in (0.5, 0.8):
                score = threshold + maximum / 100
                grid.append(
                    {
                        "method": method,
                        "maximum_regions": maximum,
                        "threshold": threshold,
                        "scene_macro": {
                            "topology_score": score,
                            "iou": score,
                            "f1": score,
                            "contamination": 0.0,
                            "giant_excess": 0.0,
                        },
                    }
                )
    selected = trainer.deterministic_proxy_shortlist(grid)
    keys = {
        (row["method"], row["maximum_regions"], row["threshold"])
        for row in selected
    }
    assert (trainer.METHODS[0], 1, 0.95) in keys
    for method in trainer.METHODS:
        for maximum in trainer.MAXIMUM_REGIONS[1:]:
            assert (method, maximum, 0.8) in keys
    assert len(selected) <= 18


def test_proxy_grid_is_complete_and_shortlist_exact_candidate_runs() -> None:
    scene = _scene()
    probability = {scene.scene_id: torch.tensor([0.9, 0.1])}
    grid = trainer._proxy_grid((scene,), probability)
    assert len(grid) == 1 + len(trainer.THRESHOLDS) * len(trainer.METHODS) * 3
    shortlist = trainer.deterministic_proxy_shortlist(grid)
    candidate = trainer._exact_candidate(
        epoch=25,
        rule=shortlist[0],
        scenes=(scene,),
        probabilities=probability,
        selection_cache={},
    )
    assert candidate["epoch"] == 25
    assert candidate["scene_macro"]["iou"] > 0


def test_synthetic_dry_run_preserves_epoch_zero_identity_and_gradient() -> None:
    result = trainer.synthetic_dry_run()
    assert result["epoch_zero_probability"] == 0.5
    assert result["final_layer_gradient_nonzero"] is True
    assert result["snapshot_epochs"] == [0, 25, 50, 75, 100]
