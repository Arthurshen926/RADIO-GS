from __future__ import annotations

import torch

from radio_gs.interfaces.source_monotone_missing_core_selector import (
    SELECTOR_FEATURE_NAMES,
    SOURCE_UNIT_FEATURE_INDICES,
)
from radio_gs.interfaces.source_multiscene_monotone_missing_core_selector import (
    MODEL_SCHEMA,
    multiscene_region_fold_ids,
    packed_scene_region_groups,
    scene_region_balanced_weights,
    select_largest_multiscene_safe_oof_threshold,
    validate_multiscene_selector_model_payload,
)
from radio_gs.scripts.fit_source_multiscene_monotone_missing_core_selector_v2 import (
    fixed_fit,
    source_access,
)


def test_scene_region_weights_equalize_scenes_and_regions() -> None:
    scenes = torch.tensor([0] * 12 + [1] * 10)
    regions = torch.tensor([0] * 10 + [1] * 2 + [0] * 3 + [1] * 7)
    weight = scene_region_balanced_weights(scenes, regions)
    scene_mass = torch.stack([weight[scenes == scene].sum() for scene in range(2)])
    assert torch.allclose(scene_mass, scene_mass[:1].expand_as(scene_mass))
    packed = packed_scene_region_groups(scenes, regions)
    group_mass = torch.stack([weight[packed == group].sum() for group in packed.unique()])
    assert torch.allclose(group_mass, group_mass[:1].expand_as(group_mass))


def test_fold_assignment_never_leaks_a_scene_region_group() -> None:
    scenes = torch.arange(2).repeat_interleave(60 * 3)
    regions = torch.arange(60).repeat_interleave(3).repeat(2)
    folds = multiscene_region_fold_ids(scenes, regions)
    packed = packed_scene_region_groups(scenes, regions)
    assert set(folds.tolist()) == {0, 1, 2}
    for group in packed.unique():
        assert int(folds[packed == group].unique().numel()) == 1
    for fold in range(3):
        heldout = folds == fold
        assert set(scenes[heldout].tolist()) == {0, 1}
        assert set(packed[heldout].tolist()).isdisjoint(set(packed[~heldout].tolist()))


def test_threshold_is_limited_by_each_scene_gate() -> None:
    # At 0.90 scene 1 falls below its Wilson bar.  The maximum safe common
    # coverage is therefore the tie-complete 0.95 population.
    score_parts = []
    label_parts = []
    utility_parts = []
    scene_parts = []
    for scene in range(2):
        high_label = torch.cat((torch.ones(270), torch.zeros(30))).bool()
        middle_positive = 270 if scene == 0 else 190
        middle_label = torch.cat(
            (torch.ones(middle_positive), torch.zeros(300 - middle_positive))
        ).bool()
        low_label = torch.cat((torch.ones(20), torch.zeros(380))).bool()
        labels = torch.cat((high_label, middle_label, low_label))
        score_parts.append(
            torch.cat((torch.full((300,), 0.95), torch.full((300,), 0.90), torch.full((400,), 0.10)))
        )
        label_parts.append(labels)
        utility_parts.append(torch.where(labels, torch.ones(1000), -torch.ones(1000)))
        scene_parts.append(torch.full((1000,), scene))
    result = select_largest_multiscene_safe_oof_threshold(
        torch.cat(score_parts),
        torch.cat(label_parts),
        torch.cat(utility_parts),
        torch.cat(scene_parts),
    )
    assert result["threshold_inclusive"] == torch.tensor(0.95).float().item()
    assert result["selected"] == 600
    assert all(
        row["hard_precision_wilson95_lower"] >= 0.75
        and row["signed_utility_mean"]
        > row["unconditional_signed_utility_mean"]
        for row in result["per_scene"]
    )
    assert result["hard_precision_wilson95_lower"] >= 0.80


def test_model_contract_has_no_query_or_scene_identifier_feature() -> None:
    fit = fixed_fit()
    assert fit["feature_names"] == list(SELECTOR_FEATURE_NAMES)
    assert fit["source_unit_feature_indices"] == list(SOURCE_UNIT_FEATURE_INDICES)
    assert fit["scene_or_query_identifiers_as_features"] is False
    assert not any("query_id" in name or "scene_id" in name for name in fit["feature_names"])
    assert source_access()["scene0003_membership_opened"] is False
    assert source_access()["scene0004_membership_opened"] is False

    zero = torch.zeros(len(SELECTOR_FEATURE_NAMES))
    one = torch.ones(len(SELECTOR_FEATURE_NAMES))
    payload = {
        "schema": MODEL_SCHEMA,
        "schema_version": 2,
        "feature_names": list(SELECTOR_FEATURE_NAMES),
        "source_unit_feature_indices": list(SOURCE_UNIT_FEATURE_INDICES),
        "fold_models": [
            {
                "location": zero,
                "scale": one,
                "positive_weights": one,
                "bias": torch.tensor(0.0),
            }
            for _ in range(3)
        ],
        "threshold_inclusive": 0.8,
        "target_probability": "minimum_probability_across_three_fold_models",
        "execution_authority": {"path": "/sealed/source.json", "sha256": "0" * 64},
        "training_provenance": {
            "source_scene_count": 2,
            "source_scene_ids_sha256": "1" * 64,
            "scene_identifier_used_for_balancing_and_folds_only": True,
            "query_identifier_used_as_feature": False,
            "scene_identifier_used_as_feature": False,
        },
    }
    validate_multiscene_selector_model_payload(payload)

