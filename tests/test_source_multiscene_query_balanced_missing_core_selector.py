from __future__ import annotations

import torch

from radio_gs.interfaces import source_multiscene_query_balanced_missing_core_selector as v3


def test_scene_query_region_weights_equalize_each_hierarchy_level() -> None:
    scenes = torch.tensor([0] * 8 + [1] * 4)
    queries = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    regions = torch.tensor([0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1])
    weight = v3.scene_query_region_balanced_weights(scenes, queries, regions)
    scene_mass = torch.stack([weight[scenes == scene].sum() for scene in (0, 1)])
    assert torch.allclose(scene_mass, scene_mass.mean().expand_as(scene_mass))
    scene0_query_mass = torch.stack(
        [weight[(scenes == 0) & (queries == query)].sum() for query in (0, 1)]
    )
    assert torch.allclose(
        scene0_query_mass, scene0_query_mass.mean().expand_as(scene0_query_mass)
    )
    query0_region_mass = torch.stack(
        [
            weight[(scenes == 0) & (queries == 0) & (regions == region)].sum()
            for region in (0, 1)
        ]
    )
    assert torch.allclose(
        query0_region_mass, query0_region_mass.mean().expand_as(query0_region_mass)
    )


def test_fold_assignment_never_splits_physical_region_across_queries() -> None:
    scenes = torch.tensor([0] * 8 + [1] * 8)
    queries = torch.tensor([0, 1, 2, 3] * 4)
    regions = torch.tensor([7] * 4 + [8] * 4 + [7] * 4 + [8] * 4)
    folds = v3.multiscene_query_region_fold_ids(scenes, queries, regions)
    packed = v3.packed_scene_region_groups(scenes, regions)
    for group in torch.unique(packed):
        assert torch.unique(folds[packed == group]).numel() == 1
    for fold in range(3):
        assert not (
            set(packed[folds == fold].tolist())
            & set(packed[folds != fold].tolist())
        )


def test_query_gate_rejects_negative_lower_tail_gain() -> None:
    queries = torch.arange(8).repeat_interleave(64)
    selected = torch.zeros(queries.numel(), dtype=torch.bool)
    utility = torch.full((queries.numel(),), -1.0)
    for query in range(8):
        rows = torch.where(queries == query)[0]
        selected[rows[:16]] = True
        utility[rows[:16]] = 1.0
    bad_rows = torch.where(queries == 0)[0]
    utility[bad_rows[:16]] = -0.25
    utility[bad_rows[16:]] = -1.0
    outcomes, checks = v3.evaluate_query_utility_gate(
        selected=selected,
        signed_utility=utility,
        query_indices=queries,
    )
    assert outcomes["evaluable_queries"] == 8
    assert checks["every_evaluable_query_selected_utility_nonnegative"] is False
    assert checks["lower_tail_20pct_utility_gain_CVaR_nonnegative"] is True
    assert checks["passed"] is False


def test_threshold_is_query_safe_and_not_near_unconditional() -> None:
    scenes = []
    queries = []
    scores = []
    labels = []
    utility = []
    for scene in range(2):
        for query in range(8):
            scenes.extend([scene] * 64)
            queries.extend([query] * 64)
            scores.extend([0.9] * 32 + [0.1] * 32)
            labels.extend([True] * 30 + [False] * 2 + [True] * 8 + [False] * 24)
            utility.extend([1.0] * 32 + [-1.0] * 32)
    result = v3.select_largest_query_safe_oof_threshold(
        torch.tensor(scores),
        torch.tensor(labels),
        torch.tensor(utility),
        torch.tensor(scenes),
        torch.tensor(queries),
        minimum_selected_per_scene=64,
        minimum_rejected_per_scene=64,
        maximum_selected_fraction_per_scene=0.90,
        minimum_candidate_units_per_query=64,
        minimum_selected_units_per_query=16,
        minimum_evaluable_query_fraction=0.80,
        minimum_evaluable_queries=8,
    )
    assert abs(float(result["threshold_inclusive"]) - 0.9) < 1e-6
    assert result["coverage_fraction"] == 0.5
    assert all(row["query_gate"]["passed"] for row in result["per_scene"])
