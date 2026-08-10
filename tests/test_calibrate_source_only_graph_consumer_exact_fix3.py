from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from radio_gs.scripts.calibrate_source_only_graph_consumer_exact_fix3 import (
    apply_strict_sequential_thresholds,
    exact_direct_edge_trace,
)


def _scene() -> SimpleNamespace:
    return SimpleNamespace(
        region_count=4,
        pair_indices=torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long),
        targets=torch.tensor([False, False, True]),
        core_rows=(
            torch.tensor([0, 1]),
            torch.tensor([2, 3]),
            torch.tensor([3, 4]),
            torch.tensor([5, 6]),
        ),
    )


def test_exact_trace_deduplicates_marginal_primitives_after_first_step() -> None:
    trace = exact_direct_edge_trace(
        scene=_scene(),
        probability_lower=torch.tensor([0.9, 0.85, 0.8]),
        edge_eligible_mask=torch.ones(3, dtype=torch.bool),
        target_filter=False,
        steps=2,
        novel_mass_reference=2.0,
    )
    assert trace["candidate_rows"][0].tolist() == [1, 2]
    assert trace["marginal_primitives"][0].tolist() == [2, 1]
    assert torch.allclose(trace["gains"][0], torch.tensor([0.8, 0.35]))


def test_true_and_false_filters_use_the_same_gain_units() -> None:
    false_trace = exact_direct_edge_trace(
        scene=_scene(),
        probability_lower=torch.tensor([0.9, 0.85, 0.8]),
        edge_eligible_mask=torch.ones(3, dtype=torch.bool),
        target_filter=False,
        steps=1,
        novel_mass_reference=2.0,
    )
    true_trace = exact_direct_edge_trace(
        scene=_scene(),
        probability_lower=torch.tensor([0.9, 0.85, 0.8]),
        edge_eligible_mask=torch.ones(3, dtype=torch.bool),
        target_filter=True,
        steps=1,
        novel_mass_reference=2.0,
    )
    assert float(false_trace["gains"][0, 0]) == pytest.approx(0.8)
    assert float(true_trace["gains"][0, 0]) == pytest.approx(0.6)


def test_sequential_threshold_stops_all_later_steps() -> None:
    selected = apply_strict_sequential_thresholds(
        torch.tensor([[0.8, 0.9, 1.0], [0.7, 0.7, 0.7]]),
        (0.75, 0.85, 0.95),
    )
    assert torch.equal(
        selected,
        torch.tensor([[True, True, True], [False, False, False]]),
    )
