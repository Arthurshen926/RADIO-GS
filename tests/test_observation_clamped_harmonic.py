from __future__ import annotations

import pytest
import torch

from radio_gs.querying.observation_clamped_harmonic import (
    ObservationClampedHarmonicConfig,
    method_contract,
    solve_observation_clamped_harmonic,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph


def _disconnected_chains() -> PrimitiveSupportGraph:
    return PrimitiveSupportGraph(
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
        ),
        edge_weight=torch.ones(6),
        raw_affinity=torch.ones(6),
        local_sigma=torch.ones(5),
        num_nodes=5,
    )


def test_graph_changes_only_unknown_rows_and_preserves_unobserved_component() -> None:
    graph = _disconnected_chains()
    fused_unary_prior = torch.tensor([0.9, 0.2, 0.3, 0.17, 0.83])
    source_confidence = torch.tensor([0.7, 0.0, 0.6, 0.0, 0.0])

    result = solve_observation_clamped_harmonic(
        graph,
        fused_unary_prior,
        source_confidence,
        config=ObservationClampedHarmonicConfig(
            cg_iterations=64, cg_tolerance=1e-8
        ),
    )

    observed = source_confidence > 0
    assert torch.equal(result[observed], fused_unary_prior[observed])
    assert result[1].item() == pytest.approx(0.6, abs=1e-6)
    assert torch.equal(result[3:], fused_unary_prior[3:])
    assert bool(((result >= 0) & (result <= 1)).all())


def test_no_observation_anywhere_is_exact_identity() -> None:
    prior = torch.tensor([0.9, 0.2, 0.3, 0.17, 0.83])
    result = solve_observation_clamped_harmonic(
        _disconnected_chains(), prior, torch.zeros_like(prior)
    )
    assert torch.equal(result, prior)


def test_hard_seeds_are_exact_boundaries() -> None:
    prior = torch.tensor([0.4, 0.2, 0.6, 0.17, 0.83])
    positive = torch.tensor([0.3, 0.0, 0.0, 0.0, 0.0])
    negative = torch.tensor([0.0, 0.0, 0.3, 0.0, 0.0])
    result = solve_observation_clamped_harmonic(
        _disconnected_chains(),
        prior,
        torch.zeros_like(prior),
        positive_seed_weight=positive,
        negative_seed_weight=negative,
    )
    assert result[0] == 1
    assert result[2] == 0
    assert result[1].item() == pytest.approx(0.5, abs=1e-6)
    assert torch.equal(result[3:], prior[3:])


def test_contract_is_target_independent_and_has_no_numeric_method_threshold() -> None:
    contract = method_contract()
    assert contract["source_boundary"].endswith("strictly_greater_than_zero")
    assert contract["source_boundary_rewrite"] is False
    assert contract["learned_or_scene_specific_constants"] is False
    assert contract["uses_target_rgb_mask_or_metric"] is False

