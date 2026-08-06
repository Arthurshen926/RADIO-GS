import numpy as np
import pytest
import torch

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV4
from radio_gs.interfaces.surface_region_typed_context_adaptive import (
    ADAPTIVE_CONTEXT_BUDGET,
    ADAPTIVE_WORKING_MEMORY_CEILING_BYTES,
    TERMINATION_NATURAL,
    TERMINATION_TYPED_BUDGET,
    adaptive_typed_budget_context,
    adaptive_typed_budget_context_batch,
    adaptive_typed_context_v4_contract,
    estimate_adaptive_probe_working_bytes,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph


def _line_graph(count: int, spacing: float) -> tuple[torch.Tensor, PrimitiveSupportGraph]:
    xyz = torch.stack(
        (
            torch.arange(count, dtype=torch.float32) * float(spacing),
            torch.zeros(count),
            torch.zeros(count),
        ),
        dim=1,
    )
    edge = torch.tensor(
        [(index, index + 1) for index in range(count - 1)], dtype=torch.long
    ).T.contiguous()
    affinity = torch.ones(edge.shape[1])
    return xyz, PrimitiveSupportGraph(
        edge_index=edge,
        edge_weight=affinity,
        raw_affinity=affinity,
        local_sigma=torch.full((count,), float(spacing)),
        num_nodes=count,
        edge_channels={"appearance": affinity, "boundary": affinity},
    )


def _star_graph(count: int) -> tuple[torch.Tensor, PrimitiveSupportGraph]:
    xyz = torch.zeros(count, 3)
    xyz[1:, 0] = 1.0
    edge = torch.stack(
        (torch.zeros(count - 1, dtype=torch.long), torch.arange(1, count)), dim=0
    )
    affinity = torch.ones(count - 1)
    return xyz, PrimitiveSupportGraph(
        edge_index=edge,
        edge_weight=affinity,
        raw_affinity=affinity,
        local_sigma=torch.ones(count),
        num_nodes=count,
        edge_channels={"appearance": affinity, "boundary": affinity},
    )


def _contract(*, context_ratio: float = 1.2) -> SurfaceRegionContractV4:
    base = adaptive_typed_context_v4_contract()
    specification = base.to_dict()
    specification["radii_m"] = (1.0,)
    specification["context_ratio"] = float(context_ratio)
    return SurfaceRegionContractV4(**specification)


def _oracle(contract, prepared, anchor: int, radius: float):
    eligibility = np.ones(prepared.num_nodes, dtype=np.bool_)
    rows, distances = contract._eligible_dijkstra(
        prepared.semantic_csr,
        anchor,
        eligibility,
        limit=radius * contract.context_ratio,
        maximum_candidates=prepared.num_nodes,
    )
    selected_rows, selected_distances = contract._select_semantic_candidates(
        rows, distances, anchor, radius
    )
    selected_distances = torch.from_numpy(
        np.asarray(selected_distances, dtype=np.float32).copy()
    )
    return (
        torch.from_numpy(np.asarray(selected_rows, dtype=np.int64).copy()),
        selected_distances <= radius + 1e-7,
        selected_distances,
    )


def _assert_oracle_equal(selection, oracle) -> None:
    rows, core, distances = oracle
    assert torch.equal(selection.rows, rows)
    assert torch.equal(selection.core_mask, core)
    assert torch.equal(selection.context_mask, ~core)
    assert torch.equal(selection.semantic_geodesic_distance, distances)


def test_natural_exhaustion_matches_infinite_candidate_oracle() -> None:
    xyz, graph = _line_graph(18, 0.01)
    contract = _contract()
    prepared = contract.prepare_graph(graph, xyz)
    result = adaptive_typed_budget_context(contract, prepared, 0, 0.10)
    assert result.termination == TERMINATION_NATURAL
    assert result.final_probe_width == 18
    assert result.adaptive_round_count == 1
    _assert_oracle_equal(result, _oracle(contract, prepared, 0, 0.10))


def test_dense_core_adapts_until_typed_budget_and_removes_false_negative() -> None:
    xyz, graph = _line_graph(5000, 0.001)
    contract = _contract(context_ratio=10.0)
    prepared = contract.prepare_graph(graph, xyz)
    result = adaptive_typed_budget_context(contract, prepared, 0, 1.0)
    assert result.termination == TERMINATION_TYPED_BUDGET
    assert result.final_probe_width == 4100
    assert result.adaptive_round_count == 2
    assert int(result.context_mask.sum()) == ADAPTIVE_CONTEXT_BUDGET
    _assert_oracle_equal(result, _oracle(contract, prepared, 0, 1.0))


def test_core_shortage_context_donation_matches_oracle() -> None:
    xyz, graph = _line_graph(500, 0.01)
    contract = _contract(context_ratio=200.0)
    prepared = contract.prepare_graph(graph, xyz)
    result = adaptive_typed_budget_context(
        contract,
        prepared,
        0,
        0.02,
        initial_probe_width=32,
        growth_factor=4,
    )
    assert result.termination == TERMINATION_NATURAL
    assert int(result.core_mask.sum()) == 3
    assert int(result.context_mask.sum()) == 253
    _assert_oracle_equal(result, _oracle(contract, prepared, 0, 0.02))


def test_single_and_batch_are_selection_and_audit_invariant() -> None:
    xyz, graph = _line_graph(5000, 0.001)
    contract = _contract(context_ratio=10.0)
    prepared = contract.prepare_graph(graph, xyz)
    anchors = [0, 100, 250]
    batched = adaptive_typed_budget_context_batch(
        contract, prepared, anchors, 1.0, batch_size=3
    ).selections
    singles = [
        adaptive_typed_budget_context(contract, prepared, anchor, 1.0)
        for anchor in anchors
    ]
    for left, right in zip(batched, singles):
        assert left.termination == right.termination
        assert left.final_probe_width == right.final_probe_width
        assert left.settled_candidate_count == right.settled_candidate_count
        assert left.adaptive_round_count == right.adaptive_round_count
        for field in (
            "rows",
            "core_mask",
            "context_mask",
            "semantic_geodesic_distance",
        ):
            assert torch.equal(getattr(left, field), getattr(right, field))


def test_equal_distance_candidates_use_primitive_row_tie_break() -> None:
    xyz, graph = _star_graph(301)
    contract = _contract(context_ratio=3.0)
    prepared = contract.prepare_graph(graph, xyz)
    result = adaptive_typed_budget_context(
        contract,
        prepared,
        0,
        0.5,
        initial_probe_width=100,
        growth_factor=4,
    )
    assert result.rows[0].item() == 0
    assert result.rows[1:].tolist() == list(range(1, 256))
    _assert_oracle_equal(result, _oracle(contract, prepared, 0, 0.5))


def test_working_memory_ceiling_reduces_batch_and_fails_closed() -> None:
    xyz, graph = _line_graph(200, 0.01)
    contract = _contract(context_ratio=10.0)
    prepared = contract.prepare_graph(graph, xyz)
    one_row = estimate_adaptive_probe_working_bytes(
        num_nodes=200,
        num_directed_edges=int(prepared.semantic_csr.nnz),
        probe_width=200,
        batch_size=1,
    )
    with pytest.raises(MemoryError, match="one-row"):
        adaptive_typed_budget_context_batch(
            contract,
            prepared,
            [0],
            1.0,
            initial_probe_width=200,
            memory_ceiling_bytes=one_row - 1,
        )
    result = adaptive_typed_budget_context_batch(
        contract,
        prepared,
        list(range(8)),
        1.0,
        initial_probe_width=200,
        batch_size=8,
        memory_ceiling_bytes=one_row + 16 * 200,
    )
    assert result.maximum_estimated_working_bytes <= result.memory_ceiling_bytes
    assert result.memory_ceiling_bytes < ADAPTIVE_WORKING_MEMORY_CEILING_BYTES


def test_batch_above_strict_maximum_fails_closed() -> None:
    xyz, graph = _line_graph(18, 0.01)
    contract = _contract()
    with pytest.raises(ValueError, match="strict maximum"):
        adaptive_typed_budget_context_batch(
            contract,
            contract.prepare_graph(graph, xyz),
            [0],
            0.10,
            batch_size=9,
        )
