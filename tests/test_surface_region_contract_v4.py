import torch

from radio_gs.interfaces.surface_region_contract import (
    SurfaceRegionContractV3,
    SurfaceRegionContractV4,
    SurfaceRegionExpansionV4,
)
from radio_gs.interfaces.surface_region_selection import (
    as_region_selection,
    surface_region_contract_from_specification,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph


def _line_graph(count: int) -> tuple[torch.Tensor, PrimitiveSupportGraph]:
    xyz = torch.stack(
        (
            torch.arange(count, dtype=torch.float32) * 0.01,
            torch.zeros(count),
            torch.zeros(count),
        ),
        dim=1,
    )
    edge = torch.tensor(
        [(index, index + 1) for index in range(count - 1)], dtype=torch.long,
    ).T.contiguous()
    affinity = torch.ones(edge.shape[1])
    return xyz, PrimitiveSupportGraph(
        edge_index=edge,
        edge_weight=affinity,
        raw_affinity=affinity,
        local_sigma=torch.full((count,), 0.01),
        num_nodes=count,
        edge_channels={"appearance": affinity, "boundary": affinity},
    )


def _contract_v4(**overrides: object) -> SurfaceRegionContractV4:
    values: dict[str, object] = {
        "radii_m": (0.10,),
        "context_ratio": 2.0,
        "minimum_tokens": 1,
        "maximum_tokens": 12,
        "token_candidate_limit": 64,
        "core_token_fraction": 0.5,
    }
    values.update(overrides)
    return SurfaceRegionContractV4(**values)


def test_v4_enumerates_candidate_limit_and_reserves_typed_budgets() -> None:
    xyz, graph = _line_graph(64)
    contract = _contract_v4()
    expansion = contract.expand(graph, xyz, 0, 0.10)

    assert isinstance(expansion, SurfaceRegionExpansionV4)
    assert expansion.rows.tolist() == [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16]
    assert int(expansion.core_mask.sum()) == 6
    assert int(expansion.context_mask.sum()) == 6
    assert not bool(expansion.support_fill_mask.any())
    assert expansion.rows[expansion.anchor_index].item() == 0

    # Limiting candidate enumeration to the output width reproduces the
    # information bottleneck: only one shell token has been observed.
    narrow = _contract_v4(token_candidate_limit=12).expand(graph, xyz, 0, 0.10)
    assert int(narrow.core_mask.sum()) == 11
    assert int(narrow.context_mask.sum()) == 1


def test_v4_default_separates_candidate_and_published_token_limits() -> None:
    contract = SurfaceRegionContractV4()
    assert contract.token_candidate_limit == 1024
    assert contract.maximum_tokens == 256
    assert contract.token_candidate_limit > contract.maximum_tokens
    assert contract.digest == (
        "55d051772c24f0e27bc464c4ed55b90f519d0c919231b977006152ce1f03e147"
    )


def test_v4_core_fraction_is_effective_and_single_batch_are_exact() -> None:
    xyz, graph = _line_graph(64)
    contract = _contract_v4(core_token_fraction=0.75)
    prepared = contract.prepare_graph(graph, xyz)
    single = contract.expand(graph, xyz, 0, 0.10, prepared_graph=prepared)
    repeated = contract.expand(graph, xyz, 0, 0.10, prepared_graph=prepared)
    batched = contract.expand_batch(
        graph, xyz, [0], 0.10, prepared_graph=prepared,
    )[0]

    assert int(single.core_mask.sum()) == 9
    assert int(single.context_mask.sum()) == 3
    for candidate in (repeated, batched):
        for field in (
            "rows", "core_mask", "context_mask", "support_fill_mask",
            "semantic_geodesic_distance", "recovery_distance",
        ):
            assert torch.equal(getattr(single, field), getattr(candidate, field))
        assert single.anchor_index == candidate.anchor_index


def test_v4_unused_typed_budget_backfills_without_wasting_capacity() -> None:
    xyz, graph = _line_graph(64)
    # The shell contains only three settled nodes at this float32 boundary.
    # Its complete contribution is retained and unused context slots return to
    # core while the returned representation remains core-first.
    contract = _contract_v4(context_ratio=1.4)
    expansion = contract.expand(graph, xyz, 0, 0.10)

    assert len(expansion.rows) == contract.maximum_tokens
    assert int(expansion.core_mask.sum()) == 9
    assert int(expansion.context_mask.sum()) == 3
    assert expansion.rows.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13]


def test_v4_factory_digest_and_normalized_selection_are_version_strict() -> None:
    contract = _contract_v4()
    loaded = surface_region_contract_from_specification(contract.to_dict())
    assert isinstance(loaded, SurfaceRegionContractV4)
    assert loaded.digest == contract.digest

    xyz, graph = _line_graph(64)
    selection = as_region_selection(contract.expand(graph, xyz, 0, 0.10))
    assert selection.contract_version == "surface-region-contract-v4"
    assert torch.equal(selection.token_mask, torch.ones(12, dtype=torch.bool))


def test_v4_recovery_remains_explicit_and_single_batch_exact() -> None:
    xyz, graph = _line_graph(80)
    eligibility = torch.zeros(80, dtype=torch.bool)
    eligibility[:20] = True
    eligibility[60:64] = True
    contract = _contract_v4(
        radii_m=(0.50,),
        context_ratio=1.2,
        minimum_tokens=24,
        maximum_tokens=32,
        token_candidate_limit=80,
        core_token_fraction=0.6,
    )
    prepared = contract.prepare_graph(graph, xyz)
    single = contract.expand(
        graph,
        xyz,
        0,
        0.50,
        prepared_graph=prepared,
        selection_eligibility=eligibility,
    )
    batched = contract.expand_batch(
        graph,
        xyz,
        [0],
        0.50,
        prepared_graph=prepared,
        selection_eligibility=eligibility,
    )[0]

    assert single.rows.tolist() == list(range(20)) + list(range(60, 64))
    assert int(single.core_mask.sum()) == 20
    assert int(single.context_mask.sum()) == 0
    assert int(single.support_fill_mask.sum()) == 4
    assert bool(torch.isinf(single.semantic_geodesic_distance[20:]).all())
    assert bool(torch.isfinite(single.recovery_distance[20:]).all())
    for field in (
        "rows", "core_mask", "context_mask", "support_fill_mask",
        "semantic_geodesic_distance", "recovery_distance",
    ):
        assert torch.equal(getattr(single, field), getattr(batched, field))


def test_v3_frozen_digest_and_dense_core_behavior_remain_exact() -> None:
    xyz, graph = _line_graph(64)
    contract = SurfaceRegionContractV3(
        radii_m=(0.10,),
        context_ratio=2.0,
        minimum_tokens=1,
        maximum_tokens=12,
        token_candidate_limit=64,
        core_token_fraction=0.5,
    )
    assert contract.digest == (
        "87435a52b39767efc58d968cd9f80e708501e2d7c3af1012f5f3057a3bb99824"
    )
    expansion = contract.expand(graph, xyz, 0, 0.10)
    assert expansion.rows.tolist() == list(range(12))
    assert int(expansion.core_mask.sum()) == 11
    assert int(expansion.context_mask.sum()) == 1
