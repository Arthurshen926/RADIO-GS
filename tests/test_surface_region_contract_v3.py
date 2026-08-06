from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from radio_gs.interfaces.surface_region_contract import (
    InsufficientRegionSupportError,
    PreparedSurfaceRegionGraphV3,
    SurfaceRegionContractV2,
    SurfaceRegionContractV3,
    SurfaceRegionExpansionV3,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph


def _graph(
    xyz: torch.Tensor,
    edges: list[tuple[int, int]],
    affinities: list[float],
) -> PrimitiveSupportGraph:
    edge_index = (
        torch.tensor(edges, dtype=torch.long).T.contiguous()
        if edges else torch.empty((2, 0), dtype=torch.long)
    )
    affinity = torch.tensor(affinities, dtype=torch.float32)
    return PrimitiveSupportGraph(
        edge_index=edge_index,
        edge_weight=torch.ones(len(edges)),
        raw_affinity=affinity,
        local_sigma=torch.ones(xyz.shape[0]),
        num_nodes=xyz.shape[0],
        edge_channels={"appearance": affinity, "boundary": affinity},
    )


def _three_tier_fixture() -> tuple[torch.Tensor, PrimitiveSupportGraph]:
    # Nodes 0/1 are core, node 2 is context, node 3 is reachable only through
    # the soft graph, and node 4 is disconnected and requires Euclidean fill.
    xyz = torch.tensor([
        [0.00, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.07, 0.0, 0.0],
        [0.12, 0.0, 0.0],
        [0.08, 0.0, 0.0],
    ])
    graph = _graph(xyz, [(0, 1), (1, 2), (2, 3)], [1.0, 1.0, 0.25])
    return xyz, graph


def test_v2_digest_and_behavior_remain_frozen() -> None:
    contract = SurfaceRegionContractV2()
    assert contract.digest == "ac77e31694ebe796befcc725ea60685ad6f97978a9a903e1029aa7a7a05abc07"
    assert "minimum_token_policy" not in contract.to_dict()

    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    graph = _graph(xyz, [(0, 1), (1, 0)], [1.0, 1.0])
    rows, core, distance = SurfaceRegionContractV2(
        minimum_tokens=1, maximum_tokens=2,
    ).expand(graph, xyz, 0, 0.1)
    assert rows.tolist() == [0, 1]
    assert core.tolist() == [True, True]
    torch.testing.assert_close(distance, torch.tensor([0.0, 0.05]), rtol=0, atol=0)


def test_v3_three_tiers_are_explicit_and_semantically_disjoint() -> None:
    xyz, graph = _three_tier_fixture()
    contract = SurfaceRegionContractV3(
        radii_m=(0.06,), context_ratio=1.2,
        minimum_tokens=5, maximum_tokens=5,
        minimum_appearance_affinity=0.5,
        minimum_boundary_affinity=0.5,
    )
    result = contract.expand(graph, xyz, 0, 0.06)
    assert isinstance(result, SurfaceRegionExpansionV3)
    assert result.rows.tolist() == [0, 1, 2, 3, 4]
    assert result.anchor_index == 0
    assert result.core_mask.tolist() == [True, True, False, False, False]
    assert result.context_mask.tolist() == [False, False, True, False, False]
    assert result.support_fill_mask.tolist() == [False, False, False, True, True]
    assert torch.isinf(result.semantic_geodesic_distance[3:]).all()
    torch.testing.assert_close(
        result.semantic_geodesic_distance[:3],
        torch.tensor([0.0, 0.05, 0.07]),
        rtol=0, atol=1e-7,
    )
    # Tier 2 uses relation-weighted path distance; Tier 3 uses Euclidean
    # distance and breaks equal-distance ties by node index.
    torch.testing.assert_close(result.recovery_distance[3], torch.tensor(0.27), atol=1e-6, rtol=0)
    torch.testing.assert_close(result.recovery_distance[4], torch.tensor(0.08), atol=1e-7, rtol=0)
    memberships = (
        result.core_mask.to(torch.int8)
        + result.context_mask.to(torch.int8)
        + result.support_fill_mask.to(torch.int8)
    )
    assert torch.equal(memberships, torch.ones_like(memberships))


def test_selection_eligibility_is_applied_during_selection_and_anchor_is_forced() -> None:
    xyz = torch.stack([
        torch.arange(5, dtype=torch.float32) * 0.01,
        torch.zeros(5),
        torch.zeros(5),
    ], dim=1)
    graph = _graph(xyz, [(0, 1), (1, 2), (2, 3), (3, 4)], [1.0] * 4)
    contract = SurfaceRegionContractV3(
        radii_m=(0.1,), minimum_tokens=3, maximum_tokens=3,
    )
    eligibility = torch.tensor([False, False, False, True, True])
    result = contract.expand(
        graph, xyz, 0, 0.1, selection_eligibility=eligibility,
    )
    # Ineligible nodes can neither be selected nor used as graph transit.  The
    # excluded anchor is explicitly restored, and eligible disconnected rows
    # are visibly recovered by Tier 3 rather than mislabelled as membership.
    assert result.rows.tolist() == [0, 3, 4]
    assert not bool(torch.isin(result.rows, torch.tensor([1, 2])).any())
    assert result.support_fill_mask.tolist() == [False, True, True]


def test_primary_eligibility_blocks_a_fallback_bridge_but_full_valid_fallback_expands() -> None:
    xyz = torch.tensor([
        [0.00, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.10, 0.0, 0.0],
    ])
    graph = _graph(xyz, [(0, 1), (1, 2)], [1.0, 1.0])
    contract = SurfaceRegionContractV3(
        radii_m=(0.2,), minimum_tokens=2, maximum_tokens=3,
    )

    # Nodes 0 and 2 are primary, while node 1 is a fallback bridge.  With the
    # primary-anchor policy, node 2 must not acquire a finite graph distance
    # through excluded node 1; it can enter only as explicit Tier-3 support.
    primary = contract.expand(
        graph,
        xyz,
        0,
        0.2,
        selection_eligibility=torch.tensor([True, False, True]),
    )
    assert primary.rows.tolist() == [0, 2]
    assert primary.core_mask.tolist() == [True, False]
    assert primary.support_fill_mask.tolist() == [False, True]
    assert torch.isinf(primary.semantic_geodesic_distance[1])
    torch.testing.assert_close(primary.recovery_distance[1], torch.tensor(0.1), rtol=0, atol=0)

    # A fallback anchor uses the caller's full output-valid mask, so the same
    # topology remains traversable and all three rows are strict members.
    fallback = SurfaceRegionContractV3(
        radii_m=(0.2,), minimum_tokens=3, maximum_tokens=3,
    ).expand(
        graph,
        xyz,
        1,
        0.2,
        selection_eligibility=torch.ones(3, dtype=torch.bool),
    )
    assert fallback.rows.tolist() == [1, 0, 2]
    assert fallback.core_mask.tolist() == [True, True, True]
    assert not bool(fallback.support_fill_mask.any())


def test_single_and_batch_paths_are_bit_exact_with_a_prepared_graph() -> None:
    xyz, graph = _three_tier_fixture()
    contract = SurfaceRegionContractV3(
        radii_m=(0.06,), context_ratio=1.2,
        minimum_tokens=5, maximum_tokens=5,
        minimum_appearance_affinity=0.5,
        minimum_boundary_affinity=0.5,
    )
    prepared = contract.prepare_graph(graph, xyz)
    assert isinstance(prepared, PreparedSurfaceRegionGraphV3)
    assert not prepared.xyz.flags.writeable
    assert not prepared.semantic_csr.data.flags.writeable
    assert not prepared.soft_recovery_csr.indices.flags.writeable
    with pytest.raises(ValueError):
        prepared.xyz[0, 0] = 1.0
    with pytest.raises(FrozenInstanceError):
        prepared.xyz = np.zeros((5, 3), dtype=np.float32)

    single = contract.expand(graph, xyz, 0, 0.06, prepared_graph=prepared)
    batched = contract.expand_batch(
        graph, xyz, [0], 0.06, prepared_graph=prepared,
    )[0]
    for field_name in (
        "rows", "core_mask", "context_mask", "support_fill_mask",
        "semantic_geodesic_distance", "recovery_distance",
    ):
        assert torch.equal(getattr(single, field_name), getattr(batched, field_name))
    assert single.anchor_index == batched.anchor_index

    moved_xyz = xyz.clone()
    moved_xyz[4, 0] += 0.01
    with pytest.raises(ValueError, match="not bound"):
        contract.expand(graph, moved_xyz, 0, 0.06, prepared_graph=prepared)

    changed_graph = _graph(xyz, [(0, 1), (1, 2), (2, 3)], [1.0, 0.9, 0.25])
    with pytest.raises(ValueError, match="fingerprint"):
        contract.expand(changed_graph, xyz, 0, 0.06, prepared_graph=prepared)

    changed_contract = SurfaceRegionContractV3(
        radii_m=(0.06,), context_ratio=1.2,
        minimum_tokens=5, maximum_tokens=5,
        minimum_appearance_affinity=0.6,
        minimum_boundary_affinity=0.5,
    )
    with pytest.raises(ValueError, match="different region contract"):
        changed_contract.expand(graph, xyz, 0, 0.06, prepared_graph=prepared)


def test_random_multi_anchor_batch_is_bit_exact_to_single_expansion() -> None:
    generator = torch.Generator().manual_seed(71)
    count = 96
    xyz = torch.rand(count, 3, generator=generator)
    chain = [(index, index + 1) for index in range(count - 1)]
    random_edges = [
        (int(source), int(target))
        for source, target in torch.randint(
            0, count, (320, 2), generator=generator,
        ).tolist()
        if source != target
    ]
    edges = chain + random_edges
    affinities = torch.rand(len(edges), generator=generator).tolist()
    graph = _graph(xyz, edges, affinities)
    contract = SurfaceRegionContractV3(
        radii_m=(0.16,), context_ratio=1.25,
        minimum_tokens=12, maximum_tokens=24,
        minimum_appearance_affinity=0.55,
        minimum_boundary_affinity=0.55,
    )
    eligibility = torch.rand(count, generator=generator) > 0.28
    anchors = [0, 7, 19, 43, 72, 95, 7]
    eligibility[torch.tensor(anchors)] = torch.tensor([
        False, True, False, True, False, True, True,
    ])
    prepared = contract.prepare_graph(graph, xyz)
    expected = [
        contract.expand(
            graph,
            xyz,
            anchor,
            0.16,
            prepared_graph=prepared,
            selection_eligibility=eligibility,
        )
        for anchor in anchors
    ]
    actual = contract.expand_batch(
        graph,
        xyz,
        anchors,
        0.16,
        prepared_graph=prepared,
        selection_eligibility=eligibility,
    )
    assert len(actual) == len(expected)
    for single, batched in zip(expected, actual):
        for field_name in (
            "rows", "core_mask", "context_mask", "support_fill_mask",
            "semantic_geodesic_distance", "recovery_distance",
        ):
            assert torch.equal(getattr(single, field_name), getattr(batched, field_name))
        assert single.anchor_index == batched.anchor_index


def test_batch_and_single_have_the_same_insufficient_support_failure() -> None:
    xyz, graph = _three_tier_fixture()
    contract = SurfaceRegionContractV3(
        radii_m=(0.06,), minimum_tokens=4, maximum_tokens=5,
    )
    eligibility = torch.tensor([False, True, False, False, False])
    with pytest.raises(InsufficientRegionSupportError) as single:
        contract.expand(
            graph, xyz, 0, 0.06, selection_eligibility=eligibility,
        )
    with pytest.raises(InsufficientRegionSupportError) as batched:
        contract.expand_batch(
            graph, xyz, [0, 1], 0.06, selection_eligibility=eligibility,
        )
    assert str(single.value) == str(batched.value)
    assert single.value.available_tokens == batched.value.available_tokens


def test_expand_core_does_not_recover_or_require_minimum_support() -> None:
    xyz, graph = _three_tier_fixture()
    contract = SurfaceRegionContractV3(
        radii_m=(0.06,), minimum_tokens=5, maximum_tokens=5,
        minimum_appearance_affinity=0.5,
        minimum_boundary_affinity=0.5,
    )
    result = contract.expand_core(graph, xyz, 0, 0.06)
    assert result.rows.tolist() == [0, 1]
    assert result.core_mask.tolist() == [True, True]
    assert not bool(result.context_mask.any())
    assert not bool(result.support_fill_mask.any())


def test_insufficient_global_eligibility_has_a_typed_failure() -> None:
    xyz, graph = _three_tier_fixture()
    contract = SurfaceRegionContractV3(
        radii_m=(0.06,), minimum_tokens=4, maximum_tokens=5,
    )
    eligibility = torch.tensor([False, True, False, False, False])
    with pytest.raises(InsufficientRegionSupportError) as caught:
        contract.expand(
            graph, xyz, 0, 0.06, selection_eligibility=eligibility,
        )
    assert caught.value.anchor == 0
    assert caught.value.available_tokens == 2
    assert caught.value.minimum_tokens == 4


def test_tier3_euclidean_ties_are_broken_by_node_index() -> None:
    xyz = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    graph = _graph(xyz, [], [])
    contract = SurfaceRegionContractV3(
        radii_m=(0.1,), minimum_tokens=3, maximum_tokens=3,
    )
    result = contract.expand(graph, xyz, 0, 0.1)
    assert result.rows.tolist() == [0, 1, 2]
    assert result.support_fill_mask.tolist() == [False, True, True]
    torch.testing.assert_close(result.recovery_distance[1:], torch.ones(2), rtol=0, atol=0)


def test_v3_manifest_is_digest_locked_and_rejects_ambiguous_modes() -> None:
    contract = SurfaceRegionContractV3()
    metadata = {
        "region_contract_version": contract.version,
        "region_contract_sha256": contract.digest,
    }
    contract.assert_compatible(metadata)
    assert contract.to_dict()["minimum_token_policy"] == "eligible_adaptive_support_v1"
    assert contract.feature_normalization == "l2_direction_plus_log_raw_norm_v1"
    assert contract.digest != SurfaceRegionContractV2().digest
    with pytest.raises(ValueError, match="euclidean"):
        SurfaceRegionContractV3(path_cost_mode="appearance_boundary_geometric")
    with pytest.raises(ValueError, match="nearest-geodesic"):
        SurfaceRegionContractV3(token_subsampling="core_context_radial_stratified_v1")
    with pytest.raises(ValueError, match="feature_normalization"):
        SurfaceRegionContractV3(feature_normalization="l2_direction")
