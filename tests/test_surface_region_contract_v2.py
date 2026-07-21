import torch

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.querying.support_solver import PrimitiveSupportGraph


def test_contract_expansion_is_exactly_shared_and_digest_locked() -> None:
    contract = SurfaceRegionContractV2(
        radii_m=(0.20, 0.40, 0.70), minimum_tokens=1, maximum_tokens=32,
        neighbors=2,
    )
    xyz = torch.stack([torch.arange(12) * 0.05, torch.zeros(12), torch.zeros(12)], 1)
    appearance = torch.nn.functional.normalize(torch.randn(12, 5), dim=-1)
    boundary = torch.nn.functional.normalize(torch.randn(12, 7), dim=-1)
    graph = contract.build_graph(
        xyz, appearance_features=appearance, boundary_features=boundary
    )
    for anchor in range(12):
        for radius in contract.radii_m:
            train_rows, train_core, train_distance = contract.expand(
                graph, xyz, anchor, radius
            )
            infer_rows, infer_core, infer_distance = contract.expand(
                graph, xyz, anchor, radius
            )
            assert torch.equal(train_rows, infer_rows)
            assert torch.equal(train_core, infer_core)
            torch.testing.assert_close(train_distance, infer_distance, rtol=0, atol=0)
    metadata = {
        "region_contract_version": contract.version,
        "region_contract_sha256": contract.digest,
    }
    contract.assert_compatible(metadata)
    changed = SurfaceRegionContractV2(
        radii_m=(0.21, 0.40, 0.70), minimum_tokens=1, maximum_tokens=32,
        neighbors=2,
    )
    try:
        changed.assert_compatible(metadata)
    except ValueError:
        pass
    else:
        raise AssertionError("changed contract must fail closed")


def test_affinity_weighted_path_consumes_more_budget_at_weak_relation() -> None:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    graph = PrimitiveSupportGraph(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_weight=torch.ones(2), raw_affinity=torch.ones(2),
        local_sigma=torch.ones(2), num_nodes=2,
        edge_channels={
            "appearance": torch.full((2,), 0.25),
            "boundary": torch.full((2,), 0.25),
        },
    )
    euclidean = SurfaceRegionContractV2(
        radii_m=(1.5,), minimum_tokens=1, maximum_tokens=8,
    )
    weighted = SurfaceRegionContractV2(
        radii_m=(1.5,), minimum_tokens=1, maximum_tokens=8,
        path_cost_mode="appearance_boundary_geometric",
    )
    assert euclidean.expand(graph, xyz, 0, 1.5)[0].tolist() == [0, 1]
    assert weighted.expand(graph, xyz, 0, 1.5)[0].tolist() == [0]
    assert weighted.digest != euclidean.digest
    assert (
        weighted.to_dict()["expansion"]
        == "undirected_dijkstra_relation_weighted_physical_edge_cost"
    )


def test_stratified_sampling_is_batch_exact_and_keeps_context() -> None:
    xyz = torch.stack([torch.arange(64).float() * 0.01, torch.zeros(64), torch.zeros(64)], 1)
    graph = PrimitiveSupportGraph(
        edge_index=torch.stack([
            torch.cat([torch.arange(63), torch.arange(1, 64)]),
            torch.cat([torch.arange(1, 64), torch.arange(63)]),
        ]),
        edge_weight=torch.ones(126), raw_affinity=torch.ones(126),
        local_sigma=torch.full((64,), 0.02), num_nodes=64,
        edge_channels={"appearance": torch.ones(126), "boundary": torch.ones(126)},
    )
    contract = SurfaceRegionContractV2(
        radii_m=(0.20,), context_ratio=1.20, minimum_tokens=1,
        maximum_tokens=12, token_candidate_limit=64,
        token_subsampling="core_context_radial_stratified_v1",
    )
    single = contract.expand(graph, xyz, 0, 0.20)
    batched = contract.expand_batch(graph, xyz, [0], 0.20)[0]
    for left, right in zip(single, batched):
        assert torch.equal(left, right)
    assert int((~single[1]).sum()) >= 4
