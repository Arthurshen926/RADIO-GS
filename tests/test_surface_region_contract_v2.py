import torch

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2


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
