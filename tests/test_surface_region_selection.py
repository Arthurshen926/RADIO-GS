import pytest
import torch

from radio_gs.interfaces.surface_region_contract import (
    SurfaceRegionContractV2,
    SurfaceRegionContractV3,
    SurfaceRegionExpansionV3,
)
from radio_gs.interfaces.surface_region_selection import (
    RegionSelection,
    as_region_selection,
    surface_region_contract_from_metadata,
    surface_region_contract_from_specification,
)


def _v3_expansion() -> SurfaceRegionExpansionV3:
    return SurfaceRegionExpansionV3(
        rows=torch.tensor([4, 7, 9]),
        core_mask=torch.tensor([True, False, False]),
        context_mask=torch.tensor([False, True, False]),
        support_fill_mask=torch.tensor([False, False, True]),
        semantic_geodesic_distance=torch.tensor([0.0, 0.2, torch.inf]),
        recovery_distance=torch.tensor([torch.inf, torch.inf, 0.5]),
        anchor_index=0,
    )


def test_factory_preserves_frozen_v2_digest_and_rejects_unknown_versions() -> None:
    frozen = SurfaceRegionContractV2()
    loaded = surface_region_contract_from_specification(frozen.to_dict())
    assert type(loaded) is SurfaceRegionContractV2
    assert loaded.digest == "ac77e31694ebe796befcc725ea60685ad6f97978a9a903e1029aa7a7a05abc07"
    assert loaded.to_dict() == frozen.to_dict()

    malformed = dict(frozen.to_dict())
    malformed["version"] = "surface-region-contract-v99"
    with pytest.raises(ValueError, match="unsupported"):
        surface_region_contract_from_specification(malformed)
    with pytest.raises(ValueError, match="malformed"):
        surface_region_contract_from_specification({**frozen.to_dict(), "extra": 1})


def test_factory_loads_v3_and_verifies_metadata_bindings() -> None:
    frozen = SurfaceRegionContractV3()
    metadata = {
        "region_contract": frozen.to_dict(),
        "region_contract_version": frozen.version,
        "region_contract_sha256": frozen.digest,
    }
    loaded = surface_region_contract_from_metadata(metadata)
    assert type(loaded) is SurfaceRegionContractV3
    assert loaded.digest == frozen.digest

    with pytest.raises(ValueError, match="version binding"):
        surface_region_contract_from_metadata(
            {**metadata, "region_contract_version": "surface-region-contract-v2"}
        )
    with pytest.raises(ValueError, match="digest binding"):
        surface_region_contract_from_metadata(
            {**metadata, "region_contract_sha256": "0" * 64}
        )


def test_v2_adapter_is_bit_exact_and_has_no_support_fill() -> None:
    rows = torch.tensor([3, 8, 10], dtype=torch.long)
    core = torch.tensor([True, True, False])
    distance = torch.tensor([0.0, 0.1, 0.3], dtype=torch.float32)
    selection = as_region_selection((rows, core, distance), anchor_row=3)

    assert selection.contract_version == "surface-region-contract-v2"
    assert torch.equal(selection.rows, rows)
    assert torch.equal(selection.core_mask, core)
    assert torch.equal(selection.context_mask, ~core)
    assert torch.equal(selection.semantic_geodesic_distance, distance)
    assert not bool(selection.support_fill_mask.any())
    assert bool(selection.token_mask.all())
    assert torch.isinf(selection.recovery_distance).all()


def test_v3_adapter_keeps_support_fill_as_a_real_selected_token() -> None:
    result = _v3_expansion()
    selection = as_region_selection(result, anchor_row=4)

    assert selection.contract_version == "surface-region-contract-v3"
    assert torch.equal(selection.rows, result.rows)
    assert torch.equal(selection.core_mask, result.core_mask)
    assert torch.equal(selection.context_mask, result.context_mask)
    assert torch.equal(selection.support_fill_mask, result.support_fill_mask)
    assert bool(selection.token_mask.all())
    assert bool(selection.token_mask[selection.support_fill_mask].all())
    with pytest.raises(ValueError, match="differs"):
        as_region_selection(result, anchor_row=5)


def test_padding_has_one_authority_and_canonical_inactive_values() -> None:
    padded = RegionSelection.from_v3(_v3_expansion()).pad_to(6)
    assert padded.selected_count == 3
    assert padded.width == 6
    assert padded.token_mask.tolist() == [True, True, True, False, False, False]
    assert padded.rows.tolist() == [4, 7, 9, 0, 0, 0]
    assert not bool(padded.core_mask[~padded.token_mask].any())
    assert not bool(padded.context_mask[~padded.token_mask].any())
    assert not bool(padded.support_fill_mask[~padded.token_mask].any())
    assert torch.isinf(padded.semantic_geodesic_distance[~padded.token_mask]).all()
    assert torch.isinf(padded.recovery_distance[~padded.token_mask]).all()
    assert bool(padded.token_mask[padded.support_fill_mask].all())

    with pytest.raises(ValueError, match="truncates"):
        padded.pad_to(2)
    with pytest.raises(ValueError, match="already padded"):
        padded.pad_to(7)


def test_region_selection_rejects_role_and_padding_ambiguity() -> None:
    with pytest.raises(ValueError, match="partition"):
        RegionSelection(
            rows=torch.tensor([1, 2]),
            core_mask=torch.tensor([True, False]),
            context_mask=torch.tensor([False, False]),
            support_fill_mask=torch.tensor([False, False]),
            token_mask=torch.tensor([True, True]),
            semantic_geodesic_distance=torch.tensor([0.0, 0.1]),
            recovery_distance=torch.tensor([torch.inf, torch.inf]),
            anchor_index=0,
            anchor_row=1,
            contract_version="surface-region-contract-v2",
        )
    with pytest.raises(ValueError, match="left prefix"):
        RegionSelection(
            rows=torch.tensor([1, 0, 2]),
            core_mask=torch.tensor([True, False, False]),
            context_mask=torch.tensor([False, False, True]),
            support_fill_mask=torch.zeros(3, dtype=torch.bool),
            token_mask=torch.tensor([True, False, True]),
            semantic_geodesic_distance=torch.tensor([0.0, torch.inf, 0.1]),
            recovery_distance=torch.full((3,), torch.inf),
            anchor_index=0,
            anchor_row=1,
            contract_version="surface-region-contract-v2",
        )
