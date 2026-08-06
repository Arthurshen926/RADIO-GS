import copy

import pytest
import torch

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV4
from radio_gs.interfaces.surface_region_typed_context import (
    TERMINATION_CANDIDATE_CAP,
    TERMINATION_COMPLETE,
    TYPED_CONTEXT_FEATURE_DIM,
    candidate_complete_typed_context,
    candidate_complete_typed_context_batch,
    pool_typed_context_radio,
    typed_context_channel_sha256,
    validate_typed_context_authority,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts.materialize_accepted_v2_typed_context_authority import (
    _pool_candidate_rows,
    assemble_authority_payload,
    validate_local_global_row_mapping,
)


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
        [(index, index + 1) for index in range(count - 1)], dtype=torch.long
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


def _contract(*, candidate_limit: int = 64) -> SurfaceRegionContractV4:
    return SurfaceRegionContractV4(
        radii_m=(0.10,),
        context_ratio=2.0,
        minimum_tokens=1,
        maximum_tokens=12,
        token_candidate_limit=candidate_limit,
        core_token_fraction=0.5,
    )


def test_non_exhausted_selection_is_bitwise_v4_typed_selection() -> None:
    xyz, graph = _line_graph(18)
    contract = _contract(candidate_limit=64)
    prepared = contract.prepare_graph(graph, xyz)
    audited = candidate_complete_typed_context(contract, prepared, 0, 0.10)
    frozen = contract.expand(graph, xyz, 0, 0.10, prepared_graph=prepared)

    assert audited.search_complete is True
    assert audited.termination == TERMINATION_COMPLETE
    assert audited.candidate_probe_count == 18
    assert torch.equal(audited.rows, frozen.rows)
    assert torch.equal(audited.core_mask, frozen.core_mask)
    assert torch.equal(audited.context_mask, frozen.context_mask)
    assert torch.equal(
        audited.semantic_geodesic_distance,
        frozen.semantic_geodesic_distance,
    )


def test_candidate_cap_is_explicit_and_disables_pool() -> None:
    xyz, graph = _line_graph(64)
    contract = _contract(candidate_limit=12)
    prepared = contract.prepare_graph(graph, xyz)
    audited = candidate_complete_typed_context(contract, prepared, 0, 0.10)
    assert audited.search_complete is False
    assert audited.termination == TERMINATION_CANDIDATE_CAP
    assert audited.candidate_probe_count == 13
    context_count = int(audited.context_mask.sum())
    values = torch.randn(context_count, TYPED_CONTEXT_FEATURE_DIM)
    pooled = pool_typed_context_radio(
        values,
        torch.ones(context_count),
        audited.semantic_geodesic_distance[audited.context_mask],
        raw_anchor_radio=torch.randn(TYPED_CONTEXT_FEATURE_DIM),
        radius_m=0.10,
        selected_semantic_token_count=audited.rows.numel(),
        search_complete=False,
        context_ratio=2.0,
    )
    assert pooled.context_present is (context_count > 0)
    assert pooled.pool_valid is False
    assert not bool(pooled.direction.count_nonzero())
    assert not bool(pooled.statistics.count_nonzero())


def test_candidate_batch_is_exactly_repeated_single() -> None:
    xyz, graph = _line_graph(40)
    contract = _contract(candidate_limit=32)
    prepared = contract.prepare_graph(graph, xyz)
    anchors = [0, 5, 17]
    batched = candidate_complete_typed_context_batch(
        contract, prepared, anchors, 0.10
    )
    repeated = [
        candidate_complete_typed_context(contract, prepared, anchor, 0.10)
        for anchor in anchors
    ]
    for left, right in zip(batched, repeated):
        assert left.termination == right.termination
        assert left.candidate_probe_count == right.candidate_probe_count
        for name in (
            "rows",
            "core_mask",
            "context_mask",
            "semantic_geodesic_distance",
        ):
            assert torch.equal(getattr(left, name), getattr(right, name))


class _SyntheticField(torch.nn.Module):
    def radio_features(self, rows: torch.Tensor) -> torch.Tensor:
        values = torch.zeros(rows.numel(), TYPED_CONTEXT_FEATURE_DIM, device=rows.device)
        values[:, 0] = 1.0
        values[:, 1] = rows.float() / 100.0
        return values


class _ForbiddenDecodeField(torch.nn.Module):
    def radio_features(self, rows: torch.Tensor) -> torch.Tensor:
        raise AssertionError("cap-hit rows must not decode context features")


def test_sparse_pool_preserves_global_local_mapping_without_dense_tokens() -> None:
    xyz, graph = _line_graph(18)
    contract = _contract(candidate_limit=64)
    selection = candidate_complete_typed_context(
        contract, contract.prepare_graph(graph, xyz), 0, 0.10
    )
    graph_global = torch.arange(100, 118)
    pooled = _pool_candidate_rows(
        [selection],
        radii_m=(0.10,),
        context_ratio=2.0,
        scale_indices=torch.tensor([0]),
        anchor_local_rows=torch.tensor([0]),
        graph_global_rows=graph_global,
        reliability=torch.ones(18),
        field=_SyntheticField(),
        field_batch_size=4,
        device=torch.device("cpu"),
    )
    assert pooled["pooled_context_radio_direction"].shape == (1, 1280)
    assert pooled["typed_context_valid"].tolist() == [True]
    assert torch.equal(
        graph_global[pooled["context_token_local_rows"]],
        pooled["context_token_global_rows"],
    )
    assert int(pooled["context_token_count"][0]) == int(selection.context_mask.sum())


def test_materializer_cap_hit_keeps_sparse_audit_but_zeroes_carrier() -> None:
    xyz, graph = _line_graph(64)
    contract = _contract(candidate_limit=12)
    selection = candidate_complete_typed_context(
        contract, contract.prepare_graph(graph, xyz), 0, 0.10
    )
    assert not selection.search_complete and bool(selection.context_mask.any())
    pooled = _pool_candidate_rows(
        [selection],
        radii_m=(0.10,),
        context_ratio=2.0,
        scale_indices=torch.tensor([0]),
        anchor_local_rows=torch.tensor([0]),
        graph_global_rows=torch.arange(100, 164),
        reliability=torch.ones(64),
        field=_ForbiddenDecodeField(),
        field_batch_size=4,
        device=torch.device("cpu"),
    )
    assert pooled["context_present"].tolist() == [True]
    assert pooled["typed_context_valid"].tolist() == [False]
    assert not bool(pooled["pooled_context_radio_direction"].count_nonzero())
    assert not bool(pooled["typed_context_statistics"].count_nonzero())
    assert int(pooled["context_token_row_offsets"][-1]) > 0


def test_pool_is_unit_finite_and_nan_fails_closed() -> None:
    values = torch.zeros(2, TYPED_CONTEXT_FEATURE_DIM)
    values[0, 0] = 2.0
    values[1, 1] = 1.0
    anchor = torch.zeros(TYPED_CONTEXT_FEATURE_DIM)
    anchor[0] = 1.0
    pooled = pool_typed_context_radio(
        values,
        torch.tensor([1.0, 0.5]),
        torch.tensor([0.11, 0.15]),
        raw_anchor_radio=anchor,
        radius_m=0.10,
        selected_semantic_token_count=6,
        search_complete=True,
        context_ratio=2.0,
    )
    assert pooled.context_present and pooled.pool_valid
    assert pooled.direction.dtype == torch.float16
    assert torch.allclose(
        pooled.direction.float().norm(), torch.tensor(1.0), atol=1e-3, rtol=0.0
    )
    assert pooled.statistics.shape == (12,)
    assert bool(torch.isfinite(pooled.statistics).all())

    broken = values.clone()
    broken[0, 3] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        pool_typed_context_radio(
            broken,
            torch.ones(2),
            torch.tensor([0.11, 0.15]),
            raw_anchor_radio=anchor,
            radius_m=0.10,
            selected_semantic_token_count=6,
            search_complete=True,
            context_ratio=2.0,
        )


def _sha(character: str) -> str:
    return character * 64


def _synthetic_payload() -> dict:
    first = candidate_complete_typed_context_selection(
        rows=[0, 1], core=[True, False], complete=True
    )
    second = candidate_complete_typed_context_selection(
        rows=[2], core=[True], complete=True
    )
    directions = torch.zeros(2, TYPED_CONTEXT_FEATURE_DIM, dtype=torch.float16)
    directions[0, 0] = 1.0
    statistics = torch.zeros(2, 12)
    statistics[0, 0] = -1.0
    accepted = {
        "scene_id": "scene0001_00",
        "physical_space_id": "scene0001",
        "region_fingerprints": [_sha("1"), _sha("2")],
        "channel_sha256": {"region_rows": _sha("3")},
        "canonical_region_indices": torch.tensor([4, 9]),
        "scale_indices": torch.tensor([0, 1]),
    }
    record = {"path": "/synthetic/source.pt", "sha256": _sha("a")}
    return assemble_authority_payload(
        accepted=accepted,
        accepted_file=record,
        field_file={"path": "/synthetic/field.pt", "sha256": _sha("b")},
        state_file={"path": "/synthetic/state.pt", "sha256": _sha("c")},
        factorized_radio_cache_sha256=_sha("d"),
        graph_file={"path": "/synthetic/graph.pt", "sha256": _sha("e")},
        primitive_row_authority_sha256=_sha("f"),
        anchor_local_rows=torch.tensor([0, 2]),
        anchor_global_rows=torch.tensor([100, 102]),
        selections=[first, second],
        pooled={
            "pooled_context_radio_direction": directions,
            "typed_context_statistics": statistics,
            "context_present": torch.tensor([True, False]),
            "typed_context_valid": torch.tensor([True, False]),
            "context_token_count": torch.tensor([1, 0]),
            "context_token_row_offsets": torch.tensor([0, 1, 1]),
            "context_token_local_rows": torch.tensor([1]),
            "context_token_global_rows": torch.tensor([101]),
        },
    )


def candidate_complete_typed_context_selection(
    *, rows: list[int], core: list[bool], complete: bool
):
    from radio_gs.interfaces.surface_region_typed_context import (
        CandidateCompleteTypedSelection,
    )

    distances = torch.tensor([0.0 if value else 0.11 for value in core])
    return CandidateCompleteTypedSelection(
        rows=torch.tensor(rows),
        core_mask=torch.tensor(core),
        context_mask=~torch.tensor(core),
        semantic_geodesic_distance=distances,
        candidate_probe_count=len(rows),
        search_complete=complete,
        termination=TERMINATION_COMPLETE if complete else TERMINATION_CANDIDATE_CAP,
    )


def test_payload_validates_mapping_hash_and_inactive_zero_gates() -> None:
    payload = _synthetic_payload()
    assert "accepted_v2_e0" not in payload
    assert validate_typed_context_authority(payload)["typed_context_valid"].tolist() == [
        True,
        False,
    ]
    validate_local_global_row_mapping(
        torch.tensor([0, 1]),
        torch.tensor([100, 101]),
        torch.tensor([100, 101, 102]),
    )
    with pytest.raises(ValueError, match="mapping"):
        validate_local_global_row_mapping(
            torch.tensor([0, 1]),
            torch.tensor([100, 999]),
            torch.tensor([100, 101, 102]),
        )

    tampered = copy.deepcopy(payload)
    tampered["typed_context_statistics"][1, 0] = 1.0
    tampered["channel_sha256"] = typed_context_channel_sha256(tampered)
    with pytest.raises(ValueError, match="inactive"):
        validate_typed_context_authority(tampered)

    tampered = copy.deepcopy(payload)
    tampered["typed_context_statistics"][0, 0] = torch.nan
    tampered["channel_sha256"] = typed_context_channel_sha256(tampered)
    with pytest.raises(ValueError, match="layout"):
        validate_typed_context_authority(tampered)

    tampered = copy.deepcopy(payload)
    tampered["context_token_global_rows"][0] = 777
    with pytest.raises(ValueError, match="SHA"):
        validate_typed_context_authority(tampered)
