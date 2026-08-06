from __future__ import annotations

from argparse import Namespace
import copy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.interfaces.full_scalar_sparse_teacher_selection import (
    select_scale_stratified_indices,
    validate_selection_audit,
    validate_sparse_pair_cardinality,
)
from radio_gs.scripts import materialize_accepted_v2_canonical_region_authority as producer
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard


def _geometry() -> dict[str, object]:
    return {"num_gaussians": 5, "xyz_sha256": "a" * 64}


def _input_authority() -> dict:
    geometry = _geometry()
    return {
        "geometry_authority": {
            "kind": "factorized_primitive_state_v2",
            "factorized_primitive_state_file_sha256": "1" * 64,
            "factorized_primitive_state_contract_sha256": (
                shard.FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256
            ),
            "factorized_field_checkpoint_file_sha256": "2" * 64,
            "factorized_radio_cache_file_sha256": "3" * 64,
            "primitive_row_authority_sha256": "4" * 64,
            "geometry_fingerprint": geometry,
        },
        "support_graph_authority": {
            "kind": "canonical_query_free_support_graph_v1",
            "support_graph_file_sha256": "5" * 64,
            "primitive_row_authority_sha256": "4" * 64,
        },
        "selection_authority": {
            "kind": "exact_marginal_anchor_visibility_sparse_selection_v1",
            "exact_marginal_responsibility_authority_file_sha256": "6" * 64,
            "exact_marginal_formula_sha256": "7" * 64,
            "responsibility_view_records_sha256": "8" * 64,
            "sampling_contract_sha256": shard.SAMPLING_CONTRACT_SHA256,
        },
        "accepted_v2_checkpoint_authority": shard.trainer._accepted_v2_authority(),
        "official_summary_head_authority": (
            shard.accepted_region_official_head_authority()
        ),
    }


def _aligned_tensors():
    # Deliberately anchor-major and unsorted by the formal scale/anchor order.
    rows = torch.tensor(
        [[3, 2, -1], [1, 0, 4], [0, 1, -1], [2, 3, 4]],
        dtype=torch.long,
    )
    mask = torch.tensor(
        [[True, True, False], [True, True, True], [True, True, False], [True, True, True]]
    )
    anchor = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    scales = torch.tensor([1, 0, 0, 1], dtype=torch.long)
    e0 = torch.zeros(4, shard.trainer.DESCRIPTOR_DIM, dtype=torch.float32)
    e0[0, 0] = e0[1, 1] = e0[2, 2] = e0[3, 3] = 1.0
    return rows, mask, anchor, scales, e0


def _payload_from_permutation(permutation: torch.Tensor) -> dict:
    rows, mask, anchor, scales, e0 = _aligned_tensors()
    selected_rows = rows[permutation]
    selected_mask = mask[permutation]
    selected_anchor = anchor[permutation]
    selected_scales = scales[permutation]
    identities = shard._canonical_region_identity(
        "scene0001_00",
        selected_rows,
        selected_mask,
        selected_anchor,
        selected_scales,
    )
    return producer.build_authority_payload(
        scene_id="scene0001_00",
        geometry_fingerprint=_geometry(),
        accepted_base_valid=torch.ones(5, dtype=torch.bool),
        canonical_region_indices=torch.arange(4),
        region_fingerprints=[shard.canonical_json_sha256(item) for item in identities],
        selection_audit={
            "sampling_contract_sha256": shard.SAMPLING_CONTRACT_SHA256,
            "canonical_candidate_region_count": 8,
            "exact_overlap_candidate_count": 8,
            "teacher_visible_candidate_count": 4,
            "selected_region_count": 4,
            "selected_count_by_scale": [2, 2],
        },
        region_rows=selected_rows,
        token_mask=selected_mask,
        anchor_index=selected_anchor,
        scale_indices=selected_scales,
        accepted_v2_e0=e0[permutation],
        input_authority=_input_authority(),
    )


def test_payload_is_stably_canonical_and_passes_training_shard_validator() -> None:
    canonical_order = torch.tensor([2, 1, 3, 0])
    payload = _payload_from_permutation(canonical_order)
    assert payload["scale_indices"].tolist() == [0, 0, 1, 1]
    assert payload["region_rows"][:, 0].tolist() == [0, 1, 2, 3]
    validated = shard.validate_accepted_region_authority(payload)
    assert validated.keys() == payload.keys()
    assert torch.equal(validated["accepted_v2_e0"], payload["accepted_v2_e0"])
    assert validated["region_fingerprints"] == payload["region_fingerprints"]

    with pytest.raises(ValueError, match="canonical region order"):
        _payload_from_permutation(torch.tensor([3, 0, 2, 1]))


def test_input_sha_and_contamination_contracts_fail_closed() -> None:
    payload = _payload_from_permutation(torch.tensor([2, 1, 3, 0]))
    wrong_graph = copy.deepcopy(payload)
    wrong_graph["input_authority"]["support_graph_authority"][
        "primitive_row_authority_sha256"
    ] = "9" * 64
    with pytest.raises(ValueError, match="support graph row authority"):
        shard.validate_accepted_region_authority(wrong_graph)

    wrong_checkpoint = copy.deepcopy(payload)
    wrong_checkpoint["input_authority"]["accepted_v2_checkpoint_authority"][
        "checkpoint_sha256"
    ] = "8" * 64
    with pytest.raises(ValueError, match="checkpoint input authority"):
        shard.validate_accepted_region_authority(wrong_checkpoint)

    contaminated = copy.deepcopy(payload)
    contaminated["source_access"]["text_queries_opened"] = True
    with pytest.raises(ValueError, match="contract differs"):
        shard.validate_accepted_region_authority(contaminated)


def test_no_clobber_precedes_expensive_input_loading(tmp_path: Path) -> None:
    output = tmp_path / "accepted.pt"
    output.write_bytes(b"occupied")
    with pytest.raises(FileExistsError, match="refuses to clobber"):
        producer.materialize(Namespace(output=str(output), preflight_only=False))


def test_missing_real_input_reports_the_formal_upstream_chain(tmp_path: Path) -> None:
    missing = tmp_path / "clean_support_graph.pt"
    with pytest.raises(FileNotFoundError, match="Required producer chain"):
        producer._require_expected_file(
            missing,
            "a" * 64,
            label="clean support graph",
        )


def test_support_graph_parameters_must_match_accepted_v2_contract() -> None:
    contract = SurfaceRegionContractV2()
    graph = {"metadata": {"graph_config": asdict(contract.graph_config())}}
    # Pure construction chunking is permitted to differ.
    graph["metadata"]["graph_config"]["affinity_chunk_size"] = 65536
    producer.validate_accepted_v2_graph_contract(graph, contract)

    changed = copy.deepcopy(graph)
    changed["metadata"]["graph_config"]["neighbors"] = 32
    with pytest.raises(ValueError, match="AcceptedV2 graph contract"):
        producer.validate_accepted_v2_graph_contract(changed, contract)


def test_selection_batch_size_is_bitwise_execution_only() -> None:
    contract = SurfaceRegionContractV2()
    xyz = torch.tensor(
        [[0.00, 0, 0], [0.02, 0, 0], [0.04, 0, 0], [0.06, 0, 0], [0.08, 0, 0]],
        dtype=torch.float32,
    )
    appearance = torch.eye(5, dtype=torch.float32)
    boundary = appearance.clone()
    support = contract.build_graph(
        xyz,
        appearance_features=appearance,
        boundary_features=boundary,
    )
    runtime = producer._Runtime(
        scene_id="scene0001_00",
        field=None,
        state=None,
        graph_payload={"global_rows": torch.arange(5), "xyz": xyz},
        support=support,
        contract=contract,
        readout=None,
        input_authority={},
        input_records={},
        anchor_visible=torch.ones(5, dtype=torch.bool),
    )
    one = producer._select_canonical_regions(runtime, batch_size=1)
    wide = producer._select_canonical_regions(runtime, batch_size=8192)
    assert torch.equal(
        one["canonical_region_indices"], wide["canonical_region_indices"]
    )
    assert one["region_fingerprints"] == wide["region_fingerprints"]
    assert one["selection_audit"] == wide["selection_audit"]


def test_sparse_selection_rejects_4097_rows() -> None:
    with pytest.raises(ValueError, match="selection audit counts differ"):
        validate_selection_audit(
            {
                "sampling_contract_sha256": shard.SAMPLING_CONTRACT_SHA256,
                "canonical_candidate_region_count": 4097,
                "exact_overlap_candidate_count": 4097,
                "teacher_visible_candidate_count": 4097,
                "selected_region_count": 4097,
                "selected_count_by_scale": [4097],
            },
            selected_count=4097,
        )


def test_three_scale_quotas_no_redistribution_and_hash_tie_break() -> None:
    scales = torch.arange(3).repeat_interleave(4097)
    fingerprints = [f"{index:064x}" for index in range(scales.numel())]
    all_candidates = torch.ones(scales.numel(), dtype=torch.bool)
    selected, counts = select_scale_stratified_indices(
        scales, fingerprints, all_candidates
    )
    assert counts == [1366, 1365, 1365]
    assert selected.numel() == 4096

    short = all_candidates.clone()
    short[:4097] = False
    short[0] = True
    selected_short, short_counts = select_scale_stratified_indices(
        scales, fingerprints, short
    )
    assert short_counts == [1, 1365, 1365]
    assert selected_short.numel() == 2731

    tied_selected, tied_counts = select_scale_stratified_indices(
        torch.zeros(4097, dtype=torch.long),
        ["a" * 64] * 4097,
        torch.ones(4097, dtype=torch.bool),
    )
    assert tied_counts == [4096]
    assert torch.equal(tied_selected, torch.arange(4096))
    repeated, repeated_counts = select_scale_stratified_indices(
        scales, fingerprints, all_candidates
    )
    assert torch.equal(selected, repeated)
    assert counts == repeated_counts


def test_sparse_pair_cardinality_rejects_more_than_16384_pairs() -> None:
    with pytest.raises(ValueError, match="pair cardinality"):
        validate_sparse_pair_cardinality(
            selected_region_count=4096, pair_count=16385
        )
