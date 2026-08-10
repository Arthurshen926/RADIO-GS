from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    target_physical_space_authority,
    validate_target_accepted_v2_authority,
)
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts.materialize_target_accepted_v2_canonical_region_authority import (
    build_target_authority_payload,
)
from radio_gs.utils.immutable_artifacts import file_record


GEOMETRY_SHA = "9" * 64


def _geometry() -> dict[str, object]:
    return {"num_gaussians": 5, "xyz_sha256": "a" * 64}


def _input_authority() -> dict:
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
            "geometry_fingerprint": _geometry(),
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
        "official_summary_head_authority": shard.accepted_region_official_head_authority(),
    }


def _payload() -> dict:
    scene = "figurines"
    physical = target_physical_space_authority(
        dataset_id="lerf",
        scene_id=scene,
        geometry_checkpoint_sha256=GEOMETRY_SHA,
    )["physical_space_id"]
    rows = torch.tensor([[0, 1, -1], [2, 3, 4]], dtype=torch.long)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    anchor = torch.tensor([0, 0], dtype=torch.long)
    scales = torch.tensor([0, 1], dtype=torch.long)
    identities = shard._canonical_region_identity(scene, rows, mask, anchor, scales)
    descriptor = torch.zeros(2, shard.trainer.DESCRIPTOR_DIM)
    descriptor[0, 0] = descriptor[1, 1] = 1.0
    return build_target_authority_payload(
        scene_id=scene,
        dataset_id="lerf",
        geometry_checkpoint_sha256=GEOMETRY_SHA,
        physical_space_id=physical,
        geometry_fingerprint=_geometry(),
        accepted_base_valid=torch.ones(5, dtype=torch.bool),
        canonical_region_indices=torch.tensor([0, 6]),
        region_fingerprints=[
            shard.canonical_json_sha256(identity) for identity in identities
        ],
        selection_audit={
            "sampling_contract_sha256": shard.SAMPLING_CONTRACT_SHA256,
            "canonical_candidate_region_count": 10,
            "exact_overlap_candidate_count": 10,
            "teacher_visible_candidate_count": 2,
            "selected_region_count": 2,
            "selected_count_by_scale": [1, 1],
        },
        region_rows=rows,
        token_mask=mask,
        anchor_index=anchor,
        scale_indices=scales,
        accepted_v2_e0=descriptor,
        input_authority=_input_authority(),
        producer=file_record(
            Path(
                "radio_gs/scripts/"
                "materialize_target_accepted_v2_canonical_region_authority.py"
            ).resolve()
        ),
    )


def test_target_schema_accepts_lerf_scene_without_scannet_aliasing() -> None:
    payload = _payload()
    validated = validate_target_accepted_v2_authority(payload)
    assert validated["scene_id"] == "figurines"
    assert validated["physical_space_id"].startswith(
        "lerf:figurines:geometry-checkpoint-sha256:"
    )
    assert validated["access_audit"]["target_metrics_computed"] is False


def test_target_physical_space_is_geometry_checkpoint_bound() -> None:
    payload = _payload()
    tampered = copy.deepcopy(payload)
    tampered["physical_space_id"] = (
        "lerf:figurines:geometry-checkpoint-sha256:" + "c" * 64
    )
    with pytest.raises(ValueError, match="physical-space binding"):
        validate_target_accepted_v2_authority(tampered)


def test_target_channel_and_scene_fingerprint_fail_closed() -> None:
    payload = _payload()
    tampered = copy.deepcopy(payload)
    tampered["region_fingerprints"][0] = "d" * 64
    tampered["channel_sha256"] = shard.accepted_region_channel_sha256(tampered)
    with pytest.raises(ValueError, match="canonical region order"):
        validate_target_accepted_v2_authority(tampered)


def test_target_payload_has_no_query_or_metric_channel() -> None:
    payload = _payload()
    forbidden = {"query", "queries", "text_scores", "labels", "masks", "metrics"}
    assert forbidden.isdisjoint(payload)
    assert payload["access_audit"]["benchmark_queries_opened"] is False
    assert payload["access_audit"]["target_metrics_computed"] is False
