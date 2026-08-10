from __future__ import annotations

import copy

import pytest
import torch

from radio_gs.interfaces import (
    factorized_native_contrast_v21_frozen_relative_primitive_readout as formal,
)


def _inputs() -> dict[str, torch.Tensor]:
    relevance = torch.tensor(
        [
            [0.95, 0.00],
            [0.80, 0.00],
            [0.00, 0.90],
            [0.00, 0.85],
            [0.00, 0.00],
        ],
        dtype=torch.float32,
    )
    eligibility = torch.tensor(
        [
            [True, False],
            [True, False],
            [False, True],
            [False, True],
            [False, False],
        ]
    )
    rows = torch.tensor(
        [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7]], dtype=torch.int64
    )
    return {
        "relative_relevance": relevance,
        "selected_scale_eligibility": eligibility,
        "unary_candidate_mask": eligibility & (relevance > 0.6),
        "region_rows": rows,
        "token_mask": torch.ones_like(rows, dtype=torch.bool),
        "primitive_valid": torch.tensor(
            [True, True, True, True, True, False, True, True]
        ),
    }


def test_union_is_bounded_query_opaque_and_never_crosses_selected_scale() -> None:
    inputs = _inputs()
    result = formal.frozen_relative_primitive_readout(**inputs)
    assert result.selected_region_indices == ((0, 1), (2, 3))
    assert all(len(value) <= 8 for value in result.selected_region_indices)
    for query, selected in enumerate(result.selected_region_indices):
        assert all(
            bool(inputs["selected_scale_eligibility"][index, query])
            for index in selected
        )
    assert result.primitive_membership[:, 0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert result.primitive_membership[:, 1].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    assert result.invalid_primitive_memberships_removed == 1


def test_strict_threshold_is_preserved_before_union_inclusive_gate() -> None:
    inputs = _inputs()
    inputs["relative_relevance"][0, 0] = 0.6
    inputs["unary_candidate_mask"] = inputs["selected_scale_eligibility"] & (
        inputs["relative_relevance"] > 0.6
    )
    result = formal.frozen_relative_primitive_readout(**inputs)
    assert result.candidate_probability[0, 0] == 0.0
    assert 0 not in result.selected_region_indices[0]


def test_cross_scale_candidate_drift_fails_closed() -> None:
    inputs = _inputs()
    inputs["unary_candidate_mask"][4, 0] = True
    with pytest.raises(ValueError, match="inputs differ"):
        formal.frozen_relative_primitive_readout(**inputs)


def _payload() -> dict:
    inputs = _inputs()
    result = formal.frozen_relative_primitive_readout(**inputs)
    payload = {
        "schema": formal.READOUT_SCHEMA,
        "schema_version": formal.READOUT_SCHEMA_VERSION,
        "contract": formal.readout_contract(),
        "contract_sha256": formal.READOUT_CONTRACT_SHA256,
        "scene_id": "scene",
        "physical_space_id": "space",
        "producer": {"path": "/tmp/producer.py", "sha256": "1" * 64},
        "execution_authority": {"path": "/tmp/execution", "sha256": "2" * 64},
        "input_authority": {},
        "region_fingerprints_sha256": "3" * 64,
        "query_axis_count": 2,
        "canonical_region_indices": torch.arange(5, dtype=torch.int64),
        "selected_scale_indices": torch.tensor([0, 1], dtype=torch.int64),
        "selected_scale_eligibility": inputs["selected_scale_eligibility"],
        "relative_relevance": inputs["relative_relevance"],
        "unary_candidate_mask": inputs["unary_candidate_mask"],
        "candidate_probability": result.candidate_probability,
        "region_rows": inputs["region_rows"],
        "token_mask": inputs["token_mask"],
        "primitive_valid": inputs["primitive_valid"],
        "primitive_membership": result.primitive_membership,
        "selected_region_indices": result.selected_region_indices,
        "selected_region_scores": result.selected_region_scores,
        "selected_marginal_core_rows": result.selected_marginal_core_rows,
        "audit": {
            "opaque_query_axes": 2,
            "query_gate_passed": 2,
            "maximum_union_regions": 2,
            "selected_region_total": 4,
            "selected_cross_scale_regions": 0,
            "selected_non_candidate_regions": 0,
            "primitive_memberships": int(result.primitive_membership.sum()),
            "invalid_primitive_memberships_removed": 1,
            "graph_or_relation_applied": False,
            "query_identifiers_consumed": False,
            "target_metric_computed": False,
        },
        "channel_sha256": {},
        "access_audit": formal.access_audit(),
    }
    payload["channel_sha256"] = formal.channel_sha256(payload)
    return payload


def test_authority_replays_union_and_rejects_membership_tamper() -> None:
    payload = _payload()
    assert formal.validate_readout_authority(payload)["audit"][
        "selected_cross_scale_regions"
    ] == 0
    tampered = copy.deepcopy(payload)
    tampered["primitive_membership"][7, 0] = 1.0
    tampered["channel_sha256"] = formal.channel_sha256(tampered)
    with pytest.raises(ValueError, match="authority differs"):
        formal.validate_readout_authority(tampered)
