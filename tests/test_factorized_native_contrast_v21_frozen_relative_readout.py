from __future__ import annotations

import copy

import pytest
import torch

from radio_gs.interfaces import (
    factorized_native_contrast_v21_frozen_relative_readout as formal,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    vala_knn_smoothed_scores,
    vala_minmax_remap_scores,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scales = torch.arange(3, dtype=torch.int64).repeat_interleave(4)
    xyz = torch.stack(
        [
            torch.arange(12, dtype=torch.float32),
            torch.zeros(12, dtype=torch.float32),
            scales.float(),
        ],
        dim=1,
    )
    raw = torch.tensor(
        [
            [0.10, 0.10],
            [0.20, 0.20],
            [0.30, 0.30],
            [0.95, 0.35],
            [0.10, 0.10],
            [0.20, 0.20],
            [0.40, 0.40],
            [0.80, 0.50],
            [0.10, 0.10],
            [0.20, 0.20],
            [0.30, 0.30],
            [0.70, 0.99],
        ],
        dtype=torch.float32,
    )
    return raw, scales, xyz


def test_matches_frozen_helpers_within_each_scale_and_peak_selects() -> None:
    raw, scales, xyz = _inputs()
    result = formal.frozen_relative_region_readout(
        raw_relevance=raw,
        scale_indices=scales,
        anchor_xyz=xyz,
        chunk_size=2,
    )
    expected_smoothed = torch.empty_like(raw)
    expected_remapped = torch.empty_like(raw)
    expected_peaks = torch.empty(3, 2)
    for level in range(3):
        rows = torch.where(scales == level)[0]
        smooth = vala_knn_smoothed_scores(
            raw[rows], xyz[rows], k=10, chunk_size=2
        )
        expected_smoothed[rows] = smooth
        expected_remapped[rows] = vala_minmax_remap_scores(smooth)
        expected_peaks[level] = smooth.amax(dim=0)
    selected = expected_peaks.argmax(dim=0)
    eligible = scales[:, None] == selected[None, :]
    expected_relative = torch.where(
        eligible, expected_remapped, torch.zeros_like(expected_remapped)
    )
    assert torch.equal(result.smoothed_relevance, expected_smoothed)
    assert torch.equal(result.remapped_relevance, expected_remapped)
    assert torch.equal(result.raw_smoothed_peaks, expected_peaks)
    assert torch.equal(result.selected_scale_indices, selected)
    assert torch.equal(result.selected_scale_eligibility, eligible)
    assert torch.equal(result.relative_relevance, expected_relative)
    assert torch.equal(
        result.unary_candidate_mask,
        eligible & (expected_relative > formal.MASK_THRESHOLD),
    )


def test_unselected_scale_is_ineligible_for_seed_graph_and_union() -> None:
    raw, scales, xyz = _inputs()
    result = formal.frozen_relative_region_readout(
        raw_relevance=raw[:, :1], scale_indices=scales, anchor_xyz=xyz
    )
    selected = int(result.selected_scale_indices[0])
    outside = scales != selected
    assert selected == 0
    assert not bool(result.selected_scale_eligibility[outside, 0].any())
    assert not bool(result.relative_relevance[outside, 0].count_nonzero())
    assert not bool(result.unary_candidate_mask[outside, 0].any())
    assert formal.readout_contract()["graph_or_relation"] == "forbidden_in_v1"
    assert (
        "seed_edge_path_relation_and_union"
        in formal.readout_contract()["scale_eligibility"][
            "future_graph_requirement"
        ]
    )


def test_equal_peaks_tie_to_lowest_scale_and_degenerate_span_fails_gate() -> None:
    scales = torch.arange(3, dtype=torch.int64).repeat_interleave(2)
    raw = torch.full((6, 1), 0.5, dtype=torch.float32)
    xyz = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    result = formal.frozen_relative_region_readout(
        raw_relevance=raw, scale_indices=scales, anchor_xyz=xyz
    )
    assert result.selected_scale_indices.tolist() == [0]
    assert not bool(result.relative_relevance.count_nonzero())
    assert result.query_gate.tolist() == [False]


@pytest.mark.parametrize("drift", ["missing_scale", "nan_xyz", "bad_probability"])
def test_input_drift_fails_closed(drift: str) -> None:
    raw, scales, xyz = _inputs()
    if drift == "missing_scale":
        scales[scales == 2] = 1
    elif drift == "nan_xyz":
        xyz[0, 0] = float("nan")
    else:
        raw[0, 0] = 1.1
    with pytest.raises(ValueError, match="inputs differ"):
        formal.frozen_relative_region_readout(
            raw_relevance=raw, scale_indices=scales, anchor_xyz=xyz
        )


def _authority_payload() -> dict:
    raw, scales, xyz = _inputs()
    result = formal.frozen_relative_region_readout(
        raw_relevance=raw, scale_indices=scales, anchor_xyz=xyz
    )
    counts = result.unary_candidate_mask.sum(dim=0)
    payload = {
        "schema": formal.READOUT_SCHEMA,
        "schema_version": formal.READOUT_SCHEMA_VERSION,
        "contract": formal.readout_contract(),
        "contract_sha256": formal.READOUT_CONTRACT_SHA256,
        "scene_id": "scene",
        "physical_space_id": "space",
        "producer": {"path": "/tmp/producer.py", "sha256": "1" * 64},
        "execution_authority": {
            "path": "/tmp/execution.json",
            "sha256": "2" * 64,
        },
        "input_authority": {},
        "region_fingerprints_sha256": "3" * 64,
        "query_axis_count": raw.shape[1],
        "canonical_region_indices": torch.arange(12, dtype=torch.int64),
        "scale_indices": scales,
        "anchor_rows": torch.arange(12, dtype=torch.int64),
        "anchor_xyz": xyz,
        "raw_relevance": result.raw_relevance,
        "smoothed_relevance": result.smoothed_relevance,
        "remapped_relevance": result.remapped_relevance,
        "raw_smoothed_peaks": result.raw_smoothed_peaks,
        "selected_scale_indices": result.selected_scale_indices,
        "selected_scale_eligibility": result.selected_scale_eligibility,
        "relative_relevance": result.relative_relevance,
        "query_gate": result.query_gate,
        "unary_candidate_mask": result.unary_candidate_mask,
        "audit": {
            "opaque_query_axes": raw.shape[1],
            "semantic_levels": 3,
            "selected_scale_counts": {
                str(level): int((result.selected_scale_indices == level).sum())
                for level in range(3)
            },
            "query_gate_passed": int(result.query_gate.sum()),
            "query_gate_failed": int((~result.query_gate).sum()),
            "candidate_count_min": int(counts.min()),
            "candidate_count_median": int(counts.float().median()),
            "candidate_count_max": int(counts.max()),
            "outside_selected_scale_nonzero": 0,
            "outside_selected_scale_candidates": 0,
            "graph_or_relation_applied": False,
            "query_identifiers_consumed_by_readout": False,
            "target_metric_computed": False,
        },
        "channel_sha256": {},
        "access_audit": formal.access_audit(),
    }
    payload["channel_sha256"] = formal.channel_sha256(payload)
    return payload


def test_output_authority_recomputes_readout_and_rejects_cross_scale_leak() -> None:
    payload = _authority_payload()
    assert formal.validate_readout_authority(payload)["audit"][
        "outside_selected_scale_nonzero"
    ] == 0
    tampered = copy.deepcopy(payload)
    outside = ~tampered["selected_scale_eligibility"]
    row, query = torch.nonzero(outside, as_tuple=True)
    tampered["relative_relevance"][row[0], query[0]] = 0.9
    tampered["channel_sha256"] = formal.channel_sha256(tampered)
    with pytest.raises(ValueError, match="authority differs"):
        formal.validate_readout_authority(tampered)
