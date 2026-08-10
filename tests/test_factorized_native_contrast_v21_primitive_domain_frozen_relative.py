from __future__ import annotations

import copy

import pytest
import torch

from radio_gs.interfaces import factorized_native_contrast_v21_primitive_domain_frozen_relative as formal
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    vala_knn_smoothed_scores,
    vala_minmax_remap_scores,
)


def _inputs():
    raw = torch.tensor(
        [
            [0.9, 0.2], [0.3, 0.4],
            [0.2, 0.8], [0.6, 0.3],
            [0.4, 0.1], [0.5, 0.95],
        ], dtype=torch.float32,
    )
    scales = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    rows = torch.tensor(
        [[0, 1], [1, 2], [0, 3], [3, 4], [2, 5], [5, 6]], dtype=torch.int64
    )
    mask = torch.ones_like(rows, dtype=torch.bool)
    xyz = torch.stack(
        (torch.arange(8, dtype=torch.float32), torch.zeros(8), torch.zeros(8)), dim=1
    )
    valid = torch.tensor([True, True, True, True, True, True, True, False])
    return raw, scales, rows, mask, xyz, valid


def test_covering_region_max_preserves_strongest_overlap_evidence() -> None:
    raw, scales, rows, mask, xyz, _ = _inputs()
    counts, projected = formal.project_covering_region_max(
        region_raw_relevance=raw,
        scale_indices=scales,
        region_rows=rows,
        token_mask=mask,
        num_primitives=xyz.shape[0],
    )
    assert counts[0, 1].item() == 2
    assert projected[0, 1, 0].item() == pytest.approx(0.9)
    assert projected[0, 1, 1].item() == pytest.approx(0.4)
    assert projected[0, 7].count_nonzero() == 0


def test_max_is_duplicate_invariant_while_overlap_mean_is_not() -> None:
    raw = torch.tensor([[0.9], [0.1]], dtype=torch.float32)
    scales = torch.tensor([0, 0])
    rows = torch.tensor([[0], [0]])
    mask = torch.ones_like(rows, dtype=torch.bool)
    _, first = formal.project_covering_region_max(
        region_raw_relevance=raw, scale_indices=scales, region_rows=rows,
        token_mask=mask, num_primitives=1,
    )
    _, duplicated = formal.project_covering_region_max(
        region_raw_relevance=torch.tensor([[0.9], [0.1], [0.1]]),
        scale_indices=torch.tensor([0, 0, 0]),
        region_rows=torch.tensor([[0], [0], [0]]),
        token_mask=torch.ones(3, 1, dtype=torch.bool), num_primitives=1,
    )
    assert first[0, 0, 0].item() == duplicated[0, 0, 0].item() == pytest.approx(0.9)
    assert float(raw.mean()) != pytest.approx(float(torch.tensor([0.9, 0.1, 0.1]).mean()))
    audit = formal.projection_rule_audit()
    assert audit["frozen_rule"] == "covering_region_max"
    assert audit["candidate_rules"]["coverage_weighted_mean"][
        "idempotent_under_duplicate_region_support"
    ] is False


def test_readout_executes_knn_and_remap_on_covered_primitive_xyz() -> None:
    raw, scales, rows, mask, xyz, valid = _inputs()
    result = formal.primitive_domain_frozen_relative_readout(
        region_raw_relevance=raw, scale_indices=scales, region_rows=rows,
        token_mask=mask, primitive_xyz=xyz, primitive_valid=valid, chunk_size=2,
    )
    for level in range(3):
        primitive = torch.where(result.projection_coverage[level])[0]
        smooth = vala_knn_smoothed_scores(
            result.projected_raw_relevance[level, primitive], xyz[primitive],
            k=10, chunk_size=2,
        )
        remapped = vala_minmax_remap_scores(smooth)
        assert torch.equal(result.smoothed_relevance[level, primitive], smooth)
        assert torch.equal(result.remapped_relevance[level, primitive], remapped)
        assert torch.equal(result.raw_smoothed_peaks[level], smooth.amax(dim=0))
    assert torch.equal(
        result.unary_candidate_mask,
        result.selected_scale_eligibility & (result.relative_relevance > 0.6),
    )


def test_uncovered_and_unselected_primitives_are_strictly_ineligible() -> None:
    raw, scales, rows, mask, xyz, valid = _inputs()
    result = formal.primitive_domain_frozen_relative_readout(
        region_raw_relevance=raw, scale_indices=scales, region_rows=rows,
        token_mask=mask, primitive_xyz=xyz, primitive_valid=valid,
    )
    assert not bool(result.projection_coverage[:, 7].any())
    assert not bool(result.selected_scale_eligibility[7].any())
    assert not bool(result.relative_relevance[~result.selected_scale_eligibility].count_nonzero())
    assert not bool(result.unary_candidate_mask[~result.selected_scale_eligibility].any())


def _payload():
    raw, scales, rows, mask, xyz, valid = _inputs()
    result = formal.primitive_domain_frozen_relative_readout(
        region_raw_relevance=raw, scale_indices=scales, region_rows=rows,
        token_mask=mask, primitive_xyz=xyz, primitive_valid=valid,
    )
    payload = {
        "schema": formal.READOUT_SCHEMA,
        "schema_version": formal.READOUT_SCHEMA_VERSION,
        "contract": formal.readout_contract(),
        "contract_sha256": formal.READOUT_CONTRACT_SHA256,
        "scene_id": "scene", "physical_space_id": "space",
        "producer": {"path": "/tmp/p.py", "sha256": "1" * 64},
        "execution_authority": {"path": "/tmp/a.json", "sha256": "2" * 64},
        "input_authority": {},
        "query_axis_count": raw.shape[1],
        "region_raw_relevance": raw,
        "scale_indices": scales, "region_rows": rows, "token_mask": mask,
        "primitive_xyz": xyz, "primitive_valid": valid,
        "projection_coverage_count": result.projection_coverage_count,
        "projection_coverage": result.projection_coverage,
        "projected_raw_relevance": result.projected_raw_relevance,
        "smoothed_relevance": result.smoothed_relevance,
        "remapped_relevance": result.remapped_relevance,
        "raw_smoothed_peaks": result.raw_smoothed_peaks,
        "selected_scale_indices": result.selected_scale_indices,
        "selected_scale_eligibility": result.selected_scale_eligibility,
        "relative_relevance": result.relative_relevance,
        "query_gate": result.query_gate,
        "unary_candidate_mask": result.unary_candidate_mask,
        "audit": formal.expected_audit(result),
        "channel_sha256": {}, "access_audit": formal.access_audit(),
    }
    payload["channel_sha256"] = formal.channel_sha256(payload)
    return payload


def test_output_replays_exactly_and_rejects_candidate_tamper() -> None:
    payload = _payload()
    assert formal.validate_readout_authority(payload)["audit"]["target_metric_computed"] is False
    tampered = copy.deepcopy(payload)
    row, query = torch.nonzero(tampered["unary_candidate_mask"], as_tuple=True)
    if row.numel() == 0:
        pytest.skip("synthetic response has no candidate")
    tampered["unary_candidate_mask"][row[0], query[0]] = False
    tampered["channel_sha256"] = formal.channel_sha256(tampered)
    with pytest.raises(ValueError, match="authority differs"):
        formal.validate_readout_authority(tampered)


def test_contract_is_query_opaque_parameter_free_and_independent() -> None:
    contract = formal.readout_contract()
    assert contract["fixed_primitive_rule"]["knn_domain"].startswith("covered_valid_primitive_xyz")
    assert contract["scene_specific_parameters"] is False
    assert contract["query_specific_parameters"] is False
    assert contract["threshold_scan"] is False
    assert contract["metric_access"] is False
    assert formal.access_audit()["existing_region_relative_candidate_opened"] is False
    assert formal.access_audit()["existing_candidate_modified"] is False
