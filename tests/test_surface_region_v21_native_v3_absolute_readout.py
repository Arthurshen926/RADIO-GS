from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import surface_region_v21_native_v3_absolute_readout as formal
from radio_gs.interfaces import surface_region_v21_query_relevance as relevance_formal
from radio_gs.utils.immutable_artifacts import canonical_json_sha256
from radio_gs.scripts import (
    materialize_surface_region_v21_native_v3_absolute_readout as materializer,
)


class QueryOpaqueMapping(dict):
    """Fail if the adapter attempts to read the opaque query-id channel."""

    def __getitem__(self, key):
        if key == "query_ids":
            raise AssertionError("query identifiers were consumed")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "query_ids":
            raise AssertionError("query identifiers were consumed")
        return super().get(key, default)


def _axes(regions: int = 10):
    canonical = torch.arange(regions, dtype=torch.int64)
    pairs = torch.stack(
        (torch.arange(regions - 1), torch.arange(1, regions))
    ).long()
    rows = torch.arange(regions, dtype=torch.int64)[:, None]
    mask = torch.ones_like(rows, dtype=torch.bool)
    active = torch.ones(regions - 1, dtype=torch.bool)
    active[-3:] = False
    fallback = ~active
    fingerprints = [f"{index:064x}" for index in range(regions)]
    fingerprint_sha = canonical_json_sha256(fingerprints)
    return canonical, pairs, rows, mask, active, fallback, fingerprint_sha


def _feature_and_inference(regions: int = 10):
    canonical, pairs, rows, mask, active, fallback, fingerprint_sha = _axes(
        regions
    )
    execution = {"path": "/authority", "sha256": "a" * 64}
    feature = {
        "schema": formal.FEATURE_SCHEMA,
        "schema_version": formal.NATIVE_V3_SCHEMA_VERSION,
        "domain": "target",
        "scene_id": "scene",
        "target_execution_authority": execution,
        "region_fingerprints_sha256": fingerprint_sha,
        "canonical_region_indices": canonical,
        "pair_indices": pairs,
        "native_pair_active_mask": active,
        "legacy_v2_fallback_pair_mask": fallback,
        "region_rows": rows,
        "token_mask": mask,
    }
    inference = {
        "schema": formal.INFERENCE_SCHEMA,
        "schema_version": formal.NATIVE_V3_SCHEMA_VERSION,
        "domain": "target",
        "scene_id": "scene",
        "target_execution_authority": execution,
        "region_fingerprints_sha256": fingerprint_sha,
        "canonical_region_indices": canonical.clone(),
        "pair_indices": pairs.clone(),
        "native_pair_active_mask": active.clone(),
        "legacy_v2_fallback_pair_mask": fallback.clone(),
        "pair_probabilities": torch.ones(regions - 1),
        "selected_rule": {
            "method": "dual_path_widest",
            "maximum_regions": 8,
            "threshold": 0.85,
        },
    }
    return feature, inference


def _relevance_authority(regions: int = 10):
    canonical, _, _, _, _, _, fingerprint_sha = _axes(regions)
    values = torch.full((regions, 2), 0.1, dtype=torch.float32)
    values[0, 0] = 0.9
    values[:, 1] = 0.2
    values[0, 1] = 0.5
    channels = {
        "region_row_ids": "1" * 64,
        "canonical_region_indices": tensor_sha256(canonical),
        "region_fingerprints": fingerprint_sha,
        "query_ids": "2" * 64,
        "region_absolute_relevance": tensor_sha256(values),
    }
    return QueryOpaqueMapping(
        {
            "schema": relevance_formal.QUERY_RELEVANCE_SCHEMA,
            "schema_version": 1,
            "contract": relevance_formal.query_relevance_contract(),
            "contract_sha256": relevance_formal.QUERY_RELEVANCE_CONTRACT_SHA256,
            "scene_id": "scene",
            "physical_space_id": "space",
            "producer": {"path": "/producer", "sha256": "3" * 64},
            "query_execution_authority": {
                "path": "/execution",
                "sha256": "4" * 64,
            },
            "input_authority": {},
            "region_row_ids": None,
            "canonical_region_indices": canonical,
            "region_fingerprints": None,
            "query_ids": object(),
            "region_absolute_relevance": values,
            "channel_sha256": channels,
            "access_audit": relevance_formal.query_relevance_access_audit(),
        }
    )


@pytest.fixture(autouse=True)
def _validated_authorities(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(formal, "validate_feature_authority", lambda value: value)
    monkeypatch.setattr(formal, "validate_inference_authority", lambda value: value)


def test_query_axis_is_opaque_and_failed_gate_is_exact_unary() -> None:
    view = formal.query_opaque_absolute_relevance_view(_relevance_authority())
    feature, inference = _feature_and_inference()
    output = formal.apply_native_v3_absolute_readout(
        relevance=view,
        feature_authority=feature,
        inference_authority=inference,
        primitive_valid=torch.ones(10, dtype=torch.bool),
    )

    assert output.query_gate.tolist() == [True, False]
    assert torch.equal(output.final_relevance[:, 1], view.values[:, 1])
    assert formal.access_audit()["query_identifiers_opened"] is False
    assert formal.access_audit()["query_strings_opened"] is False


def test_unary_only_increases_seed_is_exact_and_support_is_bounded() -> None:
    view = formal.query_opaque_absolute_relevance_view(_relevance_authority())
    feature, inference = _feature_and_inference()
    output = formal.apply_native_v3_absolute_readout(
        relevance=view,
        feature_authority=feature,
        inference_authority=inference,
        primitive_valid=torch.ones(10, dtype=torch.bool),
    )

    assert bool((output.final_relevance >= view.values).all())
    query = torch.arange(view.values.shape[1])
    assert torch.equal(
        output.final_relevance[output.seed_region_indices, query],
        view.values[output.seed_region_indices, query],
    )
    assert int(output.relation_selected_region_masks.sum(dim=0).max()) == 8
    assert max(map(len, output.union_selected_region_indices)) <= 8
    assert formal.readout_contract()["fixed_rule"] == {
        "source_selected_method": "dual_path_widest",
        "applied_path_method": "widest_path",
        "relation_threshold": 0.85,
        "maximum_regions": 8,
        "absolute_boundary": 0.5,
        "candidate_chunk_rows": 4096,
    }


def test_native_v3_fallback_probability_is_consumed_and_invalid_primitive_removed() -> None:
    view = formal.query_opaque_absolute_relevance_view(_relevance_authority())
    feature, inference = _feature_and_inference()
    valid = torch.ones(10, dtype=torch.bool)
    valid[7] = False
    output = formal.apply_native_v3_absolute_readout(
        relevance=view,
        feature_authority=feature,
        inference_authority=inference,
        primitive_valid=valid,
    )

    # Edge 6->7 is a native-ineligible fallback edge. Its frozen probability
    # remains available to the widest path, so region 7 receives completion.
    assert bool(feature["legacy_v2_fallback_pair_mask"][6])
    assert output.fallback_pair_count == 3
    assert output.fallback_pairs_above_relation_threshold == 3
    assert output.final_relevance[7, 0] > view.values[7, 0]
    assert output.invalid_primitive_memberships_removed == 1
    assert not bool(output.primitive_membership[7].any())


@pytest.mark.parametrize("drift", ["relevance", "inference", "fallback", "rule"])
def test_canonical_or_rule_mismatch_fails_closed(drift: str) -> None:
    view = formal.query_opaque_absolute_relevance_view(_relevance_authority())
    feature, inference = _feature_and_inference()
    if drift == "relevance":
        view = formal.QueryOpaqueAbsoluteRelevance(
            scene_id=view.scene_id,
            physical_space_id=view.physical_space_id,
            canonical_region_indices=torch.arange(1, 11),
            region_fingerprints_sha256=view.region_fingerprints_sha256,
            values=view.values,
        )
    elif drift == "inference":
        inference["canonical_region_indices"][0] = 99
    elif drift == "fallback":
        inference["legacy_v2_fallback_pair_mask"][0] = True
    else:
        inference["selected_rule"]["threshold"] = 0.80
    with pytest.raises(ValueError, match="canonical binding"):
        formal.apply_native_v3_absolute_readout(
            relevance=view,
            feature_authority=feature,
            inference_authority=inference,
            primitive_valid=torch.ones(10, dtype=torch.bool),
        )


def test_primitive_valid_axis_mismatch_fails_closed() -> None:
    view = formal.query_opaque_absolute_relevance_view(_relevance_authority())
    feature, inference = _feature_and_inference()
    with pytest.raises(ValueError, match="primitive-valid.*axis"):
        formal.apply_native_v3_absolute_readout(
            relevance=view,
            feature_authority=feature,
            inference_authority=inference,
            primitive_valid=torch.ones(9, dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="primitive-valid.*axis"):
        formal.apply_native_v3_absolute_readout(
            relevance=view,
            feature_authority=feature,
            inference_authority=inference,
            primitive_valid=torch.ones(10, dtype=torch.float32),
        )


def test_materializer_emits_query_opaque_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    relevance = _relevance_authority()
    feature, inference = _feature_and_inference()
    feature["input_authority"] = {
        "factorized_state": {"path": "/state", "sha256": "5" * 64}
    }
    records = {
        "relevance": (relevance, "1" * 64, tmp_path / "relevance.pt"),
        "feature": (feature, "2" * 64, tmp_path / "feature.pt"),
        "inference": (inference, "3" * 64, tmp_path / "inference.pt"),
    }

    def fake_load(path, **kwargs):
        name = Path(path).stem
        return records[name]

    captured = {}
    monkeypatch.setattr(materializer, "load_torch_mapping", fake_load)
    monkeypatch.setattr(
        materializer, "validate_feature_authority", lambda value: value
    )
    monkeypatch.setattr(
        materializer,
        "load_factorized_primitive_state",
        lambda *args, **kwargs: SimpleNamespace(
            valid=torch.ones(10, dtype=torch.bool)
        ),
    )
    monkeypatch.setattr(
        materializer,
        "write_torch_noclobber",
        lambda path, payload: captured.update(payload=payload) or path,
    )
    monkeypatch.setattr(
        materializer,
        "file_record",
        lambda path: {"path": str(path), "sha256": "f" * 64},
    )
    result = materializer.materialize(
        Namespace(
            output=str(tmp_path / "output.pt"),
            absolute_relevance_authority=str(tmp_path / "relevance"),
            expected_absolute_relevance_authority_sha256="1" * 64,
            native_v3_feature_authority=str(tmp_path / "feature"),
            expected_native_v3_feature_authority_sha256="2" * 64,
            native_v3_inference_authority=str(tmp_path / "inference"),
            expected_native_v3_inference_authority_sha256="3" * 64,
        )
    )
    payload = captured["payload"]
    assert formal.validate_readout_authority(payload) == payload
    assert "query_ids" not in payload
    assert payload["audit"]["query_identifiers_consumed"] is False
    assert payload["audit"]["query_strings_consumed"] is False
    assert result["selected_rule"]["method"] == "dual_path_widest"
    assert result["applied_rule"]["path_method"] == "widest_path"
