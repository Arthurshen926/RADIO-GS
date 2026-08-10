from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from radio_gs.interfaces import (
    factorized_native_contrast_v21_exact_native_v3_bridge as bridge,
)
from radio_gs.interfaces import factorized_native_contrast_v21_lerf_exact as contrast
from radio_gs.interfaces import (
    surface_region_v21_native_v3_absolute_readout as native_readout,
)
from radio_gs.scripts import (
    materialize_factorized_native_contrast_v21_exact_native_v3_bridge as materializer,
)


def _record(name: str, digit: str) -> dict[str, str]:
    return {"path": f"/tmp/{name}", "sha256": digit * 64}


def _payload(regions: int = 4, queries: int = 2) -> dict:
    canonical = torch.arange(regions, dtype=torch.int64)
    relevance = torch.linspace(
        0.1, 0.9, regions * queries, dtype=torch.float32
    ).reshape(regions, queries)
    fingerprints = [f"{index:064x}" for index in range(regions)]
    rows = [f"r{index}" for index in range(regions)]
    query_ids = [f"q{index}" for index in range(queries)]
    inputs = {
        "source_result": _record("source.json", "1"),
        "target_descriptor": _record("descriptor.pt", "2"),
        "health_v4_audit": _record("health-v4.json", "3"),
        "health_v4_preregistration": contrast.HEALTH_V4_PREREGISTRATION,
        "query_preregistration": contrast.QUERY_PREREGISTRATION,
        "exact_query_manifest": _record("manifest.json", "4"),
        "positive_text_cache": _record("positive.pt", "5"),
        "all_query_text_cache": contrast.FROZEN_ALL_QUERY_CACHE,
        "canonical_negative_bank": contrast.FROZEN_CANONICAL_NEGATIVE_BANK,
    }
    value = {
        "schema": contrast.QUERY_RELEVANCE_SCHEMA,
        "schema_version": contrast.SCHEMA_VERSION,
        "contract": contrast.query_contract(),
        "contract_sha256": contrast.QUERY_CONTRACT_SHA256,
        "scene_id": "scene",
        "physical_space_id": "space",
        "producer": _record("producer.py", "6"),
        "query_execution_authority": _record("query-authority.json", "7"),
        "input_authority": inputs,
        "region_row_ids": rows,
        "canonical_region_indices": canonical,
        "region_fingerprints": fingerprints,
        "query_ids": query_ids,
        "region_absolute_relevance": relevance,
        "channel_sha256": {},
        "access_audit": contrast.query_access_audit(),
    }
    value["channel_sha256"] = contrast.query_channel_sha256(value)
    return value


def _execution(payload: dict, relevance_record: dict[str, str]) -> dict:
    checkpoint = _record("checkpoint.pt", "8")
    descriptor_view = {
        "scene_id": payload["scene_id"],
        "physical_space_id": payload["physical_space_id"],
        "region_row_ids": list(payload["region_row_ids"]),
        "canonical_region_indices": payload["canonical_region_indices"].clone(),
        "region_fingerprints": list(payload["region_fingerprints"]),
    }
    execution = {
        **payload["input_authority"],
        "implementation": payload["producer"],
        "query_relevance_output": relevance_record["path"],
        "verified_record": payload["query_execution_authority"],
        "verified_manifest": {"query_ids": list(payload["query_ids"])},
        "verified_positive": SimpleNamespace(query_ids=tuple(payload["query_ids"])),
        "verified_prequery_gate": {
            "descriptor_view": descriptor_view,
            "descriptor": {
                "input_authority": {
                    "source_contrast_v21_checkpoint": checkpoint,
                }
            },
            "health_v4_audit": {
                "status": "pass",
                "query_authority_eligible": True,
            },
            "health_v4_audit_record": payload["input_authority"]["health_v4_audit"],
            "source_result_record": payload["input_authority"]["source_result"],
            "target_descriptor_record": payload["input_authority"]["target_descriptor"],
            "source_gate": {"result": {"checkpoint": checkpoint}},
        },
    }
    return execution


def test_dispatch_is_explicit_and_calls_strict_contrast_validator(monkeypatch) -> None:
    payload = _payload()
    called = []

    def strict(value):
        called.append(value)
        return dict(value)

    monkeypatch.setattr(contrast, "validate_query_relevance", strict)
    dispatched = bridge.dispatch_relevance_schema(payload)
    assert called == [payload]
    assert dispatched.dispatch_name == bridge.CONTRAST_EXACT_DISPATCH_NAME
    assert dispatched.schema == contrast.QUERY_RELEVANCE_SCHEMA

    future = dict(payload)
    future["schema"] = "radio_gs.future_calibrated_relevance.v1"
    with pytest.raises(ValueError, match="unsupported explicit"):
        bridge.dispatch_relevance_schema(future)
    assert called == [payload]
    assert bridge.bridge_contract()["field_guessing"] is False


def test_complete_lineage_is_bound_before_query_opaque_reduction() -> None:
    payload = _payload()
    record = _record("relevance.pt", "9")
    dispatched = bridge.dispatch_relevance_schema(payload)
    validated = bridge.validate_contrast_exact_lineage(
        dispatched=dispatched,
        relevance_record=record,
        query_execution=_execution(payload, record),
    )
    view = bridge.query_opaque_view(validated)
    assert view.scene_id == "scene"
    assert view.physical_space_id == "space"
    assert torch.equal(view.canonical_region_indices, payload["canonical_region_indices"])
    assert torch.equal(view.values, payload["region_absolute_relevance"])
    assert view.region_fingerprints_sha256 == payload["channel_sha256"][
        "region_fingerprints"
    ]
    assert bridge.bridge_access_audit()["health_v4_pass_lineage_validated"] is True
    assert bridge.bridge_access_audit()["query_identifiers_forwarded_to_readout"] is False


@pytest.mark.parametrize(
    "drift", ["authority", "health", "source", "descriptor", "query_order"]
)
def test_lineage_drift_fails_closed(drift: str) -> None:
    payload = _payload()
    record = _record("relevance.pt", "9")
    execution = _execution(payload, record)
    if drift == "authority":
        execution["verified_record"] = _record("other-authority.json", "a")
    elif drift == "health":
        execution["verified_prequery_gate"]["health_v4_audit"]["status"] = "reject"
    elif drift == "source":
        execution["verified_prequery_gate"]["source_result_record"] = _record(
            "other-source.json", "a"
        )
    elif drift == "descriptor":
        execution["verified_prequery_gate"]["descriptor_view"][
            "canonical_region_indices"
        ][0] = 99
    else:
        execution["verified_positive"] = SimpleNamespace(query_ids=("wrong", "q1"))
    with pytest.raises(ValueError):
        bridge.validate_contrast_exact_lineage(
            dispatched=bridge.dispatch_relevance_schema(payload),
            relevance_record=record,
            query_execution=execution,
        )


class AllowedFieldsOnly(dict):
    def __getitem__(self, key):
        forbidden = {
            "query_ids",
            "region_row_ids",
            "region_fingerprints",
            "input_authority",
            "query_execution_authority",
        }
        if key in forbidden:
            raise AssertionError(f"forbidden field forwarded: {key}")
        return super().__getitem__(key)


def test_reduction_reads_only_the_five_allowed_downstream_channels() -> None:
    payload = AllowedFieldsOnly(_payload())
    view = bridge.query_opaque_view(payload)
    assert view.values.shape == (4, 2)


def test_native_v3_files_remain_closed_when_query_lineage_rejects(
    monkeypatch, tmp_path: Path
) -> None:
    payload = _payload()
    paths_opened = []

    def load(path, **kwargs):
        paths_opened.append(str(path))
        if str(path) == "/tmp/relevance.pt":
            return payload, "9" * 64, Path(path)
        pytest.fail("native-V3 file opened before exact lineage passed")

    monkeypatch.setattr(materializer, "load_torch_mapping", load)
    monkeypatch.setattr(
        materializer.exact_script,
        "validate_authority",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("health lineage reject")),
    )
    args = Namespace(
        exact_relevance="/tmp/relevance.pt",
        expected_exact_relevance_sha256="9" * 64,
        native_v3_feature_authority="/tmp/feature.pt",
        expected_native_v3_feature_authority_sha256="a" * 64,
        native_v3_inference_authority="/tmp/inference.pt",
        expected_native_v3_inference_authority_sha256="b" * 64,
        output=str(tmp_path / "output.pt"),
    )
    with pytest.raises(ValueError, match="health lineage reject"):
        materializer.materialize(args)
    assert paths_opened == ["/tmp/relevance.pt"]


def test_bridge_emits_the_unchanged_native_v3_readout_payload(monkeypatch) -> None:
    regions, queries, primitives = 3, 2, 5
    unary = torch.tensor(
        [[0.6, 0.2], [0.1, 0.5], [0.2, 0.1]], dtype=torch.float32
    )
    relevance = native_readout.QueryOpaqueAbsoluteRelevance(
        scene_id="scene",
        physical_space_id="space",
        canonical_region_indices=torch.arange(regions, dtype=torch.int64),
        region_fingerprints_sha256="c" * 64,
        values=unary,
    )
    readout = native_readout.NativeV3AbsoluteReadout(
        absolute_relevance=unary,
        final_relevance=unary.clone(),
        seed_region_indices=torch.tensor([0, 1], dtype=torch.int64),
        query_gate=torch.tensor([False, False]),
        relation_selected_region_masks=torch.zeros(
            regions, queries, dtype=torch.bool
        ),
        relation_path_support=torch.zeros(regions, queries, dtype=torch.float32),
        primitive_valid=torch.ones(primitives, dtype=torch.bool),
        primitive_membership=torch.zeros(
            primitives, queries, dtype=torch.float32
        ),
        union_selected_region_indices=((), ()),
        union_selected_region_scores=((), ()),
        union_selected_marginal_core_rows=((), ()),
        invalid_primitive_memberships_removed=0,
        fallback_pair_count=0,
        fallback_pairs_above_relation_threshold=0,
    )
    monkeypatch.setattr(
        materializer,
        "file_record",
        lambda path: {"path": str(path), "sha256": "d" * 64},
    )
    records = {
        "absolute_relevance": _record("relevance.pt", "9"),
        "native_v3_feature": _record("feature.pt", "a"),
        "native_v3_inference": _record("inference.pt", "b"),
        "factorized_primitive_state": _record("state.pt", "c"),
    }
    payload = materializer._readout_payload(
        relevance=relevance,
        readout=readout,
        selected_rule={
            "method": native_readout.SOURCE_SELECTED_METHOD,
            "maximum_regions": native_readout.MAXIMUM_REGIONS,
            "threshold": native_readout.RELATION_THRESHOLD,
        },
        input_authority=records,
    )
    assert payload["schema"] == native_readout.READOUT_SCHEMA
    assert payload["contract"] == native_readout.readout_contract()
    assert bridge.BRIDGE_SCHEMA not in payload.values()
    assert "query_ids" not in payload
    assert native_readout.validate_readout_authority(payload) == payload
