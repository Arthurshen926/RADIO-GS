from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from radio_gs.interfaces import (
    factorized_native_contrast_v21_lerf_exact as contrast,
)
from radio_gs.interfaces import (
    lerf_v21_native_v3_frozen_metric_bridge as formal,
)
from radio_gs.scripts import (
    build_lerf_v21_native_v3_frozen_metric_bridge as builder,
)
from radio_gs.scripts import (
    run_lerf_v21_native_v3_frozen_metric as launcher,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
)


def _artifact(tmp_path: Path, name: str) -> dict[str, str]:
    path = (tmp_path / name).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode("utf-8"))
    return file_record(path)


def _bridge_inputs(tmp_path: Path) -> dict[str, object]:
    records = {
        name: _artifact(tmp_path, f"{name}.bin")
        for name in (
            "source_result",
            "target_descriptor",
            "health_v4_audit",
            "health_v4_preregistration",
            "query_preregistration",
            "exact_query_manifest",
            "positive_text_cache",
            "all_query_text_cache",
            "canonical_negative_text_cache",
            "query_execution",
            "relevance_authority",
            "readout_authority",
            "renderer_geometry_checkpoint",
            "factorized_primitive_state",
            "native_v3_feature",
            "native_v3_inference",
        )
    }
    query_ids = ["red cup", "tea pot"]
    fingerprints = ["a" * 64, "b" * 64, "c" * 64]
    relevance_values = torch.tensor(
        [[0.9, 0.1], [0.8, 0.7], [0.2, 0.6]], dtype=torch.float32
    )
    canonical = torch.tensor([2, 4, 9], dtype=torch.int64)
    input_authority = {
        "source_result": records["source_result"],
        "target_descriptor": records["target_descriptor"],
        "health_v4_audit": records["health_v4_audit"],
        "health_v4_preregistration": records["health_v4_preregistration"],
        "query_preregistration": records["query_preregistration"],
        "exact_query_manifest": records["exact_query_manifest"],
        "positive_text_cache": records["positive_text_cache"],
        "all_query_text_cache": records["all_query_text_cache"],
        "canonical_negative_bank": records["canonical_negative_text_cache"],
    }
    relevance = {
        "schema": contrast.QUERY_RELEVANCE_SCHEMA,
        "scene_id": "teatime",
        "physical_space_id": "lerf:teatime:synthetic",
        "query_execution_authority": records["query_execution"],
        "input_authority": input_authority,
        "query_ids": query_ids,
        "canonical_region_indices": canonical,
        "region_fingerprints": fingerprints,
        "region_absolute_relevance": relevance_values,
    }
    execution = {
        "verified_record": records["query_execution"],
        **input_authority,
        "verified_manifest": {
            "query_ids": query_ids,
            "query_ids_sha256": canonical_json_sha256(query_ids),
        },
        "verified_prequery_gate": {
            "descriptor": {
                "input_authority": {
                    "factorized_primitive_state": records[
                        "factorized_primitive_state"
                    ]
                }
            }
        },
    }
    membership = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    valid = torch.tensor([True, True, True, False])
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    readout = {
        "scene_id": relevance["scene_id"],
        "physical_space_id": relevance["physical_space_id"],
        "input_authority": {
            "absolute_relevance": records["relevance_authority"],
            "native_v3_feature": records["native_v3_feature"],
            "native_v3_inference": records["native_v3_inference"],
            "factorized_primitive_state": records["factorized_primitive_state"],
        },
        "region_fingerprints_sha256": canonical_json_sha256(fingerprints),
        "query_axis_count": len(query_ids),
        "canonical_region_indices": canonical.clone(),
        "absolute_relevance": relevance_values.clone(),
        "primitive_valid": valid.clone(),
        "primitive_membership": membership,
    }
    return {
        "records": records,
        "relevance": relevance,
        "execution": execution,
        "readout": readout,
        "valid": valid,
        "xyz": xyz,
    }


def _make_cache(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    values = _bridge_inputs(tmp_path)
    records = values["records"]
    cache = formal.build_external_query_score_cache(
        validated_relevance=values["relevance"],
        verified_query_execution=values["execution"],
        validated_readout=values["readout"],
        relevance_record=records["relevance_authority"],
        readout_record=records["readout_authority"],
        renderer_geometry_record=records["renderer_geometry_checkpoint"],
        exact_query_manifest_record=records["exact_query_manifest"],
        all_query_cache_record=records["all_query_text_cache"],
        canonical_negative_cache_record=records[
            "canonical_negative_text_cache"
        ],
        factorized_state_record=records["factorized_primitive_state"],
        state_xyz=values["xyz"],
        state_valid=values["valid"],
        renderer_xyz=values["xyz"].clone(),
    )
    return cache, values


def test_bridge_preserves_validated_query_axis_and_evaluator_shape(
    tmp_path: Path,
) -> None:
    cache, values = _make_cache(tmp_path)
    expected = values["readout"]["primitive_membership"]
    assert cache["metadata"]["query_names"] == ["red cup", "tea pot"]
    assert cache["query_scores"].shape == (4, 2)
    torch.testing.assert_close(cache["query_scores"], expected)
    assert cache["query_scores"][:, 0].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert cache["query_scores"][:, 1].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert cache["access_audit"]["benchmark_labels_opened"] is False


def test_bridge_rejects_opaque_axis_reorder_and_renderer_drift(
    tmp_path: Path,
) -> None:
    values = _bridge_inputs(tmp_path)
    records = values["records"]
    values["execution"]["verified_manifest"]["query_ids"] = [
        "tea pot",
        "red cup",
    ]
    values["execution"]["verified_manifest"]["query_ids_sha256"] = (
        canonical_json_sha256(["tea pot", "red cup"])
    )
    with pytest.raises(ValueError, match="query axis differs"):
        formal.build_external_query_score_cache(
            validated_relevance=values["relevance"],
            verified_query_execution=values["execution"],
            validated_readout=values["readout"],
            relevance_record=records["relevance_authority"],
            readout_record=records["readout_authority"],
            renderer_geometry_record=records["renderer_geometry_checkpoint"],
            exact_query_manifest_record=records["exact_query_manifest"],
            all_query_cache_record=records["all_query_text_cache"],
            canonical_negative_cache_record=records[
                "canonical_negative_text_cache"
            ],
            factorized_state_record=records["factorized_primitive_state"],
            state_xyz=values["xyz"],
            state_valid=values["valid"],
            renderer_xyz=values["xyz"],
        )

    values = _bridge_inputs(tmp_path / "renderer_case")
    records = values["records"]
    renderer_xyz = values["xyz"].clone()
    renderer_xyz[1, 0] += 1e-5
    with pytest.raises(ValueError, match="axis binding differs"):
        formal.build_external_query_score_cache(
            validated_relevance=values["relevance"],
            verified_query_execution=values["execution"],
            validated_readout=values["readout"],
            relevance_record=records["relevance_authority"],
            readout_record=records["readout_authority"],
            renderer_geometry_record=records["renderer_geometry_checkpoint"],
            exact_query_manifest_record=records["exact_query_manifest"],
            all_query_cache_record=records["all_query_text_cache"],
            canonical_negative_cache_record=records[
                "canonical_negative_text_cache"
            ],
            factorized_state_record=records["factorized_primitive_state"],
            state_xyz=values["xyz"],
            state_valid=values["valid"],
            renderer_xyz=renderer_xyz,
        )


def test_health_gate_runs_before_relevance_validator_can_restore_query_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"health_gate_passed": False, "query_ids_opened": False}
    record = {"path": "/synthetic/relevance.pt", "sha256": "a" * 64}
    execution_record = {"path": "/synthetic/execution.json", "sha256": "b" * 64}
    raw = {
        "schema": contrast.QUERY_RELEVANCE_SCHEMA,
        "query_execution_authority": execution_record,
        "query_ids": ["red cup"],
    }

    monkeypatch.setattr(
        builder,
        "load_torch_mapping",
        lambda *args, **kwargs: (raw, record["sha256"], Path(record["path"])),
    )

    def validate_health(*args, **kwargs):
        assert state["query_ids_opened"] is False
        state["health_gate_passed"] = True
        return {"verified_record": execution_record}

    def validate_relevance(value):
        assert state["health_gate_passed"] is True
        state["query_ids_opened"] = True
        return dict(value)

    monkeypatch.setattr(
        builder.contrast_materializer, "validate_authority", validate_health
    )
    monkeypatch.setattr(
        builder.contrast_relevance,
        "validate_query_relevance",
        validate_relevance,
    )
    relevance, _ = builder._load_health_gated_relevance_chain(record)
    assert relevance["query_ids"] == ["red cup"]
    assert state == {"health_gate_passed": True, "query_ids_opened": True}


def test_metric_authority_and_launcher_synthetic_dry_run_never_open_gt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache, values = _make_cache(tmp_path)
    records = values["records"]
    cache_path = (tmp_path / "external.pt").resolve()
    torch.save(cache, cache_path)
    cache_record = file_record(cache_path)
    config_record = _artifact(tmp_path, "frozen_config.yaml")
    label_root = (tmp_path / "gt_must_remain_absent").resolve()
    metric_output = (tmp_path / "metric_must_remain_absent").resolve()
    authority_path = (tmp_path / "metric_authority.json").resolve()
    args = argparse.Namespace(
        external_query_score_cache=cache_record["path"],
        expected_external_query_score_cache_sha256=cache_record["sha256"],
        relevance_authority=records["relevance_authority"]["path"],
        expected_relevance_authority_sha256=records["relevance_authority"]["sha256"],
        native_v3_readout_authority=records["readout_authority"]["path"],
        expected_native_v3_readout_authority_sha256=records["readout_authority"]["sha256"],
        renderer_geometry_checkpoint=records["renderer_geometry_checkpoint"]["path"],
        expected_renderer_geometry_checkpoint_sha256=records[
            "renderer_geometry_checkpoint"
        ]["sha256"],
        exact_query_manifest=records["exact_query_manifest"]["path"],
        expected_exact_query_manifest_sha256=records["exact_query_manifest"]["sha256"],
        all_query_text_cache=records["all_query_text_cache"]["path"],
        expected_all_query_text_cache_sha256=records["all_query_text_cache"]["sha256"],
        canonical_negative_text_cache=records[
            "canonical_negative_text_cache"
        ]["path"],
        expected_canonical_negative_text_cache_sha256=records[
            "canonical_negative_text_cache"
        ]["sha256"],
        config=config_record["path"],
        expected_config_sha256=config_record["sha256"],
        label_root=str(label_root),
        output_dir=str(metric_output),
        output_authority=str(authority_path),
    )
    monkeypatch.setattr(builder, "_materialize_inputs", lambda args: {"cache": cache})
    result = builder.build_metric_authority(args)
    assert result["protocol"] == formal.METRIC_PROTOCOL
    assert result["access_audit"]["label_root_opened"] is False
    assert not label_root.exists()
    assert not metric_output.exists()

    called = {"subprocess": False}

    def reject_subprocess(*args, **kwargs):
        called["subprocess"] = True
        raise AssertionError("synthetic dry-run must not start the evaluator")

    monkeypatch.setattr(launcher.subprocess, "run", reject_subprocess)
    authority_record = result["authority"]
    dry = launcher.launch(
        argparse.Namespace(
            execution_authority=authority_record["path"],
            expected_execution_authority_sha256=authority_record["sha256"],
            gpu=1,
            execute=False,
        )
    )
    assert dry["status"] == "native_v3_frozen_metric_synthetic_dry_run"
    assert called["subprocess"] is False
    assert "--scene" in dry["command"]
    assert "--protocol_preset" in dry["command"]
    assert "vala_paper_3d" in dry["command"]
    assert "--external_query_score_cache" in dry["command"]
    assert "--threshold_sweep" not in dry["command"]
    assert not label_root.exists()
    assert not metric_output.exists()


def test_launcher_exposes_no_scene_threshold_or_scan_parameters() -> None:
    parser = launcher.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert destinations == {
        "help",
        "execution_authority",
        "expected_execution_authority_sha256",
        "gpu",
        "execute",
    }
    assert formal.METRIC_PROTOCOL["protocol_preset"] == "vala_paper_3d"
    assert formal.METRIC_PROTOCOL["score_threshold"] == 0.6
    assert formal.METRIC_PROTOCOL["projection_mode"] == "selected_only_alpha"


def test_metric_authority_validation_is_label_root_io_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "schema": formal.METRIC_AUTHORITY_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "contract": formal.metric_authority_contract(),
        "contract_sha256": formal.METRIC_AUTHORITY_CONTRACT_SHA256,
        "status": "authorized_single_frozen_native_v3_lerf_metric",
        "scene_id": "teatime",
        "physical_space_id": "lerf:teatime:synthetic",
        "implementation": {"path": "/a", "sha256": "a" * 64},
        "launcher": {"path": "/b", "sha256": "b" * 64},
        "frozen_evaluator": formal.FROZEN_EVALUATOR,
        "frozen_summary_head": formal.FROZEN_SUMMARY_HEAD,
        "external_query_score_cache": {"path": "/c", "sha256": "c" * 64},
        "relevance_authority": {"path": "/d", "sha256": "d" * 64},
        "native_v3_readout_authority": {"path": "/e", "sha256": "e" * 64},
        "renderer_geometry_checkpoint": {"path": "/f", "sha256": "f" * 64},
        "exact_query_manifest": {"path": "/g", "sha256": "0" * 64},
        "all_query_text_cache": {"path": "/h", "sha256": "1" * 64},
        "canonical_negative_text_cache": {"path": "/i", "sha256": "2" * 64},
        "config": {"path": "/j", "sha256": "3" * 64},
        "label_root": "/never/open/labels",
        "output_dir": "/never/create/output",
        "protocol": formal.METRIC_PROTOCOL,
        "single_candidate_no_sweep": True,
        "scene_specific_parameters": False,
        "metric_execution_authorized": True,
        "access_audit": formal.metric_build_access_audit(),
    }
    validated = formal.validate_metric_authority_payload(payload)
    assert validated["label_root"] == "/never/open/labels"
