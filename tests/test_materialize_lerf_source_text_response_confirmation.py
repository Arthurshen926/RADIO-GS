import copy

import pytest
import torch

import radio_gs.scripts.materialize_lerf_source_text_response_confirmation as confirmation
import radio_gs.scripts.materialize_lerf_source_text_response_summaries as base


def _record(seed: str) -> dict[str, str]:
    return {"path": f"/tmp/{seed}", "sha256": seed[0] * 64}


def _authority() -> dict[str, object]:
    audit = dict(confirmation.AUDIT_BANK)
    return {
        "schema": confirmation.AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": confirmation.AUTHORITY_STATUS,
        "implementation": confirmation.IMPLEMENTATION,
        "base_materializer": confirmation.BASE_IMPLEMENTATION,
        "frame_evaluator_implementation": base.FRAME_EVALUATOR_IMPLEMENTATION,
        "scene_id": "ramen",
        "source_heldout_frame_ids": [2, 45],
        "forbidden_target_frame_ids": [6, 24],
        "inputs": {
            "source_gate_preregistration": _record("a"),
            "source_view_preregistration": _record("b"),
            "scene_config": _record("c"),
            "geometry_checkpoint": _record("d"),
            "query_bank_artifact": {
                "path": audit["path"],
                "sha256": audit["sha256"],
            },
            "query_bank_manifest": {
                "path": audit["manifest_path"],
                "sha256": audit["manifest_sha256"],
            },
            "source_reseal": {
                "path": "/tmp/reseal.json",
                "sha256": "7" * 64,
                "required_schema": "radio_gs.lerf_official_crop_summary_reseal.v1",
                "required_mode": "content_addressed_immutable_reseal",
            },
        },
        "methods": [
            {
                "method_id": "legacy_o2",
                "role": "control",
                "descriptor_payload": _record("e"),
                "descriptor_payload_kind": base.SPARSE_TEACHER_MEAN_KIND,
                "descriptor_payload_contract": {
                    "schema": "radio_gs.lerf_source_teacher_mean_siglip.v2",
                    "schema_version": 2,
                    "descriptor_dimension": 1536,
                },
                "descriptor_provenance_authority": _record("f"),
                "descriptor_geometry_authority": _record("7"),
            },
            {
                "method_id": "crop_summary_mpr",
                "role": "candidate",
                "descriptor_payload": _record("9"),
                "descriptor_payload_kind": base.DENSE_OFFICIAL_CROP_SUMMARY_MPR_KIND,
                "descriptor_payload_contract": {
                    "metadata_schema_version": 1,
                    "feature_space": "semantic_descriptor",
                    "construction": "semantic_descriptor_raster_gaussian_top1_contribution_mean",
                    "descriptor_dimension": 1536,
                },
                "descriptor_provenance_authority": _record("8"),
                "descriptor_geometry_authority": _record("8"),
            },
        ],
        "equivalence_smoke": False,
        "query_bank": audit,
        "geometry": {"num_gaussians": 5, "xyz_sha256": "4" * 64},
        "execution": {
            "required_cuda_visible_devices": "0",
            "program_device": "cuda:0",
            "thermal_guard": base.THERMAL_GUARD,
            "thermal_poll_seconds": 300,
            "maximum_temperature_c": 88,
            "soft_pause_temperature_c": 0,
        },
        "outputs": {
            "control_summary": "/tmp/control.json",
            "candidate_summary": "/tmp/candidate.json",
            "result": "/tmp/result.json",
        },
        "confirmation": {
            "candidate_locked_after_dev_gate": True,
            "dev_gate_result": _record("1"),
            "reserved_bank_opened_once": True,
            "dev_audit_disjointness_proof": _record("2"),
            "end_to_end_interface_ab_not_capacity_matched": True,
        },
        "access_audit": {
            "benchmark_queries_opened": False,
            "benchmark_masks_or_labels_opened": False,
            "target_metric_execution_authorized": False,
        },
    }


def _allow_confirmation_records(monkeypatch) -> None:
    proof = {
        "schema": "radio_gs.lerf_source_text_response_dev_audit_disjointness_proof.v1",
        "schema_version": 1,
        "status": "sealed_disjoint_before_reserved_audit_confirmation",
        "dev_manifest": confirmation.DEV_MANIFEST,
        "audit_manifest": {
            "path": confirmation.AUDIT_BANK["manifest_path"],
            "sha256": confirmation.AUDIT_BANK["manifest_sha256"],
            "split": "audit",
            "queries": 90,
            "synsets": 90,
            "ordered_records_sha256": "0bfe94aeb6b5e0fecc978c6c66d77bba0fc0b5b7be59d922a801915843bd748f",
            "split_synset_tab_query_lf_sha256": "3b78a2e81e2750dd7314d6431ac44ddea05dd505948d775e9d1e33e87ae0bc7b",
        },
        "proof": {
            "query_intersection_count": 0,
            "synset_intersection_count": 0,
            "dev_audit_disjoint": True,
            "benchmark_vocabulary_opened_by_banks": False,
        },
        "access_audit": {
            "reserved_audit_embedding_tensor_opened": False,
            "benchmark_queries_masks_or_labels_opened": False,
            "target_metric_execution_authorized": False,
        },
    }
    gate = {
        "schema": "radio_gs.lerf_source_text_response_ranking_gate.v1",
        "status": "passed",
        "candidate_method_id": "crop_summary_mpr",
        "decision": {"candidate_eligible_for_next_source_gate": True},
        "metric_execution_authorized": False,
        "metric_executed": False,
    }
    dev_manifest = {
        "split": "dev",
        "benchmark_vocabulary_opened": False,
        "queries": [f"dev-{index}" for index in range(101)],
        "synsets": [f"dev-synset-{index}" for index in range(101)],
        "ordered_records_sha256": confirmation.DEV_MANIFEST["ordered_records_sha256"],
    }
    audit_manifest = {
        "split": "audit",
        "benchmark_vocabulary_opened": False,
        "queries": [f"audit-{index}" for index in range(90)],
        "synsets": [f"audit-synset-{index}" for index in range(90)],
        "ordered_records_sha256": "0bfe94aeb6b5e0fecc978c6c66d77bba0fc0b5b7be59d922a801915843bd748f",
    }
    monkeypatch.setattr(confirmation, "validate_file_record", lambda *args, **kwargs: None)

    def fake_load(path, *, expected_sha256, label):
        if "disjointness" in label:
            value = proof
        elif "development source" in label:
            value = gate
        elif "development query-bank manifest" in label:
            value = dev_manifest
        else:
            value = audit_manifest
        return value, expected_sha256, path

    monkeypatch.setattr(confirmation, "load_json_object", fake_load)


def test_confirmation_authority_is_exact_audit_90_and_target_blind(monkeypatch) -> None:
    _allow_confirmation_records(monkeypatch)
    assert confirmation.validate_authority(_authority())["query_bank"] == confirmation.AUDIT_BANK
    leaked = copy.deepcopy(_authority())
    leaked["query_bank"]["query_split"] = "dev"
    with pytest.raises(ValueError, match="reserved audit query-bank authority differs"):
        confirmation.validate_authority(leaked)
    leaked = copy.deepcopy(_authority())
    leaked["query_bank"]["queries"] = 101
    with pytest.raises(ValueError, match="reserved audit query-bank authority differs"):
        confirmation.validate_authority(leaked)


def test_confirmation_response_is_descriptor_first_and_exact_90() -> None:
    descriptor = torch.zeros(1536, 1, 1)
    descriptor[0, 0, 0] = 0.5
    descriptor[1, 0, 0] = 0.5
    text = torch.zeros(90, 1536)
    text[:, 2] = 1.0
    text[0].zero_()
    text[0, 0] = 1.0
    response = confirmation.descriptor_map_to_text_responses(descriptor, text)
    assert response.shape == (90, 1, 1)
    assert response[0, 0, 0] == pytest.approx(2**-0.5)
    with pytest.raises(ValueError, match="reserved audit text embedding"):
        confirmation.descriptor_map_to_text_responses(descriptor, text[:89])


def test_confirmation_requires_frozen_base_implementation_record(monkeypatch) -> None:
    _allow_confirmation_records(monkeypatch)
    changed = copy.deepcopy(_authority())
    changed["base_materializer"] = _record("0")
    with pytest.raises(ValueError, match="schema differs"):
        confirmation.validate_authority(changed)
