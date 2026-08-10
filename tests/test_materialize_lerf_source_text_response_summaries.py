import copy

import pytest
import torch

import radio_gs.scripts.materialize_lerf_source_text_response_summaries as materializer


def _record(seed: str) -> dict[str, str]:
    return {"path": f"/tmp/{seed}", "sha256": seed[0] * 64}


def _authority() -> dict[str, object]:
    query = {
        "path": "/tmp/query.pt",
        "sha256": "1" * 64,
        "manifest_path": "/tmp/query.json",
        "manifest_sha256": "2" * 64,
        "query_split": "dev",
        "queries": 101,
        "embedding_tensor_sha256": "3" * 64,
    }
    return {
        "schema": materializer.AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_source_only_compact_summary",
        "implementation": materializer.IMPLEMENTATION,
        "frame_evaluator_implementation": materializer.FRAME_EVALUATOR_IMPLEMENTATION,
        "scene_id": "ramen",
        "source_heldout_frame_ids": [2, 45],
        "forbidden_target_frame_ids": [6, 24],
        "inputs": {
            "source_gate_preregistration": _record("a"),
            "source_view_preregistration": _record("b"),
            "scene_config": _record("c"),
            "geometry_checkpoint": _record("d"),
            "query_bank_artifact": {"path": query["path"], "sha256": query["sha256"]},
            "query_bank_manifest": {
                "path": query["manifest_path"], "sha256": query["manifest_sha256"]
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
                "descriptor_payload_kind": materializer.SPARSE_TEACHER_MEAN_KIND,
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
                "descriptor_payload_kind": materializer.DENSE_OFFICIAL_CROP_SUMMARY_MPR_KIND,
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
        "query_bank": query,
        "geometry": {"num_gaussians": 5, "xyz_sha256": "4" * 64},
        "execution": {
            "required_cuda_visible_devices": "0",
            "program_device": "cuda:0",
            "thermal_guard": materializer.THERMAL_GUARD,
            "thermal_poll_seconds": 300,
            "maximum_temperature_c": 88,
            "soft_pause_temperature_c": 0,
        },
        "outputs": {
            "control_summary": "/tmp/control.json",
            "candidate_summary": "/tmp/candidate.json",
            "result": "/tmp/result.json",
        },
        "access_audit": {
            "benchmark_queries_opened": False,
            "benchmark_masks_or_labels_opened": False,
            "target_metric_execution_authorized": False,
        },
    }


def test_authority_separates_sparse_control_and_genuine_mpr_candidate() -> None:
    authority = _authority()
    assert materializer.validate_authority(authority)["equivalence_smoke"] is False
    leaked = copy.deepcopy(authority)
    leaked["access_audit"]["benchmark_queries_opened"] = True
    with pytest.raises(ValueError, match="not target blind"):
        materializer.validate_authority(leaked)


def test_primitive_descriptor_placement_zeros_invalid_and_missing_rows() -> None:
    descriptors = torch.zeros(3, 1536, dtype=torch.float16)
    descriptors[0, 0] = 1
    descriptors[1, 1] = 1
    descriptors[2, 2] = 1
    output = materializer.build_primitive_descriptor_rows(
        {
            "global_rows": torch.tensor([0, 2, 4]),
            "teacher_mean": descriptors,
            "teacher_valid": torch.tensor([True, False, True]),
        },
        num_gaussians=5,
        device=torch.device("cpu"),
        chunk_rows=2,
    )
    assert output.shape == (5, 1536)
    assert output[0, 0] == 1
    assert output[4, 2] == 1
    assert torch.count_nonzero(output[1:4]) == 0


def test_descriptor_first_response_differs_from_response_first_mixture() -> None:
    descriptor_map = torch.zeros(1536, 1, 1)
    descriptor_map[0, 0, 0] = 0.5
    descriptor_map[1, 0, 0] = 0.5
    text = torch.zeros(101, 1536)
    text[:, 2] = 1.0
    text[0].zero_()
    text[0, 0] = 1.0

    descriptor_first = materializer.descriptor_map_to_text_responses(
        descriptor_map, text
    )
    primitive_response_first = 0.5  # 0.5 * <e0,e0> + 0.5 * <e1,e0>
    assert descriptor_first[0, 0, 0] == pytest.approx(2**-0.5)
    assert descriptor_first[0, 0, 0] != pytest.approx(primitive_response_first)
    expected = torch.einsum(
        "qd,dhw->qhw",
        torch.nn.functional.normalize(text, dim=-1),
        torch.nn.functional.normalize(descriptor_map, dim=0),
    )
    assert torch.equal(descriptor_first, expected)


def test_source_reseal_content_hash_is_mandatory_and_checked(
    tmp_path, monkeypatch
) -> None:
    reseal = tmp_path / "reseal.json"
    reseal.write_text("{}\n", encoding="utf-8")
    authority = _authority()
    authority["inputs"]["source_reseal"] = {
        "path": str(reseal),
        "sha256": "0" * 64,
        "required_schema": "radio_gs.lerf_official_crop_summary_reseal.v1",
        "required_mode": "content_addressed_immutable_reseal",
    }
    with pytest.raises(ValueError, match="SHA-256 differs"):
        materializer._load_source_reseal(
            authority, authority["inputs"]["source_view_preregistration"]
        )
