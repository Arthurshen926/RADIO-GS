from argparse import Namespace
import json

from radio_gs.v4.evaluation.geometry_gate import run


def test_geometry_gate_blocks_downstream_until_full_cohort(tmp_path):
    oracle = {
        "comparison_to_gaussian": {
            name: {
                "roundtrip_delta": 0.2,
                "transfer_delta": 0.1,
                "boundary_leakage_reduction": 0.1,
                "purity_delta": 0.01,
                "coverage_delta": -0.05,
                "effective_contributors_reduction": 5.0,
            }
            for name in ("mesh_oracle", "mesh_derived_sparse_surface")
        }
    }
    source = {
        "comparison_to_gaussian": {
            name: {
                "roundtrip_delta": 0.1,
                "transfer_delta": -0.01,
                "same_view_leakage_reduction": 0.1,
                "mutually_exclusive_purity_delta": 0.01,
                "coverage_delta": -0.01,
                "effective_contributors_reduction": 4.0,
            }
            for name in ("mesh_surface", "mesh_derived_sparse_surface")
        }
    }
    oracle_path, source_path = tmp_path / "oracle.json", tmp_path / "source.json"
    oracle_path.write_text(json.dumps(oracle))
    source_path.write_text(json.dumps(source))
    report = run(Namespace(
        oracle_report=oracle_path,
        source_report=source_path,
        cohort_key="synthetic",
        expected_scene_count=3,
        minimum_delta=0.0,
        output=tmp_path / "gate.json",
    ))
    assert report["scene_gate"]["sparse_surface"]["passes_scene_gate"]
    assert not report["milestone_1_complete"]
    assert not report["object_codebook_authorized"]


def test_geometry_gate_authorizes_only_oracle_codebook_after_full_cohort(tmp_path):
    comparison = {
        "roundtrip_delta": 0.2,
        "transfer_delta": 0.1,
        "boundary_leakage_reduction": 0.1,
        "purity_delta": 0.01,
        "coverage_delta": -0.05,
        "effective_contributors_reduction": 5.0,
    }
    oracle = {"comparison_to_gaussian": {
        "mesh_oracle": comparison,
        "mesh_derived_sparse_surface": comparison,
    }}
    source_comparison = {
        "roundtrip_delta": 0.1,
        "transfer_delta": -0.01,
        "same_view_leakage_reduction": 0.1,
        "mutually_exclusive_purity_delta": 0.01,
        "coverage_delta": -0.01,
        "effective_contributors_reduction": 4.0,
    }
    source = {"comparison_to_gaussian": {
        "mesh_surface": source_comparison,
        "mesh_derived_sparse_surface": source_comparison,
    }}
    oracle_path, source_path = tmp_path / "oracle.json", tmp_path / "source.json"
    oracle_path.write_text(json.dumps(oracle)); source_path.write_text(json.dumps(source))
    lerf_paths = []
    for label in ("first", "second"):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps({
            "schema": "radio_gs.surface_object_memory_v4.lerf_source_mask_geometry_gate.v1",
            "scene_label": label,
            "passes_scene_gate": True,
            "primary_directions": {"roundtrip": True, "leakage": True, "purity": True},
            "comparison_to_gaussian": {"roundtrip_delta": 0.1},
            "coverage_is_reported_not_compensatory": True,
            "projection_configuration": {"maximum_splat_radius": 1},
        }))
        lerf_paths.append(path)
    report = run(Namespace(
        oracle_report=oracle_path, source_report=source_path, lerf_report=lerf_paths,
        cohort_key="synthetic", expected_scene_count=3, minimum_delta=0.0,
        output=tmp_path / "gate.json",
    ))
    assert report["milestone_1_complete"]
    assert report["object_codebook_authorized"]
    assert report["object_codebook_authorized_scope"] == "oracle_only"
    assert not report["query_encoder_authorized"]
