import json

import pytest

from radio_gs.scripts.build_full_benchmark_final_table import build_table


def test_build_table_preserves_scope_and_diagnostic_status(tmp_path) -> None:
    pfpr = {
        "benchmark": "scannet-pfpr-small-v2",
        "protocol": {"scene_count": 20, "query_count": 200},
        "metrics_query_micro": {
            "top1_mean_error_m": 0.2,
            "top1_median_error_m": 0.1,
            "R@1_5cm": 0.1,
            "R@5_5cm": 0.2,
            "R@10_5cm": 0.3,
            "MRR_5cm": 0.15,
            "R@1_10cm": 0.4,
            "R@5_10cm": 0.5,
            "R@10_10cm": 0.6,
        },
    }
    agile = {
        "benchmark": "AGILE3D ScanNet40 single-object",
        "protocol": {
            "result_status": "diagnostic_only",
            "scenes": 312,
            "objects": 10357,
            "observation_contract": "scannet_full_observation_diagnostic_v1",
            "canonical_mpr_contract": "canonical-full-observation-mpr-v1",
            "formal_comparable": False,
            "world_query": "compile_world_3d_query",
            "observation_lift": "none",
            "official_point_readout": "continuous_opacity_weighted_gaussian",
            "background_centroids": 4,
        },
        "metrics": {
            "IoU@1": 0.1,
            "IoU@5": 0.2,
            "IoU@10": 0.3,
            "IoU@15": 0.4,
            "NoC@50": 5.0,
            "NoC@65": 6.0,
            "NoC@80": 7.0,
            "NoC@90": 8.0,
        },
    }
    pfpr_path = tmp_path / "pfpr.json"
    agile_path = tmp_path / "agile.json"
    pfpr_path.write_text(json.dumps(pfpr), encoding="utf-8")
    agile_path.write_text(json.dumps(agile), encoding="utf-8")

    report = build_table(pfpr_path, agile_path)

    assert "| 20 | 200 |" in report
    assert "| diagnostic_only | 312 | 10357 |" in report
    assert "query-free scene background modes: `4`" in report
    assert "formal comparable: `False`" in report


def test_build_table_rejects_partial_pfpr_result(tmp_path) -> None:
    pfpr = {
        "benchmark": "scannet-pfpr-small-v2",
        "protocol": {"scene_count": 1, "query_count": 10},
        "metrics_query_micro": {},
    }
    agile = {
        "benchmark": "AGILE3D ScanNet40 single-object",
        "protocol": {
            "scenes": 312,
            "objects": 10357,
            "world_query": "compile_world_3d_query",
            "observation_lift": "none",
            "official_point_readout": "continuous_opacity_weighted_gaussian",
            "background_centroids": 4,
        },
        "metrics": {},
    }
    pfpr_path = tmp_path / "pfpr.json"
    agile_path = tmp_path / "agile.json"
    pfpr_path.write_text(json.dumps(pfpr), encoding="utf-8")
    agile_path.write_text(json.dumps(agile), encoding="utf-8")

    with pytest.raises(ValueError, match="20 scenes and 200 queries"):
        build_table(pfpr_path, agile_path)


def test_build_table_rejects_old_observation_bridge_or_background_mean(tmp_path) -> None:
    pfpr = {
        "benchmark": "scannet-pfpr-small-v2",
        "protocol": {"scene_count": 20, "query_count": 200},
        "metrics_query_micro": {},
    }
    agile = {
        "benchmark": "AGILE3D ScanNet40 single-object",
        "protocol": {
            "scenes": 312,
            "objects": 10357,
            "world_query": "compile_world_3d_query",
            "observation_lift": "observed_domain",
            "official_point_readout": "continuous_opacity_weighted_gaussian",
            "background_centroids": 0,
        },
        "metrics": {},
    }
    pfpr_path = tmp_path / "pfpr.json"
    agile_path = tmp_path / "agile.json"
    pfpr_path.write_text(json.dumps(pfpr), encoding="utf-8")
    agile_path.write_text(json.dumps(agile), encoding="utf-8")

    with pytest.raises(ValueError, match="observation_lift"):
        build_table(pfpr_path, agile_path)
