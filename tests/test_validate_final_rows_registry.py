import importlib
import json
from pathlib import Path

import pytest
import yaml

VALA8_SCENES = [
    "scene0000_00",
    "scene0062_00",
    "scene0070_00",
    "scene0097_00",
    "scene0140_00",
    "scene0347_00",
    "scene0400_00",
    "scene0590_00",
]
PROMOTED_SOURCE_ARGS = {
    "scene_list": ",".join(VALA8_SCENES),
    "class_splits": "19,15,10",
    "query_mode": "knn",
    "k": "16",
    "candidate_k": "80",
    "opacity_filter_mode": "auto",
    "logit_calibration": "scene_mean",
    "logit_calibration_alpha": "0.45",
    "logit_smoothing": "spatial_knn",
    "logit_smoothing_k": "12",
    "logit_smoothing_alpha": "1.0",
    "logit_smoothing_iterations": "1",
    "prompt_templates": "{query}",
    "use_summary_head": "True",
    "use_point_summary_adapter": "False",
}


def _load_validator():
    try:
        return importlib.import_module("radio_gs.scripts.validate_final_rows_registry")
    except ImportError as exc:
        pytest.fail(f"missing validate_final_rows_registry module: {exc}")


def _rows_from_macro(macro: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    return [
        {
            "scene": scene,
            "19": dict(macro["19"]),
            "15": dict(macro["15"]),
            "10": dict(macro["10"]),
        }
        for scene in VALA8_SCENES
    ]


def _write_fixture(root: Path, *, split10_miou: float = 0.4711) -> Path:
    lerf2d_json = root / "paper/artifacts/lerf2d.json"
    lerf2d_json.parent.mkdir(parents=True)
    lerf2d_macro = {"miou": 0.588944466161357, "loc_acc": 0.8598309812650944}
    lerf2d_scenes = {
        scene: {"miou": lerf2d_macro["miou"], "loc_acc": lerf2d_macro["loc_acc"]}
        for scene in ("figurines", "ramen", "teatime", "waldo_kitchen")
    }
    lerf2d_json.write_text(
        json.dumps({"macro": lerf2d_macro, "scenes": lerf2d_scenes}),
        encoding="utf-8",
    )
    direct3d_json = root / "paper/artifacts/direct3d.json"
    direct3d_source_row = {
        "miou": 0.501373864710331,
        "acc025": 0.7044309383271872,
        "per_scene": {
            scene: {"miou": 0.501373864710331, "acc025": 0.7044309383271872}
            for scene in ("figurines", "ramen", "teatime", "waldo_kitchen")
        },
    }
    direct3d_json.write_text(
        json.dumps({"rows": {"promoted": direct3d_source_row}}),
        encoding="utf-8",
    )
    scannet_json = root / "output/scannet_pointcloud_eval/support.json"
    scannet_json.parent.mkdir(parents=True)
    support_macro = {
        "19": {"miou": 0.3806, "macc": 0.6129},
        "15": {"miou": 0.3871, "macc": 0.6315},
        "10": {"miou": 0.4711, "macc": 0.7200},
    }
    scannet_json.write_text(
        json.dumps(
            {
                "scene_count": 8,
                "scenes": VALA8_SCENES,
                "source_args": PROMOTED_SOURCE_ARGS,
                "macro": support_macro,
                "rows": _rows_from_macro(support_macro),
            }
        ),
        encoding="utf-8",
    )
    audit_json = root / "paper/artifacts/external_baseline_audit.json"
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(
        json.dumps(
            {
                "baselines": [
                    {
                        "method": "Unpublished protocol source",
                        "exists": False,
                        "blocker": "code will be publicly released upon acceptance",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    final_rows = root / "paper/artifacts/final_rows.yaml"
    final_rows.write_text(
        yaml.safe_dump(
            {
                "external_reproduction_queue": {
                    "machine_audit": {
                        "json": "paper/artifacts/external_baseline_audit.json",
                    },
                    "p0": [
                        {
                            "method": "GAGS",
                            "status": (
                                "all four LERF compatibility scenes completed. "
                                "scene-mean LocAcc 0.5000 / mIoU 0.6000 and "
                                "object-weighted LocAcc 0.5500 / mIoU 0.6500 over 10 queries"
                            ),
                        },
                        {
                            "method": "Dr. Splat",
                            "status": (
                                "all four LERF compatibility scenes completed. "
                                "mIoU 0.3100 / Acc@0.25 0.4500 / Acc@0.5 0.2000 "
                                "over 8 objects; missing rendered masks counted: 1"
                            ),
                        },
                    ],
                    "p1": [
                        {
                            "method": "LEGaussians",
                            "status": (
                                "all four LERF compatibility scenes completed. "
                                "scene-mean mIoU 0.2100 / Acc@0.25 0.3000 / "
                                "Acc@0.5 0.1200 and object-weighted mIoU 0.2200 "
                                "over 9 objects; missing rendered masks counted: 0"
                            ),
                        },
                        {
                            "method": "Semantic Gaussians",
                            "status": (
                                "all four ScanNet compatibility scenes completed. "
                                "ScanNet-20 label-PLY mean IoU 0.0280 over 4 scenes"
                            ),
                        },
                        {
                            "method": "LaGa",
                            "status": (
                                "all four LERF compatibility scenes completed. "
                                "mIoU 0.3300 / Acc@0.25 0.4000 / Acc@0.5 0.2500 "
                                "over 7 objects; missing rendered masks counted: 0"
                            ),
                        },
                    ],
                    "p2": [
                        {
                            "method": "Unpublished protocol source",
                            "status": "no public implementation was found",
                        }
                    ],
                },
                "tracks": {
                    "t1_lerf_rendered_view_ovs": {
                        "source_json": "paper/artifacts/lerf2d.json",
                        "rows": {
                            "ctfgs_rendered": {
                                "miou": lerf2d_macro["miou"],
                                "locacc": lerf2d_macro["loc_acc"],
                                "per_scene": {
                                    scene: {
                                        "miou": values["miou"],
                                        "acc": values["loc_acc"],
                                    }
                                    for scene, values in lerf2d_scenes.items()
                                },
                                "promoted": True,
                            }
                        },
                    },
                    "t2_lerf_direct_3d_selection": {
                        "source_json": "paper/artifacts/direct3d.json",
                        "rows": {
                            "ctfgs_compact_prompt_ensemble_score_component_guard_thr0p55": {
                                "source_row": "promoted",
                                "miou": direct3d_source_row["miou"],
                                "acc025": direct3d_source_row["acc025"],
                                "per_scene": direct3d_source_row["per_scene"],
                                "promoted": True,
                            }
                        },
                    },
                    "t3_scannet_ov_point_cloud_segmentation": {
                        "radio_gs_source_json": (
                            "output/scannet_pointcloud_eval/support.json"
                        ),
                        "rows": {
                            "radio_gs_dino_cv_contextual_knn_scene_mean_support": {
                                "split19": {"miou": 0.3806, "macc": 0.6129},
                                "split15": {"miou": 0.3871, "macc": 0.6315},
                                "split10": {"miou": split10_miou, "macc": 0.7200},
                                "per_scene": {
                                    scene: {
                                        "split19": dict(support_macro["19"]),
                                        "split15": dict(support_macro["15"]),
                                        "split10": dict(support_macro["10"]),
                                    }
                                    for scene in VALA8_SCENES
                                },
                                "promoted": True,
                            }
                        },
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    scannet_table = root / "paper/scannet_published_context_table.tex"
    scannet_table.parent.mkdir(parents=True, exist_ok=True)
    scannet_table.write_text(
        "\\method{} & 38.06 & 61.29 & 38.71 & 63.15 & "
        f"{100.0 * split10_miou:.2f} & 72.00 \\\\\n",
        encoding="utf-8",
    )
    return final_rows


def _write_external_summaries(root: Path, *, gags_miou: float = 0.6) -> None:
    gags_summary = root / "paper/artifacts/gags_lerf_summary.json"
    gags_summary.write_text(
        json.dumps(
            {
                "completed_rows": [{"scene": scene} for scene in ("figurines", "ramen", "teatime", "waldo_kitchen")],
                "scene_mean": {"locacc": 0.5, "miou": gags_miou},
                "object_weighted": {"query_count": 10, "locacc": 0.55, "miou": 0.65},
            }
        ),
        encoding="utf-8",
    )
    drsplat_summary = root / "paper/artifacts/drsplat_lerf_summary.json"
    drsplat_summary.write_text(
        json.dumps(
            {
                "scenes": {"figurines": {}, "ramen": {}, "teatime": {}, "waldo_kitchen": {}},
                "macro": {"miou": 0.31, "acc025": 0.45, "acc05": 0.2, "count": 8, "missing": 1},
            }
        ),
        encoding="utf-8",
    )
    legaussians_summary = root / "paper/artifacts/legaussians_lerf_summary.json"
    legaussians_summary.write_text(
        json.dumps(
            {
                "scenes": {"figurines": {}, "ramen": {}, "teatime": {}, "waldo_kitchen": {}},
                "scene_mean": {"miou": 0.21, "acc025": 0.3, "acc05": 0.12},
                "object_weighted": {"miou": 0.22, "count": 9, "missing": 0},
            }
        ),
        encoding="utf-8",
    )
    semantic_summary = root / "output/baselines/semantic_gaussians/scannet_compat_20260520/semantic_gaussians_eval_metrics.json"
    semantic_summary.parent.mkdir(parents=True, exist_ok=True)
    semantic_summary.write_text(
        json.dumps(
            {
                "metrics": {"mean_iou": 0.028},
                "scenes": {"scene0000_00": {}, "scene0062_00": {}, "scene0070_00": {}, "scene0097_00": {}},
            }
        ),
        encoding="utf-8",
    )
    laga_summary = root / "paper/artifacts/laga_lerf_summary.json"
    laga_summary.write_text(
        json.dumps(
            {
                "scenes": {"figurines": {}, "ramen": {}, "teatime": {}, "waldo_kitchen": {}},
                "macro": {"miou": 0.33, "acc025": 0.4, "acc05": 0.25, "count": 7, "missing": 0},
            }
        ),
        encoding="utf-8",
    )


def test_validate_registry_accepts_scannet_support_and_opengaff_blocker(tmp_path):
    validator = _load_validator()
    final_rows = _write_fixture(tmp_path)

    issues = validator.validate_registry(final_rows, root=tmp_path)

    assert issues == []


def test_validate_registry_flags_scannet_metric_drift(tmp_path):
    validator = _load_validator()
    final_rows = _write_fixture(tmp_path, split10_miou=0.45)

    issues = validator.validate_registry(final_rows, root=tmp_path)

    assert any("split10.miou" in issue for issue in issues)


def test_validate_registry_flags_scannet_promoted_metric_drift(tmp_path):
    validator = _load_validator()
    final_rows = _write_fixture(tmp_path)
    payload = yaml.safe_load(final_rows.read_text(encoding="utf-8"))
    payload["tracks"]["t3_scannet_ov_point_cloud_segmentation"]["rows"][
        "radio_gs_dino_cv_contextual_knn_scene_mean_support"
    ]["split19"]["miou"] = 0.35
    final_rows.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    issues = validator.validate_registry(final_rows, root=tmp_path)

    assert any("promoted split19.miou" in issue for issue in issues)


def test_validate_registry_flags_scannet_protocol_drift(tmp_path):
    validator = _load_validator()
    final_rows = _write_fixture(tmp_path)
    support_json = tmp_path / "output/scannet_pointcloud_eval/support.json"
    payload = json.loads(support_json.read_text(encoding="utf-8"))
    payload["source_args"]["query_mode"] = "nearest"
    support_json.write_text(json.dumps(payload), encoding="utf-8")

    issues = validator.validate_registry(final_rows, root=tmp_path)

    assert any("ScanNet promoted source_args.query_mode" in issue for issue in issues)


def test_validate_registry_rejects_scannet_source_without_eight_rows(tmp_path):
    validator = _load_validator()
    final_rows = _write_fixture(tmp_path)
    support_json = tmp_path / "output/scannet_pointcloud_eval/support.json"
    payload = json.loads(support_json.read_text(encoding="utf-8"))
    payload["rows"] = []
    support_json.write_text(json.dumps(payload), encoding="utf-8")

    issues = validator.validate_registry(final_rows, root=tmp_path)

    assert any("exactly 8 per-scene rows" in issue for issue in issues)


def test_validate_registry_accepts_synced_external_summary_statuses(tmp_path):
    validator = _load_validator()
    final_rows = _write_fixture(tmp_path)
    _write_external_summaries(tmp_path)

    issues = validator.validate_registry(final_rows, root=tmp_path)

    assert issues == []


def test_validate_registry_flags_external_summary_status_drift(tmp_path):
    validator = _load_validator()
    final_rows = _write_fixture(tmp_path)
    _write_external_summaries(tmp_path, gags_miou=0.61)

    issues = validator.validate_registry(final_rows, root=tmp_path)

    assert any("GAGS" in issue and "scene-mean" in issue for issue in issues)
