import json
from pathlib import Path

from radio_gs.scripts import build_submission_freeze_report as report


def test_collect_scannet_v67_results_computes_macro(tmp_path: Path) -> None:
    root = tmp_path / "scannet"
    for scene, vals in {
        "scene0000_00": (0.2, 0.3, 0.4),
        "scene0062_00": (0.4, 0.5, 0.6),
    }.items():
        out = root / f"{scene}_v67_teacherbalanced_fromv63_best_gidx_labelpoint"
        out.mkdir(parents=True)
        (out / "scannet_pointcloud_radio_gs_results.json").write_text(
            json.dumps(
                {
                    "macro": {
                        "19": {"miou": vals[0], "macc": 0.7},
                        "15": {"miou": vals[1], "macc": 0.8},
                        "10": {"miou": vals[2], "macc": 0.9},
                    },
                    "args": {
                        "config": f"configs/{scene}.yaml",
                        "checkpoint": f"checkpoints/{scene}.pth",
                        "text_embedding_cache": "checkpoints/siglip2_scannet.pt",
                        "radio_checkpoint": "checkpoints/c-radio_v4-h.pth.tar",
                        "sample_seed": "42",
                        "query_mode": "gaussian_index",
                        "opacity_filter_mode": "label_index",
                        "gaussian_index_position_mode": "label_point",
                    },
                }
            )
        )

    summary = report.collect_scannet_v67(root)

    assert summary["scene_count"] == 2
    assert summary["macro_miou"] == {"19": 0.3, "15": 0.4, "10": 0.5}
    assert summary["rows"][0]["config"].endswith("scene0000_00.yaml")
    assert summary["rows"][0]["checkpoint"].endswith("scene0000_00.pth")
    assert summary["rows"][0]["text_embedding_cache"] == "checkpoints/siglip2_scannet.pt"
    assert summary["rows"][0]["teacher_model"] == "checkpoints/c-radio_v4-h.pth.tar"
    assert summary["rows"][0]["selector_policy"] == "v67_teacherbalanced_gaussian_index_labelpoint"
    assert summary["rows"][0]["evaluator"] == "eval_scannet_pointcloud_radio_gs"
    assert not summary["warnings"]


def test_collect_scannet_v67_can_filter_vala8_subset(tmp_path: Path) -> None:
    root = tmp_path / "scannet"
    for scene, vals in {
        "scene0000_00": (0.2, 0.3, 0.4),
        "scene0062_00": (0.4, 0.5, 0.6),
        "scene0200_00": (0.9, 0.9, 0.9),
    }.items():
        out = root / f"{scene}_v67_teacherbalanced_fromv63_best_gidx_labelpoint"
        out.mkdir(parents=True)
        (out / "scannet_pointcloud_radio_gs_results.json").write_text(
            json.dumps(
                {
                    "macro": {
                        "19": {"miou": vals[0], "macc": 0.7},
                        "15": {"miou": vals[1], "macc": 0.8},
                        "10": {"miou": vals[2], "macc": 0.9},
                    },
                    "args": {
                        "query_mode": "gaussian_index",
                        "opacity_filter_mode": "label_index",
                        "gaussian_index_position_mode": "label_point",
                    },
                }
            )
        )

    summary = report.collect_scannet_v67(
        root,
        scene_subset=("scene0000_00", "scene0062_00"),
    )

    assert summary["scene_count"] == 2
    assert summary["scene_subset"] == ["scene0000_00", "scene0062_00"]
    assert summary["macro_miou"] == {"19": 0.3, "15": 0.4, "10": 0.5}


def test_collect_lerf_best_row_reads_macro(tmp_path: Path) -> None:
    csv_path = tmp_path / "current_best_lerf_ovs_per_scene.csv"
    csv_path.write_text(
        "scene,loc_acc,miou,temp,checkpoint,config,output_dir,summary\n"
        "figurines,0.8,0.4,50,ckpt,cfg,out,sum\n"
        "ramen,0.9,0.6,40,ckpt,cfg,out,sum\n"
        "macro,0.85,0.5,,,,,\n"
    )

    summary = report.collect_lerf_best(csv_path)

    assert summary["macro_loc_acc"] == 0.85
    assert summary["macro_miou"] == 0.5
    assert len(summary["rows"]) == 2


def test_collect_lerf_threshold_sweep_reads_calibrated_variant(tmp_path: Path) -> None:
    sweep_path = tmp_path / "threshold_sweep.json"
    sweep_path.write_text(
        json.dumps(
            {
                "variants": {
                    "0.60": {
                        "rows": [
                            {
                                "scene": "figurines",
                                "loc": 0.8,
                                "miou": 0.42,
                                "temp": 50.0,
                                "n": 4,
                            },
                            {
                                "scene": "ramen",
                                "loc": 0.9,
                                "miou": 0.62,
                                "temp": 40.0,
                                "n": 5,
                            },
                        ],
                        "macro": {"loc": 0.85, "miou": 0.52},
                        "weighted": {"loc": 0.8556, "miou": 0.5311},
                    }
                }
            }
        )
    )

    summary = report.collect_lerf_threshold_sweep(sweep_path, "0.60")

    assert summary["macro_loc_acc"] == 0.85
    assert summary["macro_miou"] == 0.52
    assert summary["weighted_miou"] == 0.5311
    assert summary["rows"][0]["scene"] == "figurines"
    assert summary["rows"][0]["miou"] == 0.42
    assert summary["readout"] == "threshold 0.60"
    assert summary["selector_policy"] == "fixed_threshold:0.60"
    assert summary["text_head"] == "SigLIP2"
    assert summary["evaluator"] == "eval_lerf_grounding"


def test_collect_direct3d_silhouette_sweep_reads_variant(tmp_path: Path) -> None:
    sweep_path = tmp_path / "direct_sweep.json"
    sweep_path.write_text(
        json.dumps(
            {
                "variants": {
                    "0.60": {
                        "rows": [
                            {"scene": "figurines", "miou": 0.54, "acc025": 0.78, "acc050": 0.64, "n": 4},
                            {"scene": "ramen", "miou": 0.47, "acc025": 0.74, "acc050": 0.49, "n": 5},
                        ],
                        "macro": {"miou": 0.505, "acc025": 0.76, "acc050": 0.565},
                        "weighted": {"miou": 0.5011, "acc025": 0.7578, "acc050": 0.5567},
                    }
                }
            }
        )
    )

    summary = report.collect_direct3d_silhouette_sweep(sweep_path, "0.60")

    assert summary["macro_miou"] == 0.505
    assert summary["macro_acc025"] == 0.76
    assert summary["weighted_miou"] == 0.5011
    assert summary["rows"][1]["scene"] == "ramen"


def _write_direct3d_scene_result(
    root: Path,
    scene: str,
    tag: str,
    *,
    miou: float,
    acc025: float,
    boundary_f: float,
    trimap_iou: float,
) -> None:
    result_dir = root / scene / scene
    result_dir.mkdir(parents=True)
    (result_dir / "lerf_direct_3d_selection_results.json").write_text(
        json.dumps(
            {
                "args": {
                    "config": f"configs/{scene}.yaml",
                    "checkpoint": f"checkpoints/{scene}.pth",
                    "text_embedding_cache": "checkpoints/siglip2_lerf.pt",
                    "score_cache": f"score_caches/{scene}.pt",
                    "scoring": "softmax_scene",
                    "mask_refinement": "sam3_box",
                },
                "protocol": {
                    "score_source": "direct",
                    "feature_source": "pre-refiner compact features",
                    "mask_refinement": "sam3_box",
                    "sam3_box_padding": 0,
                    "selection_min_ratio": 0.003,
                    "selection_max_ratio": 0.06,
                },
                "scene": {
                    "scene": scene,
                    "results": {
                        tag: {
                            "miou": miou,
                            "acc025": acc025,
                            "acc050": acc025 - 0.1,
                            "n": 5,
                            "boundary_f": boundary_f,
                            "trimap_iou": trimap_iou,
                            "selection_tag": tag,
                            "selection_value": 0.25,
                            "selection_min_ratio": 0.003,
                            "selection_max_ratio": 0.06,
                            "mask_refinement": "sam3_box",
                        }
                    },
                    "best_by_miou": tag,
                },
            }
        )
    )


def test_collect_direct3d_scene_readout_reads_nested_sam3_results(tmp_path: Path) -> None:
    root = tmp_path / "sam3_pad0"
    values = {
        "figurines": ("thr0p11", 0.5924, 0.7321, 0.6813, 0.3523),
        "ramen": ("thr0p16", 0.6830, 0.8028, 0.7770, 0.4210),
        "teatime": ("thr0p38", 0.6556, 0.7797, 0.7730, 0.4800),
        "waldo_kitchen": ("thr0p3", 0.3949, 0.5455, 0.4613, 0.2552),
    }
    for scene, (tag, miou, acc025, boundary_f, trimap_iou) in values.items():
        _write_direct3d_scene_result(
            root,
            scene,
            tag,
            miou=miou,
            acc025=acc025,
            boundary_f=boundary_f,
            trimap_iou=trimap_iou,
        )

    summary = report.collect_direct3d_scene_readout(
        root,
        label="direct field + official SAM3 box, pad0",
        text_head="SigLIP2+SAM3",
        protocol_label="compact direct field + official SAM3 box readout",
    )

    assert summary["scene_count"] == 4
    assert summary["macro_miou"] == 0.5815
    assert summary["macro_acc025"] == 0.715
    assert summary["macro_boundary_f"] == 0.6732
    assert summary["macro_trimap_iou"] == 0.3771
    assert summary["rows"][0]["scene"] == "figurines"
    assert summary["rows"][0]["selection"] == "thr0p11"
    assert summary["rows"][0]["source"].endswith(
        "sam3_pad0/figurines/figurines/lerf_direct_3d_selection_results.json"
    )
    assert summary["rows"][0]["config"] == "configs/figurines.yaml"
    assert summary["rows"][0]["checkpoint"] == "checkpoints/figurines.pth"
    assert summary["rows"][0]["text_embedding_cache"] == "checkpoints/siglip2_lerf.pt"
    assert summary["rows"][0]["score_cache"] == "score_caches/figurines.pt"
    assert summary["rows"][0]["selector_policy"] == "best_by_miou"
    assert summary["rows"][0]["text_head"] == "SigLIP2+SAM3"
    assert summary["rows"][0]["evaluator"] == "eval_lerf_direct_3d_selection"
    assert summary["protocol"]["mask_refinement"] == "sam3_box"
    assert summary["selector_policy"] == "best_by_miou"


def test_write_report_outputs_markdown_and_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"
    lerf = {"macro_loc_acc": 0.85, "macro_miou": 0.5, "rows": [], "warnings": []}
    scannet = {
        "scene_count": 1,
        "macro_miou": {"19": 0.3, "15": 0.4, "10": 0.5},
        "macro_macc": {"19": 0.6, "15": 0.7, "10": 0.8},
        "rows": [
            {
                "scene": "scene0000_00",
                "path": "result.json",
                "miou": {"19": 0.3, "15": 0.4, "10": 0.5},
                "macc": {"19": 0.6, "15": 0.7, "10": 0.8},
            }
        ],
        "warnings": ["external baselines unresolved"],
    }

    paths = report.write_freeze_outputs(output_dir, lerf, scannet)

    markdown = paths["markdown"].read_text()
    manifest = json.loads(paths["manifest"].read_text())
    assert "Submission Freeze Report" in markdown
    assert "LERF-OVS" in markdown
    assert "ScanNet" in markdown
    assert manifest["lerf"]["macro_loc_acc"] == 0.85
    assert manifest["scannet"]["macro_miou"]["10"] == 0.5


def test_write_report_includes_calibrated_lerf_and_direct3d(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"
    lerf = {
        "macro_loc_acc": 0.8712,
        "macro_miou": 0.5243,
        "weighted_miou": 0.5397,
        "readout": "threshold 0.60",
        "source": "threshold_sweep.json",
        "rows": [
            {"scene": "figurines", "loc_acc": 0.8214, "miou": 0.4244, "temp": 50.0, "summary": "threshold_sweep.json"}
        ],
        "warnings": [],
    }
    direct3d = {
        "silhouette": "0.60",
        "macro_miou": 0.4554,
        "macro_acc025": 0.7014,
        "macro_acc050": 0.4663,
        "weighted_miou": 0.4932,
        "rows": [],
        "source": "direct_sweep.json",
        "warnings": [],
    }
    scannet = {
        "scene_count": 0,
        "macro_miou": {"19": 0.0, "15": 0.0, "10": 0.0},
        "macro_macc": {"19": 0.0, "15": 0.0, "10": 0.0},
        "rows": [],
        "warnings": [],
    }

    paths = report.write_freeze_outputs(output_dir, lerf, scannet, direct3d=direct3d)

    markdown = paths["markdown"].read_text()
    manifest = json.loads(paths["manifest"].read_text())
    assert "threshold 0.60" in markdown
    assert "0.5243" in markdown
    assert "RGB snap silhouette 0.60" in markdown
    assert "0.4554" in markdown
    assert manifest["direct3d"]["macro_acc025"] == 0.7014


def test_write_report_includes_direct3d_readout_registry(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"
    lerf = {"macro_loc_acc": 0.85, "macro_miou": 0.5, "rows": [], "warnings": []}
    scannet = {
        "scene_count": 0,
        "macro_miou": {"19": 0.0, "15": 0.0, "10": 0.0},
        "macro_macc": {"19": 0.0, "15": 0.0, "10": 0.0},
        "rows": [],
        "warnings": [],
    }
    sam3_readout = {
        "label": "direct field + official SAM3 box, pad0",
        "text_head": "SigLIP2+SAM3",
        "protocol_label": "compact direct field + official SAM3 box readout",
        "selector_policy": "best_by_miou",
        "scene_count": 4,
        "macro_miou": 0.5815,
        "macro_acc025": 0.715,
        "macro_acc050": 0.63,
        "macro_boundary_f": 0.6731,
        "macro_trimap_iou": 0.3777,
        "rows": [
            {
                "scene": "figurines",
                "selection": "thr0p11",
                "miou": 0.5924,
                "acc025": 0.7321,
                "acc050": 0.6607,
                "boundary_f": 0.6813,
                "trimap_iou": 0.3523,
                "n": 56,
                "source": "result.json",
            }
        ],
        "source_root": "output/radio_gs/lerf_direct3d_sam3_box_pad0_best_masks_20260516",
        "protocol": {"mask_refinement": "sam3_box", "sam3_box_padding": 0},
        "warnings": [],
    }

    paths = report.write_freeze_outputs(
        output_dir,
        lerf,
        scannet,
        direct3d_readouts=[sam3_readout],
    )

    markdown = paths["markdown"].read_text()
    manifest = json.loads(paths["manifest"].read_text())
    assert "Direct-3D Readout Registry" in markdown
    assert "official SAM3 box" in markdown
    assert "Boundary-F" in markdown
    assert "thr0p11" in markdown
    assert manifest["direct3d_readouts"][0]["macro_miou"] == 0.5815
    assert manifest["direct3d_readouts"][0]["macro_boundary_f"] == 0.6731


def test_collect_profile_runs_reads_time_and_gpu_logs(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profiles" / "freeze_lerf_ramen_overlay_20260502"
    profile_dir.mkdir(parents=True)
    (profile_dir / "time.log").write_text("real 12.500\nuser 1.0\nsys 0.5\n")
    (profile_dir / "gpu_metrics.csv").write_text(
        "2026/05/02 10:00:00.000, 4, 10, 1000, 24564\n"
        "2026/05/02 10:00:01.000, 4, 50, 2000, 24564\n"
    )
    (profile_dir / "meta.txt").write_text("gpu=4\ncommand=bash eval\n")

    summary = report.collect_profile_runs([profile_dir])

    assert summary["profile_count"] == 1
    assert summary["rows"][0]["name"] == "freeze_lerf_ramen_overlay_20260502"
    assert summary["rows"][0]["wall_seconds"] == 12.5
    assert summary["rows"][0]["peak_gpu_mem_mib"] == 2000.0
    assert summary["rows"][0]["mean_gpu_util_pct"] == 30.0
    assert summary["rows"][0]["gpu"] == "4"


def test_write_report_includes_profile_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"
    lerf = {"macro_loc_acc": 0.85, "macro_miou": 0.5, "rows": [], "warnings": []}
    scannet = {
        "scene_count": 0,
        "macro_miou": {"19": 0.0, "15": 0.0, "10": 0.0},
        "macro_macc": {"19": 0.0, "15": 0.0, "10": 0.0},
        "rows": [],
        "warnings": [],
    }
    profiles = {
        "profile_count": 1,
        "rows": [
            {
                "name": "freeze_lerf_ramen_overlay_20260502",
                "path": "profiles/ramen",
                "wall": "12.500 s",
                "wall_seconds": 12.5,
                "gpu": "4",
                "peak_gpu_mem_mib": 2000.0,
                "peak_gpu_util_pct": 50.0,
                "mean_gpu_util_pct": 30.0,
                "samples": 2,
                "command": "bash eval",
            }
        ],
        "warnings": [],
    }

    paths = report.write_freeze_outputs(output_dir, lerf, scannet, profiles=profiles)

    markdown = paths["markdown"].read_text()
    manifest = json.loads(paths["manifest"].read_text())
    assert "Profile Evidence" in markdown
    assert "freeze_lerf_ramen_overlay_20260502" in markdown
    assert manifest["profiles"]["profile_count"] == 1


def test_write_report_includes_claim_artifact_matrix(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"
    lerf = {"macro_loc_acc": 0.85, "macro_miou": 0.5, "rows": [], "warnings": []}
    scannet = {
        "scene_count": 1,
        "macro_miou": {"19": 0.3, "15": 0.4, "10": 0.5},
        "macro_macc": {"19": 0.6, "15": 0.7, "10": 0.8},
        "rows": [{"scene": "scene0000_00", "path": "scan/result.json", "miou": {"19": 0.3, "15": 0.4, "10": 0.5}}],
        "warnings": [],
    }
    profiles = {"profile_count": 0, "rows": [], "warnings": []}

    paths = report.write_freeze_outputs(output_dir, lerf, scannet, profiles=profiles)

    markdown = paths["markdown"].read_text()
    assert "Claim-to-Artifact Matrix" in markdown
    assert "LERF main result" in markdown
    assert "ScanNet fair cross-domain result" in markdown
    assert "Qualitative figure shortlist" in markdown


def test_main_writes_outputs(tmp_path: Path) -> None:
    lerf_csv = tmp_path / "lerf.csv"
    lerf_csv.write_text(
        "scene,loc_acc,miou,temp,checkpoint,config,output_dir,summary\n"
        "figurines,0.8,0.4,50,ckpt,cfg,out,sum\n"
        "macro,0.8,0.4,,,,,\n"
    )
    scan_dir = tmp_path / "scan"
    result_dir = scan_dir / "scene0000_00_v67_teacherbalanced_fromv63_best_gidx_labelpoint"
    result_dir.mkdir(parents=True)
    (result_dir / "scannet_pointcloud_radio_gs_results.json").write_text(
        json.dumps(
            {
                "macro": {
                    "19": {"miou": 0.2, "macc": 0.3},
                    "15": {"miou": 0.4, "macc": 0.5},
                    "10": {"miou": 0.6, "macc": 0.7},
                },
                "args": {
                    "query_mode": "gaussian_index",
                    "opacity_filter_mode": "label_index",
                    "gaussian_index_position_mode": "label_point",
                },
            }
        )
    )
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "time.log").write_text("real 2.000\n")
    (profile_dir / "gpu_metrics.csv").write_text("2026/05/02 10:00:00.000, 5, 20, 1500, 24564\n")
    (profile_dir / "meta.txt").write_text("gpu=5\ncommand=bash eval\n")
    output_dir = tmp_path / "reports"

    report.main(
        [
            "--lerf_csv",
            str(lerf_csv),
            "--scannet_eval_root",
            str(scan_dir),
            "--profile_dirs",
            str(profile_dir),
            "--output_dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "submission_freeze_report.md").exists()
    manifest = json.loads((output_dir / "submission_freeze_manifest.json").read_text())
    assert manifest["profiles"]["profile_count"] == 1
