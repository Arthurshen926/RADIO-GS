import json

from radio_gs.scripts import summarize_opengaussian_baseline as summary


def _write_scene(root, scene, miou, acc025):
    scene_dir = root / scene
    scene_dir.mkdir(parents=True)
    payload = {
        "scene": {
            "scene": scene,
            "results": {
                "meanstd2p5": {
                    "miou": miou,
                    "acc025": acc025,
                },
                "thr0p25": {
                    "miou": miou + 0.01,
                    "acc025": acc025 + 0.01,
                }
            },
        }
    }
    (scene_dir / "lerf_direct_3d_selection_results.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_load_radio_direct_lerf_results_computes_macro(tmp_path):
    for idx, scene in enumerate(summary.LERF_SCENES):
        _write_scene(tmp_path, scene, miou=0.1 * (idx + 1), acc025=0.2 * (idx + 1))

    direct = summary._load_radio_direct_lerf_results(tmp_path, "meanstd2p5")

    assert direct is not None
    assert direct["figurines"]["miou"] == 0.1
    assert direct["waldo_kitchen"]["acc025"] == 0.8
    assert direct["macro"]["miou"] == 0.25
    assert direct["macro"]["acc025"] == 0.5


def test_default_direct_lerf_args_use_promoted_fixed_threshold():
    parser = summary.build_arg_parser()
    args = parser.parse_args([])

    assert args.radio_direct_lerf_root.endswith("lerf_direct_3d_selection_threshold_grabcut_20260515")
    assert args.radio_direct_lerf_tag == "thr0p25"


def test_lerf_lines_include_direct_3d_when_available():
    direct = {
        "figurines": {"miou": 0.4, "acc025": 0.8},
        "ramen": {"miou": 0.3, "acc025": 0.7},
        "teatime": {"miou": 0.5, "acc025": 0.9},
        "waldo_kitchen": {"miou": 0.2, "acc025": 0.4},
        "macro": {"miou": 0.35, "acc025": 0.7},
    }

    lines = "\n".join(summary._lerf_lines([], direct))

    assert "VPR direct 3D selection mIoU" in lines
    assert "VPR direct 3D selection Acc@0.25" in lines
    assert "0.3500" in lines


def test_load_opengaussian_lerf_results_reads_local_eval_report(tmp_path):
    report_path = tmp_path / "opengaussian_lerf_eval.json"
    report_path.write_text(
        json.dumps(
            {
                "scenes": {
                    "figurines": {"miou": 0.4, "acc025": 0.8},
                    "ramen": {"miou": 0.3, "acc025": 0.7},
                    "teatime": {"miou": 0.5, "acc025": 0.9},
                    "waldo_kitchen": {"miou": 0.2, "acc025": 0.4},
                },
                "macro": {"miou": 0.35, "acc025": 0.7},
            }
        ),
        encoding="utf-8",
    )

    results = summary._load_opengaussian_lerf_results(report_path)

    assert results is not None
    assert results["figurines"]["miou"] == 0.4
    assert results["macro"]["acc025"] == 0.7


def test_lerf_lines_include_local_opengaussian_reproduction_when_available():
    og_lerf = {
        "figurines": {"miou": 0.4, "acc025": 0.8},
        "ramen": {"miou": 0.3, "acc025": 0.7},
        "teatime": {"miou": 0.5, "acc025": 0.9},
        "waldo_kitchen": {"miou": 0.2, "acc025": 0.4},
        "macro": {"miou": 0.35, "acc025": 0.7},
    }

    lines = "\n".join(summary._lerf_lines([], None, None, og_lerf))

    assert "local compatibility rerun object-selection mIoU" in lines
    assert "local compatibility rerun Acc@0.25" in lines
    assert "0.3500" in lines


def test_lerf_lines_describes_completed_local_opengaussian_rerun():
    og_lerf = {
        "figurines": {"miou": 0.4, "acc025": 0.8},
        "ramen": {"miou": 0.3, "acc025": 0.7},
        "teatime": {"miou": 0.5, "acc025": 0.9},
        "waldo_kitchen": {"miou": 0.2, "acc025": 0.4},
        "macro": {"miou": 0.35, "acc025": 0.7},
    }
    asset_status = {
        scene: {
            "images": 1,
            "language_feature_masks": 1,
            "language_feature_vectors": 1,
            "labels": 1,
            "ready": True,
        }
        for scene in summary.LERF_SCENES
    }

    lines = "\n".join(summary._lerf_lines([], None, asset_status, og_lerf))

    assert "Local OpenGaussian LeRF compatibility rerun completed" in lines
    assert "macro mIoU 0.3500 and Acc@0.25 0.7000" in lines


def test_inspect_opengaussian_lerf_assets_detects_missing_language_features(tmp_path):
    scene_root = tmp_path / "figurines"
    (scene_root / "images").mkdir(parents=True)
    (scene_root / "images" / "frame_00001.jpg").write_bytes(b"x")
    (tmp_path / "label" / "figurines" / "gt" / "frame_00001").mkdir(parents=True)
    (tmp_path / "label" / "figurines" / "gt" / "frame_00001" / "object.jpg").write_bytes(b"x")

    status = summary.inspect_opengaussian_lerf_assets(tmp_path)

    assert status["figurines"]["images"] == 1
    assert status["figurines"]["labels"] == 1
    assert status["figurines"]["language_feature_masks"] == 0
    assert status["figurines"]["ready"] is False


def test_inspect_opengaussian_lerf_assets_requires_every_image_pair(tmp_path):
    scene_root = tmp_path / "figurines"
    (scene_root / "images").mkdir(parents=True)
    (scene_root / "images" / "frame_00001.jpg").write_bytes(b"x")
    (scene_root / "images" / "frame_00002.jpg").write_bytes(b"x")
    (scene_root / "language_features").mkdir()
    (scene_root / "language_features" / "frame_00001_s.npy").write_bytes(b"x")
    (scene_root / "language_features" / "frame_00001_f.npy").write_bytes(b"x")

    status = summary.inspect_opengaussian_lerf_assets(tmp_path)

    assert status["figurines"]["language_feature_masks"] == 1
    assert status["figurines"]["language_feature_vectors"] == 1
    assert status["figurines"]["ready"] is False


def test_lerf_lines_include_local_asset_blocker():
    lines = "\n".join(
        summary._lerf_lines(
            [],
            None,
            {
                scene: {
                    "images": 1,
                    "language_feature_masks": 0,
                    "language_feature_vectors": 0,
                    "labels": 1,
                    "ready": False,
                }
                for scene in summary.LERF_SCENES
            },
        )
    )

    assert "Local OpenGaussian LeRF Asset Check" in lines
    assert "language_features/*_s.npy" in lines


def test_load_radio_lerf_threshold_sweep_returns_calibrated_rows(tmp_path):
    sweep_path = tmp_path / "threshold_sweep.json"
    sweep_path.write_text(
        json.dumps(
            {
                "variants": {
                    "0.60": {
                        "rows": [
                            {"scene": "figurines", "loc": 0.8, "miou": 0.42},
                            {"scene": "ramen", "loc": 0.9, "miou": 0.62},
                            {"scene": "teatime", "loc": 0.7, "miou": 0.52},
                            {"scene": "waldo_kitchen", "loc": 0.6, "miou": 0.44},
                        ],
                        "macro": {"loc": 0.75, "miou": 0.50},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    rows = summary._load_radio_lerf_threshold_sweep(sweep_path, "0.60")
    by_scene = {row["scene"]: row for row in rows}

    assert by_scene["figurines"]["miou"] == "0.4200"
    assert by_scene["macro"]["loc_acc"] == "0.7500"
    assert by_scene["macro"]["miou"] == "0.5000"
