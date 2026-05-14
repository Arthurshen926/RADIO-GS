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
