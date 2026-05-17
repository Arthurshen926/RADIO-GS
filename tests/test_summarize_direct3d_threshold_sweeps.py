import json
from pathlib import Path

from radio_gs.scripts import summarize_direct3d_threshold_sweeps as sweeps


SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")


def _metric(miou: float, *, tag: str) -> dict:
    return {
        "miou": miou,
        "acc025": miou + 0.1,
        "acc050": miou - 0.1,
        "boundary_f": miou + 0.2,
        "trimap_iou": miou - 0.2,
        "n": 5,
        "selection_tag": tag,
        "selection_value": 0.25,
        "mask_refinement": "sam3_box",
    }


def _write_scene(root: Path, scene: str, thr10: float, thr25: float, best_tag: str) -> None:
    scene_dir = root / scene
    scene_dir.mkdir(parents=True)
    payload = {
        "protocol": {"mask_refinement": "sam3_box", "sam3_box_padding": 16},
        "scene": {
            "scene": scene,
            "results": {
                "thr0p10": _metric(thr10, tag="thr0p10"),
                "thr0p25": _metric(thr25, tag="thr0p25"),
            },
            "best_by_miou": best_tag,
        },
    }
    (scene_dir / "lerf_direct_3d_selection_results.json").write_text(json.dumps(payload))


def _write_run(root: Path) -> None:
    values = {
        "figurines": (0.55, 0.50, "thr0p10"),
        "ramen": (0.50, 0.60, "thr0p25"),
        "teatime": (0.45, 0.40, "thr0p10"),
        "waldo_kitchen": (0.35, 0.30, "thr0p10"),
    }
    for scene, (thr10, thr25, best_tag) in values.items():
        _write_scene(root, scene, thr10, thr25, best_tag)


def test_summarize_run_separates_fixed_global_posthoc_and_scene_locked(tmp_path: Path) -> None:
    root = tmp_path / "pad16"
    _write_run(root)

    summary = sweeps.summarize_runs([("pad16", root)], protocol_tag="thr0p25")

    run = summary["runs"][0]
    assert run["scene_count"] == 4
    assert run["missing_scenes"] == []
    assert run["fixed_global_threshold"]["selector_policy"] == "fixed_global_threshold:thr0p25"
    assert run["fixed_global_threshold"]["macro_miou"] == 0.45
    assert run["fixed_global_threshold"]["rows"][0]["selection"] == "thr0p25"
    assert run["best_fixed_macro_threshold"]["selector_policy"] == "diagnostic_posthoc_best_fixed_threshold"
    assert run["best_fixed_macro_threshold"]["selection"] == "thr0p10"
    assert run["best_fixed_macro_threshold"]["macro_miou"] == 0.4625
    assert run["scene_locked_best"]["selector_policy"] == "diagnostic_scene_locked_best_by_miou"
    assert run["scene_locked_best"]["macro_miou"] == 0.4875
    assert any("diagnostic" in warning for warning in summary["warnings"])


def test_summarize_run_computes_macro_before_rounding_scene_rows(tmp_path: Path) -> None:
    root = tmp_path / "pad16"
    values = {
        "figurines": 0.57044,
        "ramen": 0.57044,
        "teatime": 0.57044,
        "waldo_kitchen": 0.57049,
    }
    for scene, miou in values.items():
        _write_scene(root, scene, miou, miou, "thr0p25")

    summary = sweeps.summarize_runs([("pad16", root)], protocol_tag="thr0p25")

    fixed = summary["runs"][0]["fixed_global_threshold"]
    assert fixed["rows"][0]["miou"] == 0.5704
    assert fixed["macro_miou"] == 0.5705


def test_write_outputs_records_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "pad16"
    _write_run(root)
    summary = sweeps.summarize_runs([("pad16", root)], protocol_tag="thr0p25")
    json_path = tmp_path / "sweep.json"
    md_path = tmp_path / "sweep.md"

    paths = sweeps.write_outputs(summary, json_path=json_path, markdown_path=md_path)

    assert paths["json"] == json_path
    assert paths["markdown"] == md_path
    manifest = json.loads(json_path.read_text())
    markdown = md_path.read_text()
    assert manifest["protocol_tag"] == "thr0p25"
    assert manifest["runs"][0]["label"] == "pad16"
    assert "Fixed global threshold" in markdown
    assert "pad16" in markdown
    assert "thr0p25" in markdown
    assert "diagnostic" in markdown
