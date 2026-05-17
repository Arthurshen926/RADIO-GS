import json
from pathlib import Path

from radio_gs.scripts import build_controlled_evidence_table as table


def _write_component_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "variants": [
                    {"key": "full", "label": "Full CTF-GS", "note": "hybrid + HCD"},
                    {"key": "no_fdh", "label": "w/o FGC", "note": "no frozen head"},
                ],
                "results": {
                    "full": {
                        "figurines": {"loc_acc": 0.8, "miou": 0.4},
                        "ramen": {"loc_acc": 0.9, "miou": 0.6},
                    },
                    "no_fdh": {
                        "figurines": {"loc_acc": 0.7, "miou": 0.3},
                        "ramen": {"loc_acc": 0.8, "miou": 0.5},
                    },
                },
            }
        )
    )


def _manifest() -> dict:
    return {
        "lerf": {"macro_loc_acc": 0.8712, "macro_miou": 0.5243},
        "direct3d_readouts": [
            {
                "label": "VPR fixed threshold + RGB snap",
                "selector_policy": "fixed:thr0p25",
                "macro_miou": 0.4801,
                "macro_acc025": 0.6760,
            },
            {
                "label": "direct field + official SAM3 box, pad16 fixed global threshold",
                "selector_policy": "fixed:thr0p25",
                "macro_miou": 0.5705,
                "macro_acc025": 0.6835,
            },
        ],
    }


def test_build_rows_keeps_ablation_direct3d_unmeasured(tmp_path: Path) -> None:
    component = tmp_path / "component.json"
    _write_component_json(component)

    rows = table.build_rows(
        component_path=component,
        freeze_manifest=_manifest(),
        storage_summary={"mean_saving": 3.0},
        profile_summary={"mean_lerf_wall_seconds": 31.2},
        nearest_view_path=None,
        per_gaussian_1280d_path=None,
    )

    full = rows[1]
    no_fdh = rows[2]
    assert full["method"] == "Full CTF-GS"
    assert full["lerf_loc_acc"] == 0.8712
    assert full["lerf_miou"] == 0.5243
    assert full["direct3d"] == "VPR 0.4801/0.6760; SAM3-box 0.5705/0.6835"
    assert full["storage"] == "3.00x mean compact checkpoint saving"
    assert full["runtime"] == "31.2s mean LERF overlay"
    assert no_fdh["direct3d"] == "not evaluated"
    assert no_fdh["lerf_loc_acc"] == 0.75
    assert no_fdh["lerf_miou"] == 0.4


def test_build_rows_includes_measured_nearest_view_cache_baseline(tmp_path: Path) -> None:
    component = tmp_path / "component.json"
    nearest = tmp_path / "nearest.json"
    _write_component_json(component)
    nearest.write_text(
        json.dumps(
            {
                "macro": {"loc_acc": 0.2722, "miou": 0.1545},
                "mean_nearest_distance": 0.4582,
                "protocol": {"selection": "nearest_by_camera_center", "warp": "none"},
            }
        ),
        encoding="utf-8",
    )

    rows = table.build_rows(
        component_path=component,
        freeze_manifest=_manifest(),
        storage_summary={"mean_saving": 3.0},
        profile_summary={"mean_lerf_wall_seconds": 31.2},
        nearest_view_path=nearest,
        per_gaussian_1280d_path=None,
    )

    nearest_row = rows[1]
    assert nearest_row["method"] == "Nearest-view RADIO cache"
    assert nearest_row["compact"] == "no"
    assert nearest_row["3d_memory"] == "no"
    assert nearest_row["novel_view_feature"] == "cache-only"
    assert nearest_row["lerf_loc_acc"] == 0.2722
    assert nearest_row["lerf_miou"] == 0.1545
    assert nearest_row["source"] == "lerf_nearest_view_cache_baseline.json"


def test_build_rows_includes_measured_per_gaussian_1280d_baseline(tmp_path: Path) -> None:
    component = tmp_path / "component.json"
    explicit = tmp_path / "explicit_1280d.json"
    _write_component_json(component)
    explicit.write_text(
        json.dumps(
            {
                "macro": {"loc_acc": 0.71, "miou": 0.49},
                "mean_registered_fraction": 0.82,
                "mean_storage_mib": 980.0,
                "protocol": {
                    "feature_source": "registered RADIO 1280-D teacher features",
                    "feature_dim": 1280,
                },
            }
        ),
        encoding="utf-8",
    )

    rows = table.build_rows(
        component_path=component,
        freeze_manifest=_manifest(),
        storage_summary={"mean_saving": 3.0},
        profile_summary={"mean_lerf_wall_seconds": 31.2},
        nearest_view_path=None,
        per_gaussian_1280d_path=explicit,
    )

    explicit_row = rows[1]
    assert explicit_row["method"] == "Per-Gaussian 1280-D RADIO memory"
    assert explicit_row["compact"] == "no"
    assert explicit_row["3d_memory"] == "yes"
    assert explicit_row["novel_view_feature"] == "yes"
    assert explicit_row["direct_3d_query"] == "partial"
    assert explicit_row["lerf_loc_acc"] == 0.71
    assert explicit_row["lerf_miou"] == 0.49
    assert explicit_row["storage"] == "980.0 MiB mean fp16 feature storage"
    assert explicit_row["runtime"] == "registered fraction 0.8200"
    assert explicit_row["source"] == "lerf_per_gaussian_1280d_baseline.json"


def test_build_markdown_marks_external_teacher_not_3d_memory() -> None:
    markdown = table.build_markdown(
        [
            table.teacher_row(teacher_loc_acc=0.7985, teacher_miou=0.4634),
        ]
    )

    assert "Frame-wise RADIO teacher" in markdown
    assert "| no | no | no | no |" in markdown
    assert "0.7985" in markdown


def test_write_outputs_records_markdown_and_json(tmp_path: Path) -> None:
    rows = [table.teacher_row(0.7985, 0.4634)]

    paths = table.write_outputs(rows, tmp_path / "controlled.md", tmp_path / "controlled.json")

    assert paths["markdown"].read_text().startswith("# Controlled Evidence Table")
    payload = json.loads(paths["json"].read_text())
    assert payload["rows"][0]["method"] == "Frame-wise RADIO teacher"
