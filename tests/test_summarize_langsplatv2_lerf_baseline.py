import importlib
from pathlib import Path

import pytest


def _load_module():
    try:
        return importlib.import_module("radio_gs.scripts.summarize_langsplatv2_lerf_baseline")
    except ImportError as exc:
        pytest.fail(f"missing summarize_langsplatv2_lerf_baseline module: {exc}")


def _write_eval_log(path: Path, scene: str, *, locacc: float, miou: float, levels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"2026-05-18 11:49:42,093 - {scene} - INFO - checkpoint: 10000",
                f"2026-05-18 11:49:42,093 - {scene} - INFO - trunc thresh: 0.4",
                f"2026-05-18 11:49:42,093 - {scene} - INFO - iou chosen: {miou:.4f}",
                f"2026-05-18 11:49:42,093 - {scene} - INFO - chosen_lvl: ",
                repr(levels),
                f"2026-05-18 11:49:42,093 - {scene} - INFO - Localization accuracy: {locacc:.4f}",
            ]
        ),
        encoding="utf-8",
    )


def test_parse_eval_log_extracts_langsplatv2_metrics(tmp_path):
    module = _load_module()
    log_path = tmp_path / "eval/teatime_0/20260518_114759.log"
    _write_eval_log(log_path, "teatime", locacc=0.2034, miou=0.0968, levels=[2, 1, 2])

    row = module.parse_eval_log(log_path)

    assert row.scene == "teatime"
    assert row.index == 0
    assert row.checkpoint == 10000
    assert row.mask_thresh == 0.4
    assert row.locacc == 0.2034
    assert row.miou == 0.0968
    assert row.query_count == 3


def test_build_summary_uses_latest_nonempty_log_and_weighted_macro(tmp_path):
    module = _load_module()
    root = tmp_path / "langsplatv2"
    _write_eval_log(
        root / "eval/teatime_0/20260518_114700.log",
        "teatime",
        locacc=0.1,
        miou=0.2,
        levels=[0],
    )
    _write_eval_log(
        root / "eval/teatime_0/20260518_114759.log",
        "teatime",
        locacc=0.2034,
        miou=0.0968,
        levels=[2, 1, 2],
    )
    _write_eval_log(
        root / "eval/ramen_0/20260518_130000.log",
        "ramen",
        locacc=0.4,
        miou=0.3,
        levels=[1, 1],
    )
    (root / "eval/figurines_0/empty.log").parent.mkdir(parents=True, exist_ok=True)
    (root / "eval/figurines_0/empty.log").write_text("", encoding="utf-8")

    summary = module.build_summary(root)

    assert [row["scene"] for row in summary["completed_rows"]] == ["ramen", "teatime"]
    assert summary["scene_mean"]["locacc"] == pytest.approx((0.4 + 0.2034) / 2)
    assert summary["scene_mean"]["miou"] == pytest.approx((0.3 + 0.0968) / 2)
    assert summary["object_weighted"]["query_count"] == 5
    assert summary["object_weighted"]["miou"] == pytest.approx((0.3 * 2 + 0.0968 * 3) / 5)
