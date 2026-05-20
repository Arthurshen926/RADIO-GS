import importlib
from pathlib import Path

import pytest


def _load_module():
    try:
        return importlib.import_module("radio_gs.scripts.summarize_gags_lerf_baseline")
    except ImportError as exc:
        pytest.fail(f"missing summarize_gags_lerf_baseline module: {exc}")


def _write_eval_log(path: Path, scene: str, *, locacc: float, miou: float, frames: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"2026-05-19 23:30:00,000 - {scene} - INFO - Using the full camera list for label-frame evaluation; overriding cfg eval split.",
    ]
    for idx, (hits, total) in enumerate(frames, start=1):
        lines.append(
            f"2026-05-19 23:31:{idx:02d},000 - {scene} - INFO - eval: {idx:0>5} "
            f"acc_num: {hits}/{total} mean_iou: {miou:.4f}"
        )
    lines.extend(
        [
            "2026-05-19 23:32:00,000 - {scene} - INFO - trunc thresh: 0.4".format(scene=scene),
            "2026-05-19 23:32:00,000 - {scene} - INFO - iou chosen: {miou:.4f}".format(
                scene=scene,
                miou=miou,
            ),
            "2026-05-19 23:32:00,000 - {scene} - INFO - Localization accuracy: {locacc:.4f}".format(
                scene=scene,
                locacc=locacc,
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_eval_log_extracts_gags_metrics_and_query_count(tmp_path):
    module = _load_module()
    log_path = tmp_path / "gags" / "ramen" / "train" / "ours_30000" / "eval" / "20260519_233000.log"
    _write_eval_log(log_path, "ramen", locacc=0.75, miou=0.5, frames=[(2, 3), (4, 5)])

    row = module.parse_eval_log(log_path)

    assert row.scene == "ramen"
    assert row.checkpoint == 30000
    assert row.mask_thresh == 0.4
    assert row.locacc == 0.75
    assert row.miou == 0.5
    assert row.query_count == 8
    assert row.full_camera_list is True


def test_build_summary_uses_latest_completed_log_per_scene(tmp_path):
    module = _load_module()
    root = tmp_path / "gags"
    _write_eval_log(
        root / "ramen" / "train" / "ours_30000" / "eval" / "20260519_223000.log",
        "ramen",
        locacc=0.1,
        miou=0.2,
        frames=[(1, 1)],
    )
    _write_eval_log(
        root / "ramen" / "train" / "ours_30000" / "eval" / "20260519_233000.log",
        "ramen",
        locacc=0.4,
        miou=0.3,
        frames=[(1, 2), (1, 2)],
    )
    _write_eval_log(
        root / "teatime" / "train" / "ours_30000" / "eval" / "20260519_234000.log",
        "teatime",
        locacc=0.6,
        miou=0.7,
        frames=[(2, 3)],
    )
    (root / "figurines" / "train" / "ours_30000" / "eval" / "empty.log").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (root / "figurines" / "train" / "ours_30000" / "eval" / "empty.log").write_text(
        "",
        encoding="utf-8",
    )

    summary = module.build_summary(root)
    markdown = module.render_markdown(summary)

    assert [row["scene"] for row in summary["completed_rows"]] == ["ramen", "teatime"]
    assert summary["scene_mean"]["locacc"] == pytest.approx((0.4 + 0.6) / 2)
    assert summary["scene_mean"]["miou"] == pytest.approx((0.3 + 0.7) / 2)
    assert summary["object_weighted"]["query_count"] == 7
    assert summary["object_weighted"]["miou"] == pytest.approx((0.3 * 4 + 0.7 * 3) / 7)
    assert "# GAGS LERF Compatibility Summary" in markdown
    assert "| ramen | 30000 | 0.4000 | 4 | 0.4000 | 0.3000 | True |" in markdown
