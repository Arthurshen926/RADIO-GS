import importlib
import json
from pathlib import Path

import pytest
import yaml


def _load_module():
    try:
        return importlib.import_module("radio_gs.scripts.sync_external_reproduction_summaries")
    except ImportError as exc:
        pytest.fail(f"missing sync_external_reproduction_summaries module: {exc}")


def _write_final_rows(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "external_reproduction_queue:",
                "  p0:",
                '    - method: "GAGS"',
                '      status: "old GAGS status"',
                '      local_repo: "/root/baselines/GAGS"',
                '    - method: "Dr. Splat"',
                '      status: "old Dr. Splat status"',
                '      local_repo: "/root/baselines/Dr-Splat"',
                "  p1:",
                '    - method: "LEGaussians"',
                '      status: "old LEGaussians status"',
                '    - method: "Semantic Gaussians"',
                '      status: "old Semantic status"',
                '    - method: "LaGa"',
                '      status: "old LaGa status"',
                '    - method: "CAGS"',
                '      status: "keep me"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_build_status_updates_only_available_summaries(tmp_path):
    module = _load_module()
    final_rows = tmp_path / "paper/artifacts/final_rows.yaml"
    gags_summary = tmp_path / "paper/artifacts/gags_lerf_summary.json"
    drsplat_summary = tmp_path / "paper/artifacts/drsplat_lerf_summary.json"
    _write_final_rows(final_rows)
    gags_summary.write_text(
        json.dumps(
            {
                "completed_rows": [{"scene": scene} for scene in ("figurines", "ramen", "teatime", "waldo_kitchen")],
                "scene_mean": {"locacc": 0.5, "miou": 0.6},
                "object_weighted": {"query_count": 10, "locacc": 0.55, "miou": 0.65},
            }
        ),
        encoding="utf-8",
    )

    changed = module.sync_final_rows(
        final_rows,
        gags_summary_path=gags_summary,
        drsplat_summary_path=drsplat_summary,
        legaussians_summary_path=tmp_path / "missing_legaussians.json",
        semantic_gaussians_summary_path=tmp_path / "missing_semantic.json",
        laga_summary_path=tmp_path / "missing_laga.json",
    )

    assert changed is True
    payload = yaml.safe_load(final_rows.read_text(encoding="utf-8"))
    rows = {
        row["method"]: row["status"]
        for bucket in ("p0", "p1")
        for row in payload["external_reproduction_queue"].get(bucket, [])
    }
    assert "old GAGS status" not in rows["GAGS"]
    assert "all four LERF compatibility scenes completed" in rows["GAGS"]
    assert "scene-mean LocAcc 0.5000 / mIoU 0.6000" in rows["GAGS"]
    assert "object-weighted LocAcc 0.5500 / mIoU 0.6500 over 10 queries" in rows["GAGS"]
    assert rows["Dr. Splat"] == "old Dr. Splat status"
    assert rows["CAGS"] == "keep me"


def test_sync_final_rows_records_drsplat_mask_summary(tmp_path):
    module = _load_module()
    final_rows = tmp_path / "paper/artifacts/final_rows.yaml"
    gags_summary = tmp_path / "paper/artifacts/gags_lerf_summary.json"
    drsplat_summary = tmp_path / "paper/artifacts/drsplat_lerf_summary.json"
    _write_final_rows(final_rows)
    drsplat_summary.write_text(
        json.dumps(
            {
                "scenes": {"figurines": {}, "ramen": {}, "teatime": {}, "waldo_kitchen": {}},
                "macro": {"miou": 0.31, "acc025": 0.45, "acc05": 0.2, "count": 8, "missing": 1},
            }
        ),
        encoding="utf-8",
    )

    module.sync_final_rows(
        final_rows,
        gags_summary_path=gags_summary,
        drsplat_summary_path=drsplat_summary,
        legaussians_summary_path=tmp_path / "missing_legaussians.json",
        semantic_gaussians_summary_path=tmp_path / "missing_semantic.json",
        laga_summary_path=tmp_path / "missing_laga.json",
    )

    text = final_rows.read_text(encoding="utf-8")
    assert "old Dr. Splat status" not in text
    assert "all four LERF compatibility scenes completed" in text
    assert "mIoU 0.3100 / Acc@0.25 0.4500 / Acc@0.5 0.2000 over 8 objects" in text
    assert "missing rendered masks counted: 1" in text


def test_sync_final_rows_records_later_external_summaries(tmp_path):
    module = _load_module()
    final_rows = tmp_path / "paper/artifacts/final_rows.yaml"
    gags_summary = tmp_path / "paper/artifacts/gags_lerf_summary.json"
    drsplat_summary = tmp_path / "paper/artifacts/drsplat_lerf_summary.json"
    legaussians_summary = tmp_path / "paper/artifacts/legaussians_lerf_summary.json"
    semantic_summary = tmp_path / "semantic_gaussians_eval_metrics.json"
    laga_summary = tmp_path / "paper/artifacts/laga_lerf_summary.json"
    _write_final_rows(final_rows)
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
    semantic_summary.write_text(
        json.dumps(
            {
                "metrics": {"mean_iou": 0.028},
                "scenes": {"scene0000_00": {}, "scene0062_00": {}, "scene0070_00": {}, "scene0097_00": {}},
            }
        ),
        encoding="utf-8",
    )
    laga_summary.write_text(
        json.dumps(
            {
                "scenes": {"figurines": {}, "ramen": {}, "teatime": {}, "waldo_kitchen": {}},
                "macro": {"miou": 0.33, "acc025": 0.4, "acc05": 0.25, "count": 7, "missing": 0},
            }
        ),
        encoding="utf-8",
    )

    changed = module.sync_final_rows(
        final_rows,
        gags_summary_path=gags_summary,
        drsplat_summary_path=drsplat_summary,
        legaussians_summary_path=legaussians_summary,
        semantic_gaussians_summary_path=semantic_summary,
        laga_summary_path=laga_summary,
    )

    assert changed is True
    payload = yaml.safe_load(final_rows.read_text(encoding="utf-8"))
    rows = {
        row["method"]: row["status"]
        for bucket in ("p0", "p1")
        for row in payload["external_reproduction_queue"].get(bucket, [])
    }
    assert "old LEGaussians status" not in rows["LEGaussians"]
    assert "scene-mean mIoU 0.2100 / Acc@0.25 0.3000 / Acc@0.5 0.1200" in rows["LEGaussians"]
    assert "ScanNet-20 label-PLY mean IoU 0.0280 over 4 scenes" in rows["Semantic Gaussians"]
    assert "old LaGa status" not in rows["LaGa"]
    assert "mIoU 0.3300 / Acc@0.25 0.4000 / Acc@0.5 0.2500 over 7 objects" in rows["LaGa"]
