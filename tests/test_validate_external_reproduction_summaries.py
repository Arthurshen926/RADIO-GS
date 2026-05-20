import importlib
import json
from pathlib import Path
from typing import List, Optional

import pytest


def _load_module():
    try:
        return importlib.import_module("radio_gs.scripts.validate_external_reproduction_summaries")
    except ImportError as exc:
        pytest.fail(f"missing validate_external_reproduction_summaries module: {exc}")


def _write_gags(path: Path, *, scenes: Optional[List[str]] = None, query_count: int = 208) -> None:
    if scenes is None:
        scenes = ["figurines", "ramen", "teatime", "waldo_kitchen"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "completed_rows": [{"scene": scene} for scene in scenes],
                "object_weighted": {"query_count": query_count},
            }
        ),
        encoding="utf-8",
    )


def _write_drsplat(path: Path, *, scenes: Optional[List[str]] = None, count: int = 208, missing: int = 7) -> None:
    if scenes is None:
        scenes = ["figurines", "ramen", "teatime", "waldo_kitchen"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "scenes": {scene: {} for scene in scenes},
                "macro": {"count": count, "missing": missing},
            }
        ),
        encoding="utf-8",
    )


def test_validate_accepts_complete_gags_and_drsplat_summaries(tmp_path):
    module = _load_module()
    gags = tmp_path / "gags.json"
    drsplat = tmp_path / "drsplat.json"
    _write_gags(gags)
    _write_drsplat(drsplat)

    issues = module.validate_summaries(gags_summary_path=gags, drsplat_summary_path=drsplat)

    assert issues == []


def test_validate_flags_incomplete_scene_and_count(tmp_path):
    module = _load_module()
    gags = tmp_path / "gags.json"
    drsplat = tmp_path / "drsplat.json"
    _write_gags(gags, scenes=["ramen"], query_count=71)
    _write_drsplat(drsplat, missing=208)

    issues = module.validate_summaries(gags_summary_path=gags, drsplat_summary_path=drsplat)

    assert any("GAGS scenes" in issue for issue in issues)
    assert any("GAGS query_count" in issue for issue in issues)
    assert any("Dr. Splat missing" in issue for issue in issues)


def test_validate_flags_missing_required_summary(tmp_path):
    module = _load_module()

    issues = module.validate_summaries(
        gags_summary_path=tmp_path / "missing_gags.json",
        drsplat_summary_path=tmp_path / "missing_drsplat.json",
        require_gags=True,
        require_drsplat=True,
    )

    assert "missing required GAGS summary" in issues
    assert "missing required Dr. Splat summary" in issues
