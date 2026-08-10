import json
from pathlib import Path

import pytest

from radio_gs.scripts.aggregate_spin_reference_selected_query_interface import (
    _parse_report_specs,
    aggregate,
)


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _canonical(path: Path) -> Path:
    return _write(
        path,
        {
            "protocol_hash": "frozen",
            "macro_over_requested_scenes": {"local_ludvig_sam": 0.8},
            "scenes": {
                "a": {
                    "query_conditioned_support": 0.6,
                    "report": "/canonical/a.json",
                    "report_sha256": "a" * 64,
                    "selected_by_reference_only": {"reference_iou": 0.7},
                },
                "b": {
                    "query_conditioned_support": 0.9,
                    "report": "/canonical/b.json",
                    "report_sha256": "b" * 64,
                    "selected_by_reference_only": {"reference_iou": 0.8},
                },
            },
        },
    )


def _sam(path: Path, scene: str, reference: float, target: float) -> Path:
    return _write(
        path,
        {
            "scene_id": scene,
            "protocol_hash": "frozen",
            "foreground_iou": target,
            "prediction_persisted_before_target_mask_access": True,
            "render_resolution_mode": "registered",
            "renderer_resolution": [10, 20],
            "reference_receipt": {
                "selected_reference_iou": reference,
                "selected_candidate": 0,
                "selected_threshold": 0.5,
                "target_masks_opened": False,
            },
        },
    )


def test_reference_only_selector_and_canonical_tie(tmp_path: Path) -> None:
    canonical = _canonical(tmp_path / "canonical.json")
    a = _sam(tmp_path / "a.json", "a", 0.71, 0.95)
    b = _sam(tmp_path / "b.json", "b", 0.8, 0.1)
    result = aggregate(canonical, {"a": a, "b": b})
    assert result["scenes"]["a"]["selected_branch"] == "sam"
    assert result["scenes"]["b"]["selected_branch"] == "canonical"
    assert result["macro"]["canonical"] == pytest.approx(0.75)
    assert result["macro"]["sam"] == pytest.approx(0.525)
    assert result["macro"]["reference_selected"] == pytest.approx(0.925)


def test_requires_exact_scene_cohort_and_protocol(tmp_path: Path) -> None:
    canonical = _canonical(tmp_path / "canonical.json")
    a = _sam(tmp_path / "a.json", "a", 0.9, 0.9)
    with pytest.raises(ValueError, match="scene mismatch"):
        aggregate(canonical, {"a": a})
    b = _sam(tmp_path / "b.json", "b", 0.9, 0.9)
    value = json.loads(b.read_text())
    value["protocol_hash"] = "wrong"
    _write(b, value)
    with pytest.raises(ValueError, match="protocol hash"):
        aggregate(canonical, {"a": a, "b": b})


def test_report_specs_reject_duplicates() -> None:
    assert _parse_report_specs(["a=/tmp/a.json"])["a"] == Path("/tmp/a.json")
    with pytest.raises(ValueError, match="duplicate"):
        _parse_report_specs(["a=/tmp/a.json", "a=/tmp/b.json"])
