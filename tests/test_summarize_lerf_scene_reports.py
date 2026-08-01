import json
from pathlib import Path

import pytest

from radio_gs.scripts.summarize_lerf_scene_reports import summarize_reports


def _write_report(
    path: Path,
    *,
    scene: str,
    miou: float,
    localization: float,
    sample_count: int,
    valid_normalization: bool = True,
    coverage_power: float = 0.0,
) -> Path:
    payload = {
        "args": {
            "rendered_only": "True",
            "render_readout": "primitive_unary",
            "primitive_valid_normalization": str(valid_normalization),
            "primitive_valid_coverage_power": str(coverage_power),
            "feature_contribution_gamma": "1.0",
            "scoring": "cosine",
            "relevancy_temp": "1.0",
            "threshold_mode": "fixed",
            "iou_threshold": "0.6",
            "heatmap_upsample": "4",
            "localization_mode": "polygon_argmax",
            "mask_refinement": "none",
            "prompt_templates": "{query}",
        },
        "prompt_templates": ["{query}"],
        "feature_observation_operator": {
            "type": "alpha_normalized_mean",
            "gamma": 1.0,
            "primitive_valid_normalization": valid_normalization,
            "semantic_score_formula": (
                "sum(w*v*s)/sum(w*v) * coverage**coverage_power"
                if valid_normalization
                else "sum(w*v*s)/sum(w)"
            ),
            "semantic_coverage_power": (
                coverage_power if valid_normalization else None
            ),
            "query_dependent": False,
            "changes_geometry_or_alpha": False,
        },
        "scenes": {scene: {}},
        "aggregates": {
            "rendered": {
                "sample_micro_miou": miou,
                "localization_accuracy": localization,
                "sample_count": sample_count,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_summarizer_checks_scene_set_and_computes_both_aggregates(
    tmp_path: Path,
) -> None:
    first = _write_report(
        tmp_path / "a.json",
        scene="a",
        miou=0.2,
        localization=0.5,
        sample_count=2,
    )
    second = _write_report(
        tmp_path / "b.json",
        scene="b",
        miou=0.8,
        localization=1.0,
        sample_count=6,
    )

    summary = summarize_reports(
        [first, second], expected_scenes={"a", "b"}
    )

    aggregate = summary["aggregate"]
    assert aggregate["scene_macro_miou"] == pytest.approx(0.5)
    assert aggregate["sample_micro_miou"] == pytest.approx(0.65)
    assert aggregate["scene_macro_localization_accuracy"] == pytest.approx(0.75)
    assert aggregate["sample_micro_localization_accuracy"] == pytest.approx(0.875)
    assert aggregate["sample_count"] == 8


def test_summarizer_rejects_coverage_protocol_mismatch(tmp_path: Path) -> None:
    conditional = _write_report(
        tmp_path / "conditional.json",
        scene="a",
        miou=0.2,
        localization=0.5,
        sample_count=2,
        coverage_power=0.0,
    )
    total_alpha = _write_report(
        tmp_path / "total_alpha.json",
        scene="b",
        miou=0.2,
        localization=0.5,
        sample_count=2,
        coverage_power=1.0,
    )

    with pytest.raises(ValueError, match="protocol mismatch"):
        summarize_reports([conditional, total_alpha])


def test_summarizer_rejects_duplicate_or_missing_scenes(tmp_path: Path) -> None:
    first = _write_report(
        tmp_path / "first.json",
        scene="a",
        miou=0.2,
        localization=0.5,
        sample_count=2,
    )
    duplicate = _write_report(
        tmp_path / "duplicate.json",
        scene="a",
        miou=0.3,
        localization=0.5,
        sample_count=2,
    )

    with pytest.raises(ValueError, match="duplicate"):
        summarize_reports([first, duplicate])
    with pytest.raises(ValueError, match="scene set mismatch"):
        summarize_reports([first], expected_scenes={"a", "b"})
