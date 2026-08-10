import copy
import json
from pathlib import Path

import pytest
import torch

from radio_gs.evaluation.lerf_source_text_response_ranking import (
    FRAME_EVALUATOR_IMPLEMENTATION,
    build_scene_summary,
    evaluate_source_frame,
)
from radio_gs.scripts.select_lerf_source_text_response_ranking_audit_confirmation import (
    PREREGISTRATION_SCHEMA,
    PREREGISTRATION_STATUS,
    file_sha256,
    select,
    validate_preregistration,
)


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _bank() -> dict[str, object]:
    return {
        "path": "/tmp/target_blind_audit90.pt",
        "sha256": "1" * 64,
        "manifest_path": "/tmp/target_blind_audit90.manifest.json",
        "manifest_sha256": "2" * 64,
        "query_split": "audit",
        "queries": 90,
        "embedding_tensor_sha256": "3" * 64,
    }


def _preregistration() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    selector = root / "radio_gs/scripts/select_lerf_source_text_response_ranking_audit_confirmation.py"
    evaluator = root / "radio_gs/evaluation/lerf_source_text_response_ranking.py"
    metric = root / "radio_gs/evaluation/text_response_fidelity.py"
    scenes = {
        "ramen": {
            "source_heldout_frame_ids": [2],
            "forbidden_target_frame_ids": [6],
        },
        "teatime": {
            "source_heldout_frame_ids": [4],
            "forbidden_target_frame_ids": [25],
        },
    }
    return {
        "schema": PREREGISTRATION_SCHEMA,
        "schema_version": 1,
        "status": PREREGISTRATION_STATUS,
        "selector_implementation": _record(selector),
        "frame_evaluator_implementation": _record(evaluator),
        "shared_response_metric_implementation": _record(metric),
        "source_scenes": scenes,
        "query_bank": _bank(),
        "decision_rule": {
            "one_global_policy": True,
            "required_source_scenes": ["ramen", "teatime"],
            "strict_pooled_improvements": [
                "response_mae_lower",
                "ranking_spearman_mean_higher",
                "top_decile_overlap_mean_higher",
            ],
            "every_scene_nonregression": [
                "response_mae",
                "response_profile_cosine_mean",
                "ranking_spearman_mean",
                "ranking_spearman_p05",
                "top_decile_overlap_mean",
                "top_decile_overlap_p05",
            ],
            "tolerance": 0.0,
            "fallback": "unchanged_control",
            "per_scene_or_per_query_thresholds": False,
        },
        "one_shot_confirmation": True,
        "development_selection_complete": True,
        "target_results_used_for_gate_design": False,
        "target_metric_execution_authorized": False,
    }


def _scene(scene_id: str, frame_id: int, *, candidate: bool) -> dict[str, object]:
    y, x = torch.meshgrid(
        torch.linspace(0.1, 1.0, 4),
        torch.linspace(0.2, 1.1, 4),
        indexing="ij",
    )
    teacher = torch.stack((x, y, 0.25 + x + 0.5 * y), dim=0)
    method = teacher if candidate else teacher.flip((1, 2)).contiguous()
    base = torch.eye(3)
    text = base[torch.arange(90) % 3] + (torch.arange(90).float()[:, None] + 1) * 1e-4
    queries = [f"audit-query-{index:03d}" for index in range(90)]
    method_id = "candidate" if candidate else "control"
    frame = evaluate_source_frame(
        method,
        teacher,
        text,
        torch.ones(4, 4, dtype=torch.bool),
        scene_id=scene_id,
        frame_id=frame_id,
        method_id=method_id,
        query_ids=queries,
        query_bank=_bank(),
    )
    return build_scene_summary(
        [frame],
        scene_id=scene_id,
        method_id=method_id,
        required_frame_ids=[frame_id],
    )


def _write_summary(path: Path, value: object) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path.resolve())


def test_confirmation_requires_exact_audit90_preregistration() -> None:
    valid = _preregistration()
    assert validate_preregistration(valid)["query_bank"] == _bank()
    for key, replacement in (("query_split", "dev"), ("queries", 89)):
        invalid = copy.deepcopy(valid)
        invalid["query_bank"][key] = replacement
        with pytest.raises(ValueError, match="reserved audit-90"):
            validate_preregistration(invalid)
    invalid_schema = copy.deepcopy(valid)
    invalid_schema["schema"] = "radio_gs.lerf_source_text_response_ranking_preregistration.v1"
    with pytest.raises(ValueError, match="preregistration schema"):
        validate_preregistration(invalid_schema)


def test_confirmation_reuses_strict_paired_gate_and_never_authorizes_target(
    tmp_path: Path,
) -> None:
    preregistration = validate_preregistration(_preregistration())
    controls = {
        "ramen": _scene("ramen", 2, candidate=False),
        "teatime": _scene("teatime", 4, candidate=False),
    }
    candidates = {
        "ramen": _scene("ramen", 2, candidate=True),
        "teatime": _scene("teatime", 4, candidate=True),
    }
    control_values = [
        f"{scene}={_write_summary(tmp_path / f'{scene}_control.json', summary)}"
        for scene, summary in controls.items()
    ]
    candidate_values = [
        f"{scene}={_write_summary(tmp_path / f'{scene}_candidate.json', summary)}"
        for scene, summary in candidates.items()
    ]
    result = select(
        preregistration,
        {"path": "/tmp/audit90-prereg.json", "sha256": "a" * 64},
        control_values=control_values,
        candidate_values=candidate_values,
    )
    assert result["status"] == "passed"
    assert result["query_bank"]["query_split"] == "audit"
    assert result["query_bank"]["queries"] == 90
    assert result["confirmation_query_split"] == "audit"
    assert result["confirmation_query_count"] == 90
    assert result["decision"] == {
        "strict_pooled_response_ranking_and_top_decile_improvement": True,
        "every_scene_all_metrics_nonregressing": True,
        "candidate_eligible_for_next_source_gate": True,
    }
    assert all(row["all_metrics_nonregressing"] for row in result["scene_results"])
    assert result["protocol"]["target_metric_execution_authorized"] is False
    assert result["metric_execution_authorized"] is False
    assert result["metric_executed"] is False


def test_confirmation_rejects_summary_on_dev_axis(tmp_path: Path) -> None:
    preregistration = validate_preregistration(_preregistration())
    control = _scene("ramen", 2, candidate=False)
    candidate = _scene("ramen", 2, candidate=True)
    control["query_bank"] = {**control["query_bank"], "query_split": "dev"}
    for frame in control["frames"]:
        frame["query_bank"] = {**frame["query_bank"], "query_split": "dev"}
    control_values = [
        f"ramen={_write_summary(tmp_path / 'ramen_control.json', control)}",
        f"teatime={_write_summary(tmp_path / 'teatime_control.json', _scene('teatime', 4, candidate=False))}",
    ]
    candidate_values = [
        f"ramen={_write_summary(tmp_path / 'ramen_candidate.json', candidate)}",
        f"teatime={_write_summary(tmp_path / 'teatime_candidate.json', _scene('teatime', 4, candidate=True))}",
    ]
    with pytest.raises(ValueError, match="query bank differs"):
        select(
            preregistration,
            {"path": "/tmp/audit90-prereg.json", "sha256": "a" * 64},
            control_values=control_values,
            candidate_values=candidate_values,
        )


def test_preregistration_binds_current_frame_evaluator() -> None:
    checked = validate_preregistration(_preregistration())
    assert checked["frame_evaluator_implementation"] == FRAME_EVALUATOR_IMPLEMENTATION
