import json
from pathlib import Path

import pytest
import torch

from radio_gs.evaluation.lerf_source_text_response_ranking import (
    build_scene_summary,
    evaluate_source_frame,
    evaluate_source_response_frame,
    paired_source_gate,
)
from radio_gs.scripts.select_lerf_source_text_response_ranking_gate import (
    file_sha256,
    validate_preregistration,
)


def _query_bank(query_count: int) -> dict[str, object]:
    return {
        "path": "/tmp/target_blind_dev.pt",
        "sha256": "1" * 64,
        "manifest_path": "/tmp/target_blind_dev.manifest.json",
        "manifest_sha256": "2" * 64,
        "query_split": "dev",
        "queries": query_count,
        "embedding_tensor_sha256": "3" * 64,
    }


def _maps() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y, x = torch.meshgrid(
        torch.linspace(0.1, 1.0, 4),
        torch.linspace(0.2, 1.1, 4),
        indexing="ij",
    )
    teacher = torch.stack((x, y, 0.25 + x + 0.5 * y), dim=0)
    control = teacher.flip((1, 2)).contiguous()
    candidate = teacher.clone()
    return control, candidate, teacher


def _frame(
    *, scene_id: str, frame_id: int, method_id: str, candidate: bool
) -> dict[str, object]:
    control, exact, teacher = _maps()
    text = torch.eye(3)
    return evaluate_source_frame(
        exact if candidate else control,
        teacher,
        text,
        torch.ones(4, 4, dtype=torch.bool),
        scene_id=scene_id,
        frame_id=frame_id,
        method_id=method_id,
        query_ids=["generic-a", "generic-b", "generic-c"],
        query_bank=_query_bank(3),
    )


def _scene(scene_id: str, frame_id: int, *, candidate: bool) -> dict[str, object]:
    method_id = "candidate" if candidate else "control"
    return build_scene_summary(
        [
            _frame(
                scene_id=scene_id,
                frame_id=frame_id,
                method_id=method_id,
                candidate=candidate,
            )
        ],
        scene_id=scene_id,
        method_id=method_id,
        required_frame_ids=[frame_id],
    )


def test_exact_candidate_passes_paired_two_scene_response_ranking_gate() -> None:
    controls = [_scene("ramen", 2, candidate=False), _scene("teatime", 4, candidate=False)]
    candidates = [_scene("ramen", 2, candidate=True), _scene("teatime", 4, candidate=True)]
    result = paired_source_gate(
        controls,
        candidates,
        required_scene_ids=["ramen", "teatime"],
    )
    assert result["status"] == "passed"
    assert result["decision"] == {
        "strict_pooled_response_ranking_and_top_decile_improvement": True,
        "every_scene_all_metrics_nonregressing": True,
        "candidate_eligible_for_next_source_gate": True,
    }
    assert result["pooled"]["deltas"]["response_mae_improvement"] > 0
    assert result["pooled"]["deltas"]["ranking_spearman_mean_delta"] > 0
    assert result["protocol"]["target_metric_execution_authorized"] is False


def test_gate_rejects_unpaired_teacher_or_single_scene() -> None:
    controls = [
        _scene("ramen", 2, candidate=False),
        _scene("teatime", 4, candidate=False),
    ]
    candidates = [
        _scene("ramen", 2, candidate=True),
        _scene("teatime", 4, candidate=True),
    ]
    candidates[0]["frames"][0]["teacher_response_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="teacher_response_sha256 differs"):
        paired_source_gate(
            controls,
            candidates,
            required_scene_ids=["ramen", "teatime"],
        )
    with pytest.raises(ValueError, match=">=2"):
        paired_source_gate(
            [controls[0]],
            [_scene("ramen", 2, candidate=True)],
            required_scene_ids=["ramen"],
        )


def test_precomputed_response_path_matches_descriptor_path() -> None:
    _control, method, teacher = _maps()
    text = torch.eye(3)
    mask = torch.ones(4, 4, dtype=torch.bool)
    direct = evaluate_source_frame(
        method,
        teacher,
        text,
        mask,
        scene_id="ramen",
        frame_id=2,
        method_id="candidate",
        query_ids=["generic-a", "generic-b", "generic-c"],
        query_bank=_query_bank(3),
    )
    method_response = torch.einsum(
        "qd,dhw->qhw", torch.nn.functional.normalize(text, dim=-1),
        torch.nn.functional.normalize(method, dim=0),
    )
    teacher_response = torch.einsum(
        "qd,dhw->qhw", torch.nn.functional.normalize(text, dim=-1),
        torch.nn.functional.normalize(teacher, dim=0),
    )
    projected = evaluate_source_response_frame(
        method_response,
        teacher_response,
        mask,
        scene_id="ramen",
        frame_id=2,
        method_id="candidate",
        query_ids=["generic-a", "generic-b", "generic-c"],
        query_bank=_query_bank(3),
        method_input_sha256="a" * 64,
        teacher_input_sha256="b" * 64,
    )
    assert projected["method_response_sha256"] == direct["method_response_sha256"]
    assert projected["teacher_response_sha256"] == direct["teacher_response_sha256"]
    assert projected["sufficient_statistics"] == direct["sufficient_statistics"]
    assert projected["query_units"] == direct["query_units"]
def test_production_preregistration_binds_real_generic_bank_and_missing_gap() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "paper/artifacts/lerf_source_text_response_ranking_preregistration_20260809.json"
    )
    preregistration = validate_preregistration(
        json.loads(path.read_text(encoding="utf-8"))
    )
    bank = preregistration["query_bank"]
    assert bank["queries"] == 101
    assert bank["query_split"] == "dev"
    assert Path(bank["path"]).stat().st_size == 631176
    assert file_sha256(bank["manifest_path"]) == bank["manifest_sha256"]
    assert set(preregistration["source_scenes"]) == {"ramen", "teatime"}
    assert preregistration["source_response_asset_status"] == {
        "query_bank": "present_reusable",
        "legal_source_heldout_teacher_maps": "present_formally_content_resealed",
        "paired_control_candidate_response_summaries": "absent_requires_materialization",
        "gate_execution_now": "blocked_fail_closed",
    }
    assert preregistration["target_metric_execution_authorized"] is False
