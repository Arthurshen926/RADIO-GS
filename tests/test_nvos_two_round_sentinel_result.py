from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    REPO_ROOT
    / "paper/artifacts/"
    "nvos_method_v1_two_round_exact_consensus_sam3_fern_sentinel_result_20260817.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_failed_sentinel_result_is_closed_and_not_promoted() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "completed_failed_sentinel"
    assert result["promotion"] is False
    assert result["full8_executed"] is False
    assert result["scene_id"] == "fern"
    assert result["eligibility"] == {
        "development_use_only": True,
        "sota_claim": False,
        "strict_unseen_protocol_eligible": False,
        "target_rgb_assisted": True,
    }
    assert result["safety"]["parameters_changed_after_sentinel_metric"] is False
    assert result["safety"]["target_metric_used_for_prediction_or_selection"] is False
    assert result["safety"]["full8_target_masks_opened"] is False


def test_loss_decomposition_and_trials_match_stored_metrics() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    metrics = result["paired_fern_metrics"]
    one_shot = metrics["sealed_one_shot_box_point_supermajority_parent"][
        "foreground_iou"
    ]
    rerender = metrics["exact_W_transpose_W_rerender_threshold"]["foreground_iou"]
    round2 = metrics["round2_mask_prompt_sam3"]["foreground_iou"]
    losses = result["loss_decomposition_foreground_iou"]
    assert math.isclose(losses["exact_W_transpose_W_minus_one_shot"], rerender - one_shot)
    assert math.isclose(
        losses["round2_self_prompt_minus_exact_rerender"], round2 - rerender
    )
    assert math.isclose(losses["round2_total_minus_one_shot"], round2 - one_shot)
    assert losses["decomposition_closes"] is True
    assert rerender < one_shot and round2 < rerender

    stability = result["round2_trial_stability"]
    trial_ious = [float(row["foreground_iou"]) for row in stability["trials"]]
    assert len(trial_ious) == stability["trial_count"] == 10
    assert all(value < one_shot for value in trial_ious)
    assert math.isclose(statistics.fmean(trial_ious), stability["foreground_iou_mean"])
    assert math.isclose(
        statistics.pstdev(trial_ious), stability["foreground_iou_population_std"]
    )
    assert math.isclose(max(trial_ious) - min(trial_ious), stability["foreground_iou_range"])


def test_repository_implementation_hashes_remain_bound() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    preregistration = result["preregistration"]
    preregistration_path = Path(preregistration["path"])
    assert preregistration_path == (
        REPO_ROOT
        / "paper/artifacts/"
        "nvos_method_v1_two_round_exact_consensus_sam3_preregistration_20260817.json"
    )
    assert _sha256(preregistration_path) == preregistration["sha256"]
    for record in result["implementation"].values():
        path = Path(record["path"])
        assert path.is_file()
        assert _sha256(path) == record["sha256"]

