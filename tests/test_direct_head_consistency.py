from __future__ import annotations

import pytest
import torch

from radio_gs.scripts.diagnose_direct_head_consistency import (
    build_adapter_metadata_status,
    compute_cosine_stats,
    compute_rank_agreement,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    build_direct_head_eval_status,
    enforce_direct_head_eval_consistency,
)


def test_compute_cosine_stats_reports_mean_and_percentiles():
    pred = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    target = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ]
    )

    stats = compute_cosine_stats(pred, target)

    assert stats["mean_cos_to_vpr"] == pytest.approx(0.25)
    assert stats["p10"] == pytest.approx(-0.7)
    assert stats["p50"] == pytest.approx(0.5)
    assert stats["p90"] == pytest.approx(1.0)
    assert stats["count"] == 4


def test_compute_rank_agreement_uses_pairwise_text_ordering():
    teacher_scores = torch.tensor(
        [
            [3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0],
        ]
    )
    student_scores = torch.tensor(
        [
            [3.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],
        ]
    )

    stats = compute_rank_agreement(student_scores, teacher_scores)

    assert stats["text_rank_agreement"] == pytest.approx(0.5)
    assert stats["text_rank_pairs"] == 6


def test_adapter_metadata_warns_when_checkpoint_adapter_is_not_enabled():
    checkpoint = {"point_summary_adapter_state_dict": {"net.0.weight": torch.ones(1)}}

    status = build_adapter_metadata_status(
        checkpoint,
        use_point_summary_adapter=False,
        adapter_loaded=False,
    )

    assert status["checkpoint_has_point_summary_adapter"] is True
    assert status["adapter_loaded"] is False
    assert status["eval_adapter_disabled_with_checkpoint_adapter"] is True
    assert (
        "checkpoint_has_point_summary_adapter_but_use_point_summary_adapter_is_false"
        in status["metadata_warnings"]
    )


def test_eval_direct_head_strict_consistency_rejects_disabled_adapter():
    checkpoint = {"point_summary_adapter_state_dict": {"net.0.weight": torch.ones(1)}}
    status = build_direct_head_eval_status(
        checkpoint,
        score_source="direct",
        use_point_summary_adapter=False,
        adapter_loaded=False,
    )

    assert "checkpoint_has_point_summary_adapter_but_eval_disabled" in status["warnings"]
    with pytest.raises(ValueError, match="direct head consistency"):
        enforce_direct_head_eval_consistency(status, strict=True)


def test_eval_direct_head_strict_consistency_ignores_registered_view():
    checkpoint = {"point_summary_adapter_state_dict": {"net.0.weight": torch.ones(1)}}
    status = build_direct_head_eval_status(
        checkpoint,
        score_source="registered_view",
        use_point_summary_adapter=False,
        adapter_loaded=False,
    )

    assert status["warnings"] == []
    enforce_direct_head_eval_consistency(status, strict=True)
