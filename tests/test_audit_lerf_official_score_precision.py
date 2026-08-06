from __future__ import annotations

import pytest
import torch

from radio_gs.evaluation.openclip_readout import cosine_logits
from radio_gs.scripts.audit_lerf_official_score_precision import (
    _rank_metrics,
    compile_fp32_scores,
    downstream_audit,
    error_summary,
    per_query_scale_audit,
)


def test_compile_fp32_scores_matches_shared_cosine_without_half_cast() -> None:
    generator = torch.Generator().manual_seed(19)
    descriptors = torch.randn(4, 3, 7, generator=generator).half()
    global_rows = torch.tensor([0, 2, 3, 5], dtype=torch.int64)
    text = torch.randn(2, 7, generator=generator)

    scores = compile_fp32_scores(
        descriptors,
        global_rows,
        text,
        total_rows=6,
        chunk_size=2,
    )

    assert scores.dtype == torch.float32
    assert scores.shape == (6, 3, 2)
    assert torch.count_nonzero(scores[[1, 4]]) == 0
    for scale in range(3):
        expected = cosine_logits(descriptors[:, scale], text)
        assert torch.equal(scores[global_rows, scale], expected)


def test_error_and_rank_metrics_report_exact_identity() -> None:
    values = torch.tensor([0.1, 0.4, -0.2, 0.3])
    errors = error_summary(values, values.clone())
    ranks = _rank_metrics(values, values.clone())

    assert errors["nonzero_elements"] == 0
    assert errors["max_absolute_error"] == 0.0
    assert ranks["exact_stable_rank_fraction"] == 1.0
    assert ranks["max_absolute_rank_displacement"] == 0
    assert ranks["stable_rank_spearman"] == pytest.approx(1.0)
    assert all(row["membership_flips"] == 0 for row in ranks["topk"].values())


def test_per_query_scale_audit_counts_strict_threshold_flips() -> None:
    reference = torch.zeros(3, 3, 1)
    candidate = reference.clone()
    reference[0, 1, 0] = 0.6001
    candidate[0, 1, 0] = 0.5999
    valid = torch.tensor([True, True, False])

    rows = per_query_scale_audit(
        candidate,
        reference,
        valid=valid,
        query_ids=["query"],
        threshold=0.6,
    )

    assert len(rows) == 3
    assert rows[1]["threshold"]["membership_flips"] == 1
    assert rows[0]["threshold"]["membership_flips"] == 0


def test_downstream_audit_proves_identical_selection_for_identical_scores() -> None:
    generator = torch.Generator().manual_seed(23)
    positive = torch.randn(12, 3, 2, generator=generator) * 0.08
    negative = torch.randn(12, 3, 4, generator=generator) * 0.08
    xyz = torch.randn(12, 3, generator=generator)
    valid = torch.ones(12, dtype=torch.bool)

    report = downstream_audit(
        positive,
        negative,
        positive.clone(),
        negative.clone(),
        xyz=xyz,
        valid=valid,
        query_ids=["a", "b"],
        knn_chunk_size=5,
    )

    vala = report["vala_knn10_peak_select"]
    assert vala["selected_scale_query_flips"] == 0
    assert vala["primitive_query_membership_flips"] == 0
    assert vala["primitive_query_membership_flip_fraction"] == 0.0
    assert vala["selection_bit_exact"] is True
    assert report["exact_evaluator_consequence"].startswith("identical:")
