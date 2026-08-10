from __future__ import annotations

import torch

from radio_gs.losses.source_global_response_listwise_loss_v21b import (
    hard_negative_denominator_masks,
    source_global_response_listwise_loss_v21b,
)
from tests.test_source_global_response_listwise_loss_v21 import (
    ACCEPTED_SHA,
    FIT_SHA,
    TEACHER_CHANNEL_SHA,
    TEACHER_SHA,
    _fixture,
)


def test_hard_negative_masks_separate_pairwise_and_triplet_denominators() -> None:
    trainable = torch.tensor([True, False, False, False])
    anchors = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    negatives = torch.tensor([1, 0, 3, 2], dtype=torch.int64)
    pairwise, triplet = hard_negative_denominator_masks(
        trainable, anchors, negatives
    )
    assert pairwise.tolist() == [True, True, False, False]
    assert triplet.tolist() == [True, False, False, False]


def test_v21b_loss_reports_anchor_only_triplet_and_any_endpoint_pairwise(
    tmp_path,
) -> None:
    teacher, text, authority, canonical_negative = _fixture(tmp_path)
    student = teacher.clone().requires_grad_(True)
    total, metrics = source_global_response_listwise_loss_v21b(
        student.sum() * 0.0,
        student,
        teacher,
        torch.arange(4, dtype=torch.int64),
        text,
        torch.arange(4, dtype=torch.int64),
        authority,
        canonical_negative,
        accepted_v2_file_sha256=ACCEPTED_SHA,
        teacher_file_sha256=TEACHER_SHA,
        teacher_pair_descriptors_sha256=TEACHER_CHANNEL_SHA,
        fit_text_bank_file_sha256=FIT_SHA,
        trainable_region_mask=torch.tensor([True, False, False, False]),
    )
    assert metrics["authority_hard_negative_pairs"] == 4
    assert metrics["pairwise_objective_hard_negative_pairs"] == 2
    assert metrics["objective_hard_negative_pairs"] == 2
    assert metrics["triplet_objective_hard_negative_pairs"] == 1
    assert float(metrics["pair_trainable_endpoint_coverage"]) == 0.5
    assert float(metrics["triplet_anchor_trainable_coverage"]) == 0.25
    total.backward()
    assert student.grad is not None and float(student.grad.abs().sum()) > 0
