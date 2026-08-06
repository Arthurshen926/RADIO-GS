from __future__ import annotations

import torch

from radio_gs.scripts.train_registered_evidence_to_unary_clean_pilot import (
    SparseView,
    _binary_metrics,
    instance_split,
    prompt_masks,
    select_source_frame,
)


def _view() -> SparseView:
    image = torch.zeros(25, dtype=torch.long)
    image[[6, 7, 11, 12]] = 4
    return SparseView(
        frame_id=9,
        gaussian_ids=torch.arange(25),
        pixel_ids=torch.arange(25),
        weights=torch.ones(25),
        pixel_mass=torch.ones(25),
        instance_image=image,
    )


def test_frozen_split_and_source_frame_are_deterministic() -> None:
    assert instance_split(1) == "validation"
    assert instance_split(2) == "train"
    frames = [9, 3, 15, 4]
    assert select_source_frame(7, frames) == select_source_frame(7, reversed(frames))


def test_prompt_masks_keep_full_and_sparse_evidence_distinct() -> None:
    view = _view()
    positive, negative = prompt_masks(
        view=view, instance_id=4, mode="full_mask", height=5, width=5
    )
    assert int(positive.sum()) == 4
    assert int(negative.sum()) == 21
    sparse_positive, sparse_negative = prompt_masks(
        view=view, instance_id=4, mode="scribble", height=5, width=5
    )
    assert torch.equal(sparse_positive, positive)
    assert not bool((sparse_positive & sparse_negative).any())
    assert int(sparse_negative.sum()) == 21


def test_binary_metrics_report_perfect_ranking_and_fixed_mask() -> None:
    metrics = _binary_metrics(
        torch.tensor([0.9, 0.8, 0.2, 0.1]),
        torch.tensor([True, True, False, False]),
    )
    assert metrics["average_precision"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["iou_at_0_5"] == 1.0
    assert metrics["precision_at_0_5"] == 1.0
    assert metrics["recall_at_0_5"] == 1.0
    assert metrics["area_ratio"] == 1.0

