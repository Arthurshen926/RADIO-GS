from __future__ import annotations

import torch

from radio_gs.scripts import (
    materialize_lerf_o1_native_v3_scale_aware_fusion as materializer,
)
from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen


def test_frozen_multiscale_relevance_matches_evaluator_readout() -> None:
    positive = torch.tensor(
        [
            [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
            [[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]],
            [[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]],
            [[0.2, 0.8], [0.3, 0.7], [0.1, 0.9]],
        ],
        dtype=torch.float32,
    )
    negative = torch.zeros((4, 3, 4), dtype=torch.float32)
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    valid = torch.ones(4, dtype=torch.bool)
    per_scale, selected, peaks = materializer.frozen_multiscale_relevance(
        positive_scores=positive,
        negative_scores=negative,
        xyz=xyz,
        valid=valid,
        chunk_size=2,
    )
    probability = frozen.canonical_negative_relevancy_query_scores(
        positive, negative, logit_scale=materializer.LOGIT_SCALE
    )
    expected = frozen.vala_multiscale_knn_peak_select_scores(
        probability,
        xyz,
        k=materializer.KNN_K,
        chunk_size=2,
        valid_mask=valid,
    )
    query = torch.arange(positive.shape[2])
    gathered = per_scale[:, selected, query]
    assert torch.equal(selected, expected.selected_scale_indices)
    assert torch.equal(peaks, expected.raw_smoothed_peaks)
    assert torch.equal(gathered, expected.scores)


def test_access_audit_is_explicitly_cpu_premetric() -> None:
    audit = materializer.access_audit()
    assert audit["benchmark_images_opened"] is False
    assert audit["benchmark_masks_opened"] is False
    assert audit["benchmark_labels_opened"] is False
    assert audit["target_metrics_computed"] is False
    assert audit["gpu_used"] is False
