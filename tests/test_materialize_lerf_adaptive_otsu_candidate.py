from __future__ import annotations

from radio_gs.scripts import materialize_lerf_adaptive_otsu_candidate as candidate


def test_candidate_access_is_target_blind_and_metric_closed() -> None:
    audit = candidate.access_audit()
    assert audit["parent_primitive_score_cache_opened"] is True
    assert audit["benchmark_images_opened"] is False
    assert audit["benchmark_masks_opened"] is False
    assert audit["benchmark_labels_opened"] is False
    assert audit["target_metrics_computed"] is False
    assert audit["scene_or_query_specific_parameters"] is False
