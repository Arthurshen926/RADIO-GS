import numpy as np

from radio_gs.scripts.build_sam3_automatic_mask_cache import (
    containment_aware_deduplicate,
    mask_stability,
)


def test_containment_aware_dedup_keeps_a_nested_scale() -> None:
    large = np.zeros((10, 10), dtype=bool); large[1:9, 1:9] = True
    nested = np.zeros((10, 10), dtype=bool); nested[2:8, 2:8] = True
    duplicate = large.copy()
    kept = containment_aware_deduplicate(
        [large, nested, duplicate], [0.90, 0.80, 0.85],
        iou_threshold=0.85, minimum_area_ratio=0.90, maximum=0,
    )
    assert kept == [0, 1]


def test_mask_stability_uses_two_logit_thresholds_and_falls_back_neutrally() -> None:
    masks = np.zeros((2, 2, 2), dtype=bool)
    masks[0] = True; masks[1, 0, 0] = True
    logits = np.array([
        [[2.0]],
        [[1.2]],
    ])
    stable = mask_stability(logits, masks, offset=1.0)
    np.testing.assert_allclose(stable, [1.0, 1.0])
    neutral = mask_stability(None, masks, offset=1.0)
    np.testing.assert_allclose(neutral, [1.0, 1.0])
