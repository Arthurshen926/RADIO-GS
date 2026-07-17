import numpy as np

from radio_gs.scripts.build_sam3_automatic_mask_cache import (
    mask_nms,
    pack_masks,
    unpack_masks,
)


def test_pack_masks_is_lossless_for_non_byte_aligned_width() -> None:
    masks = np.random.default_rng(0).integers(0, 2, size=(3, 7, 13), dtype=np.uint8)
    restored = unpack_masks(pack_masks(masks), width=13)
    assert np.array_equal(restored, masks.astype(bool))


def test_mask_nms_keeps_best_duplicate_and_distinct_region() -> None:
    first = np.zeros((8, 8), dtype=bool); first[:4, :4] = True
    duplicate = first.copy()
    distinct = np.zeros((8, 8), dtype=bool); distinct[4:, 4:] = True
    kept = mask_nms(
        [first, duplicate, distinct], [0.8, 0.9, 0.7], threshold=0.8, maximum=8
    )
    assert kept == [1, 2]
