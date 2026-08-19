import numpy as np

from radio_gs.scripts.predict_nvos_joint_box_signed_points_sam3 import (
    padded_mask_box_xyxy,
)


def test_padded_mask_box_xyxy_is_clipped_and_uses_exclusive_maximum() -> None:
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:5, 3:7] = True
    np.testing.assert_array_equal(
        padded_mask_box_xyxy(mask, padding=2),
        np.asarray([1, 0, 9, 7], dtype=np.float32),
    )


def test_padded_mask_box_xyxy_returns_none_for_empty_support() -> None:
    assert padded_mask_box_xyxy(np.zeros((3, 4), dtype=bool), padding=1) is None
