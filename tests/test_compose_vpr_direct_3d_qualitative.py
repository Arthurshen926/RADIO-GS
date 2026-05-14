import cv2
import numpy as np

from radio_gs.scripts import compose_vpr_direct_3d_qualitative as compose


def test_load_pred_mask_uses_requested_selection(tmp_path):
    mask_dir = tmp_path / "pred_masks" / "meanstd2p5" / "figurines"
    mask_dir.mkdir(parents=True)
    mask_path = mask_dir / "frame_00001_green apple.png"
    cv2.imwrite(str(mask_path), np.asarray([[0, 255]], dtype=np.uint8))

    mask = compose.load_pred_mask(tmp_path, "meanstd2p5", "figurines", "00001", "green apple")

    assert mask.tolist() == [[False, True]]
