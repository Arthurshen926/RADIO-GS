import numpy as np
from PIL import Image
import pytest
import torch

from radio_gs.scripts.build_sam3_automatic_mask_cache import (
    mask_nms,
    pack_masks,
    unpack_masks,
    validate_automatic_mask_source_binding,
)
from radio_gs.utils.immutable_artifacts import sha256_file


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


def test_automatic_cache_resume_requires_source_image_sha256(tmp_path) -> None:
    image_path = tmp_path / "frame_00001.png"
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(image_path)
    payload = {
        "metadata": {
            "source": "official_sam3_interactive_grid_multimask_hierarchy",
            "official_decoder": True,
            "query_free": True,
            "image": str(image_path.resolve()),
        },
        "scores": torch.empty(0),
    }
    with pytest.raises(ValueError, match="source binding differs"):
        validate_automatic_mask_source_binding(payload, image_path)
    payload["metadata"]["source_image_sha256"] = sha256_file(image_path)
    assert validate_automatic_mask_source_binding(payload, image_path) == sha256_file(
        image_path
    )
