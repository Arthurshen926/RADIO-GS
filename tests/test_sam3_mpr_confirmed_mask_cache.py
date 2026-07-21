from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from radio_gs.scripts.build_sam3_mpr_confirmed_mask_cache import (
    _deduplicate_target_rows,
    _predict_target_masks,
    neighbouring_frames,
    select_target_frame_shard,
    select_visible_anchor_pixel,
    SourceMask,
)


def test_neighbouring_frames_preserves_frozen_mpr_order():
    assert neighbouring_frames(30, [10, 20, 30, 40, 50], per_direction=2) == [20, 40, 10, 50]
    assert neighbouring_frames(10, [10, 20, 30], per_direction=2) == [20, 30]


def test_target_sharding_is_disjoint_and_recovers_the_frozen_order():
    frames = [100, 20, 0, 60, 20]
    shards = [select_target_frame_shard(frames, shard_index=index, shard_count=3) for index in range(3)]
    assert shards == [[0, 100], [20], [60]]
    assert sorted(frame for shard in shards for frame in shard) == [0, 20, 60, 100]


def test_target_dedup_keeps_distinct_source_anchor_associations() -> None:
    mask = np.array([[True, True], [False, False]])
    rows = [
        {"mask": mask, "score": 0.9, "prompt_index": 0},
        {"mask": mask.copy(), "score": 0.8, "prompt_index": 0},
        {"mask": mask.copy(), "score": 0.95, "prompt_index": 1},
    ]
    args = SimpleNamespace(nms_iou=0.85, duplicate_minimum_area_ratio=0.9, maximum_masks_per_target=0)
    kept = _deduplicate_target_rows(rows, args)
    # The two same-prompt candidates are a duplicate, whereas prompt 1 carries
    # a different source--target track and must remain even with identical 2-D
    # support.
    assert [row["prompt_index"] for row in kept] == [0, 1]


def test_official_prompt_path_preserves_duplicate_target_mask_across_anchors() -> None:
    class FakeProcessor:
        def __init__(self):
            self.model = self

        def set_image(self, image):
            return image.size

        def predict_inst(self, state, *, point_coords, point_labels, multimask_output):
            assert state == (4, 4) and multimask_output
            return np.ones((1, 4, 4), dtype=bool), np.array([0.9], dtype=np.float32), None

    source = [
        SourceMask(0, 0, torch.tensor([0.9]), torch.tensor([True]), 0.9, 1.0),
        SourceMask(0, 1, torch.tensor([0.9]), torch.tensor([True]), 0.9, 1.0),
    ]
    anchor = {
        "xy": [1.0, 1.0], "feature_xy": [0, 0], "local_primitive": 0,
        "global_primitive": 7, "source_membership": 0.9, "mpr_weight": 0.8,
    }
    args = SimpleNamespace(
        minimum_quality=0.7, minimum_stability=0.0, stability_offset=1.0,
        minimum_area_fraction=0.001, maximum_area_fraction=1.0,
        nms_iou=0.85, duplicate_minimum_area_ratio=0.9, maximum_masks_per_target=0,
    )
    rows, logits_available, raw_candidates = _predict_target_masks(
        FakeProcessor(), Image.new("RGB", (4, 4)), [(source[0], anchor), (source[1], anchor)], args,
    )
    assert raw_candidates == 2 and not logits_available
    assert [row["source_mask_index"] for row in rows] == [0, 1]


def test_select_visible_anchor_uses_confident_local_membership_and_mpr_weight():
    # Global primitives 0 and 3 are intentionally absent from the canonical
    # graph.  The selected MPR row must therefore be a valid explicit bridge,
    # never a nearest-neighbour substitute.
    anchor = select_visible_anchor_pixel(
        torch.tensor([0.85, 0.95]),
        assignment={
            "gaussian_ids": torch.tensor([0, 1, 2, 3]),
            "pixel_ids": torch.tensor([0, 1, 2, 3]),
            "weights": torch.tensor([0.99, 0.20, 0.50, 1.0]),
        },
        global_to_local=torch.tensor([-1, 0, 1, -1]),
        feature_height=2,
        feature_width=2,
        image_height=20,
        image_width=40,
        inside_threshold=0.80,
    )
    # local primitive 2 wins (0.95 * 0.50), while primitive 0/3 cannot be
    # resurrected because they are not canonical graph rows.
    assert anchor is not None
    assert anchor["feature_xy"] == [0, 1]
    assert anchor["xy"] == [10.0, 15.0]
    assert anchor["local_primitive"] == 1
    assert anchor["global_primitive"] == 2
    assert anchor["source_membership"] == pytest.approx(0.95)
    assert anchor["mpr_weight"] == pytest.approx(0.5)


def test_select_visible_anchor_refuses_unconfident_source_support():
    anchor = select_visible_anchor_pixel(
        torch.tensor([0.79]),
        assignment={
            "gaussian_ids": torch.tensor([0]),
            "pixel_ids": torch.tensor([0]),
            "weights": torch.tensor([1.0]),
        },
        global_to_local=torch.tensor([0]),
        feature_height=1,
        feature_width=1,
        image_height=10,
        image_width=10,
        inside_threshold=0.80,
    )
    assert anchor is None
