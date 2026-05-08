import numpy as np
import torch

from radio_gs.scripts.eval_lerf_adaptor_downstream import (
    _parse_adaptor_names,
    build_masked_prototype,
    compute_prototype_heatmap,
    select_source_target_pairs,
)


def test_build_masked_prototype_averages_normalized_mask_tokens():
    feature_map = torch.zeros(2, 2, 2)
    feature_map[:, 0, 0] = torch.tensor([1.0, 0.0])
    feature_map[:, 0, 1] = torch.tensor([1.0, 0.0])
    feature_map[:, 1, 0] = torch.tensor([0.0, 1.0])
    mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)

    proto = build_masked_prototype(feature_map, mask)

    assert torch.allclose(proto, torch.tensor([1.0, 0.0]), atol=1e-6)


def test_compute_prototype_heatmap_peaks_inside_matching_region():
    feature_map = torch.zeros(2, 2, 2)
    feature_map[:, 0, 0] = torch.tensor([0.0, 1.0])
    feature_map[:, 0, 1] = torch.tensor([1.0, 0.0])
    feature_map[:, 1, 0] = torch.tensor([1.0, 0.0])
    feature_map[:, 1, 1] = torch.tensor([0.0, 1.0])
    proto = torch.tensor([1.0, 0.0])

    heatmap = compute_prototype_heatmap(feature_map, proto)

    assert heatmap[0, 1] > heatmap[0, 0]
    assert heatmap[1, 0] > heatmap[1, 1]


def test_select_source_target_pairs_uses_first_frame_as_source():
    frames_by_category = {
        "cup": [5, 9, 12],
        "plate": [2],
    }

    pairs = select_source_target_pairs(frames_by_category)

    assert pairs == [("cup", 5, 9), ("cup", 5, 12)]


def test_parse_adaptor_names_keeps_sam3_name_intact():
    assert _parse_adaptor_names("dino_v3,sam3") == ["dino_v3", "sam3"]
