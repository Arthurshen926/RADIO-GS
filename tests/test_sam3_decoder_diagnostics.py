import numpy as np
import torch

from radio_gs.scripts.diagnose_sam3_decoder_state import (
    best_output_mask,
    binary_mask_iou,
    clone_sam3_state,
    mask_to_sam3_box_prompt,
    output_mask_summary,
    sam3_box_prompt_to_xyxy_pixels,
    summarize_tensors,
)


def test_clone_sam3_state_clones_nested_tensors_without_aliasing():
    state = {
        "original_height": 20,
        "backbone_out": {
            "vision_features": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
            "backbone_fpn": [torch.ones(1, 1), torch.zeros(1, 1)],
        },
    }

    cloned = clone_sam3_state(state)
    cloned["backbone_out"]["vision_features"][0, 0, 0] = -1
    cloned["backbone_out"]["backbone_fpn"][0][0, 0] = 7

    assert state["backbone_out"]["vision_features"][0, 0, 0].item() == 0
    assert state["backbone_out"]["backbone_fpn"][0][0, 0].item() == 1
    assert cloned["backbone_out"]["vision_features"].device == state["backbone_out"]["vision_features"].device
    assert cloned["backbone_out"]["vision_features"].dtype == state["backbone_out"]["vision_features"].dtype


def test_mask_to_sam3_box_prompt_uses_normalized_cxcywh_and_roundtrips():
    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[5:15, 10:30] = 1

    box = mask_to_sam3_box_prompt(mask, padding_pixels=2)
    xyxy = sam3_box_prompt_to_xyxy_pixels(box, height=20, width=40)

    assert np.allclose(box, [0.5, 0.5, 0.6, 0.7], atol=1e-6)
    assert np.allclose(xyxy, [8.0, 3.0, 32.0, 17.0], atol=1e-6)


def test_mask_to_sam3_box_prompt_returns_none_for_empty_mask():
    assert mask_to_sam3_box_prompt(np.zeros((8, 8), dtype=np.uint8)) is None


def test_binary_mask_iou_treats_two_empty_masks_as_identity_match():
    empty = np.zeros((4, 4), dtype=bool)
    one_pixel = empty.copy()
    one_pixel[1, 1] = True

    assert binary_mask_iou(empty, empty) == 1.0
    assert binary_mask_iou(empty, one_pixel) == 0.0


def test_summarize_tensors_records_shape_dtype_and_device_for_nested_state():
    state = {
        "backbone_out": {
            "vision_features": torch.zeros(1, 256, 72, 72, dtype=torch.float16),
            "backbone_fpn": [torch.ones(1, 256, 144, 144)],
        },
        "scores": torch.tensor([0.25, 0.75]),
    }

    summary = summarize_tensors(state)

    assert summary["backbone_out.vision_features"]["shape"] == [1, 256, 72, 72]
    assert summary["backbone_out.vision_features"]["dtype"] == "torch.float16"
    assert summary["backbone_out.backbone_fpn[0]"]["shape"] == [1, 256, 144, 144]
    assert summary["scores"]["device"] == "cpu"


def test_output_mask_helpers_accept_bfloat16_logits_and_scores():
    output = {
        "masks_logits": torch.tensor(
            [[[[0.1, 0.6], [0.7, 0.2]]]],
            dtype=torch.bfloat16,
        ),
        "scores": torch.tensor([0.25], dtype=torch.bfloat16),
    }

    mask = best_output_mask(output, height=2, width=2)
    summary = output_mask_summary(output, height=2, width=2)

    assert mask.tolist() == [[False, True], [True, False]]
    assert summary["best_area"] == 2
    assert np.isclose(summary["best_score"], 0.25, atol=1e-3)
