import torch

from radio_gs.scripts.build_sam_mask_aligned_siglip2_spatial_teacher import (
    pool_mask_aligned_spatial_descriptors,
)


def test_pool_mask_aligned_spatial_descriptors_separates_foreground_context():
    features = torch.zeros(2, 2, 2)
    features[0, :, 0] = 3.0
    features[1, :, 1] = 4.0
    masks = torch.zeros(1, 4, 4, dtype=torch.bool)
    masks[:, :, :2] = True
    boxes = torch.tensor([[0, 0, 4, 4]])
    foreground, context, agreement = pool_mask_aligned_spatial_descriptors(
        features, masks, boxes, image_height=4, image_width=4
    )
    assert torch.allclose(foreground, torch.tensor([[1.0, 0.0]]), atol=1e-6)
    assert torch.allclose(context, torch.tensor([[0.0, 1.0]]), atol=1e-6)
    assert torch.allclose(agreement, torch.zeros(1), atol=1e-6)


def test_pool_mask_aligned_spatial_descriptors_falls_back_for_no_shell():
    features = torch.tensor([[[2.0]], [[0.0]]])
    masks = torch.ones(1, 2, 2, dtype=torch.bool)
    boxes = torch.tensor([[0, 0, 2, 2]])
    foreground, context, agreement = pool_mask_aligned_spatial_descriptors(
        features, masks, boxes, image_height=2, image_width=2
    )
    assert torch.equal(foreground, context)
    assert torch.allclose(agreement, torch.ones(1))
