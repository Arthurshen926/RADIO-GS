from types import SimpleNamespace

import torch

from radio_gs.v3.memory.structured_memory import StructuredSharedPrivateMemory
from radio_gs.v3.training.fit_clean_parent_instance import (
    TextMaskAuthority,
    compact_text_mask_objective,
)


def test_text_mask_objective_backpropagates_through_exact_deployment_order() -> None:
    memory = torch.randn(4, 512)
    model = StructuredSharedPrivateMemory(memory)
    with torch.no_grad():
        model.scale_adapter.weight.zero_()
        model.scale_adapter.bias.zero_()
    model.enable_owned_training_blocks("instance")
    episode = SimpleNamespace(
        scale=0.5,
        gaussian_ids=torch.tensor([0, 1, 2, 3]),
        pixel_ids=torch.tensor([0, 1, 2, 3]),
        contribution_weights=torch.ones(4),
        target=torch.tensor([[True, False], [False, False]]),
    )
    item = TextMaskAuthority(
        query_name="object",
        anchor_rows=torch.tensor([0]),
        anchor_weights=torch.tensor([1.0]),
        representative=episode,
        positive_masks=(torch.tensor([[True, False], [False, False]]),),
        negative_masks=(torch.tensor([[False, True], [False, False]]),),
    )
    loss = compact_text_mask_objective(
        model, item, temperature=0.15, unknown_growth_weight=0.25
    )
    loss.backward()
    gradient = model.owned_training_parameter("instance").grad
    assert torch.isfinite(loss)
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
