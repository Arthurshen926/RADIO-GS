from __future__ import annotations

import numpy as np
import torch

from radio_gs.scripts.build_nvos_two_round_exact_consensus import (
    exact_adjoint_probability,
    exact_forward_probability,
    robust_logodds_consensus,
)
from radio_gs.scripts.predict_nvos_two_round_mask_prompt_sam3 import (
    probability_box_xyxy,
    rerender_mask_logits,
)


def test_robust_logodds_consensus_is_coordinatewise_median() -> None:
    field = torch.tensor([0.1, 0.8, 0.7])
    box = torch.tensor([0.95, 0.95, 0.05])
    point = torch.tensor([0.2, 0.3, 0.6])
    result = robust_logodds_consensus((field, box, point))
    assert torch.allclose(result, torch.tensor([0.2, 0.8, 0.6]), atol=1e-6)


def test_exact_adjoint_and_forward_use_visibility_normalization() -> None:
    # Pixel zero sees rows 0/1 with weights .25/.75; pixel one sees row 1.
    gids = torch.tensor([0, 1, 1], dtype=torch.long)
    pids = torch.tensor([0, 0, 1], dtype=torch.long)
    weights = torch.tensor([0.25, 0.75, 0.5])
    primitive, visible = exact_adjoint_probability(
        gids, pids, weights, torch.tensor([1.0, 0.0]), num_gaussians=3
    )
    assert torch.allclose(visible, torch.tensor([0.25, 1.25, 0.0]))
    assert torch.allclose(primitive, torch.tensor([1.0, 0.6, 0.5]))
    rendered, mass = exact_forward_probability(
        gids,
        pids,
        weights,
        primitive,
        height=1,
        width=3,
        unsupported_fallback=torch.tensor([0.0, 0.0, 0.4]),
    )
    assert torch.allclose(mass, torch.tensor([[1.0, 0.5, 0.0]]))
    assert torch.allclose(rendered, torch.tensor([[0.7, 0.6, 0.4]]))


def test_rerender_mask_logits_and_fixed_box() -> None:
    probability = np.zeros((8, 12), dtype=np.float32)
    probability[2:6, 3:9] = 0.8
    logits = rerender_mask_logits(probability, size=256)
    assert logits.shape == (1, 256, 256)
    assert np.isfinite(logits).all()
    box = probability_box_xyxy(probability, padding_pixels=1)
    assert np.array_equal(box, np.asarray([2, 1, 9, 6], dtype=np.float32))

