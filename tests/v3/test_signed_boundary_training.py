from types import SimpleNamespace

import torch
from torch import nn

from radio_gs.v3.memory.structured_memory import LowRankPrivateBranchMemory
from radio_gs.v3.training.fit_clean_parent_signed_boundary import (
    _gate,
    signed_boundary_objective,
)


def test_signed_boundary_gate_allows_small_body_tolerance() -> None:
    baseline = {
        cohort: {"boundary_f": 0.0, "mask_iou": 0.5, "brier": 0.2}
        for cohort in ("oracle", "text")
    }
    candidate = {
        cohort: {"boundary_f": 0.1, "mask_iou": 0.498, "brier": 0.202}
        for cohort in ("oracle", "text")
    }
    passed, failures = _gate(baseline, candidate, tolerance=0.005)
    assert passed
    assert not failures


def test_signed_boundary_objective_backpropagates_to_D16_and_head() -> None:
    memory = torch.randn(25, 512)
    model = LowRankPrivateBranchMemory(memory)
    model.enable_owned_training_blocks("boundary")
    head = nn.Linear(16, 1)
    target = torch.zeros(5, 5, dtype=torch.bool)
    target[1:4, 1:4] = True
    episode = SimpleNamespace(
        scale=0.5,
        gaussian_ids=torch.arange(25),
        pixel_ids=torch.arange(25),
        contribution_weights=torch.ones(25),
        target=target,
    )
    loss = signed_boundary_objective(
        model,
        head,
        (torch.tensor([0]), torch.tensor([1.0])),
        episode,
        episode.target,
        torch.ones(5, 5, dtype=torch.bool),
        temperature=0.15,
        maximum_logit_residual=1.0,
        body_preservation_weight=0.25,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.owned_training_parameter("boundary").grad is not None
    assert head.weight.grad is not None
