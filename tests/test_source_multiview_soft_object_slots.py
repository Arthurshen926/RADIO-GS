import torch

from radio_gs.scripts.evaluate_source_multiview_soft_object_slots import noisy_or


def test_noisy_or_has_exact_single_slot_semantics() -> None:
    assignment = torch.tensor([[1.0], [0.5]])
    slots = torch.tensor([[0.7], [0.4]])
    assert torch.allclose(noisy_or(assignment, slots), torch.tensor([0.7, 0.2]))


def test_noisy_or_composes_multiple_slots() -> None:
    assignment = torch.ones(1, 2)
    slots = torch.tensor([[0.5, 0.5]])
    assert torch.allclose(noisy_or(assignment, slots), torch.tensor([0.75]))
