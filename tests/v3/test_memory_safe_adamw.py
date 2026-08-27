import torch

from radio_gs.v3.training.memory_safe_adamw import MemorySafeAdamW


def test_memory_safe_adamw_matches_torch_adamw():
    left = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))
    right = torch.nn.Parameter(left.detach().clone())
    reference = torch.optim.AdamW([left], lr=3e-4, weight_decay=1e-3)
    candidate = MemorySafeAdamW(
        [right], lr=3e-4, weight_decay=1e-3, chunk_elements=2
    )
    for step in range(4):
        gradient = torch.tensor([[0.2, -0.1], [0.3, 0.4]]) * (step + 1)
        left.grad = gradient.clone()
        right.grad = gradient.clone()
        reference.step()
        candidate.step()
    assert torch.allclose(left, right, atol=1e-7, rtol=1e-6)
