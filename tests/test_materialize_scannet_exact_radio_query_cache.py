import torch

from radio_gs.scripts.materialize_scannet_exact_radio_query_cache import (
    project_observed_exact_rows,
)


class _Head(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros(*value.shape[:-1], 256, device=value.device, dtype=value.dtype)
        return torch.cat((value, pad), dim=-1)


def test_exact_rows_replace_fallback_and_unobserved_rows_are_identical() -> None:
    raw = torch.zeros(3, 1280)
    raw[0, 0] = 2
    raw[2, 2] = 3
    fallback = torch.zeros(3, 1536)
    fallback[:, 10] = 1
    observed = torch.tensor([True, False, True])
    result = project_observed_exact_rows(
        raw,
        observed,
        fallback,
        _Head(),
        device=torch.device("cpu"),
        chunk_size=1,
    ).float()
    assert result.shape == (3, 1536)
    assert result[0].argmax().item() == 0
    assert result[1].argmax().item() == 10
    assert result[2].argmax().item() == 2
    assert torch.allclose(result.norm(dim=-1), torch.ones(3), atol=1e-3)
