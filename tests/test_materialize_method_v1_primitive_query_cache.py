from __future__ import annotations

import torch
from torch import nn

from radio_gs.scripts.materialize_method_v1_primitive_query_cache import (
    decode_method_v1_primitive_query_rows,
)


class _Field(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rows = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 3.0],
                ]
            )
        )
        self.num_gaussians = 3

    def radio_features(self, indices: torch.Tensor) -> torch.Tensor:
        return self.rows[indices]


class _Head(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.cat((value, value), dim=-1)


def test_decode_method_v1_primitive_rows_is_chunked_and_normalized() -> None:
    rows = decode_method_v1_primitive_query_rows(
        _Field(),
        _Head(),
        device=torch.device("cpu"),
        chunk_size=2,
    )

    assert rows.shape == (3, 6)
    assert rows.dtype == torch.float16
    assert torch.allclose(
        torch.linalg.vector_norm(rows.float(), dim=-1),
        torch.ones(3),
        atol=5e-4,
        rtol=0.0,
    )
