import pytest
import torch

from radio_gs.scripts.materialize_scannet_direct_siglip_query_cache import (
    fuse_observed_direct_rows,
)


def test_direct_rows_replace_fallback_and_preserve_totality() -> None:
    direct = torch.zeros(3, 1536)
    direct[0, 0] = 2
    direct[2, 2] = 3
    fallback = torch.zeros(3, 1536)
    fallback[:, 10] = 1
    result = fuse_observed_direct_rows(
        direct,
        torch.tensor([True, False, True]),
        fallback,
    ).float()
    assert result[0].argmax().item() == 0
    assert result[1].argmax().item() == 10
    assert result[2].argmax().item() == 2
    assert torch.allclose(result.norm(dim=-1), torch.ones(3), atol=1e-3)


def test_direct_fusion_fails_closed_on_empty_observation() -> None:
    with pytest.raises(ValueError, match="no observed"):
        fuse_observed_direct_rows(
            torch.zeros(2, 1536),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, 1536),
        )
