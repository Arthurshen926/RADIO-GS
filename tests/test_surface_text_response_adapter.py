import math

import pytest
import torch
import torch.nn.functional as F

from radio_gs.models.surface_text_response_adapter import (
    LowRankTangentSummaryAdapter,
)


def _angular_degrees(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(left, right, dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def test_tangent_adapter_is_identity_at_zero_initialized_up_projection():
    torch.manual_seed(3)
    adapter = LowRankTangentSummaryAdapter(
        feature_dim=12,
        rank=4,
        max_angle_degrees=0.1,
    )
    tokens = torch.randn(2, 3, 12)

    adapted = adapter(tokens)

    assert torch.count_nonzero(adapter.up.weight).item() == 0
    torch.testing.assert_close(adapted, tokens, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        adapted.norm(dim=-1),
        tokens.norm(dim=-1),
        rtol=1e-6,
        atol=1e-6,
    )


def test_zero_initialized_adapter_exposes_first_step_up_gradient():
    torch.manual_seed(5)
    adapter = LowRankTangentSummaryAdapter(feature_dim=12, rank=4)
    tokens = torch.randn(6, 12)
    target_direction = torch.randn(12)

    loss = (adapter(tokens) * target_direction).sum()
    loss.backward()

    assert adapter.up.weight.grad is not None
    assert torch.isfinite(adapter.up.weight.grad).all()
    assert adapter.up.weight.grad.abs().sum().item() > 0.0
    # With a zero up matrix, the down factor is intentionally dormant until
    # the first optimizer update makes the low-rank path nonzero.
    assert adapter.down.weight.grad is not None
    assert torch.equal(
        adapter.down.weight.grad,
        torch.zeros_like(adapter.down.weight.grad),
    )


def test_tangent_adapter_preserves_norm_and_enforces_point_one_degree_cap():
    torch.manual_seed(9)
    adapter = LowRankTangentSummaryAdapter(
        feature_dim=16,
        rank=6,
        max_angle_degrees=0.1,
    ).double()
    with torch.no_grad():
        adapter.up.weight.normal_(mean=0.0, std=100.0)
    # Float64 keeps acos well conditioned when verifying a 0.1-degree bound;
    # float32 dot-product roundoff is several thousandths of a degree here.
    tokens = torch.randn(32, 16, dtype=torch.float64)

    adapted = adapter(tokens)
    angles = _angular_degrees(adapted, tokens)

    torch.testing.assert_close(
        adapted.norm(dim=-1),
        tokens.norm(dim=-1),
        rtol=2e-6,
        atol=2e-6,
    )
    assert float(angles.max()) <= 0.1000001
    assert float(angles.max()) >= 0.095
    theoretical_cosine_delta_bound = 2.0 * math.sin(math.radians(0.1) / 2.0)
    assert theoretical_cosine_delta_bound < 0.002


def test_tangent_adapter_has_finite_gradients_after_nonzero_update():
    torch.manual_seed(17)
    adapter = LowRankTangentSummaryAdapter(
        feature_dim=10,
        rank=3,
        max_angle_degrees=1.0,
    )
    with torch.no_grad():
        adapter.up.weight.normal_(mean=0.0, std=0.02)
    tokens = torch.randn(5, 10, requires_grad=True)

    adapted = adapter(tokens)
    loss = adapted[:, 0].sum() + adapted[:, 1].square().mean()
    loss.backward()

    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert adapter.down.weight.grad is not None
    assert adapter.up.weight.grad is not None
    assert torch.isfinite(adapter.down.weight.grad).all()
    assert torch.isfinite(adapter.up.weight.grad).all()
    assert adapter.down.weight.grad.abs().sum().item() > 0.0
    assert adapter.up.weight.grad.abs().sum().item() > 0.0


def test_tangent_adapter_architecture_binds_rank_angle_and_initialization():
    adapter = LowRankTangentSummaryAdapter(
        feature_dim=24,
        rank=7,
        max_angle_degrees=0.25,
    )
    architecture = adapter.architecture()

    assert architecture["name"] == "low_rank_tangent_summary_adapter_v1"
    assert architecture["feature_dim"] == 24
    assert architecture["rank"] == 7
    assert architecture["max_angle_degrees"] == 0.25
    assert len(architecture["digest"]) == 64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_dim": 0},
        {"feature_dim": 4, "rank": 0},
        {"feature_dim": 4, "rank": 5},
        {"max_angle_degrees": 0.0},
        {"max_angle_degrees": 90.0},
        {"eps": 0.0},
    ],
)
def test_tangent_adapter_rejects_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        LowRankTangentSummaryAdapter(**kwargs)


def test_tangent_adapter_rejects_zero_norm_tokens():
    adapter = LowRankTangentSummaryAdapter(feature_dim=4, rank=2)
    with pytest.raises(ValueError, match="nonzero"):
        adapter(torch.zeros(1, 4))
