import torch

from radio_gs.v3.training.low_rank_writeback import (
    LowRankWritebackArm,
    pcgrad_backward,
    pcgrad_backward_sparse_anchor,
)


def make_arm() -> LowRankWritebackArm:
    return LowRankWritebackArm(
        torch.randn(6, 512),
        radio_basis=torch.randn(9, 512),
        radio_mean=torch.randn(9),
        radio_scale=torch.rand(9) + 0.5,
        rank=3,
        output_dim=4,
    )


def test_low_rank_residual_starts_at_exact_base_and_folds_to_one_d512():
    arm = make_arm()
    assert torch.equal(arm.folded_latent(), arm.base_latent)
    with torch.no_grad():
        arm.residual_codes[1, 0] = 0.25
    assert torch.equal(arm.folded_latent(), arm.coefficients())
    assert arm.folded_latent().shape == (6, 512)
    state = arm.state_dict()
    assert "base_latent" not in state
    assert "radio_basis" not in state


def test_radio_anchor_is_exact_at_initialization_and_detects_change():
    arm = make_arm()
    rows = torch.tensor([0, 2, 4])
    assert abs(float(arm.radio_anchor_loss(rows))) < 1e-6
    with torch.no_grad():
        arm.residual_codes[2, 0] = 1.0
    assert float(arm.radio_anchor_loss(rows)) > 0


def test_factorized_forward_is_exactly_the_explicit_folded_projection():
    arm = make_arm()
    with torch.no_grad():
        arm.residual_codes.normal_()
    rows = torch.tensor([0, 2, 5])
    scale = 0.3
    actual = arm(scale, rows)
    latent = arm.coefficients(rows)
    phase = latent.new_tensor([scale]) * torch.pi
    gamma, beta = arm.scale_adapter(
        torch.cat((phase.sin(), phase.cos()))
    ).chunk(2)
    expected = torch.nn.functional.normalize(
        arm.projection(latent) * (1 + 0.1 * gamma.tanh()) + 0.1 * beta,
        dim=-1,
        eps=1e-8,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.equal(actual, arm.scale_embedding(arm.projected_latent(rows), scale))


def test_pcgrad_removes_primary_conflict_before_adding_anchor_gradient():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    primary = -parameter[0] + parameter[1]
    anchor = parameter[0]
    report = pcgrad_backward(primary, anchor, [parameter], anchor_weight=0.5)
    assert report.conflict
    assert torch.allclose(parameter.grad, torch.tensor([0.5, 1.0]))


def test_sparse_anchor_pcgrad_matches_dense_row_scatter():
    table = torch.nn.Parameter(torch.tensor([[1.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))
    head = torch.nn.Parameter(torch.tensor([2.0]))
    rows = torch.tensor([0, 2])
    explicit = table.detach()[rows].clone().requires_grad_(True)
    primary = -table[0, 0] + table[0, 1] + table[1].sum() + head.square().sum()
    anchor = explicit[:, 0].sum()
    report = pcgrad_backward_sparse_anchor(
        primary,
        anchor,
        [table, head],
        anchor_parameter=table,
        anchor_rows=rows,
        anchor_values=explicit,
        anchor_weight=0.5,
    )
    assert report.conflict
    assert torch.allclose(
        table.grad,
        torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),
    )
    assert torch.allclose(head.grad, torch.tensor([4.0]))
