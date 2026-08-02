import math

import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.interfaces.surface_region_summary import (
    SurfaceRegionSummaryReadoutV2,
)
from radio_gs.models.surface_region_dual_descriptor import (
    SurfaceRegionDualDescriptor,
    SurfaceRegionDualDescriptorOutput,
)


def _small_v2_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(401)
    features = torch.randn(2, 5, 16)
    geometry = torch.randn(2, 5, 14)
    token_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    reliability = torch.rand(2, 5, 1).clamp_min(0.1)
    anchor = torch.tensor([1, 0])
    return features, geometry, token_mask, reliability, anchor


def _legacy_joint_v2_forward(
    model: SurfaceRegionSummaryReadoutV2,
    features: torch.Tensor,
    geometry: torch.Tensor,
    token_mask: torch.Tensor,
    reliability: torch.Tensor,
    anchor: torch.Tensor,
) -> torch.Tensor:
    """The pre-extension V2 joint-pooling forward, kept as a bitwise oracle."""

    values = torch.as_tensor(features).float()
    geom = torch.as_tensor(geometry, device=values.device).float()
    mask = torch.as_tensor(token_mask, device=values.device).bool()
    anchor = torch.as_tensor(anchor, device=values.device).long().reshape(-1)
    batch = torch.arange(values.shape[0], device=values.device)
    hidden = model.feature_encoder(values) + model.geometry_encoder(geom)
    query = model.query_encoder(
        torch.cat([values[batch, anchor], geom[batch, anchor]], dim=-1)
    )
    logits = (
        torch.einsum("bh,bth->bt", query, model.key(hidden))
        / model.hidden_dim**0.5
    )
    confidence = torch.as_tensor(reliability, device=values.device).float()[..., 0]
    logits = logits + confidence.clamp_min(1e-4).log()
    weights = model._masked_attention(logits, mask)
    raw_mean = torch.einsum("bt,btc->bc", weights, values)
    anchor_feature = values[batch, anchor]
    base = raw_mean + 0.25 * (anchor_feature - raw_mean)
    pooled = torch.einsum("bt,bth->bh", weights, hidden) + query
    return base + model.residual(pooled)


def test_v2_context_extension_is_bitwise_forward_and_state_dict_compatible() -> None:
    features, geometry, token_mask, reliability, anchor = _small_v2_inputs()
    torch.manual_seed(409)
    model = SurfaceRegionSummaryReadoutV2(feature_dim=16, hidden_dim=8).eval()
    state_before = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }

    legacy = _legacy_joint_v2_forward(
        model, features, geometry, token_mask, reliability, anchor
    )
    official = model(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=token_mask,
        reliability=reliability,
    )
    from_context, context = model.forward_with_context(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=token_mask,
        reliability=reliability,
    )

    assert torch.equal(official, legacy)
    assert torch.equal(from_context, legacy)
    assert context.shape == (2, 8)
    state_after = model.state_dict()
    assert state_after.keys() == state_before.keys()
    assert all(torch.equal(state_after[key], value) for key, value in state_before.items())


class _RecordingOfficialHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1280, 1536)
        self.last_shape: tuple[int, ...] | None = None

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        self.last_shape = tuple(token.shape)
        return self.projection(token)


def _dual_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(419)
    features = torch.randn(2, 4, 1280)
    geometry = torch.randn(2, 4, 14)
    token_mask = torch.ones(2, 4, dtype=torch.bool)
    reliability = torch.rand(2, 4, 1).clamp_min(0.1)
    anchor = torch.tensor([0, 2])
    return features, geometry, token_mask, reliability, anchor


def _build_dual() -> tuple[
    SurfaceRegionDualDescriptor,
    SurfaceRegionSummaryReadoutV2,
    _RecordingOfficialHead,
]:
    torch.manual_seed(421)
    base = SurfaceRegionSummaryReadoutV2(hidden_dim=128)
    head = _RecordingOfficialHead()
    return SurfaceRegionDualDescriptor(base, head), base, head


def test_dual_descriptor_zero_init_preserves_controls_and_freezes_them() -> None:
    model, base, head = _build_dual()
    features, geometry, token_mask, reliability, anchor = _dual_inputs()
    model.train()
    output = model(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=token_mask,
        reliability=reliability,
    )

    assert isinstance(output, SurfaceRegionDualDescriptorOutput)
    assert output.official_token.shape == (2, 1280)
    assert output.official_descriptor.shape == (2, 1536)
    assert output.semantic_descriptor.shape == (2, 1536)
    assert torch.equal(output.semantic_descriptor, output.official_descriptor)
    assert head.last_shape == (2, 1, 1280)
    assert not base.training and not head.training
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert all(not parameter.requires_grad for parameter in head.parameters())
    assert model.trainable_parameter_count() == 823_041
    assert torch.count_nonzero(model.film.weight) == 0
    assert torch.count_nonzero(model.film.bias) == 0

    output.semantic_descriptor[:, 0].sum().backward()
    assert model.film.bias.grad is not None
    assert torch.count_nonzero(model.film.bias.grad) > 0
    assert all(parameter.grad is None for parameter in base.parameters())
    assert all(parameter.grad is None for parameter in head.parameters())


def test_dual_descriptor_adapts_to_frozen_v3_hidden_256() -> None:
    torch.manual_seed(431)
    base = SurfaceRegionSummaryReadoutV2(hidden_dim=256)
    head = _RecordingOfficialHead()
    model = SurfaceRegionDualDescriptor(base, head).eval()
    features, geometry, token_mask, reliability, anchor = _dual_inputs()

    official_token, context = base.forward_with_context(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=token_mask,
        reliability=reliability,
    )
    output = model(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=token_mask,
        reliability=reliability,
    )

    assert model.context_dim == 256
    assert model.context_norm.normalized_shape == (256,)
    assert model.context_projection.in_features == 256
    assert model.trainable_parameter_count() == 856_065
    assert model.architecture()["trainable_parameter_count"] == 856_065
    assert context.shape == (2, 256)
    assert torch.equal(output.official_token, official_token)
    assert torch.equal(output.semantic_descriptor, output.official_descriptor)


def test_dual_descriptor_implements_bounded_film_formula() -> None:
    model, _, _ = _build_dual()
    features, geometry, token_mask, reliability, anchor = _dual_inputs()
    with torch.no_grad():
        model.film.weight.zero_()
        model.film.bias[:1536].fill_(0.2)
        model.film.bias[1536:].fill_(-0.3)
        model.gate.weight.zero_()
        model.gate.bias.fill_(0.4)

    output = model(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=token_mask,
        reliability=reliability,
    )
    gamma = math.tanh(0.2)
    beta = math.tanh(-0.3)
    alpha = torch.sigmoid(torch.tensor(0.4 + math.log(0.1 / 0.9)))
    delta = gamma * output.official_descriptor + beta / math.sqrt(1536)
    expected = F.normalize(output.official_descriptor + alpha * delta, dim=-1)

    torch.testing.assert_close(output.semantic_descriptor, expected)
    torch.testing.assert_close(
        output.semantic_descriptor.norm(dim=-1), torch.ones(2)
    )
    architecture = model.architecture()
    assert architecture["initial_gate"] == 0.1
    assert architecture["trainable_parameter_count"] == 823_041
