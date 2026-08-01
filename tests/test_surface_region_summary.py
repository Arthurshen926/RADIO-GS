import torch

from radio_gs.interfaces.surface_region_summary import (
    JOINT_CONTEXT_POOLING,
    SEPARATE_CONTEXT_POOLING,
    SurfaceRegionSummaryReadout,
    SurfaceRegionSummaryReadoutV2,
    surface_region_geometry,
    surface_region_geometry_v2,
)


def _inputs():
    torch.manual_seed(3)
    features = torch.randn(2, 9, 16)
    xyz = torch.randn(2, 9, 3)
    scales = torch.rand(2, 9, 3) * 0.04 + 0.01
    opacity = torch.rand(2, 9, 1)
    reliability = torch.rand(2, 9, 1) * 0.8 + 0.2
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[1, 7:] = False
    geometry = surface_region_geometry(
        xyz, scales, opacity, reliability, torch.tensor([0.3, 0.6]), token_mask=mask
    )
    return features, geometry, reliability, mask


def test_surface_readout_is_permutation_invariant() -> None:
    features, geometry, reliability, mask = _inputs()
    model = SurfaceRegionSummaryReadout(feature_dim=16, hidden_dim=8).eval()
    expected = model(features, geometry, token_mask=mask, reliability=reliability)
    order = torch.tensor([7, 1, 4, 0, 8, 2, 6, 5, 3])
    actual = model(
        features[:, order], geometry[:, order], token_mask=mask[:, order],
        reliability=reliability[:, order],
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_surface_geometry_is_translation_invariant_and_masks_padding() -> None:
    features, geometry, reliability, mask = _inputs()
    torch.manual_seed(3)
    xyz = torch.randn(2, 9, 3)
    scales = torch.rand(2, 9, 3) * 0.04 + 0.01
    opacity = torch.rand(2, 9, 1)
    shifted = surface_region_geometry(
        xyz + torch.tensor([19.0, -4.0, 7.0]), scales, opacity, reliability,
        torch.tensor([0.3, 0.6]), token_mask=mask,
    )
    original = surface_region_geometry(
        xyz, scales, opacity, reliability, torch.tensor([0.3, 0.6]), token_mask=mask
    )
    torch.testing.assert_close(shifted, original, atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(geometry[1, 7:]) == 0


def test_zero_initialized_readout_is_reliability_weighted_raw_mean() -> None:
    features, geometry, reliability, mask = _inputs()
    model = SurfaceRegionSummaryReadout(feature_dim=16, hidden_dim=8).eval()
    output = model(features, geometry, token_mask=mask, reliability=reliability)
    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()


def test_v2_readout_is_anchor_conditioned_and_jointly_permutation_invariant() -> None:
    torch.manual_seed(11)
    features = torch.randn(2, 7, 16)
    xyz = torch.randn(2, 7, 3)
    scale = torch.full((2, 7, 3), 0.04)
    reliability = torch.rand(2, 7, 1).clamp_min(0.1)
    mask = torch.ones(2, 7, dtype=torch.bool)
    core = torch.zeros_like(mask); core[:, :5] = True
    anchor = torch.tensor([1, 3])
    geometry = surface_region_geometry_v2(
        xyz, scale, reliability, torch.tensor([0.25, 0.45]),
        anchor_index=anchor, core_mask=core, token_mask=mask,
    )
    model = SurfaceRegionSummaryReadoutV2(feature_dim=16, hidden_dim=8).eval()
    expected = model(
        features, geometry, anchor_index=anchor, token_mask=mask,
        reliability=reliability,
    )
    order = torch.tensor([4, 2, 6, 1, 0, 5, 3])
    inverse = torch.empty_like(order); inverse[order] = torch.arange(len(order))
    actual = model(
        features[:, order], geometry[:, order], anchor_index=inverse[anchor],
        token_mask=mask[:, order], reliability=reliability[:, order],
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    other_anchor = model(
        features, geometry, anchor_index=torch.tensor([0, 0]), token_mask=mask,
        reliability=reliability,
    )
    assert not torch.allclose(other_anchor, expected)


def test_v2_input_only_mode_does_not_double_apply_reliability_prior() -> None:
    torch.manual_seed(23)
    features = torch.randn(1, 4, 16)
    geometry = torch.randn(1, 4, 14)
    mask = torch.ones(1, 4, dtype=torch.bool)
    anchor = torch.tensor([0])
    low_high = torch.tensor([[[0.1], [1.0], [1.0], [1.0]]])
    high_low = torch.tensor([[[1.0], [0.1], [1.0], [1.0]]])
    input_only = SurfaceRegionSummaryReadoutV2(
        feature_dim=16,
        hidden_dim=8,
        reliability_attention_mode="input_only",
    ).eval()
    first = input_only(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
        reliability=low_high,
    )
    second = input_only(
        features,
        geometry,
        anchor_index=anchor,
        token_mask=mask,
        reliability=high_low,
    )
    torch.testing.assert_close(first, second)
    assert (
        input_only.architecture("contract")["reliability_attention_mode"]
        == "input_only"
    )


def _core_context_inputs(context_rows: int) -> tuple[torch.Tensor, ...]:
    features = torch.zeros(1, 2 + context_rows, 16)
    features[:, 0, 0] = 1.0
    features[:, 1, 1] = 1.0
    features[:, 2:, 2] = 8.0
    geometry = torch.zeros(1, 2 + context_rows, 14)
    geometry[:, :2, 8] = 1.0
    geometry[:, 2:, 9] = 1.0
    mask = torch.ones(1, 2 + context_rows, dtype=torch.bool)
    reliability = torch.ones(1, 2 + context_rows, 1)
    anchor = torch.tensor([0])
    return features, geometry, mask, reliability, anchor


def test_v2_separate_context_pooling_protects_the_zero_residual_core_base() -> None:
    torch.manual_seed(29)
    one_context = _core_context_inputs(1)
    repeated_context = _core_context_inputs(5)
    separate = SurfaceRegionSummaryReadoutV2(
        feature_dim=16,
        hidden_dim=8,
        context_pooling_mode=SEPARATE_CONTEXT_POOLING,
    ).eval()
    joint = SurfaceRegionSummaryReadoutV2(
        feature_dim=16,
        hidden_dim=8,
        context_pooling_mode=JOINT_CONTEXT_POOLING,
    ).eval()
    joint.load_state_dict(separate.state_dict())

    def run(model: SurfaceRegionSummaryReadoutV2, values: tuple[torch.Tensor, ...]):
        features, geometry, mask, reliability, anchor = values
        return model(
            features,
            geometry,
            anchor_index=anchor,
            token_mask=mask,
            reliability=reliability,
        )

    torch.testing.assert_close(
        run(separate, one_context),
        run(separate, repeated_context),
    )
    assert not torch.allclose(
        run(joint, one_context),
        run(joint, repeated_context),
    )


def test_v2_separate_context_pooling_is_manifest_bound_and_context_trainable() -> None:
    torch.manual_seed(31)
    values = _core_context_inputs(2)
    model = SurfaceRegionSummaryReadoutV2(
        feature_dim=16,
        hidden_dim=8,
        context_pooling_mode=SEPARATE_CONTEXT_POOLING,
    ).eval()
    architecture = model.architecture("contract")
    assert architecture["context_pooling_mode"] == SEPARATE_CONTEXT_POOLING
    assert (
        "context_pooling_mode"
        not in SurfaceRegionSummaryReadoutV2(
            feature_dim=16,
            hidden_dim=8,
        ).architecture("contract")
    )

    features, geometry, mask, reliability, anchor = values
    with torch.no_grad():
        model.residual[-1].weight.normal_()
        before = model(
            features,
            geometry,
            anchor_index=anchor,
            token_mask=mask,
            reliability=reliability,
        )
        changed = features.clone()
        changed[:, 2:, 2] = -8.0
        after = model(
            changed,
            geometry,
            anchor_index=anchor,
            token_mask=mask,
            reliability=reliability,
        )
    assert not torch.allclose(before, after)


def test_v2_separate_context_pooling_rejects_missing_partition_flags() -> None:
    features, geometry, mask, reliability, anchor = _core_context_inputs(1)
    geometry[:, -1, 9] = 0.0
    model = SurfaceRegionSummaryReadoutV2(
        feature_dim=16,
        hidden_dim=8,
        context_pooling_mode=SEPARATE_CONTEXT_POOLING,
    )
    try:
        model(
            features,
            geometry,
            anchor_index=anchor,
            token_mask=mask,
            reliability=reliability,
        )
    except ValueError as exc:
        assert "core/context flags" in str(exc)
    else:
        raise AssertionError("expected an incomplete core/context partition to fail")
