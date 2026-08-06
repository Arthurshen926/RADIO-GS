import math

import pytest
import torch

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV3
from radio_gs.interfaces.surface_region_summary import (
    JOINT_CONTEXT_POOLING,
    SEPARATE_CONTEXT_POOLING,
    SURFACE_GEOMETRY_V3_DIM,
    SURFACE_GEOMETRY_V3_LEARNED_DIM,
    SURFACE_REGION_V3_FEATURE_GAUGE,
    SURFACE_REGION_V3_GATED_RAW_PRIOR,
    SURFACE_REGION_V3_GATED_RAW_PRIOR_INITIAL_WEIGHT,
    SURFACE_SUMMARY_READOUT_V3_GATED_BASE_SCHEMA_VERSION,
    SURFACE_SUMMARY_READOUT_V3_SCHEMA_VERSION,
    SurfaceRegionSummaryReadoutV2,
    SurfaceRegionSummaryReadoutV3,
    surface_region_effective_reliability_v3,
    surface_region_geometry_v2,
    surface_region_geometry_v3,
)


def _v3_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(47)
    raw = torch.randn(2, 5, 8)
    token_mask = torch.tensor([
        [True, True, True, True, True],
        [True, True, True, True, False],
    ])
    raw[1, 4] = 0.0
    raw_norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    direction = torch.nn.functional.normalize(raw, dim=-1)
    direction = direction.masked_fill(~token_mask[..., None], 0.0)
    xyz = torch.randn(2, 5, 3)
    scale = torch.rand(2, 5, 3) * 0.04 + 0.01
    primitive_reliability = torch.tensor([
        [[0.9], [0.8], [0.7], [0.6], [0.5]],
        [[0.9], [0.8], [0.7], [0.6], [0.0]],
    ])
    core = torch.tensor([
        [True, True, False, False, False],
        [True, False, False, False, False],
    ])
    context = torch.tensor([
        [False, False, True, False, False],
        [False, True, True, False, False],
    ])
    support_fill = torch.tensor([
        [False, False, False, True, True],
        [False, False, False, True, False],
    ])
    recovery = torch.tensor([
        [float("inf"), float("inf"), float("inf"), 0.3, 0.6],
        [float("inf"), float("inf"), float("inf"), 0.9, float("inf")],
    ])
    radius = torch.tensor([0.3, 0.6])
    reliability = surface_region_effective_reliability_v3(
        primitive_reliability,
        recovery,
        radius,
        support_fill_mask=support_fill,
        token_mask=token_mask,
    )
    anchor = torch.tensor([0, 0])
    geometry = surface_region_geometry_v3(
        xyz,
        scale,
        reliability,
        radius,
        raw_radio_l2_norm=raw_norm,
        anchor_index=anchor,
        core_mask=core,
        context_mask=context,
        support_fill_mask=support_fill,
        token_mask=token_mask,
    )
    return (
        raw, direction, raw_norm, xyz, scale, primitive_reliability,
        reliability, radius, core, context, support_fill, recovery,
        token_mask, anchor, geometry,
    )


def test_geometry_v3_preserves_v2_indices_and_zeroes_tensor_padding() -> None:
    (
        _, _, raw_norm, xyz, scale, _, reliability, radius, core, context,
        support_fill, _, token_mask, anchor, geometry,
    ) = _v3_inputs()
    assert geometry.shape == (2, 5, SURFACE_GEOMETRY_V3_DIM)
    assert torch.count_nonzero(geometry[1, 4]) == 0
    assert torch.equal(geometry[..., 14] > 0.5, support_fill)
    torch.testing.assert_close(geometry[..., 15][token_mask], torch.log(raw_norm[..., 0][token_mask]))

    # With no support fill, the first 14 dimensions are byte-compatible with
    # geometry-v2 and therefore preserve every existing index meaning.
    no_fill = torch.zeros_like(support_fill)
    v2_context = token_mask & ~core
    v2 = surface_region_geometry_v2(
        xyz,
        scale,
        reliability,
        radius,
        anchor_index=anchor,
        core_mask=core,
        token_mask=token_mask,
    )
    v3 = surface_region_geometry_v3(
        xyz,
        scale,
        reliability,
        radius,
        raw_radio_l2_norm=raw_norm,
        anchor_index=anchor,
        core_mask=core,
        context_mask=v2_context,
        support_fill_mask=no_fill,
        token_mask=token_mask,
    )
    assert torch.equal(v3[..., :14], v2)


def test_direction_norm_decomposition_has_the_declared_scaling_law() -> None:
    (
        raw, direction, raw_norm, xyz, scale, _, reliability, radius, core,
        context, support_fill, _, token_mask, anchor, geometry,
    ) = _v3_inputs()
    factor = 3.25
    scaled_raw = raw * factor
    scaled_direction = torch.nn.functional.normalize(scaled_raw, dim=-1)
    scaled_direction = scaled_direction.masked_fill(~token_mask[..., None], 0.0)
    torch.testing.assert_close(scaled_direction, direction, rtol=1e-6, atol=1e-7)
    scaled_geometry = surface_region_geometry_v3(
        xyz,
        scale,
        reliability,
        radius,
        raw_radio_l2_norm=raw_norm * factor,
        anchor_index=anchor,
        core_mask=core,
        context_mask=context,
        support_fill_mask=support_fill,
        token_mask=token_mask,
    )
    assert torch.equal(scaled_geometry[..., :15], geometry[..., :15])
    torch.testing.assert_close(
        scaled_geometry[..., 15][token_mask] - geometry[..., 15][token_mask],
        torch.full_like(geometry[..., 15][token_mask], math.log(factor)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_geometry_v3_rejects_invalid_norm_partition_and_anchor() -> None:
    (
        _, _, raw_norm, xyz, scale, _, reliability, radius, core, context,
        support_fill, _, token_mask, anchor, _,
    ) = _v3_inputs()
    invalid_norm = raw_norm.clone()
    invalid_norm[0, 0] = 0.0
    with pytest.raises(ValueError, match="positive and finite"):
        surface_region_geometry_v3(
            xyz, scale, reliability, radius,
            raw_radio_l2_norm=invalid_norm,
            anchor_index=anchor, core_mask=core, context_mask=context,
            support_fill_mask=support_fill, token_mask=token_mask,
        )
    overlap = context.clone()
    overlap[0, 0] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        surface_region_geometry_v3(
            xyz, scale, reliability, radius,
            raw_radio_l2_norm=raw_norm,
            anchor_index=anchor, core_mask=core, context_mask=overlap,
            support_fill_mask=support_fill, token_mask=token_mask,
        )
    not_core = core.clone()
    not_core[0, 0] = False
    context_with_anchor = context.clone()
    context_with_anchor[0, 0] = True
    with pytest.raises(ValueError, match="anchor must be a valid core"):
        surface_region_geometry_v3(
            xyz, scale, reliability, radius,
            raw_radio_l2_norm=raw_norm,
            anchor_index=anchor, core_mask=not_core, context_mask=context_with_anchor,
            support_fill_mask=support_fill, token_mask=token_mask,
        )


def test_support_fill_effective_reliability_is_parameter_free_and_exact() -> None:
    primitive = torch.tensor([[0.8], [0.6], [0.4], [0.9]])
    recovery = torch.tensor([float("inf"), 0.5, 1.0, float("inf")])
    support_fill = torch.tensor([False, True, True, False])
    token_mask = torch.tensor([True, True, True, False])
    result = surface_region_effective_reliability_v3(
        primitive,
        recovery,
        0.5,
        support_fill_mask=support_fill,
        token_mask=token_mask,
    )
    expected = torch.tensor([
        [0.8],
        [0.6 * math.exp(-1.0)],
        [0.4 * math.exp(-2.0)],
        [0.0],
    ])
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="finite and non-negative"):
        surface_region_effective_reliability_v3(
            primitive,
            torch.tensor([float("inf"), float("inf"), 1.0, 0.0]),
            0.5,
            support_fill_mask=support_fill,
            token_mask=token_mask,
        )


def test_readout_v3_enforces_direction_gauge_and_input_only_reliability() -> None:
    raw, direction, _, *rest = _v3_inputs()
    token_mask, anchor, geometry = rest[-3:]
    model = SurfaceRegionSummaryReadoutV3(feature_dim=8, hidden_dim=6).eval()
    first = model(
        direction,
        geometry,
        anchor_index=anchor,
        token_mask=token_mask,
        reliability=torch.zeros(2, 5, 1),
    )
    second = model(
        direction,
        geometry,
        anchor_index=anchor,
        token_mask=token_mask,
        reliability=torch.ones(2, 5, 1),
    )
    assert first.shape == (2, 8)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    with pytest.raises(ValueError, match="unit L2 direction gauge"):
        model(raw, geometry, anchor_index=anchor, token_mask=token_mask)
    with pytest.raises(ValueError, match="fixed to input_only"):
        SurfaceRegionSummaryReadoutV3(
            feature_dim=8, hidden_dim=6, reliability_attention_mode="log_prior",
        )
    with pytest.raises(ValueError, match="fixed to joint_attention"):
        SurfaceRegionSummaryReadoutV3(
            feature_dim=8, hidden_dim=6, context_pooling_mode=SEPARATE_CONTEXT_POOLING,
        )


def test_readout_v3_accepts_bounded_fp16_unit_direction_quantization() -> None:
    _, _, _, *rest = _v3_inputs()
    token_mask, anchor, geometry = rest[-3:]
    direction = torch.zeros(2, 5, 8, dtype=torch.float16)
    direction[..., :3] = torch.tensor(1.0 / math.sqrt(3.0)).half()
    direction = direction.masked_fill(~token_mask[..., None], 0.0)
    active_norm = torch.linalg.vector_norm(direction.float(), dim=-1)[token_mask]
    assert 2e-4 < float((active_norm - 1.0).abs().max()) < 5e-4
    model = SurfaceRegionSummaryReadoutV3(feature_dim=8, hidden_dim=6).eval()
    output = model(
        direction,
        geometry,
        anchor_index=anchor,
        token_mask=token_mask,
    )
    assert output.shape == (2, 8)
    with pytest.raises(ValueError, match="unit L2 direction gauge"):
        model(
            direction.float(),
            geometry,
            anchor_index=anchor,
            token_mask=token_mask,
        )


def test_readout_v3_base_explicitly_reconstructs_raw_radio_amplitude() -> None:
    direction = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    raw_norm = torch.tensor([[2.0], [4.0]])
    geometry = surface_region_geometry_v3(
        torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        torch.full((2, 3), 0.02),
        torch.ones(2, 1),
        0.2,
        raw_radio_l2_norm=raw_norm,
        anchor_index=0,
        core_mask=torch.ones(2, dtype=torch.bool),
        context_mask=torch.zeros(2, dtype=torch.bool),
        support_fill_mask=torch.zeros(2, dtype=torch.bool),
    )
    model = SurfaceRegionSummaryReadoutV3(feature_dim=4, hidden_dim=3).eval()
    # Force uniform attention while retaining the zero-initialized residual.
    with torch.no_grad():
        for module in (model.query_encoder, model.key):
            for parameter in module.parameters():
                parameter.zero_()
    output = model(direction, geometry, anchor_index=0)
    # mean([2,4]) + 0.25 * (anchor=2 - mean=3) = 2.75.  Averaging
    # unit directions directly would incorrectly produce 1.0 instead.
    torch.testing.assert_close(
        output,
        torch.tensor([2.75, 0.0, 0.0, 0.0]),
        rtol=0,
        atol=1e-7,
    )


def test_readout_v3_gated_raw_prior_is_small_global_and_has_no_anchor_mix() -> None:
    direction = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    raw_norm = torch.tensor([[2.0], [4.0]])
    geometry = surface_region_geometry_v3(
        torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        torch.full((2, 3), 0.02),
        torch.ones(2, 1),
        0.2,
        raw_radio_l2_norm=raw_norm,
        anchor_index=0,
        core_mask=torch.ones(2, dtype=torch.bool),
        context_mask=torch.zeros(2, dtype=torch.bool),
        support_fill_mask=torch.zeros(2, dtype=torch.bool),
    )
    model = SurfaceRegionSummaryReadoutV3(
        feature_dim=4,
        hidden_dim=3,
        base_output_mode=SURFACE_REGION_V3_GATED_RAW_PRIOR,
    ).eval()
    with torch.no_grad():
        for module in (model.query_encoder, model.key):
            for parameter in module.parameters():
                parameter.zero_()
    first = model(direction, geometry, anchor_index=0)
    second_geometry = surface_region_geometry_v3(
        torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        torch.full((2, 3), 0.02),
        torch.ones(2, 1),
        0.2,
        raw_radio_l2_norm=raw_norm,
        anchor_index=1,
        core_mask=torch.ones(2, dtype=torch.bool),
        context_mask=torch.zeros(2, dtype=torch.bool),
        support_fill_mask=torch.zeros(2, dtype=torch.bool),
    )
    second = model(direction, second_geometry, anchor_index=1)
    expected = torch.tensor([0.15, 0.0, 0.0, 0.0])
    torch.testing.assert_close(first, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(second, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        torch.sigmoid(model.raw_prior_gate_logit),
        torch.tensor(SURFACE_REGION_V3_GATED_RAW_PRIOR_INITIAL_WEIGHT),
        rtol=1e-6,
        atol=1e-7,
    )


def test_readout_v3_raw_norm_is_only_a_gauge_side_channel() -> None:
    (
        _, direction, _, _, _, _, _, _, _, _, _, _, token_mask, anchor,
        geometry,
    ) = _v3_inputs()
    model = SurfaceRegionSummaryReadoutV3(feature_dim=8, hidden_dim=6).eval()
    shifted = geometry.clone()
    shifted[..., 15][token_mask] += torch.linspace(
        -0.7,
        0.9,
        int(token_mask.sum()),
    )
    _, context = model.forward_with_context(
        direction,
        geometry,
        anchor_index=anchor,
        token_mask=token_mask,
    )
    _, shifted_context = model.forward_with_context(
        direction,
        shifted,
        anchor_index=anchor,
        token_mask=token_mask,
    )
    # The learned attention/residual path is exactly independent of raw norm;
    # only the explicit raw-gauge base may respond to this side channel.
    torch.testing.assert_close(context, shifted_context, rtol=0, atol=0)


def test_readout_v3_checkpoint_schema_digest_and_v2_are_independent(tmp_path) -> None:
    torch.manual_seed(53)
    contract = SurfaceRegionContractV3()
    model = SurfaceRegionSummaryReadoutV3(feature_dim=8, hidden_dim=6).eval()
    architecture = model.architecture(contract.digest)
    assert architecture["digest"] == (
        "5dd3af23bb0d390e578cf59e7665c36135cc2cf2ec7916b01eb197e7a69025f4"
    )
    assert architecture["feature_normalization"] == SURFACE_REGION_V3_FEATURE_GAUGE
    assert (
        architecture["base_gauge_reconstruction"]
        == "direction_times_exp_log_raw_norm_v1"
    )
    assert architecture["reliability_attention_mode"] == "input_only"
    assert architecture["context_pooling_mode"] == JOINT_CONTEXT_POOLING
    assert architecture["learned_geometry_dim"] == SURFACE_GEOMETRY_V3_LEARNED_DIM
    assert architecture["raw_radio_l2_norm_usage"] == "base_reconstruction_only_v1"
    payload = {
        "schema_version": SURFACE_SUMMARY_READOUT_V3_SCHEMA_VERSION,
        "architecture": architecture,
        "state_dict": model.state_dict(),
    }
    checkpoint = tmp_path / "readout_v3.pt"
    torch.save(payload, checkpoint)
    restored, restored_payload = SurfaceRegionSummaryReadoutV3.from_checkpoint(checkpoint)
    assert restored_payload["schema_version"] == 7
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    assert not any(parameter.requires_grad for parameter in restored.parameters())

    bad_schema = tmp_path / "old_v3_schema.pt"
    torch.save({**payload, "schema_version": 6}, bad_schema)
    with pytest.raises(ValueError, match="invalid v3"):
        SurfaceRegionSummaryReadoutV3.from_checkpoint(bad_schema)

    # Exact frozen V2 architecture serialization is untouched by the new
    # schema, gauge, and geometry dimensions.
    v2 = SurfaceRegionSummaryReadoutV2(feature_dim=16, hidden_dim=8)
    assert (
        v2.architecture("contract")["digest"]
        == "db7143bacdc7529158f3caa9e171c7130a345f222b1f4f77627d8e5dfb841e2c"
    )


def test_readout_v3_gated_checkpoint_is_schema8_and_fail_closed(tmp_path) -> None:
    contract = SurfaceRegionContractV3()
    model = SurfaceRegionSummaryReadoutV3(
        feature_dim=8,
        hidden_dim=6,
        base_output_mode=SURFACE_REGION_V3_GATED_RAW_PRIOR,
    ).eval()
    architecture = model.architecture(contract.digest)
    assert architecture["base_output_mode"] == SURFACE_REGION_V3_GATED_RAW_PRIOR
    assert architecture["raw_amplitude_prior_anchor_mix"] == "none"
    assert architecture["raw_radio_l2_norm_usage"] == (
        "gated_pooled_raw_amplitude_prior_only_v1"
    )
    payload = {
        "schema_version": SURFACE_SUMMARY_READOUT_V3_GATED_BASE_SCHEMA_VERSION,
        "architecture": architecture,
        "state_dict": model.state_dict(),
    }
    checkpoint = tmp_path / "readout-v3-gated.pt"
    torch.save(payload, checkpoint)
    restored, reopened = SurfaceRegionSummaryReadoutV3.from_checkpoint(checkpoint)
    assert reopened["schema_version"] == 8
    assert restored.base_output_mode == SURFACE_REGION_V3_GATED_RAW_PRIOR
    assert torch.equal(
        restored.raw_prior_gate_logit,
        model.raw_prior_gate_logit,
    )

    wrong_schema = tmp_path / "readout-v3-gated-schema7.pt"
    torch.save({**payload, "schema_version": 7}, wrong_schema)
    with pytest.raises(ValueError, match="schema/base-output mode mismatch"):
        SurfaceRegionSummaryReadoutV3.from_checkpoint(wrong_schema)

    legacy = SurfaceRegionSummaryReadoutV3(feature_dim=8, hidden_dim=6)
    wrong_legacy_schema = tmp_path / "readout-v3-legacy-schema8.pt"
    torch.save(
        {
            "schema_version": 8,
            "architecture": legacy.architecture(contract.digest),
            "state_dict": legacy.state_dict(),
        },
        wrong_legacy_schema,
    )
    with pytest.raises(ValueError, match="schema/base-output mode mismatch"):
        SurfaceRegionSummaryReadoutV3.from_checkpoint(wrong_legacy_schema)
