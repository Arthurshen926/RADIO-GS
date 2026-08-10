from __future__ import annotations

from copy import deepcopy
import inspect

import pytest
import torch

from radio_gs.interfaces import factorized_native_gauge_state_readout as interface
from radio_gs.interfaces.factorized_primitive_state import FactorizedPrimitiveState
from radio_gs.models.factorized_native_gauge_state_readout import (
    DIRECTION_ONLY,
    DIRECTION_PLUS_LOG_AMPLITUDE,
    DIRECTION_PLUS_LOG_AMPLITUDE_PLUS_FULL_STATE,
    FACTORIZED_NATIVE_READOUT_ARMS,
)


SHA = "a" * 64


def _state() -> FactorizedPrimitiveState:
    directions = torch.nn.functional.normalize(torch.randn(3, 1280), dim=-1)
    return FactorizedPrimitiveState(
        xyz=torch.zeros(4, 3),
        valid=torch.tensor([True, True, False, True]),
        global_rows=torch.tensor([0, 1, 3]),
        semantic_direction=directions,
        predicted_log_amplitude=torch.tensor([-1.0, 0.5, 1.0]),
        directional_dispersion=torch.tensor([0.1, 0.2, 0.3]),
        log_amplitude_std=torch.tensor([0.2, 0.3, 0.4]),
        observation_evidence=torch.tensor([0.8, 0.7, 0.6]),
        visibility_purity_value=torch.tensor([0.9, 0.0, 0.7]),
        visibility_purity_known=torch.tensor([True, False, True]),
        metadata={},
        source=None,
        sha256=SHA,
    )


def _normalization() -> dict:
    return interface.build_source_normalization(
        [_state()], source_state_cohort_authority_sha256=SHA
    )


def _inputs():
    return interface.gather_factorized_native_region_inputs(
        _state(),
        torch.tensor([[0, 1, 2, -1], [3, 2, -1, -1]]),
        torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        ),
        torch.tensor([0, 0]),
    )


def test_three_arms_are_strict_and_amplitude_has_one_learned_path() -> None:
    normalization = _normalization()
    models = {
        arm: interface.build_model(arm, normalization)
        for arm in FACTORIZED_NATIVE_READOUT_ARMS
    }
    assert models[DIRECTION_ONLY].log_amplitude_encoder is None
    assert models[DIRECTION_ONLY].state_encoder is None
    assert models[DIRECTION_PLUS_LOG_AMPLITUDE].log_amplitude_encoder is not None
    assert models[DIRECTION_PLUS_LOG_AMPLITUDE].state_encoder is None
    full = models[DIRECTION_PLUS_LOG_AMPLITUDE_PLUS_FULL_STATE]
    assert full.log_amplitude_encoder is not None
    assert full.state_encoder is not None
    assert full.state_encoder[1].in_features == 10
    architecture = full.architecture(interface.INTERFACE_CONTRACT_SHA256)
    assert architecture["raw_vector_reconstruction"] is False
    assert architecture["query_conditioning"] is False
    with pytest.raises(ValueError, match="unsupported"):
        interface.build_model("scene_specific", normalization)


@pytest.mark.parametrize("arm", FACTORIZED_NATIVE_READOUT_ARMS)
def test_forward_is_query_free_and_official_head_compatible(arm: str) -> None:
    model = interface.build_model(arm, _normalization())
    inputs = _inputs()
    output = model.forward_with_diagnostics(
        inputs.unit_direction,
        inputs.log_amplitude,
        inputs.state,
        inputs.state_known_mask,
        token_mask=inputs.token_mask,
        anchor_index=inputs.anchor_index,
    )
    assert output.summary_token.shape == (2, 1280)
    assert output.attention_weights.shape == (2, 4)
    assert torch.equal(
        output.attention_weights[~inputs.token_mask],
        torch.zeros_like(output.attention_weights[~inputs.token_mask]),
    )
    signature = inspect.signature(model.forward)
    assert not any(
        name in signature.parameters
        for name in ("query", "query_text", "scene_id", "benchmark")
    )
    assert not any(
        interface.source_access()[key]
        for key in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "benchmark_queries_opened",
            "benchmark_labels_opened",
            "target_heldout_opened",
            "text_queries_opened",
            "runtime_query_strings_consumed",
            "scene_identifiers_consumed_by_model",
            "per_scene_hyperparameters",
        )
    )


def test_gather_preserves_missingness_and_never_uses_invalid_direction() -> None:
    inputs = _inputs()
    assert inputs.token_mask.tolist() == [
        [True, True, False, False],
        [True, False, False, False],
    ]
    assert not inputs.state_known_mask[0, 1, 4]
    assert inputs.state[0, 1, 4].item() == 0.0
    assert not inputs.state_known_mask[~inputs.token_mask].any()
    assert inputs.unit_direction[~inputs.token_mask].count_nonzero() == 0
    assert torch.equal(inputs.log_amplitude, inputs.state[..., 0])


def test_fail_closed_on_raw_gauge_duplicate_unknown_and_schema_drift() -> None:
    model = interface.build_model(
        DIRECTION_PLUS_LOG_AMPLITUDE_PLUS_FULL_STATE, _normalization()
    )
    inputs = _inputs()
    raw = inputs.unit_direction * torch.exp(inputs.log_amplitude[..., None])
    with pytest.raises(ValueError, match="unit L2 gauge"):
        model(
            raw,
            inputs.log_amplitude,
            inputs.state,
            inputs.state_known_mask,
            token_mask=inputs.token_mask,
            anchor_index=inputs.anchor_index,
        )

    changed_amplitude = inputs.log_amplitude.clone()
    changed_amplitude[0, 0] += 0.1
    with pytest.raises(ValueError, match="differs from state column zero"):
        model(
            inputs.unit_direction,
            changed_amplitude,
            inputs.state,
            inputs.state_known_mask,
            token_mask=inputs.token_mask,
            anchor_index=inputs.anchor_index,
        )

    leaked = inputs.state.clone()
    leaked[0, 1, 4] = 0.2
    with pytest.raises(ValueError, match="unknown state values"):
        model(
            inputs.unit_direction,
            inputs.log_amplitude,
            leaked,
            inputs.state_known_mask,
            token_mask=inputs.token_mask,
            anchor_index=inputs.anchor_index,
        )

    normalization = _normalization()
    drifted = deepcopy(normalization)
    drifted["state_names"] = list(reversed(drifted["state_names"]))
    with pytest.raises(ValueError, match="contract differs"):
        interface.validate_source_normalization(drifted)


def test_direction_only_output_is_invariant_to_valid_scalar_changes() -> None:
    model = interface.build_model(DIRECTION_ONLY, _normalization()).eval()
    inputs = _inputs()
    baseline = model(
        inputs.unit_direction,
        inputs.log_amplitude,
        inputs.state,
        inputs.state_known_mask,
        token_mask=inputs.token_mask,
        anchor_index=inputs.anchor_index,
    )
    changed_state = inputs.state.clone()
    changed_amplitude = inputs.log_amplitude.clone()
    changed_amplitude[inputs.token_mask] += 2.0
    changed_state[..., 0] = changed_amplitude
    changed_state[..., 1:][inputs.token_mask] += 0.25
    changed_state = changed_state.masked_fill(~inputs.state_known_mask, 0.0)
    changed = model(
        inputs.unit_direction,
        changed_amplitude,
        changed_state,
        inputs.state_known_mask,
        token_mask=inputs.token_mask,
        anchor_index=inputs.anchor_index,
    )
    assert torch.equal(baseline, changed)
