import torch

from radio_gs.interfaces.factorized_native_region_relation import (
    FEATURE_NAMES,
    FactorizedNativeRegionSummary,
    factorized_native_pair_features,
    factorized_native_region_relation_features,
    factorized_native_region_summaries,
    interface_contract,
)


def _inputs():
    direction = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.6, 0.8], [0.0, 0.0]],
            [[0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]],
        ]
    )
    amplitude = torch.tensor([[1.0, 2.0, 0.0], [1.5, 2.5, 0.0], [4.0, 5.0, 0.0]])
    state = torch.zeros(3, 3, 6)
    state[..., 0] = amplitude
    state[:, :2, 1] = torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.7, 0.6]])
    state[:, :2, 2] = 0.1
    state[:, :2, 3] = torch.tensor([[0.8, 0.6], [0.9, 0.7], [0.4, 0.2]])
    state[:, :2, 4] = torch.tensor([[0.9, 0.7], [0.8, 0.6], [0.0, 0.0]])
    known = torch.zeros(3, 3, 6, dtype=torch.bool)
    known[:, :2] = True
    known[2, :2, 4] = False
    state[2, :2, 4] = 0.0
    state[:, :2, 5] = known[:, :2, 4].float()
    mask = torch.tensor([[True, True, False]] * 3)
    anchor = torch.tensor([0, 0, 0])
    pairs = torch.tensor([[0, 0, 1], [1, 2, 2]])
    return direction, amplitude, state, known, mask, anchor, pairs


def test_native_relation_is_symmetric_missingness_safe_and_query_free():
    direction, amplitude, state, known, mask, anchor, pairs = _inputs()
    output = factorized_native_region_relation_features(
        unit_direction=direction,
        log_amplitude=amplitude,
        state=state,
        state_known_mask=known,
        token_mask=mask,
        anchor_index=anchor,
        pair_indices=pairs,
    )
    assert output.pair_features.shape == (3, len(FEATURE_NAMES))
    assert torch.allclose(output.semantic_direction_concentration[:2], torch.tensor([2**-0.5, (3.2**0.5) / 2]), atol=1e-6)
    assert output.visibility_purity_known_fraction.tolist() == [1.0, 1.0, 0.0]
    assert output.mean_visibility_purity_known_value[2].item() == 0.0
    contract = interface_contract()
    assert contract["legacy_accepted_v2_default_changed"] is False
    assert contract["source_access"]["text_queries_opened"] is False
    assert contract["input_gauge"]["log_amplitude"].startswith("separate_scalar")


def test_native_relation_is_invariant_to_token_permutation():
    direction, amplitude, state, known, mask, anchor, pairs = _inputs()
    original = factorized_native_region_relation_features(
        unit_direction=direction,
        log_amplitude=amplitude,
        state=state,
        state_known_mask=known,
        token_mask=mask,
        anchor_index=anchor,
        pair_indices=pairs,
    )
    order = torch.tensor([1, 0, 2])
    permuted = factorized_native_region_relation_features(
        unit_direction=direction[:, order],
        log_amplitude=amplitude[:, order],
        state=state[:, order],
        state_known_mask=known[:, order],
        token_mask=mask[:, order],
        anchor_index=torch.ones(3, dtype=torch.long),
        pair_indices=pairs,
    )
    assert torch.equal(original.pair_indices, permuted.pair_indices)
    assert torch.allclose(original.pair_features, permuted.pair_features)


def test_native_relation_rejects_raw_gauge_or_unknown_nonzero_value():
    direction, amplitude, state, known, mask, anchor, pairs = _inputs()
    direction[0, 0] *= 2
    try:
        factorized_native_region_relation_features(
            unit_direction=direction,
            log_amplitude=amplitude,
            state=state,
            state_known_mask=known,
            token_mask=mask,
            anchor_index=anchor,
            pair_indices=pairs,
        )
    except ValueError as error:
        assert "carriers" in str(error)
    else:
        raise AssertionError("non-unit direction was accepted")

    direction, amplitude, state, known, mask, anchor, pairs = _inputs()
    state[2, 0, 4] = 0.5
    try:
        factorized_native_region_relation_features(
            unit_direction=direction,
            log_amplitude=amplitude,
            state=state,
            state_known_mask=known,
            token_mask=mask,
            anchor_index=anchor,
            pair_indices=pairs,
        )
    except ValueError as error:
        assert "carriers" in str(error)
    else:
        raise AssertionError("unknown nonzero state was accepted")


def test_chunked_region_summaries_match_monolithic_relation_exactly():
    direction, amplitude, state, known, mask, anchor, pairs = _inputs()
    monolithic = factorized_native_region_relation_features(
        unit_direction=direction,
        log_amplitude=amplitude,
        state=state,
        state_known_mask=known,
        token_mask=mask,
        anchor_index=anchor,
        pair_indices=pairs,
    )
    chunks = []
    for selected in (slice(0, 2), slice(2, 3)):
        chunks.append(
            factorized_native_region_summaries(
                unit_direction=direction[selected],
                log_amplitude=amplitude[selected],
                state=state[selected],
                state_known_mask=known[selected],
                token_mask=mask[selected],
                anchor_index=anchor[selected],
            )
        )
    summary = FactorizedNativeRegionSummary(
        **{
            name: torch.cat([getattr(chunk, name) for chunk in chunks], dim=0)
            for name in FactorizedNativeRegionSummary.__dataclass_fields__
        }
    )
    chunked = factorized_native_pair_features(summary, pairs)
    assert torch.equal(chunked, monolithic.pair_features)
