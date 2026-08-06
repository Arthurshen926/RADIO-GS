from __future__ import annotations

import copy
import math

import pytest
import torch

from radio_gs.field.factorized_radio_contract import (
    CANONICAL_FACTORIZED_RADIO_CHECKPOINT_SCHEMA_VERSION,
    CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES,
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256,
    FactorizedRadioFieldSignature,
    aggregate_factorized_radio_observations,
    canonical_factorized_radio_contract,
    factorized_radio_checkpoint_metadata,
    factorized_radio_contract_sha256,
    parse_factorized_radio_payload,
    reliability_scalar_names_sha256,
    validate_factorized_radio_checkpoint_metadata,
)
from radio_gs.field.field_signature import FeatureSpaceSignature
from radio_gs.field.checkpoint import load_canonical_field_checkpoint
from radio_gs.field.observation_lifting_contract import (
    CANONICAL_OBSERVATION_CONTRACT_NAME,
    canonical_observation_contract,
)


def _base_signature(**overrides) -> FeatureSpaceSignature:
    values = {
        "radio_version": "c-radio_v4-h",
        "radio_checkpoint_sha256": "radio-checkpoint-sha",
        "raw_feature_dim": 3,
        "adaptor_name": "backbone",
        "adaptor_sha256": "",
        "adaptor_output_dim": 0,
        "token_type": "primitive",
        "normalization": "radio_raw_full",
    }
    values.update(overrides)
    return FeatureSpaceSignature(**values)


def _observations() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observations = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 8.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    responsibility = torch.tensor(
        [[1.0, 0.5], [2.0, 0.25], [1.0, 0.0]], dtype=torch.float32
    )
    visibility = torch.tensor([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]], dtype=torch.float32)
    return observations, responsibility, visibility


def test_factorized_contract_is_a_new_schema_and_does_not_mutate_mpr_v1() -> None:
    legacy_before = canonical_observation_contract()
    factorized = canonical_factorized_radio_contract()
    legacy_after = canonical_observation_contract()

    assert factorized["name"] == CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME
    assert factorized["name"] != CANONICAL_OBSERVATION_CONTRACT_NAME
    assert factorized["legacy_canonical_mpr_v1_unchanged"] is True
    assert legacy_before == legacy_after
    assert factorized_radio_contract_sha256(factorized) == (
        factorized_radio_contract_sha256()
    )


def test_zero_amplitude_is_excluded_before_log_and_invalid_rows_are_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations, responsibility, visibility = _observations()
    original_log = torch.log

    def positive_only_log(value: torch.Tensor) -> torch.Tensor:
        assert bool((value > 0).all())
        return original_log(value)

    monkeypatch.setattr(torch, "log", positive_only_log)
    rows = aggregate_factorized_radio_observations(
        observations, responsibility, visibility
    )

    assert rows.valid.tolist() == [True, False]
    assert torch.equal(rows.semantic_direction[1], torch.zeros(3))
    assert rows.log_amplitude[1].item() == 0.0
    assert torch.equal(rows.canonical_feature[1], torch.zeros(3))
    assert torch.equal(rows.reliability[1], torch.zeros(5))
    assert bool(torch.isfinite(rows.canonical_feature).all())
    assert bool(torch.isfinite(rows.reliability).all())

    weights = torch.tensor([1.0, 2.0, 1.0])
    expected_log_amplitude = (
        weights * torch.log(torch.tensor([2.0, 4.0, 8.0]))
    ).sum() / weights.sum()
    assert rows.log_amplitude[0] == pytest.approx(
        float(expected_log_amplitude), abs=1e-6
    )
    torch.testing.assert_close(
        rows.canonical_feature[0],
        rows.semantic_direction[0] * rows.log_amplitude[0].exp(),
    )
    assert rows.reliability_scalar_names == (FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES)
    assert rows.reliability[0, 3].item() == pytest.approx(3.0 / 4.0)


def test_factorization_is_view_permutation_invariant() -> None:
    observations, responsibility, visibility = _observations()
    original = aggregate_factorized_radio_observations(
        observations, responsibility, visibility
    )
    permutation = torch.tensor([2, 0, 1])
    permuted = aggregate_factorized_radio_observations(
        observations[permutation],
        responsibility[permutation],
        visibility[permutation],
    )

    torch.testing.assert_close(
        original.semantic_direction, permuted.semantic_direction, atol=0, rtol=0
    )
    torch.testing.assert_close(
        original.log_amplitude, permuted.log_amplitude, atol=0, rtol=0
    )
    torch.testing.assert_close(
        original.canonical_feature, permuted.canonical_feature, atol=0, rtol=0
    )
    torch.testing.assert_close(
        original.reliability, permuted.reliability, atol=0, rtol=0
    )
    assert torch.equal(original.valid, permuted.valid)


def test_uniform_scaling_only_translates_log_amplitude() -> None:
    observations, responsibility, visibility = _observations()
    baseline = aggregate_factorized_radio_observations(
        observations, responsibility, visibility
    )
    scale = 5.0
    scaled = aggregate_factorized_radio_observations(
        observations * scale, responsibility, visibility
    )

    torch.testing.assert_close(
        scaled.semantic_direction, baseline.semantic_direction, atol=1e-7, rtol=1e-7
    )
    torch.testing.assert_close(
        scaled.log_amplitude[baseline.valid],
        baseline.log_amplitude[baseline.valid] + math.log(scale),
        atol=2e-7,
        rtol=1e-7,
    )
    torch.testing.assert_close(
        scaled.reliability, baseline.reliability, atol=2e-7, rtol=1e-7
    )
    torch.testing.assert_close(
        scaled.canonical_feature,
        baseline.canonical_feature * scale,
        atol=2e-6,
        rtol=2e-6,
    )


def test_opposite_direction_changes_dispersion_without_becoming_amplitude() -> None:
    aligned = torch.tensor(
        [[[[3.0, 0.0]]], [[[3.0, 0.0]]], [[[3.0, 0.0]]], [[[3.0, 0.0]]]]
    ).reshape(4, 1, 2)
    mixed = aligned.clone()
    mixed[-1, 0] *= -1
    weights = torch.ones(4, 1)

    aligned_rows = aggregate_factorized_radio_observations(aligned, weights, weights)
    mixed_rows = aggregate_factorized_radio_observations(mixed, weights, weights)

    torch.testing.assert_close(
        mixed_rows.semantic_direction, aligned_rows.semantic_direction
    )
    torch.testing.assert_close(mixed_rows.log_amplitude, aligned_rows.log_amplitude)
    torch.testing.assert_close(
        mixed_rows.reliability[:, 2:], aligned_rows.reliability[:, 2:]
    )
    assert mixed_rows.reliability[0, 0] < aligned_rows.reliability[0, 0]
    assert mixed_rows.reliability[0, 1] > aligned_rows.reliability[0, 1]


def test_weights_are_fail_closed() -> None:
    observations, responsibility, visibility = _observations()
    visibility[0, 0] = 0.5
    with pytest.raises(ValueError, match="cannot exceed"):
        aggregate_factorized_radio_observations(
            observations, responsibility, visibility
        )

    responsibility[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        aggregate_factorized_radio_observations(
            observations, responsibility, torch.ones_like(visibility) * 3
        )


def test_evidence_is_bounded_view_support_and_not_responsibility_mass() -> None:
    observations, responsibility, visibility = _observations()
    baseline = aggregate_factorized_radio_observations(
        observations, responsibility, visibility
    )
    scaled = aggregate_factorized_radio_observations(
        observations, responsibility * 7.0, visibility * 7.0
    )

    assert baseline.reliability[0, 3].item() == pytest.approx(0.75)
    assert 0.0 <= baseline.reliability[0, 3].item() < 1.0
    torch.testing.assert_close(
        scaled.reliability[:, 3], baseline.reliability[:, 3], atol=0, rtol=0
    )
    torch.testing.assert_close(
        scaled.reliability[:, 4], baseline.reliability[:, 4], atol=0, rtol=0
    )


def test_payload_round_trip_binds_explicit_scalar_names_and_digest() -> None:
    observations, responsibility, visibility = _observations()
    rows = aggregate_factorized_radio_observations(
        observations, responsibility, visibility
    )
    payload = rows.to_payload()
    assert "semantic_direction" not in payload
    parsed = parse_factorized_radio_payload(payload)

    assert parsed.reliability_scalar_names == (
        FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES
    )
    assert payload["reliability_scalar_names_sha256"] == (
        FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
    )
    assert (
        reliability_scalar_names_sha256(payload["reliability_scalar_names"])
        == FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
    )

    renamed = copy.deepcopy(payload)
    renamed["reliability_scalar_names"][0] = "agreement"
    renamed["reliability_scalar_names_sha256"] = reliability_scalar_names_sha256(
        renamed["reliability_scalar_names"]
    )
    with pytest.raises(ValueError, match="scalar columns"):
        parse_factorized_radio_payload(renamed)

    corrupted = copy.deepcopy(payload)
    corrupted["canonical_feature"][0, 0] += 0.25
    with pytest.raises(ValueError, match="reconstruction"):
        parse_factorized_radio_payload(corrupted)

    invalid_nonzero = copy.deepcopy(payload)
    invalid_nonzero["reliability"][1, 3] = 1.0
    with pytest.raises(ValueError, match="exactly zero"):
        parse_factorized_radio_payload(invalid_nonzero)

    invalid_evidence = copy.deepcopy(payload)
    invalid_evidence["reliability"][0, 3] = 1.0
    with pytest.raises(ValueError, match="outside contract"):
        parse_factorized_radio_payload(invalid_evidence)


def test_factorized_field_signature_is_fail_closed() -> None:
    signature = FactorizedRadioFieldSignature.create(_base_signature())
    restored = FactorizedRadioFieldSignature.from_mapping(signature.to_dict())
    signature.assert_compatible(restored)

    wrong_base = FactorizedRadioFieldSignature.create(
        _base_signature(radio_checkpoint_sha256="different")
    )
    with pytest.raises(ValueError, match="incompatible"):
        signature.assert_compatible(wrong_base)

    renamed = signature.to_dict()
    renamed["reliability_scalar_names"][0] = "agreement"
    renamed["reliability_scalar_names_sha256"] = reliability_scalar_names_sha256(
        renamed["reliability_scalar_names"]
    )
    with pytest.raises(ValueError, match="reliability columns"):
        FactorizedRadioFieldSignature.from_mapping(renamed)

    with pytest.raises(ValueError, match="radio_raw_full"):
        FactorizedRadioFieldSignature.create(_base_signature(normalization="l2"))


def test_checkpoint_metadata_is_new_schema_and_fails_closed(tmp_path) -> None:
    signature = FactorizedRadioFieldSignature.create(_base_signature())
    metadata = factorized_radio_checkpoint_metadata(signature)

    assert metadata["schema_version"] == (
        CANONICAL_FACTORIZED_RADIO_CHECKPOINT_SCHEMA_VERSION
    )
    assert (
        validate_factorized_radio_checkpoint_metadata(
            metadata, expected_signature=signature
        )
        == signature
    )

    legacy = copy.deepcopy(metadata)
    legacy["schema_version"] = 1
    with pytest.raises(ValueError, match="schema-v2"):
        validate_factorized_radio_checkpoint_metadata(legacy)

    stale_digest = copy.deepcopy(metadata)
    stale_digest["field_signature_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="signature digest"):
        validate_factorized_radio_checkpoint_metadata(stale_digest)

    wrong_width = copy.deepcopy(metadata)
    wrong_width["architecture"]["reliability_dim"] -= 1
    with pytest.raises(ValueError, match="architecture"):
        validate_factorized_radio_checkpoint_metadata(wrong_width)

    other = FactorizedRadioFieldSignature.create(
        _base_signature(radio_checkpoint_sha256="other")
    )
    with pytest.raises(ValueError, match="incompatible"):
        validate_factorized_radio_checkpoint_metadata(
            metadata, expected_signature=other
        )

    # The legacy canonical loader must not silently reinterpret the new
    # version-2 authority as an ordinary canonical schema-v1 checkpoint.
    path = tmp_path / "factorized_metadata.pt"
    torch.save(metadata, path)
    with pytest.raises(ValueError, match="not a canonical RADIO field schema-v1"):
        load_canonical_field_checkpoint(path)
