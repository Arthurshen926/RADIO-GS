from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from radio_gs.field import FeatureSpaceSignature
from radio_gs.field.factorized_radio_contract import (
    FactorizedRadioFieldSignature,
)
import radio_gs.interfaces.factorized_primitive_state as state_module
from radio_gs.interfaces.factorized_primitive_state import (
    FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256,
    FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256_V1,
    FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES,
    FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256,
    FactorizedFieldSupport,
    FactorizedPrimitiveState,
    build_factorized_primitive_state,
    factorized_primitive_state_contract,
    load_factorized_field_support,
    load_factorized_primitive_state,
    validate_factorized_primitive_state_payload,
)
from radio_gs.training.factorized_radio_cache import (
    FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY,
    FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY,
)


def _base_signature() -> FeatureSpaceSignature:
    return FeatureSpaceSignature(
        radio_version="test",
        radio_checkpoint_sha256="a" * 64,
        raw_feature_dim=1280,
        token_type="primitive",
        normalization="radio_raw_full",
        crop_policy="training_views_canonical_factorized_radio_v1",
    )


def _float32_rows_sha256(values: torch.Tensor) -> str:
    return hashlib.sha256(
        values.float().contiguous().numpy().astype("<f4", copy=False).tobytes()
    ).hexdigest()


class _Field(torch.nn.Module):
    def __init__(self, values: torch.Tensor, signature: FeatureSpaceSignature) -> None:
        super().__init__()
        self.register_buffer("values", values.float())
        self.num_gaussians = len(values)
        self.signature = signature

    def radio_features(self, rows: torch.Tensor) -> torch.Tensor:
        return self.values[rows]


def _support(*, exact_marginal: bool = False) -> FactorizedFieldSupport:
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    valid = torch.tensor([True, False, True])
    features = torch.zeros(3, 1280)
    features[0, 0] = 2.0
    features[2, 1] = 4.0
    base = _base_signature()
    signature = FactorizedRadioFieldSignature.create(base)
    reliability = torch.tensor(
        [
            [0.8, 0.2, 0.1, 0.5, 0.25 if exact_marginal else 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.4, 0.2, 0.75, 0.75 if exact_marginal else 0.0],
        ],
        dtype=torch.float32,
    )
    geometry = {
        "num_gaussians": 3,
        "xyz_sha256": _float32_rows_sha256(xyz),
    }
    cache = SimpleNamespace(
        xyz=xyz,
        valid=valid,
        reliability=reliability,
        geometry_fingerprint=geometry,
        sha256="c" * 64,
        source=Path("/tmp/factorized.pt"),
        metadata={
            "registration_responsibility_cache_sha256": "d" * 64,
            "visibility_purity_authority": {
                **(
                    FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY
                    if exact_marginal
                    else FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY
                ),
                "registration_responsibility_cache_sha256": "d" * 64,
            },
        },
    )
    payload = {
        "feature_output_bundle_sha256": "e" * 64,
    }
    return FactorizedFieldSupport(
        field=_Field(features, base),
        field_payload=payload,
        field_signature=signature,
        field_checkpoint=Path("/tmp/field.pt"),
        field_checkpoint_sha256="f" * 64,
        cache=cache,
    )


def test_factorized_primitive_state_contract_is_ordered_and_versioned() -> None:
    contract = factorized_primitive_state_contract()
    assert contract["schema_version"] == 2
    assert contract["scalar_names"] == list(
        FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES
    )
    assert contract["scalar_names_sha256"] == (
        FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256
    )
    assert len(FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256) == 64
    assert FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256 != (
        FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256_V1
    )


def test_unknown_purity_value_cannot_change_encoding_input() -> None:
    common = dict(
        xyz=torch.zeros(2, 3),
        valid=torch.ones(2, dtype=torch.bool),
        global_rows=torch.arange(2),
        semantic_direction=torch.zeros(2, 1280, dtype=torch.float16),
        predicted_log_amplitude=torch.tensor([1.0, 2.0]),
        directional_dispersion=torch.tensor([0.1, 0.2]),
        log_amplitude_std=torch.tensor([0.3, 0.4]),
        observation_evidence=torch.tensor([0.5, 0.6]),
        visibility_purity_known=torch.zeros(2, dtype=torch.bool),
        metadata={},
    )
    zero = FactorizedPrimitiveState(
        **common, visibility_purity_value=torch.zeros(2)
    )
    perturbed = FactorizedPrimitiveState(
        **common, visibility_purity_value=torch.tensor([0.25, 1.0])
    )
    assert torch.equal(zero.scalar_encoding_input(), perturbed.scalar_encoding_input())
    assert torch.equal(zero.scalar_encoding_input()[:, 4], torch.zeros(2))
    assert torch.equal(zero.scalar_encoding_input()[:, 5], torch.zeros(2))
    assert torch.equal(
        zero.legacy_geometric_reliability(),
        perturbed.legacy_geometric_reliability(),
    )


def test_build_factorized_primitive_state_exports_field_direction_and_amplitude() -> None:
    state = build_factorized_primitive_state(_support(), chunk_size=1)
    assert torch.equal(state.global_rows, torch.tensor([0, 2]))
    assert state.semantic_direction.dtype == torch.float16
    assert torch.equal(state.semantic_direction[:, :2], torch.eye(2).half())
    assert torch.allclose(
        state.predicted_log_amplitude, torch.log(torch.tensor([2.0, 4.0]))
    )


def test_exact_marginal_state_retains_measured_purity_as_known() -> None:
    state = build_factorized_primitive_state(
        _support(exact_marginal=True), chunk_size=1
    )
    assert bool(state.visibility_purity_known.all())
    assert torch.equal(
        state.visibility_purity_value, torch.tensor([0.25, 0.75])
    )
    assert torch.equal(
        state.scalar_encoding_input()[:, 4], torch.tensor([0.25, 0.75])
    )
    assert torch.equal(state.scalar_encoding_input()[:, 5], torch.ones(2))
    # V2 compatibility reliability deliberately remains purity-independent.
    assert torch.allclose(
        state.legacy_geometric_reliability(),
        torch.sqrt(torch.tensor([0.8 * 0.5, 0.6 * 0.75])),
    )


def test_v1_top1_payload_remains_loadable_without_rewriting_contract() -> None:
    state = build_factorized_primitive_state(_support())
    state = FactorizedPrimitiveState(
        **{
            **state.__dict__,
            "metadata": {
                **state.metadata,
                "source": "factorized_primitive_state_v1",
            },
            "schema": "radio_gs.factorized_primitive_state.v1",
            "schema_version": 1,
        }
    )
    payload = state.to_payload()
    assert payload["contract_sha256"] == FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256_V1
    validate_factorized_primitive_state_payload(payload)
    assert torch.equal(state.directional_dispersion, torch.tensor([0.2, 0.4]))
    assert torch.equal(state.observation_evidence, torch.tensor([0.5, 0.75]))
    assert not bool(state.visibility_purity_known.any())
    assert not bool(state.visibility_purity_value.any())
    assert torch.allclose(
        state.legacy_geometric_reliability(),
        torch.sqrt(torch.tensor([0.8 * 0.5, 0.6 * 0.75])),
    )


def test_state_payload_rejects_row_scalar_and_purity_authority_changes() -> None:
    payload = build_factorized_primitive_state(_support()).to_payload()
    validate_factorized_primitive_state_payload(payload)

    wrong_rows = copy.deepcopy(payload)
    wrong_rows["global_rows"] = torch.tensor([2, 0])
    with pytest.raises(ValueError, match="tensor layout"):
        validate_factorized_primitive_state_payload(wrong_rows)

    wrong_digest = copy.deepcopy(payload)
    wrong_digest["scalar_names_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="contract"):
        validate_factorized_primitive_state_payload(wrong_digest)

    wrong_purity = copy.deepcopy(payload)
    wrong_purity["visibility_purity_value"][0] = 0.5
    with pytest.raises(ValueError, match="purity/scalar"):
        validate_factorized_primitive_state_payload(wrong_purity)


def test_state_sidecar_loader_binds_field_cache_xyz_and_valid(tmp_path: Path) -> None:
    payload = build_factorized_primitive_state(_support()).to_payload()
    path = tmp_path / "state.pt"
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    state = load_factorized_primitive_state(
        path,
        expected_sha256=digest,
        expected_field_checkpoint_sha256="f" * 64,
        expected_factorized_radio_cache_sha256="c" * 64,
        expected_xyz=payload["xyz"],
        expected_valid=payload["valid"],
    )
    assert state.sha256 == digest
    with pytest.raises(ValueError, match="field SHA"):
        load_factorized_primitive_state(
            path,
            expected_sha256=digest,
            expected_field_checkpoint_sha256="0" * 64,
        )


def test_shared_schema_v2_support_loader_has_no_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "field.pt"
    checkpoint.write_bytes(b"field")
    base = _base_signature()
    signature = FactorizedRadioFieldSignature.create(base)
    field = SimpleNamespace(num_gaussians=3, signature=base)
    geometry = {"num_gaussians": 3, "xyz_sha256": "b" * 64}
    cache = SimpleNamespace(
        xyz=torch.zeros(3, 3),
        valid=torch.tensor([True, False, True]),
        geometry_fingerprint=geometry,
        sha256="c" * 64,
        source=tmp_path / "cache.pt",
        metadata={"registration_responsibility_cache_sha256": "d" * 64},
    )
    payload = {
        "mpr_cache": str(cache.source),
        "mpr_cache_sha256": cache.sha256,
        "factorized_cache_sha256": cache.sha256,
        "feature_output_bundle_sha256": "e" * 64,
        "geometry_fingerprint": geometry,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    monkeypatch.setattr(state_module, "_sha256_file", lambda _path: "f" * 64)
    monkeypatch.setattr(
        state_module,
        "load_factorized_canonical_field_checkpoint",
        lambda *_args, **_kwargs: (field, payload, signature),
    )
    monkeypatch.setattr(
        state_module,
        "load_factorized_radio_training_cache",
        lambda *_args, **_kwargs: cache,
    )
    loaded = load_factorized_field_support(
        checkpoint,
        expected_field_checkpoint_sha256="f" * 64,
    )
    assert loaded.cache is cache
    assert loaded.lineage["factorized_radio_cache_sha256"] == "c" * 64
    assert loaded.lineage["factorized_radio_field_signature_sha256"] == (
        signature.digest
    )
