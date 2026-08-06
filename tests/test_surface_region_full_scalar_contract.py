from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from radio_gs.field import FeatureSpaceSignature
from radio_gs.field.factorized_radio_contract import FactorizedRadioFieldSignature
from radio_gs.interfaces.factorized_primitive_state import (
    FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256,
    FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES,
    FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256,
    FactorizedFieldSupport,
    build_factorized_primitive_state,
)
from radio_gs.interfaces.surface_region_full_scalar_contract import (
    SURFACE_REGION_FULL_SCALAR_CONTRACT_SHA256,
    SURFACE_REGION_FULL_SCALAR_DIM,
    SURFACE_REGION_FULL_SCALAR_NAMES,
    aggregate_surface_region_full_scalars,
    apply_full_scalar_normalization,
    build_full_scalar_normalization_authority,
    build_full_scalar_support_routing,
    surface_region_full_scalar_contract,
    validate_full_scalar_normalization_authority,
)
from radio_gs.training.factorized_radio_cache import (
    FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY,
    FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY,
)


class _Field(torch.nn.Module):
    def __init__(self, values: torch.Tensor, signature: FeatureSpaceSignature) -> None:
        super().__init__()
        self.register_buffer("values", values.float())
        self.num_gaussians = len(values)
        self.signature = signature

    def radio_features(self, rows: torch.Tensor) -> torch.Tensor:
        return self.values[rows]


def _xyz_sha(values: torch.Tensor) -> str:
    array = values.float().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _state(*, exact: bool = True):
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    valid = torch.tensor([True, False, True])
    features = torch.zeros(3, 1280)
    features[0, 0] = 2.0
    features[2, 1] = 4.0
    base = FeatureSpaceSignature(
        radio_version="test",
        radio_checkpoint_sha256="a" * 64,
        raw_feature_dim=1280,
        token_type="primitive",
        normalization="radio_raw_full",
        crop_policy="training_views_canonical_factorized_radio_v1",
    )
    reliability = torch.tensor(
        [
            [0.8, 0.2, 0.1, 0.5, 0.25 if exact else 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.4, 0.2, 0.75, 0.75 if exact else 0.0],
        ]
    )
    purity_authority = (
        FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY
        if exact
        else FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY
    )
    support = FactorizedFieldSupport(
        field=_Field(features, base),
        field_payload={"feature_output_bundle_sha256": "e" * 64},
        field_signature=FactorizedRadioFieldSignature.create(base),
        field_checkpoint=Path("/tmp/field.pt"),
        field_checkpoint_sha256="f" * 64,
        cache=SimpleNamespace(
            xyz=xyz,
            valid=valid,
            reliability=reliability,
            geometry_fingerprint={
                "num_gaussians": 3,
                "xyz_sha256": _xyz_sha(xyz),
            },
            sha256="c" * 64,
            source=Path("/tmp/factorized.pt"),
            metadata={
                "registration_responsibility_cache_sha256": "d" * 64,
                "visibility_purity_authority": {
                    **purity_authority,
                    "registration_responsibility_cache_sha256": "d" * 64,
                },
            },
        ),
    )
    return build_factorized_primitive_state(support, chunk_size=1)


def test_contract_strictly_binds_state_v2_scalar_authority() -> None:
    contract = surface_region_full_scalar_contract()
    assert contract["factorized_primitive_state_contract_sha256"] == (
        FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256
    )
    assert contract["factorized_primitive_state_scalar_names"] == list(
        FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES
    )
    assert contract["factorized_primitive_state_scalar_names_sha256"] == (
        FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256
    )
    assert len(SURFACE_REGION_FULL_SCALAR_CONTRACT_SHA256) == 64
    assert len(SURFACE_REGION_FULL_SCALAR_NAMES) == SURFACE_REGION_FULL_SCALAR_DIM


def test_support_routing_has_base_fallback_and_exact_abstention() -> None:
    state = _state()
    routing = build_full_scalar_support_routing(
        torch.tensor([True, True, False]), state
    )
    assert torch.equal(routing.overlap_mask, torch.tensor([True, False, False]))
    assert torch.equal(
        routing.base_only_fallback_mask, torch.tensor([False, True, False])
    )
    assert torch.equal(
        routing.exact_only_abstain_mask, torch.tensor([False, False, True])
    )


def test_overlap_summary_is_anchor_weighted_mean_std_and_padding_safe() -> None:
    state = _state()
    result = aggregate_surface_region_full_scalars(
        state,
        torch.ones(3, dtype=torch.bool),
        torch.tensor([0, 2, 2]),
        torch.tensor([True, True, False]),
        0,
    )
    scalars = state.scalar_encoding_input()
    weights = state.legacy_geometric_reliability()
    expected_mean = (scalars * weights[:, None]).sum(dim=0) / weights.sum()
    expected_std = torch.sqrt(
        ((scalars - expected_mean).square() * weights[:, None]).sum(dim=0)
        / weights.sum()
    )
    torch.testing.assert_close(result.summary[:6], scalars[0])
    torch.testing.assert_close(result.summary[6:12], expected_mean)
    torch.testing.assert_close(result.summary[12:], expected_std)
    assert not bool(result.token_scalars[2].any())
    assert not bool(result.token_overlap_mask[2])
    assert bool(result.use_full_scalar_mask)


def test_base_only_anchor_falls_back_and_exact_only_anchor_abstains() -> None:
    state = _state()
    fallback = aggregate_surface_region_full_scalars(
        state,
        torch.tensor([True, True, False]),
        torch.tensor([1, 2]),
        torch.tensor([True, False]),
        0,
    )
    assert bool(fallback.base_fallback_mask)
    assert not bool(fallback.summary.any())

    abstain = aggregate_surface_region_full_scalars(
        state,
        torch.tensor([True, True, False]),
        torch.tensor([2]),
        torch.tensor([True]),
        0,
    )
    assert bool(abstain.abstain_mask)
    assert not bool(abstain.summary.any())


def test_top1_unknown_purity_cannot_enter_full_scalar_overlay() -> None:
    with pytest.raises(ValueError, match="exact-marginal measured-purity"):
        build_full_scalar_support_routing(torch.ones(3, dtype=torch.bool), _state(exact=False))


def test_source_normalization_freezes_median_mad_and_routes_ood_to_base() -> None:
    source = torch.stack(
        (
            torch.zeros(SURFACE_REGION_FULL_SCALAR_DIM),
            torch.ones(SURFACE_REGION_FULL_SCALAR_DIM),
            torch.full((SURFACE_REGION_FULL_SCALAR_DIM,), 2.0),
        )
    )
    source[:, 0] = 5.0
    authority = build_full_scalar_normalization_authority(
        source,
        torch.ones(3, dtype=torch.bool),
        source_state_cohort_sha256="a" * 64,
    )
    validate_full_scalar_normalization_authority(authority)
    torch.testing.assert_close(authority["median"], source[1])
    assert bool(authority["constant_coordinate_mask"][0])

    queries = torch.stack((source[1], source[2], source[1], source[1]))
    queries[2, 1] = 10.0
    queries[3, 0] = 6.0
    result = apply_full_scalar_normalization(
        queries,
        torch.tensor([True, False, True, True]),
        authority,
    )
    assert torch.equal(result.ood_mask, torch.tensor([False, False, True, True]))
    assert torch.equal(
        result.base_fallback_mask, torch.tensor([False, False, True, True])
    )
    assert not bool(result.normalized[1:].any())


def test_normalization_authority_rejects_ood_rule_rewrite() -> None:
    source = torch.stack(
        (torch.zeros(SURFACE_REGION_FULL_SCALAR_DIM), torch.ones(SURFACE_REGION_FULL_SCALAR_DIM))
    )
    authority = build_full_scalar_normalization_authority(
        source,
        torch.ones(2, dtype=torch.bool),
        source_state_cohort_sha256="a" * 64,
    )
    changed = copy.deepcopy(authority)
    changed["ood_rule"]["threshold"] = "benchmark_tuned"
    with pytest.raises(ValueError, match="contract differs"):
        validate_full_scalar_normalization_authority(changed)
