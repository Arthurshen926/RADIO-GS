"""Query-free factorization contract for multi-view raw RADIO observations.

This module deliberately does not alter ``canonical-mpr-v1``.  It defines a
new cache representation and a new checkpoint metadata boundary so legacy
canonical fields cannot silently interpret factorized rows as ordinary MPR
features (or vice versa).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch

from .field_signature import FeatureSpaceSignature


CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME = "canonical-factorized-radio-v1"
CANONICAL_FACTORIZED_RADIO_CACHE_SCHEMA = "radio_gs.canonical_factorized_radio.v1"
CANONICAL_FACTORIZED_RADIO_CHECKPOINT_CONTRACT = (
    "canonical-factorized-radio-checkpoint-v1"
)
CANONICAL_FACTORIZED_RADIO_FIELD_SIGNATURE_SCHEMA = (
    "canonical-factorized-radio-field-signature-v1"
)
CANONICAL_FACTORIZED_RADIO_CHECKPOINT_SCHEMA_VERSION = 2

FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES = (
    "directional_resultant",
    "directional_dispersion",
    "log_amplitude_std",
    "observation_evidence",
    "visibility_purity",
)


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reliability_scalar_names_sha256(
    names: Sequence[str] = FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES,
) -> str:
    """Digest the ordered scalar-column names using canonical JSON."""

    values = [str(name) for name in names]
    if (
        not values
        or any(not name for name in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("factorized RADIO scalar names must be non-empty and unique")
    return _canonical_json_sha256(values)


FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256 = reliability_scalar_names_sha256()


def canonical_factorized_radio_contract() -> dict[str, Any]:
    """Return the immutable, query-free factorization declaration."""

    return {
        "name": CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
        "schema_version": 1,
        "input_feature_space": "radio_raw_full",
        "input_layout": "view_primitive_channel",
        "direction_definition": "raw_observation_divided_by_strictly_positive_l2_norm",
        "direction_fusion": "responsibility_weighted_mean_then_unit_normalize",
        "amplitude_definition": "raw_observation_l2_norm",
        "log_amplitude_fusion": "responsibility_weighted_population_mean",
        "directional_resultant": "norm_of_responsibility_weighted_mean_unit_direction",
        "directional_dispersion": "one_minus_directional_resultant",
        "log_amplitude_std": "responsibility_weighted_population_standard_deviation",
        "observation_evidence": "positive_view_count_over_positive_view_count_plus_one",
        "visibility_purity": "positive_amplitude_responsibility_mass_over_visible_mass",
        "canonical_feature": "exp_log_amplitude_times_unit_direction",
        "semantic_direction_storage": "derived_from_canonical_feature_not_duplicated",
        "zero_amplitude_policy": "exclude_before_logarithm",
        "invalid_row_policy": "all_emitted_values_exact_zero",
        "reliability_scalar_names": list(FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES),
        "reliability_scalar_names_sha256": (
            FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
        ),
        "query_independent": True,
        "legacy_canonical_mpr_v1_unchanged": True,
    }


def factorized_radio_contract_sha256(
    contract: Mapping[str, Any] | None = None,
) -> str:
    return _canonical_json_sha256(
        dict(contract)
        if contract is not None
        else canonical_factorized_radio_contract()
    )


CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256 = factorized_radio_contract_sha256()


@dataclass(frozen=True)
class FactorizedRadioRows:
    """Canonical and reliability tensors produced by the factorized contract."""

    semantic_direction: torch.Tensor
    log_amplitude: torch.Tensor
    canonical_feature: torch.Tensor
    valid: torch.Tensor
    reliability: torch.Tensor

    @property
    def reliability_scalar_names(self) -> tuple[str, ...]:
        return FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": CANONICAL_FACTORIZED_RADIO_CACHE_SCHEMA,
            "schema_version": 1,
            "contract": canonical_factorized_radio_contract(),
            "contract_sha256": CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256,
            "reliability_scalar_names": list(self.reliability_scalar_names),
            "reliability_scalar_names_sha256": (
                FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
            ),
            "log_amplitude": self.log_amplitude,
            "canonical_feature": self.canonical_feature,
            "valid": self.valid,
            "reliability": self.reliability,
        }


def aggregate_factorized_radio_observations(
    observations: torch.Tensor,
    responsibility_weights: torch.Tensor,
    visibility_weights: torch.Tensor,
) -> FactorizedRadioRows:
    """Factor raw RADIO observations without conflating direction and norm.

    Inputs are copied to float64 CPU tensors.  A zero-norm observation is
    removed from semantic responsibility *before* logarithms are evaluated;
    it can therefore reduce visibility purity but can never produce
    ``log(0)``.  The function is an artifact-construction primitive, not a
    differentiable training operator.
    """

    raw = torch.as_tensor(observations)
    responsibility = torch.as_tensor(responsibility_weights)
    visibility = torch.as_tensor(visibility_weights)
    if raw.ndim != 3 or min(raw.shape) <= 0:
        raise ValueError("factorized RADIO observations must be non-empty [V,N,D]")
    if responsibility.shape != raw.shape[:2] or visibility.shape != raw.shape[:2]:
        raise ValueError("factorized RADIO weights must align with [view,primitive]")
    if not raw.dtype.is_floating_point:
        raise TypeError("factorized RADIO observations must be floating point")
    if (
        not responsibility.dtype.is_floating_point
        or not visibility.dtype.is_floating_point
    ):
        raise TypeError("factorized RADIO weights must be floating point")

    raw64 = raw.detach().to(device="cpu", dtype=torch.float64)
    responsibility64 = responsibility.detach().to(device="cpu", dtype=torch.float64)
    visibility64 = visibility.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(raw64).all()):
        raise ValueError("factorized RADIO observations must be finite")
    if not bool(torch.isfinite(responsibility64).all()) or not bool(
        torch.isfinite(visibility64).all()
    ):
        raise ValueError("factorized RADIO weights must be finite")
    if bool((responsibility64 < 0).any()) or bool((visibility64 < 0).any()):
        raise ValueError("factorized RADIO weights must be non-negative")
    if bool((responsibility64 > visibility64).any()):
        raise ValueError("responsibility weight cannot exceed visible weight")

    _num_views, num_primitives, feature_dim = raw64.shape
    amplitude = torch.linalg.vector_norm(raw64, dim=-1)
    positive_amplitude = amplitude > 0
    effective_weight = torch.where(
        positive_amplitude, responsibility64, torch.zeros_like(responsibility64)
    )
    responsibility_mass = effective_weight.sum(dim=0)
    positive_view_count = (effective_weight > 0).sum(dim=0).to(torch.float64)
    observation_evidence = positive_view_count / (positive_view_count + 1.0)
    visible_mass = visibility64.sum(dim=0)
    safe_responsibility_mass = torch.where(
        responsibility_mass > 0,
        responsibility_mass,
        torch.ones_like(responsibility_mass),
    )
    safe_visible_mass = torch.where(
        visible_mass > 0, visible_mass, torch.ones_like(visible_mass)
    )

    unit_observation = torch.zeros_like(raw64)
    unit_observation[positive_amplitude] = raw64[positive_amplitude] / amplitude[
        positive_amplitude
    ].unsqueeze(-1)
    direction_mean = (effective_weight[..., None] * unit_observation).sum(
        dim=0
    ) / safe_responsibility_mass[..., None]
    resultant = torch.linalg.vector_norm(direction_mean, dim=-1).clamp(0.0, 1.0)
    valid = (responsibility_mass > 0) & (resultant > 0)

    semantic_direction = torch.zeros((num_primitives, feature_dim), dtype=torch.float64)
    semantic_direction[valid] = direction_mean[valid] / resultant[valid, None]

    # Assign only at strictly positive amplitudes.  In particular, this does
    # not rely on epsilon-clamped logarithms or evaluate log(0) speculatively.
    log_observation_amplitude = torch.zeros_like(amplitude)
    log_observation_amplitude[positive_amplitude] = torch.log(
        amplitude[positive_amplitude]
    )
    log_amplitude = (effective_weight * log_observation_amplitude).sum(
        dim=0
    ) / safe_responsibility_mass
    centered = log_observation_amplitude - log_amplitude[None, :]
    log_amplitude_variance = (
        (effective_weight * centered.square()).sum(dim=0) / safe_responsibility_mass
    ).clamp_min(0.0)
    log_amplitude_std = torch.sqrt(log_amplitude_variance)

    visibility_purity = torch.where(
        visible_mass > 0,
        responsibility_mass / safe_visible_mass,
        torch.zeros_like(visible_mass),
    ).clamp(0.0, 1.0)
    directional_dispersion = 1.0 - resultant
    canonical_feature = torch.exp(log_amplitude)[..., None] * semantic_direction

    for values in (
        semantic_direction,
        log_amplitude,
        resultant,
        directional_dispersion,
        log_amplitude_std,
        observation_evidence,
        visibility_purity,
        canonical_feature,
    ):
        values[~valid] = 0.0

    reliability = torch.stack(
        (
            resultant,
            directional_dispersion,
            log_amplitude_std,
            observation_evidence,
            visibility_purity,
        ),
        dim=-1,
    )
    if not bool(torch.isfinite(canonical_feature).all()) or not bool(
        torch.isfinite(reliability).all()
    ):
        raise ValueError("factorized RADIO aggregation produced non-finite values")
    return FactorizedRadioRows(
        semantic_direction=semantic_direction.float(),
        log_amplitude=log_amplitude.float(),
        canonical_feature=canonical_feature.float(),
        valid=valid,
        reliability=reliability.float(),
    )


def parse_factorized_radio_payload(payload: Mapping[str, Any]) -> FactorizedRadioRows:
    """Parse a serialized factorized cache and validate every derived field."""

    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "reliability_scalar_names",
        "reliability_scalar_names_sha256",
        "log_amplitude",
        "canonical_feature",
        "valid",
        "reliability",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("factorized RADIO cache fields differ")
    if (
        payload.get("schema") != CANONICAL_FACTORIZED_RADIO_CACHE_SCHEMA
        or payload.get("schema_version") != 1
    ):
        raise ValueError("not a canonical factorized RADIO cache schema-v1 payload")
    expected_contract = canonical_factorized_radio_contract()
    if payload.get("contract") != expected_contract or payload.get(
        "contract_sha256"
    ) != factorized_radio_contract_sha256(expected_contract):
        raise ValueError("factorized RADIO cache contract differs")
    names = payload.get("reliability_scalar_names")
    if (
        names != list(FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES)
        or payload.get("reliability_scalar_names_sha256")
        != FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
    ):
        raise ValueError("factorized RADIO reliability scalar columns differ")

    log_amplitude = payload["log_amplitude"]
    canonical = payload["canonical_feature"]
    valid = payload["valid"]
    reliability = payload["reliability"]
    tensors = (log_amplitude, canonical, valid, reliability)
    if not all(torch.is_tensor(value) for value in tensors):
        raise TypeError("factorized RADIO cache values must be tensors")
    if canonical.ndim != 2 or min(canonical.shape) <= 0:
        raise ValueError("factorized RADIO canonical feature shape differs")
    num_rows, feature_dim = canonical.shape
    if (
        log_amplitude.shape != (num_rows,)
        or valid.shape != (num_rows,)
        or reliability.shape
        != (num_rows, len(FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES))
    ):
        raise ValueError("factorized RADIO cache tensor shapes differ")
    if valid.dtype != torch.bool:
        raise TypeError("factorized RADIO validity must be boolean")
    for value in (log_amplitude, canonical, reliability):
        if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
            raise ValueError(
                "factorized RADIO cache tensors must be finite floating point"
            )

    invalid = ~valid
    for value in (log_amplitude, canonical, reliability):
        if not bool((value[invalid] == 0).all()):
            raise ValueError("factorized RADIO invalid rows must be exactly zero")
    log_amplitude64 = log_amplitude.double()
    canonical64 = canonical.double()
    reliability64 = reliability.double()
    if bool(valid.any()):
        canonical_norm = torch.linalg.vector_norm(canonical64[valid], dim=-1)
        expected_norm = torch.exp(log_amplitude64[valid])
        tolerance = 2e-3 if canonical.dtype in {torch.float16, torch.bfloat16} else 2e-5
        if not torch.allclose(
            canonical_norm, expected_norm, atol=tolerance, rtol=tolerance
        ):
            raise ValueError(
                "factorized RADIO canonical amplitude reconstruction differs"
            )
        resultant = reliability64[valid, 0]
        dispersion = reliability64[valid, 1]
        amplitude_std = reliability64[valid, 2]
        evidence = reliability64[valid, 3]
        purity = reliability64[valid, 4]
        if (
            bool((resultant <= 0).any())
            or bool((resultant > 1).any())
            or bool((dispersion < 0).any())
            or bool((dispersion >= 1).any())
            or bool((amplitude_std < 0).any())
            or bool((evidence <= 0).any())
            or bool((evidence >= 1).any())
            or bool((purity < 0).any())
            or bool((purity > 1).any())
        ):
            raise ValueError("factorized RADIO reliability values are outside contract")
        if not torch.allclose(dispersion, 1.0 - resultant, atol=2e-6, rtol=2e-6):
            raise ValueError("factorized RADIO directional dispersion differs")
    semantic_direction = torch.zeros_like(canonical)
    if bool(valid.any()):
        # CPU does not implement ``clamp_min`` for fp16.  Derive the direction
        # in float32 and cast only the final, bounded unit vector back to the
        # persisted canonical dtype.
        canonical_valid_float = canonical[valid].float()
        semantic_direction[valid] = (
            canonical_valid_float
            / torch.linalg.vector_norm(
                canonical_valid_float, dim=-1, keepdim=True
            ).clamp_min(torch.finfo(torch.float32).tiny)
        ).to(canonical.dtype)
    return FactorizedRadioRows(
        semantic_direction=semantic_direction,
        log_amplitude=log_amplitude,
        canonical_feature=canonical,
        valid=valid,
        reliability=reliability,
    )


@dataclass(frozen=True)
class FactorizedRadioFieldSignature:
    """Fail-closed field signature for the new factorized representation."""

    base_feature_signature: FeatureSpaceSignature
    factorized_radio_contract_sha256: str
    reliability_scalar_names: tuple[str, ...]
    reliability_scalar_names_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.base_feature_signature, FeatureSpaceSignature):
            raise TypeError("base_feature_signature must be a FeatureSpaceSignature")
        if self.base_feature_signature.normalization != "radio_raw_full":
            raise ValueError(
                "factorized RADIO requires a radio_raw_full base signature"
            )
        if self.base_feature_signature.token_type != "primitive":
            raise ValueError(
                "factorized RADIO field signature must describe primitive rows"
            )
        if self.factorized_radio_contract_sha256 != (
            CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256
        ):
            raise ValueError("factorized RADIO field contract digest differs")
        names = tuple(str(name) for name in self.reliability_scalar_names)
        object.__setattr__(self, "reliability_scalar_names", names)
        if names != FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES or (
            self.reliability_scalar_names_sha256
            != FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
        ):
            raise ValueError("factorized RADIO field reliability columns differ")

    @classmethod
    def create(
        cls, base_feature_signature: FeatureSpaceSignature
    ) -> "FactorizedRadioFieldSignature":
        return cls(
            base_feature_signature=base_feature_signature,
            factorized_radio_contract_sha256=(
                CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256
            ),
            reliability_scalar_names=FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES,
            reliability_scalar_names_sha256=(
                FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CANONICAL_FACTORIZED_RADIO_FIELD_SIGNATURE_SCHEMA,
            "schema_version": 1,
            "base_feature_signature": self.base_feature_signature.to_dict(),
            "factorized_radio_contract_sha256": (self.factorized_radio_contract_sha256),
            "reliability_scalar_names": list(self.reliability_scalar_names),
            "reliability_scalar_names_sha256": (self.reliability_scalar_names_sha256),
        }

    @property
    def digest(self) -> str:
        return _canonical_json_sha256(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FactorizedRadioFieldSignature":
        required = {
            "schema",
            "schema_version",
            "base_feature_signature",
            "factorized_radio_contract_sha256",
            "reliability_scalar_names",
            "reliability_scalar_names_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("factorized RADIO field signature fields differ")
        if (
            value.get("schema") != CANONICAL_FACTORIZED_RADIO_FIELD_SIGNATURE_SCHEMA
            or value.get("schema_version") != 1
        ):
            raise ValueError("factorized RADIO field signature schema differs")
        base = value.get("base_feature_signature")
        if not isinstance(base, Mapping):
            raise TypeError("factorized RADIO base feature signature is malformed")
        return cls(
            base_feature_signature=FeatureSpaceSignature.from_mapping(base),
            factorized_radio_contract_sha256=str(
                value.get("factorized_radio_contract_sha256", "")
            ),
            reliability_scalar_names=tuple(value.get("reliability_scalar_names", ())),
            reliability_scalar_names_sha256=str(
                value.get("reliability_scalar_names_sha256", "")
            ),
        )

    def assert_compatible(self, other: "FactorizedRadioFieldSignature") -> None:
        if not isinstance(other, FactorizedRadioFieldSignature):
            raise TypeError("other must be a FactorizedRadioFieldSignature")
        if self.to_dict() != other.to_dict():
            raise ValueError("incompatible factorized RADIO field signatures")


def factorized_radio_checkpoint_metadata(
    signature: FactorizedRadioFieldSignature,
) -> dict[str, Any]:
    """Construct the minimal metadata authority for a future field checkpoint."""

    if not isinstance(signature, FactorizedRadioFieldSignature):
        raise TypeError("signature must be a FactorizedRadioFieldSignature")
    return {
        "schema_version": CANONICAL_FACTORIZED_RADIO_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_contract": CANONICAL_FACTORIZED_RADIO_CHECKPOINT_CONTRACT,
        "factorized_radio_contract": canonical_factorized_radio_contract(),
        "factorized_radio_contract_sha256": (
            CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256
        ),
        "field_signature": signature.to_dict(),
        "field_signature_sha256": signature.digest,
        "architecture": {
            "feature_dim": int(signature.base_feature_signature.raw_feature_dim),
            "reliability_dim": len(FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES),
            "canonical_feature_formula": "exp_log_amplitude_times_unit_direction",
        },
    }


def validate_factorized_radio_checkpoint_metadata(
    payload: Mapping[str, Any],
    *,
    expected_signature: FactorizedRadioFieldSignature | None = None,
) -> FactorizedRadioFieldSignature:
    """Validate the version-2 checkpoint boundary without loading model state."""

    required = {
        "schema_version",
        "checkpoint_contract",
        "factorized_radio_contract",
        "factorized_radio_contract_sha256",
        "field_signature",
        "field_signature_sha256",
        "architecture",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("factorized RADIO checkpoint metadata fields differ")
    if (
        payload.get("schema_version")
        != CANONICAL_FACTORIZED_RADIO_CHECKPOINT_SCHEMA_VERSION
        or payload.get("checkpoint_contract")
        != CANONICAL_FACTORIZED_RADIO_CHECKPOINT_CONTRACT
    ):
        raise ValueError("not a canonical factorized RADIO checkpoint schema-v2")
    contract = canonical_factorized_radio_contract()
    if (
        payload.get("factorized_radio_contract") != contract
        or payload.get("factorized_radio_contract_sha256")
        != CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256
    ):
        raise ValueError("factorized RADIO checkpoint contract differs")
    signature_value = payload.get("field_signature")
    if not isinstance(signature_value, Mapping):
        raise TypeError("factorized RADIO checkpoint field signature is malformed")
    signature = FactorizedRadioFieldSignature.from_mapping(signature_value)
    if payload.get("field_signature_sha256") != signature.digest:
        raise ValueError("factorized RADIO checkpoint field signature digest differs")
    architecture = payload.get("architecture")
    expected_architecture = {
        "feature_dim": int(signature.base_feature_signature.raw_feature_dim),
        "reliability_dim": len(FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES),
        "canonical_feature_formula": "exp_log_amplitude_times_unit_direction",
    }
    if architecture != expected_architecture:
        raise ValueError("factorized RADIO checkpoint architecture differs")
    if expected_signature is not None:
        expected_signature.assert_compatible(signature)
    return signature
