"""Authority and migration boundary for Universal Field v1.

Universal Field v1 preserves the learned D512/L512 RADIO mapping exactly and
adds the five query-independent factorized-MPR reliability scalars to the
deployment checkpoint.  The scalars are query-calibration state; they are not
fed back through the already-trained primitive fusion network.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

import torch

from radio_gs.field.factorized_radio_contract import (
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES,
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256,
)


UNIVERSAL_FIELD_ID = "radio-gs-universal-field-v1"
PRIMITIVE_READOUT_ID = "radio-gs-primitive-readout-v0"
UNIVERSAL_FIELD_CHECKPOINT_CONTRACT = "canonical-universal-radio-checkpoint-v1"
UNIVERSAL_FIELD_SCHEMA_VERSION = 3
EXACT_REGISTRATION_WEIGHT_MODE = "exact_front_to_back_marginal_responsibility"
RELIABILITY_NAMES = FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES


class UniversalFieldValidationError(ValueError):
    """Raised when an artifact cannot belong to Universal Field v1."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UniversalFieldValidationError(message)


def _sha256(value: object) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is not None


def _reliability_from_cache(
    factorized_cache: Mapping[str, Any], *, num_gaussians: int
) -> torch.Tensor:
    names = factorized_cache.get("reliability_scalar_names")
    _require(
        names == list(RELIABILITY_NAMES), "factorized reliability scalar names differ"
    )
    reliability = factorized_cache.get("reliability")
    valid = factorized_cache.get("valid")
    _require(torch.is_tensor(reliability), "factorized reliability tensor is missing")
    _require(torch.is_tensor(valid), "factorized validity tensor is missing")
    _require(
        reliability.shape == (num_gaussians, len(RELIABILITY_NAMES)),
        "factorized reliability tensor shape differs",
    )
    _require(valid.shape == (num_gaussians,), "factorized validity shape differs")
    _require(valid.dtype == torch.bool, "factorized validity dtype differs")
    _require(
        reliability.dtype == torch.float32,
        "factorized reliability must be float32 deployment state",
    )
    _require(
        bool(torch.isfinite(reliability).all()), "factorized reliability is non-finite"
    )
    _require(
        bool((reliability[~valid] == 0).all()),
        "invalid factorized rows must have zero reliability",
    )
    return reliability.detach().cpu().contiguous()


def migrate_universal_field_payload(
    source_payload: Mapping[str, Any],
    factorized_cache: Mapping[str, Any],
    *,
    source_field_sha256: str,
    factorized_cache_sha256: str,
) -> dict[str, Any]:
    """Create a schema-v3 deployment payload without changing decoded RADIO.

    The learned state is copied verbatim.  Only the zero-column reliability
    buffer is replaced with the query-independent five-column MPR state while
    ``fusion_reliability`` remains false, so field coefficients and decoded
    RADIO features are mathematically and bitwise unchanged.
    """

    _require(_sha256(source_field_sha256), "source field SHA256 is malformed")
    _require(_sha256(factorized_cache_sha256), "factorized cache SHA256 is malformed")
    _require(source_payload.get("schema_version") == 2, "source field schema differs")
    _require(
        source_payload.get("checkpoint_contract")
        == "canonical-factorized-radio-checkpoint-v1",
        "source field checkpoint contract differs",
    )
    architecture = source_payload.get("architecture")
    _require(isinstance(architecture, Mapping), "source field architecture is missing")
    _require(
        architecture.get("fusion_reliability") is False,
        "source field must disable reliability fusion",
    )
    num_gaussians = int(architecture.get("num_gaussians", -1))
    _require(num_gaussians > 0, "source field Gaussian count differs")
    source_reliability = source_payload.get("reliability")
    state = source_payload.get("state_dict")
    _require(isinstance(state, Mapping), "source field state_dict is missing")
    state_reliability = state.get("reliability")
    for value in (source_reliability, state_reliability):
        _require(
            torch.is_tensor(value) and value.shape == (num_gaussians, 0),
            "source field already has an incompatible reliability state",
        )
    _require(
        source_payload.get("mpr_cache_sha256") == factorized_cache_sha256,
        "source field and factorized cache SHA256 differ",
    )
    mpr_metadata = source_payload.get("mpr_cache_metadata")
    builder = (
        mpr_metadata.get("builder_contract")
        if isinstance(mpr_metadata, Mapping)
        else None
    )
    _require(
        isinstance(builder, Mapping)
        and builder.get("registration_weight_mode") == EXACT_REGISTRATION_WEIGHT_MODE,
        "source field is not bound to exact marginal responsibility",
    )
    reliability = _reliability_from_cache(factorized_cache, num_gaussians=num_gaussians)

    migrated = copy.deepcopy(dict(source_payload))
    migrated["schema_version"] = UNIVERSAL_FIELD_SCHEMA_VERSION
    migrated["checkpoint_contract"] = UNIVERSAL_FIELD_CHECKPOINT_CONTRACT
    migrated["reliability"] = reliability
    migrated_state = dict(migrated["state_dict"])
    # Keep the public checkpoint copy and the module state buffer on the same
    # storage.  torch serialization preserves this alias, so five scalars are
    # paid for once rather than twice in cold scene state.
    migrated_state["reliability"] = reliability
    migrated["state_dict"] = migrated_state
    migrated["universal_field_migration"] = {
        "schema_version": 1,
        "universal_field_id": UNIVERSAL_FIELD_ID,
        "baseline_readout_id": PRIMITIVE_READOUT_ID,
        "source_field_sha256": source_field_sha256,
        "source_checkpoint_contract": "canonical-factorized-radio-checkpoint-v1",
        "factorized_cache_sha256": factorized_cache_sha256,
        "registration_weight_mode": EXACT_REGISTRATION_WEIGHT_MODE,
        "reliability_scalar_names": list(RELIABILITY_NAMES),
        "reliability_scalar_names_sha256": (
            FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
        ),
        "reliability_usage": "query_posterior_calibration_only",
        "fusion_reliability": False,
        "decode_state_changed": False,
    }
    validate_universal_field_payload(migrated)
    return migrated


def validate_universal_field_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on the Universal Field v1 deployment checkpoint boundary."""

    _require(
        payload.get("schema_version") == UNIVERSAL_FIELD_SCHEMA_VERSION,
        "universal field schema differs",
    )
    _require(
        payload.get("checkpoint_contract") == UNIVERSAL_FIELD_CHECKPOINT_CONTRACT,
        "universal field checkpoint contract differs",
    )
    architecture = payload.get("architecture")
    _require(
        isinstance(architecture, Mapping), "universal field architecture is missing"
    )
    _require(
        architecture.get("feature_dim") == 1280, "universal field feature_dim differs"
    )
    _require(
        architecture.get("coefficient_dim") == 512,
        "universal field coefficient_dim differs",
    )
    _require(architecture.get("local_dim") == 512, "universal field local_dim differs")
    _require(
        architecture.get("fusion_reliability") is False,
        "universal field reliability must not alter the frozen decoder",
    )
    num_gaussians = int(architecture.get("num_gaussians", -1))
    _require(num_gaussians > 0, "universal field Gaussian count differs")
    reliability = payload.get("reliability")
    state = payload.get("state_dict")
    _require(isinstance(state, Mapping), "universal field state_dict is missing")
    state_reliability = state.get("reliability")
    _require(
        torch.is_tensor(reliability)
        and reliability.shape == (num_gaussians, len(RELIABILITY_NAMES))
        and reliability.dtype == torch.float32
        and bool(torch.isfinite(reliability).all()),
        "universal field reliability tensor differs",
    )
    _require(
        torch.is_tensor(state_reliability)
        and torch.equal(reliability, state_reliability),
        "universal field reliability copies differ",
    )
    migration = payload.get("universal_field_migration")
    _require(isinstance(migration, Mapping), "universal field migration is missing")
    _require(
        migration.get("universal_field_id") == UNIVERSAL_FIELD_ID,
        "universal field identity differs",
    )
    _require(
        migration.get("baseline_readout_id") == PRIMITIVE_READOUT_ID,
        "primitive readout identity differs",
    )
    _require(
        migration.get("reliability_scalar_names") == list(RELIABILITY_NAMES)
        and migration.get("reliability_scalar_names_sha256")
        == FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256,
        "universal field reliability scalar columns differ",
    )
    _require(
        migration.get("registration_weight_mode") == EXACT_REGISTRATION_WEIGHT_MODE,
        "universal field registration mode differs",
    )
    _require(
        migration.get("fusion_reliability") is False
        and migration.get("decode_state_changed") is False,
        "universal field decode-preservation contract differs",
    )
    _require(
        _sha256(migration.get("source_field_sha256")), "source field SHA256 differs"
    )
    _require(
        migration.get("factorized_cache_sha256") == payload.get("mpr_cache_sha256"),
        "universal field factorized cache binding differs",
    )
    mpr_metadata = payload.get("mpr_cache_metadata")
    builder = (
        mpr_metadata.get("builder_contract")
        if isinstance(mpr_metadata, Mapping)
        else None
    )
    _require(
        isinstance(builder, Mapping)
        and builder.get("registration_weight_mode") == EXACT_REGISTRATION_WEIGHT_MODE,
        "universal field MPR metadata differs",
    )
    return {
        "universal_field_id": UNIVERSAL_FIELD_ID,
        "baseline_readout_id": PRIMITIVE_READOUT_ID,
        "num_gaussians": num_gaussians,
        "reliability_dim": len(RELIABILITY_NAMES),
        "decode_state_changed": False,
    }


def validate_universal_field_authority(authority: Mapping[str, Any]) -> None:
    """Validate the checked-in architecture authority without opening fields."""

    _require(authority.get("schema_version") == 1, "authority schema differs")
    _require(
        authority.get("universal_field_id") == UNIVERSAL_FIELD_ID,
        "authority universal field identity differs",
    )
    _require(
        authority.get("baseline_readout_id") == PRIMITIVE_READOUT_ID,
        "authority primitive readout identity differs",
    )
    state = authority.get("persistent_scene_state")
    _require(isinstance(state, Mapping), "authority persistent scene state is missing")
    _require(
        state.get("checkpoint_schema_version") == 3, "authority field schema differs"
    )
    _require(
        state.get("reliability_scalar_names") == list(RELIABILITY_NAMES),
        "authority reliability scalar columns differ",
    )
    _require(
        state.get("reliability_dim") == len(RELIABILITY_NAMES),
        "authority reliability dimension differs",
    )
    _require(
        state.get("reliability_fused_into_decoder") is False,
        "authority decode preservation differs",
    )
    construction = authority.get("construction")
    _require(isinstance(construction, Mapping), "authority construction is missing")
    _require(
        construction.get("registration_weight_mode") == EXACT_REGISTRATION_WEIGHT_MODE,
        "authority registration mode differs",
    )


__all__ = [
    "EXACT_REGISTRATION_WEIGHT_MODE",
    "PRIMITIVE_READOUT_ID",
    "RELIABILITY_NAMES",
    "UNIVERSAL_FIELD_CHECKPOINT_CONTRACT",
    "UNIVERSAL_FIELD_ID",
    "UNIVERSAL_FIELD_SCHEMA_VERSION",
    "UniversalFieldValidationError",
    "migrate_universal_field_payload",
    "validate_universal_field_authority",
    "validate_universal_field_payload",
]
