"""Versioned, missingness-safe state for factorized RADIO primitives.

The semantic direction and predicted log amplitude are decoded from the
schema-v2 compact field.  Query-independent observation statistics remain a
strictly SHA-bound sidecar: they are supervision/evidence state and are never
registered as trainable ``CanonicalGaussianField.reliability`` columns.

The raster-top1 cache cannot measure visibility purity because its registration
sidecar lacks visible mass.  Schema v1 therefore recorded an unknown zero
sentinel.  Schema v2 additionally accepts the measured exact-marginal authority
without changing the legacy geometric-reliability readout.  Consumers must use
:meth:`scalar_encoding_input`, which masks unavailable values before they can
reach a learned encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.field.factorized_radio_contract import (
    CANONICAL_FACTORIZED_RADIO_CHECKPOINT_CONTRACT,
    CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256,
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES,
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256,
    FactorizedRadioFieldSignature,
)
from radio_gs.training.factorized_radio_cache import (
    FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY,
    FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY,
    FactorizedRadioTrainingCache,
    load_factorized_radio_training_cache,
)
from radio_gs.utils.immutable_artifacts import load_torch_mapping


FACTORIZED_FIELD_CHECKPOINT_SCHEMA_V2 = "factorized-v2"
FACTORIZED_PRIMITIVE_STATE_SCHEMA_V1 = "radio_gs.factorized_primitive_state.v1"
FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION_V1 = 1
FACTORIZED_PRIMITIVE_STATE_SCHEMA = "radio_gs.factorized_primitive_state.v2"
FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION = 2
FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES = (
    "predicted_log_amplitude",
    "directional_dispersion",
    "log_amplitude_std",
    "observation_evidence",
    "visibility_purity_value",
    "visibility_purity_known",
)


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256 = _canonical_json_sha256(
    list(FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES)
)


def _factorized_primitive_state_contract_v1() -> dict[str, Any]:
    """Return the frozen legacy contract exactly as originally published."""

    return {
        "schema": FACTORIZED_PRIMITIVE_STATE_SCHEMA_V1,
        "schema_version": FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION_V1,
        "parent_factorized_radio_contract_sha256": (
            CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256
        ),
        "semantic_direction": "unit_direction_of_schema_v2_field_decode",
        "predicted_log_amplitude": "log_l2_norm_of_schema_v2_field_decode",
        "directional_dispersion": "factorized_cache_one_minus_directional_resultant",
        "log_amplitude_std": "factorized_cache_population_standard_deviation",
        "observation_evidence": "factorized_cache_positive_view_count_over_count_plus_one",
        "visibility_purity_value": "factorized_cache_visibility_purity",
        "visibility_purity_known": "measurement_available_boolean",
        "current_visibility_purity_authority": dict(
            FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY
        ),
        "unknown_value_policy": "mask_value_to_exact_zero_before_encoding",
        "scalar_names": list(FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES),
        "scalar_names_sha256": FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256,
        "storage": "compact_valid_rows_in_torch_where_valid_ascending_order",
        "query_independent": True,
    }


FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256_V1 = _canonical_json_sha256(
    _factorized_primitive_state_contract_v1()
)


def factorized_primitive_state_contract() -> dict[str, Any]:
    """Return the immutable v2 state and authority-aware missingness contract."""

    return {
        "schema": FACTORIZED_PRIMITIVE_STATE_SCHEMA,
        "schema_version": FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION,
        "parent_factorized_radio_contract_sha256": (
            CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256
        ),
        "semantic_direction": "unit_direction_of_schema_v2_field_decode",
        "predicted_log_amplitude": "log_l2_norm_of_schema_v2_field_decode",
        "directional_dispersion": "factorized_cache_one_minus_directional_resultant",
        "log_amplitude_std": "factorized_cache_population_standard_deviation",
        "observation_evidence": "factorized_cache_positive_view_count_over_count_plus_one",
        "visibility_purity_value": "factorized_cache_visibility_purity",
        "visibility_purity_known": "measurement_available_boolean",
        "allowed_visibility_purity_authorities": {
            "top1_unknown": dict(FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY),
            "exact_marginal_measured": dict(
                FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY
            ),
        },
        "authority_policy": (
            "top1_requires_all_unknown_exact_zero;"
            "exact_marginal_requires_all_known_and_retains_measured_value"
        ),
        "unknown_value_policy": "mask_value_to_exact_zero_before_encoding",
        "scalar_names": list(FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES),
        "scalar_names_sha256": FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256,
        "storage": "compact_valid_rows_in_torch_where_valid_ascending_order",
        "query_independent": True,
    }


FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256 = _canonical_json_sha256(
    factorized_primitive_state_contract()
)


def _contract_for_schema(schema: object, version: object) -> tuple[dict[str, Any], str]:
    if (
        schema == FACTORIZED_PRIMITIVE_STATE_SCHEMA_V1
        and version == FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION_V1
    ):
        return (
            _factorized_primitive_state_contract_v1(),
            FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256_V1,
        )
    if (
        schema == FACTORIZED_PRIMITIVE_STATE_SCHEMA
        and version == FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION
    ):
        return (
            factorized_primitive_state_contract(),
            FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256,
        )
    raise ValueError("factorized primitive state schema differs")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float32_rows_sha256(values: torch.Tensor) -> str:
    array = (
        torch.as_tensor(values)
        .detach()
        .float()
        .cpu()
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


@dataclass(frozen=True)
class FactorizedFieldSupport:
    """Strictly loaded schema-v2 field and its exact factorized support."""

    field: torch.nn.Module
    field_payload: Mapping[str, Any]
    field_signature: FactorizedRadioFieldSignature
    field_checkpoint: Path
    field_checkpoint_sha256: str
    cache: FactorizedRadioTrainingCache

    @property
    def lineage(self) -> dict[str, Any]:
        return {
            "field_checkpoint_schema_version": 2,
            "field_checkpoint_contract": CANONICAL_FACTORIZED_RADIO_CHECKPOINT_CONTRACT,
            "factorized_radio_field_signature": self.field_signature.to_dict(),
            "factorized_radio_field_signature_sha256": self.field_signature.digest,
            "factorized_radio_contract_sha256": (
                self.field_signature.factorized_radio_contract_sha256
            ),
            "factorized_radio_cache_sha256": self.cache.sha256,
            "registration_responsibility_cache_sha256": str(
                self.cache.metadata["registration_responsibility_cache_sha256"]
            ),
            "feature_output_bundle_sha256": str(
                self.field_payload.get("feature_output_bundle_sha256", "")
            ),
            "mpr_geometry_fingerprint": dict(self.cache.geometry_fingerprint),
        }


def load_factorized_field_support(
    field_checkpoint: str | Path,
    *,
    expected_field_checkpoint_sha256: str,
    mpr_cache: str | Path | None = None,
    expected_mpr_cache_sha256: str = "",
) -> FactorizedFieldSupport:
    """Load only schema-v2; malformed inputs can never fall back to v1."""

    field_path = Path(field_checkpoint).resolve()
    expected_field = str(expected_field_checkpoint_sha256)
    if not expected_field:
        raise ValueError("factorized field support requires a trusted field SHA-256")
    actual_field = _sha256_file(field_path)
    if actual_field != expected_field:
        raise ValueError("factorized field checkpoint SHA-256 differs")
    field, raw_payload, signature = load_factorized_canonical_field_checkpoint(
        field_path,
        map_location="cpu",
        expected_sha256=expected_field,
    )
    payload = dict(raw_payload)
    embedded_cache_sha = str(payload.get("factorized_cache_sha256", ""))
    payload_mpr_sha = str(payload.get("mpr_cache_sha256", ""))
    expected_cache = str(expected_mpr_cache_sha256) or embedded_cache_sha
    if (
        not expected_cache
        or embedded_cache_sha != expected_cache
        or payload_mpr_sha != expected_cache
    ):
        raise ValueError("factorized support cache SHA-256 differs from field")
    cache_path = Path(mpr_cache or payload.get("mpr_cache", "")).resolve()
    cache = load_factorized_radio_training_cache(
        cache_path,
        expected_sha256=expected_cache,
        expected_feature_output_bundle_sha256=str(
            payload.get("feature_output_bundle_sha256", "")
        ),
    )
    if dict(payload.get("geometry_fingerprint", {})) != dict(
        cache.geometry_fingerprint
    ):
        raise ValueError("factorized field/support geometry differs")
    signature.base_feature_signature.assert_compatible(field.signature)
    if int(field.num_gaussians) != int(cache.xyz.shape[0]):
        raise ValueError("factorized field/support rows differ")
    if any(
        payload.get(name) is not False
        for name in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "text_queries_opened",
        )
    ):
        raise ValueError("factorized field is not source-only")
    return FactorizedFieldSupport(
        field=field,
        field_payload=payload,
        field_signature=signature,
        field_checkpoint=field_path,
        field_checkpoint_sha256=actual_field,
        cache=cache,
    )


@dataclass(frozen=True)
class FactorizedPrimitiveState:
    """Compact valid-row state with an explicit missing-value mask."""

    xyz: torch.Tensor
    valid: torch.Tensor
    global_rows: torch.Tensor
    semantic_direction: torch.Tensor
    predicted_log_amplitude: torch.Tensor
    directional_dispersion: torch.Tensor
    log_amplitude_std: torch.Tensor
    observation_evidence: torch.Tensor
    visibility_purity_value: torch.Tensor
    visibility_purity_known: torch.Tensor
    metadata: Mapping[str, Any]
    source: Path | None = None
    sha256: str = ""
    schema: str = FACTORIZED_PRIMITIVE_STATE_SCHEMA
    schema_version: int = FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION

    @property
    def contract_sha256(self) -> str:
        """Return the contract digest carried by this exact state version."""

        return _contract_for_schema(self.schema, self.schema_version)[1]

    def scalar_encoding_input(self) -> torch.Tensor:
        """Return state scalars after masking unknown purity to exact zero."""

        known = self.visibility_purity_known.bool()
        purity = torch.where(
            known,
            self.visibility_purity_value.float(),
            torch.zeros_like(self.visibility_purity_value, dtype=torch.float32),
        )
        return torch.stack(
            (
                self.predicted_log_amplitude.float(),
                self.directional_dispersion.float(),
                self.log_amplitude_std.float(),
                self.observation_evidence.float(),
                purity,
                known.float(),
            ),
            dim=-1,
        )

    def legacy_geometric_reliability(self) -> torch.Tensor:
        """Single-channel compatibility value for an unchanged V2 readout."""

        agreement = (1.0 - self.directional_dispersion.float()).clamp(0.0, 1.0)
        evidence = self.observation_evidence.float().clamp(0.0, 1.0)
        return (agreement * evidence).sqrt()

    def to_payload(self) -> dict[str, Any]:
        contract, contract_sha256 = _contract_for_schema(
            self.schema, self.schema_version
        )
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "contract": contract,
            "contract_sha256": contract_sha256,
            "scalar_names": list(FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES),
            "scalar_names_sha256": FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256,
            "xyz": self.xyz,
            "valid": self.valid,
            "global_rows": self.global_rows,
            "semantic_direction": self.semantic_direction,
            "predicted_log_amplitude": self.predicted_log_amplitude,
            "directional_dispersion": self.directional_dispersion,
            "log_amplitude_std": self.log_amplitude_std,
            "observation_evidence": self.observation_evidence,
            "visibility_purity_value": self.visibility_purity_value,
            "visibility_purity_known": self.visibility_purity_known,
            "metadata": dict(self.metadata),
        }


def build_factorized_primitive_state(
    support: FactorizedFieldSupport,
    *,
    chunk_size: int = 4096,
) -> FactorizedPrimitiveState:
    """Decode a compact CPU state from one strict schema-v2 support bundle."""

    size = int(chunk_size)
    if size <= 0:
        raise ValueError("factorized primitive state chunk_size must be positive")
    cache = support.cache
    rows = torch.where(cache.valid)[0]
    directions = torch.empty(rows.numel(), 1280, dtype=torch.float16)
    predicted_log_amplitude = torch.empty(rows.numel(), dtype=torch.float32)
    field = support.field.cpu().eval()
    field.requires_grad_(False)
    with torch.inference_mode():
        for start in range(0, rows.numel(), size):
            selected = rows[start : start + size]
            decoded = field.radio_features(selected).float()
            amplitude = torch.linalg.vector_norm(decoded, dim=-1)
            if not bool(torch.isfinite(decoded).all()) or bool((amplitude <= 0).any()):
                raise ValueError("factorized field decoded non-finite/zero state")
            directions[start : start + selected.numel()] = F.normalize(
                decoded, dim=-1, eps=1e-8
            ).half()
            predicted_log_amplitude[start : start + selected.numel()] = torch.log(
                amplitude
            ).cpu()
    reliability = cache.reliability[rows].float().cpu()
    registration_sha256 = str(
        cache.metadata["registration_responsibility_cache_sha256"]
    )
    purity_authority = cache.metadata.get("visibility_purity_authority")
    expected_top1_authority = {
        **FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY,
        "registration_responsibility_cache_sha256": registration_sha256,
    }
    expected_exact_authority = {
        **FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY,
        "registration_responsibility_cache_sha256": registration_sha256,
    }
    if purity_authority == expected_top1_authority:
        purity_known = torch.zeros(rows.numel(), dtype=torch.bool)
        purity_value = reliability[:, 4].clone()
        if bool(purity_value.ne(0).any()):
            raise ValueError("unknown top1 visibility purity must be exact zero")
    elif purity_authority == expected_exact_authority:
        purity_known = torch.ones(rows.numel(), dtype=torch.bool)
        purity_value = reliability[:, 4].clone()
    else:
        raise ValueError("factorized visibility-purity authority differs")
    metadata = {
        "source": "factorized_primitive_state_v2",
        "field_checkpoint": str(support.field_checkpoint),
        "field_checkpoint_sha256": support.field_checkpoint_sha256,
        "factorized_radio_cache": str(cache.source),
        "factorized_radio_cache_sha256": cache.sha256,
        "factorized_radio_field_signature": support.field_signature.to_dict(),
        "factorized_radio_field_signature_sha256": support.field_signature.digest,
        "factorized_radio_reliability_scalar_names": list(
            FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES
        ),
        "factorized_radio_reliability_scalar_names_sha256": (
            FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
        ),
        "geometry_fingerprint": dict(cache.geometry_fingerprint),
        "visibility_purity_authority": dict(purity_authority),
        "registration_responsibility_cache_sha256": registration_sha256,
        "feature_output_bundle_sha256": str(
            support.field_payload.get("feature_output_bundle_sha256", "")
        ),
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    state = FactorizedPrimitiveState(
        xyz=cache.xyz.float().cpu(),
        valid=cache.valid.bool().cpu(),
        global_rows=rows.long().cpu(),
        semantic_direction=directions,
        predicted_log_amplitude=predicted_log_amplitude,
        directional_dispersion=reliability[:, 1].clone(),
        log_amplitude_std=reliability[:, 2].clone(),
        observation_evidence=reliability[:, 3].clone(),
        visibility_purity_value=purity_value,
        visibility_purity_known=purity_known,
        metadata=metadata,
    )
    validate_factorized_primitive_state_payload(state.to_payload())
    return state


def validate_factorized_primitive_state_payload(value: object) -> dict[str, Any]:
    """Validate an in-memory payload and return its exact mapping."""

    if not isinstance(value, Mapping):
        raise ValueError("factorized primitive state must contain a mapping")
    payload = dict(value)
    required = {
        "schema", "schema_version", "contract", "contract_sha256",
        "scalar_names", "scalar_names_sha256", "xyz", "valid", "global_rows",
        "semantic_direction", "predicted_log_amplitude", "directional_dispersion",
        "log_amplitude_std", "observation_evidence", "visibility_purity_value",
        "visibility_purity_known", "metadata",
    }
    if set(payload) != required:
        raise ValueError("factorized primitive state fields differ")
    expected_contract, expected_contract_sha256 = _contract_for_schema(
        payload.get("schema"), payload.get("schema_version")
    )
    if (
        payload.get("contract") != expected_contract
        or payload.get("contract_sha256") != expected_contract_sha256
        or payload.get("scalar_names")
        != list(FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES)
        or payload.get("scalar_names_sha256")
        != FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256
    ):
        raise ValueError("factorized primitive state contract differs")
    xyz = payload["xyz"]
    valid = payload["valid"]
    rows = payload["global_rows"]
    direction = payload["semantic_direction"]
    known = payload["visibility_purity_known"]
    count = int(xyz.shape[0]) if torch.is_tensor(xyz) and xyz.ndim == 2 else -1
    compact = int(rows.numel()) if torch.is_tensor(rows) and rows.ndim == 1 else -1
    scalars = tuple(payload[name] for name in FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES[:-1])
    if (
        not torch.is_tensor(xyz) or xyz.dtype != torch.float32 or xyz.shape != (count, 3)
        or not torch.is_tensor(valid) or valid.dtype != torch.bool or valid.shape != (count,)
        or not torch.is_tensor(rows) or rows.dtype != torch.long or rows.shape != (compact,)
        or not torch.equal(rows, torch.where(valid)[0])
        or not torch.is_tensor(direction) or direction.dtype != torch.float16
        or direction.shape != (compact, 1280)
        or not torch.is_tensor(known) or known.dtype != torch.bool or known.shape != (compact,)
        or any(not torch.is_tensor(item) or item.dtype != torch.float32 or item.shape != (compact,) for item in scalars)
    ):
        raise ValueError("factorized primitive state tensor layout differs")
    if not all(bool(torch.isfinite(item).all()) for item in (xyz, direction, *scalars)):
        raise ValueError("factorized primitive state contains non-finite values")
    direction_norm = torch.linalg.vector_norm(direction.float(), dim=-1)
    if not torch.allclose(
        direction_norm, torch.ones_like(direction_norm), atol=5e-4, rtol=0.0
    ):
        raise ValueError("factorized primitive semantic direction is not unit L2")
    dispersion = payload["directional_dispersion"]
    amplitude_std = payload["log_amplitude_std"]
    evidence = payload["observation_evidence"]
    purity = payload["visibility_purity_value"]
    if (
        bool((dispersion < 0).any()) or bool((dispersion >= 1).any())
        or bool((amplitude_std < 0).any())
        or bool((evidence <= 0).any()) or bool((evidence >= 1).any())
        or bool((purity < 0).any()) or bool((purity > 1).any())
    ):
        raise ValueError("factorized primitive state scalar range differs")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("factorized primitive state metadata is malformed")
    legacy_schema = payload.get("schema") == FACTORIZED_PRIMITIVE_STATE_SCHEMA_V1
    safety = {
        "source": (
            "factorized_primitive_state_v1"
            if legacy_schema
            else "factorized_primitive_state_v2"
        ),
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    if any(metadata.get(name) != expected for name, expected in safety.items()):
        raise ValueError("factorized primitive state safety metadata differs")
    purity_authority = metadata.get("visibility_purity_authority")
    registration_sha256 = metadata.get(
        "registration_responsibility_cache_sha256"
    )
    top1_authority = {
        **FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY,
        "registration_responsibility_cache_sha256": registration_sha256,
    }
    exact_authority = {
        **FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY,
        "registration_responsibility_cache_sha256": registration_sha256,
    }
    top1_state = (
        purity_authority == top1_authority
        and not bool(known.any())
        and not bool(purity.ne(0).any())
    )
    exact_state = (
        not legacy_schema
        and purity_authority == exact_authority
        and bool(known.all())
    )
    if (
        metadata.get("factorized_radio_reliability_scalar_names")
        != list(FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES)
        or metadata.get("factorized_radio_reliability_scalar_names_sha256")
        != FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
        or not (top1_state or exact_state)
    ):
        raise ValueError("factorized primitive state purity/scalar authority differs")
    signature = FactorizedRadioFieldSignature.from_mapping(
        metadata.get("factorized_radio_field_signature", {})
    )
    geometry = metadata.get("geometry_fingerprint")
    if (
        any(
            not _is_sha256(metadata.get(name))
            for name in (
                "field_checkpoint_sha256",
                "factorized_radio_cache_sha256",
                "factorized_radio_field_signature_sha256",
                "factorized_radio_reliability_scalar_names_sha256",
                "registration_responsibility_cache_sha256",
                "feature_output_bundle_sha256",
            )
        )
        or metadata.get("factorized_radio_field_signature_sha256") != signature.digest
        or signature.factorized_radio_contract_sha256
        != CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256
        or geometry
        != {
            "num_gaussians": count,
            "xyz_sha256": _float32_rows_sha256(xyz),
        }
    ):
        raise ValueError("factorized primitive state field signature differs")
    return payload


def load_factorized_primitive_state(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_field_checkpoint_sha256: str = "",
    expected_factorized_radio_cache_sha256: str = "",
    expected_xyz: torch.Tensor | None = None,
    expected_valid: torch.Tensor | None = None,
) -> FactorizedPrimitiveState:
    """Load one immutable state sidecar with strict lineage and row checks."""

    payload, actual_sha, source = load_torch_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="factorized primitive state",
    )
    value = validate_factorized_primitive_state_payload(payload)
    metadata = dict(value["metadata"])
    if expected_field_checkpoint_sha256 and metadata.get(
        "field_checkpoint_sha256"
    ) != str(expected_field_checkpoint_sha256):
        raise ValueError("factorized primitive state field SHA-256 differs")
    if expected_factorized_radio_cache_sha256 and metadata.get(
        "factorized_radio_cache_sha256"
    ) != str(expected_factorized_radio_cache_sha256):
        raise ValueError("factorized primitive state cache SHA-256 differs")
    xyz = value["xyz"]
    valid = value["valid"]
    if expected_xyz is not None and not torch.equal(
        xyz, torch.as_tensor(expected_xyz).float().cpu()
    ):
        raise ValueError("factorized primitive state xyz differs")
    if expected_valid is not None and not torch.equal(
        valid, torch.as_tensor(expected_valid).bool().cpu()
    ):
        raise ValueError("factorized primitive state valid rows differ")
    return FactorizedPrimitiveState(
        xyz=xyz,
        valid=valid,
        global_rows=value["global_rows"],
        semantic_direction=value["semantic_direction"],
        predicted_log_amplitude=value["predicted_log_amplitude"],
        directional_dispersion=value["directional_dispersion"],
        log_amplitude_std=value["log_amplitude_std"],
        observation_evidence=value["observation_evidence"],
        visibility_purity_value=value["visibility_purity_value"],
        visibility_purity_known=value["visibility_purity_known"],
        metadata=metadata,
        source=source,
        sha256=actual_sha,
        schema=str(value["schema"]),
        schema_version=int(value["schema_version"]),
    )
