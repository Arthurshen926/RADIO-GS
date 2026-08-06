"""Fail-closed checkpoint boundary for the accepted-V2 full-scalar residual."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.models.surface_region_dual_descriptor import (
    SurfaceRegionAcceptedV2FullScalarResidualV1,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_torch_mapping,
    sha256_file,
    write_torch_noclobber,
)

from .factorized_primitive_state import (
    FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256,
)
from .surface_region_full_scalar_contract import (
    SURFACE_REGION_FULL_SCALAR_CONTRACT_SHA256,
    SURFACE_REGION_FULL_SCALAR_DIM,
    SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
    validate_full_scalar_normalization_authority,
)
from .surface_region_full_scalar_training_certificate import (
    SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_CONTRACT_SHA256,
    validate_training_certificate_payload,
)
from .surface_region_summary import (
    ACCEPTED_SURFACE_REGION_V2_ARCHITECTURE_SHA256,
    ACCEPTED_SURFACE_REGION_V2_CHECKPOINT_SHA256,
    ACCEPTED_SURFACE_REGION_V2_CONTRACT_SHA256,
    ACCEPTED_SURFACE_REGION_V2_PROVENANCE_SHA256,
    ACCEPTED_SURFACE_REGION_V2_STATE_DICT_SHA256,
    surface_region_state_dict_sha256,
)


SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_SCHEMA = (
    "radio_gs.surface_region_accepted_v2_full_scalar_residual_checkpoint.v1"
)
SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_SCHEMA_VERSION = 1
SURFACE_REGION_FULL_SCALAR_RESIDUAL_MODEL_CLASS = (
    "SurfaceRegionAcceptedV2FullScalarResidualV1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_STATE_KEYS = frozenset(
    {
        "scalar_median",
        "scalar_robust_scale",
        "descriptor_projection.weight",
        "scalar_projection.weight",
        "scalar_projection.bias",
        "fusion_projection.weight",
        "fusion_projection.bias",
        "residual_projection.weight",
        "residual_projection.bias",
    }
)


def _accepted_v2_authority() -> dict[str, str]:
    return {
        "checkpoint_sha256": ACCEPTED_SURFACE_REGION_V2_CHECKPOINT_SHA256,
        "architecture_sha256": ACCEPTED_SURFACE_REGION_V2_ARCHITECTURE_SHA256,
        "state_dict_sha256": ACCEPTED_SURFACE_REGION_V2_STATE_DICT_SHA256,
        "provenance_sha256": ACCEPTED_SURFACE_REGION_V2_PROVENANCE_SHA256,
        "contract_sha256": ACCEPTED_SURFACE_REGION_V2_CONTRACT_SHA256,
    }


def surface_region_full_scalar_residual_checkpoint_contract() -> dict[str, Any]:
    """Return the immutable loader/writer contract for the only allowed class."""

    return {
        "schema": SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_SCHEMA,
        "schema_version": (
            SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_SCHEMA_VERSION
        ),
        "allowed_model_class": SURFACE_REGION_FULL_SCALAR_RESIDUAL_MODEL_CLASS,
        "immutable_accepted_v2_authority": _accepted_v2_authority(),
        "full_scalar_contract_sha256": SURFACE_REGION_FULL_SCALAR_CONTRACT_SHA256,
        "full_scalar_names_sha256": SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
        "full_scalar_dimension": SURFACE_REGION_FULL_SCALAR_DIM,
        "factorized_primitive_state_contract_sha256": (
            FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256
        ),
        "source_authority": (
            "source_state_normalization_and_training_certificate_file_sha256"
        ),
        "training_certificate_contract_sha256": (
            SURFACE_REGION_FULL_SCALAR_TRAINING_CERTIFICATE_CONTRACT_SHA256
        ),
        "model_architecture_authority": "canonical_json_sha256",
        "model_state_dict_authority": "surface_region_state_dict_sha256",
        "normalization_buffer_policy": (
            "scalar_median_and_scalar_robust_scale_bitwise_equal_authority"
        ),
        "checkpoint_identity_policy": (
            "zero_final_projection_iff_exact_base_identity_at_checkpoint"
        ),
        "load_policy": (
            "stable_descriptor_expected_checkpoint_sha256_exact_keys_no_fallback"
        ),
        "write_policy": "atomic_same_directory_no_clobber",
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }


SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_CONTRACT_SHA256 = (
    canonical_json_sha256(surface_region_full_scalar_residual_checkpoint_contract())
)


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _source_authority(
    *,
    source_state_cohort_authority_sha256: str,
    normalization_authority_sha256: str,
    training_certificate_sha256: str,
) -> dict[str, str]:
    return {
        "source_state_cohort_authority_sha256": _require_sha256(
            source_state_cohort_authority_sha256,
            label="source-state cohort authority",
        ),
        "normalization_authority_sha256": _require_sha256(
            normalization_authority_sha256,
            label="normalization authority",
        ),
        "training_certificate_sha256": _require_sha256(
            training_certificate_sha256,
            label="training certificate",
        ),
    }


def _checkpoint_state_assertions(
    state: Mapping[str, torch.Tensor],
) -> dict[str, bool | str]:
    residual_zero = bool(
        torch.count_nonzero(state["residual_projection.weight"]) == 0
        and torch.count_nonzero(state["residual_projection.bias"]) == 0
    )
    return {
        "residual_projection_exact_zero": residual_zero,
        "exact_base_identity_at_checkpoint": residual_zero,
        "identity_proof": (
            "zero_residual_projection_structural"
            if residual_zero
            else "not_identity_trained_bounded_tangent_residual"
        ),
    }


def _provenance() -> dict[str, bool | str]:
    return {
        "training_scope": "global_cross_scene_source_only",
        "accepted_v2_immutable_external": True,
        "accepted_v2_parameters_in_model_state": False,
        "source_state_cohort_scene_disjoint": True,
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "target_state_used_for_training_or_selection": False,
    }


def _copy_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }


def _validate_normalization_binding(
    normalization_authority: Mapping[str, Any],
    *,
    expected_source_state_cohort_authority_sha256: str,
) -> dict[str, Any]:
    authority = validate_full_scalar_normalization_authority(
        normalization_authority
    )
    expected_source = _require_sha256(
        expected_source_state_cohort_authority_sha256,
        label="source-state cohort authority",
    )
    if authority.get("source_state_cohort_sha256") != expected_source:
        raise ValueError("normalization/source-state cohort authority differs")
    return authority


def build_surface_region_full_scalar_residual_checkpoint_payload(
    model: SurfaceRegionAcceptedV2FullScalarResidualV1,
    *,
    normalization_authority: Mapping[str, Any],
    normalization_authority_sha256: str,
    source_state_cohort_authority_sha256: str,
    training_certificate: Mapping[str, Any],
    training_certificate_sha256: str,
) -> dict[str, Any]:
    """Construct one exact-key payload without claiming unverifiable history."""

    if type(model) is not SurfaceRegionAcceptedV2FullScalarResidualV1:
        raise TypeError(
            "checkpoint accepts only SurfaceRegionAcceptedV2FullScalarResidualV1"
        )
    authority = _validate_normalization_binding(
        normalization_authority,
        expected_source_state_cohort_authority_sha256=(
            source_state_cohort_authority_sha256
        ),
    )
    certificate = validate_training_certificate_payload(training_certificate)
    source = _source_authority(
        source_state_cohort_authority_sha256=(
            source_state_cohort_authority_sha256
        ),
        normalization_authority_sha256=normalization_authority_sha256,
        training_certificate_sha256=training_certificate_sha256,
    )
    state = _copy_state_dict(model)
    if set(state) != _MODEL_STATE_KEYS:
        raise ValueError("full-scalar residual model state keys differ")
    if not torch.equal(state["scalar_median"], authority["median"]):
        raise ValueError("model scalar_median differs from normalization authority")
    if not torch.equal(
        state["scalar_robust_scale"], authority["robust_scale"]
    ):
        raise ValueError(
            "model scalar_robust_scale differs from normalization authority"
        )
    architecture = model.architecture()
    state_sha256 = surface_region_state_dict_sha256(state)
    certificate_model = certificate["model_authority"]
    if (
        certificate["normalization_authority"]["sha256"]
        != normalization_authority_sha256
        or certificate["source_state_manifest"]["authority_sha256"]
        != source_state_cohort_authority_sha256
        or certificate_model["architecture"] != architecture
        or certificate_model["architecture_sha256"]
        != canonical_json_sha256(architecture)
        or certificate_model["state_dict_sha256"] != state_sha256
    ):
        raise ValueError("training certificate/checkpoint authority differs")
    payload = {
        "schema": SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_SCHEMA,
        "schema_version": (
            SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_SCHEMA_VERSION
        ),
        "contract": surface_region_full_scalar_residual_checkpoint_contract(),
        "contract_sha256": (
            SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_CONTRACT_SHA256
        ),
        "accepted_v2_authority": _accepted_v2_authority(),
        "full_scalar_authority": {
            "contract_sha256": SURFACE_REGION_FULL_SCALAR_CONTRACT_SHA256,
            "names_sha256": SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
            "dimension": SURFACE_REGION_FULL_SCALAR_DIM,
            "factorized_primitive_state_contract_sha256": (
                FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256
            ),
        },
        "source_authority": source,
        "model_class": SURFACE_REGION_FULL_SCALAR_RESIDUAL_MODEL_CLASS,
        "model_architecture": architecture,
        "model_architecture_sha256": canonical_json_sha256(architecture),
        "model_state_dict": state,
        "model_state_dict_sha256": state_sha256,
        "checkpoint_state_assertions": _checkpoint_state_assertions(state),
        "provenance": _provenance(),
    }
    validate_surface_region_full_scalar_residual_checkpoint_payload(
        payload,
        normalization_authority=authority,
        expected_normalization_authority_sha256=(normalization_authority_sha256),
        expected_source_state_cohort_authority_sha256=(
            source_state_cohort_authority_sha256
        ),
        training_certificate=certificate,
        expected_training_certificate_sha256=training_certificate_sha256,
    )
    return payload


def validate_surface_region_full_scalar_residual_checkpoint_payload(
    value: object,
    *,
    normalization_authority: Mapping[str, Any],
    expected_normalization_authority_sha256: str,
    expected_source_state_cohort_authority_sha256: str,
    training_certificate: Mapping[str, Any],
    expected_training_certificate_sha256: str,
) -> tuple[dict[str, Any], SurfaceRegionAcceptedV2FullScalarResidualV1]:
    """Validate all nested keys, authorities, tensors, and truthful markers."""

    if not isinstance(value, Mapping):
        raise ValueError("full-scalar residual checkpoint must contain a mapping")
    payload = dict(value)
    required = {
        "schema", "schema_version", "contract", "contract_sha256",
        "accepted_v2_authority", "full_scalar_authority", "source_authority",
        "model_class", "model_architecture", "model_architecture_sha256",
        "model_state_dict", "model_state_dict_sha256",
        "checkpoint_state_assertions", "provenance",
    }
    if set(payload) != required:
        raise ValueError("full-scalar residual checkpoint fields differ")
    if (
        payload.get("schema")
        != SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_SCHEMA
        or payload.get("schema_version")
        != SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_SCHEMA_VERSION
        or payload.get("contract")
        != surface_region_full_scalar_residual_checkpoint_contract()
        or payload.get("contract_sha256")
        != SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_CONTRACT_SHA256
        or payload.get("accepted_v2_authority") != _accepted_v2_authority()
        or payload.get("model_class")
        != SURFACE_REGION_FULL_SCALAR_RESIDUAL_MODEL_CLASS
        or payload.get("provenance") != _provenance()
    ):
        raise ValueError("full-scalar residual checkpoint contract differs")
    expected_full_scalar = {
        "contract_sha256": SURFACE_REGION_FULL_SCALAR_CONTRACT_SHA256,
        "names_sha256": SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
        "dimension": SURFACE_REGION_FULL_SCALAR_DIM,
        "factorized_primitive_state_contract_sha256": (
            FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256
        ),
    }
    if payload.get("full_scalar_authority") != expected_full_scalar:
        raise ValueError("full-scalar residual scalar authority differs")
    expected_source = _source_authority(
        source_state_cohort_authority_sha256=(
            expected_source_state_cohort_authority_sha256
        ),
        normalization_authority_sha256=(
        expected_normalization_authority_sha256
        ),
        training_certificate_sha256=expected_training_certificate_sha256,
    )
    if payload.get("source_authority") != expected_source:
        raise ValueError("full-scalar residual source authority differs")
    authority = _validate_normalization_binding(
        normalization_authority,
        expected_source_state_cohort_authority_sha256=(
            expected_source_state_cohort_authority_sha256
        ),
    )
    certificate = validate_training_certificate_payload(training_certificate)
    if (
        certificate["normalization_authority"]["sha256"]
        != expected_normalization_authority_sha256
        or certificate["source_state_manifest"]["authority_sha256"]
        != expected_source_state_cohort_authority_sha256
    ):
        raise ValueError("training certificate source authority differs")

    architecture = payload.get("model_architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("full-scalar residual architecture must be a mapping")
    architecture = dict(architecture)
    if (
        canonical_json_sha256(architecture)
        != payload.get("model_architecture_sha256")
        or architecture.get("name")
        != SurfaceRegionAcceptedV2FullScalarResidualV1.ARCHITECTURE_NAME
        or architecture.get("scalar_dim") != SURFACE_REGION_FULL_SCALAR_DIM
        or architecture.get("hidden_dim")
        != SurfaceRegionAcceptedV2FullScalarResidualV1.HIDDEN_DIM
    ):
        raise ValueError("full-scalar residual architecture authority differs")
    try:
        model = SurfaceRegionAcceptedV2FullScalarResidualV1(
            descriptor_dim=int(architecture["descriptor_dim"]),
            scalar_median=authority["median"],
            scalar_robust_scale=authority["robust_scale"],
            max_angle_radians=float(architecture["max_angle_radians"]),
            max_alpha=float(architecture["max_alpha"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("full-scalar residual architecture differs") from exc
    if model.architecture() != architecture:
        raise ValueError("full-scalar residual architecture does not reconstruct")
    if (
        certificate["model_authority"]["architecture"] != architecture
        or certificate["model_authority"]["architecture_sha256"]
        != payload.get("model_architecture_sha256")
    ):
        raise ValueError("training certificate model architecture differs")

    raw_state = payload.get("model_state_dict")
    if not isinstance(raw_state, Mapping) or set(raw_state) != _MODEL_STATE_KEYS:
        raise ValueError("full-scalar residual model state keys differ")
    state = dict(raw_state)
    expected_initial_state = model.state_dict()
    for name in _MODEL_STATE_KEYS:
        tensor = state.get(name)
        expected_tensor = expected_initial_state[name]
        if (
            not torch.is_tensor(tensor)
            or tensor.dtype != torch.float32
            or tensor.shape != expected_tensor.shape
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"full-scalar residual state tensor {name} differs")
    if (
        surface_region_state_dict_sha256(state)
        != payload.get("model_state_dict_sha256")
    ):
        raise ValueError("full-scalar residual model state authority differs")
    if (
        certificate["model_authority"]["state_dict_sha256"]
        != payload.get("model_state_dict_sha256")
    ):
        raise ValueError("training certificate model state differs")
    if not torch.equal(state["scalar_median"], authority["median"]):
        raise ValueError("model scalar_median differs from normalization authority")
    if not torch.equal(
        state["scalar_robust_scale"], authority["robust_scale"]
    ):
        raise ValueError(
            "model scalar_robust_scale differs from normalization authority"
        )
    if payload.get("checkpoint_state_assertions") != _checkpoint_state_assertions(
        state
    ):
        raise ValueError("full-scalar residual checkpoint identity assertion differs")
    model.load_state_dict(state, strict=True)
    return payload, model


def write_surface_region_full_scalar_residual_checkpoint(
    path: str | Path,
    model: SurfaceRegionAcceptedV2FullScalarResidualV1,
    *,
    normalization_authority: Mapping[str, Any],
    normalization_authority_sha256: str,
    source_state_cohort_authority_sha256: str,
    training_certificate: Mapping[str, Any],
    training_certificate_sha256: str,
) -> tuple[Path, str]:
    """Atomically publish one immutable checkpoint without replacing a file."""

    payload = build_surface_region_full_scalar_residual_checkpoint_payload(
        model,
        normalization_authority=normalization_authority,
        normalization_authority_sha256=normalization_authority_sha256,
        source_state_cohort_authority_sha256=(
            source_state_cohort_authority_sha256
        ),
        training_certificate=training_certificate,
        training_certificate_sha256=training_certificate_sha256,
    )
    output = write_torch_noclobber(path, payload)
    return output, sha256_file(output)


def load_surface_region_full_scalar_residual_checkpoint(
    path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    normalization_authority: Mapping[str, Any],
    expected_normalization_authority_sha256: str,
    expected_source_state_cohort_authority_sha256: str,
    training_certificate: Mapping[str, Any],
    expected_training_certificate_sha256: str,
    map_location: str | torch.device = "cpu",
) -> tuple[SurfaceRegionAcceptedV2FullScalarResidualV1, dict[str, Any]]:
    """Load the sole supported class through a caller-trusted checkpoint SHA."""

    expected_checkpoint = _require_sha256(
        expected_checkpoint_sha256,
        label="full-scalar residual checkpoint",
    )
    payload, _actual_sha256, _source = load_torch_mapping(
        path,
        expected_sha256=expected_checkpoint,
        map_location="cpu",
        label="full-scalar residual checkpoint",
    )
    validated, model = validate_surface_region_full_scalar_residual_checkpoint_payload(
        payload,
        normalization_authority=normalization_authority,
        expected_normalization_authority_sha256=(
            expected_normalization_authority_sha256
        ),
        expected_source_state_cohort_authority_sha256=(
            expected_source_state_cohort_authority_sha256
        ),
        training_certificate=training_certificate,
        expected_training_certificate_sha256=expected_training_certificate_sha256,
    )
    model = model.to(torch.device(map_location)).eval().requires_grad_(False)
    return model, validated


__all__ = [
    "SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_SCHEMA",
    "SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_SCHEMA_VERSION",
    "SURFACE_REGION_FULL_SCALAR_RESIDUAL_CHECKPOINT_CONTRACT_SHA256",
    "SURFACE_REGION_FULL_SCALAR_RESIDUAL_MODEL_CLASS",
    "surface_region_full_scalar_residual_checkpoint_contract",
    "build_surface_region_full_scalar_residual_checkpoint_payload",
    "validate_surface_region_full_scalar_residual_checkpoint_payload",
    "write_surface_region_full_scalar_residual_checkpoint",
    "load_surface_region_full_scalar_residual_checkpoint",
]
