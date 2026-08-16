"""Fail-closed checkpoint I/O for a canonical Gaussian RADIO field."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Mapping

import torch

from radio_gs.utils.immutable_artifacts import load_torch_mapping

from .basis_decoder import AffineBasisDecoder, validate_basis_conditioning
from .canonical_gaussian_field import CanonicalGaussianField
from .factorized_radio_contract import (
    CANONICAL_FACTORIZED_RADIO_CHECKPOINT_CONTRACT,
    CANONICAL_FACTORIZED_RADIO_CHECKPOINT_SCHEMA_VERSION,
    FactorizedRadioFieldSignature,
    validate_factorized_radio_checkpoint_metadata,
)
from .field_signature import FeatureSpaceSignature


def load_canonical_field_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_sha256: str | None = None,
) -> tuple[CanonicalGaussianField, Mapping[str, Any]]:
    payload, _, _ = load_torch_mapping(
        Path(path),
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="canonical RADIO field checkpoint",
    )
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("not a canonical RADIO field schema-v1 checkpoint")
    field = _canonical_field_from_payload(payload, map_location=map_location)
    return field, payload


def _canonical_field_from_payload(
    payload: Mapping[str, Any],
    *,
    map_location: str | torch.device,
    signature: FeatureSpaceSignature | None = None,
) -> CanonicalGaussianField:
    """Construct a field after the caller has validated its schema boundary."""

    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("canonical field checkpoint lacks architecture metadata")
    allowed_architecture = {
        "num_gaussians",
        "feature_dim",
        "coefficient_dim",
        "local_dim",
        "coarse_dim",
        "spatial_hash",
        "position_storage",
        "fusion_reliability",
        "hidden_dim",
        "use_fusion",
        "trainable_basis",
        "trainable_statistics",
        "fusion_residual_blocks",
    }
    required_architecture = {
        "num_gaussians",
        "feature_dim",
        "coefficient_dim",
        "local_dim",
        "coarse_dim",
        "fusion_reliability",
        "hidden_dim",
        "use_fusion",
    }
    if not required_architecture.issubset(architecture) or not set(
        architecture
    ).issubset(allowed_architecture):
        raise ValueError("canonical field architecture fields differ")

    def bounded_int(name: str, minimum: int, maximum: int) -> int:
        value = architecture.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"canonical field architecture {name} is out of bounds")
        return int(value)

    num_gaussians = bounded_int("num_gaussians", 1, 10_000_000)
    feature_dim = bounded_int("feature_dim", 1, 16_384)
    coefficient_dim = bounded_int("coefficient_dim", 1, feature_dim)
    local_dim = bounded_int("local_dim", 1, 16_384)
    coarse_dim = bounded_int("coarse_dim", 0, 4_096)
    hidden_dim = bounded_int("hidden_dim", 1, 16_384)
    residual_blocks_value = architecture.get("fusion_residual_blocks", 0)
    if (
        not isinstance(residual_blocks_value, int)
        or isinstance(residual_blocks_value, bool)
        or not 0 <= residual_blocks_value <= 32
    ):
        raise ValueError("canonical field fusion_residual_blocks is out of bounds")
    residual_blocks = int(residual_blocks_value)
    for name in (
        "fusion_reliability",
        "use_fusion",
        "trainable_basis",
        "trainable_statistics",
    ):
        if name in architecture and not isinstance(architecture[name], bool):
            raise ValueError(f"canonical field architecture {name} must be boolean")
    use_fusion = bool(architecture["use_fusion"])
    fusion_reliability = bool(architecture["fusion_reliability"])
    if not use_fusion and (coarse_dim or residual_blocks):
        raise ValueError("canonical field direct mode has fusion-only dimensions")

    reliability_value = payload.get("reliability")
    if reliability_value is None:
        reliability = None
        reliability_dim = 0
    else:
        reliability = torch.as_tensor(reliability_value)
        if (
            reliability.ndim != 2
            or reliability.shape[0] != num_gaussians
            or reliability.shape[1] > 64
            or not reliability.dtype.is_floating_point
            or not bool(torch.isfinite(reliability).all())
        ):
            raise ValueError("canonical field reliability tensor is malformed")
        reliability_dim = int(reliability.shape[1])

    spatial_hash = architecture.get("spatial_hash")
    spatial_spec: dict[str, int] | None = None
    if coarse_dim:
        if not isinstance(spatial_hash, Mapping):
            raise ValueError("canonical field coarse mode lacks spatial hash")
        required_spatial = {
            "num_levels",
            "features_per_level",
            "log2_hashmap_size",
            "base_resolution",
            "max_resolution",
            "hidden_dim",
        }
        if set(spatial_hash) not in (
            required_spatial,
            required_spatial | {"output_dim"},
        ):
            raise ValueError("canonical field spatial hash fields differ")

        def spatial_int(name: str, minimum: int, maximum: int) -> int:
            value = spatial_hash.get(name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"canonical field spatial hash {name} is out of bounds"
                )
            return int(value)

        spatial_spec = {
            "output_dim": coarse_dim,
            "num_levels": spatial_int("num_levels", 1, 64),
            "features_per_level": spatial_int("features_per_level", 1, 64),
            "log2_hashmap_size": spatial_int("log2_hashmap_size", 1, 24),
            "base_resolution": spatial_int("base_resolution", 1, 65_536),
            "max_resolution": spatial_int("max_resolution", 1, 65_536),
            "hidden_dim": spatial_int("hidden_dim", 1, 8_192),
        }
        if spatial_spec["max_resolution"] < spatial_spec["base_resolution"]:
            raise ValueError("canonical field spatial hash resolution order differs")
        if "output_dim" in spatial_hash and spatial_hash["output_dim"] != coarse_dim:
            raise ValueError("canonical field spatial hash output dimension differs")
    elif spatial_hash is not None:
        raise ValueError("canonical field without coarse codes has a spatial hash")

    state_value = payload.get("state_dict")
    if not isinstance(state_value, Mapping) or len(state_value) > 512:
        raise ValueError("canonical field state_dict is malformed")
    state_dict = dict(state_value)
    reliability_state_dim = reliability_dim if fusion_reliability else 0
    expected_shapes: dict[str, tuple[int, ...]] = {
        "local_codes": (num_gaussians, local_dim),
        "reliability": (num_gaussians, reliability_dim),
        "decoder.basis": (feature_dim, coefficient_dim),
        "decoder.mean": (feature_dim,),
        "decoder.log_scale": (feature_dim,),
    }
    expected_dtypes: dict[str, torch.dtype] = {
        key: torch.float32 for key in expected_shapes
    }
    if spatial_spec is not None:
        levels = spatial_spec["num_levels"]
        features_per_level = spatial_spec["features_per_level"]
        growth = (
            math.exp(
                math.log(
                    spatial_spec["max_resolution"] / spatial_spec["base_resolution"]
                )
                / (levels - 1)
            )
            if levels > 1
            else 1.0
        )
        resolutions = [
            max(
                1,
                int(math.floor(spatial_spec["base_resolution"] * growth**level)),
            )
            for level in range(levels)
        ]
        expected_shapes.update(
            {
                "normalized_positions": (num_gaussians, 3),
                "position_minimum": (3,),
                "position_extent": (3,),
                "spatial_encoder.resolutions": (levels,),
                "spatial_encoder.corner_offsets": (8, 3),
                "spatial_encoder.hash_primes": (3,),
                "spatial_encoder.mlp.0.weight": (
                    spatial_spec["hidden_dim"],
                    levels * features_per_level,
                ),
                "spatial_encoder.mlp.0.bias": (spatial_spec["hidden_dim"],),
                "spatial_encoder.mlp.2.weight": (
                    coarse_dim,
                    spatial_spec["hidden_dim"],
                ),
                "spatial_encoder.mlp.2.bias": (coarse_dim,),
            }
        )
        for level, resolution in enumerate(resolutions):
            table_size = min(
                2 ** spatial_spec["log2_hashmap_size"],
                (resolution + 1) ** 3,
            )
            expected_shapes[f"spatial_encoder.hash_tables.{level}"] = (
                table_size,
                features_per_level,
            )
        expected_dtypes.update(
            {
                key: torch.float32
                for key in expected_shapes
                if key not in expected_dtypes
            }
        )
        expected_dtypes["normalized_positions"] = torch.float16
        for key in (
            "spatial_encoder.resolutions",
            "spatial_encoder.corner_offsets",
            "spatial_encoder.hash_primes",
        ):
            expected_dtypes[key] = torch.int64
    if use_fusion:
        fusion_input = local_dim + coarse_dim + reliability_state_dim
        expected_shapes.update(
            {
                "fusion.network.0.weight": (hidden_dim, fusion_input),
                "fusion.network.0.bias": (hidden_dim,),
                "fusion.network.2.weight": (coefficient_dim, hidden_dim),
                "fusion.network.2.bias": (coefficient_dim,),
                "fusion.gate.0.weight": (coefficient_dim, fusion_input),
                "fusion.gate.0.bias": (coefficient_dim,),
            }
        )
        for index in range(residual_blocks):
            prefix = f"fusion.residual_blocks.{index}"
            expected_shapes.update(
                {
                    f"{prefix}.0.weight": (coefficient_dim,),
                    f"{prefix}.0.bias": (coefficient_dim,),
                    f"{prefix}.1.weight": (hidden_dim, coefficient_dim),
                    f"{prefix}.1.bias": (hidden_dim,),
                    f"{prefix}.3.weight": (coefficient_dim, hidden_dim),
                    f"{prefix}.3.bias": (coefficient_dim,),
                }
            )
        if local_dim != coefficient_dim:
            expected_shapes.update(
                {
                    "fusion.base_projection.weight": (
                        coefficient_dim,
                        local_dim,
                    ),
                    "fusion.base_projection.bias": (coefficient_dim,),
                }
            )
        expected_dtypes.update(
            {
                key: torch.float32
                for key in expected_shapes
                if key not in expected_dtypes
            }
        )
    if set(state_dict) != set(expected_shapes):
        raise ValueError("canonical field state_dict keys differ from architecture")
    for name, expected_shape in expected_shapes.items():
        tensor = state_dict[name]
        if (
            not torch.is_tensor(tensor)
            or tuple(tensor.shape) != expected_shape
            or tensor.dtype != expected_dtypes[name]
        ):
            raise ValueError(f"canonical field state tensor {name} differs")
        if tensor.dtype.is_floating_point and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"canonical field state tensor {name} is non-finite")
    if reliability is not None and not torch.equal(
        reliability.float(), state_dict["reliability"]
    ):
        raise ValueError("canonical field reliability copies differ")

    # Decoder coordinates are an authority boundary, not merely trainable
    # weights.  Validate the persisted basis before constructing a model or
    # exposing any decoded feature.  Legacy schema-v1 assets remain eligible:
    # the versioned contract is recomputed from their state rather than
    # requiring newly written metadata.
    validate_basis_conditioning(state_dict["decoder.basis"])

    payload_signature = (
        FeatureSpaceSignature.from_mapping(payload["feature_signature"])
        if signature is None
        else signature
    )
    if payload_signature.raw_feature_dim != feature_dim:
        raise ValueError("canonical field signature feature dimension differs")
    decoder = AffineBasisDecoder(
        feature_dim=feature_dim,
        coefficient_dim=coefficient_dim,
        trainable_basis=bool(architecture.get("trainable_basis", True)),
        trainable_statistics=bool(architecture.get("trainable_statistics", False)),
    )
    field = CanonicalGaussianField(
        num_gaussians=num_gaussians,
        decoder=decoder,
        signature=payload_signature,
        local_dim=local_dim,
        coarse_dim=coarse_dim,
        spatial_hash=spatial_spec,
        reliability=reliability,
        fusion_reliability=bool(architecture.get("fusion_reliability", True)),
        hidden_dim=hidden_dim,
        fusion_residual_blocks=residual_blocks,
        use_fusion=use_fusion,
    )
    if spatial_spec is not None:
        for name in (
            "spatial_encoder.resolutions",
            "spatial_encoder.corner_offsets",
            "spatial_encoder.hash_primes",
        ):
            if not torch.equal(field.state_dict()[name], state_dict[name]):
                raise ValueError(f"canonical field fixed spatial buffer {name} differs")
    field.load_state_dict(state_dict, strict=True)
    target = torch.device(map_location)
    if target.type != "cpu":
        field = field.to(target)
    return field


def load_factorized_canonical_field_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_sha256: str | None = None,
    expected_signature: FactorizedRadioFieldSignature | None = None,
) -> tuple[CanonicalGaussianField, Mapping[str, Any], FactorizedRadioFieldSignature]:
    """Load a schema-v2 factorized field without accepting schema-v1 assets."""

    payload, _, _ = load_torch_mapping(
        Path(path),
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="factorized canonical RADIO field checkpoint",
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version")
        != CANONICAL_FACTORIZED_RADIO_CHECKPOINT_SCHEMA_VERSION
        or payload.get("checkpoint_contract")
        != CANONICAL_FACTORIZED_RADIO_CHECKPOINT_CONTRACT
    ):
        raise ValueError("not a canonical factorized RADIO field schema-v2 checkpoint")
    metadata = payload.get("factorized_radio_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("factorized field checkpoint lacks contract metadata")
    factorized_signature = validate_factorized_radio_checkpoint_metadata(
        metadata,
        expected_signature=expected_signature,
    )
    if "feature_signature" in payload:
        raise ValueError("factorized field forbids a legacy feature signature copy")
    if metadata.get("checkpoint_contract") != payload.get("checkpoint_contract"):
        raise ValueError("factorized field checkpoint contract copies differ")
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("factorized field checkpoint lacks model architecture")
    if architecture.get("fusion_reliability") is not False:
        raise ValueError("factorized field must disable reliability fusion")
    if int(architecture.get("feature_dim", -1)) != int(
        factorized_signature.base_feature_signature.raw_feature_dim
    ):
        raise ValueError("factorized field model feature dimension differs")
    reliability = payload.get("reliability")
    if reliability is not None and (
        not torch.is_tensor(reliability)
        or reliability.ndim != 2
        or reliability.shape[1] != 0
    ):
        raise ValueError("factorized field must not persist target reliability")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("factorized field state_dict is malformed")
    state_reliability = state.get("reliability")
    if (
        not torch.is_tensor(state_reliability)
        or state_reliability.ndim != 2
        or state_reliability.shape[1] != 0
    ):
        raise ValueError("factorized field state must have zero reliability columns")
    geometry = payload.get("geometry_fingerprint")
    if (
        not isinstance(geometry, Mapping)
        or set(geometry) != {"num_gaussians", "xyz_sha256"}
        or int(geometry.get("num_gaussians", -1))
        != int(architecture.get("num_gaussians", -2))
        or re.fullmatch(r"[0-9a-f]{64}", str(geometry.get("xyz_sha256", ""))) is None
    ):
        raise ValueError("factorized field geometry fingerprint differs")
    for name in ("factorized_cache_sha256", "feature_output_bundle_sha256"):
        value = payload.get(name)
        if re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is None:
            raise ValueError(f"factorized field {name} differs")
    field = _canonical_field_from_payload(
        payload,
        map_location=map_location,
        signature=factorized_signature.base_feature_signature,
    )
    if field.reliability.shape != (field.num_gaussians, 0):
        raise ValueError("factorized field reconstructed target reliability")
    return field, payload, factorized_signature


def load_universal_canonical_field_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_sha256: str | None = None,
) -> tuple[CanonicalGaussianField, Mapping[str, Any], FactorizedRadioFieldSignature]:
    """Load Universal Field v1 with deployment reliability kept out of fusion."""

    from radio_gs.universal_field_v1 import validate_universal_field_payload

    payload, _, _ = load_torch_mapping(
        Path(path),
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="Universal Field v1 checkpoint",
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Universal Field v1 checkpoint is not a mapping")
    validate_universal_field_payload(payload)
    metadata = payload.get("factorized_radio_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Universal Field v1 lacks factorized RADIO metadata")
    signature = validate_factorized_radio_checkpoint_metadata(metadata)
    field = _canonical_field_from_payload(
        payload,
        map_location=map_location,
        signature=signature.base_feature_signature,
    )
    if field.fusion_reliability or field.reliability.shape != (
        field.num_gaussians,
        5,
    ):
        raise ValueError("Universal Field v1 reliability reconstruction differs")
    return field, payload, signature
