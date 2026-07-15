"""Primitive-first canonical RADIO field components."""

from .basis_decoder import AffineBasisDecoder, fit_affine_basis
from .canonical_gaussian_field import CanonicalGaussianField
from .checkpoint import load_canonical_field_checkpoint
from .view_residual import ZeroMeanViewResidual, load_view_residual_checkpoint
from .observation_lifting_contract import (
    CANONICAL_OBSERVATION_CONTRACT_NAME,
    apply_canonical_observation_contract,
    canonical_observation_contract,
    observation_contract_sha256,
    validate_observation_contract_metadata,
)
from .field_signature import FeatureSpaceSignature
from .primitive_fusion import PrimitiveFusion
from .primitive_reliability import (
    PrimitiveReliability,
    canonical_primitive_reliability,
)
from .spatial_hash import PrimitiveSpatialHash

__all__ = [
    "AffineBasisDecoder",
    "CanonicalGaussianField",
    "FeatureSpaceSignature",
    "PrimitiveFusion",
    "PrimitiveReliability",
    "PrimitiveSpatialHash",
    "canonical_primitive_reliability",
    "fit_affine_basis",
    "load_canonical_field_checkpoint",
    "ZeroMeanViewResidual",
    "load_view_residual_checkpoint",
    "CANONICAL_OBSERVATION_CONTRACT_NAME",
    "apply_canonical_observation_contract",
    "canonical_observation_contract",
    "observation_contract_sha256",
    "validate_observation_contract_metadata",
]
