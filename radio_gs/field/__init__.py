"""Primitive-first canonical RADIO field components."""

from .basis_decoder import AffineBasisDecoder, fit_affine_basis
from .canonical_gaussian_field import CanonicalGaussianField
from .boundary_screen_residual import (
    BoundaryConditionedScreenResidual,
    load_boundary_screen_residual_checkpoint,
)
from .checkpoint import load_canonical_field_checkpoint
from .view_residual import ZeroMeanViewResidual, load_view_residual_checkpoint
from .observation_lifting_contract import (
    CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_CONTRACT_NAMES,
    CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    CANONICAL_OBSERVATION_CONTRACT_NAME,
    apply_canonical_observation_contract,
    canonical_observation_contract,
    is_canonical_full_observation_contract,
    observation_contract_sha256,
    select_full_observation_coverage_ranked_dataset_indices,
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
    "BoundaryConditionedScreenResidual",
    "FeatureSpaceSignature",
    "PrimitiveFusion",
    "PrimitiveReliability",
    "PrimitiveSpatialHash",
    "canonical_primitive_reliability",
    "fit_affine_basis",
    "load_canonical_field_checkpoint",
    "load_boundary_screen_residual_checkpoint",
    "ZeroMeanViewResidual",
    "load_view_residual_checkpoint",
    "CANONICAL_FULL_OBSERVATION_CONTRACT_NAME",
    "CANONICAL_FULL_OBSERVATION_CONTRACT_NAMES",
    "CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME",
    "CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME",
    "CANONICAL_OBSERVATION_CONTRACT_NAME",
    "apply_canonical_observation_contract",
    "canonical_observation_contract",
    "is_canonical_full_observation_contract",
    "observation_contract_sha256",
    "select_full_observation_coverage_ranked_dataset_indices",
    "validate_observation_contract_metadata",
]
