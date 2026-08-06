"""Primitive-first canonical RADIO field components."""

from .basis_decoder import (
    BASIS_CONDITIONING_CONTRACT_VERSION,
    MAXIMUM_BASIS_CONDITION_NUMBER_V1,
    AffineBasisDecoder,
    BasisConditioningReport,
    basis_conditioning_report,
    fit_affine_basis,
    validate_basis_conditioning,
)
from .canonical_gaussian_field import CanonicalGaussianField
from .directional_canonical_field import (
    DirectionalCanonicalField,
    load_directional_canonical_field,
)
from .directional_distribution import (
    DIRECTIONAL_PROTOTYPE_CONTRACT,
    DirectionalPrototypeSet,
    directional_prototype_observation_cosines,
    directional_prototype_coverage,
    directional_set_ranking_loss,
    directional_set_rms_loss,
    fit_two_direction_prototypes,
)
from .boundary_screen_residual import (
    BoundaryConditionedScreenResidual,
    load_boundary_screen_residual_checkpoint,
)
from .checkpoint import (
    load_canonical_field_checkpoint,
    load_factorized_canonical_field_checkpoint,
)
from .view_residual import ZeroMeanViewResidual, load_view_residual_checkpoint
from .observation_lifting_contract import (
    CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
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
    "BASIS_CONDITIONING_CONTRACT_VERSION",
    "MAXIMUM_BASIS_CONDITION_NUMBER_V1",
    "BasisConditioningReport",
    "basis_conditioning_report",
    "CanonicalGaussianField",
    "DirectionalCanonicalField",
    "DIRECTIONAL_PROTOTYPE_CONTRACT",
    "DirectionalPrototypeSet",
    "directional_prototype_observation_cosines",
    "directional_prototype_coverage",
    "directional_set_ranking_loss",
    "directional_set_rms_loss",
    "fit_two_direction_prototypes",
    "load_directional_canonical_field",
    "BoundaryConditionedScreenResidual",
    "FeatureSpaceSignature",
    "PrimitiveFusion",
    "PrimitiveReliability",
    "PrimitiveSpatialHash",
    "canonical_primitive_reliability",
    "fit_affine_basis",
    "validate_basis_conditioning",
    "load_canonical_field_checkpoint",
    "load_factorized_canonical_field_checkpoint",
    "load_boundary_screen_residual_checkpoint",
    "ZeroMeanViewResidual",
    "load_view_residual_checkpoint",
    "CANONICAL_FULL_OBSERVATION_CONTRACT_NAME",
    "CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME",
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
