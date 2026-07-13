"""Primitive-first canonical RADIO field components."""

from .basis_decoder import AffineBasisDecoder, fit_affine_basis
from .canonical_gaussian_field import CanonicalGaussianField
from .checkpoint import load_canonical_field_checkpoint
from .field_signature import FeatureSpaceSignature
from .primitive_fusion import PrimitiveFusion

__all__ = [
    "AffineBasisDecoder",
    "CanonicalGaussianField",
    "FeatureSpaceSignature",
    "PrimitiveFusion",
    "fit_affine_basis",
    "load_canonical_field_checkpoint",
]
