"""Training helpers for RADIO-GS feature-field optimization."""
from .canonical_field_losses import (
    CanonicalFieldLossConfig,
    canonical_primitive_loss,
    normalized_render_reconstruction_loss,
)
from .primitive_consensus import (
    PrimitiveConsensus,
    primitive_reconstruction_loss,
    robust_multiview_consensus,
)

__all__ = [
    "CanonicalFieldLossConfig",
    "PrimitiveConsensus",
    "canonical_primitive_loss",
    "normalized_render_reconstruction_loss",
    "primitive_reconstruction_loss",
    "robust_multiview_consensus",
]
