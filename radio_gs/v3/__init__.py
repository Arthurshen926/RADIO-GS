"""Structured Universal Gaussian Memory v3.

This namespace is intentionally isolated from historical benchmark readouts.
Only stable geometry, rendering, checkpoint, and immutable-artifact utilities
may be imported by v3 core modules.
"""

from radio_gs.v3.contracts.method import SUGM_V3_CONTRACT, validate_scene_state

__all__ = ["SUGM_V3_CONTRACT", "validate_scene_state"]
