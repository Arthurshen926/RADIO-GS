"""Scene-specific Gaussian capability fields with a shared membership core.

This research package has no dependency on the historical benchmark router.
Native encoders and source observation compilation are outside deployment I/O.
"""

from .contracts import QueryPacket, QueryResult, Support
from .field import CapabilityField

__all__ = ["CapabilityField", "QueryPacket", "QueryResult", "Support"]
