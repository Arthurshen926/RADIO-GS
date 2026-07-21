"""ScanNet-PFIR-Small: pose-free real-image exemplar to 3-D instance."""

from .protocol import (
    BENCHMARK_VERSION,
    ProtocolConfig,
    audit_manifest,
    build_scene_records,
    freeze_manifest,
)

__all__ = [
    "BENCHMARK_VERSION",
    "ProtocolConfig",
    "audit_manifest",
    "build_scene_records",
    "freeze_manifest",
]

