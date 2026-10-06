"""Explicit scene/query access and typed outputs for the current RADIO-GS system.

The source-assisted runtime freezes the map and reconstructs transient response
and selection heads. Map-only querying reads native coordinates without source
photographs. Neither mode is a claim of complete object or four-query accuracy.
"""
from .runtime import PipelineConfig, build_plan, run

__all__ = ['PipelineConfig', 'build_plan', 'run']
