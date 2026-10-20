"""Shared evidence operations; enable the migration profile explicitly.

The legacy profile remains the default until fresh five-output regression.
Neither profile changes teacher access or task-specific output semantics.
"""
from .core import (enabled, compile_domains, signed_weights, project,
                   grouped_membership_features, scope_response, trace_summary,
                   view_balanced_transfer)

__all__ = ['enabled', 'compile_domains', 'signed_weights', 'project',
           'grouped_membership_features', 'scope_response', 'trace_summary', 'view_balanced_transfer']
