"""Fail-closed source/development/benchmark identity separation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSplit:
    source_ids: frozenset[str]
    development_ids: frozenset[str]
    benchmark_ids: frozenset[str]

    def __post_init__(self) -> None:
        overlaps = {
            "source/development": self.source_ids & self.development_ids,
            "source/benchmark": self.source_ids & self.benchmark_ids,
            "development/benchmark": self.development_ids & self.benchmark_ids,
        }
        invalid = {name: sorted(values) for name, values in overlaps.items() if values}
        if invalid:
            raise ValueError(f"source split identities overlap: {invalid}")

    def require_source(self, identity: str) -> None:
        if identity not in self.source_ids:
            raise PermissionError(f"identity {identity!r} is not sealed source authority")
