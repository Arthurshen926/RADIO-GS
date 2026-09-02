"""Query-cardinality contract shared by every future v4 query adapter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuerySelectionMode(str, Enum):
    """How a query is allowed to select persistent scene memory.

    The mode is deliberately independent of query modality. Image, prompt,
    and text adapters must declare cardinality explicitly instead of choosing
    a normalization rule inside an evaluator.
    """

    SINGLE_INSTANCE = "single_instance"
    MULTI_INSTANCE = "multi_instance"
    LOCAL_SEMANTIC = "local_semantic"


@dataclass(frozen=True)
class QueryPacket:
    """Minimal v4 query boundary before modality adapters are connected.

    Phase 0 only freezes query cardinality. Encoder tokens and spatial
    payloads will be added by their later gated phases; keeping them out here
    prevents this contract work from silently introducing a text path.
    """

    selection_mode: QuerySelectionMode | str

    def __post_init__(self) -> None:
        value = self.selection_mode
        if isinstance(value, QuerySelectionMode):
            mode = value
        elif isinstance(value, str):
            try:
                mode = QuerySelectionMode(value)
            except ValueError as error:
                allowed = ", ".join(item.value for item in QuerySelectionMode)
                raise ValueError(
                    f"unsupported selection_mode {value!r}; expected one of: {allowed}"
                ) from error
        else:
            raise TypeError(
                "selection_mode must be a QuerySelectionMode or its exact string value"
            )
        object.__setattr__(self, "selection_mode", mode)


__all__ = ["QueryPacket", "QuerySelectionMode"]
