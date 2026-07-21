"""DINO multi-prototype exemplar unary."""

from __future__ import annotations

import numpy as np


def score_dino(query_tokens: np.ndarray, field_descriptors: np.ndarray) -> np.ndarray:
    tokens = np.asarray(query_tokens, dtype=np.float32)
    field = np.asarray(field_descriptors, dtype=np.float32)
    if tokens.ndim != 2 or field.ndim != 2 or tokens.shape[1] != field.shape[1]:
        raise ValueError("DINO tokens/field dimensions differ")
    tokens /= np.maximum(np.linalg.norm(tokens, axis=1, keepdims=True), 1e-8)
    field /= np.maximum(np.linalg.norm(field, axis=1, keepdims=True), 1e-8)
    # Max-over-query-tokens retains instance-specific parts rather than turning
    # the crop into a category-only global average.
    return (field @ tokens.T).max(axis=1).astype(np.float32)

