"""SigLIP summary-to-primitive cosine unary."""

from __future__ import annotations

import numpy as np


def score_siglip(query_summary: np.ndarray, field_descriptors: np.ndarray) -> np.ndarray:
    query = np.asarray(query_summary, dtype=np.float32).reshape(-1)
    field = np.asarray(field_descriptors, dtype=np.float32)
    if field.ndim != 2 or field.shape[1] != query.size:
        raise ValueError("SigLIP query/field dimensions differ")
    query /= max(float(np.linalg.norm(query)), 1e-8)
    field = field / np.maximum(np.linalg.norm(field, axis=1, keepdims=True), 1e-8)
    return (field @ query).astype(np.float32)

