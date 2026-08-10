"""Deploy V2.1 descriptors with their exact source-training relevance gauge.

This module is intentionally small.  It does not smooth, rank-normalize,
min-max remap, select a scale, or inspect a benchmark label.  It applies the
same positive-versus-hardest-canonical-negative binary softmax used by the
V2.1 source loss, so 0.5 remains the model-defined equal-logit boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    CANONICAL_NEGATIVE_MODEL,
    FrozenCanonicalNegativeBank,
    INFERENCE_LOGIT_SCALE,
)
from radio_gs.querying.unified_query import cosine_relevancy_torch
from radio_gs.utils.immutable_artifacts import load_torch_mapping


DESCRIPTOR_DIM = 1536
OFFICIAL_TEXT_CANONICALIZATION = "official_c_radio_siglip2_g"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class V21PositiveTextBank:
    """One immutable exact-query bank in the V2.1 descriptor gauge."""

    query_ids: tuple[str, ...]
    embeddings: torch.Tensor
    file_sha256: str
    embedding_tensor_sha256: str
    model_id: str
    text_canonicalization: str


def _unit_cpu_float32_matrix(value: object, *, label: str) -> torch.Tensor:
    if (
        not torch.is_tensor(value)
        or value.dtype != torch.float32
        or value.device.type != "cpu"
        or value.ndim != 2
        or min(value.shape) <= 0
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be finite CPU float32 [N,D]")
    result = value.detach().contiguous()
    norms = torch.linalg.vector_norm(result, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=2e-4):
        raise ValueError(f"{label} rows must be unit L2")
    return result


def load_v21_positive_text_bank(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> V21PositiveTextBank:
    """Load the official exact-query SigLIP2 cache without schema widening."""

    payload, digest, _ = load_torch_mapping(
        path,
        expected_sha256=expected_file_sha256,
        map_location="cpu",
        label="V2.1 positive text bank",
    )
    required = {
        "queries",
        "prompt_templates",
        "text_encoder",
        "model_name",
        "text_canonicalization",
        "embeddings",
    }
    if set(payload) != required:
        raise ValueError("V2.1 positive text bank fields differ")
    queries = payload["queries"]
    if (
        not isinstance(queries, list)
        or not queries
        or any(not isinstance(item, str) or not item.strip() for item in queries)
        or len(set(queries)) != len(queries)
        or payload["prompt_templates"] != ["{query}"]
        or payload["text_encoder"] != "siglip2"
        or payload["model_name"] != CANONICAL_NEGATIVE_MODEL
    ):
        raise ValueError("V2.1 positive text semantics differ")
    if payload["text_canonicalization"] != OFFICIAL_TEXT_CANONICALIZATION:
        raise ValueError("V2.1 positive text canonicalization differs")
    embeddings = _unit_cpu_float32_matrix(
        payload["embeddings"], label="V2.1 positive text embeddings"
    )
    if embeddings.shape != (len(queries), DESCRIPTOR_DIM):
        raise ValueError("V2.1 positive text embedding shape differs")
    return V21PositiveTextBank(
        query_ids=tuple(queries),
        embeddings=embeddings,
        file_sha256=digest,
        embedding_tensor_sha256=tensor_sha256(embeddings),
        model_id=CANONICAL_NEGATIVE_MODEL,
        text_canonicalization=OFFICIAL_TEXT_CANONICALIZATION,
    )


def calibrated_v21_absolute_relevance(
    semantic_descriptor: torch.Tensor,
    *,
    positive_bank: V21PositiveTextBank,
    canonical_negative_bank: FrozenCanonicalNegativeBank,
) -> torch.Tensor:
    """Return exact source-calibrated ``[region, query]`` probabilities."""

    if type(positive_bank) is not V21PositiveTextBank:
        raise TypeError("positive_bank must be an immutable V2.1 text bank")
    if type(canonical_negative_bank) is not FrozenCanonicalNegativeBank:
        raise TypeError("canonical_negative_bank must be frozen and SHA-bound")
    descriptor = _unit_cpu_float32_matrix(
        semantic_descriptor, label="V2.1 semantic descriptor"
    )
    positive = _unit_cpu_float32_matrix(
        positive_bank.embeddings, label="V2.1 positive text embeddings"
    )
    negative = _unit_cpu_float32_matrix(
        canonical_negative_bank.embeddings,
        label="V2.1 canonical-negative embeddings",
    )
    if (
        descriptor.shape[1] != DESCRIPTOR_DIM
        or positive.shape[1] != DESCRIPTOR_DIM
        or negative.shape != (4, DESCRIPTOR_DIM)
        or positive_bank.model_id != CANONICAL_NEGATIVE_MODEL
        or canonical_negative_bank.model_id != CANONICAL_NEGATIVE_MODEL
        or positive_bank.text_canonicalization != OFFICIAL_TEXT_CANONICALIZATION
        or len(positive_bank.query_ids) != positive.shape[0]
        or len(set(positive_bank.query_ids)) != len(positive_bank.query_ids)
        or any(not item for item in positive_bank.query_ids)
        or _SHA256.fullmatch(positive_bank.file_sha256) is None
        or _SHA256.fullmatch(positive_bank.embedding_tensor_sha256) is None
        or _SHA256.fullmatch(canonical_negative_bank.file_sha256) is None
        or _SHA256.fullmatch(canonical_negative_bank.embedding_tensor_sha256) is None
        or tensor_sha256(positive) != positive_bank.embedding_tensor_sha256
        or tensor_sha256(negative) != canonical_negative_bank.embedding_tensor_sha256
    ):
        raise ValueError("V2.1 relevance bank authority differs")
    relevance = (
        cosine_relevancy_torch(
            descriptor,
            positive,
            negative,
            logit_scale=INFERENCE_LOGIT_SCALE,
            assume_normalized=True,
        )
        .detach()
        .float()
        .cpu()
        .contiguous()
    )
    if (
        relevance.shape != (descriptor.shape[0], positive.shape[0])
        or not bool(torch.isfinite(relevance).all())
        or bool((relevance < 0).any())
        or bool((relevance > 1).any())
    ):
        raise RuntimeError("V2.1 calibrated relevance output differs")
    return relevance


__all__ = [
    "DESCRIPTOR_DIM",
    "OFFICIAL_TEXT_CANONICALIZATION",
    "V21PositiveTextBank",
    "calibrated_v21_absolute_relevance",
    "load_v21_positive_text_bank",
]
