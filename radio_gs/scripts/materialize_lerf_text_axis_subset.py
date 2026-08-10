#!/usr/bin/env python3
"""Materialize a SHA-bound single-prompt LERF text-axis subset.

The producer never re-encodes text.  It selects exact rows from an immutable
source embedding bank in the order declared by a second immutable cache and
adds the explicit official C-RADIO SigLIP2-G canonicalization contract needed
by the FP32 O0 score materializer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache import (
    SIGLIP2_MODEL_NAME,
    SIGLIP2_TEXT_CANONICALIZATION,
)
from radio_gs.utils.immutable_artifacts import (
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.lerf_text_axis_subset.v1"


def _queries(value: object, *, label: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} queries must be a non-empty sequence")
    result = [str(item) for item in value]
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{label} queries must be unique non-empty strings")
    return result


def _source(value: Mapping[str, Any]) -> tuple[list[str], torch.Tensor]:
    queries = _queries(value.get("queries"), label="source bank")
    embeddings = value.get("embeddings")
    if (
        not torch.is_tensor(embeddings)
        or embeddings.dtype != torch.float32
        or embeddings.shape != (len(queries), 1536)
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("source bank embeddings must be finite FP32 [Q,1536]")
    if value.get("prompt_templates") != ["{query}"]:
        raise ValueError("source bank is not the frozen single-prompt axis")
    if value.get("text_encoder") != "siglip2" or value.get("model_name") != SIGLIP2_MODEL_NAME:
        raise ValueError("source bank text encoder differs")
    existing = value.get("text_canonicalization")
    if existing not in (None, "", SIGLIP2_TEXT_CANONICALIZATION):
        raise ValueError("source bank text canonicalization differs")
    return queries, embeddings


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def materialize(
    *,
    source_bank: str | Path,
    expected_source_bank_sha256: str,
    query_order_cache: str | Path,
    expected_query_order_cache_sha256: str,
    output: str | Path,
) -> dict[str, Any]:
    output_path = Path(output).expanduser().resolve()
    report_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() or report_path.exists():
        raise FileExistsError("immutable text-axis output already exists")

    source, source_sha256, source_path = load_torch_mapping(
        source_bank,
        expected_sha256=expected_source_bank_sha256,
        map_location="cpu",
        label="frozen single-prompt source text bank",
    )
    order, order_sha256, order_path = load_torch_mapping(
        query_order_cache,
        expected_sha256=expected_query_order_cache_sha256,
        map_location="cpu",
        label="frozen scene query-order cache",
    )
    source_queries, source_embeddings = _source(source)
    requested = _queries(order.get("queries"), label="query-order cache")
    source_index = {query: index for index, query in enumerate(source_queries)}
    missing = [query for query in requested if query not in source_index]
    if missing:
        raise ValueError(f"query-order cache contains queries absent from source: {missing}")
    indices = torch.tensor([source_index[query] for query in requested], dtype=torch.long)
    embeddings = source_embeddings.index_select(0, indices).contiguous()
    payload = {
        "prompt_templates": ["{query}"],
        "text_encoder": "siglip2",
        "model_name": SIGLIP2_MODEL_NAME,
        "text_canonicalization": SIGLIP2_TEXT_CANONICALIZATION,
        "queries": requested,
        "embeddings": embeddings,
    }
    write_torch_noclobber(output_path, payload)
    output_sha256 = sha256_file(output_path)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete_exact_row_subset_without_text_reencoding",
        "source_bank": {"path": str(source_path), "sha256": source_sha256},
        "query_order_cache": {"path": str(order_path), "sha256": order_sha256},
        "output": {"path": str(output_path), "sha256": output_sha256},
        "queries": requested,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "embedding_tensor_sha256": _tensor_sha256(embeddings),
        "text_canonicalization": SIGLIP2_TEXT_CANONICALIZATION,
        "source_rows_copied_exactly": True,
        "text_reencoded": False,
        "ground_truth_opened": False,
        "benchmark_masks_opened": False,
        "metric_outputs_opened": False,
        "metric_computed": False,
    }
    write_frozen_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--expected-source-bank-sha256", required=True)
    parser.add_argument("--query-order-cache", required=True)
    parser.add_argument("--expected-query-order-cache-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(materialize(
        source_bank=args.source_bank,
        expected_source_bank_sha256=args.expected_source_bank_sha256,
        query_order_cache=args.query_order_cache,
        expected_query_order_cache_sha256=args.expected_query_order_cache_sha256,
        output=args.output,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
