#!/usr/bin/env python3
"""Freeze exact-query and generic-negative caches with the official text tower."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.evaluation.openclip_readout import NEGATIVE_PROMPTS
from radio_gs.scripts.eval_lerf_grounding import (
    _SIGLIP2_MODEL_NAME,
    _SIGLIP2_TEXT_CANONICALIZATION,
    encode_text_siglip2,
)


def _queries(raw: str) -> list[str]:
    value = str(raw).strip()
    path = Path(value)
    try:
        is_file = path.is_file()
    except OSError:
        # A comma-separated benchmark vocabulary can exceed the host's
        # filename limit; it is query text, not a malformed cache path.
        is_file = False
    if is_file:
        value = path.read_text(encoding="utf-8")
    result = [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]
    if not result:
        raise ValueError("queries cannot be empty")
    return result


def build(args: argparse.Namespace) -> dict:
    queries = _queries(args.queries)
    negatives = list(NEGATIVE_PROMPTS)
    device = torch.device(args.device)
    embeddings = encode_text_siglip2(queries + negatives, device).float().cpu()
    query_output = Path(args.output)
    negative_output = Path(args.negative_output)
    query_output.parent.mkdir(parents=True, exist_ok=True)
    negative_output.parent.mkdir(parents=True, exist_ok=True)
    common = {
        "prompt_templates": ["{query}"],
        "text_encoder": "siglip2",
        "model_name": _SIGLIP2_MODEL_NAME,
        "text_canonicalization": _SIGLIP2_TEXT_CANONICALIZATION,
    }
    torch.save(
        {**common, "queries": queries, "embeddings": embeddings[: len(queries)]},
        query_output,
    )
    torch.save(
        {**common, "queries": negatives, "embeddings": embeddings[len(queries) :]},
        negative_output,
    )
    return {
        "query_output": str(query_output),
        "negative_output": str(negative_output),
        "queries": queries,
        "negative_queries": negatives,
        "prompt_templates": ["{query}"],
        "text_encoder": "official_siglip2_g",
        "test_set_calibration": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--negative-output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
