#!/usr/bin/env python3
"""Materialize a target-blind LERF q,c cache from the frozen source text head."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from radio_gs.querying.lerf_source_text_likelihood import (
    EFFECTIVE_PROBABILITY_MODES,
    NEUTRAL_ABSTENTION_V1,
    build_lerf_source_text_likelihood_cache,
)
from radio_gs.querying.source_text_query_likelihood import sha256_file


def _output(path: str | Path) -> Path:
    value = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    value.parent.mkdir(parents=True, exist_ok=True)
    return value.parent.resolve(strict=True) / value.name


def _write_torch_noclobber(path: str | Path, payload: dict[str, object]) -> Path:
    output = _output(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable output already exists: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-score-cache", required=True)
    parser.add_argument("--negative-score-cache", required=True)
    support = parser.add_mutually_exclusive_group(required=True)
    support.add_argument("--factorized-state")
    support.add_argument("--canonical-field-checkpoint")
    parser.add_argument("--source-text-head-checkpoint", required=True)
    parser.add_argument("--expected-head-state-sha256", required=True)
    parser.add_argument(
        "--effective-probability-mode",
        choices=EFFECTIVE_PROBABILITY_MODES,
        default=NEUTRAL_ABSTENTION_V1,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if torch.cuda.is_initialized():
        raise RuntimeError("LERF source text likelihood materialization must remain CPU-only")
    payload = build_lerf_source_text_likelihood_cache(
        positive_score_cache=args.positive_score_cache,
        negative_score_cache=args.negative_score_cache,
        factorized_state=args.factorized_state,
        canonical_field_checkpoint=args.canonical_field_checkpoint,
        source_text_head_checkpoint=args.source_text_head_checkpoint,
        expected_head_state_sha256=args.expected_head_state_sha256,
        effective_probability_mode=args.effective_probability_mode,
    )
    output = _write_torch_noclobber(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "rows": int(payload["q"].shape[0]),
                "queries": int(payload["q"].shape[1]),
                "mean_q": float(payload["q"][payload["valid"]].mean()),
                "mean_c": float(payload["c"][payload["valid"]].mean()),
                "effective_probability_mode": payload.get(
                    "effective_probability_mode", NEUTRAL_ABSTENTION_V1
                ),
                "cuda_initialized": torch.cuda.is_initialized(),
                "lerf_ground_truth_or_metric_opened": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
