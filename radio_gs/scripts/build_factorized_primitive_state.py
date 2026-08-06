#!/usr/bin/env python3
"""Build a CPU-only, missingness-safe factorized primitive-state sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.interfaces.factorized_primitive_state import (
    build_factorized_primitive_state,
    load_factorized_field_support,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    write_frozen_json,
    write_torch_noclobber,
)


def build(args: argparse.Namespace) -> dict[str, object]:
    support = load_factorized_field_support(
        args.field_checkpoint,
        expected_field_checkpoint_sha256=str(
            args.expected_field_checkpoint_sha256
        ),
        mpr_cache=(str(args.factorized_radio_cache) or None),
        expected_mpr_cache_sha256=str(
            args.expected_factorized_radio_cache_sha256
        ),
    )
    state = build_factorized_primitive_state(
        support,
        chunk_size=int(args.chunk_size),
    )
    output = Path(args.output).resolve()
    write_torch_noclobber(output, state.to_payload())
    record = file_record(output)
    report = {
        "schema_version": state.schema_version,
        "result": str(state.metadata["source"]),
        "contract_sha256": state.contract_sha256,
        "output": record,
        "num_gaussians": int(state.valid.numel()),
        "valid_gaussians": int(state.global_rows.numel()),
        "semantic_direction_dtype": str(state.semantic_direction.dtype),
        "scalar_encoding_dim": int(state.scalar_encoding_input().shape[1]),
        "visibility_purity_known_count": int(
            state.visibility_purity_known.sum()
        ),
        "visibility_purity_unknown_count": int(
            (~state.visibility_purity_known).sum()
        ),
        "field_checkpoint_sha256": support.field_checkpoint_sha256,
        "factorized_radio_cache_sha256": support.cache.sha256,
        "factorized_radio_field_signature_sha256": support.field_signature.digest,
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--expected-field-checkpoint-sha256", required=True)
    parser.add_argument("--factorized-radio-cache", default="")
    parser.add_argument("--expected-factorized-radio-cache-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=4096)
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
