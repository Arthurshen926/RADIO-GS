#!/usr/bin/env python3
"""Materialize exact source-calibrated query relevance for V2.1 regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radio_gs.interfaces.surface_region_v21_query_relevance import (
    QUERY_RELEVANCE_CONTRACT_SHA256,
    QUERY_RELEVANCE_SCHEMA,
    query_relevance_access_audit,
    query_relevance_channel_sha256,
    query_relevance_contract,
    validate_query_execution_authority,
    validate_query_relevance_authority,
)
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    load_frozen_canonical_negative_bank,
)
from radio_gs.querying.v21_absolute_relevance_adapter import (
    calibrated_v21_absolute_relevance,
    load_v21_positive_text_bank,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    write_torch_noclobber,
)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to clobber V2.1 query relevance: {output}")
    execution = validate_query_execution_authority(
        args.query_execution_authority,
        expected_sha256=args.expected_query_execution_authority_sha256,
        expected_output=output,
    )
    descriptor = execution["verified_descriptor"]
    positive = load_v21_positive_text_bank(
        execution["positive_text_cache"]["path"],
        expected_file_sha256=execution["positive_text_cache"]["sha256"],
    )
    negative = load_frozen_canonical_negative_bank(
        execution["canonical_negative_bank"]["path"],
        expected_file_sha256=execution["canonical_negative_bank"]["sha256"],
    )
    if negative.file_sha256 != execution["verified_promoted_negative"]["sha256"]:
        raise ValueError("runtime canonical-negative bank differs from source training")
    relevance = calibrated_v21_absolute_relevance(
        descriptor["semantic_descriptor"],
        positive_bank=positive,
        canonical_negative_bank=negative,
    )
    payload: dict[str, Any] = {
        "schema": QUERY_RELEVANCE_SCHEMA,
        "schema_version": 1,
        "contract": query_relevance_contract(),
        "contract_sha256": QUERY_RELEVANCE_CONTRACT_SHA256,
        "scene_id": descriptor["scene_id"],
        "physical_space_id": descriptor["physical_space_id"],
        "producer": file_record(Path(__file__).resolve()),
        "query_execution_authority": dict(execution["verified_record"]),
        "input_authority": {
            "target_descriptor": dict(execution["target_descriptor"]),
            "positive_text_cache": dict(execution["positive_text_cache"]),
            "canonical_negative_bank": dict(execution["canonical_negative_bank"]),
        },
        "region_row_ids": list(descriptor["region_row_ids"]),
        "canonical_region_indices": descriptor["canonical_region_indices"].clone(),
        "region_fingerprints": list(descriptor["region_fingerprints"]),
        "query_ids": list(positive.query_ids),
        "region_absolute_relevance": relevance,
        "access_audit": query_relevance_access_audit(),
    }
    payload["channel_sha256"] = query_relevance_channel_sha256(payload)
    payload = validate_query_relevance_authority(payload)
    write_torch_noclobber(output, payload)
    return {
        "status": "materialized_source_calibrated_v21_query_relevance",
        "scene_id": payload["scene_id"],
        "regions": len(payload["region_row_ids"]),
        "queries": len(payload["query_ids"]),
        "output": file_record(output),
        "contract_sha256": QUERY_RELEVANCE_CONTRACT_SHA256,
        "access_audit": query_relevance_access_audit(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-execution-authority", required=True)
    parser.add_argument("--expected-query-execution-authority-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    print(json.dumps(materialize(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
