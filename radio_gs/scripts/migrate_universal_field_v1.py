#!/usr/bin/env python3
"""Migrate one frozen D512/L512 field to Universal Field v1 schema-v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.field.checkpoint import (
    _canonical_field_from_payload,
    load_factorized_canonical_field_checkpoint,
    load_universal_canonical_field_checkpoint,
)
from radio_gs.field.factorized_radio_contract import (
    parse_factorized_radio_payload,
    validate_factorized_radio_checkpoint_metadata,
)
from radio_gs.universal_field_v1 import migrate_universal_field_payload
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


@torch.inference_mode()
def migrate(args: argparse.Namespace) -> dict[str, object]:
    source_field, source_payload, _ = load_factorized_canonical_field_checkpoint(
        args.source_field,
        map_location="cpu",
        expected_sha256=args.expected_source_sha256,
    )
    factorized_path = Path(
        args.factorized_cache or str(source_payload.get("mpr_cache", ""))
    ).expanduser()
    factorized_payload, factorized_sha256, factorized_path = load_torch_mapping(
        factorized_path,
        expected_sha256=args.expected_factorized_sha256
        or str(source_payload.get("mpr_cache_sha256", "")),
        map_location="cpu",
        label="factorized RADIO cache",
    )
    if not isinstance(factorized_payload, dict):
        raise TypeError("factorized RADIO cache is not a mapping")
    factorized_rows_payload = factorized_payload.get("factorized_radio")
    if not isinstance(factorized_rows_payload, dict):
        raise TypeError("factorized RADIO builder cache lacks factorized rows")
    rows = parse_factorized_radio_payload(factorized_rows_payload)
    if rows.reliability.shape[0] != source_field.num_gaussians:
        raise ValueError("factorized cache and field row counts differ")
    migrated = migrate_universal_field_payload(
        source_payload,
        factorized_rows_payload,
        source_field_sha256=args.expected_source_sha256,
        factorized_cache_sha256=factorized_sha256,
    )

    signature = validate_factorized_radio_checkpoint_metadata(
        migrated["factorized_radio_metadata"]
    )
    migrated_field = _canonical_field_from_payload(
        migrated,
        map_location="cpu",
        signature=signature.base_feature_signature,
    )
    count = min(int(args.decode_probe_rows), int(source_field.num_gaussians))
    indices = (
        torch.linspace(
            0,
            int(source_field.num_gaussians) - 1,
            steps=count,
            dtype=torch.float64,
        )
        .round()
        .long()
        .unique()
    )
    source_coefficients = source_field.coefficients(indices)
    migrated_coefficients = migrated_field.coefficients(indices)
    source_radio = source_field.radio_features(indices)
    migrated_radio = migrated_field.radio_features(indices)
    if not torch.equal(source_coefficients, migrated_coefficients):
        raise RuntimeError("Universal Field migration changed coefficient decode")
    if not torch.equal(source_radio, migrated_radio):
        raise RuntimeError("Universal Field migration changed RADIO decode")

    output = Path(args.output).expanduser().resolve()
    write_torch_noclobber(output, migrated)
    output_record = file_record(output)
    loaded_field, loaded_payload, _ = load_universal_canonical_field_checkpoint(
        output,
        map_location="cpu",
        expected_sha256=output_record["sha256"],
    )
    loaded_radio = loaded_field.radio_features(indices)
    if not torch.equal(source_radio, loaded_radio):
        raise RuntimeError("serialized Universal Field changed RADIO decode")

    source_record = file_record(args.source_field)
    reliability_bytes = int(rows.reliability.numel() * rows.reliability.element_size())
    storage_delta_bytes = int(
        output.stat().st_size - Path(args.source_field).stat().st_size
    )
    if storage_delta_bytes > reliability_bytes + 2 * 1024 * 1024:
        raise RuntimeError("Universal Field reliability storage was duplicated")
    report = {
        "schema_version": 1,
        "artifact_type": "radio_gs_universal_field_v1_migration_report",
        "status": "complete",
        "source_field": source_record,
        "universal_field": output_record,
        "factorized_cache": {
            "path": str(factorized_path),
            "sha256": factorized_sha256,
        },
        "num_gaussians": int(source_field.num_gaussians),
        "reliability_dim": int(rows.reliability.shape[1]),
        "reliability_bytes": reliability_bytes,
        "source_field_bytes": int(Path(args.source_field).stat().st_size),
        "universal_field_bytes": int(output.stat().st_size),
        "storage_delta_bytes": storage_delta_bytes,
        "storage_overhead_bytes": storage_delta_bytes - reliability_bytes,
        "reliability_serialized_once": True,
        "decode_probe_rows": int(indices.numel()),
        "coefficient_decode_bitwise_equal": True,
        "radio_decode_bitwise_equal": True,
        "serialized_radio_decode_bitwise_equal": True,
        "field_training_rerun": False,
        "benchmark_data_opened": False,
        "migration": dict(loaded_payload["universal_field_migration"]),
    }
    report_path = output.with_suffix(output.suffix + ".json")
    write_frozen_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-field", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--factorized-cache", default="")
    parser.add_argument("--expected-factorized-sha256", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--decode-probe-rows", type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(migrate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
