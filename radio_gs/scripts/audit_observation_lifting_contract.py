#!/usr/bin/env python3
"""Audit canonical-field checkpoints against one shared MPR contract."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from radio_gs.field.observation_lifting_contract import (
    canonical_observation_contract,
    validate_observation_contract_metadata,
)


def audit(paths: list[str]) -> dict:
    rows: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = torch.load(path, map_location="cpu")
        metadata = dict(payload.get("mpr_cache_metadata", {}))
        declared = "observation_lifting_contract" in metadata
        status = "compatible_legacy"
        error = ""
        try:
            validate_observation_contract_metadata(
                metadata, require_declaration=declared
            )
            if declared:
                status = "declared_canonical_v1"
        except ValueError as exc:
            status = "incompatible"
            error = str(exc)
        rows.append(
            {
                "field_checkpoint": str(path.resolve()),
                "mpr_cache": str(payload.get("mpr_cache", "")),
                "status": status,
                "error": error,
                "observed_policy": {
                    key: metadata.get(key)
                    for key in (
                        "num_declared_views",
                        "aggregation_mode",
                        "registration_weight_mode",
                        "raster_view_fusion",
                        "normalize_each_view",
                        "depth_tolerance",
                        "relative_depth_tolerance",
                        "alpha_threshold",
                    )
                },
            }
        )
        del payload
        gc.collect()
    return {
        "schema_version": 1,
        "audit": "canonical_observation_lifting_contract",
        "expected_contract": canonical_observation_contract(),
        "fields": rows,
        "summary": {
            "total": len(rows),
            "declared_canonical_v1": sum(
                row["status"] == "declared_canonical_v1" for row in rows
            ),
            "compatible_legacy": sum(
                row["status"] == "compatible_legacy" for row in rows
            ),
            "incompatible": sum(row["status"] == "incompatible" for row in rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(args.field_checkpoints)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
