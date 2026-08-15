"""Validate one checkpoint against the frozen five-benchmark Method-v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.five_benchmark_method_v1 import (
    validate_complete_field_payload,
    validate_method_authority,
)
from radio_gs.utils.immutable_artifacts import load_json_object, load_torch_mapping


DEFAULT_AUTHORITY = (
    Path(__file__).resolve().parents[2]
    / "paper/artifacts/five_benchmark_method_v1_authority_20260815.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True)
    parser.add_argument("--expected-field-sha256", default=None)
    parser.add_argument("--authority", default=str(DEFAULT_AUTHORITY))
    args = parser.parse_args()

    authority, authority_sha256, authority_path = load_json_object(
        args.authority,
        label="five-benchmark Method-v1 authority",
    )
    validate_method_authority(authority)
    payload, field_sha256, field_path = load_torch_mapping(
        args.field,
        expected_sha256=args.expected_field_sha256,
        map_location="cpu",
        label="five-benchmark Method-v1 field",
    )
    report = validate_complete_field_payload(payload)
    report.update(
        {
            "status": "pass",
            "authority": str(authority_path),
            "authority_sha256": authority_sha256,
            "field": str(field_path),
            "field_sha256": field_sha256,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
