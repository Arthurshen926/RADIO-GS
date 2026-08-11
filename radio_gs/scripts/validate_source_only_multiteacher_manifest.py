"""Validate an immutable query-free Stage-B field-distillation manifest."""

from __future__ import annotations

import argparse
import json

from radio_gs.training.source_only_multiteacher import (
    validate_source_only_multiteacher_manifest,
)
from radio_gs.utils.immutable_artifacts import load_json_object


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    payload, digest, source = load_json_object(
        args.manifest,
        expected_sha256=args.expected_sha256,
        label="Stage-B source-only multiteacher manifest",
    )
    validated = validate_source_only_multiteacher_manifest(payload)
    print(
        json.dumps(
            {
                "status": "validated_source_only_multiteacher_manifest",
                "scene_id": validated["scene_id"],
                "manifest": {"path": str(source), "sha256": digest},
                "field_control": validated["field_control"],
                "primitive_descriptor_teacher": validated[
                    "primitive_descriptor_teacher"
                ],
                "relation_graph": validated["relation_graph"],
                "gpu_started": validated["execution"]["gpu_started"],
                "benchmark_gate": validated["source_gates"]["benchmark_gate"],
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
