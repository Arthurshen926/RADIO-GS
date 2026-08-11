"""Validate an immutable mapping-time official-SAM RADIO structure manifest."""

from __future__ import annotations

import argparse
import json

from radio_gs.training.source_only_sam_structure import (
    validate_source_only_sam_structure_manifest,
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
        label="source-only official-SAM structure manifest",
    )
    validated = validate_source_only_sam_structure_manifest(payload)
    print(
        json.dumps(
            {
                "status": "validated_source_only_sam_radio_structure_manifest",
                "scene_id": validated["scene_id"],
                "manifest": {"path": str(source), "sha256": digest},
                "persistent_semantic_feature": validated["contract"][
                    "persistent_semantic_feature"
                ],
                "mapping_teacher": validated["contract"]["mapping_teacher"],
                "query_time": validated["contract"]["query_time"],
                "gpu_started": validated["execution"]["gpu_started"],
                "benchmark_gate": validated["source_gates"][
                    "six_task_benchmark_gate"
                ],
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
