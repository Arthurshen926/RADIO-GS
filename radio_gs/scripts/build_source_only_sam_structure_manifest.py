"""Preregister one source-only official-SAM relation training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.training.source_only_sam_structure import (
    SOURCE_ONLY_SAM_STRUCTURE_CONTRACT_SHA256,
    source_only_sam_structure_contract,
    validate_source_only_sam_structure_manifest,
)
from radio_gs.utils.immutable_artifacts import (
    sha256_file,
    write_frozen_json,
)


def _record(path: str, expected: str) -> dict[str, str]:
    source = Path(path).resolve()
    if sha256_file(source) != str(expected):
        raise ValueError(f"asset SHA-256 differs: {source}")
    return {"path": str(source), "sha256": str(expected)}


def build(args: argparse.Namespace) -> dict:
    contract = source_only_sam_structure_contract()
    payload = {
        "schema": contract["schema"],
        "schema_version": contract["schema_version"],
        "contract": contract,
        "contract_sha256": SOURCE_ONLY_SAM_STRUCTURE_CONTRACT_SHA256,
        "status": "preregistered_training_not_started",
        "scene_id": str(args.scene_id),
        "field_control": _record(args.field_control, args.expected_field_control_sha256),
        "canonical_radio_cache": _record(
            args.canonical_radio_cache,
            args.expected_canonical_radio_cache_sha256,
        ),
        "relation_cache": _record(args.relation_cache, args.expected_relation_cache_sha256),
        "relation_graph": _record(args.relation_graph, args.expected_relation_graph_sha256),
        "official_sam_build_authority": _record(
            args.official_sam_build_authority,
            args.expected_official_sam_build_authority_sha256,
        ),
        "loss": contract["loss"],
        "source_gates": {
            "radio_reconstruction_no_regression": True,
            "gauge_no_regression": True,
            "sam_same_cosine_non_decrease": True,
            "sam_separate_cosine_non_increase": True,
            "sam_relation_gap_strict_improvement": True,
            "six_task_benchmark_gate": (
                "closed_until_all_source_gates_pass_then_frozen_one_shot"
            ),
        },
        "access_audit": {
            "source_rgb_opened_during_mapping": True,
            "official_sam_opened_during_mapping": True,
            "query_time_source_rgb_opened": False,
            "query_time_target_rgb_opened": False,
            "benchmark_query_or_text_opened": False,
            "benchmark_target_or_evaluation_rgb_opened": False,
            "benchmark_ground_truth_opened": False,
            "benchmark_labels_or_masks_opened": False,
            "benchmark_metrics_or_predictions_opened": False,
        },
        "execution": {
            "gpu_started": False,
            "per_scene_or_per_task_tuning": False,
            "output_no_clobber": True,
            "teacher_payload_saved_in_checkpoint": False,
        },
    }
    return validate_source_only_sam_structure_manifest(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    for name in (
        "field-control",
        "canonical-radio-cache",
        "relation-cache",
        "relation-graph",
        "official-sam-build-authority",
    ):
        parser.add_argument(f"--{name}", required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build(args)
    write_frozen_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": "preregistered_source_only_sam_radio_structure",
                "scene_id": payload["scene_id"],
                "output": str(Path(args.output).resolve()),
                "sha256": sha256_file(args.output),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
