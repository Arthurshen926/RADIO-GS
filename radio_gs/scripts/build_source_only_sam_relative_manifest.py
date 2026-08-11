"""Preregister scale-matched no-harm official-SAM supervision for one RADIO field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.training.source_only_sam_relative_structure import (
    SOURCE_ONLY_SAM_RELATIVE_CONTRACT_SHA256,
    source_only_sam_relative_contract,
    validate_source_only_sam_relative_manifest,
)
from radio_gs.utils.immutable_artifacts import sha256_file, write_frozen_json


def build(args: argparse.Namespace) -> dict:
    base = Path(args.base_structure_manifest).resolve()
    if sha256_file(base) != str(args.expected_base_structure_manifest_sha256):
        raise ValueError("base source-only SAM structure manifest SHA-256 differs")
    contract = source_only_sam_relative_contract()
    payload = {
        "schema": contract["schema"],
        "schema_version": contract["schema_version"],
        "contract": contract,
        "contract_sha256": SOURCE_ONLY_SAM_RELATIVE_CONTRACT_SHA256,
        "status": "preregistered_training_not_started",
        "scene_id": str(args.scene_id),
        "base_structure_manifest": {
            "path": str(base),
            "sha256": str(args.expected_base_structure_manifest_sha256),
        },
        "loss": contract["loss"],
        "source_gates": {
            "radio_reconstruction_no_regression": {
                "mean_cosine_max_regression": 0.005,
                "p05_cosine_max_regression": 0.01,
            },
            "official_capability_no_regression": {
                "mean_cosine_max_regression": 0.005,
                "p05_cosine_max_regression": 0.01,
            },
            "global_same_cosine_non_decrease": True,
            "global_separate_cosine_non_increase": True,
            "global_relation_gap_strict_improvement": True,
            "scale_triplet_gap_strict_improvement": True,
            "scale_triplet_violation_strict_decrease": True,
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
            "v1_candidate_checkpoint_used": False,
        },
    }
    return validate_source_only_sam_relative_manifest(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--base-structure-manifest", required=True)
    parser.add_argument("--expected-base-structure-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build(args)
    write_frozen_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": "preregistered_source_only_sam_relative_structure",
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
