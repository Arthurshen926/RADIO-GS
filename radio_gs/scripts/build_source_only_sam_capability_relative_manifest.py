#!/usr/bin/env python3
"""Seal a v3 official-SAM capability-relative source-training manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from radio_gs.training.source_only_sam_capability_relative_structure import (
    SOURCE_ONLY_SAM_CAPABILITY_RELATIVE_CONTRACT_SHA256,
    source_only_sam_capability_relative_contract,
    validate_source_only_sam_capability_relative_manifest,
)
from radio_gs.utils.immutable_artifacts import sha256_file, write_frozen_json


def build_manifest(
    *,
    scene_id: str,
    base_relative_manifest: str | Path,
    official_adaptor_checkpoint: str | Path,
) -> dict:
    base = Path(base_relative_manifest).expanduser().resolve(strict=True)
    checkpoint = Path(official_adaptor_checkpoint).expanduser().resolve(strict=True)
    contract = source_only_sam_capability_relative_contract()
    payload = {
        "schema": contract["schema"],
        "schema_version": contract["schema_version"],
        "contract": contract,
        "contract_sha256": SOURCE_ONLY_SAM_CAPABILITY_RELATIVE_CONTRACT_SHA256,
        "status": "preregistered_training_not_started",
        "scene_id": str(scene_id),
        "base_relative_manifest": {
            "path": str(base),
            "sha256": sha256_file(base),
        },
        "official_adaptor_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
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
            "global_capability_same_relation_non_decrease": True,
            "global_capability_separate_relation_non_increase": True,
            "global_capability_relation_gap_strict_improvement": True,
            "capability_triplet_gap_strict_improvement": True,
            "capability_triplet_violation_strict_decrease": True,
            "six_task_benchmark_gate": (
                "closed_after_source_gate_checkpoint_seal_no_benchmark_in_this_stage"
            ),
        },
        "access_audit": {
            "source_rgb_opened_during_mapping": True,
            "official_sam_opened_during_mapping": True,
            "official_dino_and_sam_adaptors_opened_during_training": True,
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
            "v2_candidate_checkpoint_used": False,
        },
    }
    return validate_source_only_sam_capability_relative_manifest(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--base-relative-manifest", required=True)
    parser.add_argument("--official-adaptor-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_manifest(
        scene_id=args.scene_id,
        base_relative_manifest=args.base_relative_manifest,
        official_adaptor_checkpoint=args.official_adaptor_checkpoint,
    )
    output = Path(args.output).expanduser().resolve()
    write_frozen_json(output, payload)
    print(f"{output}\t{sha256_file(output)}")


if __name__ == "__main__":
    main()
