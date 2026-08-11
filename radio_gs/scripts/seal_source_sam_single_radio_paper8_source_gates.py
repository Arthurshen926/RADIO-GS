"""Seal the eight-scene source-only SAM-to-RADIO gate cohort.

This utility consumes only training reports and checkpoint bytes.  It never
opens ScanNet semantic labels, text queries, predictions, or task metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.utils.immutable_artifacts import sha256_file, write_frozen_json


def _metric_delta(report: dict, final_key: str, control_key: str, name: str) -> float:
    return float(report[final_key][name]) - float(report[control_key][name])


def seal(candidate_specs: list[str], preregistration: str, output: str) -> dict:
    prereg = Path(preregistration).resolve(strict=True)
    if len(candidate_specs) != 8:
        raise ValueError("paper8 source gate seal requires exactly eight candidates")

    records = []
    seen = set()
    for spec in candidate_specs:
        scene_id, separator, raw_path = spec.partition("=")
        if not separator or not scene_id or scene_id in seen:
            raise ValueError("candidate must be a unique scene_id=/absolute/path pair")
        seen.add(scene_id)
        checkpoint = Path(raw_path).resolve(strict=True)
        report_path = Path(str(checkpoint) + ".json").resolve(strict=True)
        report = json.loads(report_path.read_text())
        gate = dict(report.get("source_sam_relative_gate_decision", {}))
        source = dict(report.get("source_only_sam_relative_structure", {}))
        if not bool(gate.get("all_source_gates_passed", False)):
            raise ValueError(f"{scene_id} did not pass every source gate")
        if source.get("persistent_semantic_feature") != "canonical_radio_only":
            raise ValueError(f"{scene_id} is not a single-RADIO field")
        if bool(source.get("teacher_payload_saved", True)):
            raise ValueError(f"{scene_id} checkpoint retains teacher payload")
        if bool(source.get("query_time_source_rgb", True)) or bool(
            source.get("query_time_target_rgb", True)
        ):
            raise ValueError(f"{scene_id} enables query-time RGB")
        records.append(
            {
                "scene_id": scene_id,
                "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
                "training_report": {
                    "path": str(report_path),
                    "sha256": sha256_file(report_path),
                },
                "source_manifest": source.get("manifest", {}),
                "source_gates": gate,
                "deltas": {
                    "raw_mean_cosine": float(report["final_metrics"]["mean_cosine"])
                    - float(report["initial_field_checkpoint"]["source_final_metrics"]["mean_cosine"]),
                    "raw_p05_cosine": float(report["final_metrics"]["p05_cosine"])
                    - float(report["initial_field_checkpoint"]["source_final_metrics"]["p05_cosine"]),
                    "dino_mean_cosine": float(
                        report["final_capability_metrics"]["dino_v3_target_mean_cosine"]
                    )
                    - float(
                        report["initial_field_checkpoint"]["source_final_capability_metrics"]
                        ["dino_v3_target_mean_cosine"]
                    ),
                    "sam_mean_cosine": float(
                        report["final_capability_metrics"]["sam3_target_mean_cosine"]
                    )
                    - float(
                        report["initial_field_checkpoint"]["source_final_capability_metrics"]
                        ["sam3_target_mean_cosine"]
                    ),
                    "global_relation_gap": _metric_delta(
                        report,
                        "final_source_sam_relative_pair_metrics",
                        "control_source_sam_relative_pair_metrics",
                        "sam_relation_gap",
                    ),
                    "scale_triplet_gap": _metric_delta(
                        report,
                        "final_source_sam_relative_metrics",
                        "control_source_sam_relative_metrics",
                        "sam_relative_gap",
                    ),
                    "scale_triplet_violation_rate": _metric_delta(
                        report,
                        "final_source_sam_relative_metrics",
                        "control_source_sam_relative_metrics",
                        "sam_relative_violation_rate",
                    ),
                },
            }
        )

    records.sort(key=lambda item: item["scene_id"])
    payload = {
        "schema": "radio_gs.source_sam_single_radio_paper8_source_gate_seal.v1",
        "status": "all_eight_source_gates_passed_benchmark_gate_opened",
        "training_preregistration": {
            "path": str(prereg),
            "sha256": sha256_file(prereg),
        },
        "persistent_semantic_feature": "canonical_radio_only",
        "mapping_supervision": "legal_source_rgb_official_sam_query_free",
        "evaluation_contract": {
            "source_rgb": False,
            "target_rgb": False,
            "sam_or_other_external_vision_branch": False,
            "benchmark_labels_or_metrics_opened_by_this_sealer": False,
        },
        "scenes": records,
        "all_source_gates_passed": True,
        "next_gate": "seal_all_eight_rgb_free_predictions_before_first_semantic_label_open",
    }
    write_frozen_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--training-preregistration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = seal(args.candidate, args.training_preregistration, args.output)
    print(json.dumps({"status": payload["status"], "output": str(Path(args.output).resolve()),
                      "sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
