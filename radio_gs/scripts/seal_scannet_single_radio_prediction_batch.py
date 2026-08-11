"""Seal eight RGB-free ScanNet semantic score caches before label access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.utils.immutable_artifacts import sha256_file, write_frozen_json


def seal(source_gate: str, prediction_root: str, preregistration: str, output: str) -> dict:
    gate_path = Path(source_gate).resolve(strict=True)
    gate = json.loads(gate_path.read_text())
    if not bool(gate.get("all_source_gates_passed", False)):
        raise ValueError("source gate cohort is not open")
    scenes = gate.get("scenes", [])
    if len(scenes) != 8:
        raise ValueError("prediction batch must bind exactly eight source-gated scenes")
    root = Path(prediction_root).resolve(strict=True)
    records = []
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        score = (root / f"{scene_id}.pt").resolve(strict=True)
        receipt = (root / f"{scene_id}.pt.receipt.json").resolve(strict=True)
        payload = json.loads(receipt.read_text())
        field = payload.get("canonical_field_source", {})
        if field.get("sha256") != scene["checkpoint"]["sha256"]:
            raise ValueError(f"{scene_id} receipt uses a different field")
        if Path(field.get("path", "")).resolve(strict=True) != Path(
            scene["checkpoint"]["path"]
        ).resolve(strict=True):
            raise ValueError(f"{scene_id} receipt field path differs")
        if payload.get("semantic_score_cache", {}).get("sha256") != sha256_file(score):
            raise ValueError(f"{scene_id} score hash differs from receipt")
        if payload.get("status") != "complete_immutable_gaussian_semantic_score_cache":
            raise ValueError(f"{scene_id} receipt is incomplete")
        records.append(
            {
                "scene_id": scene_id,
                "score_cache": {"path": str(score), "sha256": sha256_file(score)},
                "receipt": {"path": str(receipt), "sha256": sha256_file(receipt)},
                "field_checkpoint_sha256": field["sha256"],
            }
        )
    prereg = Path(preregistration).resolve(strict=True)
    result = {
        "schema": "radio_gs.scannet_single_radio_prediction_batch_seal.v1",
        "status": "all_eight_rgb_free_predictions_sealed_before_label_access",
        "source_gate": {"path": str(gate_path), "sha256": sha256_file(gate_path)},
        "prediction_preregistration": {"path": str(prereg), "sha256": sha256_file(prereg)},
        "evaluation_time_source_rgb": False,
        "evaluation_time_target_rgb": False,
        "evaluation_time_external_vision_branch": False,
        "semantic_labels_or_metrics_opened_by_this_sealer": False,
        "scenes": records,
    }
    write_frozen_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gate", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--prediction-preregistration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = seal(args.source_gate, args.prediction_root,
                  args.prediction_preregistration, args.output)
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve()),
                      "sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
