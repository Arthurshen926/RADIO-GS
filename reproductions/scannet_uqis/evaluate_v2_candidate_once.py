#!/usr/bin/env python3
"""One-shot evaluator-controlled scoring of a sealed UQIS v0.2 candidate batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radio_gs.benchmarks.scannet_uqis.evaluate_predictions import evaluate_predictions
from radio_gs.benchmarks.scannet_uqis.protocol import canonical_json_sha256, sha256_file


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--text-profile", type=Path, required=True)
    parser.add_argument("--sealed-batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--recover-failed-ledger",
        action="store_true",
        help="resume scoring only from an identical sealed batch after evaluator code failure",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output_dir / "controlled_evaluation_ledger.json"
    recovery = False
    if ledger_path.exists():
        previous = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not args.recover_failed_ledger or previous.get("status") != "consumed_failed_after_private_authority_claim":
            raise FileExistsError("v0.2 candidate evaluation cohort was already consumed")
        recovery = True

    # Snapshot and verify every method prediction before opening either private
    # target pairing or instance-label arrays.
    seal_path = args.sealed_batch_dir / "sealed_prediction_batch.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    predictions = {}
    legacy_text_ablation = {}
    if seal.get("status") != "sealed_complete_private_evaluator_not_opened":
        raise ValueError("prediction batch is not sealed")
    if recovery and previous.get("sealed_prediction_batch_sha256") != sha256_file(seal_path):
        raise ValueError("recovery seal differs from the originally claimed batch")
    for row in seal["predictions"]:
        path = args.sealed_batch_dir / row["relative_path"]
        if sha256_file(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("sealed prediction changed")
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or list(array.shape) != row["shape"]:
            raise ValueError("sealed prediction dtype/shape changed")
        predictions[str(row["query_id"])] = np.array(array, copy=True)
    for row in seal.get("preregistered_ablations", {}).get(
        "legacy_text_no_diffusion", []
    ):
        path = args.sealed_batch_dir / row["relative_path"]
        if sha256_file(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("sealed ablation prediction changed")
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or list(array.shape) != row["shape"]:
            raise ValueError("sealed ablation dtype/shape changed")
        legacy_text_ablation[str(row["query_id"])] = np.array(array, copy=True)
    if len(legacy_text_ablation) != 31:
        raise ValueError("legacy text ablation must be sealed before private open")

    claim = {
        "schema_version": "scannet_uqis_v2_candidate_controlled_ledger_v1",
        "status": "claimed_before_private_authority_open",
        "formal_benchmark_eligible": False,
        "sealed_prediction_batch_sha256": sha256_file(seal_path),
        "prediction_inventory_sha256": seal["prediction_inventory_sha256"],
        "recovery_after_scoring_code_failure": recovery,
        "method_predictions_changed_after_claim": False,
    }
    _write(ledger_path, claim)
    try:
        evaluator_path = args.source_release / "target_manifest.evaluator.json"
        scene_path = args.source_release / "scene_manifest.json"
        evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        profile = json.loads(args.text_profile.read_text(encoding="utf-8"))
        tiers = {
            (str(row["scene_id"]), int(row["instance_id"])): str(row["evaluation_tier"])
            for row in profile["targets"]
        }
        evaluator = json.loads(json.dumps(evaluator))
        for target in evaluator["targets"]:
            target["evaluation_tier"] = tiers[(str(target["scene_id"]), int(target["instance_id"]))]
        mesh_xyz, mesh_ids = {}, {}
        for domain in evaluator["scene_domains"]:
            scene_id = str(domain["scene_id"])
            xyz_path = Path(domain["mesh_xyz_path"])
            ids_path = Path(domain["mesh_instance_ids_path"])
            if sha256_file(xyz_path) != domain["mesh_xyz_sha256"] or sha256_file(ids_path) != domain["mesh_instance_ids_sha256"]:
                raise ValueError("private mesh authority changed")
            mesh_xyz[scene_id] = np.load(xyz_path, allow_pickle=False)
            mesh_ids[scene_id] = np.load(ids_path, allow_pickle=False)
        report = evaluate_predictions(
            evaluator,
            scene,
            predictions,
            mesh_xyz,
            mesh_ids,
            bootstrap_samples=2000,
            bootstrap_seed=20260813,
            confidence=0.95,
        )
        legacy_predictions = dict(predictions)
        legacy_predictions.update(legacy_text_ablation)
        legacy_report = evaluate_predictions(
            evaluator,
            scene,
            legacy_predictions,
            mesh_xyz,
            mesh_ids,
            bootstrap_samples=2000,
            bootstrap_seed=20260813,
            confidence=0.95,
        )
        graph_text_ap = report["core_modalities"]["text"]["scene_macro"]["average_precision"]
        legacy_text_ap = legacy_report["core_modalities"]["text"]["scene_macro"]["average_precision"]
        report["preregistered_ablation_no_graph_diffusion"] = {
            "uq_rank": legacy_report["uq_rank"],
            "uq_mask": legacy_report["uq_mask"],
            "core_text": legacy_report["core_modalities"]["text"],
            "graph_minus_legacy_core_text_ap": graph_text_ap - legacy_text_ap,
            "graph_minus_legacy_uq_rank": report["uq_rank"]["value"]
            - legacy_report["uq_rank"]["value"],
        }
        report.update(
            {
                "evaluation_mode": "v0.2_candidate_evaluator_controlled_one_shot",
                "formal_benchmark_eligible": False,
                "v0_2_text_profile_sha256": sha256_file(args.text_profile),
                "sealed_prediction_batch_sha256": sha256_file(seal_path),
            }
        )
        report_path = args.output_dir / "result.json"
        _write(report_path, report)
        completed = {
            **claim,
            "status": (
                "consumed_complete_recovered_scoring_code"
                if recovery
                else "consumed_complete"
            ),
            "evaluator_manifest_sha256": sha256_file(evaluator_path),
            "text_profile_sha256": sha256_file(args.text_profile),
            "result_sha256": sha256_file(report_path),
            "result_identity_sha256": canonical_json_sha256(report),
        }
        _write(ledger_path, completed)
        print(json.dumps({"status": completed["status"], "result_sha256": completed["result_sha256"],
                          "uq_rank": report["uq_rank"], "uq_mask": report["uq_mask"]}, indent=2))
    except Exception as error:
        _write(
            ledger_path,
            {
                **claim,
                "status": "consumed_failed_after_private_authority_claim",
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
