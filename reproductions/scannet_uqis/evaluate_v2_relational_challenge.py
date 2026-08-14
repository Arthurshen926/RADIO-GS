#!/usr/bin/env python3
"""Evaluate the sealed v0.2 relational text challenge against the frozen Core system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radio_gs.benchmarks.scannet_uqis.evaluate_predictions import evaluate_predictions
from radio_gs.benchmarks.scannet_uqis.protocol import sha256_file


def _load_seal(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    seal_path = root / "sealed_prediction_batch.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    arrays = {}
    for row in seal["predictions"]:
        path = root / row["relative_path"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError("sealed prediction changed")
        arrays[str(row["query_id"])] = np.load(path, allow_pickle=False)
    return seal, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--text-profile", type=Path, required=True)
    parser.add_argument("--complete-core-seal", type=Path, required=True)
    parser.add_argument("--relational-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    core_seal, predictions = _load_seal(args.complete_core_seal)
    relational_seal, relational = _load_seal(args.relational_seal)
    if len(relational) != 36:
        raise ValueError("relational seal must contain exactly 36 predictions")
    predictions.update(relational)
    evaluator = json.loads(
        (args.source_release / "target_manifest.evaluator.json").read_text(encoding="utf-8")
    )
    scene = json.loads((args.source_release / "scene_manifest.json").read_text(encoding="utf-8"))
    profile = json.loads(args.text_profile.read_text(encoding="utf-8"))
    tiers = {
        (str(row["scene_id"]), int(row["instance_id"])): row["evaluation_tier"]
        for row in profile["targets"]
    }
    for target in evaluator["targets"]:
        target["evaluation_tier"] = tiers[(str(target["scene_id"]), int(target["instance_id"]))]
    xyz, ids = {}, {}
    for domain in evaluator["scene_domains"]:
        scene_id = str(domain["scene_id"])
        xyz[scene_id] = np.load(domain["mesh_xyz_path"], allow_pickle=False)
        ids[scene_id] = np.load(domain["mesh_instance_ids_path"], allow_pickle=False)
    report = evaluate_predictions(
        evaluator, scene, predictions, xyz, ids,
        bootstrap_samples=2000, bootstrap_seed=20260813, confidence=0.95,
    )
    payload = {
        "schema_version": "scannet_uqis_v2_relational_challenge_result_v1",
        "status": "complete_nonformal",
        "formal_benchmark_eligible": False,
        "evaluation_timing": (
            "predictions sealed before this challenge score; evaluator labels had already "
            "been opened by the earlier Core candidate evaluation"
        ),
        "complete_core_seal_sha256": sha256_file(args.complete_core_seal / "sealed_prediction_batch.json"),
        "relational_seal_sha256": sha256_file(args.relational_seal / "sealed_prediction_batch.json"),
        "relational_text_challenge": report["relational_text_challenge"],
        "core_uq_rank_consistency_check": report["uq_rank"],
        "core_uq_mask_consistency_check": report["uq_mask"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["relational_text_challenge"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
