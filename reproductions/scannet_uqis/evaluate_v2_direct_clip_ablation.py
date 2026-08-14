#!/usr/bin/env python3
"""Score the post-hoc deterministic same-expression direct-CLIP ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radio_gs.benchmarks.scannet_uqis.evaluate_predictions import evaluate_predictions
from radio_gs.benchmarks.scannet_uqis.protocol import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--text-profile", type=Path, required=True)
    parser.add_argument("--sealed-batch-dir", type=Path, required=True)
    parser.add_argument("--direct-clip-dir", type=Path, required=True)
    parser.add_argument("--graph-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    seal_path = args.sealed_batch_dir / "sealed_prediction_batch.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    predictions = {}
    for row in seal["predictions"]:
        path = args.sealed_batch_dir / row["relative_path"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError("sealed system prediction changed")
        predictions[str(row["query_id"])] = np.load(path, allow_pickle=False)
    direct_manifest_path = args.direct_clip_dir / "run_manifest.json"
    direct = json.loads(direct_manifest_path.read_text(encoding="utf-8"))
    for row in direct["predictions"]:
        path = args.direct_clip_dir / f"{row['query_id']}.npy"
        if sha256_file(path) != row["prediction_sha256"]:
            raise ValueError("direct CLIP derivation changed")
        predictions[str(row["query_id"])] = np.load(path, allow_pickle=False)

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
        evaluator,
        scene,
        predictions,
        xyz,
        ids,
        bootstrap_samples=2000,
        bootstrap_seed=20260813,
        confidence=0.95,
    )
    graph = json.loads(args.graph_result.read_text(encoding="utf-8"))
    direct_text_ap = report["core_modalities"]["text"]["scene_macro"]["average_precision"]
    graph_text_ap = graph["core_modalities"]["text"]["scene_macro"]["average_precision"]
    payload = {
        "schema_version": "scannet_uqis_v2_same_expression_direct_clip_ablation_v1",
        "status": "posthoc_deterministic_correction_complete",
        "formal_benchmark_eligible": False,
        "interpretation_boundary": (
            "same frozen v0.2 expressions and saved pre-diffusion CLIP primitives; "
            "derived after evaluator claim because the preregistered legacy comparator "
            "was discovered to use mismatched v0.1 expressions"
        ),
        "parameters_changed_after_evaluator_open": False,
        "sealed_graph_result_sha256": sha256_file(args.graph_result),
        "direct_clip_derivation_sha256": sha256_file(direct_manifest_path),
        "direct_clip_uq_rank": report["uq_rank"],
        "direct_clip_uq_mask": report["uq_mask"],
        "direct_clip_core_text": report["core_modalities"]["text"],
        "graph_core_text_ap": graph_text_ap,
        "graph_minus_direct_core_text_ap": graph_text_ap - direct_text_ap,
        "graph_minus_direct_uq_rank": graph["uq_rank"]["value"] - report["uq_rank"]["value"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"direct_text_ap": direct_text_ap, "graph_text_ap": graph_text_ap,
                      "delta_text_ap": payload["graph_minus_direct_core_text_ap"],
                      "direct_uq_rank": report["uq_rank"]["value"],
                      "graph_uq_rank": graph["uq_rank"]["value"]}, indent=2))


if __name__ == "__main__":
    main()
