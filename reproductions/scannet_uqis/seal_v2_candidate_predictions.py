#!/usr/bin/env python3
"""Seal a complete v0.2 candidate batch before evaluator-private scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from radio_gs.benchmarks.scannet_uqis.protocol import (
    BENCHMARK_VERSION_V2_CANDIDATE,
    QUERY_MANIFEST_NAMES,
    canonical_json_sha256,
    sha256_file,
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--core-workspace-inventory", type=Path, required=True)
    parser.add_argument("--legacy-prediction-dir", type=Path, required=True)
    parser.add_argument("--diffused-text-prediction-dir", type=Path, required=True)
    parser.add_argument("--queue-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    queue = json.loads(args.queue_receipt.read_text(encoding="utf-8"))
    workspaces = json.loads(args.core_workspace_inventory.read_text(encoding="utf-8"))
    if queue.get("status") != "complete" or queue.get("completed") != 31:
        raise ValueError("Core diffusion queue is incomplete")
    core_ids = {str(row["query_id"]) for row in workspaces["workspaces"]}
    if len(core_ids) != 31:
        raise ValueError("Core query inventory changed")
    public_queries = {}
    for modality, name in QUERY_MANIFEST_NAMES.items():
        payload = json.loads((args.source_release / name).read_text(encoding="utf-8"))
        for row in payload["queries"]:
            query_id = str(row["query_id"])
            if query_id in public_queries:
                raise ValueError("query IDs are not globally unique")
            public_queries[query_id] = modality
    args.output_dir.mkdir(parents=True)
    snapshot = args.output_dir / "predictions"
    snapshot.mkdir()
    records = []
    for query_id, modality in sorted(public_queries.items()):
        source_kind = "legacy_sealed_unchanged_query"
        source = args.legacy_prediction_dir / f"{query_id}.npy"
        if modality == "text" and query_id in core_ids:
            source_kind = "v2_core_clip_dino_diffusion"
            source = args.diffused_text_prediction_dir / f"{query_id}.npy"
        array = np.load(source, allow_pickle=False)
        if array.dtype != np.float32 or array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError(f"{query_id}: prediction is not finite float32 [V]")
        destination = snapshot / f"{query_id}.npy"
        shutil.copyfile(source, destination)
        records.append(
            {
                "query_id": query_id,
                "modality": modality,
                "relative_path": f"predictions/{query_id}.npy",
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
                "dtype": "float32",
                "shape": [int(array.shape[0])],
                "source_kind": source_kind,
            }
        )
    ablation_root = args.output_dir / "ablations" / "legacy_text_no_diffusion"
    ablation_root.mkdir(parents=True)
    ablation_records = []
    for query_id in sorted(core_ids):
        source = args.legacy_prediction_dir / f"{query_id}.npy"
        array = np.load(source, allow_pickle=False)
        if array.dtype != np.float32 or array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError(f"{query_id}: legacy text ablation is invalid")
        destination = ablation_root / f"{query_id}.npy"
        shutil.copyfile(source, destination)
        ablation_records.append(
            {
                "query_id": query_id,
                "relative_path": f"ablations/legacy_text_no_diffusion/{query_id}.npy",
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
                "dtype": "float32",
                "shape": [int(array.shape[0])],
            }
        )
    body = {
        "schema_version": "scannet_uqis_v2_candidate_prediction_seal_v1",
        "benchmark_version": BENCHMARK_VERSION_V2_CANDIDATE,
        "status": "sealed_complete_private_evaluator_not_opened",
        "formal_benchmark_eligible": False,
        "row_scope": "universal_complete",
        "query_count": len(records),
        "core_diffused_text_query_count": len(core_ids),
        "legacy_unchanged_query_count": len(records) - len(core_ids),
        "source_release_sha256": sha256_file(args.source_release / "release.json"),
        "core_workspace_inventory_sha256": sha256_file(args.core_workspace_inventory),
        "diffusion_queue_receipt_sha256": sha256_file(args.queue_receipt),
        "evaluator_private_manifest_opened": False,
        "predictions": records,
        "preregistered_ablations": {
            "legacy_text_no_diffusion": ablation_records,
        },
    }
    seal = {
        **body,
        "prediction_inventory_sha256": canonical_json_sha256(records),
        "ablation_inventory_sha256": canonical_json_sha256(ablation_records),
    }
    seal_path = args.output_dir / "sealed_prediction_batch.json"
    _write(seal_path, seal)
    print(json.dumps({"status": seal["status"], "query_count": len(records),
                      "seal_sha256": sha256_file(seal_path)}, indent=2))


if __name__ == "__main__":
    main()
