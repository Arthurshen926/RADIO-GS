#!/usr/bin/env python3
"""Seal one complete v0.2 text-tier prediction inventory before scoring it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from radio_gs.benchmarks.scannet_uqis.protocol import (
    BENCHMARK_VERSION_V2_CANDIDATE,
    canonical_json_sha256,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-inventory", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--queue-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    inventory = json.loads(args.workspace_inventory.read_text(encoding="utf-8"))
    queue = json.loads(args.queue_receipt.read_text(encoding="utf-8"))
    rows = inventory.get("workspaces", [])
    if (
        inventory.get("benchmark_version") != BENCHMARK_VERSION_V2_CANDIDATE
        or queue.get("status") != "complete"
        or queue.get("completed") != len(rows)
        or len(rows) not in {31, 36}
    ):
        raise ValueError("text-tier execution is incomplete")
    args.output_dir.mkdir(parents=True)
    snapshot = args.output_dir / "predictions"
    snapshot.mkdir()
    records = []
    for row in sorted(rows, key=lambda value: value["query_id"]):
        query_id = str(row["query_id"])
        source = args.prediction_dir / f"{query_id}.npy"
        array = np.load(source, allow_pickle=False)
        if array.dtype != np.float32 or array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError(f"{query_id}: invalid text-tier prediction")
        output = snapshot / source.name
        shutil.copyfile(source, output)
        records.append(
            {
                "query_id": query_id,
                "scene_id": str(row["scene_id"]),
                "relative_path": f"predictions/{source.name}",
                "sha256": sha256_file(output),
                "bytes": output.stat().st_size,
                "dtype": "float32",
                "shape": [len(array)],
            }
        )
    seal = {
        "schema_version": "scannet_uqis_v2_text_tier_prediction_seal_v1",
        "benchmark_version": BENCHMARK_VERSION_V2_CANDIDATE,
        "status": "sealed_complete",
        "formal_benchmark_eligible": False,
        "query_count": len(records),
        "workspace_inventory_sha256": sha256_file(args.workspace_inventory),
        "queue_receipt_sha256": sha256_file(args.queue_receipt),
        "prediction_inventory_sha256": canonical_json_sha256(records),
        "predictions": records,
    }
    path = args.output_dir / "sealed_prediction_batch.json"
    path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": seal["status"], "query_count": len(records),
                      "seal_sha256": sha256_file(path)}, indent=2))


if __name__ == "__main__":
    main()
