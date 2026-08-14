#!/usr/bin/env python3
"""Run one frozen UQIS v0.2 text tier with CLIP+DINO on one physical GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from radio_gs.benchmarks.scannet_uqis.protocol import (
    BENCHMARK_VERSION_V2_CANDIDATE,
    sha256_file,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _field(root: Path, scene_id: str) -> tuple[Path, str]:
    directory = root / f"{scene_id}_v1"
    manifest = directory / "run_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return directory, sha256_file(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--dino-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--queue-receipt", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--ludvig-upstream", type=Path, required=True)
    parser.add_argument("--driver-library-dir", type=Path, required=True)
    parser.add_argument("--device-index", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    inventory = json.loads((args.input_root / "workspace_inventory.json").read_text(encoding="utf-8"))
    if (
        inventory.get("benchmark_version") != BENCHMARK_VERSION_V2_CANDIDATE
        or inventory.get("status")
        not in {"core_text_workspaces_complete", "relational_text_workspaces_complete"}
        or len(inventory.get("workspaces", [])) not in {31, 36}
    ):
        raise ValueError("v0.2 Core workspace inventory changed")
    jobs = sorted(inventory["workspaces"], key=lambda row: (row["scene_id"], row["query_id"]))
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.prediction_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device_index)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        value for value in (str(args.driver_library_dir.resolve()), environment.get("LD_LIBRARY_PATH", "")) if value
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (
            str(Path(__file__).resolve().parents[2]),
            str(args.ludvig_upstream.resolve() / ".reproduction-deps-sm86"),
            environment.get("PYTHONPATH", ""),
        ) if value
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed_rows = []
    for ordinal, row in enumerate(jobs, start=1):
        query_id, scene_id = str(row["query_id"]), str(row["scene_id"])
        workspace = args.input_root / "workspaces" / "text" / query_id
        output = args.run_root / query_id
        prediction = args.prediction_dir / f"{query_id}.npy"
        if output.is_dir() and (output / "run_manifest.json").is_file() and (output / f"{query_id}.npy").is_file():
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            if (
                manifest.get("benchmark_version") != BENCHMARK_VERSION_V2_CANDIDATE
                or manifest.get("query_id") != query_id
                or manifest.get("diffusion", {}).get("test_fitting") is not False
            ):
                raise ValueError(f"{query_id}: immutable completed output changed")
            if not prediction.is_file():
                shutil.copyfile(output / f"{query_id}.npy", prediction)
            completed_rows.append({"query_id": query_id, "scene_id": scene_id, "status": "already_complete",
                                   "run_manifest_sha256": sha256_file(output / "run_manifest.json"),
                                   "prediction_sha256": sha256_file(prediction)})
            continue
        if output.exists() or prediction.exists():
            raise FileExistsError(f"{query_id}: refusing incomplete immutable state")
        clip, clip_hash = _field(args.clip_root, scene_id)
        dino, dino_hash = _field(args.dino_root, scene_id)
        command = [
            str(args.python.resolve()),
            str(Path(__file__).resolve().parent / "run_uqis_text.py"),
            "--query-manifest", str(workspace / "query_manifest.json"),
            "--workspace-receipt", str(workspace / "workspace_receipt.json"),
            "--field-dir", str(clip),
            "--field-manifest-sha256", clip_hash,
            "--dino-field-dir", str(dino),
            "--dino-field-manifest-sha256", dino_hash,
            "--diffusion-neighbors", "64",
            "--diffusion-iterations", "20",
            "--diffusion-feature-bandwidth", "0.5",
            "--diffusion-regularizer-bandwidth", "2.0",
            "--diffusion-seed-quantile", "0.999",
            "--device", "cuda:0",
            "--ludvig-upstream", str(args.ludvig_upstream.resolve()),
            "--output-dir", str(output),
        ]
        log = args.run_root / "logs" / f"{query_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=environment,
                                     stdout=handle, stderr=subprocess.STDOUT)
        elapsed = time.monotonic() - started
        if process.returncode != 0:
            raise RuntimeError(f"{query_id}: failed; see {log}")
        shutil.copyfile(output / f"{query_id}.npy", prediction)
        runtime = {
            "schema_version": "scannet_uqis_v2_candidate_runtime_receipt_v1",
            "status": "complete",
            "benchmark_version": BENCHMARK_VERSION_V2_CANDIDATE,
            "query_id": query_id,
            "scene_id": scene_id,
            "modality": "text",
            "fresh_process": True,
            "cross_query_state_retained": False,
            "workspace_read_only": True,
            "workspace_receipt_sha256": sha256_file(workspace / "workspace_receipt.json"),
            "physical_gpu_index": args.device_index,
            "elapsed_seconds": elapsed,
            "exit_code": process.returncode,
            "run_manifest_sha256": sha256_file(output / "run_manifest.json"),
            "prediction_sha256": sha256_file(prediction),
            "evaluator_private_inputs_opened": False,
        }
        _write(output / "runtime_receipt.json", runtime)
        completed_rows.append({"query_id": query_id, "scene_id": scene_id, "status": "complete",
                               "run_manifest_sha256": runtime["run_manifest_sha256"],
                               "prediction_sha256": runtime["prediction_sha256"]})
        _write(args.queue_receipt, {
            "schema_version": "scannet_uqis_v2_text_tier_queue_v1",
            "status": "running",
            "benchmark_version": BENCHMARK_VERSION_V2_CANDIDATE,
            "physical_gpu_index": args.device_index,
            "completed": ordinal,
            "total": len(jobs),
            "jobs": completed_rows,
        })
    result = {
        "schema_version": "scannet_uqis_v2_text_tier_queue_v1",
        "status": "complete",
        "benchmark_version": BENCHMARK_VERSION_V2_CANDIDATE,
        "physical_gpu_index": args.device_index,
        "completed": len(completed_rows),
        "total": len(jobs),
        "jobs": completed_rows,
    }
    _write(args.queue_receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
