#!/usr/bin/env python3
"""Run one UQIS modality as isolated, resumable LUDVIG query processes.

This is a trusted workspace issuer plus method-process queue.  The issuer
audits the release before staging, while every method child receives only a
fresh read-only one-query workspace and the already constructed method field.
No evaluator manifest or ScanNet instance labels are passed to a child.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from radio_gs.benchmarks.scannet_uqis.protocol import (
    BENCHMARK_VERSION,
    QUERY_MANIFEST_NAMES,
    QueryModality,
    sha256_file,
)
from radio_gs.benchmarks.scannet_uqis.workspace import stage_query_workspace


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _make_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _completed(output: Path, query_id: str) -> bool:
    manifest_path = output / "run_manifest.json"
    prediction_path = output / f"{query_id}.npy"
    if not manifest_path.is_file() or not prediction_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    manifest_query_ids = []
    if manifest.get("query_id") is not None:
        manifest_query_ids.append(str(manifest["query_id"]))
    if isinstance(manifest.get("queries"), list):
        manifest_query_ids.extend(
            str(row.get("query_id"))
            for row in manifest["queries"]
            if isinstance(row, dict)
        )
    return (
        manifest.get("benchmark_version") == BENCHMARK_VERSION
        and manifest_query_ids == [query_id]
        and manifest.get("status") == "exact_runtime_smoke_complete"
        and manifest.get("privacy_boundary", {}).get("evaluator_manifest_opened") is False
    )


def _field_manifest(field_root: Path, scene_id: str) -> tuple[Path, str]:
    directory = field_root / f"{scene_id}_v1"
    manifest = directory / "run_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return directory, sha256_file(manifest)


def _command(
    args: argparse.Namespace,
    *,
    modality: QueryModality,
    scene_id: str,
    workspace: Path,
    output: Path,
) -> list[str]:
    common = [
        str(args.python.resolve()),
        str((Path(__file__).resolve().parent / f"run_uqis_{modality.value.replace('point_', 'point')}.py")),
        "--query-manifest", str(workspace / "query_manifest.json"),
        "--workspace-receipt", str(workspace / "workspace_receipt.json"),
    ]
    if modality is QueryModality.IMAGE:
        phase_b, phase_b_hash = _field_manifest(args.phase_b_root, scene_id)
        uplift, uplift_hash = _field_manifest(args.dino_root, scene_id)
        return common + [
            "--phase-b-dir", str(phase_b),
            "--phase-b-manifest-sha256", phase_b_hash,
            "--phase-c-dir", str(uplift),
            "--phase-c-manifest-sha256", uplift_hash,
            "--ludvig-upstream", str(args.ludvig_upstream.resolve()),
            "--driver-library-dir", str(args.driver_library_dir.resolve()),
            "--device", "cuda:0",
            "--output-dir", str(output),
        ]
    field_root = args.clip_root if modality is QueryModality.TEXT else args.dino_root
    field, field_hash = _field_manifest(field_root, scene_id)
    return common + [
        "--field-dir", str(field),
        "--field-manifest-sha256", field_hash,
        "--ludvig-upstream", str(args.ludvig_upstream.resolve()),
        "--output-dir", str(output),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--modality", choices=[row.value for row in QueryModality], required=True)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--dino-root", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--queue-receipt", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--ludvig-upstream", type=Path, required=True)
    parser.add_argument("--driver-library-dir", type=Path, required=True)
    parser.add_argument("--device-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    modality = QueryModality(args.modality)
    manifest_path = args.benchmark_dir.resolve() / QUERY_MANIFEST_NAMES[modality.value]
    public = json.loads(manifest_path.read_text(encoding="utf-8"))
    if public.get("benchmark_version") != BENCHMARK_VERSION or public.get("modality") != modality.value:
        raise ValueError("public query manifest identity changed")
    source_queries = public.get("queries")
    if not isinstance(source_queries, list) or not source_queries:
        raise ValueError("public query inventory is empty")
    # Public order is deliberately opaque and interleaves scenes.  Execution
    # order is not benchmark semantics; grouping by public scene_id avoids
    # repeatedly evicting multi-gigabyte scene fields while each query still
    # runs in a fresh workspace/process with no retained method state.
    queries = sorted(
        source_queries,
        key=lambda row: (str(row["scene_id"]), str(row["query_id"])),
    )
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    queries = [
        row for index, row in enumerate(queries)
        if index % args.shard_count == args.shard_index
    ]

    args.workspace_root.mkdir(parents=True, exist_ok=True)
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.prediction_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
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

    for index, query in enumerate(queries):
        query_id = str(query["query_id"])
        scene_id = str(query["scene_id"])
        workspace = args.workspace_root / modality.value / query_id
        output = args.run_root / modality.value / query_id
        prediction = args.prediction_dir / f"{query_id}.npy"
        if _completed(output, query_id):
            if not prediction.is_file():
                shutil.copyfile(output / f"{query_id}.npy", prediction)
            runtime_path = output / "runtime_receipt.json"
            if not runtime_path.is_file():
                _write_json(runtime_path, {
                    "schema_version": "scannet_uqis_query_runtime_receipt_v1",
                    "status": "recovered_after_child_success",
                    "benchmark_version": BENCHMARK_VERSION,
                    "query_id": query_id,
                    "scene_id": scene_id,
                    "modality": modality.value,
                    "fresh_process": True,
                    "cross_query_state_retained": False,
                    "workspace_read_only": True,
                    "workspace_receipt_sha256": sha256_file(workspace / "workspace_receipt.json"),
                    "command": _command(
                        args, modality=modality, scene_id=scene_id,
                        workspace=workspace, output=output,
                    ),
                    "physical_gpu_index": args.device_index,
                    "elapsed_seconds": None,
                    "exit_code": 0,
                    "run_manifest_sha256": sha256_file(output / "run_manifest.json"),
                    "prediction_sha256": sha256_file(prediction),
                    "evaluator_private_inputs_opened": False,
                })
            jobs.append({
                "query_id": query_id,
                "scene_id": scene_id,
                "status": "already_complete",
                "runtime_receipt_sha256": sha256_file(runtime_path),
            })
            continue
        if workspace.exists() or output.exists():
            raise FileExistsError(f"refusing incomplete immutable query state: {query_id}")
        receipt = stage_query_workspace(
            args.benchmark_dir,
            modality=modality,
            query_id=query_id,
            workspace_dir=workspace,
        )
        _make_read_only(workspace)
        command = _command(
            args, modality=modality, scene_id=scene_id, workspace=workspace, output=output
        )
        log_path = args.run_root / "logs" / modality.value / f"{query_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        elapsed = time.monotonic() - started
        if completed.returncode or not _completed(output, query_id):
            raise RuntimeError(f"{query_id}: query failed; see {log_path}")
        shutil.copyfile(output / f"{query_id}.npy", prediction)
        runtime = {
            "schema_version": "scannet_uqis_query_runtime_receipt_v1",
            "status": "complete",
            "benchmark_version": BENCHMARK_VERSION,
            "query_id": query_id,
            "scene_id": scene_id,
            "modality": modality.value,
            "fresh_process": True,
            "cross_query_state_retained": False,
            "workspace_read_only": True,
            "workspace_receipt_sha256": sha256_file(workspace / "workspace_receipt.json"),
            "command": command,
            "physical_gpu_index": args.device_index,
            "elapsed_seconds": elapsed,
            "exit_code": completed.returncode,
            "run_manifest_sha256": sha256_file(output / "run_manifest.json"),
            "prediction_sha256": sha256_file(prediction),
            "evaluator_private_inputs_opened": False,
        }
        _write_json(output / "runtime_receipt.json", runtime)
        jobs.append({
            "query_id": query_id,
            "scene_id": scene_id,
            "status": "complete",
            "runtime_receipt_sha256": sha256_file(output / "runtime_receipt.json"),
        })
        _write_json(args.queue_receipt, {
            "schema_version": "scannet_uqis_ludvig_query_queue_v1",
            "status": "running",
            "benchmark_version": BENCHMARK_VERSION,
            "modality": modality.value,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "completed": index + 1,
            "total": len(queries),
            "jobs": jobs,
        })
    final = {
        "schema_version": "scannet_uqis_ludvig_query_queue_v1",
        "status": "complete",
        "benchmark_version": BENCHMARK_VERSION,
        "modality": modality.value,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "completed": len(queries),
        "total": len(queries),
        "jobs": jobs,
    }
    _write_json(args.queue_receipt, final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
