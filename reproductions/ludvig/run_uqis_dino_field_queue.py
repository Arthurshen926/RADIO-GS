#!/usr/bin/env python3
"""Follow GPU1 geometries with both UQIS LUDVIG fields on GPU0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from radio_gs.benchmarks.scannet_uqis.protocol import canonical_json_sha256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_hash(root: Path) -> str:
    manifest = root / "run_manifest.json"
    declared = root / "run_manifest.sha256"
    if not manifest.is_file() or not declared.is_file():
        raise ValueError(f"incomplete immutable field output: {root}")
    digest = _sha256(manifest)
    if declared.read_text(encoding="ascii").strip() != digest:
        raise ValueError(f"field manifest digest changed: {root}")
    return digest


def _run(command: list[str], log: Path, environment: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(f"field stage failed; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--field-root", type=Path, required=True)
    parser.add_argument("--observation-receipt", type=Path, required=True)
    parser.add_argument("--observation-receipt-sha256", required=True)
    parser.add_argument("--construction-authority", type=Path, required=True)
    parser.add_argument("--construction-authority-sha256", required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--open-clip-site-packages", type=Path, required=True)
    parser.add_argument("--open-clip-checkpoint", type=Path, required=True)
    parser.add_argument("--ludvig-upstream", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--scene-id", action="append", required=True)
    args = parser.parse_args()
    if len(set(args.scene_id)) != len(args.scene_id):
        raise ValueError("scene IDs must be unique")
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        raise ValueError("poll interval must be in (0,60]")

    root = args.field_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    queue_receipt = root / "dino_field_queue_receipt.json"
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = "/root/baselines/LUDVIG/.driver535"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(Path(__file__).resolve().parents[2]), str(args.ludvig_upstream.resolve() / ".reproduction-deps-sm86"))
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    jobs: list[dict[str, str]] = []
    for scene_id in args.scene_id:
        geometry_root = args.geometry_root.resolve() / f"{scene_id}_f7a_30k_v1"
        geometry_receipt = geometry_root / "geometry_run_receipt.json"
        while not geometry_receipt.is_file():
            queue_receipt.write_text(
                json.dumps({"status": "waiting_for_geometry", "scene_id": scene_id, "jobs": jobs}, indent=2) + "\n",
                encoding="utf-8",
            )
            time.sleep(args.poll_seconds)
        geometry = json.loads(geometry_receipt.read_text(encoding="utf-8"))
        body = {key: value for key, value in geometry.items() if key != "receipt_sha256"}
        if (
            geometry.get("status") != "geometry_complete"
            or geometry.get("formal_field_eligible") is not True
            or geometry.get("receipt_sha256")
            != canonical_json_sha256(body)
        ):
            raise ValueError(f"geometry is not formally eligible: {scene_id}")

        bridge = root / "dino_phase_a_bridge" / f"{scene_id}_v1"
        phase_b = root / "dino_phase_b" / f"{scene_id}_v1"
        uplift = root / "dino_uplift" / f"{scene_id}_v1"
        clip_field = root / "clip_field" / f"{scene_id}_v1"
        logs = root / "logs" / scene_id
        if not bridge.exists():
            _run([
                str(args.python), "reproductions/ludvig/stage_uqis_dino_field.py",
                "--observation-receipt", str(args.observation_receipt.resolve()),
                "--expected-observation-receipt-sha256", args.observation_receipt_sha256,
                "--construction-authority", str(args.construction_authority.resolve()),
                "--expected-construction-authority-sha256", args.construction_authority_sha256,
                "--geometry-run-receipt", str(geometry_receipt.resolve()),
                "--expected-geometry-run-receipt-sha256", str(geometry["receipt_sha256"]),
                "--dino-checkpoint", str(args.dino_checkpoint.resolve()),
                "--ludvig-upstream", str(args.ludvig_upstream.resolve()),
                "--output-dir", str(bridge),
            ], logs / "phase_a_bridge.log", environment)
        phase_a_hash = _manifest_hash(bridge)
        ledger = bridge / "source_adapter_ledger.json"
        ledger_hash = _sha256(ledger)
        if not phase_b.exists():
            _run([
                str(args.python), "reproductions/ludvig/run_uqis_dino_phase_b.py",
                "--phase-a-dir", str(bridge),
                "--phase-a-manifest-sha256", phase_a_hash,
                "--source-adapter-ledger", str(ledger),
                "--source-adapter-ledger-sha256", ledger_hash,
                "--dino-checkpoint", str(args.dino_checkpoint.resolve()),
                "--ludvig-upstream", str(args.ludvig_upstream.resolve()),
                "--dinov2-source", str(args.ludvig_upstream.resolve()),
                "--output-dir", str(phase_b),
                "--device", args.device,
                "--views", "120",
            ], logs / "phase_b.log", environment)
        phase_b_hash = _manifest_hash(phase_b)
        if not uplift.exists():
            _run([
                str(args.python), "reproductions/ludvig/run_uqis_dino_uplift.py",
                "--phase-b-dir", str(phase_b),
                "--phase-b-manifest-sha256", phase_b_hash,
                "--ludvig-upstream", str(args.ludvig_upstream.resolve()),
                "--output-dir", str(uplift),
                "--device", args.device,
            ], logs / "uplift.log", environment)
        uplift_hash = _manifest_hash(uplift)
        if not clip_field.exists():
            _run([
                str(args.python), "reproductions/ludvig/run_uqis_clip_field.py",
                "--phase-a-dir", str(bridge),
                "--phase-a-manifest-sha256", phase_a_hash,
                "--ludvig-upstream", str(args.ludvig_upstream.resolve()),
                "--open-clip-site-packages", str(args.open_clip_site_packages.resolve()),
                "--open-clip-checkpoint", str(args.open_clip_checkpoint.resolve()),
                "--output-dir", str(clip_field),
                "--device", args.device,
            ], logs / "clip_field.log", environment)
        clip_hash = _manifest_hash(clip_field)
        jobs.append({
            "scene_id": scene_id,
            "geometry_receipt_sha256": str(geometry["receipt_sha256"]),
            "phase_a_manifest_sha256": phase_a_hash,
            "phase_b_manifest_sha256": phase_b_hash,
            "uplift_manifest_sha256": uplift_hash,
            "clip_field_manifest_sha256": clip_hash,
        })
        queue_receipt.write_text(
            json.dumps({"status": "running", "jobs": jobs}, indent=2) + "\n",
            encoding="utf-8",
        )
    queue_receipt.write_text(
        json.dumps({"status": "complete", "jobs": jobs}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
