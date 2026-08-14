#!/usr/bin/env python3
"""Run remaining UQIS LUDVIG geometry jobs serially on one thermally safe GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.benchmarks.scannet_uqis.ludvig_geometry import (
    run_ludvig_geometry_training,
)
from radio_gs.benchmarks.scannet_uqis.protocol import PREREGISTERED_TEST_SCENES


def _completed(path: Path) -> bool:
    receipt = path / "geometry_run_receipt.json"
    if not receipt.is_file():
        return False
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    return (
        payload.get("status") == "geometry_complete"
        and payload.get("formal_field_eligible") is True
        and payload.get("iterations") == 30000
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ludvig-upstream", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--scene-id",
        action="append",
        choices=PREREGISTERED_TEST_SCENES,
        help="Repeat to select/order scenes; defaults to the frozen cohort order",
    )
    args = parser.parse_args()
    scenes = tuple(args.scene_id or PREREGISTERED_TEST_SCENES)
    if len(set(scenes)) != len(scenes):
        raise ValueError("geometry queue scene IDs must be unique")
    args.output_root.mkdir(parents=True, exist_ok=True)
    queue_path = args.output_root / "geometry_queue_receipt.json"
    jobs = []
    for scene_id in scenes:
        staging = (args.staging_root / scene_id).resolve()
        staging_receipt = json.loads(
            (staging / "geometry_staging_receipt.json").read_text(encoding="utf-8")
        )
        output = (args.output_root / f"{scene_id}_f7a_30k_v1").resolve()
        if _completed(output):
            jobs.append({"scene_id": scene_id, "status": "already_complete"})
        else:
            if output.exists():
                raise FileExistsError(
                    f"refusing incomplete/failed immutable queue output: {output}"
                )
            jobs.append({"scene_id": scene_id, "status": "running"})
            queue_path.write_text(
                json.dumps({"status": "running", "jobs": jobs}, indent=2) + "\n",
                encoding="utf-8",
            )
            result = run_ludvig_geometry_training(
                staging,
                expected_staging_receipt_sha256=staging_receipt["receipt_sha256"],
                ludvig_upstream=args.ludvig_upstream,
                python=args.python,
                output_dir=output,
                iterations=30000,
                device_index=args.device_index,
            )
            jobs[-1] = {
                "scene_id": scene_id,
                "status": "complete",
                "geometry_run_receipt_sha256": result["receipt_sha256"],
            }
        queue_path.write_text(
            json.dumps({"status": "running", "jobs": jobs}, indent=2) + "\n",
            encoding="utf-8",
        )
    payload = {"status": "complete", "scene_count": len(scenes), "jobs": jobs}
    queue_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
