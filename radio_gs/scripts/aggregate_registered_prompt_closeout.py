#!/usr/bin/env python3
"""Aggregate a frozen registered-prompt closeout without changing predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-plan", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    queue_path = args.queue_plan.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()
    queue = _read_json(queue_path)
    expected = [str(row["scene_id"]) for row in queue["scenes"]]
    protocol_hash = str(queue["protocol_hash"])
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for scene_id in expected:
        path = (
            result_root
            / scene_id
            / "eval_full_mask_random_walker"
            / f"{scene_id}_evaluation.json"
        )
        if not path.is_file():
            missing.append(scene_id)
            continue
        result = _read_json(path)
        if result.get("protocol_hash") != protocol_hash:
            raise ValueError(f"{scene_id}: protocol hash mismatch")
        safety = result.get("safety", {})
        forbidden = {
            "target_ground_truth_opened_before_prediction_write": False,
            "target_rgb_opened": False,
            "target_camera_used_as_support": False,
            "test_calibration": False,
        }
        for key, required in forbidden.items():
            if safety.get(key) is not required:
                raise ValueError(f"{scene_id}: unsafe {key}={safety.get(key)!r}")
        stages = result["stage_metrics"]
        rows.append(
            {
                "scene_id": scene_id,
                "foreground_iou": float(result["foreground_iou"]),
                "pixel_accuracy": float(result["pixel_accuracy"]),
                "unary_iou": float(stages["unary_prior"]["foreground_iou"]),
                "propagated_iou": float(stages["propagated"]["foreground_iou"]),
                "connected_iou": float(stages["connected"]["foreground_iou"]),
                "result": str(path),
                "result_sha256": _sha256(path),
            }
        )

    if missing and not args.allow_incomplete:
        raise RuntimeError(f"missing {len(missing)} scenes: {', '.join(missing)}")
    summary = {
        "schema_version": 1,
        "benchmark": queue.get("benchmark"),
        "protocol_hash": protocol_hash,
        "queue_plan": str(queue_path),
        "queue_plan_sha256": _sha256(queue_path),
        "expected_scene_count": len(expected),
        "completed_scene_count": len(rows),
        "complete": not missing,
        "missing_scenes": missing,
        "macro": {
            "foreground_iou": _mean([row["foreground_iou"] for row in rows]),
            "pixel_accuracy": _mean([row["pixel_accuracy"] for row in rows]),
            "unary_iou": _mean([row["unary_iou"] for row in rows]),
            "propagated_iou": _mean([row["propagated_iou"] for row in rows]),
            "connected_iou": _mean([row["connected_iou"] for row in rows]),
        },
        "scenes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
