#!/usr/bin/env python3
"""Gate prompt-proposal SAM3 video memory with the frozen NVOS authority.

The pre-existing reliability rule is reused unchanged: authorize a transient
region when the target official-SAM proposal score is at least 0.5 or its
overlap with the registered field prior is below 0.5.  Otherwise retain the
sealed target-only Method-v1 prediction.  No target annotation is opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from radio_gs.scripts.materialize_nvos_ludvig_region_reliability_gate import (
    _sha256,
    _write_json,
    _write_numpy,
    authorize_region_agreement,
)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    base_manifest_path = Path(args.base_manifest).resolve(strict=True)
    if _sha256(base_manifest_path) != args.expected_base_manifest_sha256:
        raise ValueError("target-only manifest hash differs")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    scenes = [str(value) for value in base_manifest["scene_order"]]
    if len(scenes) != 8 or len(set(scenes)) != 8:
        raise ValueError("NVOS target-only cohort differs")
    receipt_paths = {
        str(row["scene_id"]): Path(row["path"]).resolve(strict=True)
        for row in base_manifest["receipts"]
    }
    base_root = Path(str(base_manifest.get("prediction_root", ".")))
    if not base_root.is_absolute():
        base_root = base_manifest_path.parent / base_root
    video_root = Path(args.video_root).resolve(strict=True)
    output = Path(args.output_dir).resolve()
    predictions: dict[str, dict[str, str]] = {}
    hashes: dict[str, dict[str, str]] = {}
    records: list[dict[str, Any]] = []
    for scene in scenes:
        frame, raw_base_path = next(iter(base_manifest["predictions"][scene].items()))
        base_path = Path(raw_base_path)
        if not base_path.is_absolute():
            base_path = base_root / base_path
        base_path = base_path.resolve(strict=True)
        if _sha256(base_path) != base_manifest["prediction_sha256"][scene][frame]:
            raise ValueError(f"{scene}: target-only prediction hash differs")
        base_receipt_path = receipt_paths[scene]
        base_receipt = json.loads(base_receipt_path.read_text(encoding="utf-8"))
        proposal = base_receipt["sam3"]
        authorized = authorize_region_agreement(
            proposal["selected_score"], proposal["best_initial_overlap"]
        )
        video_receipt_path = (video_root / scene / "prediction_receipt.json").resolve(strict=True)
        video_receipt = json.loads(video_receipt_path.read_text(encoding="utf-8"))
        if (
            video_receipt.get("scene_id") != scene
            or video_receipt.get("target_frame_id") != frame
            or video_receipt.get("target_mask_opened") is not False
            or video_receipt.get("target_metric_opened") is not False
            or video_receipt.get("prompt_seed")
            != "official_signed_scribble_selected_sam3_proposal_box_plus_points"
        ):
            raise ValueError(f"{scene}: video receipt authority differs")
        video_record = video_receipt["prediction"]
        video_path = Path(video_record["path"]).resolve(strict=True)
        if _sha256(video_path) != video_record["sha256"]:
            raise ValueError(f"{scene}: video prediction hash differs")
        base_margin = np.load(base_path, allow_pickle=False)
        if authorized:
            video = np.load(video_path, allow_pickle=False) >= 0.5
            video = np.asarray(
                Image.fromarray(video.astype(np.uint8)).resize(
                    (base_margin.shape[1], base_margin.shape[0]),
                    Image.Resampling.NEAREST,
                )
            ).astype(bool)
            candidate = np.where(video, 0.5, -0.5).astype(np.float32)
            branch = "prompt_proposal_video_memory"
        else:
            candidate = np.asarray(base_margin, dtype=np.float32)
            branch = "target_only_fallback"
        target = output / "scores" / scene / f"{frame}.npy"
        digest = _write_numpy(target, candidate)
        predictions[scene] = {frame: str(target)}
        hashes[scene] = {frame: digest}
        records.append(
            {
                "scene_id": scene,
                "frame_id": frame,
                "branch": branch,
                "region_memory_authorized": authorized,
                "official_sam_selected_score": float(proposal["selected_score"]),
                "best_initial_overlap": float(proposal["best_initial_overlap"]),
                "base_prediction": {"path": str(base_path), "sha256": _sha256(base_path)},
                "video_prediction": {"path": str(video_path), "sha256": _sha256(video_path)},
                "video_receipt": {"path": str(video_receipt_path), "sha256": _sha256(video_receipt_path)},
                "output": {"path": str(target), "sha256": digest},
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "nvos_sam3_prompt_proposal_video_reliability_gate_v1",
        "scene_order": scenes,
        "predictions": predictions,
        "prediction_sha256": hashes,
        "records": records,
        "rule": "target_sam_score>=0.5 OR target_sam_field_overlap<0.5",
        "all_eight_predictions_sealed": True,
        "target_mask_opened": False,
        "target_metric_opened": False,
        "development_only": True,
    }
    manifest_path = output / "prediction_manifest.json"
    _write_json(manifest_path, payload)
    return {**payload, "manifest": str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--expected-base-manifest-sha256", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    print(json.dumps(materialize(parser.parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
