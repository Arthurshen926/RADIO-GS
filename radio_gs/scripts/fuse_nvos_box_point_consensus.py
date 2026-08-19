#!/usr/bin/env python3
"""Fuse sealed NVOS SAM box extent with independent point-SAM consensus.

The box proposal is the primary object extent.  Only a fixed supermajority
(at least seven of ten point-prompt trials) may add pixels outside that extent.
No RGB, target mask, or evaluation metric is opened by this stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from radio_gs.scripts.predict_nvos_method_v1_field_box_sam3 import (
    _write_json,
    _write_numpy,
)
from radio_gs.scripts.predict_nvos_method_v1_transient_sam import _sha256


CANDIDATE_ID = "nvos-method-v1-box-plus-point-supermajority-v1"


def fuse_box_and_point_consensus(
    box_margin: np.ndarray,
    point_vote_margin: np.ndarray,
    *,
    point_margin_threshold: float = 0.2,
) -> np.ndarray:
    """Return binary-margin union with a fixed point-SAM supermajority gate."""

    box = np.asarray(box_margin, dtype=np.float32)
    point = np.asarray(point_vote_margin, dtype=np.float32)
    if box.shape != point.shape or box.ndim != 2:
        raise ValueError("box and point margins must be aligned [H,W] arrays")
    if not bool(np.isfinite(box).all()) or not bool(np.isfinite(point).all()):
        raise ValueError("box and point margins must be finite")
    # point margin = mean(binary trials) - 0.5; threshold 0.2 therefore means
    # at least 70% independent prompt trials agree on foreground.
    fused = (box >= 0.0) | (point >= float(point_margin_threshold))
    return np.where(fused, 0.5, -0.5).astype(np.float32)


def _load_manifest(path: str | Path, expected_sha256: str) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve(strict=True)
    if len(expected_sha256) != 64 or _sha256(source) != expected_sha256:
        raise ValueError(f"prediction manifest SHA-256 differs: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("prediction manifest must be a JSON object")
    return value, source


def _resolve_prediction(
    manifest: Mapping[str, Any], manifest_path: Path, scene: str, frame: str
) -> Path:
    relative = Path(str(manifest["predictions"][scene][frame]))
    root = Path(str(manifest.get("prediction_root", ".")))
    if not root.is_absolute():
        root = manifest_path.parent / root
    path = relative if relative.is_absolute() else root / relative
    path = path.resolve(strict=True)
    if _sha256(path) != str(manifest["prediction_sha256"][scene][frame]):
        raise ValueError(f"prediction SHA-256 differs: {scene}/{frame}")
    return path


def fuse(args: argparse.Namespace) -> dict[str, Any]:
    box, box_path = _load_manifest(args.box_manifest, args.expected_box_manifest_sha256)
    point, point_path = _load_manifest(
        args.point_manifest, args.expected_point_manifest_sha256
    )
    scene_order = [str(value) for value in box.get("scene_order", [])]
    if not scene_order:
        scene_order = [str(value) for value in box.get("predictions", {})]
    if (
        box.get("kind") != "promptable_nvs_method_v1_field_box_sam3_predictions"
        or point.get("kind") != "promptable_nvs_method_v1_transient_sam_predictions"
        or box.get("protocol_hash") != point.get("protocol_hash")
        or scene_order != [str(value) for value in point.get("predictions", {})]
        or len(scene_order) != 8
        or bool(box.get("target_mask_opened", True))
        or bool(point.get("target_mask_opened", True))
        or bool(box.get("target_metric_opened", True))
        or bool(point.get("target_metric_opened", True))
    ):
        raise ValueError("sealed NVOS box/point prediction contracts differ")
    output_root = Path(args.output_dir).expanduser().resolve()
    predictions: dict[str, dict[str, str]] = {}
    hashes: dict[str, dict[str, str]] = {}
    receipts: list[dict[str, str]] = []
    for scene in scene_order:
        frames = list(box["predictions"][scene])
        if len(frames) != 1 or frames != list(point["predictions"][scene]):
            raise ValueError(f"NVOS target frame differs: {scene}")
        frame = str(frames[0])
        box_score = _resolve_prediction(box, box_path, scene, frame)
        point_score = _resolve_prediction(point, point_path, scene, frame)
        fused = fuse_box_and_point_consensus(
            np.load(box_score, allow_pickle=False),
            np.load(point_score, allow_pickle=False),
            point_margin_threshold=args.point_margin_threshold,
        )
        relative = Path("scores") / scene / f"{frame}.npy"
        output = output_root / relative
        digest = _write_numpy(output, fused)
        receipt = {
            "schema_version": 1,
            "artifact_type": "radio_gs_nvos_box_point_consensus_receipt",
            "candidate_id": CANDIDATE_ID,
            "scene_id": scene,
            "frame_id": frame,
            "box_prediction": {"path": str(box_score), "sha256": _sha256(box_score)},
            "point_prediction": {
                "path": str(point_score),
                "sha256": _sha256(point_score),
            },
            "point_margin_threshold": float(args.point_margin_threshold),
            "minimum_trial_agreement": float(args.point_margin_threshold + 0.5),
            "output": {"path": str(output), "sha256": digest},
            "safety": {
                "target_rgb_opened_by_fusion": False,
                "target_mask_opened": False,
                "target_metric_opened": False,
                "selection_or_threshold_fit_on_target_metric": False,
            },
        }
        receipt_path = output_root / "receipts" / f"{scene}.json"
        _write_json(receipt_path, receipt)
        predictions[scene] = {frame: relative.as_posix()}
        hashes[scene] = {frame: digest}
        receipts.append(
            {"scene_id": scene, "path": str(receipt_path), "sha256": _sha256(receipt_path)}
        )
    manifest = {
        "schema_version": 1,
        "kind": "promptable_nvs_method_v1_box_point_consensus_predictions",
        "candidate_id": CANDIDATE_ID,
        "protocol_hash": box["protocol_hash"],
        "scene_order": scene_order,
        "prediction_root": ".",
        "predictions": predictions,
        "prediction_sha256": hashes,
        "receipts": receipts,
        "parent_manifests": {
            "box": {"path": str(box_path), "sha256": args.expected_box_manifest_sha256},
            "point": {
                "path": str(point_path),
                "sha256": args.expected_point_manifest_sha256,
            },
        },
        "method": {
            "primary_extent": "signed-evidence-selected official SAM box proposal",
            "complement": "independent point-SAM trial supermajority",
            "point_margin_threshold": float(args.point_margin_threshold),
            "minimum_trial_agreement": float(args.point_margin_threshold + 0.5),
        },
        "all_eight_predictions_sealed": True,
        "evaluation_performed": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
    }
    manifest_path = output_root / "prediction_manifest.json"
    _write_json(manifest_path, manifest)
    return {**manifest, "prediction_manifest": str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-manifest", required=True)
    parser.add_argument("--expected-box-manifest-sha256", required=True)
    parser.add_argument("--point-manifest", required=True)
    parser.add_argument("--expected-point-manifest-sha256", required=True)
    parser.add_argument("--point-margin-threshold", type=float, default=0.2)
    parser.add_argument("--output-dir", required=True)
    report = fuse(parser.parse_args(argv))
    print(json.dumps({"prediction_manifest": report["prediction_manifest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
