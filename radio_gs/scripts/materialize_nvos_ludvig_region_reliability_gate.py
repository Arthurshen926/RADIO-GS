#!/usr/bin/env python3
"""Materialize the frozen NVOS Method-v1/LUDVIG reliability gate.

This is a query-transient image-space bridge.  It consumes sealed Method-v1
binary margins, audited LUDVIG fixed-threshold selectors, and proposal
diagnostics that were emitted before target scoring.  It never reads target
masks or metrics and never transfers rows between Gaussian carriers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from radio_gs.evaluation.promptable_segmentation import resize_mask_nearest


CANDIDATE_ID = "nvos-method-v1-ludvig-region-reliability-gate-v1"
QUALITY_THRESHOLD = 0.5
OVERLAP_THRESHOLD = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authorize_region_agreement(
    official_sam_selected_score: float,
    best_initial_overlap: float,
) -> bool:
    """Use the region veto under strong 2D authority or field disagreement."""

    quality = float(official_sam_selected_score)
    overlap = float(best_initial_overlap)
    if not np.isfinite(quality) or not np.isfinite(overlap):
        raise ValueError("proposal reliability diagnostics must be finite")
    if not 0.0 <= quality <= 1.0 or not 0.0 <= overlap <= 1.0:
        raise ValueError("proposal reliability diagnostics must lie in [0,1]")
    return quality >= QUALITY_THRESHOLD or overlap < OVERLAP_THRESHOLD


def _write_numpy(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.method_v1_manifest).resolve(strict=True)
    inventory_path = Path(args.ludvig_inventory).resolve(strict=True)
    if _sha256(manifest_path) != args.expected_method_v1_manifest_sha256:
        raise ValueError("Method-v1 prediction manifest SHA-256 differs")
    if _sha256(inventory_path) != args.expected_ludvig_inventory_sha256:
        raise ValueError("LUDVIG asset inventory SHA-256 differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    scene_order = [str(value) for value in manifest["scene_order"]]
    requested = scene_order if args.scenes == "all" else args.scenes.split(",")
    if not requested or len(requested) != len(set(requested)) or any(
        scene not in scene_order for scene in requested
    ):
        raise ValueError("requested scene cohort differs")
    prediction_root = Path(str(manifest.get("prediction_root", ".")))
    if not prediction_root.is_absolute():
        prediction_root = manifest_path.parent / prediction_root
    receipts = {
        str(record["scene_id"]): Path(str(record["path"])).resolve(strict=True)
        for record in manifest["receipts"]
    }
    output = Path(args.output_dir).resolve()
    predictions: dict[str, dict[str, str]] = {}
    hashes: dict[str, dict[str, str]] = {}
    records: list[dict[str, Any]] = []
    for scene in requested:
        frames = manifest["predictions"][scene]
        if len(frames) != 1:
            raise ValueError(f"{scene} must have exactly one target prediction")
        frame, relative = next(iter(frames.items()))
        current_path = Path(str(relative))
        if not current_path.is_absolute():
            current_path = prediction_root / current_path
        current_path = current_path.resolve(strict=True)
        expected = manifest["prediction_sha256"][scene][frame]
        if _sha256(current_path) != expected:
            raise ValueError(f"{scene} Method-v1 prediction SHA-256 differs")
        receipt = json.loads(receipts[scene].read_text(encoding="utf-8"))
        proposal = receipt["sam3"]
        selected_score = float(proposal["selected_score"])
        initial_overlap = float(proposal["best_initial_overlap"])
        authorized = authorize_region_agreement(selected_score, initial_overlap)
        margin = np.load(current_path, allow_pickle=False)
        if margin.ndim != 2 or not np.isfinite(margin).all():
            raise ValueError(f"{scene} Method-v1 margin is invalid")
        current = margin >= 0.0
        selector = inventory["scenes"][scene]["runs"][str(args.ludvig_seed)][
            "rendered_binary_selector"
        ]
        ludvig_path = Path(selector["path"]).resolve(strict=True)
        if _sha256(ludvig_path) != selector["sha256"]:
            raise ValueError(f"{scene} LUDVIG selector SHA-256 differs")
        ludvig = resize_mask_nearest(np.asarray(Image.open(ludvig_path)) > 0, current.shape)
        candidate = np.logical_and(current, ludvig) if authorized else current
        candidate_margin = np.where(candidate, 0.5, -0.5).astype(np.float32)
        target = output / "scores" / scene / f"{frame}.npy"
        digest = _write_numpy(target, candidate_margin)
        predictions[scene] = {frame: str(target)}
        hashes[scene] = {frame: digest}
        records.append(
            {
                "scene_id": scene,
                "frame_id": frame,
                "official_sam_selected_score": selected_score,
                "best_initial_overlap": initial_overlap,
                "region_agreement_authorized": authorized,
                "branch": "foreground_intersection" if authorized else "method_v1_unchanged",
                "method_v1_prediction": {"path": str(current_path), "sha256": expected},
                "ludvig_selector": {"path": str(ludvig_path), "sha256": selector["sha256"]},
                "output": {"path": str(target), "sha256": digest},
            }
        )
    result = {
        "schema_version": 1,
        "artifact_type": "nvos_ludvig_region_reliability_gate_prediction_batch",
        "candidate_id": CANDIDATE_ID,
        "scene_order": requested,
        "predictions": predictions,
        "prediction_sha256": hashes,
        "records": records,
        "rule": {
            "official_sam_quality_threshold": QUALITY_THRESHOLD,
            "field_proposal_overlap_threshold": OVERLAP_THRESHOLD,
            "operator": "quality>=0.5 OR overlap<0.5",
        },
        "target_mask_opened": False,
        "target_metric_opened": False,
        "nearest_neighbor_carrier_transfer": False,
        "carrier_modified": False,
    }
    manifest_output = output / "prediction_manifest.json"
    _write_json(manifest_output, result)
    return {**result, "manifest": str(manifest_output)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-v1-manifest", required=True)
    parser.add_argument("--expected-method-v1-manifest-sha256", required=True)
    parser.add_argument("--ludvig-inventory", required=True)
    parser.add_argument("--expected-ludvig-inventory-sha256", required=True)
    parser.add_argument("--ludvig-seed", type=int, default=0)
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--output-dir", required=True)
    report = materialize(parser.parse_args(argv))
    print(json.dumps({"manifest": report["manifest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
