#!/usr/bin/env python3
"""Apply the frozen NVOS identity-supported extent rule to sealed SPIn scenes.

This is a cross-benchmark confirmation producer, not a parameter search.  It
uses the already-authorized 0.05 support fraction and opens no SPIn target
mask or metric.  The output can only remove disconnected SAM extent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from radio_gs.evaluation.promptable_segmentation import resize_mask_nearest
from radio_gs.querying.transient_rgb_sam import FROZEN_POLICY, deterministic_signed_point_trials
from radio_gs.scripts.build_nvos_identity_component_threshold_authority import (
    ARTIFACT_TYPE as THRESHOLD_AUTHORITY_TYPE,
)
from radio_gs.scripts.filter_nvos_sam3_components_by_identity_support import (
    MINIMUM_COARSE_SUPPORT_FRACTION,
    identity_supported_components,
    identity_supported_components_local_density,
)
from radio_gs.scripts.predict_nvos_method_v1_field_box_sam3 import _write_json, _write_numpy
from radio_gs.scripts.predict_nvos_method_v1_transient_sam import _sha256


CANDIDATE_ID = "spin9-method-v1-identity-supported-sam3-components-v1"
LOCAL_DENSITY_CANDIDATE_ID = (
    "spin9-method-v1-component-local-density-sam3-components-v1"
)


def _load(path: str | Path, expected_sha: str, label: str) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve(strict=True)
    if len(expected_sha) != 64 or _sha256(source) != expected_sha:
        raise ValueError(f"{label} SHA-256 differs")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, source


def _prediction_path(manifest: Mapping[str, Any], source: Path, scene: str, frame: str) -> Path:
    root = Path(str(manifest.get("prediction_root", ".")))
    if not root.is_absolute():
        root = source.parent / root
    path = (root / str(manifest["predictions"][scene][frame])).resolve(strict=True)
    if _sha256(path) != str(manifest["prediction_sha256"][scene][frame]):
        raise ValueError(f"sealed prediction differs: {scene}/{frame}")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    extent, extent_path = _load(args.extent_manifest, args.expected_extent_manifest_sha256, "SPIn extent manifest")
    authority, authority_path = _load(args.threshold_authority, args.expected_threshold_authority_sha256, "threshold authority")
    threshold = float(args.minimum_coarse_support_fraction)
    scenes = [str(value) for value in extent.get("scene_order", [])]
    if (
        extent.get("kind") != "promptable_nvs_method_v1_spin9_transient_sam_predictions"
        or not scenes
        or extent.get("evaluation_performed") is not False
        or extent.get("target_mask_opened") is not False
        or extent.get("target_metric_opened") is not False
        or authority.get("artifact_type") != THRESHOLD_AUTHORITY_TYPE
        or authority.get("target_mask_opened") is not False
        or authority.get("target_metric_opened") is not False
        or float(authority.get("selected_threshold", -1)) != threshold
    ):
        raise ValueError("sealed SPIn extent or frozen threshold authority differs")

    receipt_index = {str(row["scene_id"]): row for row in extent.get("receipts", [])}
    output = Path(args.output_dir).expanduser().resolve()
    predictions: dict[str, dict[str, str]] = {}
    hashes: dict[str, dict[str, str]] = {}
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        receipt_row = receipt_index.get(scene)
        if receipt_row is None:
            raise ValueError(f"SPIn extent receipt absent: {scene}")
        receipt_path = Path(str(receipt_row["path"])).resolve(strict=True)
        if _sha256(receipt_path) != str(receipt_row["sha256"]):
            raise ValueError(f"SPIn extent receipt differs: {scene}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        signed_receipt_path = Path(str(receipt["signed_field_receipt"])).resolve(strict=True)
        if _sha256(signed_receipt_path) != str(receipt["signed_field_receipt_sha256"]):
            raise ValueError(f"SPIn signed-field receipt differs: {scene}")
        signed_receipt = json.loads(signed_receipt_path.read_text(encoding="utf-8"))
        signed_rows = {str(row["frame_id"]): row for row in signed_receipt["target_scores"]}
        predictions[scene] = {}
        hashes[scene] = {}
        for frame, relative_source in extent["predictions"][scene].items():
            frame = str(frame)
            extent_score = np.load(_prediction_path(extent, extent_path, scene, frame), allow_pickle=False)
            extent_mask = extent_score >= 0
            signed_row = signed_rows[frame]
            signed_path = Path(str(signed_row["path"])).resolve(strict=True)
            if _sha256(signed_path) != str(signed_row["sha256"]):
                raise ValueError(f"SPIn signed margin differs: {scene}/{frame}")
            signed_margin = np.load(signed_path, allow_pickle=False).astype(np.float32, copy=False)
            coarse = resize_mask_nearest(signed_margin >= 0, extent_mask.shape).astype(bool)
            points, labels = deterministic_signed_point_trials(
                np.maximum(signed_margin, 0), np.maximum(-signed_margin, 0),
                image_shape=extent_mask.shape, policy=FROZEN_POLICY,
            )
            if args.support_denominator == "component_area":
                filtered = identity_supported_components_local_density(
                    extent_mask, coarse, points[labels == 1],
                    minimum_local_identity_density=threshold,
                )
            else:
                filtered = identity_supported_components(
                    extent_mask, coarse, points[labels == 1],
                    minimum_coarse_support_fraction=threshold,
                )
            relative = Path("scores") / scene / f"{frame}.npy"
            digest = _write_numpy(output / relative, np.where(filtered, 0.5, -0.5).astype(np.float32))
            predictions[scene][frame] = relative.as_posix()
            hashes[scene][frame] = digest
            rows.append({
                "scene_id": scene, "frame_id": frame,
                "input_components": int(cv2.connectedComponents(extent_mask.astype(np.uint8), 8)[0] - 1),
                "output_components": int(cv2.connectedComponents(filtered.astype(np.uint8), 8)[0] - 1),
                "input_foreground_pixels": int(extent_mask.sum()),
                "output_foreground_pixels": int(filtered.sum()),
            })
    manifest = {
        "schema_version": 1,
        "kind": "promptable_nvs_method_v1_spin9_identity_supported_sam3_predictions",
        "candidate_id": (
            LOCAL_DENSITY_CANDIDATE_ID
            if args.support_denominator == "component_area"
            else CANDIDATE_ID
        ),
        "protocol_hash": extent["protocol_hash"],
        "scene_order": scenes,
        "prediction_root": ".",
        "predictions": predictions,
        "prediction_sha256": hashes,
        "parents": {
            "extent": {"path": str(extent_path), "sha256": args.expected_extent_manifest_sha256},
            "threshold_authority": {"path": str(authority_path), "sha256": args.expected_threshold_authority_sha256},
        },
        "method": {
            "operation": "delete_disconnected_extent_components_only",
            "minimum_component_overlap_over_coarse_identity_support": threshold,
            "support_denominator": args.support_denominator,
            "threshold_transferred_from_nvos_without_spin_target_selection": True,
            "adds_foreground": False,
        },
        "rows": rows,
        "development_subset": bool(extent.get("development_subset", False)),
        "evaluation_performed": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
    }
    manifest_path = output / "prediction_manifest.json"
    _write_json(manifest_path, manifest)
    return {**manifest, "prediction_manifest": str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extent-manifest", required=True)
    parser.add_argument("--expected-extent-manifest-sha256", required=True)
    parser.add_argument("--threshold-authority", required=True)
    parser.add_argument("--expected-threshold-authority-sha256", required=True)
    parser.add_argument("--minimum-coarse-support-fraction", type=float, default=MINIMUM_COARSE_SUPPORT_FRACTION)
    parser.add_argument(
        "--support-denominator",
        choices=("coarse_identity_mass", "component_area"),
        default="coarse_identity_mass",
    )
    parser.add_argument("--output-dir", required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({"prediction_manifest": report["prediction_manifest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
