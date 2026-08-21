#!/usr/bin/env python3
"""Bind a target-blind threshold authority for NVOS component filtering.

NVOS provides a reference RGB and signed scribbles, not a full reference mask.
Consequently the component threshold is not identifiable from reference IoU.
This audit fails closed on that fact and inherits the already frozen coarse
overlap acceptance floor from the parent signed-evidence SAM3 selector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from radio_gs.scripts.predict_nvos_method_v1_field_box_sam3 import _write_json
from radio_gs.scripts.predict_nvos_method_v1_transient_sam import _sha256


ARTIFACT_TYPE = "nvos_identity_component_threshold_authority_v1"


def build(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset_manifest).expanduser().resolve(strict=True)
    extent_path = Path(args.extent_manifest).expanduser().resolve(strict=True)
    if _sha256(dataset_path) != args.expected_dataset_manifest_sha256:
        raise ValueError("dataset manifest SHA-256 differs")
    if _sha256(extent_path) != args.expected_extent_manifest_sha256:
        raise ValueError("extent manifest SHA-256 differs")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    extent = json.loads(extent_path.read_text(encoding="utf-8"))
    scenes = [str(value) for value in extent.get("scene_order", [])]
    dataset_scenes = {str(row["scene_id"]): row for row in dataset.get("scenes", [])}
    if len(scenes) != 8 or set(scenes) != set(dataset_scenes):
        raise ValueError("NVOS full-eight scene authority differs")

    reference_rows: list[dict[str, Any]] = []
    no_reference_full_mask = True
    for scene in scenes:
        row = dataset_scenes[scene]
        prompt = row.get("prompt", {})
        prompt_frame = str(prompt.get("frame_id", ""))
        frames = {str(frame["frame_id"]): frame for frame in row.get("frames", [])}
        frame = frames.get(prompt_frame, {})
        if (
            prompt.get("type") != "positive_negative_scribbles"
            or not prompt_frame
            or frame.get("ground_truth") is not None
            or frame.get("gt_mask_path") is not None
        ):
            no_reference_full_mask = False
        reference_rows.append(
            {
                "scene_id": scene,
                "frame_id": prompt_frame,
                "reference_full_mask_available": False,
                "positive_scribble": {
                    "path": str(prompt.get("positive_path", "")),
                    "sha256": _sha256(Path(str(prompt["positive_path"])).resolve(strict=True)),
                },
                "negative_scribble": {
                    "path": str(prompt.get("negative_path", "")),
                    "sha256": _sha256(Path(str(prompt["negative_path"])).resolve(strict=True)),
                },
            }
        )
    if not no_reference_full_mask:
        raise ValueError("NVOS reference-mask authority unexpectedly changed")

    parent_thresholds: list[float] = []
    parent_receipts: list[dict[str, str]] = []
    for record in extent.get("receipts", []):
        path = Path(str(record.get("path", ""))).resolve(strict=True)
        if _sha256(path) != str(record.get("sha256", "")):
            raise ValueError("parent extent receipt SHA-256 differs")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if (
            receipt.get("safety", {}).get("target_mask_opened") is not False
            or receipt.get("safety", {}).get("target_metric_opened") is not False
        ):
            raise ValueError("parent selector opened target supervision")
        parent_thresholds.append(float(receipt["sam3"]["minimum_coarse_overlap"]))
        parent_receipts.append({"path": str(path), "sha256": str(record["sha256"])})
    if len(parent_thresholds) != 8 or len(set(parent_thresholds)) != 1:
        raise ValueError("parent coarse-overlap floor is not one frozen full8 value")
    threshold = parent_thresholds[0]

    completion_root = Path(args.reference_completion_root).expanduser().resolve(strict=True)
    completion_rows: list[dict[str, str]] = []
    missing_completion: list[str] = []
    for scene in scenes:
        path = completion_root / "receipts" / f"{scene}.json"
        if not path.is_file():
            missing_completion.append(scene)
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        source = receipt.get("source", {})
        expected = next(row for row in reference_rows if row["scene_id"] == scene)
        if (
            receipt.get("target_mask_opened") is not False
            or receipt.get("target_metric_opened") is not False
            or source.get("frame_id") != expected["frame_id"]
            or source.get("positive_scribble_sha256") != expected["positive_scribble"]["sha256"]
            or source.get("negative_scribble_sha256") != expected["negative_scribble"]["sha256"]
            or receipt.get("method", {}).get("scribble_overwrite")
            != "raw positive true then raw negative false"
        ):
            raise ValueError("reference completion authority differs")
        completion_rows.append({"scene_id": scene, "path": str(path), "sha256": _sha256(path)})

    authority = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "fixed_analytic_inheritance_reference_loo_not_identifiable",
        "selected_threshold": threshold,
        "selection": {
            "mode": "inherit_parent_target_blind_coarse_overlap_floor",
            "parent_parameter": "sam3.minimum_coarse_overlap",
            "target_mask_or_metric_used": False,
            "scene_specific_parameter": False,
        },
        "reference_loo": {
            "available": False,
            "reason": (
                "NVOS has signed reference scribbles but no reference full mask; "
                "official completion overwrites positive/negative scribbles, so direct "
                "scribble utility is saturated and cannot identify a component threshold"
            ),
            "full_reference_masks": 0,
            "official_completion_receipts": len(completion_rows),
            "missing_completion_scenes": missing_completion,
        },
        "dataset_manifest": {"path": str(dataset_path), "sha256": args.expected_dataset_manifest_sha256},
        "extent_manifest": {"path": str(extent_path), "sha256": args.expected_extent_manifest_sha256},
        "parent_receipts": parent_receipts,
        "reference_prompts": reference_rows,
        "reference_completion_receipts": completion_rows,
        "target_mask_opened": False,
        "target_metric_opened": False,
    }
    output = Path(args.output).expanduser().resolve()
    _write_json(output, authority)
    return {**authority, "output": str(output), "sha256": _sha256(output)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--expected-dataset-manifest-sha256", required=True)
    parser.add_argument("--extent-manifest", required=True)
    parser.add_argument("--expected-extent-manifest-sha256", required=True)
    parser.add_argument("--reference-completion-root", required=True)
    parser.add_argument("--output", required=True)
    report = build(parser.parse_args(argv))
    print(json.dumps({"output": report["output"], "sha256": report["sha256"], "selected_threshold": report["selected_threshold"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
