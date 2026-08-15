#!/usr/bin/env python3
"""Aggregate canonical and SAM query posteriors using reference-only authority."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from radio_gs.querying.transient_rgb_sam import (
    PromptMode,
    transient_adapter_contract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _metric(value: object, label: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} is outside [0,1]: {result}")
    return result


def _parse_report_specs(specs: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        scene, separator, raw_path = spec.partition("=")
        if not separator or not scene or not raw_path:
            raise ValueError("--sam-report must have the form SCENE=PATH")
        if scene in result:
            raise ValueError(f"duplicate SAM report for scene {scene!r}")
        result[scene] = Path(raw_path).resolve()
    return result


def aggregate(canonical_path: Path, sam_paths: dict[str, Path]) -> dict[str, Any]:
    canonical_path = canonical_path.resolve()
    canonical = _load(canonical_path)
    canonical_scenes = canonical.get("scenes")
    if not isinstance(canonical_scenes, dict) or not canonical_scenes:
        raise ValueError("canonical audit has no scenes")
    if set(sam_paths) != set(canonical_scenes):
        missing = sorted(set(canonical_scenes) - set(sam_paths))
        extra = sorted(set(sam_paths) - set(canonical_scenes))
        raise ValueError(f"SAM report scene mismatch: missing={missing}, extra={extra}")

    protocol_hash = str(canonical["protocol_hash"])
    scenes: dict[str, Any] = {}
    for scene in sorted(canonical_scenes):
        canonical_row = canonical_scenes[scene]
        sam_path = sam_paths[scene].resolve()
        sam = _load(sam_path)
        if str(sam.get("scene_id")) != scene:
            raise ValueError(f"SAM report scene mismatch for {scene}: {sam.get('scene_id')!r}")
        if str(sam.get("protocol_hash")) != protocol_hash:
            raise ValueError(f"SAM protocol hash mismatch for {scene}")
        if sam.get("prediction_persisted_before_target_mask_access") is not True:
            raise ValueError(f"SAM prediction receipt is not sealed for {scene}")
        if sam.get("persisted_field_used_target_rgb_during_original_generation") is not True:
            raise ValueError(f"SAM target-RGB provenance is not declared for {scene}")
        if sam.get("target_masks_used_during_original_generation") is not False:
            raise ValueError(f"SAM persistent field used a target mask for {scene}")
        receipt = sam.get("reference_receipt")
        if not isinstance(receipt, dict) or receipt.get("target_masks_opened") is not False:
            raise ValueError(f"SAM reference receipt is invalid for {scene}")

        canonical_reference = _metric(
            canonical_row["selected_by_reference_only"]["reference_iou"],
            f"{scene} canonical reference IoU",
        )
        canonical_target = _metric(
            canonical_row["query_conditioned_support"],
            f"{scene} canonical target IoU",
        )
        sam_reference = _metric(
            receipt["selected_reference_iou"], f"{scene} SAM reference IoU"
        )
        sam_target = _metric(sam["foreground_iou"], f"{scene} SAM target IoU")
        selected_branch = "sam" if sam_reference > canonical_reference else "canonical"
        selected_reference = sam_reference if selected_branch == "sam" else canonical_reference
        selected_target = sam_target if selected_branch == "sam" else canonical_target
        scenes[scene] = {
            "canonical": {
                "reference_iou": canonical_reference,
                "target_foreground_iou": canonical_target,
                "report": canonical_row["report"],
                "report_sha256": canonical_row["report_sha256"],
            },
            "sam": {
                "reference_iou": sam_reference,
                "target_foreground_iou": sam_target,
                "report": str(sam_path),
                "report_sha256": _sha256(sam_path),
                "render_resolution_mode": sam.get("render_resolution_mode"),
                "renderer_resolution": sam.get("renderer_resolution"),
                "selected_candidate": receipt.get("selected_candidate"),
                "selected_threshold": receipt.get("selected_threshold"),
            },
            "selected_branch": selected_branch,
            "selected_reference_iou": selected_reference,
            "selected_target_foreground_iou": selected_target,
        }

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in scenes.values()) / len(scenes)

    canonical_macro = sum(
        row["canonical"]["target_foreground_iou"] for row in scenes.values()
    ) / len(scenes)
    sam_macro = sum(
        row["sam"]["target_foreground_iou"] for row in scenes.values()
    ) / len(scenes)
    selected_macro = mean("selected_target_foreground_iou")
    local_ludvig = float(
        canonical.get("macro_over_requested_scenes", {}).get("local_ludvig_sam", "nan")
    )
    return {
        "schema_version": "spin9_reference_selected_query_interface_result_v1",
        "selector_contract": {
            "authority": "single permitted full reference mask",
            "rule": "select greater reference foreground IoU",
            "exact_tie": "canonical",
            "target_metric_used_for_selection": False,
            "scene_identifier_used_for_selection": False,
        },
        "transient_adapter_contract": transient_adapter_contract(
            PromptMode.FULL_REFERENCE_MASK
        ),
        "protocol_hash": protocol_hash,
        "canonical_audit": str(canonical_path),
        "canonical_audit_sha256": _sha256(canonical_path),
        "scene_count": len(scenes),
        "scenes": scenes,
        "macro": {
            "canonical": canonical_macro,
            "sam": sam_macro,
            "reference_selected": selected_macro,
            "local_ludvig_sam": local_ludvig,
            "delta_selected_vs_canonical": selected_macro - canonical_macro,
            "delta_selected_vs_local_ludvig_sam": selected_macro - local_ludvig,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-audit", type=Path, required=True)
    parser.add_argument(
        "--sam-report",
        action="append",
        default=[],
        help="Repeat once per scene using SCENE=PATH.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.canonical_audit, _parse_report_specs(args.sam_report))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["macro"]))


if __name__ == "__main__":
    main()
