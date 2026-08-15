#!/usr/bin/env python3
"""Verify the complete Method-v1 NVOS batch before opening any target mask."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from radio_gs.data.promptable_nvs_manifest import (
    validate_manifest as validate_dataset_manifest,
)
from radio_gs.evaluation.promptable_segmentation import evaluate_manifest
from radio_gs.five_benchmark_method_v1 import METHOD_ID, validate_method_authority


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METHOD_AUTHORITY = (
    REPO_ROOT / "paper/artifacts/five_benchmark_method_v1_authority_20260815.json"
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path, *, label: str) -> tuple[dict[str, Any], Path, str]:
    source = Path(path).expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, source, _sha256(source)


def _write_json_noclobber(path: Path, value: Mapping[str, Any]) -> None:
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
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _nested_record(
    value: Mapping[str, Any], scene_id: str, frame_id: str, *, label: str
) -> str:
    scene = value.get(scene_id)
    if not isinstance(scene, Mapping) or frame_id not in scene:
        raise ValueError(f"{label} is absent for {scene_id}/{frame_id}")
    return str(scene[frame_id])


def verify_full8_before_gt(
    *,
    dataset_manifest_path: str | Path,
    prediction_manifest_path: str | Path,
    method_authority_path: str | Path,
) -> dict[str, Any]:
    """Return a sealed batch record without reading any target mask bytes."""

    dataset, dataset_path, dataset_sha = _load_json(
        dataset_manifest_path, label="NVOS dataset manifest"
    )
    # check_files=False is the pre-GT boundary: roles and the declared protocol
    # are validated, but annotation paths are not opened here.
    normalized = validate_dataset_manifest(dataset, check_files=False)
    authority, authority_path, authority_sha = _load_json(
        method_authority_path, label="Method-v1 authority"
    )
    validate_method_authority(authority)
    scene_order = [str(value) for value in authority["frozen_cohorts"]["nvos"]]
    if [str(row["scene_id"]) for row in normalized["scenes"]] != scene_order:
        raise ValueError("dataset and Method-v1 NVOS full8 order differ")

    prediction, prediction_path, prediction_sha = _load_json(
        prediction_manifest_path, label="Method-v1 prediction manifest"
    )
    if (
        prediction.get("kind")
        != "promptable_nvs_method_v1_transient_sam_predictions"
        or prediction.get("protocol_hash") != normalized["protocol_hash"]
        or prediction.get("method", {}).get("id") != METHOD_ID
        or prediction.get("evaluation_performed") is not False
        or prediction.get("target_mask_opened") is not False
        or prediction.get("target_metric_opened") is not False
    ):
        raise ValueError("Method-v1 prediction manifest contract differs")
    predictions = prediction.get("predictions")
    prediction_hashes = prediction.get("prediction_sha256")
    if (
        not isinstance(predictions, Mapping)
        or not isinstance(prediction_hashes, Mapping)
        or list(predictions) != scene_order
        or list(prediction_hashes) != scene_order
    ):
        raise ValueError("Method-v1 prediction batch is not the ordered full8")
    receipt_rows = prediction.get("receipts")
    if not isinstance(receipt_rows, list):
        raise ValueError("Method-v1 prediction receipts are absent")
    receipt_index = {
        str(row.get("scene_id")): row
        for row in receipt_rows
        if isinstance(row, Mapping)
    }
    if list(receipt_index) != scene_order:
        raise ValueError("Method-v1 receipt batch is not the ordered full8")

    prediction_root = Path(str(prediction.get("prediction_root", ".")))
    if not prediction_root.is_absolute():
        prediction_root = prediction_path.parent / prediction_root
    scene_index = {str(row["scene_id"]): row for row in normalized["scenes"]}
    verified: dict[str, Any] = {}
    for scene_id in scene_order:
        scene = scene_index[scene_id]
        evaluation_ids = list(map(str, scene["evaluation_frame_ids"]))
        if len(evaluation_ids) != 1:
            raise ValueError(f"{scene_id} does not have one frozen target")
        frame_id = evaluation_ids[0]
        relative = Path(
            _nested_record(predictions, scene_id, frame_id, label="prediction")
        )
        score_path = relative if relative.is_absolute() else prediction_root / relative
        score_path = score_path.resolve(strict=True)
        expected_score_sha = _nested_record(
            prediction_hashes, scene_id, frame_id, label="prediction hash"
        )
        if _sha256(score_path) != expected_score_sha:
            raise ValueError(f"{scene_id} sealed prediction SHA256 differs")

        receipt_row = receipt_index[scene_id]
        receipt_path = Path(str(receipt_row.get("path", ""))).resolve(strict=True)
        receipt_sha = _sha256(receipt_path)
        if receipt_sha != receipt_row.get("sha256"):
            raise ValueError(f"{scene_id} transient receipt SHA256 differs")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        safety = receipt.get("safety", {})
        output = receipt.get("output", {})
        authorities = receipt.get("authorities", {})
        render = authorities.get("feature_render_authority", {})
        if (
            receipt.get("artifact_type")
            != "radio_gs_method_v1_nvos_transient_sam_receipt"
            or receipt.get("method_id") != METHOD_ID
            or receipt.get("scene_id") != scene_id
            or receipt.get("frame_id") != frame_id
            or receipt.get("signed_field_prompt", {}).get(
                "sealed_before_target_rgb_open"
            )
            is not True
            or output.get("continuous_margin_sha256") != expected_score_sha
            or Path(str(output.get("continuous_margin_path", ""))).resolve()
            != score_path
            or safety.get("target_rgb_opened") is not True
            or safety.get("target_mask_opened") is not False
            or safety.get("target_metric_opened") is not False
            or safety.get("reference_mask_selection") is not False
            or safety.get("graph_used") is not False
            or safety.get("connected_component_used") is not False
            or authorities.get("method_authority_sha256") != authority_sha
            or render.get("field_checkpoint_schema") != "factorized-v2"
        ):
            raise ValueError(f"{scene_id} transient receipt contract differs")
        field_path = Path(str(render.get("field_checkpoint", ""))).resolve(strict=True)
        field_sha = str(render.get("field_checkpoint_sha256", ""))
        if len(field_sha) != 64 or _sha256(field_path) != field_sha:
            raise ValueError(f"{scene_id} final Method-v1 field SHA256 differs")
        verified[scene_id] = {
            "frame_id": frame_id,
            "field": str(field_path),
            "field_sha256": field_sha,
            "prediction": str(score_path),
            "prediction_sha256": expected_score_sha,
            "receipt": str(receipt_path),
            "receipt_sha256": receipt_sha,
        }
    return {
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_sha256": dataset_sha,
        "prediction_manifest": str(prediction_path),
        "prediction_manifest_sha256": prediction_sha,
        "method_authority": str(authority_path),
        "method_authority_sha256": authority_sha,
        "protocol_hash": normalized["protocol_hash"],
        "scene_order": scene_order,
        "verified": verified,
        "all_eight_receipts_verified_before_first_target_ground_truth_open": True,
    }


def score(args: argparse.Namespace) -> dict[str, Any]:
    barrier = verify_full8_before_gt(
        dataset_manifest_path=args.manifest,
        prediction_manifest_path=args.prediction_manifest,
        method_authority_path=args.method_authority,
    )
    # The complete prediction barrier is now sealed, so target annotation bytes
    # may be checked against the frozen dataset authority and then scored.
    dataset = json.loads(
        Path(barrier["dataset_manifest"]).read_text(encoding="utf-8")
    )
    validate_dataset_manifest(dataset, check_files=True)
    # The generic evaluator re-verifies every prediction hash and applies the
    # already-frozen metric, resize, threshold, and aggregation implementation.
    report = evaluate_manifest(
        barrier["dataset_manifest"],
        prediction_manifest=barrier["prediction_manifest"],
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "radio_gs_method_v1_nvos_full8_results",
        "method_id": METHOD_ID,
        "pre_gt_barrier": barrier,
        "evaluation": report,
        "eligibility": {
            "exact_frozen_scene_cohort": True,
            "exact_frozen_metric_implementation": True,
            "method_v1_target_rgb_authorized": True,
            "legacy_strict_unseen_manifest_target_rgb_at_query": "forbidden",
            "strict_unseen_protocol_eligible": False,
            "reason": (
                "Method-v1 explicitly authorizes transient target RGB, while the "
                "legacy strict-unseen NVOS manifest forbids it. Metrics remain useful "
                "development evidence but are not an exact strict-unseen SOTA row."
            ),
        },
        "safety": {
            "all_eight_receipts_verified_before_first_target_ground_truth_open": True,
            "target_metrics_used_for_method_selection": False,
        },
    }
    output = Path(args.output).expanduser().resolve()
    _write_json_noclobber(output, payload)
    return {**payload, "output": str(output), "output_sha256": _sha256(output)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--method-authority", default=str(DEFAULT_METHOD_AUTHORITY))
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = score(build_parser().parse_args(argv))
    dataset = report["evaluation"]["dataset"]
    print(
        json.dumps(
            {
                "output": report["output"],
                "output_sha256": report["output_sha256"],
                "foreground_iou": dataset["foreground_iou"],
                "pixel_accuracy": dataset["pixel_accuracy"],
                "strict_unseen_protocol_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
