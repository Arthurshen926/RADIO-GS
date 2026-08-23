#!/usr/bin/env python3
"""Apply the frozen component-local identity limiter to sealed NVOS predictions.

The input extent is produced by the carrier-native all-view SAM compiler.  The
signed field unary remains the identity authority.  This stage may only delete
disconnected foreground components; it cannot add pixels or inspect a target
mask.  Every parent receipt is verified before any output is materialized so a
full-eight invocation preserves the prediction barrier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from radio_gs.evaluation.promptable_segmentation import resize_mask_nearest
from radio_gs.querying.transient_rgb_sam import (
    FROZEN_POLICY,
    deterministic_signed_point_trials,
)
from radio_gs.scripts.build_nvos_identity_component_threshold_authority import (
    ARTIFACT_TYPE as THRESHOLD_AUTHORITY_TYPE,
)
from radio_gs.scripts.filter_nvos_sam3_components_by_identity_support import (
    LOCAL_DENSITY_CANDIDATE_ID,
    identity_supported_components_local_density,
)
from radio_gs.scripts.render_nvos_synchronous_candidate_marginal import (
    RECEIPT_TYPE,
    _atomic_json,
    _atomic_numpy,
    _bound,
    _sha256,
)


FILTERED_RECEIPT_TYPE = (
    "nvos_synchronous_multiview_identity_supported_target_prediction_v1"
)


def _load_manifest(
    path: str | Path, expected_sha256: str, *, label: str
) -> tuple[dict[str, Any], Path]:
    source = _bound(path, expected_sha256, label)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, source


def _prediction_path(
    manifest: Mapping[str, Any], source: Path, scene_id: str, frame_id: str
) -> Path:
    scene = manifest.get("predictions", {}).get(scene_id, {})
    if set(scene) != {frame_id}:
        raise ValueError("signed unary target-frame authority differs")
    root = Path(str(manifest.get("prediction_root", ".")))
    if not root.is_absolute():
        root = source.parent / root
    record = manifest.get("prediction_sha256", {}).get(scene_id, {})
    return _bound(root / str(scene[frame_id]), record.get(frame_id, ""), "signed unary")


def filter_batch(args: argparse.Namespace) -> dict[str, Any]:
    authority_bound_replay = bool(
        getattr(args, "authority_bound_replay_after_prior_metrics", False)
    )
    unary, unary_path = _load_manifest(
        args.signed_unary_manifest,
        args.expected_signed_unary_manifest_sha256,
        label="signed unary manifest",
    )
    authority, authority_path = _load_manifest(
        args.threshold_authority,
        args.expected_threshold_authority_sha256,
        label="component threshold authority",
    )
    threshold = float(authority.get("selected_threshold", -1))
    if (
        unary.get("kind") != "promptable_nvs_continuous_score_predictions"
        or unary.get("safety", {}).get("evaluation_ground_truth_opened") is not False
        or authority.get("artifact_type") != THRESHOLD_AUTHORITY_TYPE
        or authority.get("target_mask_opened") is not False
        or authority.get("target_metric_opened") is not False
        or threshold != float(args.minimum_local_identity_density)
    ):
        raise ValueError("identity authority contract differs")

    # Barrier: validate the complete requested cohort and every prediction byte
    # before producing the first filtered output.
    sealed: list[tuple[dict[str, Any], Path, Path]] = []
    for value in args.receipt:
        receipt_path = Path(value).expanduser().resolve(strict=True)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("artifact_type") != RECEIPT_TYPE
            or receipt.get("prediction_sealed_before_target_ground_truth") is not True
            or receipt.get("target_mask_opened") is not False
            or receipt.get("target_metric_opened") is not False
        ):
            raise ValueError("parent prediction receipt opened target evaluation")
        prediction = receipt.get("prediction", {})
        prediction_path = _bound(
            prediction.get("path", ""), prediction.get("sha256", ""), "parent prediction"
        )
        sealed.append((receipt, receipt_path, prediction_path))
    scene_ids = [str(receipt["scene_id"]) for receipt, _, _ in sealed]
    if not sealed or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("prediction batch is empty or contains duplicate scenes")
    if args.require_full8 and len(scene_ids) != 8:
        raise ValueError("formal NVOS batch must contain exactly eight scenes")

    output_root = Path(args.output_root).expanduser().resolve()
    outputs: list[dict[str, Any]] = []
    for receipt, receipt_path, prediction_path in sealed:
        scene_id = str(receipt["scene_id"])
        frame_id = str(receipt["target_frame_id"])
        signed_path = _prediction_path(unary, unary_path, scene_id, frame_id)
        probability = np.load(prediction_path, allow_pickle=False)
        if (
            probability.ndim != 2
            or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))
        ):
            raise ValueError("parent probability is not finite in [0,1]")
        extent = probability >= float(receipt.get("threshold", 0.5))
        signed_margin = np.load(signed_path, allow_pickle=False).astype(
            np.float32, copy=False
        )
        if signed_margin.ndim != 2 or not np.isfinite(signed_margin).all():
            raise ValueError("signed identity unary is invalid")
        coarse = resize_mask_nearest(signed_margin >= 0, extent.shape).astype(bool)
        points, labels = deterministic_signed_point_trials(
            np.maximum(signed_margin, 0),
            np.maximum(-signed_margin, 0),
            image_shape=extent.shape,
            policy=FROZEN_POLICY,
        )
        filtered = identity_supported_components_local_density(
            extent,
            coarse,
            points[labels == 1],
            minimum_local_identity_density=threshold,
        )
        if bool((filtered & ~extent).any()):
            raise RuntimeError("identity limiter added foreground")

        scene_root = output_root / scene_id
        output_path = scene_root / "target_probability.npy"
        output_sha256 = _atomic_numpy(output_path, filtered.astype(np.float32))
        before_components = int(
            cv2.connectedComponents(extent.astype(np.uint8), connectivity=8)[0] - 1
        )
        after_components = int(
            cv2.connectedComponents(filtered.astype(np.uint8), connectivity=8)[0] - 1
        )
        filtered_receipt = {
            "schema_version": 1,
            "artifact_type": FILTERED_RECEIPT_TYPE,
            "candidate_id": LOCAL_DENSITY_CANDIDATE_ID,
            "scene_id": scene_id,
            "target_frame_id": frame_id,
            "parent_prediction_receipt": {
                "path": str(receipt_path),
                "sha256": _sha256(receipt_path),
            },
            "signed_unary_manifest": {
                "path": str(unary_path),
                "sha256": args.expected_signed_unary_manifest_sha256,
            },
            "signed_unary": {"path": str(signed_path), "sha256": _sha256(signed_path)},
            "threshold_authority": {
                "path": str(authority_path),
                "sha256": args.expected_threshold_authority_sha256,
            },
            "minimum_local_identity_density": threshold,
            "operation": "delete_disconnected_extent_components_only",
            "adds_foreground": False,
            "input_components": before_components,
            "output_components": after_components,
            "input_foreground_pixels": int(extent.sum()),
            "output_foreground_pixels": int(filtered.sum()),
            "prediction": {"path": str(output_path), "sha256": output_sha256},
            "prediction_sealed_before_target_ground_truth": not authority_bound_replay,
            "parent_prediction_pre_metric_sealed": True,
            "authority_bound_deterministic_replay_after_prior_development_metrics": (
                authority_bound_replay
            ),
            "operator_read_target_mask_or_metric": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
            "threshold": 0.5,
        }
        filtered_receipt_path = scene_root / "prediction_receipt.json"
        _atomic_json(filtered_receipt_path, filtered_receipt)
        outputs.append(
            {
                "scene_id": scene_id,
                "receipt": str(filtered_receipt_path),
                "receipt_sha256": _sha256(filtered_receipt_path),
            }
        )
    return {
        "artifact_type": "nvos_synchronous_multiview_identity_supported_batch_v1",
        "scene_order": scene_ids,
        "outputs": outputs,
        "all_outputs_sealed": True,
        "prediction_batch_fully_sealed_before_first_target_mask_open": (
            not authority_bound_replay
        ),
        "authority_bound_deterministic_replay_after_prior_development_metrics": (
            authority_bound_replay
        ),
        "target_mask_opened": False,
        "target_metric_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", action="append", required=True)
    parser.add_argument("--require-full8", action="store_true")
    parser.add_argument("--signed-unary-manifest", required=True)
    parser.add_argument("--expected-signed-unary-manifest-sha256", required=True)
    parser.add_argument("--threshold-authority", required=True)
    parser.add_argument("--expected-threshold-authority-sha256", required=True)
    parser.add_argument("--minimum-local-identity-density", type=float, default=0.05)
    parser.add_argument(
        "--authority-bound-replay-after-prior-metrics",
        action="store_true",
        help=(
            "Record that the operator and threshold were frozen but this mechanical "
            "replay was executed after development metrics had already been opened"
        ),
    )
    parser.add_argument("--output-root", required=True)
    print(json.dumps(filter_batch(parser.parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
