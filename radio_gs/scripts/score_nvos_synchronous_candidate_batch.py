#!/usr/bin/env python3
"""Score a fully sealed NVOS synchronous prediction batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from radio_gs.evaluation.promptable_segmentation import (
    compute_binary_metrics,
    load_ground_truth_mask,
)
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _resize_nvos_score_for_evaluation,
)
from radio_gs.scripts.render_nvos_synchronous_candidate_marginal import (
    RECEIPT_TYPE,
    _atomic_json,
    _bound,
)
from radio_gs.scripts.filter_nvos_synchronous_prediction_by_identity_support import (
    FILTERED_RECEIPT_TYPE,
)


RESULT_TYPE = "nvos_synchronous_multiview_candidate_batch_result_v1"


def _scene(manifest: Mapping[str, Any], scene_id: str) -> Mapping[str, Any]:
    values = [row for row in manifest.get("scenes", []) if row.get("scene_id") == scene_id]
    if len(values) != 1:
        raise ValueError("NVOS result scene authority differs")
    return values[0]


def score(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not args.receipt:
        raise ValueError("at least one sealed prediction receipt is required")
    sealed: list[tuple[dict[str, Any], Path]] = []
    for value in args.receipt:
        path = Path(value).expanduser().resolve(strict=True)
        receipt = json.loads(path.read_text(encoding="utf-8"))
        prediction = receipt.get("prediction", {})
        replay = bool(
            receipt.get(
                "authority_bound_deterministic_replay_after_prior_development_metrics",
                False,
            )
        )
        if (
            receipt.get("artifact_type") not in {RECEIPT_TYPE, FILTERED_RECEIPT_TYPE}
            or (
                receipt.get("prediction_sealed_before_target_ground_truth") is not True
                and not replay
            )
            or receipt.get("target_mask_opened") is not False
            or receipt.get("target_metric_opened") is not False
            or (replay and receipt.get("operator_read_target_mask_or_metric") is not False)
        ):
            raise ValueError("prediction receipt opened target evaluation")
        _bound(prediction.get("path", ""), prediction.get("sha256", ""), "prediction")
        sealed.append((receipt, path))
    scene_ids = [str(receipt["scene_id"]) for receipt, _ in sealed]
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("prediction batch contains a duplicate scene")
    if args.require_full8 and len(scene_ids) != 8:
        raise ValueError("formal NVOS batch must contain exactly eight scenes")

    # Only after every prediction/receipt in the requested batch is validated
    # do target mask bytes become reachable.
    per_scene: dict[str, Any] = {}
    for receipt, receipt_path in sealed:
        scene_id = str(receipt["scene_id"])
        scene = _scene(manifest, scene_id)
        frames = scene.get("frames", [])
        values = frames.values() if isinstance(frames, Mapping) else frames
        matches = [
            row
            for row in values
            if str(row.get("frame_id")) == str(receipt["target_frame_id"])
        ]
        if len(matches) != 1:
            raise ValueError("target frame metadata differs")
        target_path = Path(
            str(matches[0].get("ground_truth") or matches[0].get("gt_mask_path"))
        ).resolve(strict=True)
        declared = str(matches[0].get("ground_truth_sha256") or "")
        _bound(target_path, declared, "target ground truth")
        ground_truth = load_ground_truth_mask(target_path)
        probability = np.load(receipt["prediction"]["path"], allow_pickle=False)
        resized = _resize_nvos_score_for_evaluation(
            probability,
            tuple(map(int, ground_truth.shape)),
            registered_forward_unary="none",
        )
        metrics = compute_binary_metrics(resized >= 0.5, ground_truth)
        per_scene[scene_id] = {
            **metrics,
            "receipt": {"path": str(receipt_path)},
            "target_frame_id": str(receipt["target_frame_id"]),
        }
    macro_iou = float(np.mean([row["foreground_iou"] for row in per_scene.values()]))
    macro_accuracy = float(np.mean([row["pixel_accuracy"] for row in per_scene.values()]))
    result = {
        "schema_version": 1,
        "artifact_type": RESULT_TYPE,
        "status": "complete_full8" if len(per_scene) == 8 else "complete_sentinel",
        "scene_order": scene_ids,
        "per_scene": per_scene,
        "macro_foreground_iou": macro_iou,
        "macro_pixel_accuracy": macro_accuracy,
        "prediction_batch_fully_sealed_before_first_target_mask_open": all(
            receipt.get("prediction_sealed_before_target_ground_truth") is True
            for receipt, _ in sealed
        ),
        "authority_bound_replay_after_prior_development_metrics": any(
            receipt.get(
                "authority_bound_deterministic_replay_after_prior_development_metrics",
                False,
            )
            for receipt, _ in sealed
        ),
        "development_only": True,
        "sota_claim": False,
    }
    output = Path(args.output).expanduser().resolve()
    _atomic_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--receipt", action="append", required=True)
    parser.add_argument("--require-full8", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(score(build_parser().parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
