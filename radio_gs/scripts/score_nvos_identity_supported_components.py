#!/usr/bin/env python3
"""Score a sealed full-eight identity-supported SAM3 component batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from radio_gs.evaluation.promptable_segmentation import evaluate_manifest
from radio_gs.scripts.filter_nvos_sam3_components_by_identity_support import (
    CANDIDATE_ID,
    LOCAL_DENSITY_CANDIDATE_ID,
)
from radio_gs.scripts.predict_nvos_method_v1_field_box_sam3 import _write_json
from radio_gs.scripts.predict_nvos_method_v1_transient_sam import _sha256


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--expected-prediction-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    prediction = Path(args.prediction_manifest).expanduser().resolve(strict=True)
    if _sha256(prediction) != args.expected_prediction_manifest_sha256:
        raise ValueError("prediction manifest SHA-256 differs")
    manifest = json.loads(prediction.read_text(encoding="utf-8"))
    if (
        manifest.get("candidate_id") not in (CANDIDATE_ID, LOCAL_DENSITY_CANDIDATE_ID)
        or manifest.get("kind")
        != "promptable_nvs_method_v1_identity_supported_sam3_predictions"
        or len(manifest.get("scene_order", [])) != 8
        or manifest.get("all_eight_predictions_sealed") is not True
        or manifest.get("evaluation_performed") is not False
        or manifest.get("target_mask_opened") is not False
        or manifest.get("target_metric_opened") is not False
    ):
        raise ValueError("identity-supported prediction barrier differs")
    evaluation = evaluate_manifest(
        Path(args.dataset_manifest).expanduser().resolve(strict=True),
        prediction_manifest=prediction,
    )
    result = {
        "schema_version": 1,
        "artifact_type": "radio_gs_nvos_identity_supported_sam3_components_result",
        "candidate_id": manifest["candidate_id"],
        "pre_gt_barrier": {
            "prediction_manifest": str(prediction),
            "prediction_manifest_sha256": args.expected_prediction_manifest_sha256,
            "all_eight_predictions_verified_before_scoring": True,
        },
        "evaluation": evaluation,
        "eligibility": {
            "development_use_only": True,
            "target_rgb_assisted": True,
            "prospective_promotion_eligible": False,
            "reason": "component support threshold inspected during development after prior target metrics",
        },
    }
    output = Path(args.output).expanduser().resolve()
    _write_json(output, result)
    print(json.dumps({"output": str(output), **evaluation["dataset"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
