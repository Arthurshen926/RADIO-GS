#!/usr/bin/env python3
"""Evaluate a source-gated categorical calibration on frozen paper8 caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS
from radio_gs.scripts.eval_ours_scannet_vala_gaussian_protocol import (
    PAPER_CLASS_IDS,
    volume_weighted_split_metrics,
)
from radio_gs.scripts.train_scannet_source_categorical_calibration_v3 import SCHEMA
from radio_gs.utils.immutable_artifacts import sha256_file


OUTPUT_SCHEMA = "radio_gs.scannet_source_calibrated_paper8_development.v1"


def calibrated_predictions(
    scores: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
) -> dict[str, np.ndarray]:
    values = torch.as_tensor(scores).float() * torch.as_tensor(scale).float()[None]
    values = values + torch.as_tensor(bias).float()[None]
    column = {class_id: index for index, class_id in enumerate(PAPER_CLASS_IDS)}
    output = {}
    for split in ("19", "15", "10"):
        class_ids = OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
        indices = torch.tensor([column[value] for value in class_ids])
        output[split] = np.asarray(class_ids, dtype=np.int32)[
            values.index_select(1, indices).argmax(1).numpy()
        ]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration_path = args.calibration.expanduser().resolve(strict=True)
    calibration = torch.load(calibration_path, map_location="cpu", weights_only=False)
    if (
        calibration.get("schema") != SCHEMA
        or calibration.get("all_loso_folds_passed") is not True
        or calibration.get("access_contract", {}).get("paper8_labels_opened") is not False
        or list(calibration.get("class_ids", [])) != list(PAPER_CLASS_IDS)
    ):
        raise ValueError("categorical calibration did not pass the independent source gate")
    scale = torch.as_tensor(calibration["positive_scale"]).float()
    bias = torch.as_tensor(calibration["bias"]).float()
    if scale.shape != (19,) or bias.shape != (19,) or bool((scale <= 0).any()):
        raise ValueError("categorical calibration parameters differ")
    baseline_path = args.baseline_result.expanduser().resolve(strict=True)
    baseline = json.loads(baseline_path.read_text())
    rows: dict[str, Any] = {}
    for scene_id, scene in baseline["scenes"].items():
        score_path = Path(scene["semantic_score_cache"]).resolve(strict=True)
        if sha256_file(score_path) != scene["semantic_score_cache_sha256"]:
            raise ValueError("frozen paper8 semantic score cache changed")
        cache = torch.load(score_path, map_location="cpu", weights_only=False)
        scores = torch.as_tensor(cache["semantic_scores"]).float()
        prediction_path = Path(scene["prediction_npz"]).resolve(strict=True)
        frozen = np.load(prediction_path, allow_pickle=False)
        pseudo = np.asarray(frozen["pseudo_labels"], dtype=np.int32)
        significance = np.asarray(frozen["significance"], dtype=np.float64)
        predictions = calibrated_predictions(scores, scale, bias)
        splits = {
            split: volume_weighted_split_metrics(
                pseudo,
                predictions[split],
                significance,
                OPENGAUSSIAN_NYU40_CLASS_SPLITS[split],
            )
            for split in ("19", "15", "10")
        }
        rows[scene_id] = {"splits": splits, "frozen_prediction": str(prediction_path), "semantic_score_cache": str(score_path)}
    macro = {
        split: {
            metric: float(np.mean([row["splits"][split][metric] for row in rows.values()]))
            for metric in ("miou", "macc")
        }
        for split in ("19", "15", "10")
    }
    delta = {
        split: {
            metric: macro[split][metric] - float(baseline["macro"][split][metric])
            for metric in ("miou", "macc")
        }
        for split in ("19", "15", "10")
    }
    result = {
        "schema": OUTPUT_SCHEMA,
        "schema_version": 1,
        "status": "complete_development_after_independent_source_loso_gate",
        "calibration": {"path": str(calibration_path), "sha256": sha256_file(calibration_path)},
        "baseline": {"path": str(baseline_path), "sha256": sha256_file(baseline_path), "macro": baseline["macro"]},
        "macro": macro,
        "delta": delta,
        "scenes": rows,
        "method": "frozen_region_scores_then_source_only_positive_diagonal_scale_plus_bias",
        "access_contract": {
            "source_loso_passed_before_paper8_metrics": True,
            "paper8_metrics_opened_for_development_evaluation": True,
            "scene_specific_parameters": False,
            "test_metric_tuning": False,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "macro": macro, "delta": delta}, indent=2))


if __name__ == "__main__":
    main()
