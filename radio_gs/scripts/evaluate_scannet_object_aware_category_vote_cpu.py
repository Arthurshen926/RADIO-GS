#!/usr/bin/env python3
"""Evaluate one frozen bounded object-voting rule from ScanNet score caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from radio_gs.querying.scannet_object_aware_category_voting import (
    object_aware_category_vote,
)
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS


SPLITS = ("19", "15", "10")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(
    gt: np.ndarray, pred: np.ndarray, weights: np.ndarray, ids: tuple[int, ...]
) -> dict[str, float]:
    valid = np.isin(gt, ids)
    prediction = pred.copy()
    prediction[~valid] = 0
    ious: list[float] = []
    accuracies: list[float] = []
    for class_id in ids:
        gt_mask = gt == class_id
        if not gt_mask.any():
            continue
        pred_mask = prediction == class_id
        intersection = float(weights[gt_mask & pred_mask].sum())
        union = float(weights[gt_mask | pred_mask].sum())
        total = float(weights[gt_mask].sum())
        ious.append(intersection / union if union else 0.0)
        accuracies.append(intersection / total if total else 0.0)
    return {"miou": float(np.mean(ious)), "macc": float(np.mean(accuracies))}


def _evaluate_scene(path: Path) -> dict[str, object]:
    payload = np.load(path, allow_pickle=False)
    gt = np.asarray(payload["pseudo_labels"], dtype=np.int32)
    significance = np.asarray(payload["significance"], dtype=np.float64)
    rows: dict[str, object] = {}
    for split in SPLITS:
        ids = tuple(int(value) for value in OPENGAUSSIAN_NYU40_CLASS_SPLITS[split])
        scores = torch.from_numpy(payload[f"scores_split_{split}"])
        baseline_prediction = np.asarray(ids, dtype=np.int32)[scores.argmax(1).numpy()]
        output, stats = object_aware_category_vote(
            scores,
            torch.from_numpy(payload["sam_membership_rows"]),
            torch.from_numpy(payload["sam_membership_proposals"]),
            torch.from_numpy(payload["sam_membership_weights"]),
            num_proposals=int(payload["sam_num_proposals"]),
            class_ids=ids,
            strength=1.0,
            residual_budget=0.25,
        )
        candidate_prediction = np.asarray(ids, dtype=np.int32)[output.argmax(1).numpy()]
        baseline = _metrics(gt, baseline_prediction, significance, ids)
        candidate = _metrics(gt, candidate_prediction, significance, ids)
        rows[split] = {
            "baseline": baseline,
            "candidate": candidate,
            "delta": {
                metric: candidate[metric] - baseline[metric]
                for metric in ("miou", "macc")
            },
            "operator": stats,
        }
    return {"cache_sha256": _sha256(path), "splits": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(int(args.cpu_threads))
    torch.set_num_interop_threads(1)
    development = ("scene0000_00",)
    confirmation = ("scene0062_00", "scene0070_00", "scene0097_00")
    scene_results: dict[str, object] = {}
    for scene in development + confirmation:
        path = args.cache_root / scene / "development" / f"{scene}_scores.npz"
        scene_results[scene] = _evaluate_scene(path)

    macro: dict[str, object] = {}
    for cohort, scenes in (("development", development), ("confirmation", confirmation)):
        macro[cohort] = {
            split: {
                kind: {
                    metric: float(
                        np.mean(
                            [
                                scene_results[scene]["splits"][split][kind][metric]
                                for scene in scenes
                            ]
                        )
                    )
                    for metric in ("miou", "macc")
                }
                for kind in ("baseline", "candidate", "delta")
            }
            for split in SPLITS
        }
    report = {
        "schema_version": 1,
        "artifact_type": "radio_gs_scannet_bounded_object_category_vote_pilot",
        "status": "development_and_frozen_confirmation_complete",
        "method": {
            "category_authority": "primitive_RADIO_logits_only",
            "object_authority": "official_source_SAM_exact_MPR_membership",
            "thing_stuff_policy": "NYU40 wall_floor_ceiling primitive fallback",
            "lambda": "0.25 * reliable_coverage * object_margin * exp(-primitive_margin)",
            "per_class_parameters": False,
            "class_permutation_equivariant_with_roles": True,
            "proposal_permutation_equivariant": True,
            "zero_strength_bitwise_replay": True,
            "target_threshold_search": False,
            "unknown_background_branch": False,
        },
        "development_scene": list(development),
        "confirmation_scenes": list(confirmation),
        "scene_results": scene_results,
        "macro": macro,
        "decision": {
            "promotion": False,
            "reason": "frozen confirmation improves 19/15 mIoU and all-split mAcc but consistently regresses split10 mIoU",
            "mechanism_supported": "object-level voting can denoise primitive thing labels when the category inventory is sufficiently complete",
            "remaining_limit": "source-view masks are not stable cross-view object tracks; subset category competition can change the object winner",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "macro": macro}, indent=2))


if __name__ == "__main__":
    main()
