#!/usr/bin/env python3
"""Select one conservative SAM-instance posterior on a disjoint dev cohort."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from radio_gs.querying.sam_categorical_instance_posterior import (
    propagate_categorical_identity_over_proposals,
)
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import (
    volume_weighted_split_metrics,
)


SPLITS = ("19", "15", "10")


def _metrics(cache: dict[str, np.ndarray], predictions: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        split: {
            key: float(value)
            for key, value in volume_weighted_split_metrics(
                cache["pseudo_labels"],
                predictions[split],
                cache["significance"],
                OPENGAUSSIAN_NYU40_CLASS_SPLITS[split],
            ).items()
            if key in {"miou", "macc"}
        }
        for split in SPLITS
    }


def _labels(scores: torch.Tensor, split: str) -> np.ndarray:
    class_ids = np.asarray(OPENGAUSSIAN_NYU40_CLASS_SPLITS[split], dtype=np.int32)
    return class_ids[scores.argmax(dim=1).numpy()]


def _macro(rows: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    return {
        split: {
            metric: float(np.mean([row[split][metric] for row in rows.values()]))
            for metric in ("miou", "macc")
        }
        for split in SPLITS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--scenes",
        default="scene0000_00,scene0062_00,scene0070_00,scene0097_00",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(int(args.cpu_threads))
    torch.set_num_interop_threads(1)
    scenes = tuple(value.strip() for value in args.scenes.split(",") if value.strip())
    caches: dict[str, dict[str, np.ndarray]] = {}
    baseline_rows: dict[str, dict[str, dict[str, float]]] = {}
    for scene in scenes:
        path = args.cache_root / scene / "development" / f"{scene}_scores.npz"
        payload = np.load(path, allow_pickle=False)
        cache = {key: np.asarray(payload[key]) for key in payload.files}
        caches[scene] = cache
        baseline_rows[scene] = _metrics(
            cache,
            {
                split: _labels(torch.from_numpy(cache[f"scores_split_{split}"]), split)
                for split in SPLITS
            },
        )
    baseline_macro = _macro(baseline_rows)

    variants: list[dict[str, object]] = []
    settings_grid = itertools.product(
        (0.005, 0.01),
        (0.0001, 0.00025, 0.0005, 0.001),
        (0.80, 0.90, 1.0),
        (4, 6, 8),
        (3, 4, 5),
    )
    for update_margin, tolerance, consensus, min_proposals, min_views in settings_grid:
        if tolerance > update_margin:
            continue
        settings = {
            "seed_margin_threshold": 0.04,
            "update_margin_threshold": update_margin,
            "semantic_tolerance": tolerance,
            "consensus_threshold": consensus,
            "minimum_supporting_proposals": min_proposals,
            "minimum_supporting_views": min_views,
            "iterations": 1,
        }
        scene_rows: dict[str, dict[str, dict[str, float]]] = {}
        changed = 0
        for scene, cache in caches.items():
            predictions: dict[str, np.ndarray] = {}
            proposal_views = torch.from_numpy(cache["sam_proposal_view_indices"])
            for split in SPLITS:
                updated, stats = propagate_categorical_identity_over_proposals(
                    torch.from_numpy(cache[f"scores_split_{split}"]),
                    torch.from_numpy(cache["sam_membership_rows"]),
                    torch.from_numpy(cache["sam_membership_proposals"]),
                    torch.from_numpy(cache["sam_membership_weights"]),
                    num_proposals=int(cache["sam_num_proposals"]),
                    proposal_view_indices=proposal_views,
                    **settings,
                )
                changed += int(stats["changed_rows"])
                predictions[split] = _labels(updated, split)
            scene_rows[scene] = _metrics(cache, predictions)
        macro = _macro(scene_rows)
        deltas = {
            split: {
                metric: macro[split][metric] - baseline_macro[split][metric]
                for metric in ("miou", "macc")
            }
            for split in SPLITS
        }
        variants.append(
            {
                "settings": settings,
                "macro": macro,
                "delta": deltas,
                "changed_rows_across_scene_split_pairs": changed,
            }
        )

    def objective(row: dict[str, object]) -> tuple[float, float, float, int]:
        delta = row["delta"]
        miou = [float(delta[split]["miou"]) for split in SPLITS]
        macc = [float(delta[split]["macc"]) for split in SPLITS]
        all_positive = float(all(value > 0.0 for value in miou + macc))
        return (
            all_positive,
            min(miou),
            sum(miou) + 0.25 * sum(macc),
            -int(row["changed_rows_across_scene_split_pairs"]),
        )

    selected = max(variants, key=objective)
    report = {
        "artifact_type": "radio_gs_scannet_sam_instance_consensus_dev4_screen",
        "status": "complete_development_selection_before_confirmation4_labels",
        "development_scenes": list(scenes),
        "confirmation_scenes": [
            "scene0140_00",
            "scene0347_00",
            "scene0400_00",
            "scene0590_00",
        ],
        "candidate_family": "field_identity_markers_plus_query_free_official_sam_exact_mpr_cross_view_consensus",
        "persistent_second_semantic_field": False,
        "baseline_macro": baseline_macro,
        "selected": selected,
        "selection_objective": "require all 19/15/10 mIoU and mAcc deltas positive when possible; then maximize worst mIoU delta, aggregate utility, and sparsity",
        "variant_count": len(variants),
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "baseline_macro", "selected", "variant_count")}, indent=2))


if __name__ == "__main__":
    main()
