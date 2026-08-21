#!/usr/bin/env python3
"""Frozen paper-scene audit for source-only cross-view ScanNet object tracks."""

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
from radio_gs.querying.source_multiview_object_tracks import (
    build_source_multiview_object_tracks,
)
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS
from radio_gs.scripts.evaluate_scannet_object_aware_category_vote_cpu import _metrics


SCENES = ("scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00")
SPLITS = ("19", "15", "10")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--additional-cache-root", type=Path, action="append", default=[])
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument("--development-scenes", default="scene0000_00")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(int(args.cpu_threads))
    torch.set_num_interop_threads(1)
    scenes = tuple(value.strip() for value in args.scenes.split(",") if value.strip())
    development = tuple(
        value.strip() for value in args.development_scenes.split(",") if value.strip()
    )
    if not scenes or not development or not set(development).issubset(scenes):
        raise ValueError("development scenes must be a non-empty subset of scenes")
    confirmation = tuple(scene for scene in scenes if scene not in set(development))
    cache_roots = (args.cache_root, *tuple(args.additional_cache_root))
    results: dict[str, object] = {}
    for scene in scenes:
        candidates = [
            root / scene / "development" / f"{scene}_scores.npz"
            for root in cache_roots
        ]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) != 1:
            raise ValueError(f"{scene} must resolve to exactly one score cache")
        path = existing[0]
        payload = np.load(path, allow_pickle=False)
        tracks = build_source_multiview_object_tracks(
            torch.from_numpy(payload["sam_membership_rows"]),
            torch.from_numpy(payload["sam_membership_proposals"]),
            torch.from_numpy(payload["sam_membership_weights"]),
            torch.from_numpy(payload["sam_proposal_view_indices"]),
            num_rows=len(payload["pseudo_labels"]),
            num_proposals=int(payload["sam_num_proposals"]),
            minimum_soft_cosine=0.5,
        )
        split_results: dict[str, object] = {}
        for split in SPLITS:
            ids = tuple(int(value) for value in OPENGAUSSIAN_NYU40_CLASS_SPLITS[split])
            scores = torch.from_numpy(payload[f"scores_split_{split}"])
            baseline_labels = np.asarray(ids)[scores.argmax(1).numpy()]
            candidate, stats = object_aware_category_vote(
                scores,
                tracks.row_indices,
                tracks.track_indices,
                tracks.membership_weights,
                num_proposals=tracks.num_tracks,
                class_ids=ids,
                strength=1.0,
                residual_budget=0.25,
            )
            candidate_labels = np.asarray(ids)[candidate.argmax(1).numpy()]
            baseline = _metrics(
                payload["pseudo_labels"], baseline_labels, payload["significance"], ids
            )
            voted = _metrics(
                payload["pseudo_labels"], candidate_labels, payload["significance"], ids
            )
            split_results[split] = {
                "baseline": baseline,
                "candidate": voted,
                "delta": {key: voted[key] - baseline[key] for key in ("miou", "macc")},
                "voting": stats,
            }
        results[scene] = {
            "cache": {"path": str(path), "sha256": _sha256(path)},
            "tracks": tracks.stats,
            "splits": split_results,
        }

    macro: dict[str, object] = {}
    for cohort, cohort_scenes in (("development", development), ("confirmation", confirmation)):
        macro[cohort] = {
            split: {
                kind: {
                    metric: float(
                        np.mean(
                            [
                                results[scene]["splits"][split][kind][metric]
                                for scene in cohort_scenes
                            ]
                        )
                    )
                    for metric in ("miou", "macc")
                }
                for kind in ("baseline", "candidate", "delta")
            }
            for split in SPLITS
        }
    split10_miou_regressions = [
        scene
        for scene in scenes
        if results[scene]["splits"]["10"]["delta"]["miou"] < 0.0
    ]
    split10_macc_regressions = [
        scene
        for scene in scenes
        if results[scene]["splits"]["10"]["delta"]["macc"] < 0.0
    ]
    report = {
        "schema_version": 1,
        "artifact_type": "radio_gs_scannet_source_multiview_object_tracks_pilot",
        "status": "development_and_prefixed_confirmation_complete_not_promoted",
        "authority": {
            "association_inputs": "official source-view SAM exact-MPR Gaussian memberships and source view IDs only",
            "association": "different-view reciprocal-best soft-cosine overlap >= 0.5; strongest-first union with unique-view constraint",
            "track_membership": "clipped union of source proposal memberships",
            "category_authority": "primitive RADIO logits only after query-independent tracks are frozen",
            "class_permutation_equivariant_with_thing_stuff_roles": True,
            "proposal_permutation_equivariant": True,
            "gt_or_target_metric_used_for_association": False,
            "development_scenes": list(development),
            "confirmation_scenes_prefixed": list(confirmation),
            "per_class_parameters": False,
            "unknown_background_branch": False
        },
        "results": results,
        "macro": macro,
        "decision": {
            "promotion": False,
            "reason": "confirmation macro improves all 19/15/10 mIoU and mAcc metrics, but the frozen rule still causes per-scene split10 regressions and therefore fails the no-regression safety gate",
            "split10_miou_regression_scenes": split10_miou_regressions,
            "split10_macc_regression_scenes": split10_macc_regressions,
            "stop_rule": "do not add a post-metric class-count fallback or tune overlap/residual thresholds; stop the analytic track-voting branch",
            "next_bridge": "replace reciprocal mask-overlap tracks by the learned compact object-affinity head while retaining this exact typed-voting interface"
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "macro": macro}, indent=2))


if __name__ == "__main__":
    main()
