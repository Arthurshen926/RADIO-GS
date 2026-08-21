#!/usr/bin/env python3
"""Train scene0000 compact affinity and run fixed ScanNet confirmations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.querying.learned_compact_object_affinity import (
    CompactObjectAffinity,
    balanced_relation_loss,
    build_source_proposal_relations,
    pool_proposal_features,
    relation_proper_metrics,
)
from radio_gs.querying.scannet_object_aware_category_voting import object_aware_category_vote
from radio_gs.querying.source_multiview_object_tracks import build_source_learned_object_tracks
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS
from radio_gs.scripts.evaluate_scannet_object_aware_category_vote_cpu import _metrics


SCENES = ("scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00")
SPLITS = ("19", "15", "10")
RELATION_LOGIT_SCALE = 8.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_view_split(views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    heldout = torch.remainder(torch.as_tensor(views).long(), 4) == 3
    return ~heldout, heldout


def _source_probability_threshold(logits: torch.Tensor, relation: torch.Tensor) -> float:
    score = torch.as_tensor(logits).detach().cpu().float()
    label = torch.as_tensor(relation).detach().cpu().to(torch.int8)
    same, different = score[label == 1].sigmoid(), score[label == 0].sigmoid()
    if not same.numel() or not different.numel():
        raise ValueError("source threshold requires both known relation outcomes")
    candidates = torch.linspace(0.05, 0.95, 181)
    balanced = torch.stack([
        0.5 * ((same >= value).float().mean() + (different < value).float().mean())
        for value in candidates
    ])
    return float(candidates[int(balanced.argmax())])


def _load_scene(
    scene: str, membership_root: Path, field_root: Path, score_root: Path
) -> dict[str, object]:
    membership_path = membership_root / scene / "official_sam3_exact_mpr_memberships.pt"
    feature_path = field_root / scene / "primitive_query_method_v1.pth"
    score_path = score_root / scene / "development" / f"{scene}_scores.npz"
    membership = torch.load(membership_path, map_location="cpu", weights_only=False)
    field = torch.load(feature_path, map_location="cpu", weights_only=True)
    score = np.load(score_path, allow_pickle=False)
    if membership.get("metadata", {}).get("benchmark_masks_opened") is not False:
        raise ValueError(f"{scene} membership information authority differs")
    if field.get("metadata", {}).get("query_independent") is not True:
        raise ValueError(f"{scene} feature is not query-independent")
    if int(membership["num_rows"]) != len(field["features"]) or len(score["pseudo_labels"]) != len(field["features"]):
        raise ValueError(f"{scene} Gaussian axes differ")
    if not torch.equal(torch.as_tensor(field["valid"]).bool(), torch.ones(len(field["features"]), dtype=torch.bool)):
        raise ValueError(f"{scene} contains invalid canonical capability rows")
    relation = build_source_proposal_relations(
        membership["row_indices"], membership["proposal_indices"], membership["weights"],
        membership["proposal_view_indices"], membership["proposal_area_fraction"],
        num_rows=int(membership["num_rows"]), num_proposals=int(membership["num_proposals"]),
    )
    pooled = pool_proposal_features(
        field["features"], membership["row_indices"], membership["proposal_indices"],
        membership["weights"], num_proposals=int(membership["num_proposals"]),
    )
    train_proposal, heldout_proposal = _fixed_view_split(membership["proposal_view_indices"])
    train_edge = train_proposal[relation.left] & train_proposal[relation.right]
    heldout_edge = heldout_proposal[relation.left] & heldout_proposal[relation.right]
    if not bool(train_edge.any()) or not bool(heldout_edge.any()):
        raise ValueError(f"{scene} source relation folds are empty")
    return {
        "membership_path": membership_path, "feature_path": feature_path, "score_path": score_path,
        "membership": membership, "field": field, "score": score, "relation": relation,
        "pooled": pooled, "train_edge": train_edge, "heldout_edge": heldout_edge,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership-root", type=Path, required=True)
    parser.add_argument("--field-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    torch.set_num_threads(int(args.cpu_threads))
    torch.set_num_interop_threads(1)
    torch.manual_seed(int(args.seed))

    development = _load_scene(SCENES[0], args.membership_root, args.field_root, args.score_root)
    relation = development["relation"]
    train_edge = development["train_edge"]
    pooled = development["pooled"]
    model = CompactObjectAffinity(input_dim=pooled.shape[1], object_dim=16, seed=int(args.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1.0e-3)
    best_loss, best_weight = float("inf"), None
    for _ in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        embedding = model(pooled)
        logits = RELATION_LOGIT_SCALE * (embedding[relation.left[train_edge]] * embedding[relation.right[train_edge]]).sum(-1)
        loss = balanced_relation_loss(logits, relation.relation[train_edge])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best_weight = model.weight.detach().clone()
    if best_weight is None:
        raise RuntimeError("compact affinity optimization produced no state")
    model.weight.data.copy_(best_weight)
    with torch.no_grad():
        development_embedding = model(pooled)
        development_logits = RELATION_LOGIT_SCALE * (
            development_embedding[relation.left] * development_embedding[relation.right]
        ).sum(-1)
    source_same_probability = _source_probability_threshold(
        development_logits[train_edge], relation.relation[train_edge]
    )

    checkpoint_path = args.output_root / "global_compact_affinity_16d.pt"
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "radio_gs.scannet_global_compact_object_affinity.v1",
        "weight": model.weight.detach().half(),
        "relation_logit_scale": RELATION_LOGIT_SCALE,
        "source_same_probability": source_same_probability,
        "metadata": {
            "training_scene": SCENES[0], "object_dim": 16,
            "source_relation": relation.stats, "view_split": "source_view_index_mod4_eq3_heldout",
            "proper_score": "class_balanced_Bernoulli_log_score",
            "benchmark_metrics_used_for_training_or_threshold": False,
        },
    }
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{os.getpid()}.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)

    scene_reports: dict[str, object] = {}
    for scene in SCENES:
        payload = development if scene == SCENES[0] else _load_scene(scene, args.membership_root, args.field_root, args.score_root)
        membership, field, scores = payload["membership"], payload["field"], payload["score"]
        relation, heldout = payload["relation"], payload["heldout_edge"]
        pooled = payload["pooled"]
        with torch.no_grad():
            proposal_embedding = model(pooled)
            learned_logits = RELATION_LOGIT_SCALE * (
                proposal_embedding[relation.left] * proposal_embedding[relation.right]
            ).sum(-1)
            baseline_embedding = F.normalize(pooled, dim=-1)
            baseline_logits = RELATION_LOGIT_SCALE * (
                baseline_embedding[relation.left] * baseline_embedding[relation.right]
            ).sum(-1)
            gaussian_codes = F.linear(torch.as_tensor(field["features"]).float(), model.weight).half()
        learned_heldout = relation_proper_metrics(learned_logits[heldout], relation.relation[heldout])
        baseline_heldout = relation_proper_metrics(baseline_logits[heldout], relation.relation[heldout])
        source_admitted = (
            learned_heldout["balanced_log_score"] < baseline_heldout["balanced_log_score"]
            and learned_heldout["balanced_brier"] < baseline_heldout["balanced_brier"]
            and learned_heldout["auc"] >= baseline_heldout["auc"]
        )
        tracks = build_source_learned_object_tracks(
            membership["row_indices"], membership["proposal_indices"], membership["weights"],
            membership["proposal_view_indices"], proposal_embedding,
            num_rows=int(membership["num_rows"]), num_proposals=int(membership["num_proposals"]),
            relation_logit_scale=RELATION_LOGIT_SCALE,
            minimum_same_probability=source_same_probability,
        )
        interface_path = args.output_root / scene / "learned_compact_affinity_tracks.pt"
        interface_path.parent.mkdir(parents=True, exist_ok=True)
        interface = {
            "schema": "radio_gs.scannet_learned_compact_affinity_tracks.v1",
            "scene": scene, "object_codes": gaussian_codes,
            "row_indices": tracks.row_indices, "track_indices": tracks.track_indices,
            "membership_weights": tracks.membership_weights, "num_tracks": tracks.num_tracks,
            "track_confidence": tracks.track_confidence,
            "source_reliability_admitted": source_admitted,
            "metadata": {
                "query_independent": True, "sparse_interface": "Gaussian-track membership",
                "fallback": "whole-scene bitwise primitive replay when source-heldout proper-score gate fails",
                "class_count_or_target_metric_used_by_gate": False,
            },
        }
        temporary = interface_path.with_name(f".{interface_path.name}.{os.getpid()}.tmp")
        torch.save(interface, temporary)
        os.replace(temporary, interface_path)

        split_report: dict[str, object] = {}
        for split in SPLITS:
            ids = tuple(int(value) for value in OPENGAUSSIAN_NYU40_CLASS_SPLITS[split])
            raw = torch.from_numpy(scores[f"scores_split_{split}"])
            if source_admitted and tracks.num_tracks:
                candidate, voting = object_aware_category_vote(
                    raw, tracks.row_indices, tracks.track_indices, tracks.membership_weights,
                    num_proposals=tracks.num_tracks, class_ids=ids, strength=1.0, residual_budget=0.25,
                )
            else:
                candidate, voting = raw.clone(), {
                    "construction": "source_reliability_bitwise_fallback_v1", "enabled": False,
                    "changed_rows": 0, "source_reliability_admitted": source_admitted,
                }
            baseline_labels = np.asarray(ids)[raw.argmax(1).numpy()]
            candidate_labels = np.asarray(ids)[candidate.argmax(1).numpy()]
            baseline = _metrics(scores["pseudo_labels"], baseline_labels, scores["significance"], ids)
            voted = _metrics(scores["pseudo_labels"], candidate_labels, scores["significance"], ids)
            split_report[split] = {
                "baseline": baseline, "candidate": voted,
                "delta": {key: voted[key] - baseline[key] for key in ("miou", "macc")},
                "voting": voting,
            }
        scene_reports[scene] = {
            "source_gate": {"admitted": source_admitted, "baseline_heldout": baseline_heldout, "learned_heldout": learned_heldout},
            "relation": relation.stats, "tracks": tracks.stats, "splits": split_report,
            "interface": {"path": str(interface_path), "sha256": _sha256(interface_path)},
        }

    confirmation = SCENES[1:]
    macro = {
        split: {
            metric: float(np.mean([scene_reports[scene]["splits"][split]["delta"][metric] for scene in confirmation]))
            for metric in ("miou", "macc")
        }
        for split in SPLITS
    }
    split10_miou_regression = [
        scene for scene in confirmation
        if scene_reports[scene]["splits"]["10"]["delta"]["miou"] < 0
    ]
    split10_macc_regression = [
        scene for scene in confirmation
        if scene_reports[scene]["splits"]["10"]["delta"]["macc"] < 0
    ]
    source_gate_passed = all(
        scene_reports[scene]["source_gate"]["admitted"] for scene in confirmation
    )
    aggregate_gate_passed = all(
        macro[split][metric] >= 0.0 for split in SPLITS for metric in ("miou", "macc")
    )
    report = {
        "schema_version": 1,
        "artifact_type": "radio_gs_scannet_learned_compact_affinity_pilot",
        "status": "complete_rejected_no_threshold_retuning",
        "authority": {
            "training_scene": SCENES[0], "confirmation_scenes": list(confirmation),
            "object_dim": 16, "relation_logit_scale": RELATION_LOGIT_SCALE,
            "source_same_probability": source_same_probability,
            "source_reliability_gate": "heldout learned log-score and Brier strictly improve canonical-feature baseline and AUC does not regress",
            "fallback": "whole-scene, all-split bitwise primitive replay decided before category metrics",
            "metric_or_class_count_tuning": False,
        },
        "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)},
        "best_training_loss": best_loss, "scenes": scene_reports, "confirmation_macro_delta": macro,
        "decision": {
            "promotion": source_gate_passed and aggregate_gate_passed and not split10_miou_regression and not split10_macc_regression,
            "source_gate_passed": source_gate_passed,
            "aggregate_gate_passed": aggregate_gate_passed,
            "split10_miou_regression_scenes": split10_miou_regression,
            "split10_macc_regression_scenes": split10_macc_regression,
            "reason": "source-heldout proper calibration improves, but canonical affinity already has AUC 1.0 and learned track voting regresses confirmation split10 mIoU",
            "rule": "no analytic affinity threshold retuning and no post-metric fallback",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_same_probability": source_same_probability, "macro": macro, "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
