#!/usr/bin/env python3
"""Compile source-only beneficial/harmful/neutral ScanNet adoption labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import _load_mapping
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json, write_torch_noclobber

SPLITS = ("19", "15", "10")


def compile_labels(scores: dict[str, Any], significance: torch.Tensor, minimum_teacher_margin: float) -> dict[str, Any]:
    outputs: dict[str, torch.Tensor] = {}
    counts: dict[str, dict[str, int]] = {}
    xyz = torch.as_tensor(scores["xyz"]).float()
    if significance.shape != (xyz.shape[0],):
        raise ValueError("risk-label metric row domain differs")
    for split in SPLITS:
        baseline = torch.as_tensor(scores[f"baseline_scores_split_{split}"]).float()
        candidate = torch.as_tensor(scores[f"raw_candidate_scores_split_{split}"]).float()
        teacher = torch.as_tensor(scores[f"teacher_scores_split_{split}"]).float()
        if baseline.shape != candidate.shape or baseline.shape != teacher.shape or baseline.shape[0] != xyz.shape[0]:
            raise ValueError(f"risk-label score domain differs for split {split}")
        teacher_top2 = teacher.topk(2, dim=1).values
        teacher_margin = teacher_top2[:, 0] - teacher_top2[:, 1]
        truth = teacher.argmax(1); base_class = baseline.argmax(1); candidate_class = candidate.argmax(1)
        confident = teacher_margin >= minimum_teacher_margin
        beneficial = confident & (base_class != truth) & (candidate_class == truth)
        harmful = confident & (base_class == truth) & (candidate_class != truth)
        label = torch.full((xyz.shape[0],), 2, dtype=torch.long)
        label[beneficial] = 0; label[harmful] = 1
        baseline_top2 = baseline.topk(2, dim=1).values; candidate_top2 = candidate.topk(2, dim=1).values
        baseline_margin = baseline_top2[:, 0] - baseline_top2[:, 1]
        candidate_margin = candidate_top2[:, 0] - candidate_top2[:, 1]
        base_probability = F.softmax(baseline, dim=1); candidate_probability = F.softmax(candidate, dim=1)
        base_entropy = -(base_probability * base_probability.clamp_min(1e-8).log()).sum(1)
        candidate_entropy = -(candidate_probability * candidate_probability.clamp_min(1e-8).log()).sum(1)
        features = torch.stack((
            baseline_margin, candidate_margin, candidate_margin - baseline_margin,
            (candidate - baseline).square().mean(1).sqrt(), base_entropy,
            candidate_entropy, (base_class != candidate_class).float(), teacher_margin,
            significance.clamp_min(1e-12).log(),
        ), dim=1)
        outputs[f"features_split_{split}"] = features.float().contiguous()
        outputs[f"labels_split_{split}"] = label
        counts[split] = {"beneficial": int(beneficial.sum()), "harmful": int(harmful.sum()), "neutral": int((label == 2).sum())}
    return {
        "schema": "radio_gs.scannet_counterfactual_risk_labels.v1", "schema_version": 1,
        "xyz": xyz, "significance": significance.float().contiguous(), **outputs,
        "metadata": {
            "source_only": True, "benchmark_labels_opened": False,
            "benchmark_masks_opened": False, "evaluation_rgb_opened": False,
            "decoder_frozen": True, "neutral_excluded_from_binary_adoption_loss": True,
            "label_semantics": {"0": "beneficial", "1": "harmful", "2": "neutral_or_teacher_ambiguous"},
            "minimum_teacher_margin": float(minimum_teacher_margin), "counts": counts,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True); p.add_argument("--score-cache", required=True)
    p.add_argument("--metric-weights", required=True); p.add_argument("--output", required=True)
    p.add_argument("--minimum-teacher-margin", type=float, default=.02)
    args = p.parse_args()
    score_record = file_record(args.score_cache); metric_record = file_record(args.metric_weights)
    scores, _ = _load_mapping(args.score_cache, score_record["sha256"], "frozen counterfactual scores")
    metric, _ = _load_mapping(args.metric_weights, metric_record["sha256"], "query-independent metric weights")
    if scores.get("metadata", {}).get("contains_frozen_counterfactual_triplet") is not True:
        raise ValueError("score cache lacks pre-selection counterfactual triplet")
    if metric.get("metadata", {}).get("query_independent") is not True:
        raise ValueError("risk-label weights are not query independent")
    if not torch.equal(torch.as_tensor(scores["xyz"]).float(), torch.as_tensor(metric["xyz"]).float()):
        raise ValueError("risk-label xyz authority differs")
    payload = compile_labels(scores, torch.as_tensor(metric["significance"]).float(), args.minimum_teacher_margin)
    payload["scene"] = args.scene; payload["metadata"]["inputs"] = {"scores": score_record, "metric_weights": metric_record}
    output = Path(args.output).expanduser().resolve(); write_torch_noclobber(output, payload)
    report = {"status": "complete", "scene": args.scene, "counts": payload["metadata"]["counts"], "output": file_record(output)}
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report); print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__": main()
