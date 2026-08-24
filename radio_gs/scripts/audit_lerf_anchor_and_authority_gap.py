#!/usr/bin/env python3
"""Audit cross-modal anchor purity and source-episode row authority for LERF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import visible_membership_target
from radio_gs.scripts.train_lerf_query_native_joint_cross_scene_decoder import _load_scene, _split
from radio_gs.utils.immutable_artifacts import write_frozen_json


def _anchor_rows(score: torch.Tensor, xyz: torch.Tensor, topk: int, radius_fraction: float | None) -> torch.Tensor:
    peak = int(score.argmax())
    eligible = torch.ones(score.numel(), dtype=torch.bool)
    if radius_fraction is not None:
        diagonal = torch.linalg.vector_norm(xyz.max(0).values - xyz.min(0).values).clamp_min(1e-8)
        eligible = torch.linalg.vector_norm(xyz - xyz[peak], dim=1) <= float(radius_fraction) * diagonal
        eligible[peak] = True
    rows = torch.where(eligible)[0]
    return rows[score[rows].topk(min(int(topk), rows.numel())).indices]


def _anchor_metrics(rows: torch.Tensor, truth: torch.Tensor, xyz: torch.Tensor, component_radius: float) -> dict[str, float | int]:
    purity = float(truth[rows].float().mean())
    distance = torch.cdist(xyz[rows], xyz[rows])
    parent = list(range(rows.numel()))
    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]; value = parent[value]
        return value
    for left in range(rows.numel()):
        for right in range(left + 1, rows.numel()):
            if float(distance[left, right]) <= component_radius:
                a, b = find(left), find(right)
                if a != b: parent[b] = a
    return {
        "purity": purity,
        "components": len({find(index) for index in range(rows.numel())}),
        "maximum_anchor_distance": float(distance.max()) if rows.numel() else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    specs = json.loads(Path(args.scene_specs).read_text())
    spec = next((value for value in specs if value["scene"] == args.scene), None)
    if spec is None: raise ValueError(f"scene {args.scene} absent from specs")
    data = _load_scene(
        spec["seed_model"], spec["seed_model_sha256"], spec["episodes"],
        spec["episodes_sha256"], args.evaluation_membership_threshold,
        spec.get("instance_text", spec.get("generic_text")),
    )
    training, _validation, heldout = _split(
        data, args.holdout_stride, args.validation_residue, args.heldout_residue,
    )
    positive_authority = torch.zeros(data["latent"].shape[0], dtype=torch.bool)
    negative_authority = torch.zeros_like(positive_authority)
    for sample in torch.where(training)[0].tolist():
        target = int(data["episode_target"][sample])
        positive_authority[data["hard_support"][target]] = True
        negative_authority[data["negative_support"][sample]] = True
    observed = data["observed"].any(0)
    bucket = torch.full((data["latent"].shape[0],), 3, dtype=torch.long)
    bucket[observed] = 2; bucket[negative_authority] = 1; bucket[positive_authority] = 0
    radii = [float(value) for value in args.radius_fractions.split(",")]
    modes: dict[str, list[dict[str, Any]]] = {"global": []}
    for radius in radii: modes[f"peak_local_{radius:g}"] = []
    authority_truth = torch.zeros(4); authority_rows = torch.zeros(4)
    eligible = heldout & data["generic_text"][args.text_split]["eligible"]
    for sample in torch.where(eligible)[0].tolist():
        query = int(data["episode_query"][sample]); target = int(data["episode_target"][sample])
        visible = torch.where(data["observed"][int(data["views"][target])])[0]
        xyz = data["xyz"][visible]
        truth = visible_membership_target(visible, data["hard_support"][target], num_rows=data["latent"].shape[0]).bool()
        image_identity = data["baseline"][visible] @ data["semantic"][query]
        text_identity = data["baseline"][visible] @ data["generic_text"][args.text_split]["embedding"][sample]
        diagonal = float(torch.linalg.vector_norm(xyz.max(0).values - xyz.min(0).values).clamp_min(1e-8))
        for name, radius in [("global", None)] + [(f"peak_local_{value:g}", value) for value in radii]:
            image_rows = _anchor_rows(image_identity, xyz, args.topk, radius)
            text_rows = _anchor_rows(text_identity, xyz, args.topk, radius)
            image_metric = _anchor_metrics(image_rows, truth, xyz, args.component_radius_fraction * diagonal)
            text_metric = _anchor_metrics(text_rows, truth, xyz, args.component_radius_fraction * diagonal)
            intersection = len(set(image_rows.tolist()) & set(text_rows.tolist()))
            union = len(set(image_rows.tolist()) | set(text_rows.tolist()))
            modes[name].append({
                "image": image_metric, "text": text_metric,
                "image_text_anchor_jaccard": intersection / max(union, 1),
                "peak_distance": float(torch.linalg.vector_norm(xyz[int(image_identity.argmax())] - xyz[int(text_identity.argmax())])),
            })
        local_bucket = bucket[visible]
        for index in range(4):
            rows = local_bucket == index; authority_rows[index] += rows.sum(); authority_truth[index] += truth[rows].sum()
    summary: dict[str, Any] = {}
    for name, values in modes.items():
        summary[name] = {
            "episodes": len(values),
            "image_anchor_purity": sum(value["image"]["purity"] for value in values) / len(values),
            "text_anchor_purity": sum(value["text"]["purity"] for value in values) / len(values),
            "image_components": sum(value["image"]["components"] for value in values) / len(values),
            "text_components": sum(value["text"]["components"] for value in values) / len(values),
            "image_text_anchor_jaccard": sum(value["image_text_anchor_jaccard"] for value in values) / len(values),
            "peak_distance": sum(value["peak_distance"] for value in values) / len(values),
        }
    report = {
        "schema": "radio_gs.lerf_anchor_authority_gap_audit.v1", "scene": args.scene,
        "source_only": True, "benchmark_vocabulary_opened": False,
        "text_authority": "instance_text" if spec.get("instance_text") else "generic_text",
        "text_split": args.text_split, "topk": args.topk, "anchor_summary": summary,
        "training_authority": {
            "positive_rows": int(positive_authority.sum()), "negative_rows": int(negative_authority.sum()),
            "observed_rows": int(observed.sum()), "total_rows": int(observed.numel()),
            "heldout_visible_rows_by_bucket": authority_rows.tolist(),
            "heldout_positive_rows_by_bucket": authority_truth.tolist(),
            "bucket_names": ["training_positive", "training_explicit_negative", "observed_unknown", "never_observed"],
        },
    }
    write_frozen_json(Path(args.output).resolve(), report); return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--scene-specs",required=True); parser.add_argument("--scene",required=True); parser.add_argument("--output",required=True)
    parser.add_argument("--text-split",default="audit"); parser.add_argument("--topk",type=int,default=6); parser.add_argument("--radius-fractions",default="0.01,0.02,0.04,0.08"); parser.add_argument("--component-radius-fraction",type=float,default=0.02)
    parser.add_argument("--evaluation-membership-threshold",type=float,default=.5); parser.add_argument("--holdout-stride",type=int,default=4); parser.add_argument("--heldout-residue",type=int,default=3); parser.add_argument("--validation-residue",type=int,default=2)
    print(json.dumps(run(parser.parse_args()),indent=2,sort_keys=True))


if __name__=="__main__": main()
