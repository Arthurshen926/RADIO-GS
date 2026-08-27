#!/usr/bin/env python3
"""Shardable target-driven source-dev Gaussian posterior ceiling."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.evaluation.mpr_geometry_oracle import (
    ExactMembership,
    exact_mpr_membership,
    optimize_same_view_posterior,
    positive_hit_coverage,
    render_exact_membership,
    score_oracle_prediction,
)
from radio_gs.v3.training.instance_upper_bound import sha256_file, validate_source_only_inputs
from radio_gs.v3.training.run_instance_upper_bound import load_episodes


def run(args: argparse.Namespace) -> dict[str, object]:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("posterior oracle shard identity differs")
    membership_path = Path(args.membership).resolve(strict=True)
    relation_path = Path(args.relation).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu")
    relation = torch.load(relation_path, map_location="cpu")
    validate_source_only_inputs(membership, relation)
    episodes, _supports = load_episodes(membership, relation)
    dev = [
        item for item in episodes
        if item.view_index % 4 == 3
        and bool((item.known & ~item.target).any())
        and item.proposal_index % args.num_shards == args.shard_index
    ]
    device = torch.device(args.device)
    records = []
    for episode in dev:
        initial = exact_mpr_membership(episode, int(membership["num_rows"]))
        dense, loss = optimize_same_view_posterior(
            initial, episode, device=device, steps=args.steps,
            learning_rate=args.learning_rate,
        )
        optimized = ExactMembership(dense, initial.observed, initial.semantic_mass)
        prediction = render_exact_membership(optimized, episode)
        metrics = score_oracle_prediction(prediction, episode)
        records.append({
            "proposal_index": episode.proposal_index,
            "view_index": episode.view_index,
            "scale": episode.scale,
            "positive_pixels": int(episode.target.sum()),
            "known_negative_pixels": int((episode.known & ~episode.target).sum()),
            "best_loss": loss,
            "metrics": asdict(metrics),
            "positive_hit_coverage": positive_hit_coverage(optimized, episode),
        })
    if not records:
        raise ValueError("posterior oracle shard contains no proper dev episode")
    report = {
        "schema": "radio_gs.sugm_v3.same_view_posterior_oracle.v1",
        "evaluation_split": "dev",
        "source_audit_opened": False,
        "benchmark_data_opened": False,
        "diagnostic_only_target_mask_optimized": True,
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "optimization": {"steps": args.steps, "learning_rate": args.learning_rate},
        "inputs": {
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "relation": {"path": str(relation_path), "sha256": sha256_file(relation_path)},
        },
        "records": records,
    }
    write_frozen_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
