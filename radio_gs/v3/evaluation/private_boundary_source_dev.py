"""Re-evaluate trained private checkpoints with their actual D16 boundary head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.memory.structured_memory import (
    LowRankPrivateBranchMemory,
    SharedPrivateLayout,
    StructuredSharedPrivateMemory,
)
from radio_gs.v3.training.instance_upper_bound import sha256_file, validate_source_only_inputs
from radio_gs.v3.training.run_instance_upper_bound import evaluate, load_episodes


def _model(
    architecture: str,
    initial: torch.Tensor,
    layout: SharedPrivateLayout,
) -> StructuredSharedPrivateMemory:
    types = {
        "hard_block_shared_private": StructuredSharedPrivateMemory,
        "shared_core_low_rank_private_branches": LowRankPrivateBranchMemory,
    }
    if architecture not in types:
        raise ValueError("boundary correction supports only retained private architectures")
    return types[architecture](initial, layout=layout)


@torch.no_grad()
def _load_compatible(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    compatible = {
        key: torch.as_tensor(value) for key, value in state.items()
        if key in current and torch.as_tensor(value).shape == current[key].shape
    }
    model.load_state_dict(compatible, strict=False)


def run(args: argparse.Namespace) -> dict[str, object]:
    membership_path = Path(args.membership).resolve(strict=True)
    relation_path = Path(args.relation).resolve(strict=True)
    candidate_path = Path(args.candidate).resolve(strict=True)
    legacy_report_path = Path(args.legacy_report).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu")
    relation = torch.load(relation_path, map_location="cpu")
    candidate = torch.load(candidate_path, map_location="cpu")
    legacy = json.loads(legacy_report_path.read_text())
    validate_source_only_inputs(membership, relation)
    metadata = candidate["metadata"]
    if (
        not metadata.get("source_only")
        or metadata.get("historical_field_opened")
        or metadata.get("target_rgb_opened")
        or metadata.get("benchmark_metrics_opened")
        or metadata.get("membership", {}).get("sha256") != sha256_file(membership_path)
        or metadata.get("relation", {}).get("sha256") != sha256_file(relation_path)
    ):
        raise ValueError("private checkpoint violates the fixed source-only authority")
    parent_path = Path(metadata["initialization"]["path"]).resolve(strict=True)
    if sha256_file(parent_path) != metadata["initialization"]["sha256"]:
        raise ValueError("private checkpoint parent hash differs")
    parent = torch.load(parent_path, map_location="cpu")
    layout = SharedPrivateLayout(**metadata["layout"])
    initial = torch.as_tensor(parent["state_dict"]["memory"]).float()
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    baseline_model = _model(metadata["architecture"], initial, layout)
    _load_compatible(baseline_model, parent["state_dict"])
    baseline_model = baseline_model.to(device).eval()
    baseline_head = nn.Linear(layout.boundary, 1).to(device).eval()
    candidate_model = _model(metadata["architecture"], initial, layout)
    _load_compatible(candidate_model, candidate["state_dict"])
    candidate_model = candidate_model.to(device).eval()
    candidate_head = nn.Linear(layout.boundary, 1).to(device).eval()
    candidate_head.load_state_dict({
        "weight": torch.as_tensor(candidate["state_dict"]["boundary_head.weight"]),
        "bias": torch.as_tensor(candidate["state_dict"]["boundary_head.bias"]),
    })
    episodes, supports = load_episodes(membership, relation)
    valid = [item for item in episodes if supports[item.proposal_index][0].numel()]
    training = [item for item in valid if item.view_index % 4 in (1, 2)]
    evaluation = [item for item in valid if item.view_index % 4 == 3]
    train_proposals = {item.proposal_index for item in training}
    baseline_metrics, baseline_count = evaluate(
        baseline_model, evaluation, supports, relation, train_proposals,
        args.temperature, boundary_head=baseline_head,
    )
    candidate_metrics, candidate_count = evaluate(
        candidate_model, evaluation, supports, relation, train_proposals,
        args.temperature, boundary_head=candidate_head,
    )
    if baseline_count != candidate_count:
        raise RuntimeError("corrected boundary cohorts differ")
    for name in ("mask_iou", "brier", "unknown_fp_mass"):
        if abs(candidate_metrics[name] - legacy["candidate_metrics"][name]) > 1e-7:
            raise RuntimeError("boundary correction changed an instance metric")
    report = {
        "schema": "radio_gs.sugm_v3.private_boundary_source_dev.v1",
        "scene": membership["scene"],
        "architecture": metadata["architecture"],
        "evaluation_proposals": candidate_count,
        "boundary_metric_correction": "rendered_sigmoid_of_trained_d16_boundary_head",
        "legacy_boundary_metric": "gradient_of_rendered_instance_posterior",
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "delta": {
            name: candidate_metrics[name] - baseline_metrics[name]
            for name in candidate_metrics
        },
        "instance_metrics_match_legacy": True,
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": {
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "relation": {"path": str(relation_path), "sha256": sha256_file(relation_path)},
            "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "legacy_report": {"path": str(legacy_report_path), "sha256": sha256_file(legacy_report_path)},
        },
        "status": "development_evidence_only",
    }
    write_frozen_json(Path(args.output).resolve(), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--legacy-report", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    print(run(args))


if __name__ == "__main__":
    main()
