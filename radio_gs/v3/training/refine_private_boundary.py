"""Repair the D16 boundary capability while freezing retained D448+D48 state."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn

from radio_gs.utils.immutable_artifacts import write_frozen_json, write_torch_noclobber
from radio_gs.v3.memory.structured_memory import LowRankPrivateBranchMemory, SharedPrivateLayout
from radio_gs.v3.training.instance_upper_bound import sha256_file, validate_source_only_inputs
from radio_gs.v3.training.run_instance_upper_bound import evaluate, load_episodes
from radio_gs.v3.training.run_structured_source_mapping import (
    boundary_objective,
    has_boundary_authority,
    protected_block_deltas,
)


def _compatible_state(model: nn.Module, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    current = model.state_dict()
    return {
        key: torch.as_tensor(value) for key, value in state.items()
        if key in current and torch.as_tensor(value).shape == current[key].shape
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    candidate_path = Path(args.candidate).resolve(strict=True)
    membership_path = Path(args.membership).resolve(strict=True)
    relation_path = Path(args.relation).resolve(strict=True)
    candidate = torch.load(candidate_path, map_location="cpu")
    membership = torch.load(membership_path, map_location="cpu")
    relation = torch.load(relation_path, map_location="cpu")
    validate_source_only_inputs(membership, relation)
    metadata = candidate["metadata"]
    if (
        metadata.get("architecture") != "shared_core_low_rank_private_branches"
        or not metadata.get("source_only")
        or metadata.get("historical_field_opened")
        or metadata.get("target_rgb_opened")
        or metadata.get("benchmark_metrics_opened")
        or metadata.get("membership", {}).get("sha256") != sha256_file(membership_path)
        or metadata.get("relation", {}).get("sha256") != sha256_file(relation_path)
    ):
        raise ValueError("boundary refinement requires the retained source-only low-rank candidate")
    parent_path = Path(metadata["initialization"]["path"]).resolve(strict=True)
    if sha256_file(parent_path) != metadata["initialization"]["sha256"]:
        raise ValueError("boundary refinement parent hash differs")
    parent = torch.load(parent_path, map_location="cpu")
    layout = SharedPrivateLayout(**metadata["layout"])
    initial = torch.as_tensor(candidate["state_dict"]["memory"]).float()
    parent_memory = torch.as_tensor(parent["state_dict"]["memory"]).float()
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    model = LowRankPrivateBranchMemory(initial, layout=layout)
    model.load_state_dict(_compatible_state(model, candidate["state_dict"]), strict=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = model.to(device)
    model.enable_owned_training_blocks("boundary")
    model.boundary_down.reset_parameters()
    with torch.no_grad():
        model.owned_training_parameter("boundary").copy_(
            parent_memory[:, layout.slices["boundary"]].to(device)
        )
        model.boundary_up.weight.zero_()
    branch_parameters: tuple[nn.Parameter, ...]
    if args.boundary_architecture == "low_rank":
        model.boundary_down.weight.requires_grad_(True)
        model.boundary_up.weight.requires_grad_(True)
        branch_parameters = (model.boundary_down.weight, model.boundary_up.weight)
    else:
        branch_parameters = ()
    boundary_head = nn.Linear(layout.boundary, 1).to(device)
    nn.init.zeros_(boundary_head.weight)
    nn.init.zeros_(boundary_head.bias)
    with torch.no_grad():
        boundary_head.weight[0, 0] = 1.0
    trainable = (
        model.owned_training_parameter("boundary"),
        *branch_parameters,
        *boundary_head.parameters(),
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    episodes, supports = load_episodes(membership, relation)
    valid = [item for item in episodes if supports[item.proposal_index][0].numel()]
    training = [
        item for item in valid
        if item.view_index % 4 in (1, 2) and has_boundary_authority(item)
    ]
    if not training:
        raise ValueError("source training split has no valid boundary authority")
    evaluation = [item for item in valid if item.view_index % 4 == 3]
    train_proposals = {item.proposal_index for item in training}
    baseline, baseline_count = evaluate(
        model, evaluation, supports, relation, train_proposals, args.temperature,
        boundary_head=boundary_head,
    )
    rng = random.Random(args.seed)
    best_loss = float("inf")
    best_model = None
    best_head = None
    total_loss = 0.0
    for step in range(args.steps):
        selected = rng.sample(training, k=min(args.episodes_per_step, len(training)))
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for episode in selected:
            loss = boundary_objective(model, boundary_head, episode)
            losses.append(float(loss.detach()))
            (loss / len(selected)).backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        value = sum(losses) / len(losses)
        total_loss += value
        snapshot = (step + 1) % args.snapshot_interval == 0 or step + 1 == args.steps
        if snapshot:
            print({"step": step + 1, "balanced_boundary_loss": value}, flush=True)
        if snapshot and value < best_loss:
            best_loss = value
            best_model = {
                key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()
            }
            best_head = {
                key: tensor.detach().cpu().clone() for key, tensor in boundary_head.state_dict().items()
            }
    if best_model is None or best_head is None:
        raise RuntimeError("boundary refinement produced no checkpoint")
    model.load_state_dict(best_model)
    boundary_head.load_state_dict(best_head)
    corrected, corrected_count = evaluate(
        model, evaluation, supports, relation, train_proposals, args.temperature,
        boundary_head=boundary_head,
    )
    deployed = model.deployment_memory().cpu()
    deltas = protected_block_deltas(initial, deployed, layout)
    if any(deltas[name] != 0.0 for name in ("shared", "semantic", "instance")):
        raise RuntimeError("boundary refinement rewrote a protected block")
    boundary_std = float(deployed[:, layout.slices["boundary"]].std())
    head_weight_norm = float(best_head["weight"].norm())
    branch_up_norm = float(best_model["boundary_up.weight"].norm())
    if boundary_std <= 1e-6 or head_weight_norm <= 1e-6:
        raise RuntimeError("boundary refinement collapsed to a constant predictor")
    state = {
        key: value for key, value in best_model.items()
        if not key.startswith("_owned_training_blocks.")
    }
    carried = {
        key: value for key, value in candidate["state_dict"].items()
        if key.startswith("visual_codec.") or key.startswith("codec.")
    }
    output = Path(args.output).resolve()
    payload = {
        "schema": candidate["schema"],
        "state_dict": {
            **state, **carried, "memory": deployed,
            "boundary_head.weight": best_head["weight"],
            "boundary_head.bias": best_head["bias"],
        },
        "metadata": {
            **metadata,
            "phase_order": "protected_low_rank_instance_then_balanced_boundary_refinement",
            "boundary_refinement": {
                "architecture": args.boundary_architecture,
                "objective": "class_balanced_bce_plus_soft_dice",
                "head_initialization": "fixed_unit_axis_0",
                "boundary_branch_initialization": "fresh_down_zero_up",
                "boundary_d16_initialization": "sealed_parent_boundary_block",
                "shared_semantic_instance_frozen": True,
                "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            },
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "schema": "radio_gs.sugm_v3.private_boundary_refinement.report.v1",
        "scene": membership["scene"],
        "boundary_architecture": args.boundary_architecture,
        "steps": args.steps,
        "best_train_loss": best_loss,
        "mean_train_loss": total_loss / args.steps,
        "baseline_evaluation_proposals": baseline_count,
        "boundary_training_proposals": len(training),
        "evaluation_proposals": corrected_count,
        "baseline_metrics": baseline,
        "candidate_metrics": corrected,
        "delta": {name: corrected[name] - baseline[name] for name in corrected},
        "protected_block_max_abs_delta": deltas,
        "nondegeneracy": {
            "boundary_d16_std": boundary_std,
            "head_weight_norm": head_weight_norm,
            "boundary_branch_up_norm": branch_up_norm,
            "pass": True,
        },
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "checkpoint": {"path": str(output), "sha256": sha256_file(output)},
        "status": "development_evidence_only",
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--boundary-architecture", choices=("hard", "low_rank"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--episodes-per-step", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--snapshot-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.steps, args.episodes_per_step, args.snapshot_interval) <= 0:
        raise ValueError("boundary refinement budgets must be positive")
    print(run(args))


if __name__ == "__main__":
    main()
