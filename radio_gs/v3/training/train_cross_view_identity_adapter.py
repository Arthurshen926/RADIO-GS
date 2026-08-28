"""Fit one cross-scene, constant-size SUGM-v3 identity metric on source views."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.v3.query.identity_adapter import LowRankIdentityAdapter
from radio_gs.v3.query.interface import load_query_interface
from radio_gs.v3.training.instance_upper_bound import (
    proposal_supports,
    sha256_file,
    validate_source_only_inputs,
)


def _parse_scene(value: str) -> tuple[Path, Path, Path, Path]:
    parts = value.split("::")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "scene must be STATE::MEMBERSHIP::RELATION::SIGLIP"
        )
    return tuple(Path(item).resolve(strict=True) for item in parts)  # type: ignore[return-value]


@torch.no_grad()
def _scene_pairs(
    paths: tuple[Path, Path, Path, Path], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    state_path, membership_path, relation_path, siglip_path = paths
    membership = torch.load(membership_path, map_location="cpu")
    relation = torch.load(relation_path, map_location="cpu")
    siglip = torch.load(siglip_path, map_location="cpu")
    validate_source_only_inputs(membership, relation)
    interface = load_query_interface(state_path, device=device)
    descriptors = torch.as_tensor(siglip["descriptors"]).float().to(device)
    count = int(membership["num_proposals"])
    if descriptors.shape != (count, 1536):
        raise ValueError("identity teacher proposal axis differs")
    query = F.normalize(
        (descriptors - interface.siglip_mean) @ interface.siglip_basis,
        dim=-1,
        eps=1e-8,
    )
    supports = proposal_supports(
        membership["row_indices"], membership["proposal_indices"],
        membership["weights"], count,
    )
    semantic = interface.model.semantic_view()
    prototypes = []
    valid = torch.zeros(count, dtype=torch.bool, device=device)
    for index, (rows, weights) in enumerate(supports):
        if rows.numel():
            value = (semantic[rows.to(device)] * weights.to(device)[:, None]).sum(0)
            prototypes.append(F.normalize(value, dim=0, eps=1e-8))
            valid[index] = True
        else:
            prototypes.append(torch.zeros(128, device=device))
    prototype = torch.stack(prototypes)
    views = torch.as_tensor(membership["proposal_view_indices"], device=device).long()
    train = ((views % 4 == 1) | (views % 4 == 2)) & valid
    left = torch.as_tensor(relation["edge_left"], device=device).long()
    right = torch.as_tensor(relation["edge_right"], device=device).long()
    labels = torch.as_tensor(relation["edge_relation"], device=device).to(torch.int8)
    selected = train[left] & train[right] & (views[left] != views[right]) & (labels >= 0)
    left, right, labels = left[selected], right[selected], labels[selected]
    positive: dict[int, list[int]] = {}
    negative: dict[int, list[int]] = {}
    for a, b, label in zip(left.tolist(), right.tolist(), labels.tolist()):
        destination = positive if label == 1 else negative
        destination.setdefault(a, []).append(b)
        destination.setdefault(b, []).append(a)
    triple_query, triple_positive, triple_negative = [], [], []
    for source in sorted(set(positive) & set(negative)):
        positives = sorted(set(positive[source]))
        negatives = sorted(set(negative[source]))
        for offset, target in enumerate(positives):
            triple_query.append(query[source])
            triple_positive.append(prototype[target])
            triple_negative.append(prototype[negatives[offset % len(negatives)]])
    if not triple_query:
        raise ValueError("source train relation cohort lacks both classes")
    inputs = torch.stack(triple_query)
    targets = torch.stack(triple_positive)
    negatives = torch.stack(triple_negative)
    receipt = {
        "scene": membership["scene"],
        "ranking_triples": int(inputs.shape[0]),
        "queries_with_positive_and_negative_authority": len(set(positive) & set(negative)),
        "inputs": {
            "scene_state": {"path": str(state_path), "sha256": sha256_file(state_path)},
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "relation": {"path": str(relation_path), "sha256": sha256_file(relation_path)},
            "siglip_teacher": {"path": str(siglip_path), "sha256": sha256_file(siglip_path)},
        },
    }
    return inputs, targets, negatives, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", type=_parse_scene, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.scene) < 2:
        raise ValueError("cross-view adapter requires at least two source scenes")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cohorts = [_scene_pairs(paths, device) for paths in args.scene]
    adapter = LowRankIdentityAdapter(rank=args.rank).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    final_loss = 0.0
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for inputs, targets, negatives, _ in cohorts:
            count = min(args.batch_size, inputs.shape[0])
            indices = torch.randint(
                inputs.shape[0], (count,), generator=generator, device=device
            )
            mapped = adapter(inputs[indices])
            positive_score = (mapped * targets[indices]).sum(-1)
            negative_score = (mapped * negatives[indices]).sum(-1)
            ranking = F.softplus((negative_score - positive_score) / 0.07).mean()
            preservation = (1.0 - (mapped * inputs[indices]).sum(-1)).mean()
            losses.append(ranking + 0.5 * preservation)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
        if step % 200 == 0 or step + 1 == args.steps:
            print({"step": step + 1, "loss": final_loss})
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "radio_gs.sugm_v3.cross_view_identity_adapter.v1",
        "state_dict": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        "rank": args.rank,
        "dimension": 128,
        "residual_bound": 0.25,
        "steps": args.steps,
        "seed": args.seed,
        "final_train_loss": final_loss,
        "scene_receipts": [item[3] for item in cohorts],
        "metadata": {
            "source_only": True,
            "train_view_residues": [1, 2],
            "dev_residue_opened": False,
            "audit_residue_opened": False,
            "target_rgb_opened": False,
            "benchmark_metrics_opened": False,
            "gaussian_indexed_state_added": 0,
            "frozen_scene_state": True,
            "shared_across_scenes": True,
        },
    }
    torch.save(payload, output)
    print({"output": str(output), "sha256": sha256_file(output)})


if __name__ == "__main__":
    main()
