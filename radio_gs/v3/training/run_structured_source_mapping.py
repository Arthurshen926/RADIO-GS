#!/usr/bin/env python3
"""Train the fresh shared-private D512 on source-only mapping authority."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_frozen_json, write_torch_noclobber
from radio_gs.v3.memory.structured_memory import (
    LowRankPrivateBranchMemory,
    OrthogonalProductMemory,
    SharedPrivateLayout,
    StructuredSharedPrivateMemory,
)
from radio_gs.v3.training.instance_upper_bound import (
    sha256_file,
    validate_source_only_inputs,
)
from radio_gs.v3.query.membership import membership_from_prototype, pool_prototype
from radio_gs.v3.training.rendered_mask import rendered_mask_loss
from radio_gs.v3.training.partition_owned_adamw import PartitionOwnedAdamW
from radio_gs.v3.training.rendered_mask import render_membership
from radio_gs.v3.training.run_instance_upper_bound import evaluate, load_episodes
from radio_gs.v3.training.structured_initialization import initialize_structured_memory
from radio_gs.v3.training.structured_initialization import fixed_jl_projection
from radio_gs.v3.training.learned_source_codec import apply_codec


def relation_training_edges(
    training, relation: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    train_proposals = {value.proposal_index for value in training}
    labels = torch.as_tensor(relation["edge_relation"]).to(torch.int8)
    source_left = torch.as_tensor(relation["edge_left"]).long()
    source_right = torch.as_tensor(relation["edge_right"]).long()
    same = labels == 1
    different_pairs = sorted({
        (min(item.proposal_index, peer), max(item.proposal_index, peer))
        for item in training for peer in item.different_proposals
        if peer in train_proposals
    })
    left = torch.cat((
        source_left[same],
        torch.tensor([value[0] for value in different_pairs], dtype=torch.long),
    ))
    right = torch.cat((
        source_right[same],
        torch.tensor([value[1] for value in different_pairs], dtype=torch.long),
    ))
    label = torch.cat((
        torch.ones(int(same.sum()), dtype=torch.int8),
        torch.zeros(len(different_pairs), dtype=torch.int8),
    ))
    train = torch.tensor([
        int(a) in train_proposals and int(b) in train_proposals
        for a, b in zip(left.tolist(), right.tolist())
    ])
    same_edges = torch.where(train & (label == 1))[0].tolist()
    different_edges = torch.where(train & (label == 0))[0].tolist()
    if not same_edges or not different_edges:
        raise ValueError("structured source split lacks same/different edges")
    return left, right, label, same_edges, different_edges


def boundary_objective(
    model: StructuredSharedPrivateMemory,
    head: nn.Linear,
    episode,
) -> torch.Tensor:
    device = model.memory.device
    unique_rows, inverse = torch.unique(
        episode.gaussian_ids, sorted=True, return_inverse=True
    )
    unique_logits = head(
        model.boundary_view(unique_rows.to(device))
    ).squeeze(-1)
    logits = unique_logits[inverse.to(device)]
    rendered = render_membership(
        logits.sigmoid(),
        torch.arange(logits.numel(), device=device),
        episode.pixel_ids.to(device),
        episode.contribution_weights.to(device),
        num_pixels=episode.target.numel(),
    )
    known = episode.known.to(device).flatten()
    target = episode.boundary.to(device).float().flatten()
    return F.binary_cross_entropy(rendered[known].clamp(1e-6, 1 - 1e-6), target[known])


def compact_episode_objective(
    model: StructuredSharedPrivateMemory,
    support: tuple[torch.Tensor, torch.Tensor],
    episode,
    *,
    temperature: float,
    unknown_growth_weight: float,
) -> torch.Tensor:
    """Exact episode loss without materializing an embedding for every row."""

    support_rows, support_weights = support
    combined = torch.cat((support_rows, episode.gaussian_ids))
    unique, inverse = torch.unique(combined, sorted=True, return_inverse=True)
    embedding = model(episode.scale, unique.to(model.memory.device))
    support_count = support_rows.numel()
    prototype = pool_prototype(
        embedding[inverse[:support_count].to(embedding.device)],
        support_weights.to(embedding.device),
    )
    hit_embedding = embedding[inverse[support_count:].to(embedding.device)]
    hit_posterior = membership_from_prototype(
        hit_embedding, prototype, temperature=temperature
    )
    prediction = render_membership(
        hit_posterior,
        torch.arange(hit_posterior.numel(), device=embedding.device),
        episode.pixel_ids.to(embedding.device),
        episode.contribution_weights.to(embedding.device),
        num_pixels=episode.target.numel(),
    )
    target = episode.target.to(embedding.device).float()
    known = episode.known.to(embedding.device)
    boundary = episode.boundary.to(embedding.device).float()
    height, width = episode.target.shape
    image = prediction.reshape(height, width)
    gradient = (
        F.max_pool2d(image[None, None], 3, 1, 1)
        + F.max_pool2d(-image[None, None], 3, 1, 1)
    )[0, 0]
    supervised = rendered_mask_loss(
        prediction,
        target.flatten(),
        known=known.flatten(),
        boundary_target=boundary.flatten(),
        boundary_prediction=(gradient * 16.0 - 4.0).flatten(),
    ).total
    unknown = episode.unknown.to(embedding.device).flatten()
    growth = (
        F.relu(prediction[unknown] - 0.5).square().mean()
        if bool(unknown.any()) else prediction.new_zeros(())
    )
    return supervised + float(unknown_growth_weight) * growth


def compact_relation_contrastive_loss(
    model: StructuredSharedPrivateMemory,
    supports: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    edge_left: torch.Tensor,
    edge_right: torch.Tensor,
    edge_relation: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Exact prototype relation loss evaluated only on selected supports."""

    device = model.memory.device
    proposal_ids = sorted(set(edge_left.tolist()) | set(edge_right.tolist()))
    prototypes = {}
    for proposal in proposal_ids:
        rows, weights = supports[proposal]
        prototypes[proposal] = pool_prototype(
            model(0.5, rows.to(device)), weights.to(device)
        )
    left = torch.stack([prototypes[int(value)] for value in edge_left.tolist()])
    right = torch.stack([prototypes[int(value)] for value in edge_right.tolist()])
    logits = (left * right).sum(-1) / float(temperature)
    return F.binary_cross_entropy_with_logits(
        logits, edge_relation.to(device).float()
    )


def sampled_shared_visual_loss(
    model: StructuredSharedPrivateMemory,
    episode,
    teacher: torch.Tensor,
    available_pixels: list[int],
    rng: random.Random,
    *,
    pixel_budget: int,
) -> torch.Tensor:
    chosen = sorted(rng.sample(available_pixels, k=min(pixel_budget, len(available_pixels))))
    chosen_tensor = torch.tensor(chosen, dtype=torch.long)
    left = torch.searchsorted(episode.pixel_ids, chosen_tensor, right=False)
    right = torch.searchsorted(episode.pixel_ids, chosen_tensor, right=True)
    hit_rows, hit_pixels, hit_weights = [], [], []
    for local, (start, stop) in enumerate(zip(left.tolist(), right.tolist())):
        if stop <= start:
            continue
        hit_rows.append(episode.gaussian_ids[start:stop])
        hit_pixels.append(torch.full((stop - start,), local, dtype=torch.long))
        hit_weights.append(episode.contribution_weights[start:stop])
    if not hit_rows:
        raise ValueError("sampled structured visual pixels have no exact hits")
    device = model.memory.device
    rows = torch.cat(hit_rows).to(device)
    local_pixels = torch.cat(hit_pixels).to(device)
    weights = torch.cat(hit_weights).to(device)
    features = model.block("shared", rows)
    rendered = torch.zeros(len(chosen), features.shape[1], device=device)
    rendered.index_add_(0, local_pixels, features * weights[:, None])
    target = teacher[chosen_tensor].to(device)
    rendered = model.shared_capability_view(rendered)
    target = model.shared_target_view(target)
    return (1.0 - F.cosine_similarity(rendered, target, dim=-1, eps=1e-8)).mean()


@torch.no_grad()
def protected_block_deltas(
    before: torch.Tensor,
    after: torch.Tensor,
    layout: SharedPrivateLayout,
) -> dict[str, float]:
    output = {}
    for name, columns in layout.slices.items():
        difference = (after[:, columns] - before[:, columns]).abs()
        output[name] = float(difference.max()) if difference.numel() else 0.0
    return output


def run(args: argparse.Namespace) -> dict[str, object]:
    membership_path = Path(args.membership).resolve(strict=True)
    relation_path = Path(args.relation).resolve(strict=True)
    siglip_path = Path(args.siglip_teacher).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu")
    relation = torch.load(relation_path, map_location="cpu")
    validate_source_only_inputs(membership, relation)
    episodes, supports = load_episodes(membership, relation)
    valid = [value for value in episodes if supports[value.proposal_index][0].numel()]
    training = [value for value in valid if value.view_index % 4 in (1, 2)]
    evaluation = [value for value in valid if value.view_index % 4 == 3]
    train_proposals = {value.proposal_index for value in training}
    left, right, labels, same_edges, different_edges = relation_training_edges(
        training, relation
    )
    layout = SharedPrivateLayout(
        shared=args.shared_dim,
        semantic=args.semantic_dim,
        instance=args.instance_dim,
        boundary=args.boundary_dim,
    )
    initial, initialization = initialize_structured_memory(
        membership,
        radio_teacher_root=args.radio_teacher_root,
        siglip_teacher_path=siglip_path,
        layout=layout,
        seed=args.seed,
        hit_chunk=args.initialization_hit_chunk,
        codec_path=args.codec,
    )
    initial_protected = initial[:, : layout.shared + layout.semantic].clone()
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    model_types = {
        "hard_block": StructuredSharedPrivateMemory,
        "orthogonal_product": OrthogonalProductMemory,
        "low_rank_private": LowRankPrivateBranchMemory,
    }
    model = model_types[args.architecture](initial, layout=layout)
    if args.architecture == "orthogonal_product" and args.orthogonal_angle_init > 0:
        generator = torch.Generator(device="cpu").manual_seed(int(args.seed + 3))
        with torch.no_grad():
            model.basis_angles.uniform_(
                -args.orthogonal_angle_init,
                args.orthogonal_angle_init,
                generator=generator,
            )
    model = model.to(device)
    # First causal proof has no cross-block bridge. The zero-initialized bridge
    # remains serialized as a global constant-size component for later ablation.
    model.visual_to_instance.requires_grad_(False)
    model.context_to_boundary.requires_grad_(False)
    boundary_head = nn.Linear(layout.boundary, 1).to(device)
    baseline, baseline_count = evaluate(
        model, evaluation, supports, relation, train_proposals, args.temperature
    )
    if args.visual_weight > 0 and args.freeze_shared_visual:
        visual_parameters = model.visual_auxiliary_parameters()
        if not visual_parameters:
            raise ValueError("frozen shared visual phase requires global visual parameters")
        visual_optimizer = torch.optim.AdamW(
            visual_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.visual_weight > 0:
        visual_optimizer = PartitionOwnedAdamW(
            model.memory,
            layout.slices["shared"],
            model.visual_auxiliary_parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        visual_optimizer = None
    instance_optimizer = PartitionOwnedAdamW(
        model.memory,
        layout.slices["instance"],
        model.instance_auxiliary_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    boundary_optimizer = PartitionOwnedAdamW(
        model.memory,
        layout.slices["boundary"],
        (*boundary_head.parameters(), *model.boundary_auxiliary_parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = random.Random(args.seed)
    visual_training = []
    codec_state = (
        torch.load(Path(args.codec).resolve(strict=True), map_location="cpu")["state_dict"]
        if args.codec else None
    )
    seen_views = set()
    for item in training:
        if item.view_index in seen_views:
            continue
        seen_views.add(item.view_index)
        record = next(
            value for value in membership["metadata"]["source_records"]
            if int(value["source_view_index"]) == item.view_index
        )
        teacher = torch.load(
            Path(args.radio_teacher_root).resolve(strict=True)
            / "backbone" / f"rgb_{int(record['frame_id'])}.pt",
            map_location="cpu",
        ).float()
        if codec_state is not None:
            projection = torch.as_tensor(codec_state["radio_basis"]).float()
            teacher = apply_codec(
                teacher.permute(1, 2, 0).reshape(-1, teacher.shape[0]),
                torch.as_tensor(codec_state["radio_mean"]).float(),
                projection,
            )
        else:
            projection = fixed_jl_projection(teacher.shape[0], layout.shared, args.seed)
            teacher = teacher.permute(1, 2, 0).reshape(-1, teacher.shape[0]) @ projection
        teacher = F.normalize(
            teacher,
            dim=-1,
            eps=1e-8,
        )
        visual_training.append((item, teacher, torch.unique_consecutive(item.pixel_ids).tolist()))
    best_loss = float("inf")
    best_state = None
    best_boundary_head = None
    boundary_sum = 0.0
    visual_sum = 0.0
    for step in range(args.steps):
        if visual_optimizer is not None:
            visual_episode, visual_teacher, visual_pixels = rng.choice(visual_training)
            visual_loss = sampled_shared_visual_loss(
                model,
                visual_episode,
                visual_teacher,
                visual_pixels,
                rng,
                pixel_budget=args.visual_pixels_per_step,
            )
            visual_optimizer.zero_grad(set_to_none=True)
            (args.visual_weight * visual_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                [model.memory, *model.visual_auxiliary_parameters()], 5.0
            )
            visual_optimizer.step()
            visual_sum += float(visual_loss.detach())
        else:
            visual_loss = model.memory.new_zeros(())

        selected = rng.sample(training, k=min(args.episodes_per_step, len(training)))
        instance_optimizer.zero_grad(set_to_none=True)
        render_values = []
        for item in selected:
            episode_loss = compact_episode_objective(
                model,
                supports[item.proposal_index],
                item,
                temperature=args.temperature,
                unknown_growth_weight=args.unknown_growth_weight,
            )
            render_values.append(float(episode_loss.detach()))
            (episode_loss / len(selected)).backward()
        each = min(args.relation_edges_per_step // 2, len(same_edges), len(different_edges))
        edge_indices = torch.tensor(
            rng.sample(same_edges, each) + rng.sample(different_edges, each),
            dtype=torch.long,
        )
        relation_value = 0.0
        for start in range(0, edge_indices.numel(), args.relation_backward_chunk):
            chunk = edge_indices[start : start + args.relation_backward_chunk]
            chunk_relation_loss = compact_relation_contrastive_loss(
                model, supports,
                left[chunk], right[chunk], labels[chunk],
                temperature=args.relation_temperature,
            )
            fraction = chunk.numel() / edge_indices.numel()
            (args.relation_weight * fraction * chunk_relation_loss).backward()
            relation_value += fraction * float(chunk_relation_loss.detach())
        instance_loss = model.memory.new_tensor(
            sum(render_values) / len(render_values)
            + args.relation_weight * relation_value
        )
        torch.nn.utils.clip_grad_norm_(
            [model.memory, *model.instance_auxiliary_parameters()], 5.0
        )
        instance_optimizer.step()

        boundary_optimizer.zero_grad(set_to_none=True)
        boundary_values = []
        for item in selected:
            episode_boundary_loss = boundary_objective(
                model, boundary_head, item
            )
            boundary_values.append(float(episode_boundary_loss.detach()))
            (episode_boundary_loss / len(selected)).backward()
        boundary_loss = model.memory.new_tensor(
            sum(boundary_values) / len(boundary_values)
        )
        torch.nn.utils.clip_grad_norm_(
            [
                model.memory,
                *boundary_head.parameters(),
                *model.boundary_auxiliary_parameters(),
            ],
            5.0,
        )
        boundary_optimizer.step()
        boundary_sum += float(boundary_loss.detach())
        value = (
            args.visual_weight * float(visual_loss.detach())
            + float(instance_loss.detach())
            + float(boundary_loss.detach())
        )
        snapshot = (step + 1) % args.snapshot_interval == 0 or step + 1 == args.steps
        if snapshot:
            print({
                "step": step + 1,
                "visual_loss": float(visual_loss.detach()),
                "instance_loss": float(instance_loss.detach()),
                "boundary_loss": float(boundary_loss.detach()),
            }, flush=True)
        if snapshot and value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            best_boundary_head = {
                key: tensor.detach().cpu().clone()
                for key, tensor in boundary_head.state_dict().items()
            }
    if best_state is None or best_boundary_head is None:
        raise RuntimeError("structured training produced no checkpoint")
    model.load_state_dict(best_state)
    boundary_head.load_state_dict(best_boundary_head)
    candidate, candidate_count = evaluate(
        model, evaluation, supports, relation, train_proposals, args.temperature
    )
    deployed = model.deployment_memory().cpu()
    deltas = protected_block_deltas(initial, deployed, layout)
    if deltas["semantic"] != 0.0:
        raise RuntimeError("private losses rewrote a protected capability block")
    output = Path(args.output).resolve()
    payload = {
        "schema": "radio_gs.sugm_v3.structured_source_mapping.v1",
        "state_dict": {
            **best_state,
            **({
                "codec.radio_mean": codec_state["radio_mean"],
                "codec.radio_basis": codec_state["radio_basis"],
                "codec.siglip_mean": codec_state["siglip_mean"],
                "codec.siglip_basis": codec_state["siglip_basis"],
            } if codec_state is not None else {}),
            "memory": deployed,
            "boundary_head.weight": best_boundary_head["weight"],
            "boundary_head.bias": best_boundary_head["bias"],
        },
        "metadata": {
            "state_representation": "single_structured_shared_private_d512_plus_global_heads",
            "architecture": model.architecture,
            "layout": dict(layout.__dict__),
            "partition_owned_writes": True,
            "phase_order": (
                "global_basis_visual_then_instance_then_boundary"
                if args.freeze_shared_visual and visual_optimizer is not None
                else (
                    "visual_then_instance_then_boundary"
                    if visual_optimizer is not None
                    else "frozen_visual_then_instance_then_boundary"
                )
            ),
            "shared_memory_frozen_during_visual_phase": args.freeze_shared_visual,
            "cross_block_bridges_enabled": args.architecture == "low_rank_private",
            "orthogonal_basis_parameterization": (
                "disjoint_global_givens" if args.architecture == "orthogonal_product" else None
            ),
            "orthogonal_angle_init": (
                args.orthogonal_angle_init
                if args.architecture == "orthogonal_product" else None
            ),
            "private_branch_ranks": (
                {"instance": 8, "boundary": 4}
                if args.architecture == "low_rank_private" else None
            ),
            "gaussian_indexed_sidecars": 0,
            "historical_field_opened": False,
            "source_only": True,
            "target_rgb_opened": False,
            "benchmark_metrics_opened": False,
            "view_split": {"train_residues": [1, 2], "dev_residue": 3, "audit_residue": 0},
            "relation_backward_chunk": args.relation_backward_chunk,
            "initialization": initialization,
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "relation": {"path": str(relation_path), "sha256": sha256_file(relation_path)},
            "siglip_teacher": {"path": str(siglip_path), "sha256": sha256_file(siglip_path)},
            "radio_teacher_root": str(Path(args.radio_teacher_root).resolve()),
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "schema": "radio_gs.sugm_v3.structured_source_mapping.report.v1",
        "architecture": model.architecture,
        "layout": dict(layout.__dict__),
        "steps": args.steps,
        "best_train_loss": best_loss,
        "mean_boundary_train_loss": boundary_sum / args.steps,
        "mean_visual_train_loss": (
            visual_sum / args.steps if visual_optimizer is not None else None
        ),
        "baseline_evaluation_proposals": baseline_count,
        "evaluation_proposals": candidate_count,
        "baseline_metrics": baseline,
        "candidate_metrics": candidate,
        "delta": {name: candidate[name] - baseline[name] for name in candidate},
        "protected_block_max_abs_delta": deltas,
        "capability_preservation": {
            "shared_trained_only_by_visual_authority": (
                visual_optimizer is not None and not args.freeze_shared_visual
            ),
            "shared_frozen": visual_optimizer is None or args.freeze_shared_visual,
            "semantic_exact": deltas["semantic"] == 0.0,
            "raw_radio_is_hard_gate": False,
            "status": "structural_preservation_only_pending_source_capability_suite",
        },
        "checkpoint": {"path": str(output), "sha256": sha256_file(output)},
        "status": "development_evidence_only",
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--membership", required=True)
    value.add_argument("--relation", required=True)
    value.add_argument("--radio-teacher-root", required=True)
    value.add_argument("--siglip-teacher", required=True)
    value.add_argument("--codec")
    value.add_argument("--device", default="cuda:0")
    value.add_argument(
        "--architecture",
        choices=("hard_block", "orthogonal_product", "low_rank_private"),
        default="hard_block",
    )
    value.add_argument("--steps", type=int, default=600)
    value.add_argument("--episodes-per-step", type=int, default=4)
    value.add_argument("--temperature", type=float, default=0.15)
    value.add_argument("--unknown-growth-weight", type=float, default=0.25)
    value.add_argument("--relation-edges-per-step", type=int, default=32)
    value.add_argument("--relation-temperature", type=float, default=0.1)
    value.add_argument("--relation-weight", type=float, default=1.0)
    value.add_argument("--relation-backward-chunk", type=int, default=16)
    value.add_argument("--learning-rate", type=float, default=1e-3)
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument("--visual-weight", type=float, default=1.0)
    value.add_argument("--freeze-shared-visual", action="store_true")
    value.add_argument("--orthogonal-angle-init", type=float, default=0.0)
    value.add_argument("--visual-pixels-per-step", type=int, default=64)
    value.add_argument("--shared-dim", type=int, default=320)
    value.add_argument("--semantic-dim", type=int, default=128)
    value.add_argument("--instance-dim", type=int, default=48)
    value.add_argument("--boundary-dim", type=int, default=16)
    value.add_argument("--initialization-hit-chunk", type=int, default=32768)
    value.add_argument("--snapshot-interval", type=int, default=50)
    value.add_argument("--seed", type=int, default=20260826)
    value.add_argument("--output", required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if (
        args.steps <= 0
        or args.episodes_per_step <= 0
        or args.snapshot_interval <= 0
        or args.visual_weight < 0
        or args.visual_pixels_per_step <= 0
        or args.relation_backward_chunk <= 0
        or args.orthogonal_angle_init < 0
    ):
        raise ValueError("structured source-mapping budgets must be positive")
    if args.freeze_shared_visual and args.architecture != "orthogonal_product":
        raise ValueError("only the orthogonal arm owns a global visual basis")
    if args.orthogonal_angle_init > 0 and args.architecture != "orthogonal_product":
        raise ValueError("only the orthogonal arm owns basis angles")
    print(run(args))


if __name__ == "__main__":
    main()
