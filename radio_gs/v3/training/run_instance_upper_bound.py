#!/usr/bin/env python3
"""Run source-heldout SUGM-v3 Arm A or temporary Arm B."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.utils.immutable_artifacts import write_frozen_json, write_torch_noclobber
from radio_gs.v3.evaluation.source_heldout import evaluate_source_heldout
from radio_gs.v3.training.instance_upper_bound import (
    ExtraCodeArm,
    FrozenProjectionArm,
    MaskEpisode,
    align_masks,
    build_known_pixel_authority,
    episode_objective,
    mask_boundary,
    proposal_supports,
    render_episode,
    relation_contrastive_loss,
    sha256_file,
    same_view_different_peers,
    unpack_masks,
    validate_source_only_inputs,
)
from radio_gs.v3.training.low_rank_writeback import (
    LowRankWritebackArm,
    pcgrad_backward,
    pcgrad_backward_sparse_anchor,
)
from radio_gs.v3.training.joint_d512 import JointD512Arm
from radio_gs.v3.training.memory_safe_adamw import MemorySafeAdamW


@torch.no_grad()
def materialize_canonical_memory(
    field: torch.nn.Module, *, chunk_size: int = 8192
) -> torch.Tensor:
    """Materialize the public post-fusion D512 field without exposing internals."""

    if chunk_size <= 0:
        raise ValueError("canonical-memory chunk size must be positive")
    count = int(field.num_gaussians)
    chunks = []
    for start in range(0, count, int(chunk_size)):
        rows = torch.arange(start, min(start + int(chunk_size), count))
        chunks.append(
            field.query_memory(rows, representation="coefficients").detach().cpu()
        )
    memory = torch.cat(chunks, dim=0)
    expected = (count, int(field.decoder.coefficient_dim))
    if memory.shape != expected:
        raise ValueError("canonical post-fusion memory shape differs")
    return memory


@torch.no_grad()
def radio_preservation_summary(
    model: LowRankWritebackArm | JointD512Arm, *, chunk_size: int = 8192
) -> dict[str, float]:
    values = []
    for start in range(0, int(model.base_latent.shape[0]), int(chunk_size)):
        rows = torch.arange(
            start,
            min(start + int(chunk_size), int(model.base_latent.shape[0])),
            device=model.base_latent.device,
        )
        values.append(model.radio_cosine(rows).cpu())
    cosine = torch.cat(values)
    return {
        "mean_cosine": float(cosine.mean()),
        "p05_cosine": float(torch.quantile(cosine, 0.05)),
        "minimum_cosine": float(cosine.min()),
    }


def sampled_source_visual_loss(
    model: LowRankWritebackArm | JointD512Arm,
    episode: MaskEpisode,
    teacher: torch.Tensor,
    available_pixels: list[int],
    rng: random.Random,
    *,
    pixel_budget: int,
) -> torch.Tensor:
    """Train RADIO fidelity on sampled source-train pixels and exact hits."""

    chosen = sorted(rng.sample(available_pixels, k=min(pixel_budget, len(available_pixels))))
    chosen_tensor = torch.tensor(chosen, dtype=torch.long)
    pixels = episode.pixel_ids
    left = torch.searchsorted(pixels, chosen_tensor, right=False)
    right = torch.searchsorted(pixels, chosen_tensor, right=True)
    hit_rows = []
    hit_pixels = []
    hit_weights = []
    for local, (start, stop) in enumerate(zip(left.tolist(), right.tolist())):
        if stop <= start:
            continue
        hit_rows.append(episode.gaussian_ids[start:stop])
        hit_pixels.append(torch.full((stop - start,), local, dtype=torch.long))
        hit_weights.append(episode.contribution_weights[start:stop])
    if not hit_rows:
        raise ValueError("sampled source visual pixels have no exact hits")
    device = model.base_latent.device
    rows = torch.cat(hit_rows).to(device)
    local_pixels = torch.cat(hit_pixels).to(device)
    weights = torch.cat(hit_weights).to(device)
    features = model.decode_radio(model.coefficients(rows))
    rendered = torch.zeros(len(chosen), features.shape[1], device=device)
    rendered.index_add_(0, local_pixels, features * weights[:, None])
    target = teacher[chosen_tensor].to(device)
    return (1.0 - F.cosine_similarity(rendered, target, dim=-1, eps=1e-8)).mean()


def load_episodes(
    membership: dict,
    relation: dict,
) -> tuple[list[MaskEpisode], tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
    metadata = membership["metadata"]
    height, width = int(metadata["feature_height"]), int(metadata["feature_width"])
    proposal_count = int(membership["num_proposals"])
    supports = proposal_supports(
        membership["row_indices"], membership["proposal_indices"],
        membership["weights"], proposal_count,
    )
    edge_relation = torch.as_tensor(relation["edge_relation"]).to(torch.int8)
    different = edge_relation == 0
    different_left = torch.as_tensor(relation["edge_left"]).long()[different]
    different_right = torch.as_tensor(relation["edge_right"]).long()[different]
    episodes: list[MaskEpisode] = []
    proposal_offset = 0
    for record in metadata["source_records"]:
        mask_payload = torch.load(record["mask_cache"], map_location="cpu")
        shard = torch.load(record["responsibility_view"], map_location="cpu")
        mask_height, mask_width = (int(value) for value in mask_payload["mask_shape"])
        masks = align_masks(
            unpack_masks(mask_payload["packed_masks"], mask_width), height, width
        )
        count = int(record["num_proposals"])
        if masks.shape != (count, height, width) or int(shard["num_pixels"]) != height * width:
            raise ValueError("source mask and exact contribution raster axes differ")
        globals_for_view = torch.arange(proposal_offset, proposal_offset + count)
        areas = torch.as_tensor(mask_payload["proposal_area_fraction"]).float()
        for local in range(count):
            global_index = proposal_offset + local
            target, known = build_known_pixel_authority(
                masks, local, global_index, globals_for_view,
                different_left, different_right,
            )
            different_proposals = same_view_different_peers(
                masks, local, globals_for_view
            )
            episodes.append(MaskEpisode(
                proposal_index=global_index,
                view_index=int(record["source_view_index"]),
                gaussian_ids=torch.as_tensor(shard["gaussian_ids"]).long(),
                pixel_ids=torch.as_tensor(shard["pixel_ids"]).long(),
                contribution_weights=torch.as_tensor(shard["base_weights"]).float(),
                target=target,
                known=known,
                boundary=mask_boundary(target),
                unknown=~known,
                scale=float(areas[local]),
                different_proposals=different_proposals,
            ))
        proposal_offset += count
    if proposal_offset != proposal_count or len(episodes) != proposal_count:
        raise ValueError("source episode proposal axis differs")
    return episodes, supports


def training_support_for_heldout(
    proposal: int,
    train_proposals: set[int],
    supports: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    relation: dict,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    edge_relation = torch.as_tensor(relation["edge_relation"]).to(torch.int8)
    left = torch.as_tensor(relation["edge_left"]).long()
    right = torch.as_tensor(relation["edge_right"]).long()
    peers: list[int] = []
    for a, b, label in zip(left.tolist(), right.tolist(), edge_relation.tolist()):
        if label != 1:
            continue
        if a == proposal and b in train_proposals:
            peers.append(b)
        elif b == proposal and a in train_proposals:
            peers.append(a)
    if not peers:
        return None
    rows = torch.cat([supports[value][0] for value in peers])
    weights = torch.cat([supports[value][1] for value in peers])
    return rows, weights


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    episodes: list[MaskEpisode],
    supports: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    relation: dict,
    train_proposals: set[int],
    temperature: float,
) -> tuple[dict[str, float], int]:
    metrics = []
    shared_projection = (
        model.projected_latent()
        if isinstance(model, (LowRankWritebackArm, JointD512Arm))
        else None
    )
    for episode in episodes:
        if not bool((episode.known & ~episode.target).any()):
            continue
        support = training_support_for_heldout(
            episode.proposal_index, train_proposals, supports, relation
        )
        if support is None:
            continue
        embedding = (
            model.scale_embedding(shared_projection, episode.scale)
            if shared_projection is not None
            else model(episode.scale)
        )
        prediction, _ = render_episode(
            embedding, support, episode, temperature=temperature
        )
        height, width = episode.target.shape
        image = prediction.reshape(height, width)
        dilation = F.max_pool2d(image[None, None], 3, 1, 1)
        erosion = -F.max_pool2d(-image[None, None], 3, 1, 1)
        edge = torch.sigmoid((dilation - erosion)[0, 0] * 16.0 - 4.0)
        metrics.append(evaluate_source_heldout(
            prediction.cpu(), episode.target.flatten(), episode.known.flatten(),
            episode.unknown.flatten(), edge.flatten().cpu(), episode.boundary.flatten(),
        ))
    if not metrics:
        raise ValueError("no heldout proposal has source-only same-object training authority")
    summary = {
        name: sum(getattr(value, name) for value in metrics) / len(metrics)
        for name in ("mask_iou", "brier", "boundary_f", "unknown_fp_mass")
    }
    return summary, len(metrics)


def run(args: argparse.Namespace) -> dict[str, object]:
    membership_path = Path(args.membership).resolve(strict=True)
    relation_path = Path(args.relation).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu")
    relation = torch.load(relation_path, map_location="cpu")
    validate_source_only_inputs(membership, relation)
    episodes, supports = load_episodes(membership, relation)
    valid_episodes = [
        value for value in episodes
        if supports[value.proposal_index][0].numel() > 0
    ]
    evaluation_residue = 3 if args.evaluation_split == "dev" else 0
    evaluation = [
        value for value in valid_episodes
        if value.view_index % 4 == evaluation_residue
    ]
    training = [
        value for value in valid_episodes
        if value.view_index % 4 in (1, 2)
    ]
    train_proposals = {value.proposal_index for value in training}
    source_relation_label = torch.as_tensor(relation["edge_relation"]).to(torch.int8)
    source_relation_left = torch.as_tensor(relation["edge_left"]).long()
    source_relation_right = torch.as_tensor(relation["edge_right"]).long()
    same = source_relation_label == 1
    different_pairs = sorted({
        (min(item.proposal_index, peer), max(item.proposal_index, peer))
        for item in training for peer in item.different_proposals
        if peer in train_proposals
    })
    relation_left = torch.cat((
        source_relation_left[same],
        torch.tensor([value[0] for value in different_pairs], dtype=torch.long),
    ))
    relation_right = torch.cat((
        source_relation_right[same],
        torch.tensor([value[1] for value in different_pairs], dtype=torch.long),
    ))
    relation_label = torch.cat((
        torch.ones(int(same.sum()), dtype=torch.int8),
        torch.zeros(len(different_pairs), dtype=torch.int8),
    ))
    train_edge = torch.tensor([
        int(left) in train_proposals and int(right) in train_proposals
        for left, right in zip(relation_left.tolist(), relation_right.tolist())
    ])
    same_edges = torch.where(train_edge & (relation_label == 1))[0].tolist()
    different_edges = torch.where(train_edge & (relation_label == 0))[0].tolist()
    if not same_edges or not different_edges:
        raise ValueError("source train split lacks same or different relation authority")
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    if args.arm == "oracle16":
        model = ExtraCodeArm(int(membership["num_rows"]), seed=args.seed)
        field_record = None
    else:
        field_path = Path(args.field).resolve(strict=True)
        field, field_payload, _signature = load_factorized_canonical_field_checkpoint(
            field_path, map_location="cpu", expected_sha256=args.expected_field_sha256
        )
        canonical_memory = materialize_canonical_memory(field)
        if args.arm == "frozen512":
            model = FrozenProjectionArm(canonical_memory)
        elif args.arm == "joint512":
            model = JointD512Arm(
                canonical_memory,
                radio_basis=field.decoder.basis.detach().cpu(),
                radio_mean=field.decoder.mean.detach().cpu(),
                radio_scale=field.decoder.scale.detach().cpu(),
            )
        else:
            model = LowRankWritebackArm(
                canonical_memory,
                radio_basis=field.decoder.basis.detach().cpu(),
                radio_mean=field.decoder.mean.detach().cpu(),
                radio_scale=field.decoder.scale.detach().cpu(),
                rank=args.residual_rank,
                seed=args.seed,
            )
        architecture = field_payload["architecture"]
        field_record = {
            "path": str(field_path),
            "sha256": sha256_file(field_path),
            "representation": "coefficients",
            "post_fusion": True,
            "dimension": int(canonical_memory.shape[1]),
            "num_gaussians": int(canonical_memory.shape[0]),
            "source_use_fusion": bool(architecture["use_fusion"]),
            "source_local_dim": int(architecture["local_dim"]),
        }
    model.to(device)
    baseline_metrics, baseline_evaluated = evaluate(
        model, evaluation, supports, relation, train_proposals, args.temperature
    )
    optimizer = (
        MemorySafeAdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            chunk_elements=args.optimizer_chunk_elements,
        )
        if args.arm == "joint512"
        else torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    )
    rng = random.Random(args.seed)
    visual_training = []
    if args.alternating_visual_weight > 0:
        if args.arm not in ("writeback512", "joint512") or not args.visual_teacher_root:
            raise ValueError("alternating visual steps require a trainable D512 and teacher root")
        teacher_root = Path(args.visual_teacher_root).resolve(strict=True)
        seen_views = set()
        for item in training:
            if item.view_index in seen_views:
                continue
            seen_views.add(item.view_index)
            teacher_path = teacher_root / "backbone" / f"rgb_{item.view_index + 1}.pt"
            # Source record frame IDs are authoritative and need not equal the
            # dense view index; recover the matching record explicitly.
            record = next(
                value for value in membership["metadata"]["source_records"]
                if int(value["source_view_index"]) == item.view_index
            )
            teacher_path = teacher_root / "backbone" / f"rgb_{int(record['frame_id'])}.pt"
            teacher = torch.load(teacher_path, map_location="cpu").float()
            teacher = teacher.permute(1, 2, 0).reshape(-1, teacher.shape[0])
            available = torch.unique_consecutive(item.pixel_ids).tolist()
            visual_training.append((item, teacher, available))
        if not visual_training:
            raise ValueError("alternating visual step lacks source-train authority")
    best_loss = float("inf")
    best_state = None
    pcgrad_conflicts = 0
    visual_loss_sum = 0.0
    for step in range(args.steps):
        visual_value = 0.0
        if visual_training:
            visual_episode, visual_teacher, visual_pixels = rng.choice(visual_training)
            visual_loss = sampled_source_visual_loss(
                model,
                visual_episode,
                visual_teacher,
                visual_pixels,
                rng,
                pixel_budget=args.visual_pixels_per_step,
            )
            optimizer.zero_grad(set_to_none=True)
            (args.alternating_visual_weight * visual_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            visual_value = float(visual_loss.detach())
            visual_loss_sum += visual_value
        selected = rng.sample(training, k=min(args.episodes_per_step, len(training)))
        shared_projection = (
            model.projected_latent()
            if args.arm in ("writeback512", "joint512")
            else None
        )
        render_loss = torch.stack([
            episode_objective(
                (
                    model.scale_embedding(shared_projection, item.scale)
                    if shared_projection is not None
                    else model(item.scale)
                ),
                supports[item.proposal_index], item,
                temperature=args.temperature,
                unknown_growth_weight=args.unknown_growth_weight,
            )
            for item in selected
        ]).mean()
        each_class = min(args.relation_edges_per_step // 2, len(same_edges), len(different_edges))
        edge_indices = rng.sample(same_edges, each_class) + rng.sample(different_edges, each_class)
        edge_indices = torch.tensor(edge_indices, dtype=torch.long)
        relation_loss = relation_contrastive_loss(
            (
                model.scale_embedding(shared_projection, 0.5)
                if shared_projection is not None
                else model(0.5)
            ),
            supports,
            relation_left[edge_indices], relation_right[edge_indices],
            relation_label[edge_indices], temperature=args.relation_temperature,
        )
        loss = render_loss + args.relation_weight * relation_loss
        optimizer.zero_grad(set_to_none=True)
        selection_loss = loss
        if args.arm in ("writeback512", "joint512"):
            anchor_candidates = torch.unique(torch.cat(tuple(
                [item.gaussian_ids for item in selected]
                + [supports[item.proposal_index][0] for item in selected]
                + [supports[int(index)][0] for index in relation_left[edge_indices].tolist()]
                + [supports[int(index)][0] for index in relation_right[edge_indices].tolist()]
            )))
            if anchor_candidates.numel() > args.radio_anchor_rows_per_step:
                positions = rng.sample(
                    range(anchor_candidates.numel()), args.radio_anchor_rows_per_step
                )
                anchor_candidates = anchor_candidates[torch.tensor(positions)]
            if args.arm == "joint512":
                anchor_values = (
                    model.coefficients(anchor_candidates).detach().requires_grad_(True)
                )
                anchor_loss = model.radio_anchor_loss_from_coefficients(
                    anchor_candidates, anchor_values
                )
                projection_report = pcgrad_backward_sparse_anchor(
                    loss,
                    anchor_loss,
                    model.parameters(),
                    anchor_parameter=model.latent,
                    anchor_rows=anchor_candidates,
                    anchor_values=anchor_values,
                    anchor_weight=args.radio_anchor_weight,
                )
            else:
                anchor_loss = model.radio_anchor_loss(anchor_candidates)
                projection_report = pcgrad_backward(
                    loss,
                    anchor_loss,
                    model.parameters(),
                    anchor_weight=args.radio_anchor_weight,
                )
            pcgrad_conflicts += int(projection_report.conflict)
            selection_loss = loss + args.radio_anchor_weight * anchor_loss
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        value = float(selection_loss.detach()) + args.alternating_visual_weight * visual_value
        snapshot_step = (
            args.arm != "joint512"
            or (step + 1) % args.joint_snapshot_interval == 0
            or step + 1 == args.steps
        )
        if snapshot_step and value < best_loss:
            best_loss = value
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("upper-bound training produced no checkpoint")
    model.load_state_dict(best_state)
    metrics, evaluated = evaluate(
        model, evaluation, supports, relation, train_proposals, args.temperature
    )
    output = Path(args.output).resolve()
    if args.arm == "writeback512":
        deployment_state = {
            "latent": model.folded_latent().cpu(),
            "projection.weight": model.projection.weight.detach().cpu().clone(),
            "scale_adapter.weight": model.scale_adapter.weight.detach().cpu().clone(),
            "scale_adapter.bias": model.scale_adapter.bias.detach().cpu().clone(),
        }
        radio_preservation = radio_preservation_summary(model)
        state_representation = "folded_single_d512_plus_global_head"
    elif args.arm == "joint512":
        deployment_state = {
            "latent": model.deployment_latent().cpu(),
            "projection.weight": model.projection.weight.detach().cpu().clone(),
            "scale_adapter.weight": model.scale_adapter.weight.detach().cpu().clone(),
            "scale_adapter.bias": model.scale_adapter.bias.detach().cpu().clone(),
        }
        radio_preservation = radio_preservation_summary(model)
        state_representation = "single_joint_d512_plus_global_head"
    else:
        deployment_state = best_state
        radio_preservation = None
        state_representation = "development_training_state"
    payload = {
        "schema": "radio_gs.sugm_v3.instance_upper_bound.v2",
        "arm": args.arm,
        "state_dict": deployment_state,
        "metadata": {
            "deployment_eligible": bool(model.deployment_eligible),
            "source_only": True,
            "benchmark_rgb_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_metrics_opened": False,
            "view_split": {
                "train_residues": [1, 2],
                "dev_residue": 3,
                "audit_residue": 0,
                "opened_evaluation_split": args.evaluation_split,
            },
            "unknown_policy": "excluded_from_negative_loss_with_one_sided_neutral_prior_growth_restraint",
            "relation_loss": "cross_view_same_plus_same_view_comparable_disjoint_different_prototype_log_score_unknown_excluded",
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "relation": {"path": str(relation_path), "sha256": sha256_file(relation_path)},
            "field": field_record,
            "state_representation": state_representation,
            "low_rank_training_parameterization_discarded": args.arm == "writeback512",
            "joint_d512_source_mapping": args.arm == "joint512",
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "schema": "radio_gs.sugm_v3.instance_upper_bound.report.v2",
        "arm": args.arm,
        "best_train_loss": best_loss,
        "evaluation_split": args.evaluation_split,
        "evaluation_proposals": evaluated,
        "baseline_evaluation_proposals": baseline_evaluated,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": metrics,
        "delta": {name: metrics[name] - baseline_metrics[name] for name in metrics},
        "checkpoint": {"path": str(output), "sha256": sha256_file(output)},
        "status": "development_evidence_only",
        "radio_preservation": radio_preservation,
        "pcgrad_conflict_fraction": (
            pcgrad_conflicts / args.steps
            if args.arm in ("writeback512", "joint512")
            else None
        ),
        "alternating_visual": {
            "enabled": bool(visual_training),
            "mean_train_loss": visual_loss_sum / args.steps if visual_training else None,
            "weight": args.alternating_visual_weight,
            "pixels_per_step": args.visual_pixels_per_step,
            "teacher_root": str(Path(args.visual_teacher_root).resolve()) if visual_training else None,
            "split": "source_train_view_residues_1_2",
        },
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "--arm",
        choices=("frozen512", "oracle16", "writeback512", "joint512"),
        required=True,
    )
    value.add_argument("--membership", required=True)
    value.add_argument("--relation", required=True)
    value.add_argument("--field")
    value.add_argument("--expected-field-sha256")
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--steps", type=int, default=2000)
    value.add_argument("--episodes-per-step", type=int, default=4)
    value.add_argument("--temperature", type=float, default=0.15)
    value.add_argument("--learning-rate", type=float, default=1e-3)
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument("--unknown-growth-weight", type=float, default=0.25)
    value.add_argument("--evaluation-split", choices=("dev", "audit"), default="dev")
    value.add_argument("--relation-edges-per-step", type=int, default=32)
    value.add_argument("--relation-temperature", type=float, default=0.1)
    value.add_argument("--relation-weight", type=float, default=1.0)
    value.add_argument("--residual-rank", type=int, default=16)
    value.add_argument("--radio-anchor-weight", type=float, default=1.0)
    value.add_argument("--radio-anchor-rows-per-step", type=int, default=2048)
    value.add_argument("--visual-teacher-root")
    value.add_argument("--alternating-visual-weight", type=float, default=0.0)
    value.add_argument("--visual-pixels-per-step", type=int, default=64)
    value.add_argument("--joint-snapshot-interval", type=int, default=50)
    value.add_argument("--optimizer-chunk-elements", type=int, default=1048576)
    value.add_argument("--seed", type=int, default=20260826)
    value.add_argument("--output", required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.arm != "oracle16" and (not args.field or not args.expected_field_sha256):
        raise ValueError("D512 arms require a hash-bound field")
    if (
        args.residual_rank <= 0
        or args.radio_anchor_rows_per_step <= 0
        or args.visual_pixels_per_step <= 0
        or args.alternating_visual_weight < 0
        or args.joint_snapshot_interval <= 0
        or args.optimizer_chunk_elements <= 0
    ):
        raise ValueError("writeback rank and RADIO anchor row budget must be positive")
    print(run(args))


if __name__ == "__main__":
    main()
