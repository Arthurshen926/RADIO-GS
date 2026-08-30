"""Fit the same 11-parameter posterior on real exact-render source masks."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch
from torch import distributed as dist
from torch import nn

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.evaluation.evaluate_exact_render_posterior import _input_paths
from radio_gs.v3.query.calibrated_posterior import NullCalibratedPosterior
from radio_gs.v3.query.interface import load_query_interface
from radio_gs.v3.query.packet import QueryPacket
from radio_gs.v3.training.instance_upper_bound import (
    sha256_file,
    validate_source_only_inputs,
)
from radio_gs.v3.training.run_instance_upper_bound import load_episodes


class _FeatureCalibrator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.posterior = NullCalibratedPosterior()

    def forward(
        self, positive_features: torch.Tensor, negative_features: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.posterior.logit_from_features(positive_features, negative_features)
        )


def _exact_mask_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    known: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    score = torch.as_tensor(probability).float().reshape(-1).clamp(1e-6, 1 - 1e-6)
    truth = torch.as_tensor(target, device=score.device).bool().reshape(-1)
    authority = torch.as_tensor(known, device=score.device).bool().reshape(-1)
    if score.shape != truth.shape or truth.shape != authority.shape or not bool(authority.any()):
        raise ValueError("exact-render calibrator mask axes differ")
    positive = truth & authority
    negative = ~truth & authority
    if not bool(negative.any()):
        raise ValueError("exact-render calibrator lacks explicit negative pixels")
    if not bool(positive.any()):
        background = score[negative]
        loss = -(1.0 - background).log().mean()
        loss = loss + background.square().mean() + background.mean().square()
        return loss, True
    balanced_bce = -0.5 * (
        score[positive].log().mean() + (1.0 - score[negative]).log().mean()
    )
    known_score = score[authority]
    known_truth = truth[authority].float()
    dice = 1.0 - (2.0 * (known_score * known_truth).sum() + 1.0) / (
        known_score.sum() + known_truth.sum() + 1.0
    )
    brier = (known_score - known_truth).square().mean()
    return balanced_bce + dice + brier, False


def _source_scene(
    evidence_path: Path,
    *,
    device: torch.device,
    topk: int,
    scale: float,
    temperature: float,
    posterior_chunk_size: int,
) -> tuple[dict, list[torch.Tensor], list[torch.Tensor], list[dict]]:
    evidence = torch.load(evidence_path, map_location="cpu")
    paths = _input_paths(evidence, evidence_path)
    membership = torch.load(paths["membership"], map_location="cpu")
    authority = torch.load(paths["authority"], map_location="cpu")
    text = torch.load(paths["text_embeddings"], map_location="cpu")
    validate_source_only_inputs(membership, authority)
    interface = load_query_interface(
        paths["scene_state"],
        device=device,
        text_negative_path=paths["text_negatives"],
        text_logit_scale=10.0,
    )
    episodes, supports = load_episodes(membership, authority)
    views = torch.as_tensor(authority["proposal_view_indices"]).long()
    states = torch.as_tensor(authority["query_state"]).to(torch.int8)
    valid = torch.tensor([rows.numel() > 0 for rows, _weights in supports])
    lookup = {str(name).casefold(): index for index, name in enumerate(text["queries"])}
    embeddings = torch.as_tensor(text["embeddings"]).float()
    examples: list[dict] = []
    active_queries: set[int] = set()
    train_views = torch.unique(views[(views % 4 == 1) | (views % 4 == 2)])
    for column, raw_name in enumerate(authority["query_names"]):
        for view in train_views.tolist():
            in_view = (views == int(view)) & valid
            positive = torch.where(in_view & (states[:, column] == 1))[0]
            negative = torch.where(in_view & (states[:, column] == 0))[0]
            if not negative.numel():
                continue
            representative = episodes[int(positive[0] if positive.numel() else negative[0])]
            target = (
                torch.stack([episodes[int(index)].target for index in positive]).any(0)
                if positive.numel()
                else torch.zeros_like(representative.target)
            )
            negative_mask = torch.stack([
                episodes[int(index)].target for index in negative
            ]).any(0) & ~target
            if not bool(negative_mask.any()):
                continue
            examples.append({
                "query_index": column,
                "query_name": str(raw_name),
                "view": int(view),
                "episode": representative,
                "target": target.flatten(),
                "known": (target | negative_mask).flatten(),
                "empty": not bool(target.any()),
            })
            active_queries.add(column)
    positive_examples = [item for item in examples if not item["empty"]]
    empty_examples = [item for item in examples if item["empty"]]
    if not positive_examples or not empty_examples:
        raise ValueError("exact-render calibrator scene lacks a positive or empty cohort")
    feature_builder = NullCalibratedPosterior().to(device).eval()
    positive_features: list[torch.Tensor] = []
    negative_features: list[torch.Tensor] = []
    for column, raw_name in enumerate(authority["query_names"]):
        if column not in active_queries:
            positive_features.append(torch.empty(0, 7, device=device))
            negative_features.append(torch.empty(0, 3, device=device))
            continue
        token_index = lookup.get(str(raw_name).casefold())
        if token_index is None:
            raise ValueError(f"exact-render calibrator lacks query token: {raw_name}")
        packet = QueryPacket("text", embeddings[token_index])
        identity, null, unknown = interface.semantic_text_evidence(packet)
        base, returned_identity = interface.posterior_from_packet(
            packet,
            scale=scale,
            topk=topk,
            temperature=temperature,
            posterior_chunk_size=posterior_chunk_size,
            text_anchor_policy="positive",
        )
        if not torch.equal(identity, returned_identity):
            raise RuntimeError("exact-render calibrator changed clean identity")
        instance, boundary = interface.refine_instance_with_boundary(
            base,
            maximum_logit_residual=interface.maximum_boundary_logit_residual,
        )
        positive, negative = feature_builder.evidence_features(
            identity=identity,
            instance=instance,
            null=null,
            negative=torch.sigmoid((null - identity) * 10.0),
            unknown=unknown,
            boundary=boundary,
            reliability=interface.reliability,
        )
        positive_features.append(positive.detach())
        negative_features.append(negative.detach())
    receipt = {
        "scene": evidence["scene"],
        "evidence": {"path": str(evidence_path), "sha256": sha256_file(evidence_path)},
        "positive_train_examples": len(positive_examples),
        "empty_train_examples": len(empty_examples),
    }
    return receipt, positive_features, negative_features, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", action="append", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--positive-examples-per-step", type=int, default=2)
    parser.add_argument("--empty-examples-per-step", type=int, default=2)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--posterior-chunk-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--snapshot-interval", type=int, default=25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if (
        world_size != len(args.evidence)
        or world_size < 2
        or args.steps <= 0
        or args.learning_rate <= 0
        or args.positive_examples_per_step <= 0
        or args.empty_examples_per_step <= 0
    ):
        raise ValueError("exact-render calibrator requires one distributed rank per scene")
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    evidence_paths = [Path(value).resolve(strict=True) for value in args.evidence]
    receipt, positive_features, negative_features, examples = _source_scene(
        evidence_paths[rank],
        device=device,
        topk=args.topk,
        scale=args.scale,
        temperature=args.temperature,
        posterior_chunk_size=args.posterior_chunk_size,
    )
    positive_examples = [item for item in examples if not item["empty"]]
    empty_examples = [item for item in examples if item["empty"]]
    torch.manual_seed(args.seed)
    module = _FeatureCalibrator().to(device)
    model = nn.parallel.DistributedDataParallel(module, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    rng = random.Random(args.seed + rank)
    history = []
    best_loss = float("inf")
    best_state = None
    for step in range(args.steps):
        selected = (
            rng.sample(
                positive_examples,
                k=min(args.positive_examples_per_step, len(positive_examples)),
            )
            + rng.sample(
                empty_examples,
                k=min(args.empty_examples_per_step, len(empty_examples)),
            )
        )
        positive = torch.cat([
            positive_features[int(item["query_index"])] for item in selected
        ])
        negative = torch.cat([
            negative_features[int(item["query_index"])] for item in selected
        ])
        counts = [positive_features[int(item["query_index"])].shape[0] for item in selected]
        posteriors = model(positive, negative).split(counts)
        positive_losses, empty_losses = [], []
        for posterior, item in zip(posteriors, selected):
            episode = item["episode"]
            rendered = episode.contribution_weights.to(device) * posterior[
                episode.gaussian_ids.to(device)
            ]
            image = posterior.new_zeros(item["target"].numel())
            image.index_add_(0, episode.pixel_ids.to(device), rendered)
            loss, empty = _exact_mask_loss(
                image.clamp(0, 1), item["target"], item["known"]
            )
            (empty_losses if empty else positive_losses).append(loss)
        loss = torch.stack(positive_losses).mean() + torch.stack(empty_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        global_loss = loss.detach().clone()
        dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
        global_loss /= world_size
        value = float(global_loss)
        if value < best_loss:
            best_loss = value
            if rank == 0:
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.module.posterior.state_dict().items()
                }
        if step == 0 or (step + 1) % args.snapshot_interval == 0 or step + 1 == args.steps:
            if rank == 0:
                record = {"step": step + 1, "distributed_train_loss": value}
                history.append(record)
                print(record, flush=True)
    gathered: list[dict | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, receipt)
    if rank == 0:
        if best_state is None:
            raise RuntimeError("exact-render calibrator produced no checkpoint")
        payload = {
            "schema": "radio_gs.sugm_v3.null_calibrated_posterior.v1",
            "state_dict": best_state,
            "history": history,
            "training_receipts": gathered,
            "feature_names": {
                "positive": [
                    "raw_positive_identity",
                    "raw_positive_anchor_d48_instance",
                    "signed_D16_magnitude_identity_minus_instance_contrast",
                    "r5_visual_write_authority",
                    "r5_coverage_confidence",
                    "r5_structural_confidence",
                    "r5_membership_strength",
                ],
                "negative": [
                    "canonical_null_similarity",
                    "canonical_null_over_positive_probability",
                    "semantic_or_r5_unknown",
                ],
            },
            "metadata": {
                "source_only": True,
                "source_train_residues": [1, 2],
                "source_dev_opened_for_fit": False,
                "source_audit_opened_for_fit": False,
                "unknown_pairs_used_as_negative": False,
                "target_rgb_opened": False,
                "benchmark_metrics_opened": False,
                "objective": "query_balanced_exact_MPR_mask_BCE_Dice_Brier_plus_empty_target_background_mass",
                "training_order": "gaussian_features_then_global_calibrator_sigmoid_then_exact_MPR_render",
                "parameter_count": sum(
                    value.numel() for value in model.module.posterior.parameters()
                ),
                "disabled_positive_feature_indices": [],
                "private_structure_policy": "retained_frozen_fresh_D48_plus_signed_D16",
                "same_gaussian_posterior_for_2d_and_3d": True,
                "global_threshold": 0.5,
                "seed": args.seed,
                "steps": args.steps,
                "learning_rate": args.learning_rate,
                "positive_examples_per_step_per_scene": args.positive_examples_per_step,
                "empty_examples_per_step_per_scene": args.empty_examples_per_step,
                "topk": args.topk,
                "scale": args.scale,
                "temperature": args.temperature,
                "distributed_scenes": world_size,
            },
        }
        output = Path(args.output).resolve()
        write_torch_noclobber(output, payload)
        print({"output": str(output), "sha256": sha256_file(output), "history": history[-1]})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()


__all__ = ["_FeatureCalibrator", "_exact_mask_loss"]
