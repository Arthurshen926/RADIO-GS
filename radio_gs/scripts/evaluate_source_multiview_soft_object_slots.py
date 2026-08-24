#!/usr/bin/env python3
"""Source-heldout sentinel for joint multi-view soft object slots.

The slots are mapping-time teacher variables, not deployed Gaussian state.  A
fixed pair of source-view residues fits slots jointly from official SAM3 masks;
a disjoint calibration residue selects one global readout and a final residue
measures descriptor-to-slot transfer.  Invisible rows are always unknown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import (
    _proposal_soft_support,
    _proposal_support,
    compose_membership_query_features,
    visible_membership_target,
)
from radio_gs.utils.immutable_artifacts import (
    load_sha_bound_project_checkpoint_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


def noisy_or(assignment: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Compose aligned [B,K] assignment and slot probabilities."""

    if assignment.ndim != 2 or slots.ndim != 2 or assignment.shape != slots.shape:
        raise ValueError("assignment and slot probability matrices must align")
    survival = torch.log1p(-(assignment * slots).clamp(max=1.0 - 1e-6)).sum(-1)
    return 1.0 - torch.exp(survival)


def _load(path: str, digest: str, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    value, actual, source = load_sha_bound_project_checkpoint_mapping(
        path, expected_sha256=digest, map_location="cpu", label=label
    )
    return dict(value), {"path": str(source), "sha256": actual}


def _balanced_pairs(
    supports: list[torch.Tensor],
    support_values: list[torch.Tensor],
    proposal_views: torch.Tensor,
    observed: torch.Tensor,
    selected: torch.Tensor,
    *,
    positive_cap: int,
    negative_cap: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    ps: list[torch.Tensor] = []
    rs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    ws: list[torch.Tensor] = []
    for proposal in torch.where(selected)[0].tolist():
        positive, values = supports[proposal], support_values[proposal]
        visible = torch.where(observed[int(proposal_views[proposal])])[0]
        if not positive.numel() or not visible.numel():
            continue
        if positive.numel() > int(positive_cap):
            take = torch.randperm(positive.numel(), generator=generator)[:positive_cap]
            positive, values = positive[take], values[take]
        lookup = torch.zeros(observed.shape[1], dtype=torch.bool)
        lookup[positive] = True
        negative = visible[~lookup[visible]]
        if negative.numel() > int(negative_cap):
            take = torch.randperm(negative.numel(), generator=generator)[:negative_cap]
            negative = negative[take]
        row = torch.cat((positive, negative))
        label = torch.cat((values.float(), torch.zeros(negative.numel())))
        weight = torch.cat((
            torch.full((positive.numel(),), 0.5 / positive.numel()),
            torch.full((negative.numel(),), 0.5 / max(negative.numel(), 1)),
        ))
        ps.append(torch.full((row.numel(),), proposal, dtype=torch.long))
        rs.append(row)
        ys.append(label)
        ws.append(weight)
    if not ps:
        raise ValueError("slot training split has no visible pairs")
    return tuple(torch.cat(values) for values in (ps, rs, ys, ws))


def _farthest_initialization(features: torch.Tensor, count: int) -> torch.Tensor:
    features = F.normalize(features.float(), dim=-1)
    chosen = [0]
    nearest = torch.full((features.shape[0],), float("inf"))
    for _ in range(1, min(int(count), features.shape[0])):
        distance = 1.0 - features @ features[chosen[-1]]
        nearest = torch.minimum(nearest, distance)
        chosen.append(int(nearest.argmax()))
    return torch.tensor(chosen, dtype=torch.long)


def _iou(score: torch.Tensor, truth: torch.Tensor, threshold: float) -> float:
    prediction = score >= float(threshold)
    intersection = int((prediction & truth).sum())
    union = int((prediction | truth).sum())
    return float(intersection / union) if union else 1.0


@torch.inference_mode()
def _evaluate(
    proposals: torch.Tensor,
    descriptors: torch.Tensor,
    slot_descriptors: torch.Tensor,
    slot_probability: torch.Tensor,
    proposal_views: torch.Tensor,
    observed: torch.Tensor,
    supports: list[torch.Tensor],
    *,
    topk: int,
    threshold: float,
) -> list[float]:
    values: list[float] = []
    for proposal in proposals.tolist():
        visible = torch.where(observed[int(proposal_views[proposal])])[0]
        truth = visible_membership_target(
            visible, supports[proposal], num_rows=observed.shape[1]
        )
        scores = slot_descriptors @ descriptors[proposal]
        selected = scores.topk(k=min(int(topk), scores.numel())).indices
        local = slot_probability.index_select(0, selected).index_select(1, visible)
        prediction = 1.0 - torch.prod(1.0 - local, dim=0)
        values.append(_iou(prediction, truth, threshold))
    return values


def run(args: argparse.Namespace) -> dict[str, Any]:
    membership, membership_record = _load(
        args.membership, args.expected_membership_sha256, "source SAM3 memberships"
    )
    language, language_record = _load(
        args.language_teacher, args.expected_language_teacher_sha256,
        "source native SigLIP2 mask teacher",
    )
    rows = torch.as_tensor(membership["row_indices"]).long()
    proposals = torch.as_tensor(membership["proposal_indices"]).long()
    weights = torch.as_tensor(membership["weights"]).float()
    proposal_views = torch.as_tensor(membership["proposal_view_indices"]).long()
    observed = torch.as_tensor(membership["view_observed"]).bool()
    proposal_count = int(membership["num_proposals"])
    row_count = int(membership["num_rows"])
    if observed.shape[1] != row_count:
        raise ValueError("source visibility and membership carriers differ")
    hard_selected = weights >= float(args.evaluation_membership_threshold)
    hard_support = _proposal_support(
        rows[hard_selected], proposals[hard_selected], proposal_count
    )
    soft_support, soft_values = _proposal_soft_support(
        rows, proposals, weights, proposal_count
    )
    if bool(args.hard_training_targets):
        soft_support = hard_support
        soft_values = [torch.ones(value.numel()) for value in hard_support]
    valid = torch.tensor([value.numel() > 0 for value in hard_support])
    residues = torch.remainder(proposal_views, int(args.split_stride))
    training = valid & (residues != int(args.calibration_residue)) & (
        residues != int(args.evaluation_residue)
    )
    calibration = torch.where(valid & (residues == int(args.calibration_residue)))[0]
    evaluation = torch.where(valid & (residues == int(args.evaluation_residue)))[0]
    if min(int(training.sum()), calibration.numel(), evaluation.numel()) <= 0:
        raise ValueError("fixed source split has an empty cohort")
    descriptor = F.normalize(
        0.75 * F.normalize(torch.as_tensor(language["descriptors"]).float(), dim=-1)
        + 0.25 * F.normalize(
            torch.as_tensor(language["context_descriptors"]).float(), dim=-1
        ), dim=-1,
    )
    if descriptor.shape != (proposal_count, 1536):
        raise ValueError("proposal descriptor domain differs")

    pair_p, pair_r, pair_y, pair_w = _balanced_pairs(
        soft_support, soft_values, proposal_views, observed, training,
        positive_cap=int(args.positive_cap), negative_cap=int(args.negative_cap),
        seed=int(args.seed),
    )
    train_ids = torch.where(training)[0]
    centers_local = _farthest_initialization(
        descriptor.index_select(0, train_ids), int(args.num_slots)
    )
    centers = descriptor.index_select(0, train_ids.index_select(0, centers_local))
    slot_count = int(centers.shape[0])
    nearest = (descriptor @ centers.T).argmax(dim=1)
    assignment_init = torch.full((proposal_count, slot_count), -4.0)
    assignment_init.scatter_(1, nearest[:, None], 4.0)
    slot_init = torch.full((slot_count, row_count), -4.0)
    for proposal in train_ids.tolist():
        slot = int(nearest[proposal])
        support_rows = soft_support[proposal]
        if support_rows.numel():
            probability = soft_values[proposal].clamp(0.02, 0.98)
            logits = torch.logit(probability)
            slot_init[slot, support_rows] = torch.maximum(
                slot_init[slot, support_rows], logits
            )

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    assignment_logits = nn.Parameter(assignment_init.to(device))
    slot_logits = nn.Parameter(slot_init.to(device))
    slot_descriptor = nn.Parameter(centers.to(device).clone())
    optimizer = torch.optim.AdamW(
        (assignment_logits, slot_logits, slot_descriptor),
        lr=float(args.learning_rate), weight_decay=float(args.weight_decay),
    )
    generator = torch.Generator().manual_seed(int(args.seed) + 1)
    best_loss = float("inf")
    best: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for step in range(1, int(args.steps) + 1):
        index = torch.randint(
            pair_r.numel(),
            (min(int(args.batch_size), pair_r.numel()),),
            generator=generator,
        )
        bp, br = pair_p[index].to(device), pair_r[index].to(device)
        assignment = torch.sigmoid(assignment_logits.index_select(0, bp))
        slots = torch.sigmoid(slot_logits.index_select(1, br).T)
        prediction = noisy_or(assignment, slots)
        loss_values = F.binary_cross_entropy(
            prediction.clamp(1e-6, 1.0 - 1e-6), pair_y[index].to(device),
            reduction="none",
        )
        local_weight = pair_w[index].to(device)
        mask_loss = (loss_values * local_weight).sum() / local_weight.sum().clamp_min(1e-8)
        selected_proposals = train_ids[
            torch.randint(train_ids.numel(), (min(64, train_ids.numel()),), generator=generator)
        ].to(device)
        selected_assignment = torch.sigmoid(
            assignment_logits.index_select(0, selected_proposals)
        )
        reconstructed = F.normalize(
            selected_assignment @ F.normalize(slot_descriptor, dim=-1), dim=-1
        )
        identity_loss = 1.0 - (
            reconstructed * descriptor.index_select(0, selected_proposals.cpu()).to(device)
        ).sum(-1).mean()
        sampled_slots = torch.sigmoid(slot_logits.index_select(1, br.unique()))
        gram = sampled_slots @ sampled_slots.T / max(sampled_slots.shape[1], 1)
        overlap = (gram.sum() - gram.diag().sum()) / max(slot_count * (slot_count - 1), 1)
        sparsity = selected_assignment.mean()
        loss = (
            mask_loss
            + float(args.identity_weight) * identity_loss
            + float(args.sparsity_weight) * sparsity
            + float(args.overlap_weight) * overlap
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (assignment_logits, slot_logits, slot_descriptor), 5.0
        )
        optimizer.step()
        value = float(loss.detach())
        # The slot table is millions of values.  Checkpoint it only at the
        # declared validation cadence; copying it to CPU on every improving
        # minibatch makes an otherwise small GPU optimization CPU-bound.
        checkpoint_step = step % int(args.log_interval) == 0 or step == int(args.steps)
        if checkpoint_step and value < best_loss:
            best_loss = value
            best = (
                assignment_logits.detach().clone(),
                slot_logits.detach().clone(),
                slot_descriptor.detach().clone(),
            )
        if checkpoint_step:
            history.append({
                "step": step, "loss": value, "mask_loss": float(mask_loss.detach()),
                "identity_loss": float(identity_loss.detach()),
            })
    if best is None:
        raise RuntimeError("slot optimization did not run")
    assignment = torch.sigmoid(best[0].float()).cpu()
    slot_probability = torch.sigmoid(best[1].float()).cpu()
    # Re-estimate the query-facing slot identity from source proposal teachers;
    # this removes arbitrary optimizer scale from descriptor matching.
    slot_identity = F.normalize(assignment[training].T @ descriptor[training], dim=-1)
    candidates: list[tuple[float, int, float]] = []
    calibration_table: dict[str, float] = {}
    for topk in tuple(int(value) for value in args.topk.split(",")):
        for threshold in tuple(float(value) for value in args.thresholds.split(",")):
            values = _evaluate(
                calibration, descriptor, slot_identity, slot_probability,
                proposal_views, observed, hard_support, topk=topk, threshold=threshold,
            )
            macro = float(torch.tensor(values).mean())
            calibration_table[f"k{topk}_t{threshold:g}"] = macro
            candidates.append((macro, -topk, -threshold))
    _, negative_topk, negative_threshold = max(candidates)
    selected_topk, selected_threshold = -negative_topk, -negative_threshold
    evaluation_values = _evaluate(
        evaluation, descriptor, slot_identity, slot_probability, proposal_views,
        observed, hard_support, topk=selected_topk, threshold=selected_threshold,
    )
    evaluation_iou = float(torch.tensor(evaluation_values).mean())
    passed = evaluation_iou >= float(args.minimum_heldout_iou)
    output_path = Path(args.output).expanduser().resolve()
    payload = {
        "schema": "radio_gs.source_multiview_soft_object_slots.v1",
        "schema_version": 1,
        "scene": str(args.scene),
        "slot_membership": slot_probability.half(),
        "slot_identity": slot_identity.half(),
        "metadata": {
            "mapping_time_teacher_only": True,
            "deployed_per_gaussian_state_added": False,
            "source_only": True,
            "benchmark_masks_opened": False,
            "hard_training_targets": bool(args.hard_training_targets),
            "view_split": f"mod_{args.split_stride}:train=other,cal={args.calibration_residue},eval={args.evaluation_residue}",
            "inputs": {"membership": membership_record, "language_teacher": language_record},
        },
    }
    write_torch_noclobber(output_path, payload)
    report = {
        "status": "source_gate_pass" if passed else "source_gate_fail",
        "scene": str(args.scene),
        "num_slots": slot_count,
        "hard_training_targets": bool(args.hard_training_targets),
        "cohorts": {"training": int(training.sum()), "calibration": int(calibration.numel()), "evaluation": int(evaluation.numel())},
        "best_train_loss": best_loss,
        "calibration_macro_iou": calibration_table,
        "selected_readout": {"topk": selected_topk, "threshold": selected_threshold},
        "source_heldout_macro_iou": evaluation_iou,
        "source_heldout_median_iou": float(torch.tensor(evaluation_values).median()),
        "source_heldout_minimum_iou": min(evaluation_values),
        "comparison": {"current_frozen_l512_membership": 0.13913016021251678, "delta": evaluation_iou - 0.13913016021251678},
        "gate": {"minimum_heldout_iou": float(args.minimum_heldout_iou), "passed": passed},
        "history": history,
    }
    write_frozen_json(output_path.with_suffix(output_path.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--expected-membership-sha256", required=True)
    parser.add_argument("--language-teacher", required=True)
    parser.add_argument("--expected-language-teacher-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-slots", type=int, default=32)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--identity-weight", type=float, default=0.25)
    parser.add_argument("--sparsity-weight", type=float, default=0.01)
    parser.add_argument("--overlap-weight", type=float, default=0.05)
    parser.add_argument("--positive-cap", type=int, default=1024)
    parser.add_argument("--negative-cap", type=int, default=2048)
    parser.add_argument("--split-stride", type=int, default=4)
    parser.add_argument("--calibration-residue", type=int, default=2)
    parser.add_argument("--evaluation-residue", type=int, default=3)
    parser.add_argument("--evaluation-membership-threshold", type=float, default=0.5)
    parser.add_argument("--hard-training-targets", action="store_true")
    parser.add_argument("--topk", default="1,2,4")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--minimum-heldout-iou", type=float, default=0.16)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260824)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
