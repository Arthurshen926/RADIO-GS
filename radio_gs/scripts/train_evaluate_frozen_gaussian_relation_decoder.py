#!/usr/bin/env python3
"""Source-heldout gate for identity-seeded Gaussian object membership.

Text/mask language selects a small frozen set of identity seeds.  One global
decoder then predicts ``P(i same-object seed)`` from frozen L512 pairs and
relative 3-D position.  Source SAM same/different/unknown authority supplies
only known relation pairs; unknown and invisible evidence never becomes a
negative.  The field and all per-Gaussian state remain frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.models.frozen_gaussian_relation_decoder import (
    FrozenGaussianRelationDecoder,
)
from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import (
    _calibrate_threshold,
    _iou_from_scores,
    _proposal_support,
    visible_membership_target,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_sha_bound_project_checkpoint_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


def sample_gaussian_relation_pairs(
    *,
    edge_left: torch.Tensor,
    edge_right: torch.Tensor,
    edge_relation: torch.Tensor,
    supports: list[torch.Tensor],
    selected_proposals: torch.Tensor,
    samples_per_edge: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift known proposal relations to sampled Gaussian-pair authority."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    left_rows: list[torch.Tensor] = []
    right_rows: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for left, right, relation in zip(
        edge_left.tolist(), edge_right.tolist(), edge_relation.tolist()
    ):
        if relation not in (0, 1) or not (
            bool(selected_proposals[int(left)])
            and bool(selected_proposals[int(right)])
        ):
            continue
        left_support, right_support = supports[int(left)], supports[int(right)]
        if not left_support.numel() or not right_support.numel():
            continue
        count = int(samples_per_edge)
        left_index = torch.randint(left_support.numel(), (count,), generator=generator)
        right_index = torch.randint(right_support.numel(), (count,), generator=generator)
        left_rows.append(left_support[left_index])
        right_rows.append(right_support[right_index])
        labels.append(torch.full((count,), float(relation)))
    if not labels:
        raise ValueError("training split has no known Gaussian relation pair")
    output = tuple(torch.cat(values) for values in (left_rows, right_rows, labels))
    if not bool((output[2] == 0).any()) or not bool((output[2] == 1).any()):
        raise ValueError("Gaussian relation training requires same and different pairs")
    return output


@torch.inference_mode()
def identity_seeded_relation_score(
    model: FrozenGaussianRelationDecoder,
    latent: torch.Tensor,
    xyz: torch.Tensor,
    baseline_feature: torch.Tensor,
    query: torch.Tensor,
    visible: torch.Tensor,
    *,
    device: torch.device,
    scene_extent: torch.Tensor,
    seed_count: int,
    seed_logit_scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginalize same-object scores over fixed language identity seeds."""

    rows = torch.as_tensor(visible).long().reshape(-1)
    identity = baseline_feature[rows] @ F.normalize(query.float(), dim=0)
    count = min(int(seed_count), int(rows.numel()))
    values, positions = torch.topk(identity, count)
    seeds = rows[positions]
    probability = torch.softmax(
        float(seed_logit_scale) * (values - values.max()), dim=0
    ).to(device)
    parts: list[torch.Tensor] = []
    seed_latent = latent[seeds].to(device)
    seed_xyz = xyz[seeds].to(device)
    for chunk in rows.split(int(chunk_size)):
        local_latent = latent[chunk].to(device)
        local_xyz = xyz[chunk].to(device)
        batch = int(chunk.numel())
        candidates = local_latent[:, None, :].expand(-1, count, -1).reshape(
            batch * count, -1
        )
        candidate_xyz = local_xyz[:, None, :].expand(-1, count, -1).reshape(
            batch * count, 3
        )
        references = seed_latent[None, :, :].expand(batch, -1, -1).reshape(
            batch * count, -1
        )
        reference_xyz = seed_xyz[None, :, :].expand(batch, -1, -1).reshape(
            batch * count, 3
        )
        same = torch.sigmoid(
            model(
                candidates,
                references,
                candidate_xyz,
                reference_xyz,
                scene_extent,
            )
        ).reshape(batch, count)
        parts.append((same * probability[None]).sum(dim=1).cpu())
    score = torch.cat(parts)
    for seed in seeds.tolist():
        score[rows == int(seed)] = 1.0
    return score, seeds


def train_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    membership, membership_sha, membership_path = load_sha_bound_project_checkpoint_mapping(
        args.membership,
        expected_sha256=args.expected_membership_sha256,
        map_location="cpu",
        label="source SAM Gaussian membership",
    )
    authority, authority_sha, authority_path = load_sha_bound_project_checkpoint_mapping(
        args.relation_authority,
        expected_sha256=args.expected_relation_authority_sha256,
        map_location="cpu",
        label="source SAM ternary relation authority",
    )
    teacher, teacher_sha, teacher_path = load_sha_bound_project_checkpoint_mapping(
        args.language_teacher,
        expected_sha256=args.expected_language_teacher_sha256,
        map_location="cpu",
        label="source mask language teacher",
    )
    query_cache, query_sha, query_path = load_sha_bound_project_checkpoint_mapping(
        args.query_cache,
        expected_sha256=args.expected_query_cache_sha256,
        map_location="cpu",
        label="primitive query cache",
    )
    field_path = Path(args.field).expanduser().resolve(strict=True)
    field, _payload, _signature = load_factorized_canonical_field_checkpoint(
        field_path, map_location="cpu", expected_sha256=args.expected_field_sha256
    )
    if membership.get("metadata", {}).get("query_independent_proposal_set") is not True:
        raise ValueError("proposal membership is not query-independent")
    if authority.get("metadata", {}).get("source_only") is not True:
        raise ValueError("relation authority is not source-only")
    if authority.get("metadata", {}).get("benchmark_masks_opened") is not False:
        raise ValueError("relation authority opened benchmark masks")
    if teacher.get("metadata", {}).get("source_only") is not True:
        raise ValueError("language teacher is not source-only")

    rows = torch.as_tensor(membership["row_indices"]).long()
    proposals = torch.as_tensor(membership["proposal_indices"]).long()
    membership_weight = torch.as_tensor(membership["weights"]).float()
    proposal_views = torch.as_tensor(membership["proposal_view_indices"]).long()
    view_observed = torch.as_tensor(membership["view_observed"]).bool()
    proposal_count = int(membership["num_proposals"])
    supports = _proposal_support(
        rows[membership_weight >= float(args.membership_threshold)],
        proposals[membership_weight >= float(args.membership_threshold)],
        proposal_count,
    )
    valid = torch.tensor([value.numel() > 0 for value in supports], dtype=torch.bool)
    heldout = (
        torch.remainder(proposal_views, int(args.holdout_stride))
        == int(args.holdout_residue)
    ) & valid
    training = (~heldout) & valid
    authority_views = torch.as_tensor(authority["proposal_view_indices"]).long()
    if not torch.equal(authority_views, proposal_views):
        raise ValueError("relation and membership proposal axes differ")
    edge_left = torch.as_tensor(authority["edge_left"]).long()
    edge_right = torch.as_tensor(authority["edge_right"]).long()
    edge_relation = torch.as_tensor(authority["edge_relation"]).to(torch.int8)
    left_rows, right_rows, pair_label = sample_gaussian_relation_pairs(
        edge_left=edge_left,
        edge_right=edge_right,
        edge_relation=edge_relation,
        supports=supports,
        selected_proposals=training,
        samples_per_edge=int(args.samples_per_edge),
        seed=int(args.seed),
    )

    latent = field.local_codes.detach().cpu().float()
    xyz = torch.as_tensor(query_cache["xyz"]).detach().cpu().float()
    baseline_feature = F.normalize(
        torch.as_tensor(
            query_cache.get("features", query_cache.get("summary_features"))
        ).detach().cpu().float(),
        dim=-1,
    )
    descriptors = F.normalize(torch.as_tensor(teacher["descriptors"]).float(), dim=-1)
    contexts = F.normalize(torch.as_tensor(teacher["context_descriptors"]).float(), dim=-1)
    language = F.normalize(0.75 * descriptors + 0.25 * contexts, dim=-1)
    if (
        latent.shape != (xyz.shape[0], 512)
        or baseline_feature.shape != (xyz.shape[0], 1536)
        or language.shape != (proposal_count, 1536)
        or view_observed.shape[1] != xyz.shape[0]
    ):
        raise ValueError("field, proposal, and query row domains differ")
    scene_extent = (xyz.amax(0) - xyz.amin(0)).clamp_min(1e-6)
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    model = FrozenGaussianRelationDecoder(hidden_dim=int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    positive = torch.where(pair_label == 1)[0]
    negative = torch.where(pair_label == 0)[0]
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + 1)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    half_batch = max(1, int(args.batch_size) // 2)
    for _step in range(int(args.steps)):
        selected = torch.cat(
            (
                positive[torch.randint(positive.numel(), (half_batch,), generator=generator)],
                negative[torch.randint(negative.numel(), (half_batch,), generator=generator)],
            )
        )
        logits = model(
            latent[left_rows[selected]].to(device),
            latent[right_rows[selected]].to(device),
            xyz[left_rows[selected]].to(device),
            xyz[right_rows[selected]].to(device),
            scene_extent.to(device),
        )
        loss = F.binary_cross_entropy_with_logits(logits, pair_label[selected].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Gaussian relation decoder did not train")
    model.load_state_dict(best_state)
    model.eval()

    def evaluate_proposals(selected: torch.Tensor) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor]],
        list[tuple[torch.Tensor, torch.Tensor]],
        int,
        int,
    ]:
        relation_examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        baseline_examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        seed_hits, seed_total = 0, 0
        for proposal in torch.where(selected)[0].tolist():
            visible = torch.where(view_observed[int(proposal_views[proposal])])[0]
            truth = visible_membership_target(
                visible, supports[proposal], num_rows=field.num_gaussians
            )
            score, seeds = identity_seeded_relation_score(
                model,
                latent,
                xyz,
                baseline_feature,
                language[proposal],
                visible,
                device=device,
                scene_extent=scene_extent.to(device),
                seed_count=int(args.identity_seeds),
                seed_logit_scale=float(args.identity_seed_logit_scale),
                chunk_size=int(args.eval_chunk_size),
            )
            relation_examples.append((score, truth))
            baseline_examples.append((baseline_feature[visible] @ language[proposal], truth))
            seed_hits += int(any(int(seed) in set(supports[proposal].tolist()) for seed in seeds))
            seed_total += 1
        return relation_examples, baseline_examples, seed_hits, seed_total

    calibration = torch.where(training)[0]
    if calibration.numel() > int(args.calibration_proposals):
        positions = torch.linspace(
            0, calibration.numel() - 1, int(args.calibration_proposals)
        ).round().long()
        calibration_mask = torch.zeros_like(training)
        calibration_mask[calibration[positions]] = True
    else:
        calibration_mask = training
    relation_calibration, baseline_calibration, train_seed_hits, train_seed_total = (
        evaluate_proposals(calibration_mask)
    )
    relation_threshold, relation_train_iou = _calibrate_threshold(
        relation_calibration, int(args.threshold_candidates)
    )
    baseline_threshold, baseline_train_iou = _calibrate_threshold(
        baseline_calibration, int(args.threshold_candidates)
    )
    relation_heldout, baseline_heldout, heldout_seed_hits, heldout_seed_total = (
        evaluate_proposals(heldout)
    )
    relation_iou = float(
        torch.tensor(
            [
                _iou_from_scores(score, target, relation_threshold)
                for score, target in relation_heldout
            ]
        ).mean()
    )
    baseline_iou = float(
        torch.tensor(
            [
                _iou_from_scores(score, target, baseline_threshold)
                for score, target in baseline_heldout
            ]
        ).mean()
    )
    passed = (
        relation_iou >= float(args.minimum_heldout_iou)
        and relation_iou >= baseline_iou + float(args.minimum_gain)
    )
    output = Path(args.output).expanduser().resolve()
    inputs = {
        "field": file_record(field_path),
        "membership": {"path": str(membership_path), "sha256": membership_sha},
        "relation_authority": {"path": str(authority_path), "sha256": authority_sha},
        "language_teacher": {"path": str(teacher_path), "sha256": teacher_sha},
        "query_cache": {"path": str(query_path), "sha256": query_sha},
    }
    write_torch_noclobber(
        output,
        {
            "schema": "radio_gs.frozen_l512_gaussian_relation_decoder.v1",
            "schema_version": 1,
            "scene": str(args.scene),
            "state_dict": best_state,
            "threshold": relation_threshold,
            "metadata": {
                "field_frozen": True,
                "per_gaussian_parameters_added": False,
                "identity_owner": "primitive_siglip_topk_seed_marginal",
                "extent_owner": "global_gaussian_pair_same_object_decoder",
                "relation_authority": "source_sam_same_different_known_unknown_excluded",
                "view_split": f"source_view_mod_{args.holdout_stride}_eq_{args.holdout_residue}",
                "benchmark_masks_opened": False,
                "evaluation_rgb_opened": False,
                "text_queries_opened": False,
                "inputs": inputs,
            },
        },
    )
    report = {
        "status": "source_gate_pass" if passed else "source_gate_fail",
        "scene": str(args.scene),
        "training_proposals": int(training.sum()),
        "heldout_proposals": int(heldout.sum()),
        "training_relation_pairs": int(pair_label.numel()),
        "training_same_pairs": int((pair_label == 1).sum()),
        "training_different_pairs": int((pair_label == 0).sum()),
        "best_train_loss": best_loss,
        "source_train_calibration": {
            "relation_threshold": relation_threshold,
            "relation_iou": relation_train_iou,
            "primitive_similarity_threshold": baseline_threshold,
            "primitive_similarity_iou": baseline_train_iou,
            "identity_seed_hit_rate": train_seed_hits / max(train_seed_total, 1),
        },
        "source_heldout_macro_iou": {
            "primitive_similarity": baseline_iou,
            "gaussian_relation": relation_iou,
            "delta": relation_iou - baseline_iou,
            "identity_seed_hit_rate": heldout_seed_hits / max(heldout_seed_total, 1),
        },
        "gate": {
            "minimum_heldout_iou": float(args.minimum_heldout_iou),
            "minimum_gain": float(args.minimum_gain),
            "passed": passed,
        },
        "output": file_record(output),
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--expected-field-sha256", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--expected-membership-sha256", required=True)
    parser.add_argument("--relation-authority", required=True)
    parser.add_argument("--expected-relation-authority-sha256", required=True)
    parser.add_argument("--language-teacher", required=True)
    parser.add_argument("--expected-language-teacher-sha256", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--samples-per-edge", type=int, default=4)
    parser.add_argument("--membership-threshold", type=float, default=0.5)
    parser.add_argument("--identity-seeds", type=int, default=3)
    parser.add_argument("--identity-seed-logit-scale", type=float, default=16.0)
    parser.add_argument("--holdout-stride", type=int, default=4)
    parser.add_argument("--holdout-residue", type=int, default=3)
    parser.add_argument("--calibration-proposals", type=int, default=32)
    parser.add_argument("--threshold-candidates", type=int, default=64)
    parser.add_argument("--eval-chunk-size", type=int, default=16384)
    parser.add_argument("--minimum-heldout-iou", type=float, default=0.20)
    parser.add_argument("--minimum-gain", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260823)
    print(json.dumps(train_and_evaluate(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
