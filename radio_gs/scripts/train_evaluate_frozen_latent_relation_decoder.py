#!/usr/bin/env python3
"""Train and source-holdout test a global decoder over frozen L512 latents.

This sentinel changes neither the Gaussian field nor its per-primitive state.
Source SAM proposals are pooled through exact-MPR memberships.  A single
decoder learns ternary-authority known same/different edges and mask-level
SigLIP2 targets on training source views, then reconstructs proposals from
held-out source views by retrieving a training-view proposal.  Benchmark
masks, evaluation RGB, and text queries are never opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.models.frozen_latent_relation_decoder import FrozenLatentRelationDecoder
from radio_gs.models.object_aware_universal_field_v2 import sparse_proposal_pool
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_sha_bound_project_checkpoint_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


def relation_auc(score: torch.Tensor, relation: torch.Tensor) -> float:
    """Mann--Whitney AUC with half credit for ties."""

    values = torch.as_tensor(score).detach().cpu().float()
    labels = torch.as_tensor(relation).detach().cpu().to(torch.int8)
    same, different = values[labels == 1], values[labels == 0]
    if not same.numel() or not different.numel():
        raise ValueError("relation AUC requires same and different examples")
    wins = values.new_zeros(())
    for chunk in same.split(4096):
        delta = chunk[:, None] - different[None, :]
        wins += (delta > 0).sum() + 0.5 * (delta == 0).sum()
    return float(wins / (same.numel() * different.numel()))


def support_iou(left: set[int], right: set[int]) -> float:
    union = len(left | right)
    return float(len(left & right) / union) if union else 1.0


def proposal_supports(
    row_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    num_proposals: int,
) -> list[set[int]]:
    supports = [set() for _ in range(int(num_proposals))]
    for row, proposal in zip(
        torch.as_tensor(row_indices).tolist(),
        torch.as_tensor(proposal_indices).tolist(),
    ):
        supports[int(proposal)].add(int(row))
    return supports


def visibility_normalized_track_posterior(
    relation_weights: torch.Tensor,
    train_support: torch.Tensor,
    train_visibility: torch.Tensor,
) -> torch.Tensor:
    """Estimate object occupancy without treating occlusion as negative evidence.

    For each query proposal and Gaussian, this is the relation-weighted fraction
    of *views in which that Gaussian was observable* whose matched proposal
    contains the Gaussian.  It is therefore a marginal track posterior rather
    than a hard union of partial masks.
    """

    relation = torch.as_tensor(relation_weights).detach().cpu().float()
    support = torch.as_tensor(train_support).detach().cpu().bool()
    visibility = torch.as_tensor(train_visibility).detach().cpu().bool()
    if (
        relation.ndim != 2
        or support.ndim != 2
        or visibility.shape != support.shape
        or relation.shape[1] != support.shape[0]
    ):
        raise ValueError("track posterior inputs do not align")
    if not bool(torch.isfinite(relation).all()) or bool((relation < 0).any()):
        raise ValueError("track relation weights must be finite and nonnegative")
    # Proposal support is meaningful only when its source view observes the
    # Gaussian.  Enforce the contract even if an upstream cache is malformed.
    supported_and_visible = support & visibility
    numerator = relation @ supported_and_visible.float()
    denominator = relation @ visibility.float()
    posterior = numerator / denominator.clamp_min(1e-8)
    posterior[denominator <= 0] = 0.0
    return posterior.clamp_(0.0, 1.0)


def soft_support_iou(
    posterior: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Per-row soft IoU on the observable domain."""

    probability = torch.as_tensor(posterior).float()
    truth = torch.as_tensor(target).bool()
    visible = torch.as_tensor(visibility).bool()
    if probability.shape != truth.shape or truth.shape != visible.shape:
        raise ValueError("soft-IoU inputs do not align")
    probability = probability * visible.float()
    truth = truth & visible
    intersection = (probability * truth.float()).sum(dim=1)
    union = (probability + truth.float() - probability * truth.float()).sum(dim=1)
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def balanced_relation_loss(logits: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
    """Class-balanced categorical log score for different/same/unknown."""

    labels = torch.as_tensor(relation, device=logits.device).to(torch.int8)
    if logits.shape != (labels.numel(), 3):
        raise ValueError("ternary relation logits must have shape [E,3]")
    # Authority values {-1,0,1} map to categorical indices {2,0,1}.
    target = labels.long().clone()
    target[labels == -1] = 2
    losses = []
    for class_index in range(3):
        selected = target == class_index
        if not bool(selected.any()):
            raise ValueError("training relation split requires all three classes")
        losses.append(F.cross_entropy(logits[selected], target[selected]))
    return torch.stack(losses).mean()


def conditional_same_score(logits: torch.Tensor) -> torch.Tensor:
    """P(same | relation is known), independent of the unknown logit."""

    values = torch.as_tensor(logits)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("ternary relation logits must have shape [E,3]")
    return torch.sigmoid(values[:, 1] - values[:, 0])


def _load(args: argparse.Namespace) -> dict[str, object]:
    membership, membership_sha, membership_path = load_sha_bound_project_checkpoint_mapping(
        args.membership,
        expected_sha256=args.expected_membership_sha256,
        map_location="cpu",
        label="source-SAM exact-MPR proposal membership",
    )
    authority, authority_sha, authority_path = load_sha_bound_project_checkpoint_mapping(
        args.relation_authority,
        expected_sha256=args.expected_relation_authority_sha256,
        map_location="cpu",
        label="source-only ternary relation authority",
    )
    teacher, teacher_sha, teacher_path = load_sha_bound_project_checkpoint_mapping(
        args.language_teacher,
        expected_sha256=args.expected_language_teacher_sha256,
        map_location="cpu",
        label="source-mask SigLIP2 language teacher",
    )
    query, query_sha, query_path = load_sha_bound_project_checkpoint_mapping(
        args.query_cache,
        expected_sha256=args.expected_query_cache_sha256,
        map_location="cpu",
        label="Method-v1 primitive query cache geometry",
    )
    field, _payload, _signature = load_factorized_canonical_field_checkpoint(
        args.field,
        map_location="cpu",
        expected_sha256=args.expected_field_sha256,
    )
    if membership.get("metadata", {}).get("query_independent_proposal_set") is not True:
        raise ValueError("membership proposal set is not query-independent")
    for label, value in (("relation", authority), ("language", teacher)):
        metadata = value.get("metadata", {})
        if metadata.get("source_only") is not True:
            raise ValueError(f"{label} authority is not source-only")
        if metadata.get("benchmark_masks_opened") is not False:
            raise ValueError(f"{label} authority opened benchmark masks")
    if field.local_codes.ndim != 2 or int(field.local_codes.shape[1]) != 512:
        raise ValueError("frozen relation sentinel requires the canonical L512 latent")
    xyz = torch.as_tensor(query.get("xyz")).detach().cpu().float()
    if xyz.shape != (field.num_gaussians, 3):
        raise ValueError("query-cache geometry and field row domains differ")
    if int(membership.get("num_rows", -1)) != field.num_gaussians:
        raise ValueError("membership and field row domains differ")
    return {
        "membership": membership,
        "authority": authority,
        "teacher": teacher,
        "field": field,
        "xyz": xyz,
        "records": {
            "field": file_record(Path(args.field).expanduser().resolve(strict=True)),
            "membership": {"path": str(membership_path), "sha256": membership_sha},
            "relation_authority": {"path": str(authority_path), "sha256": authority_sha},
            "language_teacher": {"path": str(teacher_path), "sha256": teacher_sha},
            "query_cache": {"path": str(query_path), "sha256": query_sha},
        },
    }


def train_and_evaluate(args: argparse.Namespace) -> dict[str, object]:
    loaded = _load(args)
    membership = loaded["membership"]
    authority = loaded["authority"]
    teacher = loaded["teacher"]
    field = loaded["field"]
    xyz = loaded["xyz"]
    assert isinstance(membership, dict) and isinstance(authority, dict)
    assert isinstance(teacher, dict) and isinstance(field, torch.nn.Module)
    assert isinstance(xyz, torch.Tensor)

    rows = torch.as_tensor(membership["row_indices"]).long()
    proposals = torch.as_tensor(membership["proposal_indices"]).long()
    weights = torch.as_tensor(membership["weights"]).float()
    views = torch.as_tensor(membership["proposal_view_indices"]).long()
    areas = torch.as_tensor(membership["proposal_area_fraction"]).float()
    view_observed = torch.as_tensor(membership["view_observed"]).bool()
    proposal_count = int(membership["num_proposals"])
    left = torch.as_tensor(authority["edge_left"]).long()
    right = torch.as_tensor(authority["edge_right"]).long()
    relation = torch.as_tensor(authority["edge_relation"]).to(torch.int8)
    authority_views = torch.as_tensor(authority["proposal_view_indices"]).long()
    if not torch.equal(views, authority_views):
        raise ValueError("relation and membership proposal view axes differ")
    if view_observed.shape != (int(views.max()) + 1, field.num_gaussians):
        raise ValueError("source-view visibility and field row domains differ")

    with torch.inference_mode():
        latent = field.local_codes.detach().cpu().float()
        pooled, mass = sparse_proposal_pool(latent, rows, proposals, weights, proposal_count)
        centroids, _ = sparse_proposal_pool(xyz, rows, proposals, weights, proposal_count)
    valid_proposal = mass > 0
    if not bool(valid_proposal.any()):
        raise ValueError("no proposal has exact-MPR support")
    pooled = F.normalize(pooled, dim=-1, eps=1e-8)
    scene_extent = (xyz.amax(0) - xyz.amin(0)).clamp_min(1e-6)
    descriptors = F.normalize(torch.as_tensor(teacher["descriptors"]).float(), dim=-1)
    contexts = F.normalize(torch.as_tensor(teacher["context_descriptors"]).float(), dim=-1)
    language_teacher = F.normalize(0.75 * descriptors + 0.25 * contexts, dim=-1)
    if language_teacher.shape != (proposal_count, 1536):
        raise ValueError("mask-language teacher proposal domain differs")

    heldout_split = torch.remainder(views, int(args.holdout_stride)) == int(
        args.holdout_residue
    )
    heldout_proposal = heldout_split & valid_proposal
    train_proposal = (~heldout_split) & valid_proposal
    known = relation >= 0
    train_edge = train_proposal[left] & train_proposal[right]
    cross_edge = (heldout_proposal[left] ^ heldout_proposal[right]) & known
    if not bool(train_edge.any()) or not bool(cross_edge.any()):
        raise ValueError("fixed source split lacks train or cross-view known edges")

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    model = FrozenLatentRelationDecoder(
        latent_dim=pooled.shape[1], hidden_dim=int(args.hidden_dim), language_dim=1536
    ).to(device)
    pooled_d = pooled.to(device)
    centroids_d = centroids.to(device)
    areas_d = areas.to(device)
    extent_d = scene_extent.to(device)
    left_d, right_d, relation_d = left.to(device), right.to(device), relation.to(device)
    train_edge_d = train_edge.to(device)
    train_proposal_d = train_proposal.to(device)
    language_d = language_teacher.to(device)
    initial_record: dict[str, str] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    if str(args.initial_checkpoint).strip():
        initial, initial_sha, initial_path = load_sha_bound_project_checkpoint_mapping(
            args.initial_checkpoint,
            expected_sha256=args.expected_initial_checkpoint_sha256,
            map_location="cpu",
            label="frozen L512 global relation decoder initialization",
        )
        if initial.get("schema") != "radio_gs.frozen_l512_global_relation_decoder.v1":
            raise ValueError("initial relation decoder schema differs")
        if initial.get("scene") != str(args.scene):
            raise ValueError("initial relation decoder scene differs")
        if initial.get("metadata", {}).get("inputs") != loaded["records"]:
            raise ValueError("initial relation decoder input authorities differ")
        model.load_state_dict(initial["state_dict"], strict=True)
        initial_record = {"path": str(initial_path), "sha256": initial_sha}
        best_state = {
            key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()
        }
        with torch.inference_mode():
            initial_embedding = model.encode(pooled_d)
            initial_logits = model.relation_logits(
                initial_embedding,
                left_d[train_edge_d],
                right_d[train_edge_d],
                centroids_d,
                areas_d,
                extent_d,
            )
            initial_relation_loss = balanced_relation_loss(
                initial_logits, relation_d[train_edge_d]
            )
            initial_language = model.decode_language(initial_embedding[train_proposal_d])
            initial_language_loss = 1.0 - (
                initial_language * language_d[train_proposal_d]
            ).sum(-1).mean()
            best_loss = float(
                initial_relation_loss
                + float(args.language_weight) * initial_language_loss
            )
    elif int(args.steps) <= 0:
        raise ValueError("zero-step evaluation requires --initial-checkpoint")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    for _step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        embedding = model.encode(pooled_d)
        logits = model.relation_logits(
            embedding,
            left_d[train_edge_d],
            right_d[train_edge_d],
            centroids_d,
            areas_d,
            extent_d,
        )
        relation_loss = balanced_relation_loss(logits, relation_d[train_edge_d])
        decoded = model.decode_language(embedding[train_proposal_d])
        language_loss = 1.0 - (decoded * language_d[train_proposal_d]).sum(-1).mean()
        loss = relation_loss + float(args.language_weight) * language_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()
            }
    assert best_state is not None
    model.load_state_dict(best_state)

    with torch.inference_mode():
        embedding = model.encode(pooled_d)
        all_logits = model.relation_logits(
            embedding, left_d, right_d, centroids_d, areas_d, extent_d
        )
        heldout_language = model.decode_language(embedding[heldout_proposal.to(device)])
        heldout_language_cosine = float(
            (heldout_language * language_d[heldout_proposal.to(device)]).sum(-1).mean()
        )
    decoder_auc = relation_auc(
        conditional_same_score(all_logits[cross_edge.to(device)]), relation[cross_edge]
    )
    raw_similarity = (pooled[left] * pooled[right]).sum(-1)
    raw_auc = relation_auc(raw_similarity[cross_edge], relation[cross_edge])

    train_indices = torch.where(train_proposal)[0]
    heldout_indices = torch.where(heldout_proposal)[0]
    hh = heldout_indices[:, None].expand(-1, train_indices.numel()).reshape(-1).to(device)
    tt = train_indices[None, :].expand(heldout_indices.numel(), -1).reshape(-1).to(device)
    with torch.inference_mode():
        matrix_logits = model.relation_logits(
            embedding, hh, tt, centroids_d, areas_d, extent_d
        )
        # Retrieval must account for unknown edges, unlike the conditional
        # same-vs-different diagnostic AUC.
        matrix = F.softmax(matrix_logits, dim=-1)[:, 1].reshape(
            heldout_indices.numel(), train_indices.numel()
        ).cpu()
    raw_matrix = pooled[heldout_indices] @ pooled[train_indices].T
    supports = proposal_supports(rows, proposals, proposal_count)
    visible_supports = [
        set(torch.where(view_observed[view])[0].tolist())
        for view in range(view_observed.shape[0])
    ]
    known_same: dict[int, set[int]] = {int(item): set() for item in heldout_indices.tolist()}
    for edge_left, edge_right, edge_relation in zip(left.tolist(), right.tolist(), relation.tolist()):
        if edge_relation != 1:
            continue
        if heldout_proposal[edge_left] and train_proposal[edge_right]:
            known_same[int(edge_left)].add(int(edge_right))
        elif heldout_proposal[edge_right] and train_proposal[edge_left]:
            known_same[int(edge_right)].add(int(edge_left))

    # Dense only over the proposal sentinel's source-domain rows.  The
    # visibility-normalized posterior is the first-principles alternative to
    # a proposal union: disagreement from an occluded view is missing evidence,
    # while disagreement from an observable view is genuine negative evidence.
    train_lookup = torch.full((proposal_count,), -1, dtype=torch.long)
    train_lookup[train_indices] = torch.arange(train_indices.numel())
    train_support_matrix = torch.zeros(
        train_indices.numel(), field.num_gaussians, dtype=torch.bool
    )
    selected_train_entries = train_lookup[proposals] >= 0
    train_support_matrix[
        train_lookup[proposals[selected_train_entries]], rows[selected_train_entries]
    ] = True
    train_visibility_matrix = view_observed[views[train_indices]]
    target_support_matrix = torch.zeros(
        heldout_indices.numel(), field.num_gaussians, dtype=torch.bool
    )
    heldout_lookup = torch.full((proposal_count,), -1, dtype=torch.long)
    heldout_lookup[heldout_indices] = torch.arange(heldout_indices.numel())
    selected_heldout_entries = heldout_lookup[proposals] >= 0
    target_support_matrix[
        heldout_lookup[proposals[selected_heldout_entries]], rows[selected_heldout_entries]
    ] = True
    target_visibility_matrix = view_observed[views[heldout_indices]]
    known_same_weights = torch.zeros_like(matrix)
    recoverable_rows: list[int] = []
    train_position = {int(value): index for index, value in enumerate(train_indices.tolist())}
    for row_index, heldout in enumerate(heldout_indices.tolist()):
        candidates = known_same[int(heldout)]
        if candidates:
            recoverable_rows.append(row_index)
            for candidate in candidates:
                known_same_weights[row_index, train_position[int(candidate)]] = 1.0
    marginal = visibility_normalized_track_posterior(
        matrix, train_support_matrix, train_visibility_matrix
    )
    known_same_marginal = visibility_normalized_track_posterior(
        known_same_weights, train_support_matrix, train_visibility_matrix
    )
    recoverable_tensor = torch.as_tensor(recoverable_rows, dtype=torch.long)
    marginal_soft_iou = soft_support_iou(
        marginal[recoverable_tensor],
        target_support_matrix[recoverable_tensor],
        target_visibility_matrix[recoverable_tensor],
    )
    known_same_marginal_soft_iou = soft_support_iou(
        known_same_marginal[recoverable_tensor],
        target_support_matrix[recoverable_tensor],
        target_visibility_matrix[recoverable_tensor],
    )
    marginal_binary_iou = [
        support_iou(
            set(torch.where(target_support_matrix[item])[0].tolist()),
            set(
                torch.where(
                    (marginal[item] >= 0.5) & target_visibility_matrix[item]
                )[0].tolist()
            ),
        )
        for item in recoverable_rows
    ]
    known_same_marginal_binary_iou = [
        support_iou(
            set(torch.where(target_support_matrix[item])[0].tolist()),
            set(
                torch.where(
                    (known_same_marginal[item] >= 0.5)
                    & target_visibility_matrix[item]
                )[0].tolist()
            ),
        )
        for item in recoverable_rows
    ]
    reconstructed: list[float] = []
    reconstructed_union: list[float] = []
    baseline: list[float] = []
    relation_oracle: list[float] = []
    relation_union_oracle: list[float] = []
    geometry_oracle: list[float] = []
    for row_index, heldout in enumerate(heldout_indices.tolist()):
        same_candidates = known_same[int(heldout)]
        if not same_candidates:
            continue
        visible = visible_supports[int(views[heldout])]
        predicted = int(train_indices[int(matrix[row_index].argmax())])
        raw_predicted = int(train_indices[int(raw_matrix[row_index].argmax())])
        reconstructed.append(
            support_iou(supports[heldout], supports[predicted] & visible)
        )
        predicted_same = train_indices[matrix[row_index] >= 0.5].tolist()
        predicted_union: set[int] = set()
        for item in predicted_same:
            predicted_union.update(supports[int(item)])
        reconstructed_union.append(
            support_iou(supports[heldout], predicted_union & visible)
        )
        baseline.append(
            support_iou(supports[heldout], supports[raw_predicted] & visible)
        )
        relation_oracle.append(
            max(
                support_iou(supports[heldout], supports[item] & visible)
                for item in same_candidates
            )
        )
        same_union: set[int] = set()
        for item in same_candidates:
            same_union.update(supports[item])
        relation_union_oracle.append(
            support_iou(supports[heldout], same_union & visible)
        )
        geometry_oracle.append(
            max(
                support_iou(supports[heldout], supports[int(item)] & visible)
                for item in train_indices
            )
        )
    if not reconstructed:
        raise ValueError("no held-out proposal has a known same-object training proposal")
    mean = lambda values: float(torch.tensor(values).mean())
    best_decoder_reconstruction = max(
        mean(reconstructed),
        mean(reconstructed_union),
        mean(marginal_binary_iou),
    )
    passed = decoder_auc > raw_auc and best_decoder_reconstruction > mean(baseline)

    output = Path(args.output).expanduser().resolve()
    payload = {
        "schema": "radio_gs.frozen_l512_global_relation_decoder.v1",
        "schema_version": 1,
        "scene": str(args.scene),
        "state_dict": best_state,
        "metadata": {
            "query_independent": True,
            "field_frozen": True,
            "per_gaussian_parameters_added": False,
            "relation_target": "source_sam_same_different_unknown",
            "relation_proper_score": "class_balanced_categorical_log_score",
            "relation_classes": ["different", "same", "unknown"],
            "view_split": (
                f"source_view_index_mod_{int(args.holdout_stride)}_eq_"
                f"{int(args.holdout_residue)}"
            ),
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "text_queries_opened": False,
            "inputs": loaded["records"],
            "initial_checkpoint": initial_record,
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "status": "source_gate_pass" if passed else "source_gate_fail",
        "scene": str(args.scene),
        "best_train_loss": best_loss,
        "train_proposals": int(train_proposal.sum()),
        "heldout_proposals": int(heldout_proposal.sum()),
        "invalid_zero_support_proposals": int((~valid_proposal).sum()),
        "train_relation_edges": int(train_edge.sum()),
        "heldout_cross_known_edges": int(cross_edge.sum()),
        "heldout_cross_relation_auc": {"raw_l512_cosine": raw_auc, "decoder": decoder_auc},
        "heldout_mask_language_cosine": heldout_language_cosine,
        "heldout_recoverable_proposals": len(reconstructed),
        "heldout_visibility_clamped_support_iou": {
            "raw_l512_nearest": mean(baseline),
            "decoder_nearest": mean(reconstructed),
            "decoder_same_probability_union_at_0p5": mean(reconstructed_union),
            "known_same_oracle": mean(relation_oracle),
            "known_same_union_oracle": mean(relation_union_oracle),
            "all_training_geometry_oracle": mean(geometry_oracle),
            "decoder_visibility_normalized_marginal_soft": float(
                marginal_soft_iou.mean()
            ),
            "decoder_visibility_normalized_marginal_at_0p5": mean(
                marginal_binary_iou
            ),
            "known_same_visibility_normalized_marginal_soft": float(
                known_same_marginal_soft_iou.mean()
            ),
            "known_same_visibility_normalized_marginal_at_0p5": mean(
                known_same_marginal_binary_iou
            ),
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
    parser.add_argument("--initial-checkpoint", default="")
    parser.add_argument("--expected-initial-checkpoint-sha256", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--language-weight", type=float, default=0.20)
    parser.add_argument("--holdout-stride", type=int, default=4)
    parser.add_argument("--holdout-residue", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260823)
    print(json.dumps(train_and_evaluate(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
