#!/usr/bin/env python3
"""Train a field-frozen source-SAM Object-Aware Universal Field v2 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.models.object_aware_universal_field_v2 import (
    ObjectAwareFieldHead,
    object_aware_proper_loss,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_object_aware_universal_field_v2_pilot.v1"


def fixed_view_split(proposal_views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reserve every fourth source view before training or metric access."""

    views = torch.as_tensor(proposal_views).long()
    heldout = torch.remainder(views, 4) == 3
    return ~heldout, heldout


def relation_metrics(similarity: torch.Tensor, relation: torch.Tensor) -> dict[str, float]:
    """Threshold-free pair AUC and mean same/different similarities."""

    score = torch.as_tensor(similarity).detach().cpu().float()
    label = torch.as_tensor(relation).detach().cpu().to(torch.int8)
    same = score[label == 1]
    different = score[label == 0]
    if not same.numel() or not different.numel():
        return {"auc": float("nan"), "same_mean": float("nan"), "different_mean": float("nan")}
    # Mann--Whitney AUC in bounded chunks; ties receive half credit.
    wins = score.new_zeros(())
    for chunk in same.split(4096):
        delta = chunk[:, None] - different[None, :]
        wins += (delta > 0).sum() + 0.5 * (delta == 0).sum()
    return {
        "auc": float(wins / (same.numel() * different.numel())),
        "same_mean": float(same.mean()),
        "different_mean": float(different.mean()),
    }


def source_threshold(similarity: torch.Tensor, relation: torch.Tensor) -> float:
    """Choose one train-source threshold maximizing balanced accuracy."""

    score = torch.as_tensor(similarity).detach().cpu().float()
    label = torch.as_tensor(relation).detach().cpu().to(torch.int8)
    same, different = score[label == 1], score[label == 0]
    if not same.numel() or not different.numel():
        raise ValueError("threshold source split lacks both known relations")
    candidates = torch.linspace(-1.0, 1.0, 401)
    balanced = torch.stack(
        [0.5 * ((same >= value).float().mean() + (different < value).float().mean()) for value in candidates]
    )
    return float(candidates[int(balanced.argmax())])


def build(args: argparse.Namespace) -> dict[str, Any]:
    membership_path = Path(args.membership).expanduser().resolve()
    teacher_path = Path(args.teacher).expanduser().resolve()
    relation_path = Path(args.relation_authority).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"pilot output exists: {output}")
    membership = torch.load(membership_path, map_location="cpu", weights_only=False)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    authority = torch.load(relation_path, map_location="cpu", weights_only=False)
    if membership.get("metadata", {}).get("query_independent_proposal_set") is not True:
        raise ValueError("membership is not query-independent")
    if teacher.get("metadata", {}).get("source_only") is not True or teacher.get("metadata", {}).get("benchmark_masks_opened") is not False:
        raise ValueError("language teacher information contract differs")
    if authority.get("metadata", {}).get("source_only") is not True or authority.get("metadata", {}).get("benchmark_masks_opened") is not False:
        raise ValueError("relation authority information contract differs")
    rows = torch.as_tensor(membership["row_indices"]).long()
    proposals = torch.as_tensor(membership["proposal_indices"]).long()
    weights = torch.as_tensor(membership["weights"]).float()
    proposal_views = torch.as_tensor(membership["proposal_view_indices"]).long()
    areas = torch.as_tensor(membership["proposal_area_fraction"]).float()
    proposal_count = int(membership["num_proposals"])
    num_rows = int(membership["num_rows"])
    descriptors = F.normalize(torch.as_tensor(teacher["descriptors"]).float(), dim=-1)
    contexts = F.normalize(torch.as_tensor(teacher["context_descriptors"]).float(), dim=-1)
    language_teacher = F.normalize(0.75 * descriptors + 0.25 * contexts, dim=-1)
    left = torch.as_tensor(authority["edge_left"]).long()
    right = torch.as_tensor(authority["edge_right"]).long()
    relation = torch.as_tensor(authority["edge_relation"]).to(torch.int8)
    if proposal_views.shape != (proposal_count,) or areas.shape != (proposal_count,) or descriptors.shape != (proposal_count, 1536):
        raise ValueError("proposal authority axes differ")
    if not torch.equal(proposal_views, torch.as_tensor(authority["proposal_view_indices"]).long()):
        raise ValueError("relation/membership proposal views differ")

    train_proposal, heldout_proposal = fixed_view_split(proposal_views)
    train_edge = train_proposal[left] & train_proposal[right] & (relation >= 0)
    heldout_edge = heldout_proposal[left] & heldout_proposal[right] & (relation >= 0)
    if not bool(train_edge.any()) or not bool(heldout_edge.any()):
        raise ValueError("fixed source split lacks known relation edges")

    device = torch.device(args.device)
    model = ObjectAwareFieldHead(
        num_rows,
        object_dim=int(args.object_dim),
        language_dim=1536,
        seed=int(args.seed),
    ).to(device)
    rows_d, proposals_d, weights_d, areas_d = (
        value.to(device) for value in (rows, proposals, weights, areas)
    )
    left_d, right_d, relation_d = left.to(device), right.to(device), relation.to(device)
    teacher_d = language_teacher.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)

    def forward_metrics() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding, _ = model.proposal_embeddings(rows_d, proposals_d, weights_d, areas_d, proposal_count)
        decoded = model.decode_language(embedding)
        similarity = (embedding[left_d] * embedding[right_d]).sum(-1)
        return embedding, decoded, similarity

    with torch.no_grad():
        _, initial_decoded, initial_similarity = forward_metrics()
        initial_heldout = relation_metrics(initial_similarity[heldout_edge.to(device)], relation_d[heldout_edge.to(device)])
        initial_language = float((initial_decoded[heldout_proposal.to(device)] * teacher_d[heldout_proposal.to(device)]).sum(-1).mean())

    best_state: dict[str, torch.Tensor] | None = None
    best_source_loss = float("inf")
    train_indices = torch.where(train_edge)[0].to(device)
    train_proposal_d = train_proposal.to(device)
    for step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        embedding, decoded, _ = forward_metrics()
        loss = object_aware_proper_loss(
            embedding,
            decoded[train_proposal_d],
            teacher_d[train_proposal_d],
            left_d[train_indices],
            right_d[train_indices],
            relation_d[train_indices],
            language_weight=float(args.language_weight),
            relation_logit_scale=float(args.relation_logit_scale),
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        value = float(loss.total.detach())
        if value < best_source_loss:
            best_source_loss = value
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    with torch.no_grad():
        embedding, decoded, similarity = forward_metrics()
    similarity_cpu = similarity.cpu()
    final_train = relation_metrics(similarity_cpu[train_edge], relation[train_edge])
    final_heldout = relation_metrics(similarity_cpu[heldout_edge], relation[heldout_edge])
    heldout_language = float((decoded[heldout_proposal.to(device)] * teacher_d[heldout_proposal.to(device)]).sum(-1).mean())
    threshold = source_threshold(similarity_cpu[train_edge], relation[train_edge])
    passed = final_heldout["auc"] >= initial_heldout["auc"] and heldout_language >= initial_language

    checkpoint = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": str(args.scene),
        "object_codes": model.object_codes.detach().cpu().half(),
        "scale_log_gates": model.scale_log_gates.detach().cpu(),
        "language_decoder_weight": model.language_decoder.weight.detach().cpu().half(),
        "proposal_embeddings": embedding.detach().cpu().half(),
        "decoded_object_language": decoded.detach().cpu().half(),
        "edge_affinity": similarity_cpu.half(),
        "source_same_threshold": threshold,
        "metadata": {
            "query_independent": True,
            "field_frozen": True,
            "object_dim": int(args.object_dim),
            "scale_bins": "fixed_area_[1/256,1/64,1/16]",
            "view_split": "heldout_source_view_index_mod_4_eq_3",
            "unknown_relation_loss": "excluded",
            "relation_proper_score": "class_balanced_Bernoulli_log_score",
            "relation_logit": f"{float(args.relation_logit_scale):g}_times_cosine",
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "target_metrics_opened_before_training": False,
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "teacher": {"path": str(teacher_path), "sha256": sha256_file(teacher_path)},
            "relation_authority": {"path": str(relation_path), "sha256": sha256_file(relation_path)},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)
    report = {
        "schema": SCHEMA,
        "status": "source_gate_pass" if passed else "source_gate_fail",
        "scene": str(args.scene),
        "object_dim": int(args.object_dim),
        "train_proposals": int(train_proposal.sum()),
        "heldout_proposals": int(heldout_proposal.sum()),
        "train_known_edges": int(train_edge.sum()),
        "heldout_known_edges": int(heldout_edge.sum()),
        "best_source_loss": best_source_loss,
        "relation_proper_score": "class_balanced_Bernoulli_log_score",
        "initial_heldout": {**initial_heldout, "language_cosine": initial_language},
        "final_train": final_train,
        "final_heldout": {**final_heldout, "language_cosine": heldout_language},
        "source_same_threshold": threshold,
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--relation-authority", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--object-dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--language-weight", type=float, default=0.20)
    parser.add_argument("--relation-logit-scale", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260821)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
