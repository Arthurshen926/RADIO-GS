"""Source-dev search for a D128-to-D48 anchor coupling after Gate-2 failure."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.evaluation.evaluate_clean_query_posterior import _proposal_scores
from radio_gs.v3.evaluation.semantic_mapping_error_ladder import _metrics
from radio_gs.v3.query.interface import load_query_interface
from radio_gs.v3.query.membership import membership_from_prototype, pool_prototype
from radio_gs.v3.query.packet import QueryPacket
from radio_gs.v3.training.instance_upper_bound import proposal_supports, sha256_file


def _coherent_anchors(
    identity: torch.Tensor,
    instance: torch.Tensor,
    *,
    candidate_k: int,
    anchor_k: int,
    affinity_weight: float,
    seed_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep strong semantic candidates coherent with the best valid identity seed."""

    if (
        candidate_k < anchor_k
        or anchor_k <= 0
        or affinity_weight < 0
        or seed_k <= 0
    ):
        raise ValueError("coherent anchor budget differs")
    valid = instance.norm(dim=1) > 1e-6
    available = int(valid.sum())
    if available < anchor_k:
        raise ValueError("coherent anchor compiler lacks written D48 rows")
    budget = min(int(candidate_k), available)
    masked = identity.masked_fill(~valid, -torch.inf)
    _values, candidates = masked.topk(budget)
    seed_budget = min(int(seed_k), budget)
    candidate_instance = instance[candidates]
    seed_similarity = candidate_instance @ candidate_instance[:seed_budget].T
    semantic_mass = torch.softmax(
        identity[candidates] - identity[candidates].max(), dim=0
    )
    cluster_mass = (
        seed_similarity.clamp_min(0).square() * semantic_mass[:, None]
    ).sum(0)
    seed = candidate_instance[cluster_mass.argmax()]
    affinity = candidate_instance @ seed
    combined = identity[candidates] + float(affinity_weight) * affinity
    selected = combined.topk(anchor_k).indices
    rows = candidates[selected]
    weights = torch.softmax(combined[selected] - combined[selected].max(), dim=0)
    return rows, weights


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-parent", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--residue", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--candidate-k", type=int, nargs="+", default=(32, 128, 512))
    parser.add_argument("--anchor-k", type=int, nargs="+", default=(8, 32))
    parser.add_argument("--affinity-weight", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--seed-k", type=int, nargs="+", default=(1,))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.residue != 3:
        raise ValueError("coupling exploration is restricted to source dev")
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("clean_parent", args.clean_parent),
            ("candidate", args.candidate),
            ("membership", args.membership),
            ("authority", args.authority),
            ("text_embeddings", args.text_embeddings),
        )
    }
    candidate_payload = torch.load(paths["candidate"], map_location="cpu")
    gate1 = candidate_payload.get("metadata", {}).get("instance_gate1", {})
    if (
        gate1.get("status") != "passed"
        or gate1.get("parent", {}).get("sha256") != sha256_file(paths["clean_parent"])
    ):
        raise ValueError("coupling exploration requires passed Gate 1")
    membership = torch.load(paths["membership"], map_location="cpu")
    authority = torch.load(paths["authority"], map_location="cpu")
    text = torch.load(paths["text_embeddings"], map_location="cpu")
    clean = load_query_interface(paths["clean_parent"], device=args.device)
    interface = load_query_interface(paths["candidate"], device=args.device)
    supports = proposal_supports(
        membership["row_indices"], membership["proposal_indices"],
        membership["weights"], int(membership["num_proposals"]),
    )
    views = torch.as_tensor(authority["proposal_view_indices"]).long()
    selected = torch.tensor([
        index for index, (rows, _weights) in enumerate(supports)
        if rows.numel() and int(views[index] % 4) == args.residue
    ])
    states = torch.as_tensor(authority["query_state"])[selected].to(torch.int8)
    active = (states == 1).any(0) & (states == 0).any(0)
    states = states[:, active]
    names = [authority["query_names"][index] for index in torch.where(active)[0].tolist()]
    lookup = {str(name).casefold(): index for index, name in enumerate(text["queries"])}
    policies = [
        (candidate_k, anchor_k, affinity, seed_k)
        for candidate_k, anchor_k, affinity, seed_k in itertools.product(
            args.candidate_k, args.anchor_k, args.affinity_weight, args.seed_k
        )
        if candidate_k >= anchor_k
    ]
    columns: dict[tuple[int, int, float], list[torch.Tensor]] = {
        policy: [] for policy in policies
    }
    raw_columns = []
    direct_columns = []
    identity_exact = True
    instance = interface.model.instance_view(0.5)
    for name in names:
        token = torch.as_tensor(text["embeddings"])[lookup[str(name).casefold()]].float()
        packet = QueryPacket("text", token)
        clean_rows, clean_weights, clean_identity = clean.compile_identity_anchors(
            packet, topk=8, text_anchor_policy="positive"
        )
        direct_rows, direct_weights, identity = interface.compile_identity_anchors(
            packet, topk=8, text_anchor_policy="positive"
        )
        identity_exact &= (
            torch.equal(clean_identity, identity)
            and torch.equal(clean_rows, direct_rows)
            and torch.equal(clean_weights, direct_weights)
        )
        raw_columns.append(_proposal_scores(identity.cpu(), supports, selected))
        direct_prototype = pool_prototype(instance[direct_rows], direct_weights)
        direct = membership_from_prototype(
            instance, direct_prototype, temperature=args.temperature
        )
        direct_columns.append(_proposal_scores(direct.cpu(), supports, selected))
        for policy in policies:
            candidate_k, anchor_k, affinity, seed_k = policy
            rows, weights = _coherent_anchors(
                identity,
                instance,
                candidate_k=candidate_k,
                anchor_k=anchor_k,
                affinity_weight=affinity,
                seed_k=seed_k,
            )
            prototype = pool_prototype(instance[rows], weights)
            posterior = membership_from_prototype(
                instance, prototype, temperature=args.temperature
            )
            columns[policy].append(
                _proposal_scores(posterior.cpu(), supports, selected)
            )
    raw = _metrics(torch.stack(raw_columns, dim=1), states)
    direct = _metrics(torch.stack(direct_columns, dim=1), states)
    results = []
    for policy in policies:
        metric = _metrics(torch.stack(columns[policy], dim=1), states)
        results.append({
            "candidate_k": policy[0],
            "anchor_k": policy[1],
            "affinity_weight": policy[2],
            "seed_k": policy[3],
            "metrics": metric,
            "delta_vs_raw": {
                name: metric[name] - raw[name] for name in metric if name != "queries"
            },
            "delta_vs_direct_top8": {
                name: metric[name] - direct[name] for name in metric if name != "queries"
            },
        })
    payload = {
        "schema": "radio_gs.sugm_v3.text_instance_coupling_exploration.v1",
        "scene": membership["scene"],
        "residue": args.residue,
        "identity_bitwise_preserved": identity_exact,
        "raw_D128_identity": raw,
        "direct_top8_D48": direct,
        "coherent_anchor_grid": results,
        "selection_authority": "source_dev_only_global_policy_must_be_frozen_before_audit",
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print({
        "scene": membership["scene"],
        "raw": raw,
        "direct": direct,
        "best": max(results, key=lambda item: (item["metrics"]["mrr"], item["metrics"]["recall_at_1"])),
    })


if __name__ == "__main__":
    main()


__all__ = ["_coherent_anchors"]
