"""Gate-2 source-dev test of D128 text anchors expanded through fresh D48."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.evaluation.evaluate_clean_query_posterior import _proposal_scores
from radio_gs.v3.evaluation.semantic_mapping_error_ladder import _metrics
from radio_gs.v3.query.interface import load_query_interface
from radio_gs.v3.query.packet import QueryPacket
from radio_gs.v3.training.instance_upper_bound import proposal_supports, sha256_file


def _gate_decision(
    identity: dict[str, float], instance: dict[str, float], *, identity_exact: bool
) -> tuple[bool, list[str]]:
    failures = []
    if not identity_exact:
        failures.append("candidate changed protected D128 identity scores or anchors")
    if instance["recall_at_1"] < identity["recall_at_1"]:
        failures.append("text-anchored D48 recall@1 regressed")
    if instance["mrr"] < identity["mrr"]:
        failures.append("text-anchored D48 MRR regressed")
    if instance["margin"] <= 0:
        failures.append("text-anchored D48 has no positive proposal margin")
    return not failures, failures


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
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.residue not in (0, 3):
        raise ValueError("text-anchor instance gate is source-heldout only")
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
        raise ValueError("Gate-2 requires a passed Gate-1 child of this clean parent")
    membership = torch.load(paths["membership"], map_location="cpu")
    authority = torch.load(paths["authority"], map_location="cpu")
    text = torch.load(paths["text_embeddings"], map_location="cpu")
    clean = load_query_interface(paths["clean_parent"], device=args.device)
    candidate = load_query_interface(paths["candidate"], device=args.device)
    supports = proposal_supports(
        membership["row_indices"],
        membership["proposal_indices"],
        membership["weights"],
        int(membership["num_proposals"]),
    )
    views = torch.as_tensor(authority["proposal_view_indices"]).long()
    selected = torch.tensor(
        [
            index
            for index, (rows, _weights) in enumerate(supports)
            if rows.numel() and int(views[index] % 4) == args.residue
        ],
        dtype=torch.long,
    )
    states = torch.as_tensor(authority["query_state"])[selected].to(torch.int8)
    active = (states == 1).any(0) & (states == 0).any(0)
    if not bool(active.any()):
        raise ValueError("text-anchor instance split has no evaluable query")
    states = states[:, active]
    names = [
        authority["query_names"][index]
        for index in torch.where(active)[0].tolist()
    ]
    lookup = {
        str(name).casefold(): index for index, name in enumerate(text["queries"])
    }
    identity_columns = []
    instance_columns = []
    identity_exact = True
    for name in names:
        token = torch.as_tensor(text["embeddings"])[lookup[str(name).casefold()]].float()
        packet = QueryPacket("text", token)
        clean_rows, clean_weights, clean_score = clean.compile_identity_anchors(
            packet, topk=args.topk, text_anchor_policy="positive"
        )
        rows, weights, identity = candidate.compile_identity_anchors(
            packet, topk=args.topk, text_anchor_policy="positive"
        )
        identity_exact &= (
            torch.equal(clean_score, identity)
            and torch.equal(clean_rows, rows)
            and torch.equal(clean_weights, weights)
        )
        instance = candidate.gaussian_posterior(
            rows,
            weights,
            scale=args.scale,
            temperature=args.temperature,
            posterior_chunk_size=65536,
        )
        identity_columns.append(_proposal_scores(identity.cpu(), supports, selected))
        instance_columns.append(_proposal_scores(instance.cpu(), supports, selected))
    identity_metrics = _metrics(torch.stack(identity_columns, dim=1), states)
    instance_metrics = _metrics(torch.stack(instance_columns, dim=1), states)
    passed, failures = _gate_decision(
        identity_metrics, instance_metrics, identity_exact=identity_exact
    )
    payload = {
        "schema": "radio_gs.sugm_v3.text_anchor_instance_gate.v1",
        "scene": membership["scene"],
        "residue": args.residue,
        "evaluable_queries": len(names),
        "identity_bitwise_preserved": identity_exact,
        "raw_D128_identity": identity_metrics,
        "text_anchor_D48_instance": instance_metrics,
        "delta": {
            name: instance_metrics[name] - identity_metrics[name]
            for name in instance_metrics
            if name != "queries"
        },
        "gate": {
            "passed": passed,
            "failures": failures,
            "rule": "bitwise_identity_preservation_and_no_R1_or_MRR_regression_and_positive_instance_margin",
        },
        "method": {
            "topk": args.topk,
            "scale": args.scale,
            "temperature": args.temperature,
            "text_anchor_policy": "raw_positive_D128_before_null",
            "calibrator": "disabled",
            "boundary": "disabled",
        },
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)


if __name__ == "__main__":
    main()


__all__ = ["_gate_decision"]
