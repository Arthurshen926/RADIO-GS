"""Gate-0 verification that the disabled-child pipeline is exact D128 identity."""

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


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--residue", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.residue not in (0, 3):
        raise ValueError("identity replay evaluation is held-out source only")
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("scene_state", args.scene_state),
            ("membership", args.membership),
            ("authority", args.authority),
            ("text_embeddings", args.text_embeddings),
        )
    }
    membership = torch.load(paths["membership"], map_location="cpu")
    authority = torch.load(paths["authority"], map_location="cpu")
    text_payload = torch.load(paths["text_embeddings"], map_location="cpu")
    interface = load_query_interface(paths["scene_state"], device=args.device)
    supports = proposal_supports(
        membership["row_indices"],
        membership["proposal_indices"],
        membership["weights"],
        int(membership["num_proposals"]),
    )
    views = torch.as_tensor(authority["proposal_view_indices"]).long()
    selected = torch.where(views % 4 == args.residue)[0]
    states = torch.as_tensor(authority["query_state"])[selected].to(torch.int8)
    active = (states == 1).any(0) & (states == 0).any(0)
    if not bool(active.any()):
        raise ValueError("identity replay split has no evaluable query")
    states = states[:, active]
    names = [
        authority["query_names"][index]
        for index in torch.where(active)[0].tolist()
    ]
    lookup = {
        str(value).casefold(): index
        for index, value in enumerate(text_payload["queries"])
    }
    score_columns = []
    bitwise_scores = True
    bitwise_rows = True
    bitwise_weights = True
    for name in names:
        token = torch.as_tensor(text_payload["embeddings"])[
            lookup[str(name).casefold()]
        ].float()
        packet = QueryPacket("text", token)
        clean_score, _null, _unknown = interface.semantic_text_evidence(packet)
        clean_rows, clean_weights, compiled_score = interface.compile_identity_anchors(
            packet, topk=args.topk, text_anchor_policy="positive"
        )
        replay_score, replay_rows, replay_weights = interface.replay_identity_from_packet(
            packet, topk=args.topk
        )
        bitwise_scores &= torch.equal(clean_score, compiled_score) and torch.equal(
            clean_score, replay_score
        )
        bitwise_rows &= torch.equal(clean_rows, replay_rows)
        bitwise_weights &= torch.equal(clean_weights, replay_weights)
        score_columns.append(_proposal_scores(replay_score.cpu(), supports, selected))
    replay_scores = torch.stack(score_columns, dim=1)
    if not (bitwise_scores and bitwise_rows and bitwise_weights):
        raise RuntimeError("Gate-0 identity replay is not bitwise exact")
    payload = {
        "schema": "radio_gs.sugm_v3.identity_replay_gate.v1",
        "scene": membership["scene"],
        "residue": args.residue,
        "evaluable_queries": len(names),
        "metrics": _metrics(replay_scores, states),
        "invariants": {
            "identity_scores_bitwise_equal": bitwise_scores,
            "anchor_rows_bitwise_equal": bitwise_rows,
            "anchor_weights_bitwise_equal": bitwise_weights,
            "instance_residual": 0.0,
            "boundary_residual": 0.0,
            "calibrator": "disabled",
            "null_conversion": "disabled",
            "reliability_tempering": "disabled",
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
