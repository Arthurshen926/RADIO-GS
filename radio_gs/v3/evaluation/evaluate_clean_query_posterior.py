"""Evaluate the actual text-query Gaussian posterior on clean ternary authority."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.evaluation.semantic_mapping_error_ladder import _metrics
from radio_gs.v3.query.calibrated_posterior import (
    NullCalibratedPosterior,
    load_null_calibrated_posterior,
)
from radio_gs.v3.query.interface import load_query_interface
from radio_gs.v3.query.packet import QueryPacket
from radio_gs.v3.training.instance_upper_bound import proposal_supports, sha256_file


def _proposal_scores(
    row_score: torch.Tensor, supports: list[tuple[torch.Tensor, torch.Tensor]], indices: torch.Tensor
) -> torch.Tensor:
    values = []
    for index in indices.tolist():
        rows, weight = supports[index]
        mass = weight.float()
        values.append((row_score[rows] * mass).sum() / mass.sum().clamp_min(1e-8))
    return torch.stack(values)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--text-negatives", required=True)
    parser.add_argument("--residue", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--calibrator")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.residue not in (0, 3):
        raise ValueError("clean posterior evaluation is held-out source only")
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("scene_state", args.scene_state), ("membership", args.membership),
            ("authority", args.authority), ("text_embeddings", args.text_embeddings),
            ("text_negatives", args.text_negatives),
        )
    }
    membership = torch.load(paths["membership"], map_location="cpu")
    authority = torch.load(paths["authority"], map_location="cpu")
    text_payload = torch.load(paths["text_embeddings"], map_location="cpu")
    interface = load_query_interface(
        paths["scene_state"], device=args.device,
        text_negative_path=paths["text_negatives"], text_logit_scale=10.0,
    )
    supports = proposal_supports(
        membership["row_indices"], membership["proposal_indices"], membership["weights"],
        int(membership["num_proposals"]),
    )
    views = torch.as_tensor(authority["proposal_view_indices"]).long()
    indices = torch.where(views % 4 == args.residue)[0]
    states = torch.as_tensor(authority["query_state"])[indices].to(torch.int8)
    active = (states == 1).any(0) & (states == 0).any(0)
    if not bool(active.any()):
        raise ValueError("clean posterior split has no evaluable query")
    states = states[:, active]
    names = [authority["query_names"][index] for index in torch.where(active)[0].tolist()]
    lookup = {str(value).casefold(): index for index, value in enumerate(text_payload["queries"])}
    raw_identity_columns, null_columns, null_margin_columns = [], [], []
    posterior_columns = []
    calibrated_columns = []
    calibrator = (
        load_null_calibrated_posterior(args.calibrator, device=args.device)
        if args.calibrator
        else NullCalibratedPosterior().to(args.device).eval()
    )
    for name in names:
        token = torch.as_tensor(text_payload["embeddings"])[lookup[str(name).casefold()]].float()
        packet = QueryPacket("text", token)
        raw_identity, canonical_null, _unknown = interface.semantic_text_evidence(packet)
        calibrated, returned_identity, posterior = interface.calibrated_posterior_from_packet(
            packet, calibrator, scale=args.scale, topk=args.topk,
            temperature=args.temperature, posterior_chunk_size=65536,
        )
        if not torch.equal(raw_identity, returned_identity):
            raise RuntimeError("positive-anchor interface changed the identity evidence")
        null_margin = torch.sigmoid((raw_identity - canonical_null) * 10.0)
        raw_identity_columns.append(_proposal_scores(raw_identity.cpu(), supports, indices))
        null_columns.append(_proposal_scores(canonical_null.cpu(), supports, indices))
        null_margin_columns.append(_proposal_scores(null_margin.cpu(), supports, indices))
        posterior_columns.append(_proposal_scores(posterior.cpu(), supports, indices))
        calibrated_columns.append(_proposal_scores(calibrated.cpu(), supports, indices))
    raw_identity_scores = torch.stack(raw_identity_columns, dim=1)
    null_scores = torch.stack(null_columns, dim=1)
    null_margin_scores = torch.stack(null_margin_columns, dim=1)
    posterior_scores = torch.stack(posterior_columns, dim=1)
    calibrated_scores = torch.stack(calibrated_columns, dim=1)
    payload = {
        "schema": "radio_gs.sugm_v3.clean_query_posterior_evaluation.v1",
        "scene": membership["scene"], "residue": args.residue,
        "evaluable_queries": len(names),
        "raw_positive_identity": _metrics(raw_identity_scores, states),
        "canonical_null": _metrics(null_scores, states),
        "positive_vs_null_probability": _metrics(null_margin_scores, states),
        "raw_positive_anchor_instance_posterior": _metrics(posterior_scores, states),
        (
            "trained_calibrated_posterior"
            if args.calibrator else "initial_calibrated_posterior"
        ): _metrics(calibrated_scores, states),
        "source_only": True, "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "method": {"topk": args.topk, "scale": args.scale, "temperature": args.temperature},
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    if args.calibrator:
        calibrator_path = Path(args.calibrator).resolve(strict=True)
        payload["inputs"]["calibrator"] = {
            "path": str(calibrator_path),
            "sha256": sha256_file(calibrator_path),
        }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)


if __name__ == "__main__":
    main()
