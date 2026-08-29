"""Evaluate a shared semantic codec on held-out fresh ternary authority."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.evaluation.semantic_mapping_error_ladder import _metrics
from radio_gs.v3.training.instance_upper_bound import sha256_file


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--siglip-teacher", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--residue", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.residue not in (0, 3):
        raise ValueError("semantic codec evaluation is held-out source only")
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("codec", args.codec), ("authority", args.authority),
            ("siglip_teacher", args.siglip_teacher),
            ("text_embeddings", args.text_embeddings),
        )
    }
    codec = torch.load(paths["codec"], map_location="cpu")
    authority = torch.load(paths["authority"], map_location="cpu")
    teacher = torch.load(paths["siglip_teacher"], map_location="cpu")
    text_payload = torch.load(paths["text_embeddings"], map_location="cpu")
    if (
        codec.get("schema") != "radio_gs.sugm_v3.query_discriminative_semantic_codec.v1"
        or authority.get("schema") != "radio_gs.sugm_v3.native_language_authority.v3"
    ):
        raise ValueError("semantic codec evaluation lineage differs")
    lookup = {str(value).casefold(): index for index, value in enumerate(text_payload["queries"])}
    text = torch.as_tensor(text_payload["embeddings"])[
        [lookup[str(value).casefold()] for value in authority["query_names"]]
    ].float()
    views = torch.as_tensor(authority["proposal_view_indices"]).long()
    selected = views % 4 == args.residue
    states = torch.as_tensor(authority["query_state"])[selected].to(torch.int8)
    active = (states == 1).any(0) & (states == 0).any(0)
    if not bool(active.any()):
        raise ValueError("semantic codec split has no evaluable query")
    states = states[:, active]
    text = text[active]
    descriptor = torch.as_tensor(teacher["descriptors"])[selected].float()
    native_scores = F.normalize(descriptor, dim=-1) @ F.normalize(text, dim=-1).T
    state = codec["state_dict"]
    mean = torch.as_tensor(state["siglip_mean"]).float()
    basis = torch.as_tensor(state["siglip_basis"]).float()
    projected_scores = F.normalize((descriptor - mean) @ basis, dim=-1) @ F.normalize(
        (text - mean) @ basis, dim=-1
    ).T
    payload = {
        "schema": "radio_gs.sugm_v3.semantic_codec_evaluation.v1",
        "scene": authority["scene"], "residue": args.residue,
        "evaluable_queries": int(active.sum()),
        "positive_pairs": int((states == 1).sum()),
        "negative_pairs": int((states == 0).sum()),
        "unknown_pairs": int((states == -1).sum()),
        "native_d1536": _metrics(native_scores, states),
        "candidate_d128": _metrics(projected_scores, states),
        "source_only": True, "target_rgb_opened": False,
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
