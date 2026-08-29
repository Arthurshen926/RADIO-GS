"""Measure proposal codec quality before and after clean Gaussian writing."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.evaluation.semantic_mapping_error_ladder import _metrics
from radio_gs.v3.training.instance_upper_bound import proposal_supports, sha256_file


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-memory", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--siglip-teacher", required=True)
    parser.add_argument("--semantic-codec", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--residue", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.residue not in (0, 3):
        raise ValueError("semantic writer evaluation is held-out source only")
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("semantic_memory", args.semantic_memory), ("membership", args.membership),
            ("authority", args.authority), ("siglip_teacher", args.siglip_teacher),
            ("semantic_codec", args.semantic_codec), ("text_embeddings", args.text_embeddings),
        )
    }
    memory = torch.load(paths["semantic_memory"], map_location="cpu")
    membership = torch.load(paths["membership"], map_location="cpu")
    authority = torch.load(paths["authority"], map_location="cpu")
    teacher = torch.load(paths["siglip_teacher"], map_location="cpu")
    codec = torch.load(paths["semantic_codec"], map_location="cpu")
    text_payload = torch.load(paths["text_embeddings"], map_location="cpu")
    if memory.get("schema") not in (
        "radio_gs.sugm_v3.conflict_aware_semantic_memory.v1",
        "radio_gs.sugm_v3.conflict_aware_semantic_memory.v2",
    ):
        raise ValueError("semantic writer evaluation memory differs")
    lookup = {str(value).casefold(): index for index, value in enumerate(text_payload["queries"])}
    text = torch.as_tensor(text_payload["embeddings"])[
        [lookup[str(value).casefold()] for value in authority["query_names"]]
    ].float()
    state = codec["state_dict"]
    mean = torch.as_tensor(state["siglip_mean"]).float()
    basis = torch.as_tensor(state["siglip_basis"]).float()
    text = F.normalize((text - mean) @ basis, dim=-1, eps=1e-8)
    descriptor = F.normalize(
        (torch.as_tensor(teacher["descriptors"]).float() - mean) @ basis,
        dim=-1, eps=1e-8,
    )
    views = torch.as_tensor(authority["proposal_view_indices"]).long()
    selected = torch.where(views % 4 == args.residue)[0]
    states = torch.as_tensor(authority["query_state"])[selected].to(torch.int8)
    active = (states == 1).any(0) & (states == 0).any(0)
    if not bool(active.any()):
        raise ValueError("semantic writer split has no evaluable query")
    states = states[:, active]
    text = text[active]
    proposal_scores = descriptor[selected] @ text.T
    supports = proposal_supports(
        membership["row_indices"], membership["proposal_indices"],
        membership["weights"], int(membership["num_proposals"]),
    )
    semantic = torch.as_tensor(memory["semantic"]).float()
    prototypes = []
    for index in selected.tolist():
        rows, weights = supports[index]
        prototypes.append(F.normalize(
            (semantic[rows] * weights.float()[:, None]).sum(0), dim=0, eps=1e-8
        ))
    memory_scores = torch.stack(prototypes) @ text.T
    payload = {
        "schema": "radio_gs.sugm_v3.semantic_writer_evaluation.v1",
        "scene": membership["scene"], "residue": args.residue,
        "evaluable_queries": int(active.sum()),
        "codec_proposal_d128": _metrics(proposal_scores, states),
        "written_gaussian_d128": _metrics(memory_scores, states),
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
