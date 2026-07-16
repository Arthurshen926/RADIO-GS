#!/usr/bin/env python3
"""Audit matched-MPR capability fidelity before any 2-D rendering.

The screen-space capability audit conflates primitive reconstruction with
alpha compositing.  This audit uses the same frozen support-graph edges to
compare canonical primitive capability rows against the official-adaptor-
before-MPR teacher rows, cleanly locating which side of the renderer owns a
local-relation failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.evaluation.capability_fidelity import relation_fidelity_summary
from radio_gs.utils.checkpoint_io import load_trusted_checkpoint


def _edge_affinity(
    rows: torch.Tensor, edge_index: torch.Tensor, *, chunk_size: int
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for start in range(0, edge_index.shape[1], chunk_size):
        edge = edge_index[:, start : start + chunk_size]
        values.append((rows[edge[0]] * rows[edge[1]]).sum(dim=-1).cpu())
    return torch.cat(values, dim=0)


def audit(args: argparse.Namespace) -> dict:
    capability = load_trusted_checkpoint(Path(args.capability_cache), map_location="cpu")
    target = load_trusted_checkpoint(Path(args.target_mpr_cache), map_location="cpu")
    graph = load_trusted_checkpoint(Path(args.support_graph), map_location="cpu")
    global_rows = torch.as_tensor(graph["global_rows"]).long()
    edge_index = torch.as_tensor(graph["edge_index"]).long()
    predicted_all = torch.as_tensor(capability[args.capability_key]).float()
    target_all = torch.as_tensor(target["features"]).float()
    if predicted_all.shape != target_all.shape:
        raise ValueError(
            f"capability/target shape mismatch: {predicted_all.shape} vs {target_all.shape}"
        )
    if not torch.equal(
        torch.as_tensor(capability["xyz"]).float(), torch.as_tensor(target["xyz"]).float()
    ):
        raise ValueError("capability and matched-MPR geometry rows do not align")
    valid = (
        torch.as_tensor(capability["valid"]).bool()
        & torch.as_tensor(target["valid"]).bool()
    )
    selected_valid = valid[global_rows]
    keep_edge = selected_valid[edge_index[0]] & selected_valid[edge_index[1]]
    edge_index = edge_index[:, keep_edge]
    predicted = F.normalize(predicted_all[global_rows], dim=-1, eps=1e-8)
    teacher = F.normalize(target_all[global_rows], dim=-1, eps=1e-8)
    row_cosine = (predicted[selected_valid] * teacher[selected_valid]).sum(dim=-1)
    predicted_affinity = _edge_affinity(
        predicted, edge_index, chunk_size=args.chunk_size
    )
    teacher_affinity = _edge_affinity(teacher, edge_index, chunk_size=args.chunk_size)
    report = {
        "schema_version": 1,
        "audit": "primitive_capability_relation_v1",
        "capability_cache": str(Path(args.capability_cache).resolve()),
        "target_mpr_cache": str(Path(args.target_mpr_cache).resolve()),
        "support_graph": str(Path(args.support_graph).resolve()),
        "capability_key": args.capability_key,
        "official_adaptor_before_mpr_target": True,
        "rendering_used": False,
        "valid_rows": int(selected_valid.sum()),
        "row_cosine": {
            "mean": float(row_cosine.mean()),
            "p05": float(torch.quantile(row_cosine, 0.05)),
        },
        "local_relation": relation_fidelity_summary(
            predicted_affinity, teacher_affinity
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument("--target-mpr-cache", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--capability-key", default="boundary_sam3")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
