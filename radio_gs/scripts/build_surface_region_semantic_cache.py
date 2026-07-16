#!/usr/bin/env python3
"""Derive a disposable text-space cache from the canonical field and v3 readout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.interfaces.surface_region_summary import (
    SurfaceRegionSummaryReadout, surface_region_geometry,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _adjacency(graph: dict, neighbors: int) -> torch.Tensor:
    """Keep strongest surface-conditioned outgoing edges plus a self slot."""
    count = int(graph["xyz"].shape[0]); k = int(neighbors)
    edge = torch.as_tensor(graph["edge_index"]).long()
    affinity = torch.as_tensor(graph["raw_affinity"]).float()
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(count)]
    for src, dst, weight in zip(edge[0].tolist(), edge[1].tolist(), affinity.tolist()):
        buckets[src].append((float(weight), int(dst)))
    result = torch.arange(count)[:, None].expand(-1, k).clone()
    for row, entries in enumerate(buckets):
        selected = [dst for _weight, dst in sorted(entries, reverse=True)[:k]]
        if selected:
            result[row, :len(selected)] = torch.tensor(selected)
    return result


def two_hop_physical_regions(
    centers: torch.Tensor, adjacency: torch.Tensor, xyz: torch.Tensor, radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unique two-hop surface candidates clipped by physical radius."""
    center = torch.as_tensor(centers, device=adjacency.device).long()
    first = adjacency[center]
    second = adjacency[first].flatten(1)
    rows = torch.cat([center[:, None], first, second], dim=1)
    rows, _ = rows.sort(dim=1)
    unique = torch.ones_like(rows, dtype=torch.bool)
    unique[:, 1:] = rows[:, 1:] != rows[:, :-1]
    distance = torch.linalg.vector_norm(xyz[rows] - xyz[center, None], dim=-1)
    mask = unique & (distance <= float(radius))
    # The center is guaranteed to survive sorting and the radius test.
    if not bool(mask.any(dim=1).all()):
        raise RuntimeError("physical surface region lost its center")
    return rows, mask


@torch.no_grad()
def build(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    field_path, graph_path, readout_path = map(Path, (
        args.field_checkpoint, args.support_graph, args.readout_checkpoint,
    ))
    field, field_payload = load_canonical_field_checkpoint(field_path, map_location="cpu")
    graph = torch.load(graph_path, map_location="cpu")
    readout, readout_payload = SurfaceRegionSummaryReadout.from_checkpoint(readout_path)
    if readout_payload["provenance"].get("uses_benchmark_scenes", True):
        raise ValueError("readout provenance is benchmark contaminated")
    mpr = torch.load(Path(field_payload["mpr_cache"]), map_location="cpu")
    xyz_global = torch.as_tensor(mpr["xyz"]).float().cpu()
    global_rows = torch.as_tensor(graph["global_rows"]).long().cpu()
    xyz = torch.as_tensor(graph["xyz"]).float().cpu()
    if not torch.equal(xyz, xyz_global[global_rows]):
        raise ValueError("support graph and canonical field geometry differ")
    adjacency = _adjacency(graph, int(args.graph_neighbors)).to(device)
    field, readout = field.to(device).eval(), readout.to(device).eval()
    head = SigLIP2SummaryHead.from_radio_checkpoint(args.radio_checkpoint).to(device).eval()
    for module in (field, readout, head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    radio = torch.empty(len(global_rows), 1280, dtype=torch.float16, device=device)
    for start in range(0, len(global_rows), int(args.radio_batch_size)):
        stop = min(start + int(args.radio_batch_size), len(global_rows))
        radio[start:stop] = field.radio_features(global_rows[start:stop].to(device)).half()
    reliability_source = torch.as_tensor(mpr.get("reliability")).float()[global_rows]
    if reliability_source.ndim != 2 or reliability_source.shape[1] < 2:
        raise ValueError("canonical MPR reliability needs coverage/agreement channels")
    reliability = reliability_source[:, :2].clamp_min(1e-6).log().mean(-1).exp()
    reliability[(reliability_source[:, :2] <= 0).any(-1)] = 0.0
    reliability = reliability.to(device)
    local_scale = torch.as_tensor(graph["local_sigma"]).float().clamp_min(1e-4).to(device)
    xyz_device = xyz.to(device)
    radii = tuple(float(value) for value in str(args.region_radii).replace(",", " ").split())
    descriptors = torch.zeros(len(global_rows), 1536, dtype=torch.float16)
    for start in range(0, len(global_rows), int(args.semantic_batch_size)):
        stop = min(start + int(args.semantic_batch_size), len(global_rows))
        centers = torch.arange(start, stop, device=device)
        scale_outputs = []
        for radius in radii:
            rows, mask = two_hop_physical_regions(centers, adjacency, xyz_device, radius)
            token_xyz = xyz_device[rows]
            token_scale = local_scale[rows, None].expand(-1, -1, 3)
            token_reliability = reliability[rows, None]
            geometry = surface_region_geometry(
                token_xyz, token_scale, torch.ones_like(token_reliability),
                token_reliability, float(radius), token_mask=mask,
            )
            summary = readout(
                radio[rows], geometry, token_mask=mask,
                reliability=token_reliability,
            )
            scale_outputs.append(F.normalize(head(summary[:, None])[:, 0].float(), dim=-1))
        descriptors[start:stop] = F.normalize(
            torch.stack(scale_outputs, dim=1).mean(1), dim=-1
        ).half().cpu()
    output_valid = torch.zeros(len(xyz_global), dtype=torch.bool)
    output_valid[global_rows] = True
    metadata = {
        "schema_version": 4, "feature_space": "official_siglip2_summary_descriptor",
        "source": "canonical_radio_surface_region_readout",
        "construction": "canonical_radio_surface_region_readout_then_official_summary_head",
        "canonical_radio_source": "field_decode_only", "mpr_radio_features_opened": False,
        "readout_checkpoint": str(readout_path.resolve()),
        "readout_checkpoint_sha256": _sha256(readout_path),
        "field_checkpoint": str(field_path.resolve()),
        "field_checkpoint_sha256": _sha256(field_path),
        "support_graph": str(graph_path.resolve()),
        "support_graph_sha256": _sha256(graph_path),
        "official_radio_checkpoint_sha256": _sha256(Path(args.radio_checkpoint)),
        "region_radii_m": list(radii), "region_topology": "two_hop_surface_graph_physical_clip",
        "query_set_invariant": True, "benchmark_images_opened": False,
        "official_summary_head": True, "custom_text_projection": False,
        "benchmark_masks_opened": False, "text_queries_opened": False,
        "cache_role": "disposable_derivative_not_scene_memory",
        "row_storage": "sparse_valid_rows_with_global_row_index",
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    # Semantic descriptors dominate cache size (1536 fp16 values per row).  Do
    # not materialize zero descriptors for invalid/background primitives.  The
    # global geometry and explicit row index retain an exact, lossless mapping;
    # consumers expand only when their downstream score representation needs it.
    torch.save({"xyz": xyz_global, "features": descriptors,
                "summary_features": descriptors, "global_rows": global_rows,
                "valid": output_valid, "metadata": metadata}, output)
    report = {"output": str(output.resolve()), "valid_primitives": int(output_valid.sum()),
              "total_primitives": len(output_valid), "metadata": metadata}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2))
    output.with_suffix(output.suffix + ".provenance.json").write_text(
        json.dumps({"cache": str(output.resolve()), "inputs": metadata}, indent=2)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--readout-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--region-radii", default="0.20,0.40,0.70")
    parser.add_argument("--graph-neighbors", type=int, default=16)
    parser.add_argument("--radio-batch-size", type=int, default=4096)
    parser.add_argument("--semantic-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radio-checkpoint", default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    args = parser.parse_args(); print(json.dumps(build(args), indent=2))


if __name__ == "__main__": main()
