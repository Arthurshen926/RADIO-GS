#!/usr/bin/env python3
"""Audit exact single/batched region equivalence on a real support graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.querying.support_solver import PrimitiveSupportGraph


def run(args: argparse.Namespace) -> dict:
    payload = torch.load(args.support_graph, map_location="cpu")
    xyz = torch.as_tensor(payload["xyz"]).float()
    graph = PrimitiveSupportGraph(
        edge_index=payload["edge_index"], edge_weight=payload["edge_weight"],
        raw_affinity=payload["raw_affinity"], local_sigma=payload["local_sigma"],
        num_nodes=len(xyz), edge_channels=payload.get("edge_channels", {}),
    )
    contract = SurfaceRegionContractV2()
    prepared = contract.prepare_graph(graph, xyz)
    rng = random.Random(int(args.seed))
    anchors = rng.sample(range(graph.num_nodes), min(int(args.anchors), graph.num_nodes))
    comparisons = 0; minimum_jaccard = 1.0; exact = True
    empty_core = 0; region_sizes = []
    for start in range(0, len(anchors), int(args.batch_size)):
        chunk = anchors[start:start + int(args.batch_size)]
        for radius in contract.radii_m:
            batched = contract.expand_batch(
                graph, xyz, chunk, radius, prepared_graph=prepared
            )
            for anchor, inferred in zip(chunk, batched):
                trained = contract.expand(
                    graph, xyz, anchor, radius, prepared_graph=prepared
                )
                left, right = set(trained[0].tolist()), set(inferred[0].tolist())
                union = left | right
                jaccard = len(left & right) / max(1, len(union))
                minimum_jaccard = min(minimum_jaccard, jaccard)
                exact &= torch.equal(trained[0], inferred[0])
                exact &= torch.equal(trained[1], inferred[1])
                exact &= torch.equal(trained[2], inferred[2])
                empty_core += int(not bool(trained[1].any()))
                region_sizes.append(len(trained[0])); comparisons += 1
    report = {
        "schema_version": 1,
        "audit": "surface_region_contract_v2_train_inference_exactness",
        "support_graph": str(Path(args.support_graph).resolve()),
        "region_contract": contract.to_dict(),
        "region_contract_sha256": contract.digest,
        "anchors": len(anchors), "scales": len(contract.radii_m),
        "comparisons": comparisons, "minimum_jaccard": minimum_jaccard,
        "ordered_rows_masks_distances_exact": bool(exact),
        "empty_core_regions": empty_core,
        "region_size_min": min(region_sizes),
        "region_size_mean": sum(region_sizes) / len(region_sizes),
        "region_size_max": max(region_sizes),
        "regions_below_minimum_tokens": sum(
            size < contract.minimum_tokens for size in region_sizes
        ),
        "acceptance_passed": bool(exact and minimum_jaccard == 1.0 and empty_core == 0),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchors", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
