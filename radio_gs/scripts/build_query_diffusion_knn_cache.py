#!/usr/bin/env python3
"""Build a query-independent k=200(+self) Euclidean diffusion topology."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from radio_gs.interfaces.query_diffusion_cache import (
    build_exact_euclidean_knn,
    tensor_sha256,
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-neighbors", type=int, default=200)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--experiment-registration", default="")
    args = parser.parse_args()

    source = Path(args.support_graph).resolve()
    payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported canonical support graph")
    required = {"global_rows", "num_global_rows", "xyz"}
    if not required.issubset(payload):
        raise ValueError(f"support graph lacks keys: {sorted(required - set(payload))}")
    rows = torch.as_tensor(payload["global_rows"]).long().cpu()
    xyz = torch.as_tensor(payload["xyz"]).float().cpu()
    if xyz.shape != (rows.numel(), 3):
        raise ValueError("support graph xyz/global rows do not align")
    registration_path = (
        Path(args.experiment_registration).resolve()
        if str(args.experiment_registration).strip()
        else None
    )
    if registration_path is not None and not registration_path.is_file():
        raise FileNotFoundError(registration_path)
    neighbors = build_exact_euclidean_knn(
        xyz,
        num_neighbors=int(args.num_neighbors),
        include_self=True,
        workers=int(args.workers),
    )
    result = {
        "schema_version": 1,
        "artifact_type": "query_conditioned_diffusion_euclidean_knn",
        "neighbor_indices": neighbors,
        "global_rows": rows,
        "num_global_rows": int(payload["num_global_rows"]),
        "xyz_sha256": tensor_sha256(xyz),
        "metadata": {
            "source_graph": str(source),
            "source_graph_sha256": _sha256_file(source),
            "official_num_neighbors_parameter": int(args.num_neighbors),
            "include_self": True,
            "effective_k": int(neighbors.shape[1]),
            "construction": "scipy_ckdtree_exact_euclidean_query",
            "release_semantics": "query_k_is_num_neighbors_plus_one_and_self_is_retained",
            "experiment_registration": (
                str(registration_path) if registration_path is not None else ""
            ),
            "experiment_registration_sha256": (
                _sha256_file(registration_path) if registration_path is not None else ""
            ),
            "query_independent": True,
            "labels_opened": False,
            "target_masks_opened": False,
            "target_metrics_opened": False,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    sidecar = {
        key: value
        for key, value in result.items()
        if key not in {"neighbor_indices", "global_rows"}
    }
    sidecar["num_nodes"] = int(neighbors.shape[0])
    sidecar["effective_k"] = int(neighbors.shape[1])
    sidecar["output"] = str(output)
    sidecar["output_sha256"] = _sha256_file(output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sidecar, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
