#!/usr/bin/env python3
"""Project canonical DINO rows into a frozen query-independent relation cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.interfaces.query_diffusion_cache import tensor_sha256
from radio_gs.scripts.build_canonical_support_graph import deterministic_feature_hash


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hash-dimension", type=int, default=256)
    parser.add_argument("--hash-batch-size", type=int, default=1024)
    parser.add_argument("--experiment-registration", default="")
    args = parser.parse_args()

    capability_path = Path(args.capability_cache).resolve()
    graph_path = Path(args.support_graph).resolve()
    graph = torch.load(graph_path, map_location="cpu")
    if not isinstance(graph, dict) or int(graph.get("schema_version", -1)) != 1:
        raise ValueError("unsupported canonical support graph")
    rows = torch.as_tensor(graph.get("global_rows")).long().cpu()
    xyz = torch.as_tensor(graph.get("xyz")).float().cpu()
    expected_field_hash = str(
        dict(graph.get("metadata", {}))
        .get("capability_metadata", {})
        .get("field_checkpoint_sha256", "")
    )
    bank = load_canonical_capability_bank(
        capability_path,
        expected_field_checkpoint_sha256=expected_field_hash,
    )
    if not torch.equal(rows, bank.global_rows) or not torch.equal(xyz, bank.xyz[rows]):
        raise ValueError("capability cache and support graph rows/geometry differ")
    declared_capability = Path(
        str(dict(graph.get("metadata", {})).get("capability_cache", ""))
    ).resolve()
    if declared_capability != capability_path:
        raise ValueError("support graph declares a different capability cache")
    features = bank.valid_feature_banks()["appearance"]
    projected = deterministic_feature_hash(
        features,
        int(args.hash_dimension),
        batch_size=int(args.hash_batch_size),
    )
    if projected.shape != (rows.numel(), int(args.hash_dimension)):
        raise RuntimeError("feature hashing returned an unexpected shape")
    registration_path = (
        Path(args.experiment_registration).resolve()
        if str(args.experiment_registration).strip()
        else None
    )
    if registration_path is not None and not registration_path.is_file():
        raise FileNotFoundError(registration_path)
    sidecar_path = capability_path.with_suffix(capability_path.suffix + ".json")
    result = {
        "schema_version": 1,
        "artifact_type": "query_conditioned_diffusion_relation_features",
        "features": projected,
        "global_rows": rows,
        "num_global_rows": int(graph["num_global_rows"]),
        "xyz_sha256": tensor_sha256(xyz),
        "metadata": {
            "source_capability_cache": str(capability_path),
            "source_capability_sidecar_sha256": (
                _sha256_file(sidecar_path) if sidecar_path.is_file() else ""
            ),
            "source_graph": str(graph_path),
            "source_graph_sha256": _sha256_file(graph_path),
            "field_checkpoint_sha256": expected_field_hash,
            "source_feature": "official_c_radio_v4_dino_v3_7b_feature_projection",
            "projection": "signed_multiplicative_hash",
            "input_dimension": int(features.shape[1]),
            "output_dimension": int(projected.shape[1]),
            "normalization": "l2_after_projection",
            "native_ludvig_dinov2_pca40_exact": False,
            "kernel_compatibility_scope": "release_kernel_compatible_c_radio_relation",
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
        if key not in {"features", "global_rows"}
    }
    sidecar.update(
        {
            "num_nodes": int(projected.shape[0]),
            "feature_dimension": int(projected.shape[1]),
            "feature_sha256": tensor_sha256(projected),
            "output": str(output),
            "output_sha256": _sha256_file(output),
        }
    )
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sidecar, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
