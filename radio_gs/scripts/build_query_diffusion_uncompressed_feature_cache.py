#!/usr/bin/env python3
"""Export official C-RADIO DINOv3 primitive rows without PCA or hashing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.interfaces.query_diffusion_cache import tensor_sha256


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(64 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostic-declaration", required=True)
    args = parser.parse_args()

    capability_path = Path(args.capability_cache).resolve()
    graph_path = Path(args.support_graph).resolve()
    declaration_path = Path(args.diagnostic_declaration).resolve()
    if not declaration_path.is_file():
        raise FileNotFoundError(declaration_path)
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    if (
        not isinstance(declaration, dict)
        or declaration.get("status") != "post_hoc_causal_diagnostic"
        or declaration.get("formal_preregistered_result") is not False
    ):
        raise ValueError("diagnostic declaration must be explicitly post-hoc")

    graph = torch.load(graph_path, map_location="cpu")
    if not isinstance(graph, dict) or int(graph.get("schema_version", -1)) != 1:
        raise ValueError("unsupported canonical support graph")
    rows = torch.as_tensor(graph.get("global_rows")).long().cpu()
    xyz = torch.as_tensor(graph.get("xyz")).float().cpu()
    graph_metadata = dict(graph.get("metadata", {}))
    expected_field_hash = str(
        dict(graph_metadata.get("capability_metadata", {})).get(
            "field_checkpoint_sha256", ""
        )
    )
    bank = load_canonical_capability_bank(
        capability_path,
        expected_field_checkpoint_sha256=expected_field_hash,
    )
    if not torch.equal(rows, bank.global_rows) or not torch.equal(xyz, bank.xyz[rows]):
        raise ValueError("capability cache and support graph rows/geometry differ")
    declared_capability = Path(str(graph_metadata.get("capability_cache", ""))).resolve()
    if declared_capability != capability_path:
        raise ValueError("support graph declares a different capability cache")
    signature = bank.signatures.get("appearance")
    if (
        signature is None
        or signature.adaptor_name != "dino_v3_7b.feature_projection"
        or int(signature.adaptor_output_dim) != 4096
    ):
        raise ValueError("appearance rows are not official DINOv3 4096-D features")
    features = bank.valid_feature_banks()["appearance"].contiguous()
    if features.shape != (rows.numel(), 4096):
        raise RuntimeError("official DINOv3 relation rows have an unexpected shape")
    if features.dtype != torch.float16:
        raise ValueError("official capability rows must retain their source fp16 storage")

    capability_sidecar = capability_path.with_suffix(capability_path.suffix + ".json")
    result = {
        "schema_version": 1,
        "artifact_type": "query_conditioned_diffusion_relation_features",
        "features": features,
        "global_rows": rows,
        "num_global_rows": int(graph["num_global_rows"]),
        "xyz_sha256": tensor_sha256(xyz),
        "metadata": {
            "source_capability_cache": str(capability_path),
            "source_capability_sidecar_sha256": _sha256_file(capability_sidecar),
            "source_graph": str(graph_path),
            "source_graph_sha256": _sha256_file(graph_path),
            "field_checkpoint_sha256": expected_field_hash,
            "source_feature": "official_c_radio_v4_dino_v3_7b_feature_projection",
            "projection": "none_uncompressed",
            "input_dimension": 4096,
            "output_dimension": 4096,
            "storage_dtype": "float16_source_exact",
            "normalization": "source_l2_then_runtime_fp32_l2",
            "lossy_relation_compression": False,
            "native_ludvig_dinov2_pca40_exact": False,
            "kernel_compatibility_scope": (
                "release_kernel_compatible_uncompressed_c_radio_relation"
            ),
            "diagnostic_status": "post_hoc_causal_diagnostic",
            "formal_preregistered_result": False,
            "scene_selection_after_full9": True,
            "diagnostic_declaration": str(declaration_path),
            "diagnostic_declaration_sha256": _sha256_file(declaration_path),
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
        key: value for key, value in result.items() if key not in {"features", "global_rows"}
    }
    sidecar.update(
        {
            "num_nodes": int(features.shape[0]),
            "feature_dimension": int(features.shape[1]),
            "feature_sha256": tensor_sha256(features),
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
