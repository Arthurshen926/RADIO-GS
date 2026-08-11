"""Materialize the frozen query-independent typed primitive graph for AGILE v7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.querying.seed_conditioned_graph_residual import (
    reliability_weighted_support_graph,
)
from radio_gs.querying.support_solver import (
    SupportGraphConfig,
    build_primitive_support_graph,
    mix_support_graph_channels,
)
from radio_gs.scripts.train_query_likelihood_head_fixed import (
    _sha256,
    _write_json_no_clobber,
    _write_torch_no_clobber,
)


ARTIFACT_TYPE = "agile3d-query-independent-typed-primitive-graph-v7"
ALLOWED_SCENES = {"scene0000_00", "scene0002_00", "scene0005_00"}
CHANNEL_WEIGHTS = {"geometry": 0.2, "appearance": 0.4, "boundary": 0.4}


def _load_preregistered_bundle(
    path: Path, *, preregistration: Path
) -> dict[str, object]:
    contract = json.loads(preregistration.read_text(encoding="utf-8"))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    scene_id = str(payload.get("scene_id"))
    expected = contract["data_contract"]["canonical_gaussian_bundle_sha256"].get(
        scene_id
    )
    if scene_id not in ALLOWED_SCENES or expected is None or _sha256(path) != expected:
        raise PermissionError("primitive bundle is outside the preregistered source set")
    safety = payload.get("safety", {})
    for key, value in {
        "query_independent": True,
        "object_id_used": False,
        "clicks_opened": False,
        "gt_labels_opened": False,
        "test_labels_opened": False,
        "point_as_primitive_used": False,
    }.items():
        if safety.get(key) is not value:
            raise PermissionError(f"primitive bundle violates safety field {key}")
    return payload


def materialize(args: argparse.Namespace) -> tuple[Path, Path]:
    bundle_path = Path(args.primitive_bundle).resolve()
    preregistration = Path(args.preregistration).resolve()
    bundle = _load_preregistered_bundle(
        bundle_path, preregistration=preregistration
    )
    scene_id = str(bundle["scene_id"])
    config = SupportGraphConfig(
        neighbors=16,
        spatial_scale=2.0,
        appearance_temperature=0.10,
        boundary_temperature=0.10,
        minimum_sigma=1e-4,
        affinity_chunk_size=8192,
        topology_mode="symmetric_union",
    )
    raw = build_primitive_support_graph(
        torch.as_tensor(bundle["primitive_xyz"]),
        appearance_features=torch.as_tensor(bundle["appearance"]),
        boundary_features=torch.as_tensor(bundle["boundary"]),
        config=config,
        feature_affinity_device=args.device,
    )
    typed = mix_support_graph_channels(raw, CHANNEL_WEIGHTS)
    graph = reliability_weighted_support_graph(
        typed, torch.as_tensor(bundle["reliability"])
    )
    row = graph.edge_index[0]
    row_sum = torch.zeros(graph.num_nodes)
    if row.numel():
        row_sum.index_add_(0, row, graph.edge_weight.cpu())
    nonempty = torch.zeros(graph.num_nodes, dtype=torch.bool)
    if row.numel():
        nonempty[row.cpu()] = True
    if bool((graph.edge_weight < 0).any()) or (
        bool(nonempty.any())
        and not torch.allclose(
            row_sum[nonempty], torch.ones_like(row_sum[nonempty]), atol=2e-5, rtol=0
        )
    ):
        raise RuntimeError("materialized v7 transition violates nonnegative row sum")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 7,
        "artifact_type": ARTIFACT_TYPE,
        "scene_id": scene_id,
        "num_nodes": graph.num_nodes,
        "edge_index": graph.edge_index.to(torch.int32),
        "edge_weight": graph.edge_weight.half(),
        "raw_affinity": graph.raw_affinity.half(),
        "local_sigma": graph.local_sigma.float(),
        "edge_channels": {
            name: values.half() for name, values in raw.edge_channels.items()
        },
        "channel_weights": dict(CHANNEL_WEIGHTS),
        "support_graph_config": vars(config),
        "reliability_conductance": "sqrt(endpoint product), row-normalized",
        "primitive_bundle": {
            "path": str(bundle_path),
            "sha256": _sha256(bundle_path),
        },
        "preregistration": {
            "path": str(preregistration),
            "sha256": _sha256(preregistration),
        },
        "safety": {
            "query_independent": True,
            "labels_opened": False,
            "clicks_opened": False,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
            "point_as_primitive_used": False,
        },
    }
    graph_path = _write_torch_no_clobber(
        output_dir / f"{scene_id}.typed_graph_v7.pt", artifact
    )
    receipt = {
        "schema_version": 1,
        "artifact_type": "agile3d-typed-primitive-graph-v7-materialization-receipt",
        "scene_id": scene_id,
        "graph": {"path": str(graph_path), "sha256": _sha256(graph_path)},
        "num_nodes": graph.num_nodes,
        "directed_edge_count": int(graph.edge_index.shape[1]),
        "nonempty_row_count": int(nonempty.sum()),
        "maximum_row_sum_error": float((row_sum[nonempty] - 1).abs().max())
        if bool(nonempty.any())
        else 0.0,
        "minimum_edge_weight": float(graph.edge_weight.min())
        if graph.edge_weight.numel()
        else 0.0,
        "safety": artifact["safety"],
    }
    receipt_path = _write_json_no_clobber(
        output_dir / f"{scene_id}.typed_graph_v7.receipt.json", receipt
    )
    return graph_path, receipt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primitive-bundle", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    graph_path, receipt_path = materialize(parse_args())
    print(json.dumps({"graph": str(graph_path), "receipt": str(receipt_path)}))


if __name__ == "__main__":
    main()
