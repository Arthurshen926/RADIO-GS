"""Materialize a sealed, label-free primitive/view co-visibility carrier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.querying.view_support_carrier import (
    build_compact_view_support,
    dense_support_to_csr,
    edge_covisibility_from_support,
)
from radio_gs.scripts.train_query_likelihood_head_fixed import (
    _sha256,
    _write_json_no_clobber,
    _write_torch_no_clobber,
)


ARTIFACT = "agile3d-query-independent-view-support-carrier-v11"


def _validate_safe_metadata(metadata: dict[str, object]) -> None:
    for key in ("benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened"):
        if metadata.get(key, False) is not False:
            raise PermissionError(f"responsibility authority violates safety field {key}")
    if metadata.get("query_independent", True) is not True:
        raise PermissionError("responsibility authority is not query independent")


def _load_legacy_top1(path: Path) -> tuple[list[torch.Tensor], list[int], dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported legacy top1 responsibility authority")
    metadata = payload.get("metadata")
    assignments = payload.get("assignments")
    if not isinstance(metadata, dict) or not isinstance(assignments, list):
        raise ValueError("legacy responsibility authority is incomplete")
    _validate_safe_metadata(metadata)
    frame_indices = metadata.get("selected_frame_indices")
    if not isinstance(frame_indices, list) or len(frame_indices) != len(assignments):
        raise ValueError("legacy responsibility frame axis is inconsistent")
    view_ids: list[torch.Tensor] = []
    total_hits = 0
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("legacy responsibility assignment is malformed")
        ids = torch.as_tensor(assignment["gaussian_ids"]).long().cpu()
        weights = torch.as_tensor(assignment["weights"]).float().cpu()
        if ids.ndim != 1 or weights.shape != ids.shape:
            raise ValueError("legacy responsibility tensors are inconsistent")
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("legacy responsibility weights must be positive finite")
        view_ids.append(ids)
        total_hits += int(ids.numel())
    source = {
        "format": "legacy_top1_contribution",
        "path": str(path),
        "sha256": _sha256(path),
        "view_records": [],
        "total_pixel_primitive_hits": total_hits,
        "incidence_rule": "any positive accepted top1 responsibility hit in the view",
        "metadata": metadata,
    }
    return view_ids, [int(v) for v in frame_indices], source


def _load_exact_marginal(path: Path) -> tuple[list[torch.Tensor], list[int], dict[str, object]]:
    authority = json.loads(path.read_text())
    if authority.get("schema") != "radio_gs.sparse_exact_marginal_responsibility_authority.v1":
        raise ValueError("unsupported exact marginal responsibility authority")
    metadata = authority.get("metadata")
    views = authority.get("views")
    if not isinstance(metadata, dict) or not isinstance(views, list):
        raise ValueError("exact marginal responsibility authority is incomplete")
    _validate_safe_metadata(metadata)
    root = path.parent.resolve()
    view_ids: list[torch.Tensor] = []
    records: list[dict[str, object]] = []
    for expected_view, record in enumerate(views):
        sidecar = (root / str(record["relative_path"])).resolve()
        if root not in sidecar.parents or not sidecar.is_file():
            raise ValueError("unsafe or missing responsibility view sidecar")
        observed_sha = _sha256(sidecar)
        if observed_sha != record["sha256"]:
            raise ValueError("responsibility view sidecar SHA mismatch")
        payload = torch.load(sidecar, map_location="cpu", weights_only=True)
        if (
            payload.get("schema") != "radio_gs.sparse_exact_marginal_responsibility_view.v1"
            or int(payload["view_index"]) != expected_view
            or int(payload["frame_index"]) != int(record["frame_index"])
            or payload["formula_sha256"] != authority["formula_sha256"]
        ):
            raise ValueError("responsibility view sidecar differs from authority")
        ids = torch.as_tensor(payload["gaussian_ids"]).long().cpu()
        weights = torch.as_tensor(payload["base_weights"]).float().cpu()
        if ids.ndim != 1 or weights.shape != ids.shape:
            raise ValueError("exact marginal responsibility tensors are inconsistent")
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("exact marginal base weights must be positive finite")
        view_ids.append(ids)
        records.append(
            {
                "view_index": expected_view,
                "frame_index": int(record["frame_index"]),
                "path": str(sidecar),
                "sha256": observed_sha,
                "num_hits": int(ids.numel()),
            }
        )
    source = {
        "format": "sparse_exact_marginal",
        "path": str(path),
        "sha256": _sha256(path),
        "view_records": records,
        "total_pixel_primitive_hits": sum(int(v["num_hits"]) for v in records),
        "formula_sha256": authority["formula_sha256"],
        "formula_contract": authority["formula_contract"],
        "incidence_rule": "any positive accepted exact-marginal responsibility hit in the view",
        "metadata": metadata,
    }
    return view_ids, [int(v) for v in authority["frame_indices"]], source


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> tuple[Path, Path]:
    bundle_path = Path(args.primitive_bundle).resolve()
    graph_path = Path(args.typed_graph).resolve()
    responsibility_path = Path(args.responsibility).resolve()
    for path, expected, label in [
        (bundle_path, args.primitive_bundle_sha256, "primitive bundle"),
        (graph_path, args.typed_graph_sha256, "typed graph"),
        (responsibility_path, args.responsibility_sha256, "responsibility authority"),
    ]:
        if _sha256(path) != expected:
            raise ValueError(f"{label} SHA mismatch")
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    graph = torch.load(graph_path, map_location="cpu", weights_only=True)
    scene_id = str(bundle["scene_id"])
    if scene_id != args.scene_id or graph.get("scene_id") != scene_id:
        raise ValueError("scene identity differs across sealed inputs")
    bundle_safety = bundle.get("safety", {})
    graph_safety = graph.get("safety", {})
    for safety in (bundle_safety, graph_safety):
        if (
            safety.get("query_independent") is not True
            or safety.get("point_as_primitive_used") is not False
            or safety.get("test_labels_opened") is not False
        ):
            raise PermissionError("sealed input violates v11 carrier safety")
    if bundle_safety.get("gt_labels_opened") is not False:
        raise PermissionError("primitive bundle opened GT labels")
    global_rows = torch.as_tensor(bundle["global_rows"]).long().cpu()
    edge_index = torch.as_tensor(graph["edge_index"]).long().cpu()
    if int(graph["num_nodes"]) != global_rows.numel():
        raise ValueError("typed graph and primitive bundle row counts differ")
    if args.responsibility_format == "legacy_top1":
        view_ids, frame_indices, source = _load_legacy_top1(responsibility_path)
    else:
        view_ids, frame_indices, source = _load_exact_marginal(responsibility_path)
    support = build_compact_view_support(
        global_rows=global_rows, view_global_ids=view_ids
    )
    crow, col = dense_support_to_csr(support)
    unique_edge_mask = edge_index[0] < edge_index[1]
    unique_edges = edge_index[:, unique_edge_mask]
    covisibility = edge_covisibility_from_support(
        support=support,
        edge_index=unique_edges,
        chunk_size=args.chunk_size,
    )
    primitive_counts = support.sum(dim=1)
    shared = covisibility["shared_view_count"].long()
    artifact = {
        "schema_version": 11,
        "artifact_type": ARTIFACT,
        "scene_id": scene_id,
        "primitive_count": int(support.shape[0]),
        "view_count": int(support.shape[1]),
        "view_frame_indices": torch.tensor(frame_indices, dtype=torch.int32),
        "primitive_view_support_csr": {
            "crow_indices": crow,
            "col_indices": col,
            "shape": [int(support.shape[0]), int(support.shape[1])],
            "value_semantics": True,
        },
        "primitive_view_count": primitive_counts.to(torch.uint8),
        "edge_index": unique_edges.to(torch.int32),
        "edge_covisibility": covisibility,
        "contracts": {
            "support": "binary accepted-renderer-responsibility incidence; repeated pixel hits collapse within each view",
            "jaccard": "shared_view_count / union_view_count; zero when union is zero",
            "source_given_target": "shared_view_count / target_view_count; zero for unobserved target",
            "target_given_source": "shared_view_count / source_view_count; zero for unobserved source",
            "overlap_coefficient": "shared_view_count / min(endpoint_view_counts); zero if either endpoint is unobserved",
            "edge_axis": "one canonical source<target row per symmetric typed-graph edge",
        },
        "source_responsibility": source,
        "primitive_bundle": {"path": str(bundle_path), "sha256": args.primitive_bundle_sha256},
        "typed_graph": {"path": str(graph_path), "sha256": args.typed_graph_sha256},
        "summary": {
            "support_nnz": int(support.sum()),
            "observed_primitive_count": int((primitive_counts > 0).sum()),
            "unobserved_primitive_count": int((primitive_counts == 0).sum()),
            "mean_views_per_primitive": float(primitive_counts.float().mean()),
            "median_views_per_primitive": float(primitive_counts.float().median()),
            "unique_edge_count": int(unique_edges.shape[1]),
            "edges_with_shared_view": int((shared > 0).sum()),
            "mean_edge_jaccard": float(covisibility["jaccard"].float().mean()) if shared.numel() else 0.0,
            "mean_edge_overlap_coefficient": float(covisibility["overlap_coefficient"].float().mean()) if shared.numel() else 0.0,
        },
        "safety": {
            "query_independent": True,
            "labels_opened": False,
            "clicks_opened": False,
            "masks_opened": False,
            "source_rgb_opened": False,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
            "point_as_primitive_used": False,
        },
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = _write_torch_no_clobber(
        output_dir / f"{scene_id}.view_support_v11.pt", artifact
    )
    receipt = {
        "schema_version": 11,
        "artifact_type": "agile3d-view-support-carrier-v11-receipt",
        "scene_id": scene_id,
        "artifact": {"path": str(artifact_path), "sha256": _sha256(artifact_path)},
        "summary": artifact["summary"],
        "input_authorities": {
            "primitive_bundle": artifact["primitive_bundle"],
            "typed_graph": artifact["typed_graph"],
            "responsibility": {
                "path": source["path"],
                "sha256": source["sha256"],
                "format": source["format"],
                "view_records": source["view_records"],
            },
        },
        "safety": artifact["safety"],
    }
    receipt_path = _write_json_no_clobber(
        output_dir / f"{scene_id}.view_support_v11.receipt.json", receipt
    )
    return artifact_path, receipt_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--primitive-bundle", required=True)
    parser.add_argument("--primitive-bundle-sha256", required=True)
    parser.add_argument("--typed-graph", required=True)
    parser.add_argument("--typed-graph-sha256", required=True)
    parser.add_argument("--responsibility", required=True)
    parser.add_argument("--responsibility-sha256", required=True)
    parser.add_argument(
        "--responsibility-format",
        choices=("legacy_top1", "exact_marginal"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=131072)
    args = parser.parse_args()
    artifact, receipt = materialize(args)
    print(json.dumps({"artifact": str(artifact), "receipt": str(receipt)}, indent=2))


if __name__ == "__main__":
    main()
