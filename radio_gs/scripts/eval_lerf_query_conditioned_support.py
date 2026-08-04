#!/usr/bin/env python3
"""Evaluate the preregistered target-blind LERF text support candidate.

The frozen LERF evaluator remains byte-for-byte unchanged.  Before it can
open benchmark labels, this wrapper validates the text-score authority and
query-independent surface graph, computes every propagated primitive support,
freezes a receipt, and substitutes only the final primitive membership.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.querying import adaptive_support as adaptive_support_impl
from radio_gs.querying import support_solver as support_solver_impl
from radio_gs.querying.adaptive_support import select_adaptive_otsu_support
from radio_gs.querying.support_solver import (
    PrimitiveSupportGraph,
    SupportSolverConfig,
    gate_support_graph_by_query_compatibility,
    mix_support_graph_channels,
    solve_primitive_support,
)
from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen_evaluator
from radio_gs.scripts.eval_lerf_adaptive_support_diagnostic import (
    _load_cache_inputs,
    authority_tensor_sha256,
    build_frozen_evaluator_argv,
    sha256_file,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


REGISTRATION_PATH = Path(
    "paper/artifacts/lerf_text_evidence_to_support_v1_registration_20260803.json"
)
EXPECTED_REGISTRATION_SHA256 = (
    "280d5083fd413f1aafc8b8e3a0dd59b719643ad9f640081543a67846ede0cd73"
)
CHANNEL_WEIGHTS = {"geometry": 0.2, "appearance": 0.4, "boundary": 0.4}
SOLVER_CONFIG = SupportSolverConfig(
    iterations=12,
    residual=0.30,
    unary_temperature=1.0,
    support_threshold=0.50,
    solver_type="diffusion",
)
OTSU_STAGES = 3
PROBABILITY_EPS = 1e-6


def _load_surface_graph(
    path: str | Path,
    *,
    score_cache: Mapping[str, Any],
) -> tuple[PrimitiveSupportGraph, torch.Tensor, dict[str, Any]]:
    source = Path(path).resolve()
    payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported canonical support graph")
    required = {
        "global_rows",
        "num_global_rows",
        "xyz",
        "edge_index",
        "edge_weight",
        "raw_affinity",
        "edge_channels",
        "local_sigma",
        "metadata",
    }
    if not required.issubset(payload):
        raise ValueError(f"support graph lacks keys: {sorted(required - set(payload))}")
    rows = torch.as_tensor(payload["global_rows"]).long().cpu()
    valid = torch.as_tensor(score_cache["valid"]).bool().cpu()
    expected_rows = torch.where(valid)[0]
    if not torch.equal(rows, expected_rows):
        raise ValueError("support graph nodes differ from score-authority valid rows")
    if int(payload["num_global_rows"]) != int(valid.numel()):
        raise ValueError("support graph global row count differs from text scores")
    graph_xyz = torch.as_tensor(payload["xyz"]).float().cpu()
    score_xyz = torch.as_tensor(score_cache["xyz"]).float().cpu()
    if graph_xyz.shape != (rows.numel(), 3) or not torch.equal(
        graph_xyz, score_xyz[rows]
    ):
        raise ValueError("support graph geometry/row order differs from text scores")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("support graph metadata is malformed")
    if any(
        metadata.get(key) is not False
        for key in ("benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened")
    ):
        raise ValueError("support graph is not query-independent")
    graph_field_sha = str(
        dict(metadata.get("capability_metadata", {})).get(
            "field_checkpoint_sha256", ""
        )
    )
    if not graph_field_sha or graph_field_sha != str(
        score_cache.get("field_checkpoint_sha256", "")
    ):
        raise ValueError("support graph and text scores bind different canonical fields")
    channels = {
        str(name): torch.as_tensor(values).float()
        for name, values in dict(payload["edge_channels"]).items()
    }
    missing = sorted(set(CHANNEL_WEIGHTS) - set(channels))
    if missing:
        raise ValueError(f"support graph lacks preregistered channels: {missing}")
    graph = PrimitiveSupportGraph(
        edge_index=payload["edge_index"],
        edge_weight=torch.as_tensor(payload["edge_weight"]).float(),
        raw_affinity=torch.as_tensor(payload["raw_affinity"]).float(),
        local_sigma=torch.as_tensor(payload["local_sigma"]).float(),
        num_nodes=int(rows.numel()),
        edge_channels=channels,
    )
    receipt = {
        "path": str(source),
        "sha256": sha256_file(source),
        "field_checkpoint_sha256": graph_field_sha,
        "num_nodes": int(rows.numel()),
        "num_edges": int(graph.edge_index.shape[1]),
        "global_rows_sha256": authority_tensor_sha256(rows),
        "xyz_sha256": frozen_evaluator.tensor_sha256_float32(graph_xyz),
        "available_channels": sorted(channels),
    }
    return graph, rows, receipt


def precompute_query_conditioned_membership(
    cache: Mapping[str, Any],
    graph: PrimitiveSupportGraph,
    global_rows: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Freeze all graph propagation and Otsu selection without target data."""

    readout = frozen_evaluator.vala_multiscale_knn_peak_select_scores(
        cache["query_scores"],
        cache["xyz"],
        k=10,
        valid_mask=cache["valid"],
    )
    processed = readout.scores.detach().float().cpu()
    typed = mix_support_graph_channels(graph, CHANNEL_WEIGHTS).to(device)
    propagated = torch.zeros_like(processed)
    for query_index in range(int(processed.shape[1])):
        prior = processed[global_rows, query_index].to(device).clamp(
            PROBABILITY_EPS, 1.0 - PROBABILITY_EPS
        )
        gated = gate_support_graph_by_query_compatibility(typed, prior)
        unary = torch.logit(prior)
        solved = solve_primitive_support(
            gated,
            unary,
            config=SOLVER_CONFIG,
        )
        if not bool(torch.isfinite(solved).all()):
            raise FloatingPointError("query-conditioned text support is non-finite")
        propagated[global_rows, query_index] = solved.detach().cpu()
        del gated, unary, solved, prior
    selection = select_adaptive_otsu_support(
        propagated,
        cache["valid"],
        otsu_stages=OTSU_STAGES,
    )
    return {
        "selection": selection,
        "processed_scores": processed,
        "propagated_scores": propagated,
        "processed_scores_sha256": frozen_evaluator.tensor_sha256_float32(processed),
        "propagated_scores_sha256": frozen_evaluator.tensor_sha256_float32(propagated),
        "membership_sha256": authority_tensor_sha256(selection.selected.bool()),
        "selected_scale_indices": [
            int(value) for value in readout.selected_scale_indices.tolist()
        ],
    }


def _write_prelabel_receipt(path: Path, receipt: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"pre-label receipt already differs: {path}")
    else:
        path.write_text(serialized, encoding="utf-8")
    return sha256_file(path)


def _result_path(output_dir: str | Path, scene: str) -> Path:
    return Path(output_dir) / scene / "lerf_direct_3d_selection_results.json"


def run(args: argparse.Namespace) -> Path:
    # Every operation through receipt_path happens before frozen_evaluator.main,
    # whose category/annotation load is the first target-label access.
    cache = _load_cache_inputs(args.ours_multiscale_query_score_cache)
    renderer_sha = sha256_file(args.checkpoint)
    if renderer_sha != cache["renderer_geometry_checkpoint_sha256"]:
        raise ValueError("renderer checkpoint differs from text-score authority")
    registration = REGISTRATION_PATH.resolve()
    if not registration.is_file() or sha256_file(registration) != EXPECTED_REGISTRATION_SHA256:
        raise ValueError("LERF text support registration is missing or changed")
    graph, global_rows, graph_receipt = _load_surface_graph(
        args.support_graph,
        score_cache=cache,
    )
    precomputed = precompute_query_conditioned_membership(
        cache,
        graph,
        global_rows,
        device=torch.device(f"cuda:{int(args.gpu)}"),
    )
    selection = precomputed["selection"]
    method = {
        "schema_version": "lerf_text_evidence_to_support_v1_prelabel_receipt",
        "scene": args.scene,
        "method_config": {
            "channel_weights": CHANNEL_WEIGHTS,
            "query_edge_gate": "sqrt(P_i*P_j)",
            "solver": asdict(SOLVER_CONFIG),
            "probability_eps": PROBABILITY_EPS,
            "otsu_stages": OTSU_STAGES,
        },
        "method_config_sha256": canonical_json_sha256(
            {
                "channel_weights": CHANNEL_WEIGHTS,
                "query_edge_gate": "sqrt(P_i*P_j)",
                "solver": asdict(SOLVER_CONFIG),
                "probability_eps": PROBABILITY_EPS,
                "otsu_stages": OTSU_STAGES,
            }
        ),
        "experiment_registration": str(registration),
        "experiment_registration_sha256": EXPECTED_REGISTRATION_SHA256,
        "query_score_cache": str(Path(args.ours_multiscale_query_score_cache).resolve()),
        "query_score_cache_sha256": sha256_file(args.ours_multiscale_query_score_cache),
        "query_scores_authority_sha256": cache["query_scores_sha256"],
        "valid_authority_sha256": cache["valid_sha256"],
        "surface_graph": graph_receipt,
        "renderer_checkpoint_sha256": renderer_sha,
        "processed_scores_sha256": precomputed["processed_scores_sha256"],
        "propagated_scores_sha256": precomputed["propagated_scores_sha256"],
        "membership_sha256": precomputed["membership_sha256"],
        "thresholds": [float(value) for value in selection.thresholds.tolist()],
        "selected_counts": [int(value) for value in selection.selected_counts.tolist()],
        "selected_scale_indices": precomputed["selected_scale_indices"],
        "query_ids": list(cache["query_ids"]),
        "target_rgb_opened": False,
        "target_masks_opened": False,
        "target_metrics_opened": False,
        "frozen_evaluator_source_sha256": sha256_file(frozen_evaluator.__file__),
        "implementation_source_sha256": sha256_file(__file__),
        "implementation_dependency_source_sha256": {
            "adaptive_support": sha256_file(adaptive_support_impl.__file__),
            "support_solver": sha256_file(support_solver_impl.__file__),
        },
    }
    receipt_path = Path(args.output_dir).resolve() / "method_receipt.prelabel.json"
    receipt_sha = _write_prelabel_receipt(receipt_path, method)

    calls: list[str] = []
    original_selector = frozen_evaluator.select_gaussians_from_scores

    def query_conditioned_selector(scores, spec, *, min_select=1):
        if spec != frozen_evaluator.SelectionSpec(
            "score_threshold", frozen_evaluator.OURS_VALA_MASK_THRESHOLD
        ):
            raise ValueError("query-conditioned support requires the frozen singleton spec")
        if int(min_select) != 0:
            raise ValueError("query-conditioned support requires frozen min_select=0")
        if calls:
            raise RuntimeError("query-conditioned selector must be called exactly once")
        actual_sha = frozen_evaluator.tensor_sha256_float32(scores)
        if actual_sha != precomputed["processed_scores_sha256"]:
            raise ValueError("frozen evaluator scores differ from precomputed evidence")
        if tuple(scores.shape) != tuple(selection.selected.shape):
            raise ValueError("precomputed membership shape differs")
        calls.append(actual_sha)
        return selection.selected

    previous_argv = sys.argv
    frozen_evaluator.select_gaussians_from_scores = query_conditioned_selector
    try:
        sys.argv = build_frozen_evaluator_argv(args)
        frozen_evaluator.main()
    finally:
        frozen_evaluator.select_gaussians_from_scores = original_selector
        sys.argv = previous_argv
    if len(calls) != 1:
        raise RuntimeError(f"query-conditioned selector call count differs: {len(calls)}")

    result_path = _result_path(args.output_dir, args.scene)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["query_conditioned_text_support"] = {
        "status": "preregistered_target_blind_candidate_scored_by_frozen_evaluator",
        "prelabel_receipt": str(receipt_path),
        "prelabel_receipt_sha256": receipt_sha,
        **method,
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return result_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", required=True, choices=list(frozen_evaluator.LERF_OVS_SCENES))
    parser.add_argument("--ours_multiscale_query_score_cache", required=True)
    parser.add_argument("--support_graph", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_dir", default="/mnt/pool/sqy/3d_understanding/lerf_ovs/label")
    parser.add_argument("--summary_head_weights", default="checkpoints/siglip2_summary_head.pth")
    parser.add_argument("--text_embedding_cache", default="checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt")
    parser.add_argument("--canonical_embedding_cache", default="checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt")
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(run(args))


if __name__ == "__main__":
    main()
