#!/usr/bin/env python3
"""Compare sparse AcceptedV2 and dense SurfaceRegion support on source labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts.build_source_region_comembership_v1 import (
    _load_exact_instance_mass,
)
from radio_gs.scripts.eval_lerf_support_readout_oracle_d0_d5 import (
    _candidate_memberships,
    _enumerate_region_candidates,
)
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
)


PREREGISTRATION = Path(
    "paper/artifacts/source_region_support_axis_4096_vs_dense_oracle_preregistration_20260807.json"
)


def _source_targets(dense_mass: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    mass = torch.as_tensor(dense_mass).detach().float().cpu()
    if mass.ndim != 2 or mass.shape[1] < 2 or not bool(torch.isfinite(mass).all()):
        raise ValueError("source primitive instance mass differs")
    positive = mass[:, 1:]
    positive_mass = positive.sum(dim=1)
    observed = positive_mass > 0
    dominant = positive.argmax(dim=1) + 1
    dominant = torch.where(observed, dominant, torch.zeros_like(dominant))
    instance_ids = sorted(set(dominant[observed].tolist()))
    if not instance_ids:
        raise RuntimeError("source support audit found no positive instance")
    target = torch.stack([dominant == value for value in instance_ids], dim=1)
    seen = observed[:, None].expand_as(target).contiguous()
    return target.contiguous(), seen, instance_ids


def _score_global_rows(
    rows: torch.Tensor,
    *,
    target: torch.Tensor,
    observed: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    candidates = torch.as_tensor(rows).detach().long().cpu()
    truth = torch.as_tensor(target).detach().bool().cpu()
    seen = torch.as_tensor(observed).detach().bool().cpu()
    if candidates.ndim != 2 or truth.shape != seen.shape:
        raise ValueError("sparse support audit axes differ")
    target_count = truth.sum(dim=0).float()
    scores = torch.zeros(candidates.shape[0], truth.shape[1], dtype=torch.float32)
    for start in range(0, candidates.shape[0], int(batch_size)):
        batch = candidates[start : start + int(batch_size)]
        valid = batch >= 0
        safe = batch.clamp_min(0)
        active = valid[:, :, None] & seen[safe]
        intersection = (active & truth[safe]).sum(dim=1).float()
        predicted = active.sum(dim=1).float()
        union = target_count[None] + predicted - intersection
        scores[start : start + batch.shape[0]] = torch.where(
            union > 0, intersection / union, torch.ones_like(union)
        )
    return scores


def _summarize(choices: list[dict[str, Any]]) -> dict[str, Any]:
    if not choices:
        raise RuntimeError("source support audit produced no oracle choice")
    single = [float(row["D2"]["primitive_iou"]) for row in choices]
    union = [float(row["D3"]["primitive_iou"]) for row in choices]
    regions = [int(row["D3"]["regions"]) for row in choices]
    primitives = [int(row["D3"]["selected_rows"]) for row in choices]
    return {
        "instances": len(choices),
        "best_single_region_macro_primitive_iou": sum(single) / len(single),
        "up_to_eight_region_macro_primitive_iou": sum(union) / len(union),
        "mean_selected_regions": sum(regions) / len(regions),
        "mean_selected_primitives": sum(primitives) / len(primitives),
        "per_instance": choices,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"source support audit output exists: {output}")
    root = Path(__file__).resolve().parents[2]
    preregistration = root / PREREGISTRATION
    if sha256_file(preregistration) != args.expected_preregistration_sha256:
        raise ValueError("source support audit preregistration SHA-256 differs")
    accepted_raw, accepted_sha, accepted_path = load_torch_mapping(
        args.accepted_v2,
        expected_sha256=args.expected_accepted_v2_sha256,
        map_location="cpu",
        label="source support audit AcceptedV2",
    )
    accepted = shard.validate_accepted_region_authority(accepted_raw)
    if accepted["scene_id"] != "scene0001_00":
        raise ValueError("source support audit scene differs")
    graph, graph_sha, graph_path = load_torch_mapping(
        args.support_graph,
        expected_sha256=args.expected_support_graph_sha256,
        map_location="cpu",
        label="source support audit graph",
    )
    global_rows = torch.as_tensor(graph["global_rows"]).long().cpu()
    xyz = torch.as_tensor(graph["xyz"]).float().cpu()
    if (
        not bool((global_rows[1:] > global_rows[:-1]).all())
        or accepted["input_authority"]["support_graph_authority"][
            "support_graph_file_sha256"
        ]
        != graph_sha
    ):
        raise ValueError("source support graph and AcceptedV2 differ")
    dense_mass, instance_audit, marginal_record = _load_exact_instance_mass(
        manifest_path=Path(args.exact_marginal_authority),
        manifest_sha256=args.expected_exact_marginal_authority_sha256,
        instance_zip=Path(args.instance_zip),
        instance_zip_sha256=args.expected_instance_zip_sha256,
    )
    if dense_mass.shape[0] <= int(global_rows.max()):
        raise ValueError("source instance evidence and support rows differ")
    target, observed, instance_ids = _source_targets(dense_mass)

    sparse_rows = torch.as_tensor(accepted["region_rows"]).long().cpu()
    sparse_scores = _score_global_rows(
        sparse_rows,
        target=target,
        observed=observed,
        batch_size=args.score_batch_size,
    )
    sparse_scale = torch.as_tensor(accepted["scale_indices"]).to(torch.int8).cpu()
    anchor_token = torch.as_tensor(accepted["anchor_index"]).long().cpu()
    sparse_anchor = sparse_rows[
        torch.arange(sparse_rows.shape[0]), anchor_token
    ].to(torch.int32)
    full_rows = torch.arange(dense_mass.shape[0], dtype=torch.long)
    _, _, sparse_choices = _candidate_memberships(
        rows_by_candidate=sparse_rows.to(torch.int32),
        candidate_scores=sparse_scores,
        anchor_local=sparse_anchor,
        scale_index=sparse_scale,
        global_rows=full_rows,
        target_global=target,
        observed_global=observed,
    )

    support = PrimitiveSupportGraph(
        edge_index=graph["edge_index"],
        edge_weight=graph["edge_weight"],
        raw_affinity=graph["raw_affinity"],
        local_sigma=graph["local_sigma"],
        num_nodes=len(xyz),
        edge_channels=graph.get("edge_channels", {}),
    )
    contract = SurfaceRegionContractV2()
    dense_rows, dense_scores, dense_anchor, dense_scale = _enumerate_region_candidates(
        contract=contract,
        support=support,
        xyz=xyz,
        global_rows=global_rows,
        target_global=target,
        observed_global=observed,
        batch_size=args.region_batch_size,
    )
    _, _, dense_choices = _candidate_memberships(
        rows_by_candidate=dense_rows,
        candidate_scores=dense_scores,
        anchor_local=dense_anchor,
        scale_index=dense_scale,
        global_rows=global_rows,
        target_global=target,
        observed_global=observed,
    )
    sparse = _summarize(sparse_choices)
    dense = _summarize(dense_choices)
    delta = (
        dense["up_to_eight_region_macro_primitive_iou"]
        - sparse["up_to_eight_region_macro_primitive_iou"]
    )
    result = {
        "schema": "radio_gs.source_region_support_axis_audit.v1",
        "status": "source_only_support_axis_oracle_complete",
        "scene_id": "scene0001_00",
        "preregistration": file_record(preregistration),
        "inputs": {
            "accepted_v2": {"path": str(accepted_path), "sha256": accepted_sha},
            "support_graph": {"path": str(graph_path), "sha256": graph_sha},
            "exact_marginal": marginal_record,
            "instance_zip": {
                "path": str(Path(args.instance_zip).resolve()),
                "sha256": args.expected_instance_zip_sha256,
            },
        },
        "instance_ids": instance_ids,
        "instance_evidence_audit": instance_audit,
        "sparse_accepted_v2": sparse,
        "dense_surface_region": dense,
        "dense_minus_sparse_eight_region_macro_iou": delta,
        "sparse_axis_failure_signal": delta > 0.05,
        "source_access": {
            "source_instance_labels_opened": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_queries_opened": False,
            "target_metrics_computed": False,
        },
    }
    write_frozen_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-v2", required=True)
    parser.add_argument("--expected-accepted-v2-sha256", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--expected-support-graph-sha256", required=True)
    parser.add_argument("--exact-marginal-authority", required=True)
    parser.add_argument("--expected-exact-marginal-authority-sha256", required=True)
    parser.add_argument("--instance-zip", required=True)
    parser.add_argument("--expected-instance-zip-sha256", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--region-batch-size", type=int, default=256)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
