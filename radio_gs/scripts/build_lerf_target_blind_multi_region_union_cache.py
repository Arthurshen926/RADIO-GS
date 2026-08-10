#!/usr/bin/env python3
"""Build a target-blind SurfaceRegion union score cache for frozen LERF eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces.surface_region_selection import (
    surface_region_contract_from_metadata,
)
from radio_gs.querying.multi_region_union_readout import (
    MultiRegionUnionConfig,
    greedy_connected_expected_mass_union_readout,
    greedy_novelty_union_readout,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMAS = {
    "novelty_v1": "radio_gs.lerf_target_blind_multi_region_union_external_scores.v1",
    "connected_v2": "radio_gs.lerf_target_blind_connected_multi_region_union_external_scores.v2",
}


def _require_file(path: str | Path, expected_sha256: str, label: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or sha256_file(source) != str(expected_sha256):
        raise ValueError(f"{label} is missing or has a different SHA-256")
    return source


def _surface_region_rows(
    *,
    contract: Any,
    support: PrimitiveSupportGraph,
    xyz: torch.Tensor,
    global_rows: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enumerate immutable scale-major SurfaceRegion rows and semantic cores."""

    prepared = contract.prepare_graph(support, xyz)
    anchor_count = int(global_rows.numel())
    region_count = anchor_count * len(contract.radii_m)
    width = int(contract.maximum_tokens)
    rows = torch.full((region_count, width), -1, dtype=torch.int32)
    core = torch.zeros((region_count, width), dtype=torch.bool)
    anchor_global = torch.empty(region_count, dtype=torch.int32)
    scale_index = torch.empty(region_count, dtype=torch.int8)
    anchors = torch.arange(anchor_count, dtype=torch.long)
    cursor = 0
    for scale, radius in enumerate(contract.radii_m):
        for start in range(0, anchor_count, int(batch_size)):
            selected = anchors[start : start + int(batch_size)]
            regions = contract.expand_batch(
                support,
                xyz,
                selected.tolist(),
                float(radius),
                prepared_graph=prepared,
            )
            for local, (region_rows, region_core, _distances) in enumerate(regions):
                local_rows = torch.as_tensor(region_rows).long()
                semantic_core = torch.as_tensor(region_core).bool()
                if local_rows.numel() > width:
                    raise RuntimeError(
                        "SurfaceRegion exceeds its registered token width"
                    )
                candidate = cursor + local
                rows[candidate, : local_rows.numel()] = global_rows[local_rows].to(
                    torch.int32
                )
                core[candidate, : local_rows.numel()] = semantic_core
                anchor_global[candidate] = int(global_rows[int(selected[local])])
                scale_index[candidate] = int(scale)
            cursor += len(regions)
    if cursor != region_count or not bool(core.any(dim=1).all()):
        raise RuntimeError("SurfaceRegion enumeration differs from its contract")
    return rows, core, anchor_global, scale_index


def _scale_major_anchor_probability(
    remapped_probability: torch.Tensor,
    global_rows: torch.Tensor,
) -> torch.Tensor:
    """Align independent scale responses with scale-major region candidates."""

    values = torch.as_tensor(remapped_probability).detach().float().cpu()
    rows = torch.as_tensor(global_rows).detach().long().cpu().reshape(-1)
    if values.ndim != 3 or values.shape[1] != 3 or values.shape[2] <= 0:
        raise ValueError("remapped probability must align as [N,3,Q]")
    if rows.numel() == 0 or int(rows.min()) < 0 or int(rows.max()) >= values.shape[0]:
        raise ValueError("global rows are empty or out of probability range")
    # Candidate enumeration is scale-major, then immutable local anchor row.
    # Preserve that exact ordering without peak selection or scale broadcast.
    return values[rows].permute(1, 0, 2).reshape(-1, values.shape[2]).contiguous()


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output_cache).expanduser().resolve()
    report_path = Path(args.output_report).expanduser().resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refuses to clobber multi-region union outputs")

    preregistration = _require_file(
        args.preregistration,
        args.expected_preregistration_sha256,
        "multi-region preregistration",
    )
    scale_addendum = _require_file(
        args.scale_alignment_addendum,
        args.expected_scale_alignment_addendum_sha256,
        "scale-alignment addendum",
    )
    selector_source = _require_file(
        args.selector_source,
        args.expected_selector_source_sha256,
        "multi-region selector source",
    )
    descriptor_report_path = _require_file(
        args.descriptor_report,
        args.expected_descriptor_report_sha256,
        "descriptor report",
    )
    positive_path = _require_file(
        args.positive_cache,
        args.expected_positive_cache_sha256,
        "positive O0 cache",
    )
    negative_path = _require_file(
        args.negative_cache,
        args.expected_negative_cache_sha256,
        "canonical-negative O0 cache",
    )
    graph_path = _require_file(
        args.support_graph,
        args.expected_support_graph_sha256,
        "support graph",
    )

    positive_payload, _, _ = load_torch_mapping(
        positive_path,
        expected_sha256=args.expected_positive_cache_sha256,
        map_location="cpu",
        label="positive O0 cache",
    )
    negative_payload, _, _ = load_torch_mapping(
        negative_path,
        expected_sha256=args.expected_negative_cache_sha256,
        map_location="cpu",
        label="canonical-negative O0 cache",
    )
    query_ids = tuple(str(value) for value in positive_payload["query_ids"])
    positive = frozen.validate_ours_multiscale_query_score_cache(
        positive_payload,
        expected_xyz=torch.as_tensor(positive_payload["xyz"]),
        expected_query_ids=query_ids,
        expected_renderer_geometry_checkpoint_sha256=(
            args.expected_renderer_geometry_checkpoint_sha256
        ),
    )
    negative = frozen.validate_ours_multiscale_query_score_cache(
        negative_payload,
        expected_xyz=positive_payload["xyz"],
        expected_query_ids=frozen.NEGATIVE_PROMPTS,
        expected_renderer_geometry_checkpoint_sha256=(
            args.expected_renderer_geometry_checkpoint_sha256
        ),
    )
    for field in (
        "valid",
        "scale_ids",
        "scale_radii_m",
        "xyz_sha256",
        "field_checkpoint_sha256",
        "readout_checkpoint_sha256",
        "renderer_geometry_checkpoint_sha256",
    ):
        left = getattr(positive, field)
        right = getattr(negative, field)
        equal = (
            torch.equal(left, right)
            if isinstance(left, torch.Tensor)
            else left == right
        )
        if not bool(equal):
            raise ValueError(f"positive/negative O0 cache {field} differs")

    probability = frozen.canonical_negative_relevancy_query_scores(
        positive.query_scores,
        negative.query_scores,
        logit_scale=10.0,
    )
    count, scales, queries = probability.shape
    smoothed = frozen.vala_knn_smoothed_scores(
        probability.reshape(count, scales * queries),
        positive_payload["xyz"],
        k=10,
        chunk_size=args.knn_chunk_size,
        valid_mask=positive.valid,
    ).reshape(count, scales, queries)
    remapped_probability = frozen.vala_minmax_remap_scores(
        smoothed.reshape(count, scales * queries),
        valid_mask=positive.valid,
    ).reshape(count, scales, queries)

    descriptor_report = json.loads(descriptor_report_path.read_text(encoding="utf-8"))
    contract = surface_region_contract_from_metadata(descriptor_report["metadata"])
    if tuple(contract.radii_m) != tuple(positive.scale_radii_m):
        raise ValueError("descriptor region radii and O0 scale radii differ")
    graph, _, _ = load_torch_mapping(
        graph_path,
        expected_sha256=args.expected_support_graph_sha256,
        map_location="cpu",
        label="support graph",
    )
    global_rows = torch.as_tensor(graph["global_rows"]).long().cpu()
    graph_xyz = torch.as_tensor(graph["xyz"]).float().cpu()
    full_xyz = torch.as_tensor(positive_payload["xyz"]).float().cpu()
    if not torch.equal(graph_xyz, full_xyz[global_rows]):
        raise ValueError("support graph and O0 geometry differ")
    if not torch.equal(
        positive.valid,
        torch.zeros_like(positive.valid).index_fill_(0, global_rows, True),
    ):
        raise ValueError("support graph rows and O0 valid authority differ")
    support = PrimitiveSupportGraph(
        edge_index=graph["edge_index"],
        edge_weight=graph["edge_weight"],
        raw_affinity=graph["raw_affinity"],
        local_sigma=graph["local_sigma"],
        num_nodes=len(graph_xyz),
        edge_channels=graph.get("edge_channels", {}),
    )
    region_rows, core_mask, anchor_global, scale_index = _surface_region_rows(
        contract=contract,
        support=support,
        xyz=graph_xyz,
        global_rows=global_rows,
        batch_size=args.region_batch_size,
    )
    region_probability = _scale_major_anchor_probability(
        remapped_probability,
        global_rows,
    )
    if region_probability.shape[0] != region_rows.shape[0]:
        raise RuntimeError("region probability and immutable candidate axis differ")
    union_config = MultiRegionUnionConfig(
        score_threshold=0.6,
        maximum_regions=8,
        candidate_chunk_rows=args.selector_chunk_rows,
    )
    if args.selector_mode == "connected_v2":
        local_edges = torch.as_tensor(graph["edge_index"]).long().cpu()
        global_edges = global_rows[local_edges]
        union = greedy_connected_expected_mass_union_readout(
            region_probability,
            region_rows,
            core_mask,
            global_edges,
            num_primitives=int(full_xyz.shape[0]),
            config=union_config,
        )
    else:
        union = greedy_novelty_union_readout(
            region_probability,
            region_rows,
            core_mask,
            num_primitives=int(full_xyz.shape[0]),
            config=union_config,
        )
    schema = SCHEMAS[args.selector_mode]

    cache_payload = {
        "schema": schema,
        "query_scores": union.primitive_membership.contiguous(),
        "valid": positive.valid.contiguous(),
        "xyz": full_xyz.contiguous(),
        "metadata": {
            "query_names": list(query_ids),
            "score_semantics": "binary_semantic_core_union_membership",
            "query_consumption": "inference_interface_only",
            "target_blind": True,
            "source_score": "O0_canonical_negative_probability_after_independent_per_scale_frozen_VALA_knn_and_minmax",
            "scale_alignment": "scale_major_candidate_uses_the_same_scale_probability_and_semantic_core",
            "peak_scale_selection_before_union": False,
            "selection_rule": args.selector_mode,
            "preregistration": file_record(preregistration),
            "scale_alignment_addendum": file_record(scale_addendum),
            "selector_source": file_record(selector_source),
            "positive_cache": file_record(positive_path),
            "negative_cache": file_record(negative_path),
            "descriptor_report": file_record(descriptor_report_path),
            "support_graph": file_record(graph_path),
        },
        "selection": {
            "region_indices": union.selected_region_indices,
            "region_scores": union.selected_region_scores,
            "marginal_core_rows": union.selected_marginal_core_rows,
            "anchor_global": anchor_global,
            "scale_index": scale_index,
        },
    }
    write_torch_noclobber(output, cache_payload)
    selected_counts = [len(value) for value in union.selected_region_indices]
    selected_scales = [
        [int(scale_index[index]) for index in indices]
        for indices in union.selected_region_indices
    ]
    report = {
        "schema": schema,
        "scene": "figurines",
        "target_blind": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "query_ids": list(query_ids),
        "candidate_regions": int(region_rows.shape[0]),
        "selected_region_count": selected_counts,
        "selected_scale_indices": selected_scales,
        "selected_region_scores": [
            list(value) for value in union.selected_region_scores
        ],
        "selected_marginal_core_rows": [
            list(value) for value in union.selected_marginal_core_rows
        ],
        "selected_primitive_count": [
            int(union.primitive_membership[:, query].sum())
            for query in range(len(query_ids))
        ],
        "constants": {
            "score_gate": 0.6,
            "maximum_regions": 8,
            "selection_utility": (
                "first_probability_times_core_count_then_connected_probability_times_novel_count"
                if args.selector_mode == "connected_v2"
                else "probability_times_uncovered_core_fraction"
            ),
            "tie_break": "smaller_immutable_candidate_index",
            "support_edge_connection_required_after_seed": (
                args.selector_mode == "connected_v2"
            ),
        },
        "artifacts": {
            "cache": file_record(output),
            "preregistration": file_record(preregistration),
            "scale_alignment_addendum": file_record(scale_addendum),
            "selector_source": file_record(selector_source),
            "producer": file_record(Path(__file__).resolve()),
        },
    }
    write_frozen_json(report_path, report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--scale-alignment-addendum", required=True)
    parser.add_argument("--expected-scale-alignment-addendum-sha256", required=True)
    parser.add_argument("--selector-source", required=True)
    parser.add_argument("--expected-selector-source-sha256", required=True)
    parser.add_argument("--descriptor-report", required=True)
    parser.add_argument("--expected-descriptor-report-sha256", required=True)
    parser.add_argument("--positive-cache", required=True)
    parser.add_argument("--expected-positive-cache-sha256", required=True)
    parser.add_argument("--negative-cache", required=True)
    parser.add_argument("--expected-negative-cache-sha256", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--expected-support-graph-sha256", required=True)
    parser.add_argument("--expected-renderer-geometry-checkpoint-sha256", required=True)
    parser.add_argument("--region-batch-size", type=int, default=512)
    parser.add_argument("--knn-chunk-size", type=int, default=65536)
    parser.add_argument("--selector-chunk-rows", type=int, default=4096)
    parser.add_argument(
        "--selector-mode",
        choices=sorted(SCHEMAS),
        default="novelty_v1",
    )
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output-report", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
