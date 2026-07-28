#!/usr/bin/env python3
"""Re-decode a frozen pose-free image unary with the shared support graph.

The image encoder and fused primitive unary are intentionally not rerun.  This
utility isolates the benchmark-independent support/readout policy from query
encoding so graph changes can be audited without opening query targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.interfaces import (
    load_canonical_capability_bank,
    load_canonical_support_graph,
)
from radio_gs.querying.query_spec import QueryIntent, SelectionMode
from radio_gs.querying.support_solver import (
    PrimitiveSupportGraph,
    SupportSolverConfig,
    graph_for_query_intent,
    select_support_components,
    solve_primitive_support,
)


def decode_posefree_image_unary(
    graph: PrimitiveSupportGraph,
    unary: torch.Tensor,
    *,
    solver_config: SupportSolverConfig,
    graph_policy: str = "typed_if_available",
    channel_confidence_mode: str = "none",
    selection_mode: SelectionMode = SelectionMode.TOP_COMPONENT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return support probabilities and the intent-declared component mask."""

    values = torch.as_tensor(unary).float().reshape(-1)
    if values.shape != (graph.num_nodes,) or not bool(torch.isfinite(values).all()):
        raise ValueError("pose-free unary must be finite and graph aligned")
    query_graph = graph_for_query_intent(
        graph,
        QueryIntent.INSTANCE,
        policy=graph_policy,
        channel_confidence_mode=channel_confidence_mode,
    )
    probabilities = solve_primitive_support(
        query_graph,
        values,
        config=solver_config,
    )
    selected = select_support_components(
        query_graph,
        probabilities,
        selection_mode,
        config=solver_config,
    )
    return probabilities, selected


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    source_path = Path(args.query_cache)
    payload = torch.load(source_path, map_location="cpu")
    bank = load_canonical_capability_bank(args.capability_cache)
    graph = load_canonical_support_graph(args.support_graph, bank).to(device)
    valid = torch.as_tensor(payload["valid"]).bool().cpu()
    xyz = torch.as_tensor(payload["xyz"]).float().cpu()
    if (
        valid.shape != bank.valid.shape
        or not torch.equal(valid, bank.valid)
        or xyz.shape != bank.xyz.shape
        or not torch.allclose(xyz, bank.xyz, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("query cache does not align with canonical capability bank")
    unary_full = torch.as_tensor(payload["unary"]).float().reshape(-1)
    unary = unary_full[bank.global_rows].to(device)
    config = SupportSolverConfig(
        iterations=args.iterations,
        residual=args.residual,
        unary_temperature=args.unary_temperature,
        support_threshold=args.support_threshold,
        top_k_components=args.top_k_components,
    )
    selection_mode = SelectionMode(args.selection_mode)
    probabilities, selected = decode_posefree_image_unary(
        graph,
        unary,
        solver_config=config,
        graph_policy=args.graph_policy,
        channel_confidence_mode=args.channel_confidence_mode,
        selection_mode=selection_mode,
    )
    decoded = torch.zeros(bank.num_gaussians, dtype=torch.float16)
    decoded[bank.global_rows] = (
        probabilities * selected.to(probabilities.dtype)
    ).half().cpu()
    metadata = dict(payload.get("metadata", {}))
    metadata["support_redecode"] = {
        "source_query_cache": str(source_path.resolve()),
        "query_encoder_rerun": False,
        "target_masks_opened": False,
        "test_calibration": False,
        "query_intent": QueryIntent.INSTANCE.value,
        "selection_mode": selection_mode.value,
        "graph_policy": str(args.graph_policy),
        "channel_confidence_mode": str(args.channel_confidence_mode),
        "support_graph": str(Path(args.support_graph).resolve()),
        "solver": {
            "iterations": int(config.iterations),
            "residual": float(config.residual),
            "unary_temperature": float(config.unary_temperature),
            "support_threshold": float(config.support_threshold),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **payload,
            "features": decoded[:, None],
            "metadata": metadata,
        },
        output,
    )
    report = {
        **metadata["support_redecode"],
        "output": str(output.resolve()),
        "valid_gaussians": int(bank.valid.sum()),
        "selected_gaussians": int(selected.sum()),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--residual", type=float, default=0.30)
    parser.add_argument("--unary-temperature", type=float, default=0.10)
    parser.add_argument("--support-threshold", type=float, default=0.50)
    parser.add_argument("--top-k-components", type=int, default=3)
    parser.add_argument("--graph-policy", default="typed_if_available")
    parser.add_argument(
        "--channel-confidence-mode",
        choices=("none", "affinity_mass", "max_affinity"),
        default="none",
    )
    parser.add_argument(
        "--selection-mode",
        choices=tuple(mode.value for mode in SelectionMode),
        default=SelectionMode.TOP_COMPONENT.value,
    )
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
