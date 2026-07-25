#!/usr/bin/env python3
"""Audit a hashed and native-capability support graph without evaluator data.

The compact graph hashes official DINO/SAM rows before its local cosine
affinities are computed.  The high-fidelity graph keeps those same frozen rows
at native dimensionality.  This utility verifies that the two graphs use an
identical primitive topology, then measures the distortion introduced by the
hash on every fixed edge.  It deliberately opens neither an AGILE object list
nor a PLY label property, query, click, prediction, or metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


_CAPABILITY_CHANNELS = ("appearance", "boundary")


def _load_graph(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    required = {"schema_version", "global_rows", "num_global_rows", "edge_index", "edge_channels", "metadata"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"unsupported canonical support graph: {path}")
    if int(payload["schema_version"]) != 1:
        raise ValueError(f"unsupported canonical support graph schema: {path}")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError(f"canonical support graph metadata is invalid: {path}")
    channels = payload["edge_channels"]
    if not isinstance(channels, Mapping):
        raise ValueError(f"canonical support graph edge channels are invalid: {path}")
    edge_index = torch.as_tensor(payload["edge_index"]).long().cpu()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"canonical support graph edges are invalid: {path}")
    return {
        "path": str(Path(path).resolve()),
        "global_rows": torch.as_tensor(payload["global_rows"]).long().cpu(),
        "num_global_rows": int(payload["num_global_rows"]),
        "edge_index": edge_index,
        "edge_channels": {
            str(name): torch.as_tensor(values).float().cpu().reshape(-1)
            for name, values in channels.items()
        },
        "metadata": dict(metadata),
    }


def _graph_affinity_mode(graph: Mapping[str, Any]) -> str:
    affinity = dict(graph["metadata"].get("capability_affinity", {}))
    mode = str(affinity.get("mode", ""))
    if mode not in {"signed_hash", "exact_official_capability"}:
        raise ValueError("support graph lacks a recognized capability-affinity audit")
    return mode


def _channel_values(graph: Mapping[str, Any], name: str) -> np.ndarray:
    values = graph["edge_channels"].get(name)
    if values is None:
        raise ValueError(f"support graph lacks {name!r} affinity channel")
    values = torch.as_tensor(values).float().cpu().numpy()
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"support graph has invalid {name!r} affinities")
    return values


def _sampled_outgoing_top1_agreement(
    rows: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    maximum_nodes: int = 8192,
) -> float:
    """Compare the strongest outgoing edge on a fixed deterministic node sample."""

    if rows.ndim != 1 or rows.shape != baseline.shape or rows.shape != candidate.shape:
        raise ValueError("outgoing affinity arrays do not align")
    order = np.argsort(rows, kind="stable")
    ordered_rows = rows[order]
    nodes, starts = np.unique(ordered_rows, return_index=True)
    stops = np.r_[starts[1:], len(order)]
    if len(nodes) > int(maximum_nodes):
        positions = np.linspace(0, len(nodes) - 1, int(maximum_nodes)).round().astype(np.int64)
        starts = starts[positions]
        stops = stops[positions]
    agrees = 0
    for start, stop in zip(starts.tolist(), stops.tolist()):
        indices = order[start:stop]
        # ``argmax`` is deterministic for exact ties and both vectors share
        # the same topology/order, so this is an auditable rank diagnostic.
        agrees += int(indices[int(np.argmax(baseline[indices]))] == indices[int(np.argmax(candidate[indices]))])
    return float(agrees / max(1, len(starts)))


def _channel_report(
    baseline: np.ndarray,
    candidate: np.ndarray,
    rows: np.ndarray,
) -> dict[str, float]:
    if baseline.shape != candidate.shape:
        raise ValueError("hashed/native affinity values do not align")
    delta = np.abs(candidate - baseline)
    if len(baseline) > 1 and float(np.std(baseline)) > 0 and float(np.std(candidate)) > 0:
        pearson = float(np.corrcoef(baseline, candidate)[0, 1])
    else:
        pearson = 1.0
    return {
        "edge_count": int(len(baseline)),
        "hashed_mean": float(baseline.mean()),
        "native_mean": float(candidate.mean()),
        "mean_absolute_delta": float(delta.mean()),
        "p50_absolute_delta": float(np.quantile(delta, 0.50)),
        "p95_absolute_delta": float(np.quantile(delta, 0.95)),
        "max_absolute_delta": float(delta.max(initial=0.0)),
        "edge_affinity_pearson": pearson,
        "sampled_outgoing_top1_agreement": _sampled_outgoing_top1_agreement(
            rows, baseline, candidate
        ),
    }


def audit(
    hashed_graph_path: str | Path,
    native_graph_path: str | Path,
) -> dict[str, Any]:
    """Compare exactly aligned graph affinity channels label-free."""

    hashed = _load_graph(hashed_graph_path)
    native = _load_graph(native_graph_path)
    if _graph_affinity_mode(hashed) != "signed_hash":
        raise ValueError("baseline graph must declare signed_hash capability affinity")
    if _graph_affinity_mode(native) != "exact_official_capability":
        raise ValueError("candidate graph must declare exact_official_capability")
    if int(hashed["num_global_rows"]) != int(native["num_global_rows"]) or not torch.equal(
        hashed["global_rows"], native["global_rows"]
    ):
        raise ValueError("hashed/native graphs do not share a capability-valid primitive domain")
    if not torch.equal(hashed["edge_index"], native["edge_index"]):
        raise ValueError("hashed/native graphs do not share identical topology")
    rows = hashed["edge_index"][0].numpy()
    reports = {
        name: _channel_report(
            _channel_values(hashed, name), _channel_values(native, name), rows
        )
        for name in _CAPABILITY_CHANNELS
    }
    return {
        "mode": "label_free_canonical_capability_affinity_audit",
        "hashed_graph": hashed["path"],
        "native_graph": native["path"],
        "topology": {
            "identical": True,
            "capability_valid_nodes": int(len(hashed["global_rows"])),
            "global_primitive_rows": int(hashed["num_global_rows"]),
            "directed_edges": int(hashed["edge_index"].shape[1]),
        },
        "capability_affinity": {
            "baseline_mode": "signed_hash",
            "candidate_mode": "exact_official_capability",
            "channels": reports,
        },
        "query_independent": True,
        "labels_opened": False,
        "object_list_opened": False,
        "queries_opened": False,
        "clicks_opened": False,
        "metrics_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hashed-graph", required=True)
    parser.add_argument("--native-graph", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(args.hashed_graph, args.native_graph)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
