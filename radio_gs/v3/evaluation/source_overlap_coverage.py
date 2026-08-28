"""Evaluate trustworthy union coverage of a frozen source overlap graph."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.build_multisource_correspondence_authority import _pixel_support
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _union_coverage(rows: torch.Tensor, views: list[int], *, num_pixels: int) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for view in views:
        selected = rows[rows[:, 0].long() == view]
        pixels = selected[:, 1].long()
        direct = selected[:, 6] > 0.5
        cycle = selected[:, 12] >= math.exp(-1.0)
        confident = selected[:, 13] >= 0.5

        def coverage(mask: torch.Tensor) -> float:
            return float(torch.unique(pixels[mask]).numel() / num_pixels)

        reports.append({
            "view": view,
            "direct_support_union": coverage(direct),
            "direct_cycle_union": coverage(direct & cycle),
            "any_cycle_union": coverage(cycle),
            "confidence_half_union": coverage(confident),
            "strict_union": coverage(direct & cycle & confident),
        })
    return reports


def _summary(reports: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in reports]
    return {"mean": sum(values) / len(values), "minimum": min(values), "maximum": max(values)}


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    graph_path = Path(args.graph).resolve(strict=True)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    rows = torch.as_tensor(graph["correspondences"]).float()
    metadata = graph["metadata"]
    views = [int(value) for value in metadata["selected_views"]]
    num_pixels = int(rows[:, 1].max()) + 1
    view_reports = _union_coverage(rows, views, num_pixels=num_pixels)

    membership_path = Path(metadata["inputs"]["membership"]["path"]).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu", weights_only=False)
    num_rows = int(membership["num_rows"])
    records = {int(value["source_view_index"]): value for value in membership["metadata"]["source_records"]}
    top_rows = torch.zeros(num_rows, dtype=torch.bool)
    all_rows = torch.zeros(num_rows, dtype=torch.bool)
    for view in views:
        shard = torch.load(Path(records[view]["responsibility_view"]), map_location="cpu", weights_only=False)
        support, _ = _pixel_support(shard, top_k=int(metadata["support_top_k"]))
        top_rows[support[support >= 0]] = True
        all_rows[torch.as_tensor(shard["gaussian_ids"]).long()] = True

    keys = ("direct_support_union", "direct_cycle_union", "any_cycle_union", "confidence_half_union", "strict_union")
    payload = {
        "schema": "radio_gs.sugm_v3.source_overlap_coverage.v1",
        "scene": graph["scene"],
        "view_reports": view_reports,
        "summary": {key: _summary(view_reports, key) for key in keys},
        "gaussian_row_coverage": {
            "top_support": float(top_rows.float().mean()),
            "any_exact_compositor_hit": float(all_rows.float().mean()),
        },
        "metadata": {
            "source_only": True, "benchmark_metrics_opened": False,
            "graph": {"path": str(graph_path), "sha256": sha256_file(graph_path)},
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "cycle_radius_definition": "cycle_score >= exp(-1), equivalent to configured local radius",
            "confidence_report_cut": 0.5,
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    return {"output": str(output), "sha256": sha256_file(output), "summary": payload["summary"], "gaussian_row_coverage": payload["gaussian_row_coverage"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
