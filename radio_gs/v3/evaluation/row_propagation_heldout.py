"""Source-dev fidelity gate for row propagation without D512 writeback."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.build_multisource_correspondence_authority import (
    _native,
    _pixel_support,
)
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _best_training_observation(
    records: dict[int, dict[str, Any]], views: list[int], *, num_rows: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    best_weight = torch.full((num_rows,), -torch.inf)
    best_view = torch.full((num_rows,), -1, dtype=torch.long)
    best_pixel = torch.full((num_rows,), -1, dtype=torch.long)
    for view in views:
        shard = torch.load(Path(records[view]["responsibility_view"]), map_location="cpu", weights_only=False)
        gaussian = torch.as_tensor(shard["gaussian_ids"]).long()
        pixel = torch.as_tensor(shard["pixel_ids"]).long()
        weight = torch.as_tensor(shard["base_weights"]).float()
        local_best = torch.full((num_rows,), -torch.inf)
        local_best.scatter_reduce_(0, gaussian, weight, reduce="amax", include_self=True)
        selected = weight == local_best[gaussian]
        selected_gaussian, selected_pixel, selected_weight = gaussian[selected], pixel[selected], weight[selected]
        order = torch.argsort(selected_gaussian, stable=True)
        selected_gaussian, selected_pixel, selected_weight = (
            selected_gaussian[order], selected_pixel[order], selected_weight[order]
        )
        first = torch.ones(selected_gaussian.numel(), dtype=torch.bool)
        first[1:] = selected_gaussian[1:] != selected_gaussian[:-1]
        selected_gaussian, selected_pixel, selected_weight = (
            selected_gaussian[first], selected_pixel[first], selected_weight[first]
        )
        improve = selected_weight > best_weight[selected_gaussian]
        row = selected_gaussian[improve]
        best_weight[row] = selected_weight[improve]
        best_view[row] = view
        best_pixel[row] = selected_pixel[improve]
    return best_view, best_pixel, best_weight


def _tier_report(cosine: torch.Tensor, tier: torch.Tensor, tier_value: int) -> dict[str, Any]:
    values = cosine[tier == tier_value]
    if not values.numel():
        return {"tier": tier_value, "count": 0, "mean": None, "p10": None, "median": None}
    return {
        "tier": tier_value, "count": int(values.numel()),
        "mean": float(values.mean()), "p10": float(torch.quantile(values, 0.1)),
        "median": float(values.median()),
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    propagation_path = Path(args.propagation).resolve(strict=True)
    membership_path = Path(args.membership).resolve(strict=True)
    propagation = torch.load(propagation_path, map_location="cpu", weights_only=False)
    membership = torch.load(membership_path, map_location="cpu", weights_only=False)
    metadata = membership["metadata"]
    if propagation["metadata"]["inputs"]["membership"]["sha256"] != sha256_file(membership_path):
        raise ValueError("propagation and membership authorities differ")
    if any(metadata.get(key) is not False for key in (
        "benchmark_images_opened", "benchmark_masks_opened", "evaluation_rgb_opened"
    )):
        raise ValueError("heldout propagation gate requires sealed source inputs")

    num_rows = int(membership["num_rows"])
    records = {int(value["source_view_index"]): value for value in metadata["source_records"]}
    graph_path = Path(propagation["metadata"]["inputs"]["overlap_graph"]["path"])
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    train_views = [int(value) for value in graph["metadata"]["selected_views"]]
    best_view, best_pixel, _ = _best_training_observation(records, train_views, num_rows=num_rows)

    assignments = torch.as_tensor(propagation["assignments"])
    source_for_row = torch.full((num_rows,), -1, dtype=torch.long)
    tier_for_row = torch.full((num_rows,), -1, dtype=torch.long)
    target_rows = assignments[:, 0].long()
    source_for_row[target_rows] = assignments[:, 1].long()
    tier_for_row[target_rows] = assignments[:, 2].long()
    native_cache: dict[tuple[str, int], torch.Tensor] = {}

    all_cosine: list[torch.Tensor] = []
    all_tier: list[torch.Tensor] = []
    view_reports: list[dict[str, Any]] = []
    minimum_unique = max(1, int(
        int(metadata["feature_height"]) * int(metadata["feature_width"])
        * args.min_top_diversity_fraction
    ))
    for view in sorted(value for value in records if value % args.residue_modulus == args.dev_residue):
        record = records[view]
        shard = torch.load(Path(record["responsibility_view"]), map_location="cpu", weights_only=False)
        support, mass = _pixel_support(shard, top_k=1)
        target_row = support[:, 0]
        diversity = int(torch.unique(target_row[target_row >= 0]).numel())
        if diversity < minimum_unique:
            view_reports.append({"view": view, "accepted": False, "unique_top_gaussians": diversity})
            continue
        source_row = source_for_row[target_row.clamp_min(0)]
        tier = tier_for_row[target_row.clamp_min(0)]
        valid = (target_row >= 0) & (source_row >= 0) & (best_view[source_row.clamp_min(0)] >= 0) & (mass >= args.min_pixel_mass)
        pixels = torch.where(valid)[0]
        predicted_radio = torch.zeros(pixels.numel(), 1280)
        predicted_dino = torch.zeros(pixels.numel(), 768)
        valid_source = source_row[pixels]
        for source_view in torch.unique(best_view[valid_source]).tolist():
            choose = best_view[valid_source] == source_view
            frame = int(records[int(source_view)]["frame_id"])
            for kind, root in (("radio", args.radio_root), ("dino", args.dino_root)):
                key = (kind, int(source_view))
                if key not in native_cache:
                    native_cache[key] = _native(Path(root), "radio" if kind == "radio" else "dinov2", frame)
            source_pixels = best_pixel[valid_source[choose]]
            predicted_radio[choose] = native_cache["radio", int(source_view)][source_pixels]
            predicted_dino[choose] = native_cache["dino", int(source_view)][source_pixels]
        frame = int(record["frame_id"])
        target_radio = _native(Path(args.radio_root), "radio", frame)[pixels]
        target_dino = _native(Path(args.dino_root), "dinov2", frame)[pixels]
        cosine = 0.5 * (
            (predicted_radio * target_radio).sum(1) + (predicted_dino * target_dino).sum(1)
        )
        all_cosine.append(cosine)
        all_tier.append(tier[pixels])
        view_reports.append({
            "view": view, "accepted": True, "unique_top_gaussians": diversity,
            "evaluated_pixels": int(pixels.numel()), "mean_cosine": float(cosine.mean()),
        })

    cosine = torch.cat(all_cosine)
    tier = torch.cat(all_tier)
    tier_reports = [_tier_report(cosine, tier, value) for value in range(5)]
    direct_median = tier_reports[0]["median"]
    fine_values = [row for row in tier_reports if row["tier"] in (1, 2) and row["count"]]
    fine_count = sum(int(row["count"]) for row in fine_values)
    fine_mask = (tier == 1) | (tier == 2)
    fine_median = float(cosine[fine_mask].median()) if fine_mask.any() else None
    gate_pass = bool(
        fine_count > 0 and fine_median is not None and direct_median is not None
        and fine_median >= args.minimum_fine_median
        and fine_median >= args.minimum_direct_ratio * float(direct_median)
    )
    payload = {
        "schema": "radio_gs.sugm_v3.row_propagation_source_dev.v1",
        "scene": membership["scene"], "tier_reports": tier_reports,
        "view_reports": view_reports,
        "gate": {
            "pass": gate_pass, "fine_tier_count": fine_count,
            "fine_tier_median": fine_median, "direct_tier_median": direct_median,
            "minimum_fine_median": args.minimum_fine_median,
            "minimum_direct_ratio": args.minimum_direct_ratio,
        },
        "metadata": {
            "split": "source_dev_residue", "dev_residue": args.dev_residue,
            "source_only": True, "target_rgb_opened": False,
            "benchmark_metrics_opened": False, "audit_residue_opened": False,
            "inputs": {
                "propagation": {"path": str(propagation_path), "sha256": sha256_file(propagation_path)},
                "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
                "radio_root": str(Path(args.radio_root).resolve(strict=True)),
                "dino_root": str(Path(args.dino_root).resolve(strict=True)),
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    return {"output": str(output), "sha256": sha256_file(output), "gate": payload["gate"], "tier_reports": tier_reports}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--propagation", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--radio-root", required=True)
    parser.add_argument("--dino-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dev-residue", type=int, default=3)
    parser.add_argument("--residue-modulus", type=int, default=4)
    parser.add_argument("--min-top-diversity-fraction", type=float, default=0.02)
    parser.add_argument("--min-pixel-mass", type=float, default=0.02)
    parser.add_argument("--minimum-fine-median", type=float, default=0.70)
    parser.add_argument("--minimum-direct-ratio", type=float, default=0.80)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
