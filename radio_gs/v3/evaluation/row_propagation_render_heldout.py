"""Exact-MPR source-dev render gate for frozen row propagation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.evaluation.row_propagation_heldout import _tier_report
from radio_gs.v3.training.build_multisource_correspondence_authority import _native, _pixel_support
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _top_training_observations(
    records: dict[int, dict[str, Any]], views: list[int], *, num_rows: int, top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = torch.full((num_rows, top_k), -torch.inf)
    view_ids = torch.full((num_rows, top_k), -1, dtype=torch.long)
    pixel_ids = torch.full((num_rows, top_k), -1, dtype=torch.long)
    for view in views:
        shard = torch.load(Path(records[view]["responsibility_view"]), map_location="cpu", weights_only=False)
        gaussian = torch.as_tensor(shard["gaussian_ids"]).long()
        pixel = torch.as_tensor(shard["pixel_ids"]).long()
        value = torch.as_tensor(shard["base_weights"]).float()
        local = torch.full((num_rows,), -torch.inf)
        local.scatter_reduce_(0, gaussian, value, reduce="amax", include_self=True)
        selected = value == local[gaussian]
        gaussian, pixel, value = gaussian[selected], pixel[selected], value[selected]
        order = torch.argsort(gaussian, stable=True)
        gaussian, pixel, value = gaussian[order], pixel[order], value[order]
        first = torch.ones(gaussian.numel(), dtype=torch.bool)
        first[1:] = gaussian[1:] != gaussian[:-1]
        gaussian, pixel, value = gaussian[first], pixel[first], value[first]
        candidate_weight = torch.cat((weights[gaussian], value[:, None]), dim=1)
        candidate_view = torch.cat((view_ids[gaussian], torch.full_like(pixel[:, None], view)), dim=1)
        candidate_pixel = torch.cat((pixel_ids[gaussian], pixel[:, None]), dim=1)
        new_weight, position = torch.topk(candidate_weight, top_k, dim=1)
        weights[gaussian] = new_weight
        view_ids[gaussian] = candidate_view.gather(1, position)
        pixel_ids[gaussian] = candidate_pixel.gather(1, position)
    return view_ids, pixel_ids, weights


def _render_teacher_kind(
    *, kind: str, root: Path, records: dict[int, dict[str, Any]],
    source_rows: torch.Tensor, observation_views: torch.Tensor,
    observation_pixels: torch.Tensor, observation_weights: torch.Tensor,
    hit_source_rows: torch.Tensor, hit_pixels: torch.Tensor, hit_weights: torch.Tensor,
    num_pixels: int, target_frame: int,
) -> torch.Tensor:
    dimension = 1280 if kind == "radio" else 768
    prototype = torch.zeros(source_rows.numel(), dimension)
    views = observation_views[source_rows]
    pixels = observation_pixels[source_rows]
    weights = observation_weights[source_rows]
    valid = views >= 0
    normalized = weights.masked_fill(~valid, -torch.inf).softmax(1).masked_fill(~valid, 0)
    for view in torch.unique(views[valid]).tolist():
        choose = views == view
        frame = int(records[int(view)]["frame_id"])
        feature = _native(root, "radio" if kind == "radio" else "dinov2", frame)
        row, slot = torch.where(choose)
        prototype.index_add_(0, row, feature[pixels[row, slot]] * normalized[row, slot, None])
    prototype = F.normalize(prototype, dim=1, eps=1e-8)
    position = torch.searchsorted(source_rows, hit_source_rows)
    rendered = torch.zeros(num_pixels, dimension)
    rendered.index_add_(0, hit_pixels, prototype[position] * hit_weights[:, None])
    rendered = F.normalize(rendered, dim=1, eps=1e-8)
    target = _native(root, "radio" if kind == "radio" else "dinov2", target_frame)
    return (rendered * target).sum(1)


def _bucket_report(cosine: torch.Tensor, bucket: torch.Tensor, value: int, name: str) -> dict[str, Any]:
    report = _tier_report(cosine, bucket, value)
    report["bucket"] = name
    return report


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    propagation_path = Path(args.propagation).resolve(strict=True)
    membership_path = Path(args.membership).resolve(strict=True)
    propagation = torch.load(propagation_path, map_location="cpu", weights_only=False)
    membership = torch.load(membership_path, map_location="cpu", weights_only=False)
    metadata = membership["metadata"]
    if propagation["metadata"]["inputs"]["membership"]["sha256"] != sha256_file(membership_path):
        raise ValueError("propagation and membership authorities differ")
    records = {int(value["source_view_index"]): value for value in metadata["source_records"]}
    graph_path = Path(propagation["metadata"]["inputs"]["overlap_graph"]["path"])
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    train_views = [int(value) for value in graph["metadata"]["selected_views"]]
    num_rows = int(membership["num_rows"])
    observation_views, observation_pixels, observation_weights = _top_training_observations(
        records, train_views, num_rows=num_rows, top_k=args.observations_per_row
    )
    assignments = torch.as_tensor(propagation["assignments"])
    tier_for_row = torch.full((num_rows,), -1, dtype=torch.long)
    target_rows = assignments[:, 0].long()
    tier_for_row[target_rows] = assignments[:, 2].long()
    propagation_top_k = int(propagation["metadata"].get("propagation_top_k", 1))
    mixture_source = torch.full((num_rows, propagation_top_k), -1, dtype=torch.long)
    mixture_weight = torch.zeros(num_rows, propagation_top_k)
    if "mixture_source_rows" in propagation:
        mixture_source[target_rows] = torch.as_tensor(propagation["mixture_source_rows"]).long()
        mixture_weight[target_rows] = torch.as_tensor(propagation["mixture_weights"]).float()
    else:
        mixture_source[target_rows, 0] = assignments[:, 1].long()
        mixture_weight[target_rows, 0] = 1

    all_cosine, all_bucket, view_reports = [], [], []
    minimum_unique = max(1, int(
        int(metadata["feature_height"]) * int(metadata["feature_width"])
        * args.min_top_diversity_fraction
    ))
    for view in sorted(value for value in records if value % args.residue_modulus == args.dev_residue):
        record = records[view]
        shard = torch.load(Path(record["responsibility_view"]), map_location="cpu", weights_only=False)
        support, mass = _pixel_support(shard, top_k=1)
        diversity = int(torch.unique(support[support[:, 0] >= 0, 0]).numel())
        if diversity < minimum_unique:
            view_reports.append({"view": view, "accepted": False, "unique_top_gaussians": diversity})
            continue
        gaussian = torch.as_tensor(shard["gaussian_ids"]).long()
        hit_pixels = torch.as_tensor(shard["pixel_ids"]).long()
        hit_weights = torch.as_tensor(shard["base_weights"]).float()
        hit_mixture_source = mixture_source[gaussian]
        hit_mixture_weight = mixture_weight[gaussian]
        valid_component = (
            (hit_mixture_source >= 0)
            & (observation_views[hit_mixture_source.clamp_min(0), 0] >= 0)
            & (hit_mixture_weight > 0)
        )
        valid_hit = valid_component.any(1)
        gaussian, hit_pixels, hit_weights = gaussian[valid_hit], hit_pixels[valid_hit], hit_weights[valid_hit]
        hit_mixture_source = hit_mixture_source[valid_hit]
        hit_mixture_weight = hit_mixture_weight[valid_hit]
        valid_component = valid_component[valid_hit]
        expanded_source = hit_mixture_source[valid_component]
        expanded_pixels = hit_pixels[:, None].expand_as(hit_mixture_source)[valid_component]
        expanded_weights = (
            hit_weights[:, None] * hit_mixture_weight
        )[valid_component]
        source_rows = torch.unique(expanded_source, sorted=True)
        frame = int(record["frame_id"])
        radio_cosine = _render_teacher_kind(
            kind="radio", root=Path(args.radio_root), records=records,
            source_rows=source_rows, observation_views=observation_views,
            observation_pixels=observation_pixels, observation_weights=observation_weights,
            hit_source_rows=expanded_source, hit_pixels=expanded_pixels, hit_weights=expanded_weights,
            num_pixels=int(shard["num_pixels"]), target_frame=frame,
        )
        dino_cosine = _render_teacher_kind(
            kind="dino", root=Path(args.dino_root), records=records,
            source_rows=source_rows, observation_views=observation_views,
            observation_pixels=observation_pixels, observation_weights=observation_weights,
            hit_source_rows=expanded_source, hit_pixels=expanded_pixels, hit_weights=expanded_weights,
            num_pixels=int(shard["num_pixels"]), target_frame=frame,
        )
        cosine = 0.5 * (radio_cosine + dino_cosine)
        total_mass = torch.zeros(int(shard["num_pixels"]))
        fine_mass, coarse_mass = torch.zeros_like(total_mass), torch.zeros_like(total_mass)
        total_mass.index_add_(0, hit_pixels, hit_weights)
        hit_tier = tier_for_row[gaussian]
        fine = (hit_tier == 1) | (hit_tier == 2)
        coarse = (hit_tier == 3) | (hit_tier == 4)
        fine_mass.index_add_(0, hit_pixels[fine], hit_weights[fine])
        coarse_mass.index_add_(0, hit_pixels[coarse], hit_weights[coarse])
        valid_pixel = (mass >= args.min_pixel_mass) & (total_mass >= args.min_pixel_mass)
        fine_fraction = fine_mass / total_mass.clamp_min(1e-8)
        coarse_fraction = coarse_mass / total_mass.clamp_min(1e-8)
        bucket = torch.full_like(hit_pixels[: int(shard["num_pixels"])], 3)
        bucket[(fine_fraction + coarse_fraction) <= 0.05] = 0
        bucket[(fine_fraction >= 0.20) & (coarse_fraction < 0.05)] = 1
        bucket[coarse_fraction >= 0.20] = 2
        all_cosine.append(cosine[valid_pixel])
        all_bucket.append(bucket[valid_pixel])
        view_reports.append({
            "view": view, "accepted": True, "unique_top_gaussians": diversity,
            "evaluated_pixels": int(valid_pixel.sum()), "mean_cosine": float(cosine[valid_pixel].mean()),
        })
    cosine, bucket = torch.cat(all_cosine), torch.cat(all_bucket)
    bucket_reports = [
        _bucket_report(cosine, bucket, 0, "direct_mass_95"),
        _bucket_report(cosine, bucket, 1, "fine_mass_20"),
        _bucket_report(cosine, bucket, 2, "coarse_mass_20"),
        _bucket_report(cosine, bucket, 3, "mixed"),
    ]
    direct_median, fine_median = bucket_reports[0]["median"], bucket_reports[1]["median"]
    gate_pass = bool(
        bucket_reports[1]["count"] > 0 and direct_median is not None and fine_median is not None
        and fine_median >= args.minimum_fine_median
        and fine_median >= args.minimum_direct_ratio * float(direct_median)
    )
    payload = {
        "schema": "radio_gs.sugm_v3.row_propagation_render_source_dev.v1",
        "scene": membership["scene"], "bucket_reports": bucket_reports,
        "view_reports": view_reports,
        "gate": {
            "pass": gate_pass, "fine_median": fine_median, "direct_median": direct_median,
            "minimum_fine_median": args.minimum_fine_median,
            "minimum_direct_ratio": args.minimum_direct_ratio,
        },
        "metadata": {
            "split": "source_dev_residue", "dev_residue": args.dev_residue,
            "observations_per_row": args.observations_per_row,
            "pixel_bucket_definition": {"direct": "propagated mass <= 0.05", "fine": "fine propagated mass >= 0.20 and coarse mass < 0.05", "coarse": "coarse propagated mass >= 0.20"},
            "source_only": True, "target_rgb_opened": False,
            "benchmark_metrics_opened": False, "audit_residue_opened": False,
            "inputs": {
                "propagation": {"path": str(propagation_path), "sha256": sha256_file(propagation_path)},
                "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    return {"output": str(output), "sha256": sha256_file(output), "gate": payload["gate"], "bucket_reports": bucket_reports}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--propagation", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--radio-root", required=True)
    parser.add_argument("--dino-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--observations-per-row", type=int, default=4)
    parser.add_argument("--dev-residue", type=int, default=3)
    parser.add_argument("--residue-modulus", type=int, default=4)
    parser.add_argument("--min-top-diversity-fraction", type=float, default=0.02)
    parser.add_argument("--min-pixel-mass", type=float, default=0.02)
    parser.add_argument("--minimum-fine-median", type=float, default=0.70)
    parser.add_argument("--minimum-direct-ratio", type=float, default=0.80)
    args = parser.parse_args()
    if args.observations_per_row <= 0:
        raise ValueError("observation budget must be positive")
    print(run(args))


if __name__ == "__main__":
    main()
