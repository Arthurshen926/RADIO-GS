"""Build source-only view overlap and geometry-local correspondence authority."""

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


def _pixel_geometry(
    shard: dict[str, Any], xyz: torch.Tensor, *, top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return top support, normalized weights, mass, and responsibility centroid."""
    support, mass = _pixel_support(shard, top_k=top_k)
    pixels = torch.as_tensor(shard["pixel_ids"]).long()
    gaussians = torch.as_tensor(shard["gaussian_ids"]).long()
    weights = torch.as_tensor(shard["base_weights"]).float()
    num_pixels = int(shard["num_pixels"])
    support_weights = torch.zeros(num_pixels, top_k)
    left = torch.searchsorted(pixels, torch.arange(num_pixels), right=False)
    right = torch.searchsorted(pixels, torch.arange(num_pixels), right=True)
    for pixel, (start, stop) in enumerate(zip(left.tolist(), right.tolist())):
        if stop <= start:
            continue
        order = torch.argsort(weights[start:stop], descending=True)[:top_k]
        support_weights[pixel, : order.numel()] = weights[start:stop][order]
    normalized = support_weights / support_weights.sum(1, keepdim=True).clamp_min(1e-8)
    safe_support = support.clamp_min(0)
    centroid = (xyz[safe_support] * normalized[..., None]).sum(1)
    centroid[support[:, 0] < 0] = torch.nan
    return support, normalized, mass, centroid


def _direct_support_map(
    left_support: torch.Tensor,
    left_weight: torch.Tensor,
    right_support: torch.Tensor,
    right_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map pixels through shared visible Gaussian support without dense pair matrices."""
    inverted: dict[int, list[tuple[int, float]]] = {}
    for pixel in range(right_support.shape[0]):
        for gaussian, weight in zip(right_support[pixel].tolist(), right_weight[pixel].tolist()):
            if gaussian >= 0 and weight > 0:
                inverted.setdefault(gaussian, []).append((pixel, weight))
    match = torch.full((left_support.shape[0],), -1, dtype=torch.long)
    score = torch.zeros(left_support.shape[0])
    for pixel in range(left_support.shape[0]):
        candidates: dict[int, float] = {}
        for gaussian, left_value in zip(left_support[pixel].tolist(), left_weight[pixel].tolist()):
            if gaussian < 0 or left_value <= 0:
                continue
            for right_pixel, right_value in inverted.get(gaussian, ()):
                candidates[right_pixel] = candidates.get(right_pixel, 0.0) + min(left_value, right_value)
        if candidates:
            best_pixel, best_score = max(candidates.items(), key=lambda item: item[1])
            match[pixel], score[pixel] = best_pixel, best_score
    return match, score


def _nearest_centroid(
    query: torch.Tensor, reference: torch.Tensor, *, chunk_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_query = torch.isfinite(query).all(1)
    valid_reference = torch.where(torch.isfinite(reference).all(1))[0]
    match = torch.full((query.shape[0],), -1, dtype=torch.long)
    distance = torch.full((query.shape[0],), torch.inf)
    query_ids = torch.where(valid_query)[0]
    for chunk in query_ids.split(chunk_size):
        values = torch.cdist(query[chunk], reference[valid_reference])
        best_distance, best = values.min(1)
        match[chunk] = valid_reference[best]
        distance[chunk] = best_distance
    return match, distance


def _select_graph_edges(overlap: torch.Tensor, *, neighbors: int) -> list[tuple[int, int]]:
    """Take a symmetric union of fixed top-neighbor overlap edges."""
    overlap = torch.minimum(overlap, overlap.T)
    chosen: set[tuple[int, int]] = set()
    for left in range(overlap.shape[0]):
        order = torch.argsort(overlap[left], descending=True)
        count = 0
        for right in order.tolist():
            if right == left or overlap[left, right] <= 0:
                continue
            chosen.add(tuple(sorted((left, right))))
            count += 1
            if count == neighbors:
                break
    return sorted(chosen)


def _qualify_views(
    views: list[int], geometry: dict[int, tuple[torch.Tensor, ...]], *, minimum_unique: int
) -> tuple[list[int], list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    qualified: list[int] = []
    for view in views:
        support = geometry[view][0]
        unique_top = int(torch.unique(support[support[:, 0] >= 0, 0]).numel())
        accepted = unique_top >= minimum_unique
        reports.append({
            "view": view, "unique_top_gaussians": unique_top,
            "minimum_unique_top_gaussians": minimum_unique, "accepted": accepted,
        })
        if accepted:
            qualified.append(view)
    return qualified, reports


def _window_refine(
    left_centroid: torch.Tensor,
    right_centroid: torch.Tensor,
    anchors: torch.Tensor,
    left_dino: torch.Tensor,
    right_dino: torch.Tensor,
    left_radio: torch.Tensor,
    right_radio: torch.Tensor,
    *,
    height: int,
    width: int,
    radius: int,
    scene_scale: float,
    device: torch.device,
    locked: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    left_centroid = left_centroid.to(device)
    right_centroid = right_centroid.to(device)
    anchors = anchors.to(device)
    locked = torch.zeros_like(anchors, dtype=torch.bool) if locked is None else locked.to(device)
    left_dino, right_dino = left_dino.to(device), right_dino.to(device)
    left_radio, right_radio = left_radio.to(device), right_radio.to(device)
    refined = anchors.clone()
    geometry_score = torch.zeros(anchors.shape[0], device=device)
    dino_score = torch.zeros_like(geometry_score)
    radio_score = torch.zeros_like(geometry_score)
    delta = torch.tensor([
        (dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
    ], device=device)
    for pixels in torch.arange(anchors.shape[0], device=device).split(512):
        anchor = anchors[pixels].clamp_min(0)
        ay, ax = torch.div(anchor, width, rounding_mode="floor"), anchor % width
        candidate_y = ay[:, None] + delta[:, 0]
        candidate_x = ax[:, None] + delta[:, 1]
        inside = (
            (anchors[pixels, None] >= 0) & (candidate_y >= 0) & (candidate_y < height)
            & (candidate_x >= 0) & (candidate_x < width)
            & torch.isfinite(left_centroid[pixels]).all(1, keepdim=True)
        )
        inside &= (~locked[pixels, None]) | (delta[:, 0].eq(0) & delta[:, 1].eq(0))[None]
        ids = (candidate_y.clamp(0, height - 1) * width + candidate_x.clamp(0, width - 1)).long()
        inside &= torch.isfinite(right_centroid[ids]).all(2)
        distance = torch.linalg.vector_norm(
            right_centroid[ids] - left_centroid[pixels, None], dim=2
        )
        geo = torch.exp(-distance / max(scene_scale * 0.02, 1e-8))
        dino = (right_dino[ids] * left_dino[pixels, None]).sum(2).add(1).mul(0.5).clamp(0, 1)
        radio = (right_radio[ids] * left_radio[pixels, None]).sum(2).add(1).mul(0.5).clamp(0, 1)
        objective = ((geo + dino + radio) / 3).masked_fill(~inside, -torch.inf)
        best = objective.argmax(1)
        has_candidate = inside.any(1)
        chosen = ids.gather(1, best[:, None]).squeeze(1)
        chosen_geo = geo.gather(1, best[:, None]).squeeze(1)
        chosen_dino = dino.gather(1, best[:, None]).squeeze(1)
        chosen_radio = radio.gather(1, best[:, None]).squeeze(1)
        destination = pixels[has_candidate]
        refined[destination] = chosen[has_candidate]
        geometry_score[destination] = chosen_geo[has_candidate]
        dino_score[destination] = chosen_dino[has_candidate]
        radio_score[destination] = chosen_radio[has_candidate]
    return refined.cpu(), geometry_score.cpu(), dino_score.cpu(), radio_score.cpu()


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    membership_path = Path(args.membership).resolve(strict=True)
    primitive_path = Path(args.primitive_cache).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu", weights_only=False)
    primitive = torch.load(primitive_path, map_location="cpu", weights_only=False)
    metadata = membership["metadata"]
    if any(metadata.get(key) is not False for key in (
        "benchmark_images_opened", "benchmark_masks_opened", "evaluation_rgb_opened"
    )) or int(metadata.get("source_view_count", -1)) != 32:
        raise ValueError("overlap graph requires sealed source32 inputs")
    xyz = torch.as_tensor(primitive["xyz"]).float()
    if xyz.shape != (int(membership["num_rows"]), 3):
        raise ValueError("primitive geometry does not match membership rows")
    if str(metadata.get("primitive_cache")) != str(primitive_path):
        raise ValueError("primitive cache is not the membership-bound authority")

    records = {int(value["source_view_index"]): value for value in metadata["source_records"]}
    candidate_views = sorted(v for v in records if v % args.residue_modulus in args.train_residues)
    geometry: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for view in candidate_views:
        shard = torch.load(Path(records[view]["responsibility_view"]), map_location="cpu", weights_only=False)
        geometry[view] = _pixel_geometry(shard, xyz, top_k=args.support_top_k)
    minimum_unique = max(1, int(
        int(metadata["feature_height"]) * int(metadata["feature_width"])
        * args.min_top_diversity_fraction
    ))
    selected_views, view_quality_reports = _qualify_views(
        candidate_views, geometry, minimum_unique=minimum_unique
    )
    if len(selected_views) < 2:
        raise ValueError("fewer than two source views pass the diversity authority gate")

    overlap = torch.zeros(len(selected_views), len(selected_views))
    direct_maps: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for left_index, left_view in enumerate(selected_views):
        for right_index, right_view in enumerate(selected_views):
            if left_view == right_view:
                continue
            match, score = _direct_support_map(
                geometry[left_view][0], geometry[left_view][1],
                geometry[right_view][0], geometry[right_view][1],
            )
            valid = geometry[left_view][2] >= args.min_pixel_mass
            overlap[left_index, right_index] = ((match >= 0) & valid).sum() / valid.sum().clamp_min(1)
            direct_maps[left_view, right_view] = match, score
    indexed_edges = _select_graph_edges(overlap, neighbors=args.graph_neighbors)
    edges = [(selected_views[left], selected_views[right]) for left, right in indexed_edges]

    rows: list[torch.Tensor] = []
    edge_reports: list[dict[str, Any]] = []
    scene_scale = float(torch.linalg.vector_norm(xyz.amax(0) - xyz.amin(0)))
    device = torch.device(args.device)
    native: dict[tuple[str, int], torch.Tensor] = {}
    for view in selected_views:
        frame = int(records[view]["frame_id"])
        native["dino", view] = _native(Path(args.dino_root), "dinov2", frame)
        native["radio", view] = _native(Path(args.radio_root), "radio", frame)

    for left_view, right_view in edges:
        directional: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
        for source, target in ((left_view, right_view), (right_view, left_view)):
            direct, support_score = direct_maps[source, target]
            nearest, distance = _nearest_centroid(geometry[source][3], geometry[target][3])
            anchors = torch.where(direct >= 0, direct, nearest)
            refined, geo, dino, radio = _window_refine(
                geometry[source][3], geometry[target][3], anchors,
                native["dino", source], native["dino", target],
                native["radio", source], native["radio", target],
                height=int(metadata["feature_height"]), width=int(metadata["feature_width"]),
                radius=args.local_radius, scene_scale=scene_scale, device=device,
                locked=direct >= 0,
            )
            directional[source, target] = (refined, geo, dino, radio, support_score, direct, distance)
        for source, target in ((left_view, right_view), (right_view, left_view)):
            forward = directional[source, target]
            backward = directional[target, source]
            source_ids = torch.arange(forward[0].shape[0])
            valid_match = forward[0] >= 0
            cycle_pixel = torch.full_like(source_ids, -1)
            cycle_pixel[valid_match] = backward[0][forward[0][valid_match]]
            sy, sx = torch.div(source_ids, int(metadata["feature_width"]), rounding_mode="floor"), source_ids % int(metadata["feature_width"])
            cy, cx = torch.div(cycle_pixel.clamp_min(0), int(metadata["feature_width"]), rounding_mode="floor"), cycle_pixel.clamp_min(0) % int(metadata["feature_width"])
            cycle_distance = torch.sqrt((sy - cy).float().square() + (sx - cx).float().square())
            cycle_score = torch.exp(-cycle_distance / max(float(args.local_radius), 1.0)) * valid_match
            anchor_score = torch.where(
                forward[5] >= 0, forward[4].clamp(0, 1).sqrt(), forward[1]
            )
            confidence = (anchor_score * forward[2] * forward[3] * cycle_score).clamp_min(0).pow(0.25)
            valid_source = geometry[source][2] >= args.min_pixel_mass
            keep = valid_source & valid_match
            anchor_type = (forward[5] >= 0).long()
            values = torch.stack((
                torch.full_like(source_ids, source), source_ids,
                torch.full_like(source_ids, target), forward[0],
                geometry[source][0][:, 0], geometry[target][0][forward[0].clamp_min(0), 0],
                anchor_type, cycle_pixel, forward[4], forward[1], forward[2], forward[3], cycle_score, confidence,
            ), dim=1)[keep]
            rows.append(values)
            edge_reports.append({
                "left_view": source, "right_view": target,
                "valid_source_pixels": int(valid_source.sum()),
                "direct_support_coverage": float((valid_source & (forward[5] >= 0)).sum() / valid_source.sum().clamp_min(1)),
                "geometry_local_coverage": float(keep.sum() / valid_source.sum().clamp_min(1)),
                "cycle_within_radius": float((cycle_distance[keep] <= args.local_radius).float().mean()) if keep.any() else 0.0,
                "mean_confidence": float(confidence[keep].mean()) if keep.any() else 0.0,
            })

    correspondences = torch.cat(rows) if rows else torch.empty(0, 14)
    payload = {
        "schema": "radio_gs.sugm_v3.source_overlap_graph.v1",
        "scene": membership["scene"],
        "columns": [
            "left_view", "left_pixel", "right_view", "right_pixel",
            "left_top_gaussian", "right_top_gaussian", "direct_support_anchor",
            "cycle_pixel", "shared_support_score", "geometry_score", "dino_score",
            "radio_score", "cycle_score", "confidence",
        ],
        "correspondences": correspondences,
        "view_overlap": overlap,
        "metadata": {
            "candidate_views": candidate_views, "selected_views": selected_views,
            "excluded_views": sorted(set(candidate_views) - set(selected_views)),
            "view_quality_reports": view_quality_reports,
            "min_top_diversity_fraction": args.min_top_diversity_fraction,
            "graph_edges": edges, "edge_reports": edge_reports,
            "support_top_k": args.support_top_k, "graph_neighbors": args.graph_neighbors,
            "local_radius": args.local_radius, "min_pixel_mass": args.min_pixel_mass,
            "source_only": True, "historical_field_opened": False,
            "target_rgb_opened": False, "benchmark_metrics_opened": False,
            "inputs": {
                "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
                "primitive_cache": {"path": str(primitive_path), "sha256": sha256_file(primitive_path)},
                "radio_root": str(Path(args.radio_root).resolve(strict=True)),
                "dino_root": str(Path(args.dino_root).resolve(strict=True)),
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    return {"output": str(output), "sha256": sha256_file(output), "edges": len(edges), "pairs": int(correspondences.shape[0]), "edge_reports": edge_reports}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--primitive-cache", required=True)
    parser.add_argument("--radio-root", required=True)
    parser.add_argument("--dino-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--support-top-k", type=int, default=8)
    parser.add_argument("--graph-neighbors", type=int, default=4)
    parser.add_argument("--local-radius", type=int, default=2)
    parser.add_argument("--min-pixel-mass", type=float, default=0.02)
    parser.add_argument("--min-top-diversity-fraction", type=float, default=0.02)
    parser.add_argument("--residue-modulus", type=int, default=4)
    parser.add_argument("--train-residues", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (
        args.support_top_k <= 0 or args.graph_neighbors <= 0 or args.local_radius < 0
        or not 0 <= args.min_top_diversity_fraction <= 1
    ):
        raise ValueError("authority budgets are invalid")
    print(run(args))


if __name__ == "__main__":
    main()
