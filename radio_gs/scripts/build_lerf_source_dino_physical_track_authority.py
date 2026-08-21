#!/usr/bin/env python3
"""Build source-only physical-track proxy from DINO mask transport cycles.

Unlike the legacy geometry-cycle proxy, labels here never use Gaussian overlap,
depth, camera geometry, or any feature later supplied to the association
calibrator.  Official query-free SAM proposals are transported by frozen RADIO
DINOv3 dense correspondence across an adjacent three-view cycle.  Affirmative
same-instance labels require the cycle to return to its seed proposal.
Disjoint proposals in a view with affirmative transported support are known
different; missing transport is occlusion/unknown, while transported foreground
with zero overlap against every proposal is an explicit visible-null outcome.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.models.radio_adaptors import (
    load_radio_adaptor_from_checkpoint,
    project_feature_map_with_adaptor,
)
from radio_gs.models.sam3_multiscale_hierarchy import unpack_masks
from radio_gs.scripts.eval_lerf_sam_dino_tasks import filter_matches_by_ransac
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_source_dino_physical_track_authority.v1"


def dense_mutual_ransac_map(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    cycle_max_distance: float = 1.5,
    ransac_threshold: float = 1.5,
) -> torch.Tensor:
    """Return source-token -> target-token indices after mutual/RANSAC filtering."""

    if source.shape != target.shape or source.ndim != 3:
        raise ValueError("dense feature maps must share [C,H,W]")
    channels, height, width = source.shape
    src = F.normalize(source.float().reshape(channels, -1).T, dim=-1)
    tgt = F.normalize(target.float().reshape(channels, -1).T, dim=-1)
    forward = []
    for chunk in src.split(512):
        forward.append((chunk @ tgt.T).argmax(dim=1))
    forward = torch.cat(forward)
    reverse = []
    for chunk in tgt.split(512):
        reverse.append((chunk @ src.T).argmax(dim=1))
    reverse = torch.cat(reverse)
    source_index = torch.arange(src.shape[0], device=src.device)
    returned = reverse[forward]
    sy, sx = torch.div(source_index, width, rounding_mode="floor"), source_index % width
    ry, rx = torch.div(returned, width, rounding_mode="floor"), returned % width
    mutual = torch.sqrt((sy - ry).float().square() + (sx - rx).float().square()) <= cycle_max_distance
    selected = torch.where(mutual)[0]
    matches = []
    for index in selected.detach().cpu().tolist():
        target_index = int(forward[index])
        ty, tx = divmod(target_index, width)
        matches.append({"src_y": index // width, "src_x": index % width,
                        "tgt_y": ty, "tgt_x": tx, "score": 1.0})
    matches = filter_matches_by_ransac(
        matches, model="fundamental", reproj_threshold=ransac_threshold, min_inliers=8
    )
    mapping = torch.full((height * width,), -1, dtype=torch.long)
    for match in matches:
        left = int(match["src_y"]) * width + int(match["src_x"])
        right = int(match["tgt_y"]) * width + int(match["tgt_x"])
        mapping[left] = right
    return mapping


def transport_mask(mask: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Forward-splat a binary mask through an abstaining token map."""

    flat = torch.as_tensor(mask).bool().flatten()
    if flat.shape != mapping.shape:
        raise ValueError("mask/mapping token axes differ")
    valid = flat & (mapping >= 0)
    output = torch.zeros_like(flat)
    if bool(valid.any()):
        output[mapping[valid]] = True
    return output.reshape(mask.shape)


def best_proposal_by_minimum_overlap(
    transported: torch.Tensor, candidates: torch.Tensor
) -> tuple[int, float, torch.Tensor]:
    """Match by symmetric minimum-overlap; zero support remains explicit."""

    if candidates.ndim != 3:
        raise ValueError("candidate masks must be [P,H,W]")
    intersection = (candidates.bool() & transported.bool()).flatten(1).sum(1)
    denominator = torch.minimum(
        candidates.flatten(1).sum(1), transported.sum().expand(candidates.shape[0])
    ).clamp_min(1)
    overlap = intersection.float() / denominator.float()
    if not overlap.numel():
        return -1, 0.0, intersection
    index = int(overlap.argmax())
    return index, float(overlap[index]), intersection


def _load_scene(args: argparse.Namespace) -> tuple[list[int], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    proposal_dir = Path(args.proposal_dir).resolve()
    manifest = json.loads((proposal_dir / "manifest.json").read_text())
    frame_ids = [int(item["image_id"].split("_")[-1]) for item in manifest["images"]]
    frame_masks: list[torch.Tensor] = []
    quality_parts, area_parts, view_parts = [], [], []
    for view, frame_id in enumerate(frame_ids):
        payload = torch.load(proposal_dir / f"frame_{frame_id:05d}.pt", map_location="cpu", weights_only=False)
        height, width = (int(value) for value in payload["mask_shape"])
        masks = torch.from_numpy(unpack_masks(torch.as_tensor(payload["packed_masks"]), width=width)).bool()
        if tuple(masks.shape[-2:]) != (height, width):
            raise ValueError("unpacked proposal shape differs")
        frame_masks.append(masks)
        quality_parts.append(torch.as_tensor(payload["quality"]).float())
        area_parts.append(torch.as_tensor(payload["proposal_area_fraction"]).float())
        view_parts.append(torch.full((masks.shape[0],), view, dtype=torch.long))
    return frame_ids, frame_masks, torch.cat(quality_parts), torch.cat(area_parts), torch.cat(view_parts)


def _project_feature(path: Path, adaptor: torch.nn.Module, device: torch.device) -> torch.Tensor:
    feature = torch.load(path, map_location="cpu", weights_only=False).float()
    if feature.ndim == 3:
        feature = feature.unsqueeze(0)
    with torch.no_grad():
        projected = project_feature_map_with_adaptor(feature.to(device), adaptor).squeeze(0)
    return projected.detach()


def build(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"authority exists: {output}")
    frame_ids, full_masks, quality, areas, views = _load_scene(args)
    proposal_offsets = np.cumsum([0] + [int(m.shape[0]) for m in full_masks]).tolist()
    count = int(proposal_offsets[-1])
    labels = torch.full((count, count), -1, dtype=torch.int8)
    outcome = torch.full((count, len(frame_ids)), -1, dtype=torch.int8)
    matched = torch.full((count, len(frame_ids)), -1, dtype=torch.long)
    device = torch.device(args.device)
    adaptor = load_radio_adaptor_from_checkpoint(args.radio_checkpoint, "dino_v3", kind="feature_projection").to(device).eval()
    adaptor.requires_grad_(False)
    feature_dir = Path(args.feature_dir).resolve()
    feature_cache: dict[int, torch.Tensor] = {}
    mask_cache: dict[int, torch.Tensor] = {}

    def feature(view: int) -> torch.Tensor:
        if view not in feature_cache:
            feature_cache[view] = _project_feature(
                feature_dir / "backbone" / f"rgb_{frame_ids[view]}.pt", adaptor, device
            )
        return feature_cache[view]

    def masks(view: int) -> torch.Tensor:
        if view not in mask_cache:
            height, width = feature(view).shape[-2:]
            resized = [cv2.resize(value.numpy().astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                       for value in full_masks[view]]
            mask_cache[view] = torch.from_numpy(np.stack(resized)).bool()
        return mask_cache[view]

    same_triplets = 0
    visible_null = 0
    for middle in range(1, len(frame_ids) - 1):
        left, right = middle - 1, middle + 1
        maps = {
            (middle, left): dense_mutual_ransac_map(feature(middle), feature(left)),
            (left, right): dense_mutual_ransac_map(feature(left), feature(right)),
            (right, middle): dense_mutual_ransac_map(feature(right), feature(middle)),
        }
        for local_middle, seed_mask in enumerate(masks(middle)):
            global_middle = proposal_offsets[middle] + local_middle
            transported_left = transport_mask(seed_mask, maps[(middle, left)].cpu())
            if not bool(transported_left.any()):
                continue
            local_left, score_left, intersections_left = best_proposal_by_minimum_overlap(transported_left, masks(left))
            if int((intersections_left > 0).sum()) == 0:
                outcome[global_middle, left] = 0; visible_null += 1
                continue
            if score_left < 0.5:
                continue
            global_left = proposal_offsets[left] + local_left
            transported_right = transport_mask(masks(left)[local_left], maps[(left, right)].cpu())
            if not bool(transported_right.any()):
                continue
            local_right, score_right, intersections_right = best_proposal_by_minimum_overlap(transported_right, masks(right))
            if int((intersections_right > 0).sum()) == 0:
                outcome[global_left, right] = 0; visible_null += 1
                continue
            if score_right < 0.5:
                continue
            global_right = proposal_offsets[right] + local_right
            transported_return = transport_mask(masks(right)[local_right], maps[(right, middle)].cpu())
            if not bool(transported_return.any()):
                continue
            returned, return_score, intersections_return = best_proposal_by_minimum_overlap(transported_return, masks(middle))
            if return_score < 0.5 or returned != local_middle:
                continue
            same_triplets += 1
            cycle = (global_middle, global_left, global_right)
            for first, second in ((cycle[0], cycle[1]), (cycle[1], cycle[2]), (cycle[2], cycle[0])):
                labels[first, second] = labels[second, first] = 1
            outcome[global_middle, left] = outcome[global_left, right] = outcome[global_right, middle] = 1
            matched[global_middle, left] = global_left
            matched[global_left, right] = global_right
            matched[global_right, middle] = global_middle
            # Zero intersection with an affirmatively transported object is
            # signed different. Partial overlap is a granularity conflict.
            for target_view, source_global, intersection in (
                (left, global_middle, intersections_left),
                (right, global_left, intersections_right),
                (middle, global_right, intersections_return),
            ):
                for local_other in torch.where(intersection == 0)[0].tolist():
                    other = proposal_offsets[target_view] + int(local_other)
                    if labels[source_global, other] != 1:
                        labels[source_global, other] = labels[other, source_global] = 0

    lefts, rights = torch.triu_indices(count, count, offset=1)
    cross = views[lefts] != views[rights]
    lefts, rights = lefts[cross], rights[cross]
    edge_label = labels[lefts, rights]
    teacher = torch.load(Path(args.teacher), map_location="cpu", weights_only=False)
    descriptors = F.normalize(torch.as_tensor(teacher["descriptors"]).float(), dim=-1)
    contexts = F.normalize(torch.as_tensor(teacher["context_descriptors"]).float(), dim=-1)
    edge_features = torch.stack([
        (descriptors[lefts] * descriptors[rights]).sum(-1),
        (contexts[lefts] * contexts[rights]).sum(-1),
        -torch.abs(torch.log2(areas[lefts].clamp_min(1e-8) / areas[rights].clamp_min(1e-8))),
        torch.minimum(quality[lefts], quality[rights]),
    ], dim=1)
    payload = {
        "schema": SCHEMA, "schema_version": 1, "scene": args.scene,
        "edge_left": lefts, "edge_right": rights, "edge_features": edge_features,
        "edge_label": edge_label, "proposal_views": views,
        "proposal_area_fraction": areas, "view_outcome": outcome,
        "matched_proposal": matched,
        "feature_names": ["masked_descriptor_cosine", "context_descriptor_cosine", "negative_absolute_log2_area_ratio", "minimum_sam_quality"],
        "metadata": {
            "source_only": True, "benchmark_masks_opened": False, "evaluation_rgb_opened": False,
            "label_inputs": "frozen_RADIO_DINOv3_dense_mutual_fundamental_RANSAC_transport_plus_official_query_free_SAM_masks",
            "calibrator_feature_inputs": "SigLIP2_crop_and_context_descriptors_area_and_SAM_quality_no_DINO_or_geometry_overlap",
            "same": "adjacent_three_view_mask_transport_cycle_returns_to_exact_seed_with_each_minimum_overlap_ge_0.5",
            "different": "zero_mask_intersection_in_view_with_affirmative_transported_same_object_support",
            "visible_null": "nonempty_transport_with_zero_intersection_against_every_query_free_proposal",
            "occlusion_unknown": "no_transport_inlier_or_partial_overlap_or_nonclosing_cycle",
            "figurines_opened": False,
            "radio_checkpoint": {"path": str(Path(args.radio_checkpoint).resolve()), "sha256": sha256_file(args.radio_checkpoint)},
            "proposal_manifest": {"path": str((Path(args.proposal_dir).resolve() / 'manifest.json')), "sha256": sha256_file(Path(args.proposal_dir) / 'manifest.json')},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary); os.replace(temporary, output)
    report = {
        "schema": SCHEMA, "status": "complete", "scene": args.scene,
        "proposals": count, "three_view_closed_cycles": same_triplets,
        "same_edges": int((edge_label == 1).sum()), "different_edges": int((edge_label == 0).sum()),
        "unknown_edges": int((edge_label < 0).sum()), "visible_null_outcomes": visible_null,
        "figurines_opened": False, "output": str(output), "output_sha256": sha256_file(output),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--proposal-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--radio-checkpoint", default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar")
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
