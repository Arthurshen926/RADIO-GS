"""Build non-circular adjacent-view correspondence authority for SUGM-v3.1."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import sha256_file
from radio_gs.v3.training.native_visual_codec import _best_pixel_per_gaussian, _load_dino


def _pixel_support(shard: dict[str, Any], *, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = torch.as_tensor(shard["pixel_ids"]).long()
    gaussians = torch.as_tensor(shard["gaussian_ids"]).long()
    weights = torch.as_tensor(shard["base_weights"]).float()
    num_pixels = int(shard["num_pixels"])
    support = torch.full((num_pixels, top_k), -1, dtype=torch.long)
    mass = torch.zeros(num_pixels)
    left = torch.searchsorted(pixels, torch.arange(num_pixels), right=False)
    right = torch.searchsorted(pixels, torch.arange(num_pixels), right=True)
    for pixel, (start, stop) in enumerate(zip(left.tolist(), right.tolist())):
        if stop <= start:
            continue
        order = torch.argsort(weights[start:stop], descending=True)[:top_k]
        support[pixel, : order.numel()] = gaussians[start:stop][order]
        mass[pixel] = weights[start:stop].sum()
    return support, mass


def _support_overlap(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (
        (left[:, :, None] == right[:, None, :])
        & (left[:, :, None] >= 0)
    ).any(-1).any(-1)


def _native(root: Path, kind: str, frame: int) -> torch.Tensor:
    if kind == "radio":
        value = torch.load(root / "backbone" / f"rgb_{frame}.pt", map_location="cpu")
    else:
        value = _load_dino(root / f"frame_{frame:05d}.pt")
    feature = torch.as_tensor(value).float().permute(1, 2, 0).reshape(-1, value.shape[0])
    return F.normalize(feature, dim=-1, eps=1e-8)


def _sample_valid(mass: torch.Tensor, budget: int) -> torch.Tensor:
    valid = torch.where(mass >= 0.02)[0]
    if valid.numel() <= budget:
        return valid
    positions = torch.linspace(0, valid.numel() - 1, budget).long()
    return valid[positions]


def _tier_rows(
    left_view: int,
    right_view: int,
    left_pixels: torch.Tensor,
    right_pixels: torch.Tensor,
    left_support: torch.Tensor,
    right_support: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    indices = torch.where(selected)[0]
    if not indices.numel():
        return torch.empty(0, 8, dtype=torch.long)
    return torch.stack((
        torch.full_like(indices, left_view), left_pixels[indices],
        torch.full_like(indices, right_view), right_pixels[indices],
        left_support[indices, 0], right_support[indices, 0],
        torch.full_like(indices, left_support.shape[1]),
        torch.full_like(indices, right_support.shape[1]),
    ), dim=-1)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    membership_path = Path(args.membership).resolve(strict=True)
    radio_root = Path(args.radio_root).resolve(strict=True)
    dino_root = Path(args.dino_root).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu")
    metadata = membership["metadata"]
    if (
        metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
        or metadata.get("evaluation_rgb_opened") is not False
        or int(metadata.get("source_view_count", -1)) != 32
    ):
        raise ValueError("correspondence authority requires sealed source32 inputs")
    rows = int(membership["num_rows"])
    device = torch.device(args.device)
    records = {int(value["source_view_index"]): value for value in metadata["source_records"]}
    adjacent = [(value, value + 1) for value in range(1, 32, 4)]
    high_rows, medium_rows, weak_rows, negative_rows, pair_reports = [], [], [], [], []
    for left_view, right_view in adjacent:
        left_record, right_record = records[left_view], records[right_view]
        left_shard = torch.load(Path(left_record["responsibility_view"]), map_location="cpu")
        right_shard = torch.load(Path(right_record["responsibility_view"]), map_location="cpu")
        left_support_full, left_mass = _pixel_support(left_shard, top_k=args.support_top_k)
        right_support_full, right_mass = _pixel_support(right_shard, top_k=args.support_top_k)
        left_pixels = _sample_valid(left_mass, args.pixels_per_view)
        right_candidates = _sample_valid(right_mass, args.pixels_per_view)
        left_dino = _native(dino_root, "dinov2", int(left_record["frame_id"]))[left_pixels]
        right_dino = _native(dino_root, "dinov2", int(right_record["frame_id"]))[right_candidates]
        dino_similarity = (left_dino.to(device) @ right_dino.to(device).T).cpu()
        forward = dino_similarity.argmax(1)
        backward = dino_similarity.argmax(0)
        left_indices = torch.arange(left_pixels.numel())
        mutual = backward[forward] == left_indices
        matched_right_pixels = right_candidates[forward]
        left_radio = _native(radio_root, "radio", int(left_record["frame_id"]))[left_pixels]
        right_radio = _native(radio_root, "radio", int(right_record["frame_id"]))[right_candidates]
        radio_similarity = (left_radio.to(device) @ right_radio.to(device).T).cpu()
        radio_forward = radio_similarity.argmax(1)
        radio_backward = radio_similarity.argmax(0)
        radio_confirms_pair = (
            (radio_forward == forward) & (radio_backward[forward] == left_indices)
        )
        left_support = left_support_full[left_pixels]
        matched_support = right_support_full[matched_right_pixels]
        overlap = _support_overlap(left_support, matched_support)
        same_top = left_support[:, 0] == matched_support[:, 0]
        high = mutual & same_top
        medium = mutual & overlap & ~same_top

        left_ids, left_best_pixels = _best_pixel_per_gaussian(left_record, rows)
        right_ids, right_best_pixels = _best_pixel_per_gaussian(right_record, rows)
        position = torch.searchsorted(right_ids, left_ids)
        valid = position < right_ids.numel()
        same = torch.zeros_like(valid)
        same[valid] = right_ids[position[valid]] == left_ids[valid]
        same_indices = torch.where(same)[0]
        weak_rows.append(torch.stack((
            torch.full_like(same_indices, left_view), left_best_pixels[same_indices],
            torch.full_like(same_indices, right_view), right_best_pixels[position[same_indices]],
            left_ids[same_indices], left_ids[same_indices],
            torch.zeros_like(same_indices), torch.zeros_like(same_indices),
        ), dim=-1))

        positive = dino_similarity[left_indices, forward]
        mutual_indices = torch.where(mutual)[0]
        disjoint = ~(
            (left_support[mutual_indices, None, :, None]
             == right_support_full[right_candidates][None, :, None, :])
            & (left_support[mutual_indices, None, :, None] >= 0)
        ).any(2).any(-1)
        negative_similarity = dino_similarity[mutual_indices].masked_fill(~disjoint, -torch.inf)
        finite_negative = torch.isfinite(negative_similarity.max(1).values)
        eligible_indices = mutual_indices[finite_negative]
        hardest = negative_similarity[finite_negative].argmax(1)
        negative_rows.append(torch.stack((
            torch.full_like(eligible_indices, left_view), left_pixels[eligible_indices],
            torch.full_like(eligible_indices, right_view), right_candidates[hardest],
            left_support[eligible_indices, 0],
            right_support_full[right_candidates[hardest], 0],
            torch.zeros_like(eligible_indices), torch.zeros_like(eligible_indices),
        ), dim=-1))
        high_rows.append(_tier_rows(
            left_view, right_view, left_pixels, matched_right_pixels,
            left_support, matched_support, high,
        ))
        medium_rows.append(_tier_rows(
            left_view, right_view, left_pixels, matched_right_pixels,
            left_support, matched_support, medium,
        ))
        pair_reports.append({
            "left_view": left_view, "right_view": right_view,
            "left_candidates": int(left_pixels.numel()),
            "right_candidates": int(right_candidates.numel()),
            "dino_mutual": int(mutual.sum()), "radio_confirmed_mutual": int((mutual & radio_confirms_pair).sum()),
            "high_same_top": int(high.sum()), "medium_support_overlap_different_top": int(medium.sum()),
            "dino_mutual_positive_similarity": float(positive[mutual].mean()) if mutual.any() else None,
            "dino_mutual_hard_negative_margin": (
                float((positive[eligible_indices] - negative_similarity[finite_negative].max(1).values).mean())
                if eligible_indices.numel() else None
            ),
        })
    high = torch.cat(high_rows)
    medium = torch.cat(medium_rows)
    weak = torch.cat(weak_rows)
    negatives = torch.cat(negative_rows)
    payload = {
        "schema": "radio_gs.sugm_v3.multisource_correspondence_authority.v1",
        "scene": membership["scene"],
        "columns": [
            "left_view", "left_pixel", "right_view", "right_pixel",
            "left_top_gaussian", "right_top_gaussian", "left_support_k", "right_support_k",
        ],
        "high_confidence_pairs": high,
        "medium_confidence_pairs": medium,
        "weak_same_gaussian_pairs": weak,
        "hard_support_disjoint_negatives": negatives,
        "metadata": {
            "adjacent_source_train_view_pairs": adjacent,
            "pixels_per_view": args.pixels_per_view,
            "support_top_k": args.support_top_k,
            "pair_reports": pair_reports,
            "counts": {
                "high": int(high.shape[0]), "medium": int(medium.shape[0]),
                "weak": int(weak.shape[0]), "hard_negative": int(negatives.shape[0]),
            },
            "tier_definition": {
                "high": "DINO bidirectional mutual + same exact-compositor top Gaussian; RADIO is reported confidence",
                "medium": "DINO bidirectional mutual + exact-compositor top-K overlap + different top Gaussian",
                "weak": "adjacent-view same-Gaussian best-pixel; not one-hot authority",
                "negative": "DINO-hard candidate with disjoint exact-compositor top-K support",
            },
            "source_only": True, "historical_field_opened": False,
            "target_rgb_opened": False, "benchmark_metrics_opened": False,
            "inputs": {
                "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
                "radio_root": str(radio_root), "dino_root": str(dino_root),
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    return {
        "output": str(output), "sha256": sha256_file(output),
        "counts": payload["metadata"]["counts"], "pair_reports": pair_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--radio-root", required=True)
    parser.add_argument("--dino-root", required=True)
    parser.add_argument("--pixels-per-view", type=int, default=512)
    parser.add_argument("--support-top-k", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.pixels_per_view <= 1 or args.support_top_k <= 0:
        raise ValueError("correspondence authority budgets are invalid")
    print(run(args))


if __name__ == "__main__":
    main()
