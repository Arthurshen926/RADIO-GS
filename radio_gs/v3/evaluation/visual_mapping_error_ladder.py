"""Five-level source-only visual-mapping error ladder for SUGM-v3.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.training.instance_upper_bound import (
    align_masks,
    mask_boundary,
    sha256_file,
    unpack_masks,
)
from radio_gs.v3.training.native_visual_codec import _best_pixel_per_gaussian, _load_dino


def _native_frame(root: Path, kind: str, frame: int) -> torch.Tensor:
    if kind == "radio":
        value = torch.load(root / "backbone" / f"rgb_{frame}.pt", map_location="cpu")
    else:
        value = _load_dino(root / f"frame_{frame:05d}.pt")
    feature = torch.as_tensor(value).float().permute(1, 2, 0).reshape(-1, value.shape[0])
    return F.normalize(feature, dim=-1, eps=1e-8)


def _pixel_boundary(record: dict[str, Any], height: int, width: int) -> torch.Tensor:
    payload = torch.load(Path(record["mask_cache"]), map_location="cpu")
    mask_width = int(payload["mask_shape"][1])
    masks = align_masks(unpack_masks(payload["packed_masks"], mask_width), height, width)
    if masks.shape[0] != int(record["num_proposals"]):
        raise ValueError("source SAM mask count differs")
    return torch.stack([mask_boundary(value) for value in masks]).any(0).flatten()


def _observation(record: dict[str, Any], rows: int, height: int, width: int):
    ids, pixels = _best_pixel_per_gaussian(record, rows)
    shard = torch.load(Path(record["responsibility_view"]), map_location="cpu")
    footprint = torch.bincount(
        torch.as_tensor(shard["gaussian_ids"]).long(), minlength=rows
    )
    return {
        "ids": ids,
        "pixels": pixels,
        "footprint": footprint[ids],
        "boundary": _pixel_boundary(record, height, width)[pixels],
    }


def _direction_metrics(similarity: torch.Tensor, selected: torch.Tensor) -> dict[str, float]:
    indices = torch.where(selected)[0]
    if not indices.numel():
        return {}
    values = similarity[indices]
    positive = values[torch.arange(indices.numel()), indices]
    rank = 1 + (values > positive[:, None]).sum(-1)
    negative = values.clone()
    negative[torch.arange(indices.numel()), indices] = -torch.inf
    hardest = negative.max(-1).values
    return {
        "count": int(indices.numel()),
        "recall_at_1": float((rank == 1).float().mean()),
        "recall_at_5": float((rank <= min(5, similarity.shape[1])).float().mean()),
        "mrr": float(rank.float().reciprocal().mean()),
        "positive_similarity": float(positive.mean()),
        "hardest_negative_similarity": float(hardest.mean()),
        "margin": float((positive - hardest).mean()),
    }


def _symmetric_metrics(similarity: torch.Tensor) -> dict[str, float]:
    selected = torch.ones(similarity.shape[0], dtype=torch.bool)
    left = _direction_metrics(similarity, selected)
    right = _direction_metrics(similarity.T, selected)
    return {
        name: (
            left[name] if name == "count" else (left[name] + right[name]) / 2.0
        )
        for name in left
    }


def _weighted(values: list[dict[str, float]]) -> dict[str, float] | None:
    values = [value for value in values if value]
    if not values:
        return None
    total = sum(int(value["count"]) for value in values)
    return {
        name: (
            total if name == "count" else
            sum(value[name] * int(value["count"]) for value in values) / total
        )
        for name in values[0]
    }


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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
        raise ValueError("visual ladder requires sealed source32 authority")
    height, width = int(metadata["feature_height"]), int(metadata["feature_width"])
    rows = int(membership["num_rows"])
    records = [
        record for record in metadata["source_records"]
        if int(record["source_view_index"]) % 4 in (1, 2)
    ]
    observations, radio, dino = [], [], []
    for record in records:
        frame = int(record["frame_id"])
        observations.append(_observation(record, rows, height, width))
        radio.append(_native_frame(radio_root, "radio", frame))
        dino.append(_native_frame(dino_root, "dinov2", frame))

    teacher_values = {name: [] for name in ("radio", "dinov2", "radio_dino_mean")}
    bucket_values = {
        name: [] for name in (
            "sam_boundary", "sam_non_boundary", "footprint_small_1_4",
            "footprint_medium_5_11", "footprint_large_12_plus",
        )
    }
    authority = {
        "pair_count": 0, "radio_mutual_true": 0, "dino_mutual_true": 0,
        "both_teachers_mutual_true": 0, "fused_mutual_true": 0,
        "teacher_predicted_match_agreement": 0,
    }
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    left_views = [i for i, r in enumerate(records) if int(r["source_view_index"]) % 4 == 1]
    right_views = [i for i, r in enumerate(records) if int(r["source_view_index"]) % 4 == 2]
    for left_index in left_views:
        left = observations[left_index]
        for right_index in right_views:
            right = observations[right_index]
            position = torch.searchsorted(right["ids"], left["ids"])
            valid = position < right["ids"].numel()
            matched = torch.zeros_like(valid)
            matched[valid] = right["ids"][position[valid]] == left["ids"][valid]
            chosen = torch.where(matched)[0]
            if chosen.numel() > args.pairs_per_view_pair:
                chosen = chosen[
                    torch.randperm(chosen.numel(), generator=generator)[:args.pairs_per_view_pair]
                ]
            if chosen.numel() <= 1:
                continue
            right_chosen = position[chosen]
            left_pixels, right_pixels = left["pixels"][chosen], right["pixels"][right_chosen]
            radio_similarity = radio[left_index][left_pixels] @ radio[right_index][right_pixels].T
            dino_similarity = dino[left_index][left_pixels] @ dino[right_index][right_pixels].T
            fused_similarity = (radio_similarity + dino_similarity) / 2.0
            for name, similarity in (
                ("radio", radio_similarity), ("dinov2", dino_similarity),
                ("radio_dino_mean", fused_similarity),
            ):
                teacher_values[name].append(_symmetric_metrics(similarity))
            count = chosen.numel()
            diagonal = torch.arange(count)
            radio_mutual = (
                (radio_similarity.argmax(1) == diagonal)
                & (radio_similarity.argmax(0) == diagonal)
            )
            dino_mutual = (
                (dino_similarity.argmax(1) == diagonal)
                & (dino_similarity.argmax(0) == diagonal)
            )
            fused_mutual = (
                (fused_similarity.argmax(1) == diagonal)
                & (fused_similarity.argmax(0) == diagonal)
            )
            authority["pair_count"] += count
            authority["radio_mutual_true"] += int(radio_mutual.sum())
            authority["dino_mutual_true"] += int(dino_mutual.sum())
            authority["both_teachers_mutual_true"] += int((radio_mutual & dino_mutual).sum())
            authority["fused_mutual_true"] += int(fused_mutual.sum())
            authority["teacher_predicted_match_agreement"] += int(
                (radio_similarity.argmax(1) == dino_similarity.argmax(1)).sum()
            )
            boundary = left["boundary"][chosen] | right["boundary"][right_chosen]
            footprint = torch.minimum(
                left["footprint"][chosen], right["footprint"][right_chosen]
            )
            masks = {
                "sam_boundary": boundary,
                "sam_non_boundary": ~boundary,
                "footprint_small_1_4": footprint <= 4,
                "footprint_medium_5_11": (footprint >= 5) & (footprint <= 11),
                "footprint_large_12_plus": footprint >= 12,
            }
            for name, mask in masks.items():
                bucket_values[name].append(_direction_metrics(fused_similarity, mask))

    pair_count = int(authority.pop("pair_count"))
    authority_rates = {
        "same_gaussian_pairs": pair_count,
        **{name.replace("_true", "_fraction"): value / pair_count for name, value in authority.items()},
        "definition": (
            "same-Gaussian best-pixel pair agreement with independent native-teacher "
            "bidirectional nearest-neighbor decisions"
        ),
    }
    report_paths = {
        "uncompressed_radio": Path(args.uncompressed_radio).resolve(strict=True),
        "uncompressed_dino": Path(args.uncompressed_dino).resolve(strict=True),
        "codec_mpr": Path(args.codec_mpr).resolve(strict=True),
        "renderer_refinement": Path(args.renderer_refinement).resolve(strict=True),
    }
    report = {
        "schema": "radio_gs.sugm_v3.visual_mapping_error_ladder.v1",
        "scene": membership["scene"],
        "split": "source_train_pair_authority_and_train_to_dev_mapping",
        "ladder": {
            "A_native_2d_teacher_on_current_pair_authority": {
                name: _weighted(values) for name, values in teacher_values.items()
            },
            "B_current_pair_authority": authority_rates,
            "B_fused_teacher_buckets": {
                name: _weighted(values) for name, values in bucket_values.items()
            },
            "C_uncompressed_exact_mpr": {
                "radio": _load_report(report_paths["uncompressed_radio"]),
                "dinov2": _load_report(report_paths["uncompressed_dino"]),
            },
            "D_learned_codec_plus_mpr": _load_report(report_paths["codec_mpr"]),
            "E_renderer_refinement": _load_report(report_paths["renderer_refinement"]),
        },
        "source_only": True,
        "historical_field_opened": False,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": {
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "radio_root": str(radio_root), "dino_root": str(dino_root),
            "reports": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in report_paths.items()
            },
        },
    }
    write_frozen_json(Path(args.output).resolve(), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--radio-root", required=True)
    parser.add_argument("--dino-root", required=True)
    parser.add_argument("--uncompressed-radio", required=True)
    parser.add_argument("--uncompressed-dino", required=True)
    parser.add_argument("--codec-mpr", required=True)
    parser.add_argument("--renderer-refinement", required=True)
    parser.add_argument("--pairs-per-view-pair", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.pairs_per_view_pair <= 1:
        raise ValueError("pair-authority candidate count must exceed one")
    print(run(args))


if __name__ == "__main__":
    main()
