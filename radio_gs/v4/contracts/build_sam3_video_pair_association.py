"""Build query-free source-pair identity edges with official SAM3 video memory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.evaluation.lerf_source_mask_gate import _load_sam_records, _masks


def _candidate_pairs(frame_ids: list[int], count: int) -> list[tuple[int, int]]:
    ordered = sorted(frame_ids)
    adjacent = [(right - left, left, right) for left, right in zip(ordered, ordered[1:])]
    return [(left, right) for _, left, right in sorted(adjacent)[:count]]


def _pair_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first.bool().flatten(1).float()
    second = second.bool().flatten(1).float()
    intersection = first @ second.T
    union = first.sum(-1)[:, None] + second.sum(-1)[None] - intersection
    return intersection / union.clamp_min(1)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    from sam3.model_builder import build_sam3_video_model

    authority_path = Path(args.source_rgb_authority).resolve(strict=True)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    sam_manifests = [Path(value).resolve(strict=True) for value in args.sam_manifest]
    authority = json.loads(authority_path.read_text())
    if authority.get("information_policy", {}).get("benchmark_ground_truth_used") is not False:
        raise ValueError("source authority is not label-free")
    image_paths = {
        int(str(row["image_id"]).removeprefix("frame_")): Path(row["path"]).resolve(strict=True)
        for row in authority["images"]
    }
    sam_records = _load_sam_records(sam_manifests)
    all_pairs = _candidate_pairs(list(image_paths), args.pair_count)
    if args.direction == "reverse":
        all_pairs = [(target, source) for source, target in all_pairs]
    pairs = [pair for index, pair in enumerate(all_pairs) if index % args.shard_count == args.shard_index]

    requested = torch.device(args.device)
    if requested.type == "cuda" and requested.index is not None:
        torch.cuda.set_device(requested)
    model = build_sam3_video_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        strict_state_dict_loading=True,
        apply_temporal_disambiguation=True,
        device=args.device,
        compile=False,
    )
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    pair_records = []
    for source_frame, target_frame in pairs:
        source_cache = Path(sam_records[source_frame]["output"]).resolve(strict=True)
        target_cache = Path(sam_records[target_frame]["output"]).resolve(strict=True)
        source_payload = torch.load(source_cache, map_location="cpu")
        target_payload = torch.load(target_cache, map_location="cpu")
        source_shape = tuple(map(int, source_payload["mask_shape"]))
        target_shape = tuple(map(int, target_payload["mask_shape"]))
        if source_shape != target_shape:
            raise ValueError("video pair source mask rasters differ")
        source_masks = _masks(source_cache, *source_shape) > 0.5
        target_masks = _masks(target_cache, *target_shape) > 0.5
        source_roots = torch.where(torch.as_tensor(source_payload["parent_index"]) < 0)[0]
        target_roots = torch.where(torch.as_tensor(target_payload["parent_index"]) < 0)[0]
        quality = torch.as_tensor(source_payload["quality"])
        selected_roots = source_roots[
            torch.argsort(quality[source_roots], descending=True, stable=True)[: args.maximum_roots_per_pair]
        ]
        edges = []
        if not len(selected_roots):
            pair_records.append({
                "source_frame_id": source_frame,
                "target_frame_id": target_frame,
                "temporal_frame_gap": target_frame - source_frame,
                "source_mask_cache": str(source_cache),
                "target_mask_cache": str(target_cache),
                "source_root_count": 0,
                "target_root_count": int(len(target_roots)),
                "seeded_root_count": 0,
                "accepted_edge_count": 0,
                "edges": [],
            })
            continue
        with tempfile.TemporaryDirectory(prefix="radio_gs_v4_sam3_pair_") as temporary:
            root = Path(temporary)
            os.symlink(image_paths[source_frame], root / "00000.jpg")
            os.symlink(image_paths[target_frame], root / "00001.jpg")
            state = tracker.init_state(video_path=str(root))
            for object_id, proposal_index in enumerate(selected_roots.tolist(), start=1):
                tracker.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=object_id,
                    mask=source_masks[proposal_index],
                    add_mask_to_memory=True,
                )
            target_output = None
            for frame_index, object_ids, _low, masks, object_scores in tracker.propagate_in_video(
                state,
                start_frame_idx=0,
                max_frame_num_to_track=2,
                reverse=False,
                propagate_preflight=True,
            ):
                if int(frame_index) == 1:
                    target_output = (object_ids, masks, object_scores)
            if target_output is None:
                raise RuntimeError("SAM3 video tracker returned no target frame")
            object_ids, tracked_masks, object_scores = target_output
            tracked = torch.stack([
                tracked_masks[index].detach().float().cpu().squeeze() > 0
                for index in range(len(object_ids))
            ]) if len(object_ids) else torch.empty(0, *target_shape, dtype=torch.bool)
            if tracked.shape[-2:] != target_shape:
                tracked = torch.nn.functional.interpolate(
                    tracked[:, None].float(), size=target_shape, mode="nearest"
                )[:, 0] > 0.5
            target_root_masks = target_masks[target_roots]
            overlap = _pair_iou(tracked, target_root_masks) if len(target_roots) else torch.empty(len(tracked), 0)
            for row, object_id in enumerate(object_ids):
                source_position = int(object_id) - 1
                if not 0 <= source_position < len(selected_roots):
                    raise RuntimeError("tracked object identity escaped seeded roots")
                if overlap.shape[1]:
                    best_iou, best_local = overlap[row].max(0)
                    target_proposal = int(target_roots[int(best_local)])
                else:
                    best_iou = torch.tensor(0.0)
                    target_proposal = -1
                edges.append({
                    "source_proposal_index": int(selected_roots[source_position]),
                    "target_proposal_index": target_proposal,
                    "tracked_to_target_root_iou": float(best_iou),
                    "accepted": bool(best_iou >= args.minimum_target_iou),
                    "tracker_score": float(torch.as_tensor(object_scores[row]).detach().cpu()),
                })
            if hasattr(tracker, "reset_state"):
                tracker.reset_state(state)
            del state
        pair_records.append({
            "source_frame_id": source_frame,
            "target_frame_id": target_frame,
            "temporal_frame_gap": target_frame - source_frame,
            "source_mask_cache": str(source_cache),
            "target_mask_cache": str(target_cache),
            "source_root_count": int(len(source_roots)),
            "target_root_count": int(len(target_roots)),
            "seeded_root_count": int(len(selected_roots)),
            "accepted_edge_count": sum(int(edge["accepted"]) for edge in edges),
            "edges": edges,
        })
    del tracker, model
    report = {
        "schema": "radio_gs.surface_object_memory_v4.sam3_video_pair_association.v1",
        "source_rgb_authority": {"path": str(authority_path), "sha256": sha256_file(authority_path)},
        "sam_manifests": [{"path": str(path), "sha256": sha256_file(path)} for path in sam_manifests],
        "sam3_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "information_policy": {
            "source_rgb_used": True,
            "query_text_used": False,
            "benchmark_labels_used": False,
            "target_or_evaluation_rgb_used": False,
        },
        "pair_selection": "smallest_temporal_gap_within_sealed_source_cohort",
        "tracking_direction": args.direction,
        "minimum_target_root_iou": args.minimum_target_iou,
        "maximum_roots_per_pair": args.maximum_roots_per_pair,
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "pairs": pair_records,
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rgb-authority", required=True)
    parser.add_argument("--sam-manifest", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pair-count", type=int, default=8)
    parser.add_argument("--maximum-roots-per-pair", type=int, default=4)
    parser.add_argument("--minimum-target-iou", type=float, default=0.30)
    parser.add_argument("--direction", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid pair-association shard")
    report = run(args)
    print(json.dumps({
        "pair_count": len(report["pairs"]),
        "accepted_edges": sum(row["accepted_edge_count"] for row in report["pairs"]),
    }, indent=2))


if __name__ == "__main__":
    main()
