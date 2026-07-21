#!/usr/bin/env python3
"""Construct a no-extra-decoder MPR-confirmation control from official masks.

For each frozen alpha-adjoint source mask, an MPR-visible primitive selects a
point in an adjacent training image.  This control retains every *already
cached* official-SAM3 target proposal that contains that exact point.  A later
true alpha-adjoint check decides whether the 3-D primitive is genuinely still
inside the target mask.  The script never links masks from loose 3-D overlap,
and it never opens labels, text, or evaluation masks.

It is intentionally a control for the stronger re-decoded official-SAM3
confirmation path.  If this cached-proposal control cannot improve the
teacher, it is not promoted; it merely tells us whether re-running the
official decoder is necessary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from radio_gs.scripts.build_sam3_automatic_mask_cache import pack_masks, unpack_masks
from radio_gs.scripts.build_sam3_mpr_confirmed_mask_cache import (
    _load_mpr_assignments,
    _load_source_masks,
    _parse_values,
    neighbouring_frames,
    select_visible_anchor_pixel,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def containing_mask_indices(masks: np.ndarray, *, x: float, y: float) -> np.ndarray:
    """Return all discrete official masks containing the nearest pixel centre."""

    values = np.asarray(masks, dtype=bool)
    if values.ndim != 3:
        raise ValueError("masks must be [M,H,W]")
    if not len(values):
        return np.empty(0, dtype=np.int64)
    px = int(np.clip(np.rint(float(x)), 0, values.shape[2] - 1))
    py = int(np.clip(np.rint(float(y)), 0, values.shape[1] - 1))
    return np.flatnonzero(values[:, py, px]).astype(np.int64)


def _automatic_mask_paths(roots: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw_root in roots:
        root = Path(raw_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"automatic-mask root does not exist: {root}")
        for path in root.glob("*.pt"):
            try:
                frame = int(path.stem)
            except ValueError as error:
                raise ValueError(f"automatic-mask stem is not a ScanNet frame: {path.name}") from error
            if frame in result:
                raise ValueError(f"duplicate automatic mask frame across roots: {frame}")
            result[frame] = path
    if not result:
        raise FileNotFoundError("automatic-mask roots contain no cache tensors")
    return result


def _validate_automatic_payload(payload: dict) -> None:
    metadata = dict(payload.get("metadata", {}))
    if not bool(metadata.get("official_decoder", False)) or not bool(metadata.get("query_free", False)):
        raise ValueError("automatic cache is not query-free official SAM3 output")
    if any(bool(metadata.get(key, False)) for key in ("labels_opened", "instances_opened", "text_opened")):
        raise ValueError("automatic cache was built with benchmark annotations")
    if metadata.get("source") != "official_sam3_interactive_grid_multimask_hierarchy":
        raise ValueError("automatic cache did not retain official SAM3 multimask hierarchy")


def build(args: argparse.Namespace) -> dict:
    source_paths = [Path(value).resolve() for value in args.membership_sidecars]
    source_masks, source_metadata, graph_path = _load_source_masks(source_paths)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    assignments, mpr_metadata, global_to_local = _load_mpr_assignments(
        Path(args.responsibility_cache).resolve(), graph=graph,
    )
    automatic_paths = _automatic_mask_paths(args.automatic_mask_roots)
    ordered_frames = [int(value) for value in mpr_metadata["selected_frame_indices"]]
    requested_source = {int(value) for value in _parse_values(args.source_frames)}
    requested_target = {int(value) for value in _parse_values(args.target_frames)}
    if requested_source:
        source_masks = [item for item in source_masks if item.frame_id in requested_source]
    if int(args.maximum_source_masks) > 0:
        source_masks = source_masks[: int(args.maximum_source_masks)]
    if not source_masks:
        raise RuntimeError("no source masks remain after deterministic selection")
    inside = float(source_metadata["inside_threshold"])
    requests: dict[int, list[tuple[object, dict]]] = defaultdict(list)
    skipped: list[dict] = []
    payload_cache: dict[int, tuple[dict, np.ndarray]] = {}
    for source in source_masks:
        for target in neighbouring_frames(source.frame_id, ordered_frames, per_direction=int(args.neighbours_per_direction)):
            if requested_target and target not in requested_target:
                continue
            if target not in automatic_paths or target not in assignments:
                skipped.append({"source": [source.frame_id, source.source_mask_index], "target": target, "reason": "target_cache_or_mpr_absent"})
                continue
            if target not in payload_cache:
                payload = torch.load(automatic_paths[target], map_location="cpu", weights_only=False)
                _validate_automatic_payload(payload)
                height, width = (int(value) for value in payload["mask_shape"])
                payload_cache[target] = (payload, unpack_masks(payload["packed_masks"], width))
            payload, masks = payload_cache[target]
            height, width = (int(value) for value in payload["mask_shape"])
            anchor = select_visible_anchor_pixel(
                source.membership,
                assignment=assignments[target],
                global_to_local=global_to_local,
                feature_height=int(mpr_metadata["feature_height"]),
                feature_width=int(mpr_metadata["feature_width"]),
                image_height=height,
                image_width=width,
                inside_threshold=inside,
            )
            if anchor is None:
                skipped.append({"source": [source.frame_id, source.source_mask_index], "target": target, "reason": "no_confident_mpr_visible_anchor"})
                continue
            target_indices = containing_mask_indices(
                masks, x=anchor["xy"][0], y=anchor["xy"][1]
            )
            if not len(target_indices):
                skipped.append({"source": [source.frame_id, source.source_mask_index], "target": target, "reason": "no_cached_official_mask_contains_mpr_anchor"})
                continue
            for mask_index in target_indices.tolist():
                requests[target].append((source, {**anchor, "target_mask_index": int(mask_index)}))
    if not requests:
        raise RuntimeError("no cached official SAM3 target proposals contain an MPR-visible anchor")

    output_root = Path(args.output_root).resolve(); output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for target in sorted(requests):
        output = output_root / f"{target}.pt"
        if output.exists() and args.skip_existing:
            continue
        payload, masks = payload_cache[target]
        rows = requests[target]
        selected = np.stack([masks[item[1]["target_mask_index"]] for item in rows])
        selected_indices = torch.tensor([item[1]["target_mask_index"] for item in rows], dtype=torch.long)
        source = [item[0] for item in rows]
        boxes = []
        for mask in selected:
            ys, xs = np.where(mask)
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
        target_quality = torch.as_tensor(payload["scores"]).float()[selected_indices]
        target_stability = torch.as_tensor(payload["stability"]).float()[selected_indices]
        output_payload = {
            "packed_masks": pack_masks(selected),
            "mask_shape": list(payload["mask_shape"]),
            "scores": target_quality.cpu(),
            "stability": target_stability.cpu(),
            "seed_xy": torch.tensor([item[1]["xy"] for item in rows], dtype=torch.float32),
            "prompt_index": torch.arange(len(rows), dtype=torch.int32),
            "candidate_index": selected_indices.to(torch.int32),
            "boxes_xyxy": torch.tensor(boxes, dtype=torch.int32),
            "proposal_area_fraction": torch.tensor([float(mask.mean()) for mask in selected], dtype=torch.float32),
            "source_frame": torch.tensor([item.frame_id for item in source], dtype=torch.int32),
            "source_mask_index": torch.tensor([item.source_mask_index for item in source], dtype=torch.int32),
            "source_quality": torch.tensor([item.quality for item in source], dtype=torch.float32),
            "source_stability": torch.tensor([item.stability for item in source], dtype=torch.float32),
            "target_feature_xy": torch.tensor([item[1]["feature_xy"] for item in rows], dtype=torch.int16),
            "anchor_membership": torch.tensor([item[1]["source_membership"] for item in rows], dtype=torch.float32),
            "anchor_mpr_weight": torch.tensor([item[1]["mpr_weight"] for item in rows], dtype=torch.float32),
            "anchor_local_primitive": torch.tensor([item[1]["local_primitive"] for item in rows], dtype=torch.int32),
            "anchor_global_primitive": torch.tensor([item[1]["global_primitive"] for item in rows], dtype=torch.int32),
            "metadata": {
                "schema_version": 1,
                "source": "official_sam3_mpr_confirmed_cached_multimask_teacher_control",
                "official_decoder": True,
                "query_free": True,
                "labels_opened": False,
                "instances_opened": False,
                "text_opened": False,
                "not_an_inference_representation": True,
                "teacher_only": True,
                "confirmation_mode": "reuse_existing_official_automatic_mask_at_exact_mpr_anchor",
                "image": str(dict(payload["metadata"]).get("image", "")),
                "source_membership_sidecars": [str(path) for path in source_paths],
                "source_scene_graph": str(graph_path),
                "source_scene_graph_sha256": _sha256_file(graph_path),
                "source_membership_lifting": "raster_adjoint",
                "source_raster_lifting_semantics": "true_alpha_compositing_adjoint",
                "anchor_selection": "max_confident_source_membership_times_frozen_mpr_weight",
                "target_selection": "adjacent_frozen_mpr_training_views",
                "neighbours_per_direction": int(args.neighbours_per_direction),
                "inside_threshold": inside,
                "multimask_candidates_retained_before_deduplication": len(rows),
                "decoder_logits_available": bool(dict(payload["metadata"]).get("decoder_logits_available", False)),
            },
        }
        torch.save(output_payload, output)
        reports.append({"target_frame": target, "source_target_associations": len(rows), "output": str(output)})
    report = {
        "schema_version": 1,
        "source": "official_sam3_mpr_confirmed_cached_multimask_teacher_control",
        "query_free": True,
        "labels_opened": False,
        "instances_opened": False,
        "text_opened": False,
        "source_masks": len(source_masks),
        "target_frames": len(requests),
        "associations": int(sum(len(value) for value in requests.values())),
        "skipped": skipped,
        "outputs": reports,
    }
    (output_root / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership-sidecars", nargs="+", required=True)
    parser.add_argument("--automatic-mask-roots", nargs="+", required=True)
    parser.add_argument("--responsibility-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--neighbours-per-direction", type=int, default=1)
    parser.add_argument("--source-frames", default="")
    parser.add_argument("--target-frames", default="")
    parser.add_argument("--maximum-source-masks", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if int(args.neighbours_per_direction) <= 0:
        parser.error("--neighbours-per-direction must be positive")
    if int(args.maximum_source_masks) < 0:
        parser.error("--maximum-source-masks must be non-negative")
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
