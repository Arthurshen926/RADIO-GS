#!/usr/bin/env python3
"""Build query-free cross-view SAM3 confirmations from frozen MPR anchors.

An automatic official-SAM3 mask is only a local 2-D observation.  This tool
does *not* treat a loose 3-D overlap as an object identity.  Instead it uses
the mask's already-frozen true alpha-adjoint membership to choose one actually
visible primitive in a neighbouring training view, and asks the same official
SAM3 interactive decoder to segment that target image from that positive
point.  All accepted multimask alternatives are retained, just as in the
automatic cache.

The resulting masks are an offline relation-teacher artifact only.  They are
not an end-user query path, a field representation, or an inference cache.
No labels, benchmark masks, text, or evaluation images beyond the declared
training-view RGBs are opened by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch

from radio_gs.scripts.build_sam3_automatic_mask_cache import (
    containment_aware_deduplicate,
    mask_stability,
    pack_masks,
)
from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    set_requested_cuda_device,
    sha256_file,
)


def _parse_values(raw: str) -> list[str]:
    return [value for value in str(raw).replace(",", " ").split() if value]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor(values: torch.Tensor) -> str:
    array = (
        torch.as_tensor(values)
        .detach()
        .float()
        .cpu()
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def _frame_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 2**63 - 1, value


@dataclass(frozen=True)
class SourceMask:
    """One frozen local alpha-adjoint SAM3 teacher observation."""

    frame_id: int
    source_mask_index: int
    membership: torch.Tensor
    observed: torch.Tensor
    quality: float
    stability: float


def neighbouring_frames(
    source_frame: int,
    ordered_frame_ids: Iterable[int],
    *,
    per_direction: int,
) -> list[int]:
    """Return temporal neighbours from the frozen MPR training-view order.

    This deliberately uses the MPR order rather than a spatial nearest-neighbour
    search: it is deterministic, independent of semantic evidence, and has no
    opportunity to merge visually similar but disconnected instances.
    """

    ids = [int(value) for value in ordered_frame_ids]
    if int(per_direction) <= 0:
        raise ValueError("per_direction must be positive")
    if int(source_frame) not in ids:
        raise ValueError("source frame is absent from frozen MPR frame order")
    index = ids.index(int(source_frame))
    result: list[int] = []
    for offset in range(1, int(per_direction) + 1):
        if index - offset >= 0:
            result.append(ids[index - offset])
        if index + offset < len(ids):
            result.append(ids[index + offset])
    return result


def select_target_frame_shard(
    target_frames: Iterable[int], *, shard_index: int, shard_count: int,
) -> list[int]:
    """Deterministically partition target images for independent GPU workers.

    The partition is based solely on frozen MPR frame order (here represented
    by sorted numeric IDs), never on masks, semantics, or evaluation labels.
    Taking the union of all shards is exactly the unsharded teacher cache.
    """

    if int(shard_count) <= 0:
        raise ValueError("target shard count must be positive")
    if not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("target shard index must lie in [0, target shard count)")
    ordered = sorted({int(value) for value in target_frames})
    return [frame for index, frame in enumerate(ordered) if index % int(shard_count) == int(shard_index)]


def select_visible_anchor_pixel(
    source_membership: torch.Tensor,
    *,
    assignment: dict,
    global_to_local: torch.Tensor,
    feature_height: int,
    feature_width: int,
    image_height: int,
    image_width: int,
    inside_threshold: float,
) -> dict | None:
    """Select one MPR-visible target pixel from a confident source support.

    ``assignment`` is the frozen top-1 MPR raster used only to *place* a
    positive target point.  It never lifts a mask or supplies relation labels;
    subsequent mask lifting remains the renderer's true alpha adjoint.  The
    choice is deterministic: maximum source membership times MPR pixel weight,
    then the first matching row on a tie.
    """

    membership = torch.as_tensor(source_membership).float().cpu().reshape(-1)
    primitive_ids = torch.as_tensor(assignment["gaussian_ids"]).long().cpu().reshape(-1)
    pixel_ids = torch.as_tensor(assignment["pixel_ids"]).long().cpu().reshape(-1)
    weights = torch.as_tensor(assignment["weights"]).float().cpu().reshape(-1)
    if not (primitive_ids.shape == pixel_ids.shape == weights.shape):
        raise ValueError("MPR assignment rows do not align")
    if global_to_local.ndim != 1 or global_to_local.numel() == 0:
        raise ValueError("global_to_local must be a non-empty row map")
    if primitive_ids.numel() and (
        int(primitive_ids.min()) < 0 or int(primitive_ids.max()) >= len(global_to_local)
    ):
        raise ValueError("MPR assignment primitive is outside global row map")
    if pixel_ids.numel() and (
        int(pixel_ids.min()) < 0
        or int(pixel_ids.max()) >= int(feature_height) * int(feature_width)
    ):
        raise ValueError("MPR assignment pixel is outside feature raster")
    local = global_to_local[primitive_ids]
    local_valid = local >= 0
    confident = torch.zeros_like(local_valid)
    if bool(local_valid.any()):
        local_indices = torch.where(local_valid)[0]
        confident[local_indices] = (
            membership[local[local_indices]] >= float(inside_threshold)
        )
    if not bool(confident.any()):
        return None
    score = torch.full_like(weights, float("-inf"))
    score[confident] = weights[confident] * membership[local[confident]]
    row = int(score.argmax())
    feature_pixel = int(pixel_ids[row])
    feature_x = feature_pixel % int(feature_width)
    feature_y = feature_pixel // int(feature_width)
    # Pixel centres, not corners, preserve the selected MPR raster cell under
    # the feature-to-RGB resize.  Clamp only the final floating representation.
    x = min(
        max((float(feature_x) + 0.5) * float(image_width) / float(feature_width), 0.0),
        float(image_width - 1),
    )
    y = min(
        max((float(feature_y) + 0.5) * float(image_height) / float(feature_height), 0.0),
        float(image_height - 1),
    )
    return {
        "xy": [float(x), float(y)],
        "feature_xy": [int(feature_x), int(feature_y)],
        "local_primitive": int(local[row]),
        "global_primitive": int(primitive_ids[row]),
        "source_membership": float(membership[local[row]]),
        "mpr_weight": float(weights[row]),
    }


def _load_source_masks(paths: list[Path]) -> tuple[list[SourceMask], dict, Path]:
    if not paths:
        raise ValueError("at least one alpha-adjoint membership sidecar is required")
    metadata_reference: dict | None = None
    source: list[SourceMask] = []
    seen: set[tuple[int, int]] = set()
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata", {}))
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError(f"{path} must use membership-sidecar schema v1")
        required_truths = (
            "query_free",
            "offline_teacher_audit_only",
            "not_an_inference_representation",
        )
        if not all(bool(metadata.get(key, False)) for key in required_truths):
            raise ValueError(f"{path} is not declared as query-free offline teacher evidence")
        if any(bool(metadata.get(key, False)) for key in ("labels_opened", "instances_opened", "text_opened")):
            raise ValueError(f"{path} was built with benchmark annotations")
        if metadata.get("membership_lifting") != "raster_adjoint":
            raise ValueError(f"{path} must contain true raster-adjoint memberships")
        if metadata.get("raster_lifting_semantics") != "true_alpha_compositing_adjoint":
            raise ValueError(f"{path} does not declare true alpha-adjoint semantics")
        if metadata_reference is None:
            metadata_reference = metadata
        else:
            for key in (
                "scene_graph",
                "scene_graph_sha256",
                "inside_threshold",
                "outside_threshold",
                "membership_lifting",
                "raster_lifting_semantics",
            ):
                if metadata.get(key) != metadata_reference.get(key):
                    raise ValueError(f"source sidecars differ at {key!r}")
        for record in payload.get("records", []):
            try:
                frame_id = int(Path(str(record["mask_frame"])).stem)
            except ValueError as error:
                raise ValueError(f"non-numeric ScanNet mask frame in {path}") from error
            membership = torch.as_tensor(record["membership"]).float().cpu()
            observed = torch.as_tensor(record["observed"]).bool().cpu().reshape(-1)
            source_indices = torch.as_tensor(record["source_mask_indices"]).long().cpu()
            quality = torch.as_tensor(record["quality"]).float().cpu()
            stability = torch.as_tensor(record["stability"]).float().cpu()
            if membership.ndim != 2 or source_indices.shape != (membership.shape[0],):
                raise ValueError("source membership rows do not align with source indices")
            if observed.shape != (membership.shape[1],):
                raise ValueError("source observation rows do not align with memberships")
            if quality.shape != source_indices.shape or stability.shape != source_indices.shape:
                raise ValueError("source membership quality/stability rows do not align")
            for local_index, source_index in enumerate(source_indices.tolist()):
                identity = (frame_id, int(source_index))
                if identity in seen:
                    raise ValueError(f"duplicate source SAM3 mask across sidecars: {identity}")
                seen.add(identity)
                source.append(
                    SourceMask(
                        frame_id=frame_id,
                        source_mask_index=int(source_index),
                        membership=membership[local_index],
                        observed=observed,
                        quality=float(quality[local_index]),
                        stability=float(stability[local_index]),
                    )
                )
    assert metadata_reference is not None
    graph_path = Path(str(metadata_reference.get("scene_graph", "")))
    if not graph_path.is_file() or _sha256_file(graph_path) != str(metadata_reference.get("scene_graph_sha256", "")):
        raise ValueError("source membership sidecars do not match their frozen graph")
    source.sort(key=lambda value: (value.frame_id, value.source_mask_index))
    return source, metadata_reference, graph_path


def _load_mpr_assignments(
    path: Path,
    *,
    graph: dict,
) -> tuple[dict[int, dict], dict, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    if int(payload.get("schema_version", -1)) != 1 or not isinstance(payload.get("assignments"), list):
        raise ValueError("MPR responsibility cache must use schema v1")
    if metadata.get("assignment_mode") != "raster_gaussian_top1":
        raise ValueError("cross-view point placement requires frozen MPR top-1 assignments")
    if any(bool(metadata.get(key, False)) for key in ("benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened")):
        raise ValueError("MPR responsibility cache violates query-free provenance")
    frame_ids = [int(value) for value in metadata.get("selected_frame_indices", [])]
    assignments = list(payload["assignments"])
    if not frame_ids or len(frame_ids) != len(assignments):
        raise ValueError("MPR responsibility frame IDs do not align with assignments")
    if not {"feature_height", "feature_width"}.issubset(metadata):
        raise ValueError("MPR responsibility cache lacks feature raster shape")
    global_rows = torch.as_tensor(graph.get("global_rows")).long().cpu().reshape(-1)
    num_global = int(graph.get("num_global_rows", 0))
    if global_rows.shape != (len(torch.as_tensor(graph["xyz"])),) or num_global <= 0:
        raise ValueError("relation graph lacks valid global-row identity")
    if int(global_rows.min()) < 0 or int(global_rows.max()) >= num_global:
        raise ValueError("relation graph global rows are outside MPR geometry")
    # The MPR sidecar is indexed by the checkpoint's full Gaussian geometry;
    # ``global_rows`` is the only permitted bridge to the canonical subset.
    # Refuse a spatial fallback, since it could move a positive prompt across
    # a boundary before SAM3 ever sees it.
    graph_metadata = dict(graph.get("metadata", {}))
    capability_path = Path(str(graph_metadata.get("capability_cache", "")))
    if not capability_path.is_file():
        raise ValueError("relation graph lacks the capability cache needed for MPR identity validation")
    capability = torch.load(capability_path, map_location="cpu", weights_only=False)
    full_xyz = torch.as_tensor(capability.get("xyz")).float().cpu()
    if full_xyz.shape != (num_global, 3) or metadata.get("xyz_sha256") != _sha256_tensor(full_xyz):
        raise ValueError("MPR responsibility geometry does not match canonical graph capability rows")
    if not torch.equal(torch.as_tensor(graph["xyz"]).float().cpu(), full_xyz[global_rows]):
        raise ValueError("canonical graph global rows do not match MPR capability geometry")
    global_to_local = torch.full((num_global,), -1, dtype=torch.long)
    global_to_local[global_rows] = torch.arange(len(global_rows), dtype=torch.long)
    return {frame: assignment for frame, assignment in zip(frame_ids, assignments)}, metadata, global_to_local


def _resolve_image(image_root: Path, frame_id: int) -> Path:
    candidates = [image_root / f"{int(frame_id)}{suffix}" for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".PNG")]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"cannot uniquely resolve RGB image for frame {frame_id} under {image_root}")
    return matches[0]


def _predict_target_masks(
    processor,
    image: Image.Image,
    requests: list[tuple[SourceMask, dict]],
    args: argparse.Namespace,
) -> tuple[list[dict], bool, int]:
    """Run official decoder once per MPR-confirmed positive point."""

    state = processor.set_image(image)
    rows: list[dict] = []
    logits_available = True
    for prompt_index, (source, anchor) in enumerate(requests):
        candidates, quality, low_resolution = processor.model.predict_inst(
            state,
            point_coords=np.asarray([anchor["xy"]], dtype=np.float32),
            point_labels=np.ones(1, dtype=np.int32),
            multimask_output=True,
        )
        candidates = np.asarray(candidates).astype(bool, copy=False)
        quality = np.asarray(quality, dtype=np.float32).reshape(-1)
        if candidates.ndim != 3 or candidates.shape[0] != quality.size:
            raise ValueError("official SAM3 prompted multimasks and qualities do not align")
        stability = mask_stability(low_resolution, candidates, offset=float(args.stability_offset))
        logits_available &= (
            low_resolution is not None
            and np.asarray(low_resolution).ndim == 3
            and np.asarray(low_resolution).shape[0] == candidates.shape[0]
        )
        for candidate_index, (mask, score, stable) in enumerate(zip(candidates, quality, stability)):
            fraction = float(mask.mean())
            if score < float(args.minimum_quality) or stable < float(args.minimum_stability):
                continue
            if not float(args.minimum_area_fraction) <= fraction <= float(args.maximum_area_fraction):
                continue
            rows.append(
                {
                    "mask": mask,
                    "score": float(score),
                    "stability": float(stable),
                    "seed_xy": anchor["xy"],
                    "prompt_index": int(prompt_index),
                    "candidate_index": int(candidate_index),
                    "source_frame": int(source.frame_id),
                    "source_mask_index": int(source.source_mask_index),
                    "source_quality": float(source.quality),
                    "source_stability": float(source.stability),
                    "target_feature_xy": anchor["feature_xy"],
                    "anchor_local_primitive": int(anchor["local_primitive"]),
                    "anchor_global_primitive": int(anchor["global_primitive"]),
                    "anchor_membership": float(anchor["source_membership"]),
                    "anchor_mpr_weight": float(anchor["mpr_weight"]),
                }
            )
    return _deduplicate_target_rows(rows, args), bool(logits_available), len(rows)


def _deduplicate_target_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    """Deduplicate decoder alternatives *within* one source-anchor request.

    Two source masks can legitimately trigger the same target-image mask.  In
    an ordinary 2-D proposal cache that would be redundant, but here each row
    is a different source--target correspondence and therefore contributes a
    different virtual 3-D union.  Removing it would sever precisely the
    cross-view connectivity this teacher is intended to measure.  We only
    suppress near-identical alternatives from the same official decoder call;
    different prompt/source lineages always survive the normal NMS step.
    """

    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["prompt_index"])].append(row)
    per_anchor: list[list[dict]] = []
    for prompt_index in sorted(grouped):
        candidates = grouped[prompt_index]
        kept = containment_aware_deduplicate(
            [row["mask"] for row in candidates],
            [row["score"] for row in candidates],
            iou_threshold=float(args.nms_iou),
            minimum_area_ratio=float(args.duplicate_minimum_area_ratio),
            maximum=0,
        )
        per_anchor.append([candidates[index] for index in kept])
    if int(args.maximum_masks_per_target) <= 0:
        return [row for rows_for_anchor in per_anchor for row in rows_for_anchor]

    # An explicit memory cap should not silently privilege the first source
    # anchor.  Interleave kept alternatives so every independently confirmed
    # correspondence gets one chance before any anchor receives a second.
    result: list[dict] = []
    maximum = int(args.maximum_masks_per_target)
    depth = 0
    while len(result) < maximum:
        added = False
        for rows_for_anchor in per_anchor:
            if depth < len(rows_for_anchor):
                result.append(rows_for_anchor[depth])
                added = True
                if len(result) >= maximum:
                    break
        if not added:
            break
        depth += 1
    return result


def build(args: argparse.Namespace) -> dict:
    source_paths = [Path(value).resolve() for value in args.membership_sidecars]
    source_masks, source_metadata, graph_path = _load_source_masks(source_paths)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    assignments, mpr_metadata, global_to_local = _load_mpr_assignments(
        Path(args.responsibility_cache).resolve(), graph=graph,
    )
    ordered_frame_ids = [int(value) for value in mpr_metadata["selected_frame_indices"]]
    inside_threshold = float(source_metadata["inside_threshold"])
    requested_source_frames = {int(value) for value in _parse_values(args.source_frames)}
    requested_target_frames = {int(value) for value in _parse_values(args.target_frames)}
    if requested_source_frames:
        source_masks = [row for row in source_masks if row.frame_id in requested_source_frames]
    if int(args.maximum_source_masks) > 0:
        source_masks = source_masks[: int(args.maximum_source_masks)]
    if not source_masks:
        raise RuntimeError("no source masks remain after deterministic source selection")
    image_root = Path(args.image_root).resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {image_root}")

    grouped: dict[int, list[tuple[SourceMask, dict]]] = defaultdict(list)
    skipped: list[dict] = []
    for source in source_masks:
        for target_frame in neighbouring_frames(
            source.frame_id, ordered_frame_ids, per_direction=int(args.neighbours_per_direction)
        ):
            if requested_target_frames and target_frame not in requested_target_frames:
                continue
            assignment = assignments.get(target_frame)
            if assignment is None:
                skipped.append({"source": [source.frame_id, source.source_mask_index], "target": target_frame, "reason": "target_absent_from_mpr"})
                continue
            image_path = _resolve_image(image_root, target_frame)
            with Image.open(image_path) as raw_image:
                anchor = select_visible_anchor_pixel(
                    source.membership,
                    assignment=assignment,
                    global_to_local=global_to_local,
                    feature_height=int(mpr_metadata["feature_height"]),
                    feature_width=int(mpr_metadata["feature_width"]),
                    image_height=int(raw_image.height),
                    image_width=int(raw_image.width),
                    inside_threshold=inside_threshold,
                )
            if anchor is None:
                skipped.append({"source": [source.frame_id, source.source_mask_index], "target": target_frame, "reason": "no_confident_mpr_visible_anchor"})
                continue
            grouped[target_frame].append((source, anchor))
    if not grouped:
        raise RuntimeError("no MPR-confirmed target prompts were available")
    available_target_frames = sorted(grouped)
    selected_target_frames = select_target_frame_shard(
        available_target_frames,
        shard_index=int(args.target_shard_index),
        shard_count=int(args.target_shard_count),
    )
    grouped = {frame: grouped[frame] for frame in selected_target_frames}
    if not grouped:
        raise RuntimeError("no target prompts remain in the requested deterministic target shard")

    set_requested_cuda_device(args.device)
    processor = _load_sam3_model(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        confidence_threshold=0.0,
        dtype=args.dtype,
        resolution=int(args.resolution),
        point_only=True,
    )
    checkpoint_sha = sha256_file(args.checkpoint_path)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    for target_ordinal, target_frame in enumerate(sorted(grouped), start=1):
        output = output_root / f"{target_frame}.pt"
        if output.exists() and args.skip_existing:
            print(json.dumps({
                "phase": "target", "target_frame": int(target_frame),
                "target_ordinal": target_ordinal, "target_count": len(grouped),
                "status": "skipped_existing",
            }), flush=True)
            continue
        image_path = _resolve_image(image_root, target_frame)
        image = Image.open(image_path).convert("RGB")
        rows, logits_available, candidates_after_filters = _predict_target_masks(
            processor, image, grouped[target_frame], args
        )
        if rows:
            masks = np.stack([row["mask"] for row in rows])
            boxes = []
            for mask in masks:
                ys, xs = np.where(mask)
                boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
        else:
            masks = np.empty((0, image.height, image.width), dtype=bool)
            boxes = []
        payload = {
            "packed_masks": pack_masks(masks),
            "mask_shape": [int(image.height), int(image.width)],
            "scores": torch.tensor([row["score"] for row in rows], dtype=torch.float32),
            "stability": torch.tensor([row["stability"] for row in rows], dtype=torch.float32),
            "seed_xy": torch.tensor([row["seed_xy"] for row in rows], dtype=torch.float32).reshape(-1, 2),
            "prompt_index": torch.tensor([row["prompt_index"] for row in rows], dtype=torch.int32),
            "candidate_index": torch.tensor([row["candidate_index"] for row in rows], dtype=torch.int8),
            "boxes_xyxy": torch.tensor(boxes, dtype=torch.int32).reshape(-1, 4),
            "proposal_area_fraction": torch.tensor([float(row["mask"].mean()) for row in rows], dtype=torch.float32),
            "source_frame": torch.tensor([row["source_frame"] for row in rows], dtype=torch.int32),
            "source_mask_index": torch.tensor([row["source_mask_index"] for row in rows], dtype=torch.int32),
            "source_quality": torch.tensor([row["source_quality"] for row in rows], dtype=torch.float32),
            "source_stability": torch.tensor([row["source_stability"] for row in rows], dtype=torch.float32),
            "target_feature_xy": torch.tensor([row["target_feature_xy"] for row in rows], dtype=torch.int16).reshape(-1, 2),
            "anchor_membership": torch.tensor([row["anchor_membership"] for row in rows], dtype=torch.float32),
            "anchor_mpr_weight": torch.tensor([row["anchor_mpr_weight"] for row in rows], dtype=torch.float32),
            "anchor_local_primitive": torch.tensor([row["anchor_local_primitive"] for row in rows], dtype=torch.int32),
            "anchor_global_primitive": torch.tensor([row["anchor_global_primitive"] for row in rows], dtype=torch.int32),
            "metadata": {
                "schema_version": 1,
                "source": "official_sam3_mpr_confirmed_cross_view_multimask_teacher",
                "official_decoder": True,
                "query_free": True,
                "labels_opened": False,
                "instances_opened": False,
                "text_opened": False,
                "not_an_inference_representation": True,
                "teacher_only": True,
                "image": str(image_path),
                "checkpoint_sha256": checkpoint_sha,
                "source_membership_sidecars": [str(path) for path in source_paths],
                "source_scene_graph": str(graph_path),
                "source_scene_graph_sha256": _sha256_file(graph_path),
                "source_membership_lifting": "raster_adjoint",
                "source_raster_lifting_semantics": "true_alpha_compositing_adjoint",
                "anchor_selection": "max_confident_source_membership_times_frozen_mpr_weight",
                "target_selection": "adjacent_frozen_mpr_training_views",
                "neighbours_per_direction": int(args.neighbours_per_direction),
                "inside_threshold": inside_threshold,
                "minimum_quality": float(args.minimum_quality),
                "minimum_stability": float(args.minimum_stability),
                "minimum_area_fraction": float(args.minimum_area_fraction),
                "maximum_area_fraction": float(args.maximum_area_fraction),
                "nms_iou": float(args.nms_iou),
                "duplicate_minimum_area_ratio": float(args.duplicate_minimum_area_ratio),
                "deduplication": "within_source_anchor_containment_aware_near_duplicate_only",
                "cross_anchor_target_mask_associations_retained": True,
                "multimask_candidates_retained_before_deduplication": int(candidates_after_filters),
                "decoder_logits_available": bool(logits_available),
            },
        }
        torch.save(payload, output)
        reports.append({
            "target_frame": int(target_frame),
            "target_prompts": len(grouped[target_frame]),
            "retained_masks": int(len(rows)),
            "output": str(output),
        })
        print(json.dumps({
            "phase": "target", "target_frame": int(target_frame),
            "target_ordinal": target_ordinal, "target_count": len(grouped),
            "target_prompts": len(grouped[target_frame]), "retained_masks": int(len(rows)),
            "status": "written",
        }), flush=True)
    report = {
        "schema_version": 1,
        "source": "official_sam3_mpr_confirmed_cross_view_multimask_teacher",
        "query_free": True,
        "labels_opened": False,
        "instances_opened": False,
        "text_opened": False,
        "source_masks": len(source_masks),
        "target_frames": len(grouped),
        "target_frames_available_before_sharding": len(available_target_frames),
        "target_shard_index": int(args.target_shard_index),
        "target_shard_count": int(args.target_shard_count),
        "target_prompts": int(sum(len(value) for value in grouped.values())),
        "skipped": skipped,
        "outputs": reports,
    }
    (output_root / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership-sidecars", nargs="+", required=True)
    parser.add_argument("--responsibility-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-path", default="checkpoints/sam3_modelscope/sam3.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--neighbours-per-direction", type=int, default=1)
    parser.add_argument("--source-frames", default="")
    parser.add_argument("--target-frames", default="")
    parser.add_argument("--target-shard-index", type=int, default=0)
    parser.add_argument("--target-shard-count", type=int, default=1)
    parser.add_argument("--maximum-source-masks", type=int, default=0)
    parser.add_argument("--minimum-quality", type=float, default=0.70)
    parser.add_argument("--minimum-stability", type=float, default=0.0)
    parser.add_argument("--stability-offset", type=float, default=1.0)
    parser.add_argument("--minimum-area-fraction", type=float, default=0.001)
    parser.add_argument("--maximum-area-fraction", type=float, default=0.80)
    parser.add_argument("--nms-iou", type=float, default=0.85)
    parser.add_argument("--duplicate-minimum-area-ratio", type=float, default=0.90)
    parser.add_argument("--maximum-masks-per-target", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if int(args.neighbours_per_direction) <= 0:
        parser.error("--neighbours-per-direction must be positive")
    if int(args.target_shard_count) <= 0 or not 0 <= int(args.target_shard_index) < int(args.target_shard_count):
        parser.error("target shard index must lie in [0, target shard count)")
    if int(args.maximum_source_masks) < 0 or int(args.maximum_masks_per_target) < 0:
        parser.error("maximum mask counts must be non-negative")
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
