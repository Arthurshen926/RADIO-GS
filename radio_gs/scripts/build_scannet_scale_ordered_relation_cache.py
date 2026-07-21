#!/usr/bin/env python3
"""Lift hierarchical official-SAM3 masks into scale-ordered 3-D edge votes.

This is intentionally separate from the legacy binary relation cache.  Each
mask contributes an upper or lower merge-scale constraint; no primitive is
assigned to a single smallest mask and no positive view overwrites a negative
view.  ``raster_adjoint`` is the promoted lifting route: it computes mask
membership through the renderer's true alpha-compositing color adjoint.  The
older sparse raster sidecar remains a named compatibility route.  Centre
projection remains an explicitly labelled diagnostic, never the promotion
path for a scale-ordered relation model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.interfaces.relation_calibrator import edge_relation_features
from radio_gs.interfaces.scale_ordered_relation import (
    accumulate_scale_ordered_votes,
    logarithmic_scale_bin_edges,
    merge_scale_intervals,
    robust_mask_physical_radius,
)
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.training.feature_training_utils import SimpleRadioDataset


def _sha256_tensor(values: torch.Tensor) -> str:
    array = torch.as_tensor(values).detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    # Match the frozen MPR sidecar's ``_sha256_tensor_rows`` exactly.
    return hashlib.sha256(array.tobytes()).hexdigest()


def _unpack_payload_masks(payload: dict) -> torch.Tensor:
    height, width = (int(value) for value in payload["mask_shape"])
    masks = torch.from_numpy(unpack_masks(payload["packed_masks"], width))
    if masks.ndim != 3 or tuple(masks.shape[1:]) != (height, width):
        raise ValueError("packed SAM3 masks do not align with declared mask_shape")
    return masks.bool()


def _align_masks_to_raster(
    masks: torch.Tensor, *, image_height: int, image_width: int,
) -> tuple[torch.Tensor, bool]:
    """Align official binary masks with the frozen feature raster.

    Nearest-neighbour resampling is intentional: it preserves an official
    discrete mask label at the MPR lattice, rather than inventing fractional
    SAM probabilities.  The caller records every use in cache provenance.
    """

    values = torch.as_tensor(masks).bool().cpu()
    if values.ndim != 3:
        raise ValueError("masks must be [M,H,W]")
    target_shape = (int(image_height), int(image_width))
    if tuple(values.shape[1:]) == target_shape:
        return values, False
    return F.interpolate(
        values.float().unsqueeze(1), size=target_shape, mode="nearest",
    )[:, 0].bool(), True


def _mask_cache_paths(raw_roots: str) -> list[Path]:
    """Return deterministic, non-overlapping caches from one or more roots.

    Multiple roots let independent GPUs decode disjoint source-frame shards
    without racing on a shared manifest.  A duplicate frame stem is an error:
    choosing one duplicate silently would make the relation teacher depend on
    worker order.
    """

    roots = [Path(value) for value in str(raw_roots).replace(",", " ").split() if value]
    if not roots or any(not root.is_dir() for root in roots):
        raise FileNotFoundError("mask-root must contain one or more existing cache directories")
    paths = [path for root in roots for path in root.glob("*.pt")]
    if not paths:
        raise FileNotFoundError("mask-root contains no SAM3 cache tensors")
    by_stem: dict[str, Path] = {}
    for path in paths:
        if path.stem in by_stem:
            raise ValueError(f"duplicate SAM3 mask frame stem across cache roots: {path.stem}")
        by_stem[path.stem] = path
    return [by_stem[stem] for stem in sorted(by_stem, key=lambda value: (int(value) if value.isdigit() else value, value))]


def _center_projection_membership(
    graph: dict, frame: dict, mask_payload: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy centre readout, retained only as a named diagnostic baseline."""

    xyz = torch.as_tensor(graph["xyz"]).float()
    pose = torch.as_tensor(frame["pose"]).float()
    camera = torch.cat([xyz, torch.ones(len(xyz), 1)], 1) @ torch.linalg.inv(pose).T
    z = camera[:, 2]
    kd, kc = torch.as_tensor(graph["depth_intrinsic"]).float(), torch.as_tensor(graph["color_intrinsic"]).float()
    depth = torch.from_numpy(np.asarray(Image.open(frame["depth"]), dtype=np.float32)) / 1000.0
    ud = kd[0, 0] * camera[:, 0] / z.clamp_min(1e-6) + kd[0, 2]
    vd = kd[1, 1] * camera[:, 1] / z.clamp_min(1e-6) + kd[1, 2]
    ix, iy = ud.round().long(), vd.round().long()
    depth_inside = (ix >= 0) & (iy >= 0) & (ix < depth.shape[1]) & (iy < depth.shape[0])
    observed = (z > 0.15) & depth_inside
    visible_rows = torch.where(observed)[0]
    if len(visible_rows):
        measured = depth[iy[visible_rows], ix[visible_rows]]
        observed[visible_rows] &= (measured > 0) & ((measured - z[visible_rows]).abs() < 0.10)
    masks = _unpack_payload_masks(mask_payload)
    height, width = masks.shape[1:]
    u = kc[0, 0] * camera[:, 0] / z.clamp_min(1e-6) + kc[0, 2]
    v = kc[1, 1] * camera[:, 1] / z.clamp_min(1e-6) + kc[1, 2]
    ui, vi = u.round().long(), v.round().long()
    color_inside = (ui >= 0) & (vi >= 0) & (ui < width) & (vi < height)
    observed &= color_inside
    membership = torch.zeros(len(masks), len(xyz), dtype=torch.float32)
    rows = torch.where(observed)[0]
    if len(rows) and len(masks):
        membership[:, rows] = masks[:, vi[rows], ui[rows]].float()
    return membership, observed


def raster_responsibility_membership(
    masks: torch.Tensor,
    *,
    primitive_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    weights: torch.Tensor,
    primitive_count: int,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Convert sparse pixel-to-primitive responsibilities into soft masks.

    For mask ``m`` and primitive ``i`` this computes exactly
    ``sum_{p in m} R_pi / sum_p R_pi`` over the supplied raster assignments.
    The final boolean declares whether a nearest-neighbour resize was needed
    solely to align the official mask raster with the frozen MPR raster.
    """

    values, resized = _align_masks_to_raster(
        masks, image_height=image_height, image_width=image_width,
    )
    primitive_ids = torch.as_tensor(primitive_ids).long().cpu().reshape(-1)
    pixel_ids = torch.as_tensor(pixel_ids).long().cpu().reshape(-1)
    weights = torch.as_tensor(weights).float().cpu().reshape(-1)
    if not (primitive_ids.shape == pixel_ids.shape == weights.shape):
        raise ValueError("raster responsibility rows do not align")
    if primitive_ids.numel() and (
        int(primitive_ids.min()) < 0 or int(primitive_ids.max()) >= int(primitive_count)
    ):
        raise ValueError("raster responsibility primitive id is out of range")
    pixels = int(image_height) * int(image_width)
    if pixel_ids.numel() and (int(pixel_ids.min()) < 0 or int(pixel_ids.max()) >= pixels):
        raise ValueError("raster responsibility pixel id is out of range")
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("raster responsibility weights must be finite and positive")
    denominator = torch.zeros(int(primitive_count), dtype=torch.float32)
    denominator.index_add_(0, primitive_ids, weights)
    membership = torch.zeros(values.shape[0], int(primitive_count), dtype=torch.float32)
    flattened = values.reshape(values.shape[0], -1)
    for mask_index in range(values.shape[0]):
        numerator = torch.zeros(int(primitive_count), dtype=torch.float32)
        numerator.index_add_(0, primitive_ids, weights * flattened[mask_index, pixel_ids].float())
        membership[mask_index] = numerator / denominator.clamp_min(1e-12)
    return membership, denominator > 0, resized


def _load_responsibility_assignments(
    path: Path, graph: dict,
) -> tuple[dict[int, dict], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    assignments = payload.get("assignments")
    if int(payload.get("schema_version", -1)) != 1 or not isinstance(assignments, list):
        raise ValueError("responsibility cache must use the frozen schema-v1 assignment format")
    expected_xyz = _sha256_tensor(torch.as_tensor(graph["xyz"]))
    global_to_local = None
    identity = "direct_primitive_identity"
    if metadata.get("xyz_sha256") != expected_xyz:
        # A canonical support graph often removes MPR-invalid Gaussians while
        # the raster sidecar correctly indexes the original full Gaussian
        # set.  ``global_rows`` is the explicit, lossless bridge between the
        # two—not a nearest-neighbour approximation—and is therefore safe to
        # use when its source capability geometry exactly matches the sidecar.
        if "global_rows" not in graph:
            raise ValueError(
                "raster responsibility geometry does not match the relation graph; "
                "refusing a Gaussian/voxel ID mismatch"
            )
        graph_metadata = dict(graph.get("metadata", {}))
        capability_path = Path(str(graph_metadata.get("capability_cache", "")))
        if not capability_path.is_file():
            raise ValueError("global-row graph lacks its source capability cache")
        capability = torch.load(capability_path, map_location="cpu", weights_only=False)
        full_xyz = torch.as_tensor(capability.get("xyz")).float().cpu()
        if metadata.get("xyz_sha256") != _sha256_tensor(full_xyz):
            raise ValueError("responsibility cache does not match global-row capability geometry")
        global_rows = torch.as_tensor(graph["global_rows"]).long().cpu().reshape(-1)
        if global_rows.shape != (len(graph["xyz"]),) or global_rows.numel() == 0:
            raise ValueError("global_rows do not align with the relation graph")
        if int(global_rows.min()) < 0 or int(global_rows.max()) >= len(full_xyz):
            raise ValueError("global_rows are outside source capability geometry")
        if not torch.equal(torch.as_tensor(graph["xyz"]).float().cpu(), full_xyz[global_rows]):
            raise ValueError("global-row graph geometry differs from source capability rows")
        global_to_local = torch.full((len(full_xyz),), -1, dtype=torch.long)
        global_to_local[global_rows] = torch.arange(len(global_rows), dtype=torch.long)
        identity = "global_gaussian_responsibility_to_explicit_valid_canonical_subset"
    frame_ids = metadata.get("selected_frame_indices")
    if not isinstance(frame_ids, list) or len(frame_ids) != len(assignments):
        raise ValueError("responsibility cache frame identifiers do not align with assignments")
    if not {"feature_height", "feature_width"}.issubset(metadata):
        raise ValueError("responsibility cache lacks its frozen raster shape")
    result: dict[int, dict] = {}
    for frame_id, assignment in zip(frame_ids, assignments):
        if not isinstance(assignment, dict):
            raise ValueError("malformed responsibility assignment")
        primitive_ids = assignment.get("primitive_ids", assignment.get("gaussian_ids"))
        if primitive_ids is None:
            raise ValueError("responsibility assignment lacks primitive/gaussian IDs")
        primitive_ids = torch.as_tensor(primitive_ids).long().cpu().reshape(-1)
        pixel_ids = torch.as_tensor(assignment.get("pixel_ids")).long().cpu().reshape(-1)
        weights = torch.as_tensor(assignment.get("weights")).float().cpu().reshape(-1)
        if not (primitive_ids.shape == pixel_ids.shape == weights.shape):
            raise ValueError("malformed responsibility assignment row shapes")
        if global_to_local is not None:
            if primitive_ids.numel() and (
                int(primitive_ids.min()) < 0 or int(primitive_ids.max()) >= len(global_to_local)
            ):
                raise ValueError("global responsibility primitive ID is out of range")
            primitive_ids = global_to_local[primitive_ids]
            keep = primitive_ids >= 0
            primitive_ids, pixel_ids, weights = (
                primitive_ids[keep], pixel_ids[keep], weights[keep]
            )
        result[int(frame_id)] = {
            "primitive_ids": primitive_ids, "pixel_ids": pixel_ids, "weights": weights,
        }
    metadata = {**metadata, "relation_graph_identity": identity}
    return result, metadata


def _select_relation_graph_rows(
    full_membership: torch.Tensor,
    full_observed: torch.Tensor,
    *,
    graph: dict,
    relation_graph_identity: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select an explicit canonical subset from full-renderer rows.

    The renderer operates on the checkpoint's full Gaussian set.  A canonical
    support graph may intentionally remove invalid rows; ``global_rows`` is
    the only allowed bridge.  This helper deliberately refuses a nearest-row
    or nearest-3D-point fallback.
    """

    member = torch.as_tensor(full_membership).float().cpu()
    observed = torch.as_tensor(full_observed).bool().cpu().reshape(-1)
    if member.ndim != 2 or observed.shape != (member.shape[1],):
        raise ValueError("full compositing membership and observation rows do not align")
    graph_count = len(torch.as_tensor(graph["xyz"]))
    if relation_graph_identity == "direct_primitive_identity":
        if member.shape[1] != graph_count:
            raise ValueError("direct adjoint geometry does not match relation graph rows")
        return member, observed
    if relation_graph_identity != "global_gaussian_responsibility_to_explicit_valid_canonical_subset":
        raise ValueError("unknown relation-graph identity for compositing membership")
    rows = torch.as_tensor(graph.get("global_rows")).long().cpu().reshape(-1)
    if rows.shape != (graph_count,) or rows.numel() == 0:
        raise ValueError("global rows do not align with relation graph")
    if int(rows.min()) < 0 or int(rows.max()) >= member.shape[1]:
        raise ValueError("global rows are outside full compositing geometry")
    return member[:, rows], observed[rows]


def _load_adjoint_context(
    args: argparse.Namespace,
    *,
    graph: dict,
    responsibility_metadata: dict,
) -> dict:
    """Load and fail-close the exact renderer context for mask lifting."""

    metadata = dict(responsibility_metadata)
    config_path = Path(str(args.adjoint_config or metadata.get("config", ""))).expanduser()
    checkpoint_path = Path(str(args.adjoint_checkpoint or metadata.get("checkpoint", ""))).expanduser()
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise ValueError("raster_adjoint requires a readable frozen config and checkpoint")
    expected_config = str(metadata.get("config", ""))
    expected_checkpoint = str(metadata.get("checkpoint", ""))
    if expected_config and config_path.resolve() != Path(expected_config).expanduser().resolve():
        raise ValueError("raster_adjoint config differs from frozen MPR sidecar")
    if expected_checkpoint and checkpoint_path.resolve() != Path(expected_checkpoint).expanduser().resolve():
        raise ValueError("raster_adjoint checkpoint differs from frozen MPR sidecar")

    from radio_gs.scripts.build_gaussian_multiview_teacher_cache import _gaussian_state_sha256
    from radio_gs.scripts.eval_lerf_direct_3d_selection import raster_adjoint_registered_view_features
    from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline

    device = torch.device(str(args.adjoint_device))
    config = load_config(str(config_path))
    model, _codec, renderer, _sharpener, _refiner, _cfg, _hybrid = load_render_pipeline(
        str(config_path), str(checkpoint_path), device,
        strict_checkpoint_contract=True, load_ply_rgb_features=False,
    )
    model.eval()
    full_xyz = model.get_xyz().detach().float().cpu()
    if metadata.get("xyz_sha256") != _sha256_tensor(full_xyz):
        raise ValueError("raster_adjoint checkpoint geometry differs from frozen MPR sidecar")
    expected_state = str(metadata.get("gaussian_state_sha256", ""))
    if expected_state and _gaussian_state_sha256(model) != expected_state:
        raise ValueError("raster_adjoint Gaussian state differs from frozen MPR sidecar")
    # The sidecar validation already proves that either direct rows or
    # graph.global_rows align with this full geometry.  Recheck the model row
    # count here before any expensive autograd rendering.
    identity = str(metadata.get("relation_graph_identity", ""))
    if identity == "direct_primitive_identity" and len(full_xyz) != len(graph["xyz"]):
        raise ValueError("direct raster-adjoint model rows differ from relation graph")
    if identity == "global_gaussian_responsibility_to_explicit_valid_canonical_subset":
        rows = torch.as_tensor(graph.get("global_rows")).long().cpu().reshape(-1)
        if rows.numel() == 0 or int(rows.max()) >= len(full_xyz):
            raise ValueError("raster-adjoint global rows differ from model geometry")

    height, width = int(metadata["feature_height"]), int(metadata["feature_width"])
    if (int(getattr(config, "feature_height", height)), int(getattr(config, "feature_width", width))) != (height, width):
        raise ValueError("raster_adjoint feature raster differs from frozen MPR sidecar")
    dataset = SimpleRadioDataset(
        str(getattr(config, "feature_dir")),
        pose_file=str(getattr(config, "pose_file", "") or "") or None,
        pose_dir=str(getattr(config, "pose_dir", "") or "") or None,
        feature_size=(height, width), split="train",
        dataset_type=str(getattr(config, "dataset_type", "lerf")),
    )
    poses_by_frame = {
        int(frame_id): torch.from_numpy(dataset.poses_w2c[index]).float()
        for index, frame_id in enumerate(dataset.frame_indices)
    }
    frame_ids = [int(value) for value in metadata.get("selected_frame_indices", [])]
    if not frame_ids or any(frame_id not in poses_by_frame for frame_id in frame_ids):
        raise ValueError("raster_adjoint dataset poses do not cover frozen MPR frame IDs")
    if metadata.get("pose_sha256") != _sha256_tensor(torch.stack([poses_by_frame[key] for key in frame_ids])):
        raise ValueError("raster_adjoint poses differ from frozen MPR sidecar")
    return {
        "model": model, "renderer": renderer, "device": device,
        "poses_by_frame": poses_by_frame, "height": height, "width": width,
        "alpha_threshold": float(metadata["alpha_threshold"]),
        "raster_adjoint": raster_adjoint_registered_view_features,
        "provenance": {
            "config": str(config_path.resolve()), "checkpoint": str(checkpoint_path.resolve()),
            "xyz_sha256": _sha256_tensor(full_xyz),
            "gaussian_state_sha256": _gaussian_state_sha256(model),
            "pose_sha256": metadata["pose_sha256"],
            "feature_height": height, "feature_width": width,
            "alpha_threshold": float(metadata["alpha_threshold"]),
            "channel_chunk_size": int(args.adjoint_channel_chunk_size),
        },
    }


def _raster_adjoint_membership(
    masks: torch.Tensor,
    *,
    frame_id: int,
    graph: dict,
    responsibility_metadata: dict,
    context: dict,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Compute exact alpha-compositing mask membership for one source view."""

    if int(frame_id) not in context["poses_by_frame"]:
        raise ValueError("SAM3 frame is absent from frozen raster-adjoint poses")
    values, resized = _align_masks_to_raster(
        masks, image_height=int(context["height"]), image_width=int(context["width"]),
    )
    if not len(values):
        return torch.empty(0, len(graph["xyz"])), torch.zeros(len(graph["xyz"]), dtype=torch.bool), resized
    model, renderer, device = context["model"], context["renderer"], context["device"]
    pose = context["poses_by_frame"][int(frame_id)].to(device)
    with torch.inference_mode():
        alpha = renderer.render_features(
            model, pose,
            feature_height=int(context["height"]), feature_width=int(context["width"]),
        )
    sums, denominator = context["raster_adjoint"](
        model=model, renderer=renderer, viewmat=pose,
        siglip_feat=values.unsqueeze(0).to(device=device, dtype=torch.float32),
        alpha_map=alpha["alpha_map"].unsqueeze(0),
        alpha_threshold=float(context["alpha_threshold"]),
        channel_chunk_size=int(context["provenance"]["channel_chunk_size"]),
    )
    full_member = (sums / denominator.clamp_min(1e-12).unsqueeze(1)).T
    # The adjoint of a binary mask is a weighted fraction.  Clamp only tiny
    # floating-point overshoots, then fail if a material violation remains.
    if not bool(torch.isfinite(full_member).all()):
        raise ValueError("raster-adjoint membership contains non-finite values")
    if float(full_member.min()) < -1e-4 or float(full_member.max()) > 1.0001:
        raise ValueError("raster-adjoint membership is outside its probability range")
    full_member = full_member.clamp(0.0, 1.0).cpu()
    membership, observed = _select_relation_graph_rows(
        full_member, denominator > 0, graph=graph,
        relation_graph_identity=str(responsibility_metadata["relation_graph_identity"]),
    )
    return membership, observed, resized


def _mask_metadata(payload: dict, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    quality = torch.as_tensor(payload.get("scores", torch.ones(count))).float().reshape(-1)
    stability = torch.as_tensor(payload.get("stability", torch.ones(count))).float().reshape(-1)
    if quality.shape != (count,) or stability.shape != (count,):
        raise ValueError("SAM3 quality/stability rows do not align with masks")
    return quality.clamp(0, 1), stability.clamp(0, 1)


def _assert_query_free_graph_provenance(graph_metadata: dict) -> None:
    """Require an explicit, non-leaky provenance declaration from a graph.

    The early ScanNet-frame relation graphs used ``labels_opened`` / 
    ``instances_opened`` / ``masks_opened`` / ``text_opened``.  Canonical MPR
    support graphs predate that naming and instead carry the capability-cache
    contract: ``query_independent`` together with the ``benchmark_*`` flags.
    Both are legitimate, but an absent declaration is not evidence that a
    graph is query free.  In particular, do not use ``dict.get(..., True)``
    here: it accidentally rejects the modern contract while hiding which
    contract was expected.
    """

    metadata = dict(graph_metadata)
    capability = dict(metadata.get("capability_metadata", {}))
    # A top-level field is the graph's own declaration; otherwise inherit the
    # frozen capability contract from which it was constructed.
    provenance = {**capability, **metadata}
    forbidden = (
        "labels_opened", "instances_opened", "masks_opened", "text_opened",
        "benchmark_labels_opened", "benchmark_instances_opened",
        "benchmark_masks_opened", "text_queries_opened",
    )
    if any(bool(provenance.get(key, False)) for key in forbidden):
        raise ValueError("scene graph violates query-free relation-teacher provenance")
    legacy_keys = ("labels_opened", "instances_opened", "masks_opened", "text_opened")
    legacy_explicit = all(key in provenance for key in legacy_keys)
    modern_explicit = provenance.get("query_independent") is True and all(
        key in provenance for key in ("benchmark_masks_opened", "text_queries_opened")
    )
    if not (legacy_explicit or modern_explicit):
        raise ValueError(
            "scene graph lacks an explicit query-free provenance declaration"
        )


def build(args: argparse.Namespace) -> dict:
    if not 0.0 <= float(args.minimum_stability) <= 1.0:
        raise ValueError("minimum-stability must lie in [0,1]")
    if int(args.adjoint_channel_chunk_size) <= 0:
        raise ValueError("adjoint-channel-chunk-size must be positive")
    graph_path = Path(args.scene_graph)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    graph_metadata = dict(graph.get("metadata", {}))
    _assert_query_free_graph_provenance(graph_metadata)
    graph_frames = {
        Path(frame["color"]).stem: frame for frame in graph.get("frames", [])
    }
    responsibility_by_frame: dict[int, dict] = {}
    responsibility_metadata: dict = {}
    adjoint_context: dict | None = None
    membership_mode = str(args.membership_mode)
    if membership_mode in {"raster_responsibility", "raster_adjoint"}:
        if not str(args.responsibility_cache):
            raise ValueError(f"{membership_mode} mode requires --responsibility-cache")
        responsibility_by_frame, responsibility_metadata = _load_responsibility_assignments(
            Path(args.responsibility_cache), graph
        )
    if membership_mode == "raster_adjoint":
        adjoint_context = _load_adjoint_context(
            args, graph=graph, responsibility_metadata=responsibility_metadata,
        )

    memberships, observations, radii, quality_rows, stability_rows = [], [], [], [], []
    # Optional, offline-only frozen teacher evidence.  It is deliberately not
    # part of the field or an inference input; it exists so a later GT-only
    # audit can distinguish proposal impurity from missing 3-D coverage.
    membership_records: list[dict] = []
    used, skipped, physical_radii = [], [], []
    resized_frames: list[str] = []
    resized_mask_count = 0
    mask_schema_versions = set()
    xyz = torch.as_tensor(graph["xyz"]).float()
    requested_stems = {
        value for value in str(args.mask_stems).replace(",", " ").split() if value
    }
    mask_paths = _mask_cache_paths(args.mask_root)
    for mask_path in mask_paths:
        if requested_stems and mask_path.stem not in requested_stems:
            continue
        frame = graph_frames.get(mask_path.stem)
        if membership_mode == "center_projection" and frame is None:
            skipped.append({"mask": mask_path.name, "reason": "frame_absent_from_scene_graph"})
            continue
        mask_payload = torch.load(mask_path, map_location="cpu", weights_only=False)
        mask_metadata = dict(mask_payload.get("metadata", {}))
        if not bool(mask_metadata.get("official_decoder", False)) or not bool(mask_metadata.get("query_free", False)):
            raise ValueError(f"{mask_path} is not a query-free official SAM3 cache")
        mask_schema_versions.add(int(mask_metadata.get("schema_version", 1)))
        if membership_mode == "center_projection":
            member, observed = _center_projection_membership(graph, frame, mask_payload)
        else:
            try:
                frame_id = int(mask_path.stem)
            except ValueError as error:
                raise ValueError("raster lifting requires numeric ScanNet frame stems") from error
            assignment = responsibility_by_frame.get(frame_id)
            if assignment is None:
                skipped.append({"mask": mask_path.name, "reason": "frame_absent_from_responsibility_cache"})
                continue
            masks = _unpack_payload_masks(mask_payload)
            if membership_mode == "raster_responsibility":
                member, observed, resized = raster_responsibility_membership(
                    masks, primitive_ids=assignment["primitive_ids"],
                    pixel_ids=assignment["pixel_ids"], weights=assignment["weights"],
                    primitive_count=len(xyz),
                    image_height=int(responsibility_metadata["feature_height"]),
                    image_width=int(responsibility_metadata["feature_width"]),
                )
            else:
                assert adjoint_context is not None
                member, observed, resized = _raster_adjoint_membership(
                    masks, frame_id=frame_id, graph=graph,
                    responsibility_metadata=responsibility_metadata,
                    context=adjoint_context,
                )
            if resized:
                resized_frames.append(mask_path.name)
                resized_mask_count += int(masks.shape[0])
        quality, stability = _mask_metadata(mask_payload, member.shape[0])
        scale = torch.tensor([
            robust_mask_physical_radius(
                xyz, member[index], inside_threshold=float(args.inside_threshold),
                minimum_primitives=int(args.minimum_primitives_per_mask),
            )
            for index in range(member.shape[0])
        ], dtype=torch.float32)
        valid = (
            torch.isfinite(scale) & (scale > 0)
            & (stability >= float(args.minimum_stability))
        )
        if not bool(valid.any()):
            skipped.append({"mask": mask_path.name, "reason": "no_mask_with_metric_3d_support"})
            continue
        memberships.append(member[valid]); observations.append(observed)
        radii.append(scale[valid]); quality_rows.append(quality[valid]); stability_rows.append(stability[valid])
        if str(getattr(args, "save_membership_sidecar", "")):
            membership_records.append({
                "mask_frame": mask_path.name,
                "source_mask_indices": torch.where(valid)[0].to(torch.int32).cpu(),
                "membership": member[valid].to(dtype=torch.float16).cpu(),
                "observed": observed.bool().cpu(),
                "physical_radius_m": scale[valid].float().cpu(),
                "quality": quality[valid].float().cpu(),
                "stability": stability[valid].float().cpu(),
            })
        used.append(mask_path.name); physical_radii.extend(scale[valid].tolist())
    if not memberships:
        raise RuntimeError("no query-free SAM3 masks align with usable relation observations")

    full_edge = torch.as_tensor(graph["edge_index"]).long()
    unique = full_edge[0] < full_edge[1]
    edge_rows = torch.where(unique)[0]
    edge = full_edge[:, unique]
    bins = logarithmic_scale_bin_edges(
        minimum_radius_m=float(args.scale_minimum_radius_m),
        maximum_radius_m=float(args.scale_maximum_radius_m), bins=int(args.scale_bins),
    )
    votes = accumulate_scale_ordered_votes(
        memberships, observations, radii, edge, bins,
        quality=quality_rows, stability=stability_rows,
        inside_threshold=float(args.inside_threshold), outside_threshold=float(args.outside_threshold),
        mask_chunk=int(args.mask_chunk),
    )
    intervals = merge_scale_intervals(votes)
    constrained = (intervals["same_mass"] + intervals["separate_mass"]) > 0
    both = intervals["has_lower"] & intervals["has_upper"]
    scene_name = str(graph.get("scene", "") or graph_path.parent.name)
    raster_lifting_semantics = {
        "center_projection": "projected_primitive_centres_diagnostic_only",
        "raster_responsibility": "sparse_top1_footprint_depth_proxy",
        "raster_adjoint": "true_alpha_compositing_adjoint",
    }[membership_mode]
    vote_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[str(getattr(args, "vote_storage_dtype", "float16"))]
    vote_storage = (
        "fp32_soft_same_and_separate_mass_no_overwrite"
        if vote_dtype == torch.float32
        else "fp16_soft_same_and_separate_mass_no_overwrite"
    )
    payload = {
        "schema_version": 2, "scene": scene_name,
        "edge_rows": edge_rows, "edge_index": edge,
        "features": edge_relation_features(graph)[unique],
        # Float32 shards preserve exact additive vote semantics when a large
        # scene is rendered independently on several GPUs and merged later.
        # The legacy compact fp16 option remains available for one-shot
        # diagnostic caches.
        "same_votes": votes["same_votes"].to(dtype=vote_dtype),
        "separate_votes": votes["separate_votes"].to(dtype=vote_dtype),
        "observed_votes": votes["observed_votes"].to(dtype=vote_dtype),
        "same_events": votes["same_events"], "separate_events": votes["separate_events"],
        "scale_bin_edges_log": votes["scale_bin_edges_log"],
        "merge_log_radius": intervals["merge_log_radius"].float(),
        "lower_log_radius": intervals["lower_log_radius"].float(),
        "upper_log_radius": intervals["upper_log_radius"].float(),
        "has_lower": intervals["has_lower"], "has_upper": intervals["has_upper"],
        "interval_consistent": intervals["interval_consistent"],
        "constraint_entropy": intervals["constraint_entropy"].to(dtype=vote_dtype),
        "same_mass": intervals["same_mass"].to(dtype=vote_dtype),
        "separate_mass": intervals["separate_mass"].to(dtype=vote_dtype),
        "metadata": {
            "teacher": "official_sam3_multimask_scale_ordered_regions",
            "query_free": True, "labels_opened": False, "instances_opened": False,
            "text_opened": False, "scene_graph": str(graph_path.resolve()),
            "scene_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
            "membership_lifting": membership_mode,
            "raster_responsibility_used": membership_mode in {"raster_responsibility", "raster_adjoint"},
            "raster_lifting_semantics": raster_lifting_semantics,
            "responsibility_cache": str(Path(args.responsibility_cache).resolve()) if args.responsibility_cache else "",
            "responsibility_metadata": responsibility_metadata,
            "mask_roots": [str(Path(value).resolve()) for value in str(args.mask_root).replace(",", " ").split() if value],
            "raster_adjoint_provenance": (
                dict(adjoint_context["provenance"]) if adjoint_context is not None else {}
            ),
            "mask_raster_alignment": "nearest_label_resample_to_frozen_mpr_raster",
            "resized_mask_frames": resized_frames,
            "resized_mask_count": int(resized_mask_count),
            "mask_frames": used, "skipped_mask_frames": skipped,
            "mask_schema_versions": sorted(mask_schema_versions),
            "inside_threshold": float(args.inside_threshold),
            "outside_threshold": float(args.outside_threshold),
            "minimum_primitives_per_mask": int(args.minimum_primitives_per_mask),
            "minimum_stability": float(args.minimum_stability),
            "scale_definition": "Q0.90_distance_to_coordinatewise_median_m",
            "scale_bins": int(args.scale_bins),
            "scale_minimum_radius_m": float(args.scale_minimum_radius_m),
            "scale_maximum_radius_m": float(args.scale_maximum_radius_m),
            "vote_storage": vote_storage,
        },
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    membership_sidecar = str(getattr(args, "save_membership_sidecar", ""))
    if membership_sidecar:
        sidecar_path = Path(membership_sidecar)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema_version": 1,
            "records": membership_records,
            "metadata": {
                "query_free": True, "labels_opened": False, "instances_opened": False,
                "text_opened": False, "offline_teacher_audit_only": True,
                "not_an_inference_representation": True,
                "source_relation_cache": str(output.resolve()),
                "scene_graph": str(graph_path.resolve()),
                "scene_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
                "membership_lifting": membership_mode,
                "raster_lifting_semantics": raster_lifting_semantics,
                "inside_threshold": float(args.inside_threshold),
                "outside_threshold": float(args.outside_threshold),
                "membership_storage": "fp16_full_relation_graph_rows_offline_audit_only",
                "mask_frames": used,
            },
        }, sidecar_path)
    report = {
        "output": str(output.resolve()), "scene": payload["scene"],
        "membership_lifting": membership_mode, "mask_frames": len(used),
        "masks_with_metric_3d_scale": len(physical_radii),
        "physical_radius_m": {
            "min": float(np.min(physical_radii)), "p50": float(np.quantile(physical_radii, 0.50)),
            "p90": float(np.quantile(physical_radii, 0.90)), "max": float(np.max(physical_radii)),
        },
        "edges": int(edge.shape[1]), "constrained_edges": int(constrained.sum()),
        "constrained_edge_fraction": float(constrained.float().mean()),
        "edges_with_both_bounds": int(both.sum()),
        "interval_consistent_fraction_among_both": float(
            intervals["interval_consistent"][both].float().mean() if bool(both.any()) else 0.0
        ),
        "same_events": [int(value) for value in votes["same_events"]],
        "separate_events": [int(value) for value in votes["separate_events"]],
        "raster_lifting_semantics": raster_lifting_semantics,
        "resized_mask_frames": resized_frames,
        "resized_mask_count": int(resized_mask_count),
        "skipped_mask_frames": skipped,
        "membership_sidecar": str(Path(membership_sidecar).resolve()) if membership_sidecar else "",
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-graph", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument(
        "--mask-stems", default="",
        help="Optional deterministic subset of frame stems for a diagnostic split.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--membership-mode", choices=("raster_adjoint", "raster_responsibility", "center_projection"),
        default="raster_responsibility",
    )
    parser.add_argument("--responsibility-cache", default="")
    parser.add_argument(
        "--adjoint-config", default="",
        help="Frozen field config; defaults to the responsibility sidecar contract.",
    )
    parser.add_argument(
        "--adjoint-checkpoint", default="",
        help="Frozen field checkpoint; defaults to the responsibility sidecar contract.",
    )
    parser.add_argument("--adjoint-device", default="cuda:0")
    parser.add_argument("--adjoint-channel-chunk-size", type=int, default=32)
    parser.add_argument("--inside-threshold", type=float, default=0.80)
    parser.add_argument("--outside-threshold", type=float, default=0.20)
    parser.add_argument("--minimum-primitives-per-mask", type=int, default=3)
    parser.add_argument(
        "--minimum-stability", type=float, default=0.0,
        help="Apply only at soft-vote construction; raw multimask caches remain intact.",
    )
    parser.add_argument("--scale-minimum-radius-m", type=float, default=0.05)
    parser.add_argument("--scale-maximum-radius-m", type=float, default=4.0)
    parser.add_argument("--scale-bins", type=int, default=8)
    parser.add_argument(
        "--vote-storage-dtype", choices=("float16", "float32"), default="float16",
        help=(
            "Storage precision for additive votes. Use float32 for GPU shards "
            "that will be merged; float16 is the compact diagnostic default."
        ),
    )
    parser.add_argument(
        "--mask-chunk", type=int, default=4,
        help="Bound CPU memory while vectorizing all masks in a frame over graph edges.",
    )
    parser.add_argument(
        "--save-membership-sidecar", default="",
        help=(
            "Optional fp16 query-free alpha-adjoint memberships for offline "
            "teacher-coverage audits only; never an inference representation."
        ),
    )
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
