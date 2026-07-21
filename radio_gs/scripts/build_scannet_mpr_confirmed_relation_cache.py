#!/usr/bin/env python3
"""Lift MPR-confirmed official-SAM3 masks into same-only 3-D track votes.

This is the second half of the offline cross-view teacher contract.  A target
mask is accepted only when its *true alpha-adjoint* membership still contains
the exact canonical primitive that caused the official-SAM3 target prompt.
The accepted virtual region is the union of the frozen source observation and
that independently re-segmented target observation.

Only positive (same-region) constraints are emitted here.  The base automatic
teacher keeps its original local same/separate evidence; a later explicit
additive merge may add these confirmed same constraints without fabricating a
virtual track's exterior as a negative observation.  This cache is a
query-free, no-learning teacher diagnostic, never a query-time field input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from radio_gs.interfaces.relation_calibrator import edge_relation_features
from radio_gs.interfaces.scale_ordered_relation import (
    accumulate_scale_ordered_votes,
    logarithmic_scale_bin_edges,
    merge_scale_intervals,
    robust_mask_physical_radius,
)
from radio_gs.scripts.build_sam3_mpr_confirmed_mask_cache import _load_source_masks
from radio_gs.scripts.build_scannet_scale_ordered_relation_cache import (
    _assert_query_free_graph_provenance,
    _load_adjoint_context,
    _load_responsibility_assignments,
    _mask_metadata,
    _raster_adjoint_membership,
    _unpack_payload_masks,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompted_mask_paths(roots: list[str]) -> list[Path]:
    paths: dict[str, Path] = {}
    for raw_root in roots:
        root = Path(raw_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"prompted mask root does not exist: {root}")
        for path in root.glob("*.pt"):
            if path.stem in paths:
                raise ValueError(f"duplicate prompted target frame across roots: {path.stem}")
            paths[path.stem] = path
    if not paths:
        raise FileNotFoundError("prompted mask roots contain no cache tensors")
    try:
        return [paths[key] for key in sorted(paths, key=lambda value: int(value))]
    except ValueError as error:
        raise ValueError("prompted target cache stems must be numeric ScanNet frame IDs") from error


def _select_prompted_mask_shard(
    paths: list[Path], *, shard_index: int, shard_count: int,
) -> list[Path]:
    """Partition target frames deterministically for additive relation shards."""

    if int(shard_count) <= 0:
        raise ValueError("target shard count must be positive")
    if not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("target shard index must lie in [0, target shard count)")
    ordered = sorted(paths, key=lambda path: int(path.stem))
    return [path for index, path in enumerate(ordered) if index % int(shard_count) == int(shard_index)]


def _assert_prompted_cache_contract(payload: dict, *, graph_path: Path) -> dict:
    metadata = dict(payload.get("metadata", {}))
    if int(metadata.get("schema_version", -1)) != 1:
        raise ValueError("prompted SAM3 cache must use schema v1")
    if metadata.get("source") not in {
        "official_sam3_mpr_confirmed_cross_view_multimask_teacher",
        "official_sam3_mpr_confirmed_cached_multimask_teacher_control",
    }:
        raise ValueError("unexpected prompted SAM3 teacher source")
    required_truths = ("official_decoder", "query_free", "teacher_only", "not_an_inference_representation")
    if not all(bool(metadata.get(key, False)) for key in required_truths):
        raise ValueError("prompted cache lacks official query-free teacher provenance")
    if any(bool(metadata.get(key, False)) for key in ("labels_opened", "instances_opened", "text_opened")):
        raise ValueError("prompted cache was built with benchmark annotations")
    if Path(str(metadata.get("source_scene_graph", ""))).resolve() != graph_path.resolve():
        raise ValueError("prompted cache scene graph differs from requested relation graph")
    if str(metadata.get("source_scene_graph_sha256", "")) != _sha256_file(graph_path):
        raise ValueError("prompted cache graph hash differs from requested relation graph")
    if metadata.get("source_membership_lifting") != "raster_adjoint":
        raise ValueError("prompted cache source anchor is not raster-adjoint")
    if metadata.get("source_raster_lifting_semantics") != "true_alpha_compositing_adjoint":
        raise ValueError("prompted cache lacks true alpha-adjoint source semantics")
    return metadata


def _aligned_prompt_fields(payload: dict, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_frame = torch.as_tensor(payload.get("source_frame")).long().cpu().reshape(-1)
    source_mask_index = torch.as_tensor(payload.get("source_mask_index")).long().cpu().reshape(-1)
    anchor_local = torch.as_tensor(payload.get("anchor_local_primitive")).long().cpu().reshape(-1)
    if any(values.shape != (count,) for values in (source_frame, source_mask_index, anchor_local)):
        raise ValueError("prompted masks lack row-aligned source identity / anchor rows")
    return source_frame, source_mask_index, anchor_local


def _cached_automatic_target_memberships(
    payload: dict,
    *,
    target_frame: int,
    count: int,
    source_by_identity: dict,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Reuse an already-lifted target automatic mask for the control only.

    The cache-reuse control deliberately selects a mask that was already part
    of the base automatic teacher.  Re-rendering that exact mask merely to
    reproduce its alpha-adjoint membership is both needlessly expensive and
    weaker operationally: it gives a second rendering pass an opportunity to
    differ from the frozen base teacher.  Its ``candidate_index`` is therefore
    an explicit index into the target frame's frozen automatic-mask rows.

    This shortcut is *not* available to the real cross-view decoder path:
    its candidate index denotes a newly decoded SAM3 multimask alternative,
    which has no prior alpha-adjoint row and must be lifted by the renderer.
    """

    metadata = dict(payload.get("metadata", {}))
    if metadata.get("source") != "official_sam3_mpr_confirmed_cached_multimask_teacher_control":
        return None
    if metadata.get("confirmation_mode") != "reuse_existing_official_automatic_mask_at_exact_mpr_anchor":
        raise ValueError("cached MPR control lacks its exact automatic-mask identity contract")
    indices = torch.as_tensor(payload.get("candidate_index")).long().cpu().reshape(-1)
    if indices.shape != (count,) or bool((indices < 0).any()):
        raise ValueError("cached MPR control target automatic-mask indices do not align with masks")
    target_rows = []
    observed_rows = []
    for index in indices.tolist():
        target = source_by_identity.get((int(target_frame), int(index)))
        if target is None:
            raise ValueError(
                "cached MPR control references a target automatic mask absent from the "
                "frozen alpha-adjoint source sidecars"
            )
        target_rows.append(torch.as_tensor(target.membership).float().cpu())
        observed_rows.append(torch.as_tensor(target.observed).bool().cpu())
    if not target_rows:
        return torch.empty((0, 0), dtype=torch.float32), torch.empty(0, dtype=torch.bool)
    reference_observed = observed_rows[0]
    if any(not torch.equal(reference_observed, value) for value in observed_rows[1:]):
        raise ValueError("frozen target automatic masks disagree on their frame observation support")
    return torch.stack(target_rows, dim=0), reference_observed


def _batch_virtual_tracks(
    memberships: list[torch.Tensor],
    observations: list[torch.Tensor],
    radii: list[torch.Tensor],
    qualities: list[torch.Tensor],
    stabilities: list[torch.Tensor],
    *,
    batch_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Pack independent virtual tracks without changing their evidence.

    A confirmed source-target track has its own union visibility row, unlike a
    collection of ordinary masks from one camera view.  The relation interface
    supports those per-mask rows, so packing only changes tensor scheduling:
    each output row remains one independent track and its vote is unchanged.
    """

    count = len(memberships)
    if not (count == len(observations) == len(radii) == len(qualities) == len(stabilities)):
        raise ValueError("virtual track tensors do not have a common row count")
    if int(batch_size) <= 0:
        raise ValueError("virtual track batch size must be positive")
    packed_memberships: list[torch.Tensor] = []
    packed_observations: list[torch.Tensor] = []
    packed_radii: list[torch.Tensor] = []
    packed_qualities: list[torch.Tensor] = []
    packed_stabilities: list[torch.Tensor] = []
    for start in range(0, count, int(batch_size)):
        stop = min(start + int(batch_size), count)
        packed_memberships.append(torch.cat(memberships[start:stop], dim=0))
        packed_observations.append(torch.stack(observations[start:stop], dim=0))
        packed_radii.append(torch.cat(radii[start:stop], dim=0))
        packed_qualities.append(torch.cat(qualities[start:stop], dim=0))
        packed_stabilities.append(torch.cat(stabilities[start:stop], dim=0))
    return (
        packed_memberships,
        packed_observations,
        packed_radii,
        packed_qualities,
        packed_stabilities,
    )


def _adjoint_provenance(
    *, responsibility_metadata: dict, adjoint_context: dict | None, channel_chunk_size: int,
) -> dict:
    """Return the exact lifting provenance without needlessly reloading a model.

    A real prompted mask has just been lifted and therefore supplies renderer
    provenance directly.  The cache-reuse control instead reuses the *frozen*
    alpha-adjoint rows from the same responsibility contract.  Its provenance
    is consequently recovered from that immutable contract rather than from a
    second, redundant rendering pass.
    """

    if adjoint_context is not None:
        return dict(adjoint_context["provenance"])
    required = (
        "config", "checkpoint", "xyz_sha256", "gaussian_state_sha256",
        "pose_sha256", "feature_height", "feature_width", "alpha_threshold",
    )
    if any(key not in responsibility_metadata for key in required):
        raise ValueError("frozen MPR responsibility contract lacks alpha-adjoint provenance")
    return {
        key: responsibility_metadata[key]
        for key in required
    } | {"channel_chunk_size": int(channel_chunk_size)}


def build(args: argparse.Namespace) -> dict:
    if not 0.0 <= float(args.inside_threshold) <= 1.0:
        raise ValueError("inside threshold must lie in [0,1]")
    if not 0.0 <= float(args.outside_threshold) < float(args.inside_threshold):
        raise ValueError("outside threshold must be below inside threshold")
    graph_path = Path(args.scene_graph).resolve()
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    _assert_query_free_graph_provenance(dict(graph.get("metadata", {})))
    source_masks, source_metadata, source_graph_path = _load_source_masks(
        [Path(value).resolve() for value in args.source_membership_sidecars]
    )
    if source_graph_path.resolve() != graph_path:
        raise ValueError("source alpha-adjoint memberships use a different relation graph")
    if abs(float(source_metadata["inside_threshold"]) - float(args.inside_threshold)) > 1e-8:
        raise ValueError("confirmed teacher must use its frozen source inside threshold")
    if abs(float(source_metadata["outside_threshold"]) - float(args.outside_threshold)) > 1e-8:
        raise ValueError("confirmed teacher must use its frozen source outside threshold")
    source_by_identity = {
        (item.frame_id, item.source_mask_index): item for item in source_masks
    }

    responsibility_by_frame, responsibility_metadata = _load_responsibility_assignments(
        Path(args.responsibility_cache).resolve(), graph
    )
    # Construct the expensive renderer lazily.  A pure cache-reuse control can
    # obtain every target membership from the already frozen base sidecars;
    # only genuinely re-decoded official-SAM3 masks require a new adjoint.
    adjoint_context: dict | None = None
    xyz = torch.as_tensor(graph["xyz"]).float()
    vote_device = torch.device(str(args.vote_device))
    if vote_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--vote-device requests CUDA but CUDA is unavailable")
    radius_xyz = xyz.to(vote_device)
    full_edge = torch.as_tensor(graph["edge_index"]).long()
    unique = full_edge[0] < full_edge[1]
    edge_rows = torch.where(unique)[0]
    edge = full_edge[:, unique]
    bins = logarithmic_scale_bin_edges(
        minimum_radius_m=float(args.scale_minimum_radius_m),
        maximum_radius_m=float(args.scale_maximum_radius_m), bins=int(args.scale_bins),
    )

    membership_rows: list[torch.Tensor] = []
    observed_rows: list[torch.Tensor] = []
    radius_rows: list[torch.Tensor] = []
    quality_rows: list[torch.Tensor] = []
    stability_rows: list[torch.Tensor] = []
    audit_records: list[dict] = []
    used: list[str] = []
    rejected: list[dict] = []
    reused_target_memberships = 0
    rerendered_target_memberships = 0
    resized_target_frames: set[str] = set()
    prompted_roots = [str(Path(value).resolve()) for value in args.prompted_mask_roots]
    all_prompted_paths = _prompted_mask_paths(prompted_roots)
    prompted_paths = _select_prompted_mask_shard(
        all_prompted_paths,
        shard_index=int(args.target_shard_index),
        shard_count=int(args.target_shard_count),
    )
    if not prompted_paths:
        raise RuntimeError("no prompted target masks remain in the requested relation shard")
    for path in prompted_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _assert_prompted_cache_contract(payload, graph_path=graph_path)
        target_frame = int(path.stem)
        if target_frame not in responsibility_by_frame:
            rejected.append({"target_frame": target_frame, "reason": "target_absent_from_frozen_mpr"})
            continue
        masks = _unpack_payload_masks(payload)
        if not len(masks):
            rejected.append({"target_frame": target_frame, "reason": "no_prompted_sam3_masks"})
            continue
        source_frame, source_mask_index, anchor_local = _aligned_prompt_fields(payload, len(masks))
        reused = _cached_automatic_target_memberships(
            payload, target_frame=target_frame, count=len(masks), source_by_identity=source_by_identity,
        )
        if reused is not None:
            target_member, target_observed = reused
            reused_target_memberships += len(masks)
        else:
            if adjoint_context is None:
                adjoint_context = _load_adjoint_context(
                    args, graph=graph, responsibility_metadata=responsibility_metadata,
                )
            target_member, target_observed, resized = _raster_adjoint_membership(
                masks,
                frame_id=target_frame,
                graph=graph,
                responsibility_metadata=responsibility_metadata,
                context=adjoint_context,
            )
            rerendered_target_memberships += len(masks)
            if resized:
                resized_target_frames.add(path.name)
        quality, stability = _mask_metadata(payload, len(masks))
        if target_member.shape != (len(masks), len(xyz)):
            raise ValueError("prompted alpha-adjoint target memberships do not align with graph")
        for index in range(len(masks)):
            identity = (int(source_frame[index]), int(source_mask_index[index]))
            source = source_by_identity.get(identity)
            if source is None:
                raise ValueError(f"prompted mask references absent source teacher anchor: {identity}")
            anchor = int(anchor_local[index])
            if anchor < 0 or anchor >= len(xyz):
                raise ValueError("prompted mask anchor primitive is outside relation graph")
            if float(source.membership[anchor]) < float(args.inside_threshold):
                raise ValueError("stored MPR prompt anchor is not source-confident")
            # This is the essential confirmation check.  A point merely being
            # sent to the decoder is not evidence that its returned mask still
            # owns the corresponding 3-D primitive after true raster lifting.
            if float(target_member[index, anchor]) < float(args.inside_threshold):
                rejected.append({
                    "target_frame": target_frame,
                    "target_mask_index": int(index),
                    "source": list(identity),
                    "reason": "target_alpha_adjoint_does_not_confirm_anchor",
                    "target_anchor_membership": float(target_member[index, anchor]),
                })
                continue
            virtual_membership = torch.maximum(source.membership, target_member[index])
            virtual_observed = source.observed | target_observed
            radius = robust_mask_physical_radius(
                radius_xyz, virtual_membership,
                inside_threshold=float(args.inside_threshold),
                minimum_primitives=int(args.minimum_primitives_per_mask),
                device=vote_device,
            )
            confidence = min(float(source.quality), float(quality[index]))
            stable = min(float(source.stability), float(stability[index]))
            if not torch.isfinite(torch.tensor(radius)) or radius <= 0:
                rejected.append({"target_frame": target_frame, "target_mask_index": int(index), "source": list(identity), "reason": "no_metric_union_support"})
                continue
            if stable < float(args.minimum_stability):
                rejected.append({"target_frame": target_frame, "target_mask_index": int(index), "source": list(identity), "reason": "stability_below_fixed_minimum"})
                continue
            membership_rows.append(virtual_membership[None])
            observed_rows.append(virtual_observed)
            radius_rows.append(torch.tensor([radius], dtype=torch.float32))
            quality_rows.append(torch.tensor([confidence], dtype=torch.float32))
            stability_rows.append(torch.tensor([stable], dtype=torch.float32))
            audit_records.append({
                "mask_frame": f"confirmed_{target_frame}_{index}",
                "source_mask_indices": torch.tensor([len(audit_records)], dtype=torch.int32),
                "membership": virtual_membership[None].half().cpu(),
                "observed": virtual_observed.bool().cpu(),
                "physical_radius_m": torch.tensor([radius], dtype=torch.float32),
                "quality": torch.tensor([confidence], dtype=torch.float32),
                "stability": torch.tensor([stable], dtype=torch.float32),
                "source_frame": int(source.frame_id),
                "source_mask_index": int(source.source_mask_index),
                "target_frame": int(target_frame),
                "target_mask_index": int(index),
                "anchor_local_primitive": anchor,
                "target_anchor_membership": float(target_member[index, anchor]),
            })
            used.append(f"{target_frame}:{index}<-{source.frame_id}:{source.source_mask_index}")
    if not membership_rows:
        raise RuntimeError("no target SAM3 masks survived true alpha-adjoint MPR confirmation")

    # A virtual track supplies only positive connectivity.  Its unobserved
    # exterior is intentionally not turned into a false cross-view negative.
    (
        batched_memberships,
        batched_observations,
        batched_radii,
        batched_qualities,
        batched_stabilities,
    ) = _batch_virtual_tracks(
        membership_rows, observed_rows, radius_rows, quality_rows, stability_rows,
        batch_size=int(args.track_batch_size),
    )
    votes = accumulate_scale_ordered_votes(
        batched_memberships, batched_observations, batched_radii, edge, bins,
        quality=batched_qualities, stability=batched_stabilities,
        inside_threshold=float(args.inside_threshold),
        outside_threshold=float(args.outside_threshold),
        mask_chunk=int(args.mask_chunk), include_separate=False,
        device=vote_device,
    )
    intervals = merge_scale_intervals(votes)
    vote_dtype = {"float16": torch.float16, "float32": torch.float32}[args.vote_storage_dtype]
    vote_storage = "fp32_soft_same_only_mass_no_overwrite" if vote_dtype == torch.float32 else "fp16_soft_same_only_mass_no_overwrite"
    scene_name = str(graph.get("scene", "") or graph_path.parent.name)
    adjoint_provenance = _adjoint_provenance(
        responsibility_metadata=responsibility_metadata,
        adjoint_context=adjoint_context,
        channel_chunk_size=int(args.adjoint_channel_chunk_size),
    )
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "scene": scene_name,
        "edge_rows": edge_rows,
        "edge_index": edge,
        "features": edge_relation_features(graph)[unique],
        "same_votes": votes["same_votes"].to(vote_dtype),
        "separate_votes": votes["separate_votes"].to(vote_dtype),
        "observed_votes": votes["observed_votes"].to(vote_dtype),
        "same_events": votes["same_events"],
        "separate_events": votes["separate_events"],
        "scale_bin_edges_log": votes["scale_bin_edges_log"],
        "merge_log_radius": intervals["merge_log_radius"].float(),
        "lower_log_radius": intervals["lower_log_radius"].float(),
        "upper_log_radius": intervals["upper_log_radius"].float(),
        "has_lower": intervals["has_lower"],
        "has_upper": intervals["has_upper"],
        "interval_consistent": intervals["interval_consistent"],
        "constraint_entropy": intervals["constraint_entropy"].to(vote_dtype),
        "same_mass": intervals["same_mass"].to(vote_dtype),
        "separate_mass": intervals["separate_mass"].to(vote_dtype),
        "metadata": {
            "teacher": "official_sam3_mpr_confirmed_source_target_track_same_only",
            "query_free": True,
            "labels_opened": False,
            "instances_opened": False,
            "text_opened": False,
            "scene_graph": str(graph_path),
            "scene_graph_sha256": _sha256_file(graph_path),
            "membership_lifting": "raster_adjoint",
            "raster_responsibility_used": True,
            "raster_lifting_semantics": "true_alpha_compositing_adjoint",
            "responsibility_cache": str(Path(args.responsibility_cache).resolve()),
            "responsibility_metadata": responsibility_metadata,
            "raster_adjoint_provenance": adjoint_provenance,
            "mask_raster_alignment": "nearest_label_resample_to_frozen_mpr_raster",
            "source_membership_sidecars": [str(Path(value).resolve()) for value in args.source_membership_sidecars],
            "prompted_mask_roots": prompted_roots,
            "target_shard_index": int(args.target_shard_index),
            "target_shard_count": int(args.target_shard_count),
            "target_frames_available_before_sharding": int(len(all_prompted_paths)),
            "mask_roots": prompted_roots,
            "mask_schema_versions": [1],
            "resized_mask_frames": sorted(resized_target_frames, key=lambda value: int(Path(value).stem)),
            "resized_mask_count": int(len(resized_target_frames)),
            "confirmed_track_union": "max_source_and_independently_prompted_target_alpha_adjoint_membership",
            "confirmed_track_exterior": "not_used_same_only_positive_constraints",
            "anchor_confirmation": "target_true_alpha_adjoint_membership_at_exact_mpr_prompt_primitive_ge_inside_threshold",
            "confirmed_track_batch_size": int(args.track_batch_size),
            "vote_compute_device": str(vote_device),
            "target_membership_materialization": {
                "reused_frozen_automatic_alpha_adjoint_masks": int(reused_target_memberships),
                "renderer_lifted_new_target_masks": int(rerendered_target_memberships),
            },
            "mask_frames": used,
            "skipped_mask_frames": rejected,
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
    torch.save(payload, output)
    if args.save_membership_sidecar:
        sidecar = Path(args.save_membership_sidecar).resolve(); sidecar.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema_version": 1,
            "records": audit_records,
            "metadata": {
                "query_free": True,
                "labels_opened": False,
                "instances_opened": False,
                "text_opened": False,
                "offline_teacher_audit_only": True,
                "not_an_inference_representation": True,
                "source_relation_cache": str(output),
                "scene_graph": str(graph_path),
                "scene_graph_sha256": _sha256_file(graph_path),
                "membership_lifting": "raster_adjoint",
                "raster_lifting_semantics": "true_alpha_compositing_adjoint",
                "membership_aggregation": "MPR_confirmed_source_target_union",
                "inside_threshold": float(args.inside_threshold),
                "outside_threshold": float(args.outside_threshold),
                "membership_storage": "fp16_full_relation_graph_rows_offline_audit_only",
            },
        }, sidecar)
    constrained = (intervals["same_mass"] + intervals["separate_mass"]) > 0
    report = {
        "output": str(output),
        "confirmed_tracks": len(membership_rows),
        "accepted_target_masks": len(used),
        "rejected": len(rejected),
        "edges": int(edge.shape[1]),
        "constrained_edges": int(constrained.sum()),
        "constrained_edge_fraction": float(constrained.float().mean()),
        "same_events": [int(value) for value in votes["same_events"]],
        "separate_events": [int(value) for value in votes["separate_events"]],
        "same_only": True,
        "track_batch_size": int(args.track_batch_size),
        "vote_device": str(vote_device),
        "target_shard_index": int(args.target_shard_index),
        "target_shard_count": int(args.target_shard_count),
        "reused_target_memberships": int(reused_target_memberships),
        "rerendered_target_memberships": int(rerendered_target_memberships),
        "membership_sidecar": str(Path(args.save_membership_sidecar).resolve()) if args.save_membership_sidecar else "",
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-graph", required=True)
    parser.add_argument("--source-membership-sidecars", nargs="+", required=True)
    parser.add_argument("--prompted-mask-roots", nargs="+", required=True)
    parser.add_argument("--responsibility-cache", required=True)
    parser.add_argument(
        "--adjoint-config", "--config", dest="adjoint_config", default="",
        help="Frozen field config; defaults to the responsibility-sidecar contract.",
    )
    parser.add_argument(
        "--adjoint-checkpoint", "--checkpoint", dest="adjoint_checkpoint", default="",
        help="Frozen field checkpoint; defaults to the responsibility-sidecar contract.",
    )
    parser.add_argument("--adjoint-device", default="cuda")
    parser.add_argument("--adjoint-channel-chunk-size", type=int, default=32)
    parser.add_argument("--inside-threshold", type=float, default=0.80)
    parser.add_argument("--outside-threshold", type=float, default=0.20)
    parser.add_argument("--minimum-primitives-per-mask", type=int, default=3)
    parser.add_argument("--minimum-stability", type=float, default=0.0)
    parser.add_argument("--scale-bins", type=int, default=8)
    parser.add_argument("--scale-minimum-radius-m", type=float, default=0.05)
    parser.add_argument("--scale-maximum-radius-m", type=float, default=4.0)
    parser.add_argument("--mask-chunk", type=int, default=4)
    parser.add_argument(
        "--track-batch-size", type=int, default=32,
        help="Number of independent confirmed tracks packed per vote batch; semantics are unchanged.",
    )
    parser.add_argument(
        "--vote-device", default="cpu",
        help="Optional device for exact physical-scale and edge-vote accumulation (default: cpu).",
    )
    parser.add_argument("--vote-storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--target-shard-index", type=int, default=0)
    parser.add_argument("--target-shard-count", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-membership-sidecar", default="")
    args = parser.parse_args()
    if int(args.adjoint_channel_chunk_size) <= 0 or int(args.minimum_primitives_per_mask) <= 0:
        parser.error("adjoint channel chunk and minimum primitives must be positive")
    if int(args.scale_bins) <= 0 or int(args.mask_chunk) <= 0 or int(args.track_batch_size) <= 0:
        parser.error("scale bins, mask chunk, and track batch size must be positive")
    if int(args.target_shard_count) <= 0 or not 0 <= int(args.target_shard_index) < int(args.target_shard_count):
        parser.error("target shard index must lie in [0, target shard count)")
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
