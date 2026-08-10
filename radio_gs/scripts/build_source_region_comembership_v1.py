#!/usr/bin/env python3
"""Build one SHA-bound source-only AcceptedV2 region co-membership authority."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.interfaces.surface_region_typed_context_adaptive import (
    validate_adaptive_typed_context_authority,
)
from radio_gs.models.region_comembership_v1 import PAIR_FEATURE_NAMES
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.source_region_comembership_authority.v1"
SCHEMA_VERSION = 1
DESCRIPTOR_NEIGHBORS = 16
CENTROID_NEIGHBORS = 16
INSTANCE_KEY_STRIDE = 65536
PREREGISTRATION = Path(
    "paper/artifacts/source_only_region_comembership_v1_preregistration_20260807.json"
)
TRAIN_SCENES = {
    "scene0001_00",
    "scene0002_00",
    "scene0003_00",
    "scene0005_00",
}
VALIDATION_SCENES = {"scene0004_00", "scene0008_00"}


def source_access() -> dict[str, bool]:
    return {
        "source_instance_labels_opened": True,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "target_heldout_opened": False,
        "target_metrics_computed": False,
    }


def _load_instance_members(path: Path, expected_sha256: str) -> dict[int, str]:
    if sha256_file(path) != str(expected_sha256):
        raise ValueError("source instance ZIP SHA-256 differs")
    result: dict[int, str] = {}
    with ZipFile(path) as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if info.is_dir():
                continue
            if (
                member.is_absolute()
                or ".." in member.parts
                or member.suffix.lower() != ".png"
            ):
                raise ValueError("source instance ZIP member differs")
            try:
                frame = int(member.stem)
            except ValueError as exc:
                raise ValueError("source instance frame name differs") from exc
            if frame in result:
                raise ValueError("source instance ZIP repeats a frame")
            result[frame] = info.filename
    if not result:
        raise ValueError("source instance ZIP is empty")
    return result


def _instance_raster(
    archive: ZipFile,
    member: str,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    with archive.open(member) as source:
        image = Image.open(source)
        image.load()
    if image.mode not in {"L", "I", "I;16"}:
        raise ValueError("source instance PNG must be one-channel integer")
    resized = image.resize((width, height), resample=Image.Resampling.NEAREST)
    values = np.asarray(resized, dtype=np.int64)
    if values.shape != (height, width) or np.any(values < 0):
        raise ValueError("source instance raster differs")
    return torch.from_numpy(values.copy()).long().reshape(-1)


def _exact_hit_instance_mass(
    *,
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    base_weights: torch.Tensor,
    pixel_instance_ids: torch.Tensor,
    num_gaussians: int,
    num_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gaussian = torch.as_tensor(gaussian_ids).detach().long().cpu()
    pixel = torch.as_tensor(pixel_ids).detach().long().cpu()
    base = torch.as_tensor(base_weights).detach().float().cpu()
    labels = torch.as_tensor(pixel_instance_ids).detach().long().cpu()
    if (
        gaussian.ndim != 1
        or pixel.shape != gaussian.shape
        or base.shape != gaussian.shape
        or labels.shape != (int(num_pixels),)
        or bool((gaussian < 0).any())
        or bool((gaussian >= int(num_gaussians)).any())
        or bool((pixel < 0).any())
        or bool((pixel >= int(num_pixels)).any())
        or bool((base < 0).any())
        or not bool(torch.isfinite(base).all())
    ):
        raise ValueError("exact marginal hit tensors differ")
    pixel_mass = torch.zeros(int(num_pixels), dtype=torch.float64)
    pixel_mass.index_add_(0, pixel, base.double())
    denominator = pixel_mass[pixel]
    if bool((denominator <= 0).any()):
        raise ValueError("exact marginal hit has nonpositive pixel mass")
    exact_weight = base.double().square() / denominator
    instance = labels[pixel]
    if bool((instance >= INSTANCE_KEY_STRIDE).any()):
        raise ValueError("source instance ID exceeds the frozen key stride")
    key = gaussian * INSTANCE_KEY_STRIDE + instance
    unique, inverse = torch.unique(key, sorted=True, return_inverse=True)
    mass = torch.zeros(unique.numel(), dtype=torch.float64)
    mass.index_add_(0, inverse, exact_weight)
    return unique, mass


def _load_exact_instance_mass(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    instance_zip: Path,
    instance_zip_sha256: str,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, str]]:
    manifest, digest, source = load_json_object(
        manifest_path,
        expected_sha256=manifest_sha256,
        label="exact marginal responsibility authority",
    )
    metadata = manifest.get("metadata")
    views = manifest.get("views")
    if (
        manifest.get("formula_contract", {}).get("name")
        != "sparse_exact_marginal_responsibility_authority_v1"
        or not isinstance(metadata, Mapping)
        or metadata.get("assignment_mode") != "exact_front_to_back_sparse_marginal"
        or metadata.get("benchmark_masks_opened") is not False
        or not isinstance(views, list)
        or not views
    ):
        raise ValueError("exact marginal responsibility contract differs")
    height = int(metadata["feature_height"])
    width = int(metadata["feature_width"])
    num_gaussians = int(manifest["num_gaussians"])
    if height <= 0 or width <= 0 or num_gaussians <= 0:
        raise ValueError("exact marginal dimensions differ")
    members = _load_instance_members(instance_zip, instance_zip_sha256)
    frame_ids = [int(item["frame_index"]) for item in views]
    if len(set(frame_ids)) != len(frame_ids) or not set(frame_ids).issubset(members):
        raise ValueError("exact marginal frames are absent from source instances")
    all_keys: list[torch.Tensor] = []
    all_mass: list[torch.Tensor] = []
    with ZipFile(instance_zip) as archive:
        for expected_view, record in enumerate(views):
            view_path = source.parent / str(record["relative_path"])
            payload, view_digest, _ = load_torch_mapping(
                view_path,
                expected_sha256=str(record["sha256"]),
                map_location="cpu",
                label="exact marginal responsibility view",
            )
            frame = int(record["frame_index"])
            if (
                payload.get("schema")
                != "radio_gs.sparse_exact_marginal_responsibility_view.v1"
                or int(payload.get("view_index", -1)) != expected_view
                or int(payload.get("frame_index", -1)) != frame
                or int(payload.get("num_gaussians", -1)) != num_gaussians
                or int(payload.get("num_pixels", -1)) != height * width
                or view_digest != str(record["sha256"])
            ):
                raise ValueError("exact marginal view identity differs")
            raster = _instance_raster(
                archive, members[frame], height=height, width=width
            )
            keys, mass = _exact_hit_instance_mass(
                gaussian_ids=payload["gaussian_ids"],
                pixel_ids=payload["pixel_ids"],
                base_weights=payload["base_weights"],
                pixel_instance_ids=raster,
                num_gaussians=num_gaussians,
                num_pixels=height * width,
            )
            all_keys.append(keys)
            all_mass.append(mass)
    keys = torch.cat(all_keys)
    masses = torch.cat(all_mass)
    unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    reduced = torch.zeros(unique.numel(), dtype=torch.float64)
    reduced.index_add_(0, inverse, masses)
    maximum_instance = int((unique % INSTANCE_KEY_STRIDE).max())
    dense = torch.zeros(num_gaussians, maximum_instance + 1, dtype=torch.float32)
    dense[unique // INSTANCE_KEY_STRIDE, unique % INSTANCE_KEY_STRIDE] = reduced.float()
    audit = {
        "views": len(views),
        "feature_height": height,
        "feature_width": width,
        "num_gaussians": num_gaussians,
        "maximum_instance_id": maximum_instance,
        "positive_instance_ids": int(
            torch.unique(unique % INSTANCE_KEY_STRIDE).gt(0).sum()
        ),
        "nonzero_primitive_instance_cells": int(unique.numel()),
        "exact_visible_mass": float(reduced.sum()),
    }
    return dense, audit, {"path": str(source), "sha256": digest}


def _region_instance_statistics(
    dense_mass: torch.Tensor,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    rows = torch.as_tensor(region_rows).detach().long().cpu()
    mask = torch.as_tensor(token_mask).detach().bool().cpu()
    if rows.ndim != 2 or mask.shape != rows.shape or dense_mass.ndim != 2:
        raise ValueError("region instance aggregation dimensions differ")
    region_mass = torch.zeros(rows.shape[0], dense_mass.shape[1])
    for start in range(0, rows.shape[0], 128):
        stop = min(start + 128, rows.shape[0])
        safe = rows[start:stop].clamp_min(0)
        region_mass[start:stop] = (dense_mass[safe] * mask[start:stop, :, None]).sum(
            dim=1
        )
    all_mass = region_mass.sum(dim=1)
    positive = region_mass[:, 1:]
    positive_mass = positive.sum(dim=1)
    dominant_mass, dominant_zero = positive.max(dim=1)
    observed = positive_mass > 0
    dominant = torch.where(observed, dominant_zero + 1, -torch.ones_like(dominant_zero))
    purity = torch.where(
        observed,
        dominant_mass / positive_mass.clamp_min(1e-12),
        torch.zeros_like(positive_mass),
    )
    coverage = torch.where(
        all_mass > 0,
        positive_mass / all_mass.clamp_min(1e-12),
        torch.zeros_like(all_mass),
    )
    return {
        "dominant_instance_ids": dominant.long().contiguous(),
        "dominant_instance_mass": dominant_mass.float().contiguous(),
        "positive_instance_mass": positive_mass.float().contiguous(),
        "all_visible_mass": all_mass.float().contiguous(),
        "instance_purity": purity.float().contiguous(),
        "instance_label_coverage": coverage.float().contiguous(),
        "instance_observed": observed.bool().contiguous(),
    }


def _knn_pairs(
    values: torch.Tensor, *, neighbors: int, cosine: bool
) -> set[tuple[int, int]]:
    matrix = torch.as_tensor(values).detach().float().cpu()
    if (
        matrix.ndim != 2
        or matrix.shape[0] < 2
        or not bool(torch.isfinite(matrix).all())
    ):
        raise ValueError("co-membership KNN values differ")
    if cosine:
        matrix = F.normalize(matrix, dim=-1)
    count = matrix.shape[0]
    k = min(int(neighbors), count - 1)
    result: set[tuple[int, int]] = set()
    for start in range(0, count, 256):
        stop = min(start + 256, count)
        if cosine:
            score = matrix[start:stop] @ matrix.T
            score[torch.arange(stop - start), torch.arange(start, stop)] = -torch.inf
            selected = score.topk(k, dim=1, largest=True).indices
        else:
            score = torch.cdist(matrix[start:stop], matrix)
            score[torch.arange(stop - start), torch.arange(start, stop)] = torch.inf
            selected = score.topk(k, dim=1, largest=False).indices
        for offset, targets in enumerate(selected.tolist()):
            left = start + offset
            for right in targets:
                result.add((min(left, right), max(left, right)))
    return result


def _anchor_support_pairs(
    graph: Mapping[str, Any], anchor_global_rows: torch.Tensor
) -> dict[tuple[int, int], float]:
    global_rows = torch.as_tensor(graph["global_rows"]).long().cpu()
    edge = torch.as_tensor(graph["edge_index"]).long().cpu()
    affinity = torch.as_tensor(graph["raw_affinity"]).float().cpu()
    anchors = torch.as_tensor(anchor_global_rows).long().cpu()
    by_global: dict[int, list[int]] = {}
    for region, row in enumerate(anchors.tolist()):
        by_global.setdefault(row, []).append(region)
    global_to_local = {row: index for index, row in enumerate(global_rows.tolist())}
    anchor_locals = {
        global_to_local[row]: regions
        for row, regions in by_global.items()
        if row in global_to_local
    }
    wanted = torch.zeros(global_rows.numel(), dtype=torch.bool)
    wanted[list(anchor_locals)] = True
    selected = wanted[edge[0]] & wanted[edge[1]] & (edge[0] != edge[1])
    result: dict[tuple[int, int], float] = {}
    for local_left, local_right, value in zip(
        edge[0, selected].tolist(),
        edge[1, selected].tolist(),
        affinity[selected].tolist(),
    ):
        for left in anchor_locals[local_left]:
            for right in anchor_locals[local_right]:
                if left == right:
                    continue
                pair = (min(left, right), max(left, right))
                result[pair] = max(result.get(pair, 0.0), float(value))
    return result


def _full_state_channels(state: Any) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(state.xyz.shape[0])
    observation = torch.zeros(count)
    purity = torch.zeros(count)
    observation[state.global_rows] = state.observation_evidence.float()
    known_purity = torch.where(
        state.visibility_purity_known,
        state.visibility_purity_value,
        torch.zeros_like(state.visibility_purity_value),
    )
    purity[state.global_rows] = known_purity.float()
    return observation, purity


def _region_mean(
    values: torch.Tensor, rows: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    safe = rows.clamp_min(0)
    selected = values[safe]
    weights = mask.float()
    while weights.ndim < selected.ndim:
        weights = weights.unsqueeze(-1)
    return (selected * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def _pair_features(
    *,
    pair_indices: torch.Tensor,
    accepted: Mapping[str, Any],
    context: Mapping[str, Any],
    state: Any,
    support: Mapping[tuple[int, int], float],
) -> torch.Tensor:
    pairs = torch.as_tensor(pair_indices).long().cpu()
    left, right = pairs
    descriptor = F.normalize(accepted["accepted_v2_e0"].float(), dim=-1)
    context_direction = context["pooled_context_radio_direction"].float()
    context_norm = torch.linalg.vector_norm(context_direction, dim=-1, keepdim=True)
    context_unit = torch.where(
        context_norm > 0,
        context_direction / context_norm.clamp_min(1e-12),
        torch.zeros_like(context_direction),
    )
    statistics = context["typed_context_statistics"].float()
    rows = accepted["region_rows"].long()
    mask = accepted["token_mask"].bool()
    centroid = _region_mean(state.xyz.float(), rows, mask)
    observation, visibility = _full_state_channels(state)
    region_observation = _region_mean(observation[:, None], rows, mask).squeeze(-1)
    region_visibility = _region_mean(visibility[:, None], rows, mask).squeeze(-1)
    log_radius = statistics[:, 0]
    radius = log_radius.exp()
    delta_statistics = (statistics[left] - statistics[right]).abs()
    distance = torch.linalg.vector_norm(centroid[left] - centroid[right], dim=-1)
    overlap = torch.zeros(pairs.shape[1])
    core_sets = [set(row[valid].tolist()) for row, valid in zip(rows, mask)]
    for index, (a, b) in enumerate(pairs.T.tolist()):
        overlap[index] = len(core_sets[a] & core_sets[b]) / max(
            1, min(len(core_sets[a]), len(core_sets[b]))
        )
    adjacency = torch.tensor(
        [1.0 if tuple(pair) in support else 0.0 for pair in pairs.T.tolist()]
    )
    affinity = torch.tensor(
        [support.get(tuple(pair), 0.0) for pair in pairs.T.tolist()]
    )
    features = torch.stack(
        (
            (descriptor[left] * descriptor[right]).sum(dim=-1),
            (context_unit[left] * context_unit[right]).sum(dim=-1),
            delta_statistics.mean(dim=-1),
            delta_statistics.amax(dim=-1),
            overlap,
            adjacency,
            affinity,
            distance,
            distance / torch.maximum(radius[left], radius[right]).clamp_min(1e-6),
            (log_radius[left] - log_radius[right]).abs(),
            (
                accepted["scale_indices"][left] == accepted["scale_indices"][right]
            ).float(),
            torch.minimum(region_observation[left], region_observation[right]),
            torch.minimum(region_visibility[left], region_visibility[right]),
            (region_observation[left] - region_observation[right]).abs(),
            (region_visibility[left] - region_visibility[right]).abs(),
        ),
        dim=1,
    ).float()
    if features.shape[1] != len(PAIR_FEATURE_NAMES) or not bool(
        torch.isfinite(features).all()
    ):
        raise RuntimeError("co-membership pair feature construction failed")
    return features.contiguous()


def build_query_independent_pair_features(
    *,
    accepted: Mapping[str, Any],
    context: Mapping[str, Any],
    state: Any,
    graph: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Build the identical target-safe candidate graph without instance labels."""

    rows = torch.as_tensor(accepted["region_rows"]).long().cpu()
    mask = torch.as_tensor(accepted["token_mask"]).bool().cpu()
    centroid = _region_mean(state.xyz, rows, mask)
    pair_set = _knn_pairs(
        accepted["accepted_v2_e0"],
        neighbors=DESCRIPTOR_NEIGHBORS,
        cosine=True,
    )
    pair_set |= _knn_pairs(centroid, neighbors=CENTROID_NEIGHBORS, cosine=False)
    anchor_rows = rows[torch.arange(rows.shape[0]), accepted["anchor_index"].long()]
    support = _anchor_support_pairs(graph, anchor_rows)
    pair_set |= set(support)
    pair_indices = torch.tensor(sorted(pair_set), dtype=torch.int64).T.contiguous()
    features = _pair_features(
        pair_indices=pair_indices,
        accepted=accepted,
        context=context,
        state=state,
        support=support,
    )
    return (
        pair_indices,
        features,
        {
            "candidate_pairs": int(pair_indices.shape[1]),
            "anchor_support_pairs": len(support),
        },
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"co-membership authority already exists: {output}")
    accepted_raw, accepted_sha, accepted_path = load_torch_mapping(
        args.accepted_v2,
        expected_sha256=args.expected_accepted_v2_sha256,
        map_location="cpu",
        label="co-membership AcceptedV2 authority",
    )
    accepted = shard.validate_accepted_region_authority(accepted_raw)
    context_raw, context_sha, context_path = load_torch_mapping(
        args.typed_context,
        expected_sha256=args.expected_typed_context_sha256,
        map_location="cpu",
        label="co-membership typed-context authority",
    )
    context = validate_adaptive_typed_context_authority(context_raw)
    graph, graph_sha, graph_path = load_torch_mapping(
        args.support_graph,
        expected_sha256=args.expected_support_graph_sha256,
        map_location="cpu",
        label="co-membership support graph",
    )
    state = load_factorized_primitive_state(
        args.factorized_state,
        expected_sha256=args.expected_factorized_state_sha256,
    )
    scene_id = str(args.scene_id)
    expected_scenes = (
        TRAIN_SCENES if args.split == "source_train" else VALIDATION_SCENES
    )
    if (
        scene_id not in expected_scenes
        or accepted["scene_id"] != scene_id
        or context["scene_id"] != scene_id
        or context["region_row_ids"]
        != [
            f"{scene_id}:accepted-v2-canonical-v1:{value}"
            for value in accepted["region_fingerprints"]
        ]
        or not torch.equal(
            accepted["canonical_region_indices"],
            context["canonical_region_indices"],
        )
        or not torch.equal(
            torch.as_tensor(graph["global_rows"]).long(), state.global_rows
        )
        or not torch.equal(
            torch.as_tensor(graph["xyz"]).float(), state.xyz[state.global_rows]
        )
    ):
        raise ValueError("co-membership scene/region/geometry authorities differ")
    if (
        accepted["input_authority"]["support_graph_authority"][
            "support_graph_file_sha256"
        ]
        != graph_sha
        or accepted["input_authority"]["geometry_authority"][
            "factorized_primitive_state_file_sha256"
        ]
        != state.sha256
    ):
        raise ValueError("AcceptedV2 does not bind the supplied graph/state")
    dense_mass, instance_audit, marginal_record = _load_exact_instance_mass(
        manifest_path=Path(args.exact_marginal_authority),
        manifest_sha256=args.expected_exact_marginal_authority_sha256,
        instance_zip=Path(args.instance_zip),
        instance_zip_sha256=args.expected_instance_zip_sha256,
    )
    if dense_mass.shape[0] != state.xyz.shape[0]:
        raise ValueError("instance evidence and geometry row counts differ")
    region_instance = _region_instance_statistics(
        dense_mass, accepted["region_rows"], accepted["token_mask"]
    )
    rows = accepted["region_rows"]
    pair_indices, features, pair_audit = build_query_independent_pair_features(
        accepted=accepted,
        context=context,
        state=state,
        graph=graph,
    )
    left, right = pair_indices
    dominant = region_instance["dominant_instance_ids"]
    target = (
        (dominant[left] > 0)
        & (dominant[right] > 0)
        & (dominant[left] == dominant[right])
    )
    purity = region_instance["instance_purity"]
    coverage = region_instance["instance_label_coverage"]
    target_weight = (
        purity[left]
        * purity[right]
        * torch.sqrt((coverage[left] * coverage[right]).clamp_min(0))
    ).float()
    identity = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "split": str(args.split),
        "producer": file_record(Path(__file__).resolve()),
        "preregistration": file_record(
            Path(__file__).resolve().parents[2] / PREREGISTRATION
        ),
        "input_authority": {
            "instance_zip": {
                "path": str(Path(args.instance_zip).resolve()),
                "sha256": str(args.expected_instance_zip_sha256),
            },
            "exact_marginal": marginal_record,
            "accepted_v2": {"path": str(accepted_path), "sha256": accepted_sha},
            "typed_context": {"path": str(context_path), "sha256": context_sha},
            "support_graph": {"path": str(graph_path), "sha256": graph_sha},
            "factorized_state": {
                "path": str(state.source),
                "sha256": state.sha256,
            },
        },
        "candidate_policy": {
            "descriptor_neighbors": DESCRIPTOR_NEIGHBORS,
            "centroid_neighbors": CENTROID_NEIGHBORS,
            "anchor_support_edges": True,
        },
        "feature_names": list(PAIR_FEATURE_NAMES),
        "feature_names_sha256": canonical_json_sha256(list(PAIR_FEATURE_NAMES)),
        "target_semantics": {
            "positive": "equal_positive_dominant_instance_id",
            "weight": "purity_product_times_geometric_mean_label_coverage",
        },
        "source_access": source_access(),
    }
    payload = {
        **identity,
        "content_authority_sha256": canonical_json_sha256(identity),
        "region_fingerprints": list(accepted["region_fingerprints"]),
        "canonical_region_indices": accepted["canonical_region_indices"],
        **region_instance,
        "pair_indices": pair_indices,
        "pair_features": features,
        "same_instance_targets": target.bool().contiguous(),
        "pair_evidence_weights": target_weight.contiguous(),
        "channel_sha256": {},
        "audit": {
            **instance_audit,
            "canonical_regions": int(rows.shape[0]),
            "instance_observed_regions": int(
                region_instance["instance_observed"].sum()
            ),
            "candidate_pairs": pair_audit["candidate_pairs"],
            "positive_pairs": int(target.sum()),
            "active_evidence_pairs": int((target_weight > 0).sum()),
            "anchor_support_pairs": pair_audit["anchor_support_pairs"],
        },
    }
    channels = (
        "canonical_region_indices",
        "dominant_instance_ids",
        "dominant_instance_mass",
        "positive_instance_mass",
        "all_visible_mass",
        "instance_purity",
        "instance_label_coverage",
        "instance_observed",
        "pair_indices",
        "pair_features",
        "same_instance_targets",
        "pair_evidence_weights",
    )
    payload["channel_sha256"] = {
        name: tensor_sha256(payload[name]) for name in channels
    }
    written = write_torch_noclobber(output, payload)
    return {
        "status": "source_region_comembership_authority_complete",
        "scene_id": scene_id,
        "output": file_record(written),
        "audit": payload["audit"],
        "benchmark_opened": False,
    }


def synthetic_dry_run() -> dict[str, Any]:
    keys, mass = _exact_hit_instance_mass(
        gaussian_ids=torch.tensor([0, 1, 0, 2]),
        pixel_ids=torch.tensor([0, 0, 1, 1]),
        base_weights=torch.tensor([0.6, 0.4, 0.5, 0.5]),
        pixel_instance_ids=torch.tensor([3, 4]),
        num_gaussians=3,
        num_pixels=2,
    )
    dense = torch.zeros(3, 5)
    dense[keys // INSTANCE_KEY_STRIDE, keys % INSTANCE_KEY_STRIDE] = mass.float()
    region = _region_instance_statistics(
        dense,
        torch.tensor([[0, 1], [0, 2]]),
        torch.ones(2, 2, dtype=torch.bool),
    )
    return {
        "schema": "radio_gs.source_region_comembership_v1_synthetic_dry_run.v1",
        "primitive_instance_cells": int(keys.numel()),
        "dominant_instance_ids": region["dominant_instance_ids"].tolist(),
        "instance_purity": region["instance_purity"].tolist(),
        "benchmark_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic-dry-run")
    build_command = commands.add_parser("build")
    build_command.add_argument("--scene-id", required=True)
    build_command.add_argument(
        "--split", choices=("source_train", "source_validation"), required=True
    )
    for name in (
        "instance-zip",
        "exact-marginal-authority",
        "accepted-v2",
        "typed-context",
        "support-graph",
        "factorized-state",
    ):
        build_command.add_argument(f"--{name}", required=True)
        build_command.add_argument(f"--expected-{name}-sha256", required=True)
    build_command.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = synthetic_dry_run() if args.command == "synthetic-dry-run" else build(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
