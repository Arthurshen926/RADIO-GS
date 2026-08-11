#!/usr/bin/env python3
"""Build capability-prototype likelihood data from authorized source clicks.

Spatial click kernels are retained as hard solver seeds but never used as the
instance-propagation likelihood.  Appearance and boundary likelihood channels
come from query-independent scene-centered capability cosine similarities to
the local Gaussian mixture observed at each click.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.querying.query_compilers import world_point_soft_seed_matrix
from radio_gs.querying.query_likelihood_head import QueryLikelihoodInputs

from .build_likelihood_training_dataset import (
    _authorized_source_scene,
    _load_json,
    _read_official_geometry,
    _read_official_labels,
    _write_json_no_clobber,
    _write_torch_no_clobber,
    sha256_file,
    synthesize_click_trajectory,
    verify_scene_split_sources,
)
from .protocol import quantize_scannet_points


SHARD_SCHEMA_V2 = "agile3d-capability-query-likelihood-training-shard-v2"
DATASET_SCHEMA_V2 = "agile3d-capability-query-likelihood-training-dataset-v2"
CAPABILITY_CHANNELS = ("appearance", "boundary")
AFFINITY_CALIBRATION = "scene_centered_l2_cosine_to_click_gaussian_mixture_v1"


def _tensor_bytes_sha256(value: torch.Tensor, *, rows_per_chunk: int = 4096) -> str:
    tensor = torch.as_tensor(value).detach().cpu()
    digest = hashlib.sha256()
    if tensor.ndim == 0:
        digest.update(tensor.contiguous().numpy().tobytes())
        return digest.hexdigest()
    for start in range(0, len(tensor), int(rows_per_chunk)):
        digest.update(
            tensor[start : start + int(rows_per_chunk)]
            .contiguous()
            .numpy()
            .tobytes(order="C")
        )
    return digest.hexdigest()


def _load_query_independent_bundle(path: Path) -> dict[str, torch.Tensor | object]:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("artifact_type") != "agile3d-canonical-gaussian-primitive-bundle-v1":
        raise ValueError("capability likelihood requires a canonical Gaussian bundle")
    safety = bundle.get("safety", {})
    required_safety = {
        "query_independent": True,
        "object_id_used": False,
        "clicks_opened": False,
        "gt_labels_opened": False,
        "test_labels_opened": False,
        "point_as_primitive_used": False,
    }
    for key, expected in required_safety.items():
        if safety.get(key) is not expected:
            raise ValueError(f"primitive bundle violates query-independent safety: {key}")
    required = {
        "primitive_xyz",
        "primitive_covariance",
        "primitive_opacity",
        "appearance",
        "boundary",
        "prior_probability",
        "coverage",
        "reliability",
        "official_point_xyz",
        "point_candidate_indices",
        "primitive_to_point_index",
    }
    if not required <= set(bundle):
        raise ValueError(f"primitive bundle misses {sorted(required - set(bundle))}")
    return bundle


def click_gaussian_mixture_weights(
    *,
    primitive_xyz: torch.Tensor,
    primitive_covariance: torch.Tensor,
    primitive_opacity: torch.Tensor,
    click_xyz: torch.Tensor,
    click_candidate_indices: torch.Tensor,
) -> torch.Tensor:
    xyz = torch.as_tensor(primitive_xyz).float()
    covariance = torch.as_tensor(primitive_covariance).float()
    opacity = torch.as_tensor(primitive_opacity).float().reshape(-1)
    points = torch.as_tensor(click_xyz).float()
    candidates = torch.as_tensor(click_candidate_indices).long()
    if candidates.ndim != 2 or points.shape != (candidates.shape[0], 3):
        raise ValueError("click candidates and points must be [Q,K]/[Q,3]")
    selected_covariance = covariance[candidates]
    identity = torch.eye(3, dtype=torch.float32, device=selected_covariance.device)
    precision = torch.linalg.pinv(selected_covariance + 1e-6 * identity)
    delta = xyz[candidates] - points[:, None, :]
    mahalanobis = torch.einsum("qki,qkij,qkj->qk", delta, precision, delta)
    selected_opacity = opacity[candidates]
    if bool((selected_opacity <= 0).all(dim=1).any()):
        raise ValueError("a source click has no Gaussian mixture support")
    # Normalize in log space.  The equivalent exp-times-opacity expression
    # underflows to an all-zero row when a valid official point lies many tiny
    # Gaussian standard deviations from its K nearest centers.
    negative_infinity = torch.full_like(selected_opacity, -torch.inf)
    log_opacity = torch.where(
        selected_opacity > 0, selected_opacity.log(), negative_infinity
    )
    log_weight = -0.5 * mahalanobis + log_opacity
    if not bool(torch.isfinite(log_weight).any(dim=1).all()):
        raise ValueError("a source click has no finite Gaussian mixture support")
    return torch.softmax(log_weight, dim=1)


def scene_centered_capability_affinity(
    features: torch.Tensor,
    *,
    click_candidate_indices: torch.Tensor,
    click_mixture_weights: torch.Tensor,
    chunk_size: int = 2048,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return label/query-threshold-free global similarity for each click."""

    bank = torch.as_tensor(features).cpu()
    candidates = torch.as_tensor(click_candidate_indices).long().cpu()
    mixture = torch.as_tensor(click_mixture_weights).float().cpu()
    if bank.ndim != 2 or candidates.ndim != 2 or mixture.shape != candidates.shape:
        raise ValueError("capability bank/candidates/mixtures do not align")
    if bool((candidates < 0).any()) or bool((candidates >= len(bank)).any()):
        raise ValueError("click mixture candidates are outside capability rows")
    chunk = max(1, int(chunk_size))
    centroid = torch.zeros(bank.shape[1], dtype=torch.float64)
    for start in range(0, len(bank), chunk):
        rows = F.normalize(bank[start : start + chunk].float(), dim=1, eps=1e-8)
        centroid += rows.double().sum(dim=0)
    centroid = (centroid / len(bank)).float()
    candidate_features = F.normalize(bank[candidates].float(), dim=2, eps=1e-8)
    prototypes = (candidate_features * mixture[..., None]).sum(dim=1)
    prototypes = F.normalize(prototypes - centroid[None], dim=1, eps=1e-8)
    affinity = torch.empty((len(bank), len(prototypes)), dtype=torch.float16)
    for start in range(0, len(bank), chunk):
        stop = min(start + chunk, len(bank))
        rows = F.normalize(bank[start:stop].float(), dim=1, eps=1e-8)
        centered = F.normalize(rows - centroid[None], dim=1, eps=1e-8)
        cosine = centered @ prototypes.T
        affinity[start:stop] = ((cosine + 1.0) * 0.5).clamp(0, 1).half()
    return affinity, {
        "calibration": AFFINITY_CALIBRATION,
        "scene_centroid_sha256": _tensor_bytes_sha256(centroid),
        "scene_centroid_norm": float(centroid.norm()),
        "capability_dimension": int(bank.shape[1]),
        "primitive_rows": int(len(bank)),
        "prototype_count": int(len(prototypes)),
        "uses_labels": False,
        "uses_query_threshold": False,
        "uses_metric_feedback": False,
    }


def capability_click_affinities(
    bundle: Mapping[str, object],
    *,
    click_point_indices: torch.Tensor,
    chunk_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    points = torch.as_tensor(bundle["official_point_xyz"]).float()
    point_candidates = torch.as_tensor(bundle["point_candidate_indices"]).long()
    click_rows = torch.as_tensor(click_point_indices).long().reshape(-1)
    candidates = point_candidates.index_select(0, click_rows)
    click_xyz = points.index_select(0, click_rows)
    mixture = click_gaussian_mixture_weights(
        primitive_xyz=bundle["primitive_xyz"],
        primitive_covariance=bundle["primitive_covariance"],
        primitive_opacity=bundle["primitive_opacity"],
        click_xyz=click_xyz,
        click_candidate_indices=candidates,
    )
    channel_affinity = []
    reports = {}
    for name in CAPABILITY_CHANNELS:
        affinity, report = scene_centered_capability_affinity(
            torch.as_tensor(bundle[name]),
            click_candidate_indices=candidates,
            click_mixture_weights=mixture,
            chunk_size=chunk_size,
        )
        channel_affinity.append(affinity)
        reports[name] = report
    return (
        torch.stack(channel_affinity, dim=-1),
        mixture,
        {
            "channels": list(CAPABILITY_CHANNELS),
            "channel_order_sha256": hashlib.sha256(
                json.dumps(list(CAPABILITY_CHANNELS), separators=(",", ":")).encode()
            ).hexdigest(),
            "calibration": AFFINITY_CALIBRATION,
            "per_channel": reports,
            "click_mixture_candidate_k": int(candidates.shape[1]),
        },
    )


def iter_capability_training_examples(
    payload: Mapping[str, object],
) -> Iterator[tuple[QueryLikelihoodInputs, torch.Tensor, Mapping[str, object]]]:
    if payload.get("artifact_type") != SHARD_SCHEMA_V2:
        raise ValueError("unexpected capability likelihood training shard")
    affinity = torch.as_tensor(payload["capability_click_affinity"]).float()
    if affinity.ndim != 3 or affinity.shape[2] != len(CAPABILITY_CHANNELS):
        raise ValueError("capability affinity must be [N,K,2]")
    target = torch.as_tensor(payload["primitive_target"]).float().reshape(-1)
    for step in payload["steps"]:
        positive = torch.as_tensor(step["positive_columns"], dtype=torch.long)
        negative = torch.as_tensor(step["negative_columns"], dtype=torch.long)
        inputs = QueryLikelihoodInputs(
            positive_affinity=affinity.index_select(1, positive),
            negative_affinity=affinity.index_select(1, negative),
            prior_probability=payload["prior_probability"],
            coverage=payload["coverage"],
            reliability=payload["reliability"],
        ).validated()
        yield inputs, target, step


def build_capability_training_payload(
    *,
    scene_id: str,
    object_id: int,
    point_xyz: np.ndarray,
    point_target: np.ndarray,
    bundle: Mapping[str, object],
    max_clicks: int,
    spatial_candidate_k: int,
    affinity_chunk_size: int,
) -> dict[str, object]:
    points = np.asarray(point_xyz, dtype=np.float32)
    target = torch.as_tensor(point_target).bool().reshape(-1)
    clicks = synthesize_click_trajectory(
        points, target.numpy(), max_clicks=max_clicks, click_workers=1
    )
    click_rows = torch.tensor([click.point_index for click in clicks], dtype=torch.long)
    capability_affinity, mixture, calibration = capability_click_affinities(
        bundle,
        click_point_indices=click_rows,
        chunk_size=affinity_chunk_size,
    )
    click_xyz = torch.from_numpy(np.ascontiguousarray(points[click_rows.numpy()]))
    hard_spatial_seed = world_point_soft_seed_matrix(
        torch.as_tensor(bundle["primitive_xyz"]).float(),
        torch.as_tensor(bundle["primitive_covariance"]).float(),
        click_xyz,
        euclidean_candidate_k=int(spatial_candidate_k),
    ).half()
    mapping = torch.as_tensor(bundle["primitive_to_point_index"]).long()
    primitive_target = target.index_select(0, mapping).float()
    steps = []
    for count in range(1, len(clicks) + 1):
        prefix = clicks[:count]
        steps.append(
            {
                "click_count": count,
                "positive_columns": [i for i, click in enumerate(prefix) if click.is_positive],
                "negative_columns": [i for i, click in enumerate(prefix) if not click.is_positive],
            }
        )
    return {
        "schema_version": 2,
        "artifact_type": SHARD_SCHEMA_V2,
        "head_schema_version": "monotone-query-likelihood-multichannel-v2",
        "affinity_channels": list(CAPABILITY_CHANNELS),
        "scene_id": str(scene_id),
        "object_id": int(object_id),
        "partition": "fit",
        "adapter": "canonical_capability_prototype_v2",
        "primitive_count": int(len(mapping)),
        "point_count": int(len(points)),
        "clicks": [
            {
                "point_index": int(click.point_index),
                "is_positive": bool(click.is_positive),
                "order": int(click.order),
            }
            for click in clicks
        ],
        "steps": steps,
        "capability_click_affinity": capability_affinity,
        "hard_spatial_click_seed": hard_spatial_seed,
        "click_gaussian_mixture_weights": mixture.half(),
        "prior_probability": torch.as_tensor(bundle["prior_probability"]).half(),
        "coverage": torch.as_tensor(bundle["coverage"]).half(),
        "reliability": torch.as_tensor(bundle["reliability"]).half(),
        "primitive_to_point_index": mapping.to(torch.int32),
        "primitive_target": primitive_target,
        "point_target": target.float(),
        "calibration": calibration,
        "contracts": {
            "likelihood_evidence": "appearance_boundary_capability_prototype_only",
            "hard_spatial_seed": "solver_seed_only_excluded_from_likelihood_head",
            "coverage": "capability_valid_times_query_independent_observation_support",
            "step_weighting": "equal_in_training_objective",
        },
        "safety": {
            "labels_opened": True,
            "label_scope": "official_source_train_scene_only",
            "source_train_labels_opened": True,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "labels_used_by_scene_calibration": False,
            "labels_used_by_capability_prototypes": False,
            "target_used_by_click_simulator_only": True,
            "spatial_kernel_used_as_instance_likelihood": False,
            "threshold_tuned_per_scene": False,
        },
    }


def materialize(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    if args.scene_id != "scene0000_00" or args.partition != "fit":
        raise ValueError("v2 Stage-A is sealed to fit scene0000_00")
    split_path = Path(args.split_manifest).expanduser().resolve()
    split = verify_scene_split_sources(_load_json(split_path))
    object_id = _authorized_source_scene(split, args.scene_id, "fit")
    contract_path = Path(args.preregistration).expanduser().resolve()
    prereg = _load_json(contract_path)
    if prereg.get("artifact_type") != "agile3d-capability-likelihood-v2-preregistration":
        raise ValueError("unexpected v2 preregistration")
    ply_path = Path(args.benchmark_root).expanduser().resolve() / "scans" / f"{args.scene_id}.ply"
    xyz, colors = _read_official_geometry(ply_path)
    labels = _read_official_labels(ply_path)
    quantized = quantize_scannet_points(xyz, colors, labels, voxel_size=0.05)
    point_xyz = quantized.raw_coordinates + xyz.min(axis=0, keepdims=True)
    point_target = quantized.labels == int(object_id)
    bundle_path = Path(args.primitive_bundle).expanduser().resolve()
    bundle = _load_query_independent_bundle(bundle_path)
    official = torch.as_tensor(bundle["official_point_xyz"]).float()
    if official.shape != torch.from_numpy(point_xyz).shape or not torch.allclose(
        official, torch.from_numpy(point_xyz).float(), atol=1e-6, rtol=0
    ):
        raise ValueError("bundle and source-train official point rows differ")
    payload = build_capability_training_payload(
        scene_id=args.scene_id,
        object_id=object_id,
        point_xyz=point_xyz,
        point_target=point_target,
        bundle=bundle,
        max_clicks=args.max_clicks,
        spatial_candidate_k=args.spatial_candidate_k,
        affinity_chunk_size=args.affinity_chunk_size,
    )
    payload["source_authority"] = {
        "ply": {"path": str(ply_path), "sha256": sha256_file(ply_path)},
        "split": {"path": str(split_path), "sha256": sha256_file(split_path)},
        "preregistration": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
        },
        "primitive_bundle": {
            "path": str(bundle_path),
            "sha256": sha256_file(bundle_path),
        },
    }
    output_root = Path(args.output_root).expanduser().resolve()
    shard = _write_torch_no_clobber(
        output_root / "shards" / f"{args.scene_id}.pt", payload
    )
    receipt = {
        "schema_version": 2,
        "artifact_type": "agile3d-capability-query-likelihood-training-receipt-v2",
        "scene_id": args.scene_id,
        "partition": "fit",
        "shard": {"path": str(shard), "sha256": sha256_file(shard)},
        "primitive_count": payload["primitive_count"],
        "point_count": payload["point_count"],
        "click_steps": len(payload["steps"]),
        "affinity_channels": list(CAPABILITY_CHANNELS),
        "calibration": payload["calibration"],
        "labels_opened": True,
        "label_scope": "official_source_train_scene_only",
        "development_labels_opened": False,
        "test_labels_opened": False,
    }
    receipt_path = _write_json_no_clobber(
        output_root / "receipts" / f"{args.scene_id}.json", receipt
    )
    manifest = {
        "schema_version": 2,
        "artifact_type": DATASET_SCHEMA_V2,
        "status": "sealed_ready_for_balanced_capability_likelihood_training",
        "scene_count": 1,
        "affinity_channels": list(CAPABILITY_CHANNELS),
        "records": [receipt],
        "preregistration": receipt.get("preregistration", payload["source_authority"]["preregistration"]),
        "safety": {
            "labels_opened": True,
            "label_scope": "official_source_train_scene_only",
            "development_labels_opened": False,
            "test_labels_opened": False,
            "full312_evaluation_authorized": False,
            "spatial_kernel_used_as_instance_likelihood": False,
        },
    }
    manifest_path = _write_json_no_clobber(output_root / "dataset_manifest.json", manifest)
    return manifest_path, {"manifest": str(manifest_path), "sha256": sha256_file(manifest_path), **receipt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--scene-id", default="scene0000_00")
    parser.add_argument("--partition", default="fit")
    parser.add_argument("--primitive-bundle", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-clicks", type=int, default=10)
    parser.add_argument("--spatial-candidate-k", type=int, default=64)
    parser.add_argument("--affinity-chunk-size", type=int, default=2048)
    path, receipt = materialize(parser.parse_args())
    print(json.dumps({"output": str(path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
