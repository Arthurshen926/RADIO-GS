#!/usr/bin/env python3
"""Build all-object fit data for the prior-invariant capability LLR head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .build_capability_likelihood_training_dataset import (
    AFFINITY_CALIBRATION,
    CAPABILITY_CHANNELS,
    click_gaussian_mixture_weights,
)
from .build_likelihood_training_dataset import (
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


SHARD_SCHEMA_V3 = "agile3d-capability-likelihood-ratio-training-shard-v3"
DATASET_SCHEMA_V3 = "agile3d-capability-likelihood-ratio-training-dataset-v3"
FIXED_FIT_SCENES = ("scene0000_00", "scene0002_00", "scene0005_00")


def legal_quantized_object_ids(labels: np.ndarray) -> list[int]:
    values = np.asarray(labels, dtype=np.int32).reshape(-1)
    return [
        int(object_id)
        for object_id in sorted(np.unique(values).tolist())
        if int(object_id) > 0
        and bool((values == int(object_id)).any())
        and bool((values != int(object_id)).any())
    ]


def stable_class_sample_indices(
    target: torch.Tensor,
    *,
    scene_id: str,
    object_id: int,
    click_count: int,
    per_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lowest keyed rows give deterministic prevalence-independent samples."""

    mask = torch.as_tensor(target).bool().reshape(-1).cpu()

    def select(class_value: bool) -> torch.Tensor:
        rows = torch.nonzero(mask == bool(class_value), as_tuple=False).flatten().tolist()
        keyed = []
        prefix = f"{scene_id}|{int(object_id)}|{int(click_count)}|".encode()
        for row in rows:
            key = hashlib.sha256(prefix + str(int(row)).encode()).digest()
            keyed.append((key, int(row)))
        keyed.sort()
        return torch.tensor(
            [row for _key, row in keyed[: min(int(per_class), len(keyed))]],
            dtype=torch.int32,
        )

    return select(True), select(False)


class SceneCapabilityAffinityComputer:
    """Reuse one sealed scene calibration across every legal fit object."""

    def __init__(
        self,
        bundle: Mapping[str, object],
        *,
        device: torch.device,
        chunk_size: int,
    ) -> None:
        self.device = device
        self.chunk_size = int(chunk_size)
        self.xyz = torch.as_tensor(bundle["primitive_xyz"], device=device).float()
        self.covariance = torch.as_tensor(
            bundle["primitive_covariance"], device=device
        ).float()
        self.opacity = torch.as_tensor(
            bundle["primitive_opacity"], device=device
        ).float()
        self.official_xyz = torch.as_tensor(
            bundle["official_point_xyz"], device=device
        ).float()
        self.point_candidates = torch.as_tensor(
            bundle["point_candidate_indices"], device=device, dtype=torch.long
        )
        self.features: dict[str, torch.Tensor] = {}
        self.centered: dict[str, torch.Tensor] = {}
        self.centroids: dict[str, torch.Tensor] = {}
        for name in CAPABILITY_CHANNELS:
            source = torch.as_tensor(bundle[name]).cpu()
            centroid = torch.zeros(source.shape[1], dtype=torch.float64)
            for start in range(0, len(source), self.chunk_size):
                rows = F.normalize(
                    source[start : start + self.chunk_size].float(),
                    dim=1,
                    eps=1e-8,
                )
                centroid += rows.double().sum(dim=0)
            centroid = (centroid / len(source)).float().to(device)
            feature = source.to(device)
            centered = torch.empty_like(feature, dtype=torch.float16)
            for start in range(0, len(feature), self.chunk_size):
                stop = min(start + self.chunk_size, len(feature))
                centered[start:stop] = F.normalize(
                    F.normalize(feature[start:stop].float(), dim=1, eps=1e-8)
                    - centroid[None],
                    dim=1,
                    eps=1e-8,
                ).half()
            self.features[name] = feature
            self.centered[name] = centered
            self.centroids[name] = centroid

    @torch.inference_mode()
    def affinities(self, click_rows: torch.Tensor) -> torch.Tensor:
        rows = torch.as_tensor(click_rows, device=self.device, dtype=torch.long)
        candidates = self.point_candidates.index_select(0, rows)
        click_xyz = self.official_xyz.index_select(0, rows)
        mixture = click_gaussian_mixture_weights(
            primitive_xyz=self.xyz,
            primitive_covariance=self.covariance,
            primitive_opacity=self.opacity,
            click_xyz=click_xyz,
            click_candidate_indices=candidates,
        ).to(self.device)
        outputs = []
        for name in CAPABILITY_CHANNELS:
            candidate_feature = F.normalize(
                self.features[name][candidates].float(), dim=2, eps=1e-8
            )
            prototype = (candidate_feature * mixture[..., None]).sum(dim=1)
            prototype = F.normalize(
                prototype - self.centroids[name][None], dim=1, eps=1e-8
            )
            affinity = torch.empty(
                (len(self.xyz), len(rows)),
                device=self.device,
                dtype=torch.float16,
            )
            for start in range(0, len(self.xyz), self.chunk_size):
                stop = min(start + self.chunk_size, len(self.xyz))
                cosine = self.centered[name][start:stop].float() @ prototype.T
                affinity[start:stop] = ((cosine + 1) * 0.5).clamp(0, 1).half()
            outputs.append(affinity)
        return torch.stack(outputs, dim=-1).cpu()


def _load_bundle(path: Path, *, scene_id: str) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("artifact_type")
        != "agile3d-canonical-gaussian-primitive-bundle-v1"
        or payload.get("scene_id") != scene_id
    ):
        raise ValueError("v3 requires a matching canonical Gaussian bundle")
    safety = payload.get("safety", {})
    for key, expected in {
        "query_independent": True,
        "object_id_used": False,
        "clicks_opened": False,
        "gt_labels_opened": False,
        "test_labels_opened": False,
        "point_as_primitive_used": False,
    }.items():
        if safety.get(key) is not expected:
            raise PermissionError(f"v3 primitive bundle violates {key}")
    return payload


def build(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    scene_id = str(args.scene_id)
    if scene_id not in FIXED_FIT_SCENES:
        raise PermissionError("v3 data builder is sealed to fit scenes 0000/0002/0005")
    split_path = Path(args.split_manifest).resolve()
    if sha256_file(split_path) != str(args.split_manifest_sha256):
        raise ValueError("v3 split manifest SHA-256 differs")
    split = verify_scene_split_sources(_load_json(split_path))
    if scene_id not in split["partitions"]["fit"] or scene_id in split["partitions"]["test"]:
        raise PermissionError("v3 builder crossed the fit/test scene boundary")
    bundle_path = Path(args.primitive_bundle).resolve()
    ply_path = Path(args.official_ply).resolve()
    bundle = _load_bundle(bundle_path, scene_id=scene_id)
    xyz, colors = _read_official_geometry(ply_path)
    labels = _read_official_labels(ply_path)
    quantized = quantize_scannet_points(
        xyz, colors, labels, voxel_size=float(args.voxel_size)
    )
    world = quantized.raw_coordinates + xyz.min(axis=0, keepdims=True)
    bundle_points = np.asarray(torch.as_tensor(bundle["official_point_xyz"]).float())
    if world.shape != bundle_points.shape or not np.allclose(
        world, bundle_points, atol=1e-6, rtol=0
    ):
        raise ValueError("fit labels and sealed bundle official rows differ")
    object_ids = legal_quantized_object_ids(quantized.labels)
    if not object_ids:
        raise ValueError("fit scene has no legal quantized source objects")
    device = torch.device(args.device)
    computer = SceneCapabilityAffinityComputer(
        bundle, device=device, chunk_size=int(args.affinity_chunk_size)
    )
    mapping = torch.as_tensor(bundle["primitive_to_point_index"]).long()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for object_id in object_ids:
        point_target = torch.from_numpy(
            np.ascontiguousarray(quantized.labels == int(object_id))
        )
        clicks = synthesize_click_trajectory(
            world,
            point_target.numpy(),
            max_clicks=int(args.max_clicks),
            click_workers=1,
        )
        click_rows = torch.tensor(
            [int(click.point_index) for click in clicks], dtype=torch.long
        )
        affinity = computer.affinities(click_rows)
        primitive_target = point_target.index_select(0, mapping).bool()
        steps = []
        for click_count in range(1, len(clicks) + 1):
            prefix = clicks[:click_count]
            positive_sample, negative_sample = stable_class_sample_indices(
                primitive_target,
                scene_id=scene_id,
                object_id=int(object_id),
                click_count=click_count,
                per_class=int(args.training_rows_per_class),
            )
            steps.append(
                {
                    "click_count": click_count,
                    "positive_columns": [
                        index for index, click in enumerate(prefix) if click.is_positive
                    ],
                    "negative_columns": [
                        index for index, click in enumerate(prefix) if not click.is_positive
                    ],
                    "positive_training_rows": positive_sample,
                    "negative_training_rows": negative_sample,
                }
            )
        shard = {
            "schema_version": 3,
            "artifact_type": SHARD_SCHEMA_V3,
            "scene_id": scene_id,
            "object_id": int(object_id),
            "partition": "fit",
            "affinity_channels": list(CAPABILITY_CHANNELS),
            "affinity_calibration": AFFINITY_CALIBRATION,
            "capability_click_affinity": affinity,
            "point_target": point_target,
            "primitive_target": primitive_target,
            "prior_probability": torch.as_tensor(bundle["prior_probability"]).half(),
            "coverage": torch.as_tensor(bundle["coverage"]).half(),
            "reliability": torch.as_tensor(bundle["reliability"]).half(),
            "steps": steps,
            "clicks": [
                {
                    "point_index": int(click.point_index),
                    "is_positive": bool(click.is_positive),
                    "order": int(click.order),
                }
                for click in clicks
            ],
            "primitive_foreground_prevalence": float(primitive_target.float().mean()),
            "point_foreground_prevalence": float(point_target.float().mean()),
            "safety": {
                "labels_opened": True,
                "label_scope": "all_legal_official_fit_scene_objects",
                "development_labels_opened": False,
                "test_labels_opened": False,
                "soft_dice_target_materialized": False,
                "spatial_kernel_used_as_instance_likelihood": False,
            },
        }
        shard_path = _write_torch_no_clobber(
            output_dir / f"object_{int(object_id):04d}.pt", shard
        )
        records.append(
            {
                "scene_id": scene_id,
                "object_id": int(object_id),
                "shard": {"path": str(shard_path), "sha256": sha256_file(shard_path)},
                "primitive_foreground_prevalence": shard[
                    "primitive_foreground_prevalence"
                ],
                "point_foreground_prevalence": shard["point_foreground_prevalence"],
                "click_count": len(clicks),
            }
        )
    manifest = {
        "schema_version": 3,
        "artifact_type": DATASET_SCHEMA_V3,
        "scene_id": scene_id,
        "partition": "fit",
        "object_definition": (
            "all positive instance ids in official 5cm first-occurrence labels "
            "with nonempty foreground and background"
        ),
        "object_count": len(records),
        "records": records,
        "source_authority": {
            "split_manifest": {"path": str(split_path), "sha256": sha256_file(split_path)},
            "primitive_bundle": {
                "path": str(bundle_path),
                "sha256": sha256_file(bundle_path),
            },
            "official_ply": {"path": str(ply_path), "sha256": sha256_file(ply_path)},
        },
        "safety": {
            "labels_opened": True,
            "label_scope": "all_legal_official_fit_scene_objects",
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
        },
    }
    manifest_path = _write_json_no_clobber(Path(args.manifest), manifest)
    return manifest_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--primitive-bundle", required=True)
    parser.add_argument("--official-ply", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--max-clicks", type=int, default=10)
    parser.add_argument("--training-rows-per-class", type=int, default=4096)
    parser.add_argument("--affinity-chunk-size", type=int, default=4096)
    path, manifest = build(parser.parse_args())
    print(json.dumps({"manifest": str(path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
