#!/usr/bin/env python3
"""Build query-independent hard-boundary relation triplets for Field-B."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from radio_gs.training.tensor_cache_io import load_mpr_cache
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file


SCHEMA_VERSION = "canonical_field_b_boundary_relation_triplets_v1"


def _validate_exact_capability_pair(dino: dict, sam: dict) -> None:
    dino_metadata = dict(dino.get("metadata", {}))
    sam_metadata = dict(sam.get("metadata", {}))
    for name, payload, metadata in (
        ("dino_v3", dino, dino_metadata),
        ("sam3", sam, sam_metadata),
    ):
        if metadata.get("feature_space") != name:
            raise ValueError(f"Field-B {name} feature space differs")
        expected = {
            "aggregation_mode": "raster_adjoint",
            "raster_view_fusion": "contribution_mean",
            "raster_reliability_mode": "mean_resultant",
            "capability_map_source": "project_raw",
            "capability_projection_before_mpr": True,
            "normalize_each_view": True,
            "custom_adaptor_head": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        }
        mismatched = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatched:
            raise ValueError(f"Field-B {name} exact-target policy differs: {mismatched}")
        features = torch.as_tensor(payload["features"])
        if features.ndim != 2 or not features.is_floating_point():
            raise ValueError(f"Field-B {name} features must be floating [N,D]")
    for key in ("xyz", "valid", "view_counts"):
        if not torch.equal(torch.as_tensor(dino[key]), torch.as_tensor(sam[key])):
            raise ValueError(f"Field-B DINO/SAM {key} differs")
    for key in (
        "selected_frame_indices",
        "feature_output_bundle_sha256",
        "official_adaptor_checkpoint_sha256",
    ):
        if dino_metadata.get(key) != sam_metadata.get(key):
            raise ValueError(f"Field-B DINO/SAM provenance differs: {key}")


def build_triplets_from_arrays(
    xyz: torch.Tensor,
    valid: torch.Tensor,
    dino_features: torch.Tensor,
    sam_features: torch.Tensor,
    *,
    neighbors: int = 16,
    chunk_size: int = 256,
) -> dict[str, torch.Tensor | dict[str, int]]:
    """Select one local positive and one cross-channel hard negative per anchor.

    Positive selection maximizes the geometric mean of exact DINO and SAM
    affinities inside the fixed kNN region. Negative selection maximizes local
    geometric proximity times the stronger of two cross-channel boundary
    terms: DINO-similar/SAM-dissimilar or SAM-similar/DINO-dissimilar. The
    ranking margin is the detached exact-teacher relation gap, so no scene- or
    benchmark-selected scalar margin exists.
    """

    points = torch.as_tensor(xyz).detach().float().cpu()
    active = torch.as_tensor(valid).detach().bool().cpu().reshape(-1)
    dino = torch.as_tensor(dino_features).detach().cpu()
    sam = torch.as_tensor(sam_features).detach().cpu()
    if points.ndim != 2 or points.shape[1] != 3 or active.shape != (points.shape[0],):
        raise ValueError("Field-B xyz/valid must align as [N,3]/[N]")
    if dino.ndim != 2 or sam.ndim != 2 or dino.shape[0] != points.shape[0] or sam.shape[0] != points.shape[0]:
        raise ValueError("Field-B capability features must align with xyz")
    if not bool(torch.isfinite(points).all()) or neighbors < 2 or chunk_size <= 0:
        raise ValueError("Field-B geometry/neighbors/chunk size is invalid")
    global_rows = torch.where(active)[0]
    if global_rows.numel() <= neighbors:
        raise ValueError("Field-B needs more valid rows than kNN neighbors")
    local_xyz = points[global_rows].numpy()
    raw_distances, raw_neighbors = cKDTree(local_xyz).query(
        local_xyz,
        k=int(neighbors) + 1,
        workers=1,
    )
    raw_distances = np.asarray(raw_distances, dtype=np.float32)
    raw_neighbors = np.asarray(raw_neighbors, dtype=np.int64)
    distances = np.empty((global_rows.numel(), int(neighbors)), dtype=np.float32)
    local_neighbors = np.empty(
        (global_rows.numel(), int(neighbors)), dtype=np.int64
    )
    for row in range(global_rows.numel()):
        keep = raw_neighbors[row] != row
        if int(keep.sum()) < int(neighbors):
            raise ValueError("Field-B kNN query did not return enough non-self rows")
        distances[row] = raw_distances[row, keep][: int(neighbors)]
        local_neighbors[row] = raw_neighbors[row, keep][: int(neighbors)]
    if local_neighbors.shape != (global_rows.numel(), int(neighbors)):
        raise ValueError("Field-B kNN topology shape differs")
    neighbor_rows = global_rows[torch.from_numpy(local_neighbors)]
    distances_tensor = torch.from_numpy(distances)

    anchors_out: list[torch.Tensor] = []
    positives_out: list[torch.Tensor] = []
    negatives_out: list[torch.Tensor] = []
    margins_out: list[torch.Tensor] = []
    channels_out: list[torch.Tensor] = []
    positive_relations_out: list[torch.Tensor] = []
    negative_relations_out: list[torch.Tensor] = []
    negative_hardness_out: list[torch.Tensor] = []
    for start in range(0, global_rows.numel(), int(chunk_size)):
        stop = min(start + int(chunk_size), global_rows.numel())
        anchors = global_rows[start:stop]
        candidates = neighbor_rows[start:stop]
        d_anchor = F.normalize(dino[anchors].float(), dim=-1, eps=1e-8)
        s_anchor = F.normalize(sam[anchors].float(), dim=-1, eps=1e-8)
        d_neighbor = F.normalize(dino[candidates].float(), dim=-1, eps=1e-8)
        s_neighbor = F.normalize(sam[candidates].float(), dim=-1, eps=1e-8)
        dino_affinity = (
            0.5 * (1.0 + (d_anchor[:, None] * d_neighbor).sum(dim=-1))
        ).clamp(0.0, 1.0)
        sam_affinity = (
            0.5 * (1.0 + (s_anchor[:, None] * s_neighbor).sum(dim=-1))
        ).clamp(0.0, 1.0)
        consistency = torch.sqrt((dino_affinity * sam_affinity).clamp_min(0.0))
        positive_local = consistency.argmax(dim=1)

        distance = distances_tensor[start:stop]
        local_scale = distance.median(dim=1).values.clamp_min(1e-12)
        geometry = torch.exp(-torch.square(distance / local_scale[:, None]))
        sam_boundary = torch.sqrt(
            (dino_affinity * (1.0 - sam_affinity)).clamp_min(0.0)
        )
        appearance_boundary = torch.sqrt(
            (sam_affinity * (1.0 - dino_affinity)).clamp_min(0.0)
        )
        hardness, channel = torch.stack(
            [sam_boundary, appearance_boundary], dim=-1
        ).max(dim=-1)
        hardness = hardness * geometry
        hardness.scatter_(1, positive_local[:, None], -1.0)
        negative_local = hardness.argmax(dim=1)

        row = torch.arange(stop - start)
        positive_relation = consistency[row, positive_local]
        negative_relation = consistency[row, negative_local]
        margin = (positive_relation - negative_relation).clamp_min(0.0)
        positive_relations_out.append(positive_relation)
        negative_relations_out.append(negative_relation)
        negative_hardness_out.append(hardness[row, negative_local])
        informative = margin > 0.0
        anchors_out.append(anchors[informative])
        positives_out.append(candidates[row, positive_local][informative])
        negatives_out.append(candidates[row, negative_local][informative])
        margins_out.append(margin[informative])
        channels_out.append(channel[row, negative_local][informative].to(torch.uint8))

    anchor = torch.cat(anchors_out)
    positive = torch.cat(positives_out)
    negative = torch.cat(negatives_out)
    margin = torch.cat(margins_out).float()
    boundary_channel = torch.cat(channels_out)
    positive_relation_all = torch.cat(positive_relations_out).float()
    negative_relation_all = torch.cat(negative_relations_out).float()
    negative_hardness_all = torch.cat(negative_hardness_out).float()
    raw_gap_all = positive_relation_all - negative_relation_all
    candidate_anchors = int(global_rows.numel())
    triplets = int(margin.numel())

    def summary(values: torch.Tensor) -> dict[str, float]:
        quantiles = torch.quantile(
            values,
            torch.tensor([0.05, 0.5, 0.95], dtype=values.dtype),
        )
        return {
            "min": float(values.min()),
            "p05": float(quantiles[0]),
            "median": float(quantiles[1]),
            "mean": float(values.mean()),
            "p95": float(quantiles[2]),
            "max": float(values.max()),
        }

    positive_pairs = torch.stack([anchor, positive])
    negative_pairs = torch.stack([anchor, negative])
    return {
        "pair_index": torch.cat([positive_pairs, negative_pairs], dim=1),
        "teacher_margin": margin,
        "boundary_channel": boundary_channel,
        "audit": {
            "candidate_anchors": candidate_anchors,
            "triplets": triplets,
            "retained_fraction": float(triplets / candidate_anchors),
            "dropped_nonpositive_teacher_gap": candidate_anchors - triplets,
            "nonpositive_teacher_gap_fraction": float(
                (candidate_anchors - triplets) / candidate_anchors
            ),
            "zero_selected_boundary_hardness": int(
                (negative_hardness_all <= 0.0).sum()
            ),
            "zero_selected_boundary_hardness_fraction": float(
                (negative_hardness_all <= 0.0).float().mean()
            ),
            "sam_boundary_negatives": int((boundary_channel == 0).sum()),
            "appearance_boundary_negatives": int((boundary_channel == 1).sum()),
            "teacher_margin": summary(margin),
            "teacher_positive_relation": summary(positive_relation_all),
            "teacher_negative_relation": summary(negative_relation_all),
            "selected_boundary_hardness": summary(negative_hardness_all),
            "raw_teacher_gap_min": float(raw_gap_all.min()),
        },
    }


def build(args: argparse.Namespace) -> dict:
    registration, registration_sha, registration_path = load_json_object(
        args.experiment_registration,
        expected_sha256=args.expected_experiment_registration_sha256,
        label="Field-B experiment registration",
    )
    if registration.get("schema_version") != "canonical_field_b_boundary_relation_registration_v1":
        raise ValueError("Field-B registration schema differs")
    dino, dino_sha, dino_path = load_mpr_cache(
        args.dino_mpr_cache,
        expected_sha256=args.expected_dino_mpr_cache_sha256,
        expected_feature_space="dino_v3",
        require_reliability=True,
        require_formal_safety=True,
    )
    sam, sam_sha, sam_path = load_mpr_cache(
        args.sam3_mpr_cache,
        expected_sha256=args.expected_sam3_mpr_cache_sha256,
        expected_feature_space="sam3",
        require_reliability=True,
        require_formal_safety=True,
    )
    _validate_exact_capability_pair(dino, sam)
    immutable_inputs = dict(registration.get("immutable_inputs", {}))
    if dict(immutable_inputs.get("exact_dino_v3", {})).get("sha256") != dino_sha:
        raise ValueError("Field-B registration DINOv3 SHA-256 differs")
    if dict(immutable_inputs.get("exact_sam3", {})).get("sha256") != sam_sha:
        raise ValueError("Field-B registration SAM3 SHA-256 differs")
    triplet_contract = dict(registration.get("triplet_contract", {}))
    if (
        triplet_contract.get("neighbors") != int(args.neighbors)
        or triplet_contract.get("fixed_scalar_margin", "missing") is not None
    ):
        raise ValueError("Field-B registration triplet contract differs")
    registered_builder_sha = dict(registration.get("source_hashes", {})).get(
        "triplet_builder"
    )
    if registered_builder_sha != sha256_file(Path(__file__).resolve()):
        raise ValueError("Field-B registered builder source SHA-256 differs")
    built = build_triplets_from_arrays(
        dino["xyz"],
        dino["valid"],
        dino["features"],
        sam["features"],
        neighbors=int(args.neighbors),
        chunk_size=int(args.chunk_size),
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "construction": "exact_capability_local_hard_boundary_ranking_v1",
        "neighbors": int(args.neighbors),
        "topology": "euclidean_knn_query_independent",
        "positive_rule": "argmax_sqrt(dino_affinity*sam_affinity)_within_k16",
        "negative_rule": "argmax_geometry*max(sqrt(dino*(1-sam)),sqrt(sam*(1-dino)))_excluding_positive",
        "geometry_rule": "exp(-(distance/local_median_neighbor_distance)^2)",
        "margin_rule": "exact_teacher_positive_relation_minus_negative_relation",
        "fixed_scalar_margin": None,
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "experiment_registration": {"path": str(registration_path), "sha256": registration_sha},
        "dino_mpr": {"path": str(dino_path), "sha256": dino_sha},
        "sam3_mpr": {"path": str(sam_path), "sha256": sam_sha},
        "geometry_fingerprint": dino.get("geometry_fingerprint", {}),
        **built["audit"],
    }
    output = Path(args.output).expanduser().resolve()
    report_output = output.with_suffix(output.suffix + ".json")
    if (
        output.exists()
        or output.is_symlink()
        or report_output.exists()
        or report_output.is_symlink()
    ):
        raise FileExistsError(f"Field-B triplet output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(
            {
                "schema_version": 1,
                "pair_index": built["pair_index"],
                "teacher_margin": built["teacher_margin"].half(),
                "boundary_channel": built["boundary_channel"],
                "metadata": metadata,
            },
            temporary,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {**metadata, "output": str(output), "output_sha256": sha256_file(output)}
    report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-registration", required=True)
    parser.add_argument("--expected-experiment-registration-sha256", required=True)
    parser.add_argument("--dino-mpr-cache", required=True)
    parser.add_argument("--expected-dino-mpr-cache-sha256", required=True)
    parser.add_argument("--sam3-mpr-cache", required=True)
    parser.add_argument("--expected-sam3-mpr-cache-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.neighbors != 16:
        parser.error("Field-B v1 freezes --neighbors at the existing global k=16")
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
