#!/usr/bin/env python3
"""Materialize one label-free canonical Gaussian bundle for AGILE training.

This stage deliberately stops before an object id, click, or PLY label is
opened.  It binds an existing query-independent canonical field/capability
bank to the geometry-only official 5 cm point domain.  The resulting bundle
can later be consumed by ``build_likelihood_training_dataset`` after that
builder independently authorizes and opens a source-train target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree
import torch

from radio_gs.config import load_config
from radio_gs.interfaces.capability_cache import (
    load_canonical_capability_bank,
    load_canonical_primitive_reliability,
)

from .build_likelihood_training_dataset import (
    _load_json,
    sha256_file,
    validate_scene_split,
)
from .evaluate_canonical_field import (
    _gaussian_covariances,
    _load_geometry_model,
    _read_official_geometry,
)
from .protocol import quantize_scannet_points


BUNDLE_SCHEMA = "agile3d-canonical-gaussian-primitive-bundle-v1"
RECEIPT_SCHEMA = "agile3d-canonical-gaussian-primitive-bundle-receipt-v1"
SUPPORTED_SCENE = "scene0000_00"
TENSOR_KEYS = (
    "primitive_xyz",
    "primitive_covariance",
    "primitive_opacity",
    "appearance",
    "boundary",
    "prior_probability",
    "coverage",
    "reliability",
    "global_rows",
    "official_point_xyz",
    "point_candidate_indices",
    "primitive_to_point_index",
)
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "object_id",
        "query",
        "queries",
        "click",
        "clicks",
        "label",
        "labels",
        "target",
        "targets",
        "ground_truth",
        "point_target",
        "primitive_target",
    }
)


def _tensor_sha256(value: torch.Tensor, *, chunk_rows: int = 4096) -> str:
    """Hash tensor bytes in logical C row order without a whole-tensor copy."""

    tensor = torch.as_tensor(value).detach().cpu()
    digest = hashlib.sha256()
    if tensor.ndim == 0:
        digest.update(tensor.contiguous().numpy().tobytes())
        return digest.hexdigest()
    for start in range(0, int(tensor.shape[0]), max(1, int(chunk_rows))):
        rows = tensor[start : start + int(chunk_rows)].contiguous().numpy()
        digest.update(rows.tobytes(order="C"))
    return digest.hexdigest()


def _tensor_record(value: torch.Tensor) -> dict[str, object]:
    tensor = torch.as_tensor(value)
    return {
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "sha256": _tensor_sha256(tensor),
    }


def _validate_sha256(path: str | Path, expected: str, *, role: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} is missing: {resolved}")
    actual = sha256_file(resolved)
    if actual != str(expected):
        raise ValueError(f"{role} SHA-256 differs: expected={expected}, actual={actual}")
    return resolved


def _validate_stage_stamp(
    path: str | Path,
    *,
    artifact: Path,
    expected_kind: str,
    expected_sha256: str,
) -> dict[str, object]:
    stamp_path = Path(path).expanduser().resolve()
    stamp = _load_json(stamp_path)
    if stamp.get("schema") != "scannet-canonical-mpr-v3-paper8-stage-validation-v1":
        raise ValueError(f"unexpected paper8 validation stamp: {stamp_path}")
    if stamp.get("kind") != expected_kind:
        raise ValueError(f"paper8 validation kind differs for {artifact}")
    if stamp.get("artifact_sha256") != expected_sha256:
        raise ValueError(f"paper8 validation digest differs for {artifact}")
    if Path(str(stamp.get("artifact", ""))).name != artifact.name:
        raise ValueError(f"paper8 validation artifact basename differs for {artifact}")
    return stamp


def _validate_source_train_scene(
    split_manifest: str | Path,
    *,
    expected_sha256: str,
    scene_id: str,
) -> Path:
    split_path = _validate_sha256(
        split_manifest, expected_sha256, role="sealed source-train split"
    )
    split = validate_scene_split(_load_json(split_path))
    if scene_id not in split["partitions"]["fit"]:
        raise PermissionError(f"bundle materialization requires a fit scene: {scene_id}")
    # The source object id exists in the split authority, but this query-free
    # stage neither retrieves nor propagates it.
    return split_path


def _official_geometry_only_points(
    ply_path: Path, *, voxel_size: float
) -> tuple[torch.Tensor, dict[str, object]]:
    xyz, colors = _read_official_geometry(ply_path)
    # A dummy vector is supplied solely because the released quantizer keeps
    # labels aligned.  _read_official_labels is intentionally never called.
    dummy = np.zeros(len(xyz), dtype=np.int32)
    quantized = quantize_scannet_points(xyz, colors, dummy, voxel_size=voxel_size)
    origin = xyz.min(axis=0, keepdims=True)
    world = np.ascontiguousarray(quantized.raw_coordinates + origin, dtype=np.float32)
    return torch.from_numpy(world), {
        "raw_vertex_count": int(len(xyz)),
        "quantized_point_count": int(len(world)),
        "voxel_size_m": float(voxel_size),
        "coordinate_contract": "agile3d_shifted_5cm_plus_scene_origin_world_v1",
        "label_property_opened": False,
    }


def geometry_candidate_mappings(
    primitive_xyz: torch.Tensor,
    official_point_xyz: torch.Tensor,
    *,
    candidate_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    primitives = torch.as_tensor(primitive_xyz).detach().float().cpu().numpy()
    points = torch.as_tensor(official_point_xyz).detach().float().cpu().numpy()
    if (
        primitives.ndim != 2
        or primitives.shape[1] != 3
        or points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(primitives).all()
        or not np.isfinite(points).all()
    ):
        raise ValueError("primitive and official geometry must be finite [N,3]")
    count = min(int(candidate_k), len(primitives))
    if count <= 0:
        raise ValueError("candidate_k and primitive count must be positive")
    primitive_tree = cKDTree(primitives)
    _distance, candidates = primitive_tree.query(points, k=count, workers=1)
    candidates = np.asarray(candidates, dtype=np.int32)
    if candidates.ndim == 1:
        candidates = candidates[:, None]
    point_tree = cKDTree(points)
    _distance, primitive_to_point = point_tree.query(primitives, k=1, workers=1)
    return (
        torch.from_numpy(np.ascontiguousarray(candidates)),
        torch.from_numpy(
            np.ascontiguousarray(np.asarray(primitive_to_point, dtype=np.int32))
        ),
    )


def build_bundle_payload(
    *,
    scene_id: str,
    primitive_xyz: torch.Tensor,
    primitive_covariance: torch.Tensor,
    primitive_opacity: torch.Tensor,
    appearance: torch.Tensor,
    boundary: torch.Tensor,
    prior_probability: torch.Tensor,
    coverage: torch.Tensor,
    reliability: torch.Tensor,
    global_rows: torch.Tensor,
    official_point_xyz: torch.Tensor,
    point_candidate_indices: torch.Tensor,
    primitive_to_point_index: torch.Tensor,
    provenance: Mapping[str, Any],
) -> dict[str, object]:
    tensors = {
        "primitive_xyz": torch.as_tensor(primitive_xyz).detach().cpu(),
        "primitive_covariance": torch.as_tensor(primitive_covariance).detach().cpu(),
        "primitive_opacity": torch.as_tensor(primitive_opacity).detach().cpu(),
        "appearance": torch.as_tensor(appearance).detach().cpu(),
        "boundary": torch.as_tensor(boundary).detach().cpu(),
        "prior_probability": torch.as_tensor(prior_probability).detach().cpu(),
        "coverage": torch.as_tensor(coverage).detach().cpu(),
        "reliability": torch.as_tensor(reliability).detach().cpu(),
        "global_rows": torch.as_tensor(global_rows).detach().cpu(),
        "official_point_xyz": torch.as_tensor(official_point_xyz).detach().cpu(),
        "point_candidate_indices": torch.as_tensor(
            point_candidate_indices
        ).detach().cpu(),
        "primitive_to_point_index": torch.as_tensor(
            primitive_to_point_index
        ).detach().cpu(),
    }
    rows = int(tensors["primitive_xyz"].shape[0])
    points = int(tensors["official_point_xyz"].shape[0])
    if tensors["primitive_xyz"].shape != (rows, 3):
        raise ValueError("primitive xyz must be [N,3]")
    if tensors["primitive_covariance"].shape != (rows, 3, 3):
        raise ValueError("primitive covariance must be [N,3,3]")
    if tensors["appearance"].ndim != 2 or tensors["appearance"].shape[0] != rows:
        raise ValueError("appearance rows do not align")
    if tensors["boundary"].ndim != 2 or tensors["boundary"].shape[0] != rows:
        raise ValueError("boundary rows do not align")
    for key in (
        "primitive_opacity",
        "prior_probability",
        "coverage",
        "reliability",
        "global_rows",
        "primitive_to_point_index",
    ):
        if tensors[key].reshape(-1).shape != (rows,):
            raise ValueError(f"{key} rows do not align")
        tensors[key] = tensors[key].reshape(-1)
    if tensors["official_point_xyz"].shape != (points, 3):
        raise ValueError("official point geometry must be [P,3]")
    candidates = tensors["point_candidate_indices"]
    if candidates.ndim != 2 or candidates.shape[0] != points:
        raise ValueError("point candidates must be [P,K]")
    if bool((candidates < 0).any()) or bool((candidates >= rows).any()):
        raise ValueError("point candidate mapping is outside primitive domain")
    primitive_to_point = tensors["primitive_to_point_index"]
    if bool((primitive_to_point < 0).any()) or bool((primitive_to_point >= points).any()):
        raise ValueError("primitive-to-point mapping is outside official point domain")
    if bool((tensors["prior_probability"] != 0.5).any()):
        raise ValueError("query-independent bundle requires a neutral 0.5 prior")
    for key in ("coverage", "reliability", "primitive_opacity"):
        values = tensors[key].float()
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()) or bool(
            (values > 1).any()
        ):
            raise ValueError(f"{key} must be finite in [0,1]")
    for key, tensor in tensors.items():
        if tensor.dtype.is_floating_point and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{key} contains NaN or infinity")
    lowered = {str(key).lower() for key in provenance}
    if lowered & FORBIDDEN_PAYLOAD_KEYS:
        raise ValueError("bundle provenance contains query/object/label material")
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": BUNDLE_SCHEMA,
        "scene_id": str(scene_id),
        **tensors,
        "tensor_records": {key: _tensor_record(tensors[key]) for key in TENSOR_KEYS},
        "provenance": dict(provenance),
        "contracts": {
            "primitive_row_order": "ascending_global_rows_where_capability_valid",
            "point_row_order": "official_agile3d_5cm_first_occurrence",
            "point_candidate_order": "euclidean_distance_then_ckdtree_index",
            "prior_probability": "neutral_query_independent_0.5",
            "coverage": "canonical_reliability_observation_evidence",
            "reliability": "canonical_primitive_reliability_v1",
        },
        "safety": {
            "query_independent": True,
            "object_id_used": False,
            "clicks_opened": False,
            "gt_labels_opened": False,
            "ply_label_property_opened": False,
            "test_labels_opened": False,
            "target_masks_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "point_as_primitive_used": False,
            "cuda_used": False,
        },
    }
    if set(payload) & FORBIDDEN_PAYLOAD_KEYS:
        raise AssertionError("bundle top-level keys cross the query-free boundary")
    return payload


def _write_torch_no_clobber(path: Path, value: Mapping[str, object]) -> Path:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"refusing to replace primitive bundle: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _write_json_no_clobber(path: Path, value: Mapping[str, object]) -> Path:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
    except FileExistsError:
        if output.read_bytes() != encoded:
            raise ValueError(f"refusing to replace different receipt: {output}")
    finally:
        temporary.unlink(missing_ok=True)
    return output


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    if args.scene_id != SUPPORTED_SCENE:
        raise ValueError(f"Stage-A is sealed to {SUPPORTED_SCENE}")
    if str(args.device) != "cpu":
        raise ValueError("query-independent Stage-A materialization is CPU-only")
    split_path = _validate_source_train_scene(
        args.split_manifest,
        expected_sha256=args.split_manifest_sha256,
        scene_id=args.scene_id,
    )
    config_path = _validate_sha256(
        args.config, args.config_sha256, role="paper8 render config"
    )
    geometry_path = _validate_sha256(
        args.geometry_checkpoint,
        args.geometry_checkpoint_sha256,
        role="paper8 Gaussian checkpoint",
    )
    field_path = _validate_sha256(
        args.field_checkpoint,
        args.field_checkpoint_sha256,
        role="paper8 canonical field",
    )
    capability_path = _validate_sha256(
        args.capability_cache,
        args.capability_cache_sha256,
        role="paper8 capability cache",
    )
    reliability_path = Path(args.reliability_cache).expanduser().resolve()
    if not reliability_path.is_file():
        raise FileNotFoundError(f"canonical reliability cache is missing: {reliability_path}")
    ply_path = Path(args.official_ply).expanduser().resolve()
    if not ply_path.is_file() or ply_path.stem != args.scene_id:
        raise FileNotFoundError(f"official source-train PLY differs: {ply_path}")
    _validate_stage_stamp(
        args.field_validation_stamp,
        artifact=field_path,
        expected_kind="field_v2",
        expected_sha256=args.field_checkpoint_sha256,
    )
    _validate_stage_stamp(
        args.capability_validation_stamp,
        artifact=capability_path,
        expected_kind="capability",
        expected_sha256=args.capability_cache_sha256,
    )

    bank = load_canonical_capability_bank(
        capability_path,
        expected_field_checkpoint_sha256=args.field_checkpoint_sha256,
    )
    reliability_bank = load_canonical_primitive_reliability(
        reliability_path,
        expected_xyz=bank.xyz,
        expected_valid=bank.valid,
        expected_field_checkpoint_sha256=args.field_checkpoint_sha256,
    )
    global_rows = bank.global_rows.long()
    features = bank.valid_feature_banks()
    primitive_xyz = bank.xyz.index_select(0, global_rows).float().contiguous()
    reliability = reliability_bank.confidence.index_select(0, global_rows).half()
    coverage = reliability_bank.components["observation_evidence"].index_select(
        0, global_rows
    ).half()

    config = load_config(str(config_path))
    model = _load_geometry_model(config, str(geometry_path), torch.device("cpu"))
    try:
        full_xyz = model.get_xyz().detach().float().cpu()
        if full_xyz.shape != bank.xyz.shape or not torch.allclose(
            full_xyz, bank.xyz, atol=1e-6, rtol=0.0
        ):
            raise ValueError("paper8 Gaussian geometry and capability rows differ")
        covariance = _gaussian_covariances(model).detach().index_select(
            0, global_rows
        ).float().cpu()
        opacity = model.get_opacity().detach().float().reshape(-1).index_select(
            0, global_rows
        ).cpu()
    finally:
        del model

    official_points, point_report = _official_geometry_only_points(
        ply_path, voxel_size=float(args.voxel_size)
    )
    candidates, primitive_to_point = geometry_candidate_mappings(
        primitive_xyz,
        official_points,
        candidate_k=int(args.candidate_k),
    )
    source_records = {}
    for name, path in (
        ("split_manifest", split_path),
        ("config", config_path),
        ("geometry_checkpoint", geometry_path),
        ("field_checkpoint", field_path),
        ("capability_cache", capability_path),
        ("reliability_cache", reliability_path),
        ("official_ply", ply_path),
        ("field_validation_stamp", Path(args.field_validation_stamp).resolve()),
        (
            "capability_validation_stamp",
            Path(args.capability_validation_stamp).resolve(),
        ),
    ):
        source_records[name] = {"path": str(path), "sha256": sha256_file(path)}
    payload = build_bundle_payload(
        scene_id=args.scene_id,
        primitive_xyz=primitive_xyz,
        primitive_covariance=covariance,
        primitive_opacity=opacity,
        appearance=features["appearance"],
        boundary=features["boundary"],
        prior_probability=torch.full((len(global_rows),), 0.5, dtype=torch.float16),
        coverage=coverage,
        reliability=reliability,
        global_rows=global_rows.to(torch.int32),
        official_point_xyz=official_points,
        point_candidate_indices=candidates,
        primitive_to_point_index=primitive_to_point,
        provenance={
            "source_assets": source_records,
            "official_point_domain": point_report,
            "capability_source": bank.metadata.get("source"),
            "capability_signatures": {
                name: signature.to_dict()
                for name, signature in bank.signatures.items()
            },
        },
    )
    output = _write_torch_no_clobber(Path(args.output), payload)
    receipt = {
        "schema_version": 1,
        "artifact_type": RECEIPT_SCHEMA,
        "status": "sealed_query_independent_bundle_ready",
        "scene_id": args.scene_id,
        "bundle": {"path": str(output), "sha256": sha256_file(output)},
        "primitive_count": int(len(global_rows)),
        "official_point_count": int(len(official_points)),
        "candidate_k": int(candidates.shape[1]),
        "tensor_records": payload["tensor_records"],
        "source_assets": source_records,
        "safety": payload["safety"],
    }
    receipt_path = _write_json_no_clobber(Path(args.receipt), receipt)
    return receipt_path, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default=SUPPORTED_SCENE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--geometry-checkpoint-sha256", required=True)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--field-checkpoint-sha256", required=True)
    parser.add_argument("--field-validation-stamp", required=True)
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument("--capability-cache-sha256", required=True)
    parser.add_argument("--capability-validation-stamp", required=True)
    parser.add_argument("--reliability-cache", required=True)
    parser.add_argument("--official-ply", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--candidate-k", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    receipt_path, receipt = materialize(parser.parse_args())
    print(json.dumps({"receipt": str(receipt_path), **receipt}, indent=2))


if __name__ == "__main__":
    main()
