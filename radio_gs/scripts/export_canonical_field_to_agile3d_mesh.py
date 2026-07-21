#!/usr/bin/env python3
"""Export one canonical primitive field into official AGILE3D PLY row order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.interfaces.capability_cache import load_canonical_capability_bank


def interpolate_feature_matrix(
    source_xyz: np.ndarray,
    source_features: np.ndarray,
    target_xyz: np.ndarray,
    *,
    neighbors: int,
    maximum_distance_m: float,
    chunk_size: int = 16384,
    workers: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Query-free inverse-distance interpolation for vector features."""

    source_xyz = np.asarray(source_xyz, dtype=np.float32)
    source_features = np.asarray(source_features, dtype=np.float32)
    target_xyz = np.asarray(target_xyz, dtype=np.float32)
    if source_xyz.shape != (source_features.shape[0], 3):
        raise ValueError("source xyz/features must align")
    if target_xyz.ndim != 2 or target_xyz.shape[1] != 3:
        raise ValueError("target xyz must be [N,3]")
    if neighbors <= 0 or maximum_distance_m <= 0:
        raise ValueError("neighbors and maximum distance must be positive")
    tree = cKDTree(source_xyz)
    output = np.zeros((len(target_xyz), source_features.shape[1]), dtype=np.float32)
    valid = np.zeros(len(target_xyz), dtype=bool)
    k = min(int(neighbors), len(source_xyz))
    for start in range(0, len(target_xyz), int(chunk_size)):
        stop = min(start + int(chunk_size), len(target_xyz))
        distance, index = tree.query(
            target_xyz[start:stop],
            k=k,
            workers=int(workers),
        )
        distance = np.asarray(distance, dtype=np.float32)
        index = np.asarray(index, dtype=np.int64)
        if distance.ndim == 1:
            distance, index = distance[:, None], index[:, None]
        keep = distance <= float(maximum_distance_m)
        weight = np.where(keep, 1.0 / np.maximum(distance, 1e-4), 0.0)
        denominator = weight.sum(axis=1)
        selected = denominator > 0
        values = (
            source_features[index] * weight[..., None]
        ).sum(axis=1) / np.maximum(denominator[:, None], 1e-8)
        output[start:stop][selected] = values[selected]
        valid[start:stop] = selected
    norm = np.linalg.norm(output, axis=1, keepdims=True)
    output[valid] /= np.maximum(norm[valid], 1e-8)
    return output, valid


def _mesh_xyz(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"].data
    return np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(
        np.float32
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_savez(output: Path, arrays: dict[str, np.ndarray]) -> None:
    """Publish a mesh feature cache only after the complete NPZ is written.

    Sharded AGILE3D field workers and the final evaluator run asynchronously.
    A same-directory atomic rename prevents the evaluator from treating the
    visible prefix of a still-writing ``np.savez`` archive as a finished scene.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


@torch.inference_mode()
def export(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    field, payload = load_canonical_field_checkpoint(
        args.field_checkpoint,
        map_location="cpu",
    )
    field_sha256 = _sha256(Path(args.field_checkpoint))
    bank = load_canonical_capability_bank(
        args.capability_cache,
        expected_field_checkpoint_sha256=field_sha256,
    )
    mpr = torch.load(Path(args.mpr_cache or payload["mpr_cache"]), map_location="cpu")
    xyz = torch.as_tensor(mpr["xyz"]).float()
    valid = torch.as_tensor(mpr["valid"]).bool()
    if xyz.shape != bank.xyz.shape or not torch.allclose(xyz, bank.xyz, atol=1e-6):
        raise ValueError("field MPR and capability cache geometry differ")
    if not torch.equal(valid, bank.valid):
        raise ValueError("field MPR and capability valid rows differ")
    rows = torch.where(valid)[0]
    field = field.to(device).eval().requires_grad_(False)
    radio_parts = []
    for start in range(0, rows.numel(), int(args.decode_batch_size)):
        selected = rows[start : start + int(args.decode_batch_size)].to(device)
        radio_parts.append(
            F.normalize(field.radio_features(selected).float(), dim=-1)
            .half()
            .cpu()
        )
    primitive = {
        "radio_features": torch.cat(radio_parts).float().numpy(),
        "appearance_features": bank.appearance[rows].float().numpy(),
        "boundary_features": bank.boundary[rows].float().numpy(),
    }
    mesh_xyz = _mesh_xyz(Path(args.mesh_ply))
    mapped: dict[str, np.ndarray] = {"xyz": mesh_xyz}
    coverage = None
    mapping_path = Path(str(args.quantization_map)).expanduser()
    exact_mapping = bool(str(args.quantization_map).strip())
    if exact_mapping:
        with np.load(mapping_path, allow_pickle=False) as mapping_payload:
            unique_map = np.asarray(mapping_payload["unique_map"], dtype=np.int64)
            inverse_map = np.asarray(mapping_payload["inverse_map"], dtype=np.int64)
            quantized_xyz = np.asarray(mapping_payload["quantized_xyz"], dtype=np.float32)
        if inverse_map.shape != (len(mesh_xyz),):
            raise ValueError("quantization inverse map does not align with official PLY")
        if (
            unique_map.shape != (len(quantized_xyz),)
            or quantized_xyz.shape != tuple(xyz.shape)
            or not np.allclose(quantized_xyz, xyz.numpy(), atol=1e-6, rtol=0.0)
            or not np.allclose(mesh_xyz[unique_map], quantized_xyz, atol=1e-6, rtol=0.0)
        ):
            raise ValueError("quantization map does not align with canonical geometry")
        all_rows = np.zeros((len(xyz),), dtype=bool)
        all_rows[rows.numpy()] = True
        row_positions = np.full(len(xyz), -1, dtype=np.int64)
        row_positions[rows.numpy()] = np.arange(rows.numel())
        if args.store_quantized:
            mapped = {
                "xyz": quantized_xyz,
                "unique_map": unique_map,
            }
            safe_positions = np.maximum(row_positions, 0)
            for name, matrix in primitive.items():
                values = matrix[safe_positions]
                values[~all_rows] = 0
                mapped[name] = values.astype(np.float16)
            coverage = all_rows
        else:
            valid_full = all_rows[inverse_map]
            safe_positions = np.maximum(row_positions[inverse_map], 0)
            for name, matrix in primitive.items():
                values = matrix[safe_positions]
                values[~valid_full] = 0
                mapped[name] = values.astype(np.float16)
            coverage = valid_full
    else:
        for name, matrix in primitive.items():
            values, feature_valid = interpolate_feature_matrix(
                xyz[rows].numpy(),
                matrix,
                mesh_xyz,
                neighbors=int(args.neighbors),
                maximum_distance_m=float(args.maximum_distance_m),
                chunk_size=int(args.interpolation_chunk_size),
                workers=int(args.interpolation_workers),
            )
            mapped[name] = values.astype(np.float16)
            coverage = feature_valid if coverage is None else coverage & feature_valid
    assert coverage is not None
    mapped["valid"] = coverage
    output = Path(args.output)
    _atomic_savez(output, mapped)
    report = {
        "output": str(output.resolve()),
        "mesh_ply": str(Path(args.mesh_ply).resolve()),
        "field_checkpoint": str(Path(args.field_checkpoint).resolve()),
        "field_checkpoint_sha256": field_sha256,
        "capability_cache": str(Path(args.capability_cache).resolve()),
        "mesh_vertices": len(mesh_xyz),
        "valid_vertices": int(coverage.sum()),
        "coverage": float(coverage.mean()),
        "mapping_mode": (
            (
                "exact_official_5cm_quantized_rows"
                if args.store_quantized
                else "exact_official_5cm_inverse_map"
            )
            if exact_mapping
            else "query_free_inverse_distance_knn"
        ),
        "quantization_map": (
            str(mapping_path.resolve()) if exact_mapping else ""
        ),
        "neighbors": int(args.neighbors),
        "maximum_distance_m": float(args.maximum_distance_m),
        "labels_opened": False,
        "queries_opened": False,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument("--mpr-cache", default="")
    parser.add_argument("--mesh-ply", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decode-batch-size", type=int, default=4096)
    parser.add_argument("--interpolation-chunk-size", type=int, default=16384)
    parser.add_argument(
        "--interpolation-workers",
        type=int,
        default=-1,
        help="cKDTree query workers; -1 uses all available workers.",
    )
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--maximum-distance-m", type=float, default=0.10)
    parser.add_argument(
        "--quantization-map",
        default="",
        help=(
            "Optional exact first-occurrence 5 cm map from "
            "build_agile3d_gaussian_geometry.py; bypasses KNN interpolation."
        ),
    )
    parser.add_argument(
        "--store-quantized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --quantization-map, store one feature row per official 5 cm "
            "voxel instead of expanding back to every raw PLY vertex."
        ),
    )
    args = parser.parse_args()
    print(json.dumps(export(args), indent=2))


if __name__ == "__main__":
    main()
