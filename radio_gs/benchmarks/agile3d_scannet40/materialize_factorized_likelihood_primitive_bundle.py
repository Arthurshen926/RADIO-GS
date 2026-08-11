#!/usr/bin/env python3
"""Materialize a label-free Gaussian bundle from a factorized field cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from radio_gs.config import load_config
from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)

from .build_likelihood_training_dataset import _load_json, sha256_file, validate_scene_split
from .evaluate_canonical_field import (
    _gaussian_covariances,
    _load_geometry_model,
    _read_official_geometry,
)
from .materialize_likelihood_primitive_bundle import (
    _validate_sha256,
    _write_json_no_clobber,
    _write_torch_no_clobber,
    build_bundle_payload,
    geometry_candidate_mappings,
)
from .protocol import quantize_scannet_points


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    allowed_fit = {"scene0002_00", "scene0005_00"}
    allowed_development = {"scene0003_00"}
    if args.scene_id not in allowed_fit | allowed_development:
        raise ValueError(
            "factorized bundle materializer is sealed to fit 0002/0005 and dev 0003"
        )
    if args.device != "cpu":
        raise ValueError("factorized bundle materialization is CPU-only")
    split_path = _validate_sha256(
        args.split_manifest,
        args.split_manifest_sha256,
        role="sealed likelihood scene split",
    )
    split = validate_scene_split(_load_json(split_path))
    partition = "fit" if args.scene_id in allowed_fit else "development_validation"
    if args.scene_id not in split["partitions"][partition]:
        raise PermissionError(
            f"{args.scene_id} is not in the frozen {partition} partition"
        )
    config_path = _validate_sha256(args.config, args.config_sha256, role="render config")
    geometry_path = _validate_sha256(
        args.geometry_checkpoint,
        args.geometry_checkpoint_sha256,
        role="Gaussian geometry checkpoint",
    )
    field_path = _validate_sha256(
        args.field_checkpoint,
        args.field_checkpoint_sha256,
        role="factorized canonical field",
    )
    state_path = _validate_sha256(
        args.factorized_primitive_state,
        args.factorized_primitive_state_sha256,
        role="factorized primitive state",
    )
    capability_path = _validate_sha256(
        args.capability_cache,
        args.capability_cache_sha256,
        role="compact canonical capability bank",
    )
    ply_path = Path(args.official_ply).expanduser().resolve()
    if not ply_path.is_file() or ply_path.stem != args.scene_id:
        raise FileNotFoundError("official source PLY geometry differs")

    bank = load_canonical_capability_bank(
        capability_path,
        expected_field_checkpoint_sha256=args.field_checkpoint_sha256,
        require_row_authority=True,
        require_formal_projection_order=True,
    )
    state = load_factorized_primitive_state(
        state_path,
        expected_sha256=args.factorized_primitive_state_sha256,
        expected_field_checkpoint_sha256=args.field_checkpoint_sha256,
        expected_factorized_radio_cache_sha256=args.factorized_radio_cache_sha256,
        expected_xyz=bank.xyz,
        expected_valid=bank.valid,
    )
    rows = bank.global_rows.long()
    if not torch.equal(rows, state.global_rows.long()):
        raise ValueError("capability and factorized-state compact row orders differ")
    features = bank.valid_feature_banks()
    primitive_xyz = bank.xyz.index_select(0, rows).float().contiguous()
    coverage = state.observation_evidence.float().clamp(0, 1).half()
    reliability = state.legacy_geometric_reliability().half()

    config = load_config(str(config_path))
    model = _load_geometry_model(config, str(geometry_path), torch.device("cpu"))
    try:
        full_xyz = model.get_xyz().detach().float().cpu()
        if full_xyz.shape != bank.xyz.shape or not torch.allclose(
            full_xyz, bank.xyz, atol=1e-6, rtol=0
        ):
            raise ValueError("frozen Gaussian geometry and capability rows differ")
        covariance = _gaussian_covariances(model).detach().index_select(0, rows).float().cpu()
        opacity = (
            model.get_opacity().detach().float().reshape(-1).index_select(0, rows).cpu()
        )
    finally:
        del model

    xyz, colors = _read_official_geometry(ply_path)
    # Geometry-only quantization: the PLY label property is deliberately never read.
    quantized = quantize_scannet_points(
        xyz,
        colors,
        np.zeros(len(xyz), dtype=np.int32),
        voxel_size=float(args.voxel_size),
    )
    official_points = torch.from_numpy(
        np.ascontiguousarray(
            quantized.raw_coordinates + xyz.min(axis=0, keepdims=True),
            dtype=np.float32,
        )
    )
    candidates, primitive_to_point = geometry_candidate_mappings(
        primitive_xyz,
        official_points,
        candidate_k=int(args.candidate_k),
    )
    source_paths = {
        "split_manifest": split_path,
        "config": config_path,
        "geometry_checkpoint": geometry_path,
        "field_checkpoint": field_path,
        "factorized_primitive_state": state_path,
        "capability_cache": capability_path,
        "official_ply": ply_path,
    }
    source_assets = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in source_paths.items()
    }
    payload = build_bundle_payload(
        scene_id=args.scene_id,
        primitive_xyz=primitive_xyz,
        primitive_covariance=covariance,
        primitive_opacity=opacity,
        appearance=features["appearance"],
        boundary=features["boundary"],
        prior_probability=torch.full((len(rows),), 0.5, dtype=torch.float16),
        coverage=coverage,
        reliability=reliability,
        global_rows=rows.to(torch.int32),
        official_point_xyz=official_points,
        point_candidate_indices=candidates,
        primitive_to_point_index=primitive_to_point,
        provenance={
            "source_assets": source_assets,
            "capability_source": bank.metadata.get("source"),
            "capability_signatures": {
                name: signature.to_dict() for name, signature in bank.signatures.items()
            },
            "factorized_reliability": {
                "coverage": "observation_evidence",
                "reliability": "sqrt((1-directional_dispersion)*observation_evidence)",
                "query_independent": True,
            },
            "official_point_domain": {
                "raw_vertex_count": int(len(xyz)),
                "quantized_point_count": int(len(official_points)),
                "voxel_size_m": float(args.voxel_size),
                "coordinate_contract": "agile3d_shifted_5cm_plus_scene_origin_world_v1",
                "label_property_opened": False,
            },
        },
    )
    output = _write_torch_no_clobber(Path(args.output), payload)
    receipt = {
        "schema_version": 1,
        "artifact_type": "agile3d-factorized-gaussian-bundle-receipt-v1",
        "status": f"sealed_query_independent_{partition}_bundle_ready",
        "scene_id": args.scene_id,
        "partition": partition,
        "bundle": {"path": str(output), "sha256": sha256_file(output)},
        "primitive_count": int(len(rows)),
        "official_point_count": int(len(official_points)),
        "candidate_k": int(candidates.shape[1]),
        "tensor_records": payload["tensor_records"],
        "source_assets": source_assets,
        "safety": {
            **payload["safety"],
            "development_labels_opened": False,
            "development_membership_read": partition == "development_validation",
            "fit_labels_opened": False,
            "fit_membership_read": partition == "fit",
        },
    }
    receipt_path = _write_json_no_clobber(Path(args.receipt), receipt)
    return receipt_path, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--geometry-checkpoint-sha256", required=True)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--field-checkpoint-sha256", required=True)
    parser.add_argument("--factorized-primitive-state", required=True)
    parser.add_argument("--factorized-primitive-state-sha256", required=True)
    parser.add_argument("--factorized-radio-cache-sha256", required=True)
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument("--capability-cache-sha256", required=True)
    parser.add_argument("--official-ply", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--candidate-k", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    path, receipt = materialize(parser.parse_args())
    print(json.dumps({"receipt": str(path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
