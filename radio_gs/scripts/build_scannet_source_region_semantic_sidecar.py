#!/usr/bin/env python3
"""Build an immutable official-ScanNet semantic sidecar for source regions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from plyfile import PlyData
import torch

from radio_gs.data.scannet_source_region_semantics import (
    PREREGISTERED_SOURCE_FIT_SCENES,
    build_source_region_semantic_sidecar,
    load_scannet_raw_to_nyu40,
    official_vertex_nyu40_labels,
    sha256_file,
    validate_source_region_semantic_sidecar,
)


RECEIPT_SCHEMA = "radio_gs.scannet_source_region_semantic_sidecar_receipt.v1"


def _canonical_output(path: str | Path) -> Path:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    raw.parent.mkdir(parents=True, exist_ok=True)
    return raw.parent.resolve(strict=True) / raw.name


def _write_torch_noclobber(path: str | Path, value: Mapping[str, Any]) -> Path:
    output = _canonical_output(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable output already exists: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _write_json_noclobber(path: str | Path, value: Mapping[str, Any]) -> Path:
    output = _canonical_output(path)
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
            raise ValueError(f"refusing to replace different artifact: {output}")
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _read_mesh_xyz(path: str | Path) -> np.ndarray:
    source = Path(path).expanduser().resolve(strict=True)
    vertices = PlyData.read(source)["vertex"].data
    names = set(vertices.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("official ScanNet mesh lacks xyz")
    return np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
        np.float32, copy=False
    )


def run(
    *,
    scene_id: str,
    accepted_region_authority: str | Path,
    factorized_field_authority: str | Path,
    official_mesh: str | Path,
    official_segmentation: str | Path,
    official_aggregation: str | Path,
    official_label_tsv: str | Path,
    official_train_split: str | Path,
    output: str | Path,
    receipt: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    if scene_id not in PREREGISTERED_SOURCE_FIT_SCENES:
        raise PermissionError("scene is not a preregistered source-fit scene")
    paths = {
        "accepted_region_authority": Path(accepted_region_authority)
        .expanduser()
        .resolve(strict=True),
        "factorized_field_authority": Path(factorized_field_authority)
        .expanduser()
        .resolve(strict=True),
        "official_mesh": Path(official_mesh).expanduser().resolve(strict=True),
        "official_segmentation": Path(official_segmentation)
        .expanduser()
        .resolve(strict=True),
        "official_aggregation": Path(official_aggregation)
        .expanduser()
        .resolve(strict=True),
        "official_label_tsv": Path(official_label_tsv)
        .expanduser()
        .resolve(strict=True),
        "official_train_split": Path(official_train_split)
        .expanduser()
        .resolve(strict=True),
    }
    train_scenes = {
        line.strip()
        for line in paths["official_train_split"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if scene_id not in train_scenes:
        raise PermissionError("scene is not in the official ScanNet training split")
    mesh_xyz = _read_mesh_xyz(paths["official_mesh"])
    segmentation = json.loads(
        paths["official_segmentation"].read_text(encoding="utf-8")
    )
    aggregation = json.loads(
        paths["official_aggregation"].read_text(encoding="utf-8")
    )
    vertex_labels, label_audit = official_vertex_nyu40_labels(
        scene_id=scene_id,
        vertex_count=len(mesh_xyz),
        segmentation=segmentation,
        aggregation=aggregation,
        raw_to_nyu40=load_scannet_raw_to_nyu40(paths["official_label_tsv"]),
    )
    accepted = torch.load(
        paths["accepted_region_authority"], map_location="cpu", weights_only=True
    )
    factorized = torch.load(
        paths["factorized_field_authority"], map_location="cpu", weights_only=True
    )
    sidecar = build_source_region_semantic_sidecar(
        scene_id=scene_id,
        accepted_region_payload=accepted,
        factorized_field_payload=factorized,
        official_mesh_xyz=mesh_xyz,
        official_vertex_labels=vertex_labels,
        lineage_paths=paths,
        official_label_audit=label_audit,
    )
    validate_source_region_semantic_sidecar(sidecar)
    output_path = _write_torch_noclobber(output, sidecar)
    receipt_payload = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": "complete_preregistered_source_fit_semantic_alignment",
        "scene_id": scene_id,
        "sidecar": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "statistics": dict(sidecar["statistics"]),
        "source_access": dict(sidecar["source_access"]),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    receipt_path = _write_json_noclobber(receipt, receipt_payload)
    return output_path, receipt_path, receipt_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True, choices=PREREGISTERED_SOURCE_FIT_SCENES)
    parser.add_argument("--accepted-region-authority", required=True)
    parser.add_argument("--factorized-field-authority", required=True)
    parser.add_argument("--official-mesh", required=True)
    parser.add_argument("--official-segmentation", required=True)
    parser.add_argument("--official-aggregation", required=True)
    parser.add_argument("--official-label-tsv", required=True)
    parser.add_argument("--official-train-split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    _output, _receipt, payload = run(
        scene_id=args.scene_id,
        accepted_region_authority=args.accepted_region_authority,
        factorized_field_authority=args.factorized_field_authority,
        official_mesh=args.official_mesh,
        official_segmentation=args.official_segmentation,
        official_aggregation=args.official_aggregation,
        official_label_tsv=args.official_label_tsv,
        official_train_split=args.official_train_split,
        output=args.output,
        receipt=args.receipt,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
