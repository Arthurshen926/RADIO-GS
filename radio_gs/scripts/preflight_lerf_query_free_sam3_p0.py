#!/usr/bin/env python3
"""Fail-closed preflight for a query-free LERF official-SAM3 P0 pilot.

The preflight binds a deterministic source-view subset to the frozen
exact-marginal authority and canonical Gaussian row identity.  It never opens
benchmark masks, query names, or evaluation RGB.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from radio_gs.scripts.build_lerf_query_free_sam3_exact_mpr_memberships import (
    EXPECTED_P0_GENERATION,
    validate_responsibility_authority,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_query_free_sam3_p0_preflight.v1"


def _values(raw: str) -> list[int]:
    values = [int(value) for value in str(raw).replace(",", " ").split() if value]
    if len(values) != 8 or len(values) != len(set(values)):
        raise ValueError("P0 sentinel requires exactly eight unique source frame IDs")
    return values


def _xyz_sha256(value: torch.Tensor) -> str:
    array = (
        torch.as_tensor(value).detach().float().cpu().contiguous().numpy()
        .astype("<f4", copy=False)
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON authority is not an object: {path}")
    return value


def build(args: argparse.Namespace) -> dict[str, Any]:
    responsibility_path = Path(args.responsibility_authority).expanduser().resolve()
    responsibility_bytes = responsibility_path.read_bytes()
    responsibility = json.loads(responsibility_bytes)
    metadata = dict(responsibility.get("metadata", {}))
    if (
        responsibility.get("schema")
        != "radio_gs.sparse_exact_marginal_responsibility_authority.v1"
        or int(responsibility.get("schema_version", -1)) != 1
    ):
        raise ValueError("responsibility authority schema differs")
    source_ids = _values(args.source_frame_ids)
    declared = [int(value) for value in responsibility.get("frame_indices", [])]
    excluded = {int(value) for value in metadata.get("excluded_frame_ids", [])}
    if (
        not declared
        or declared != [int(value) for value in metadata.get("selected_frame_indices", [])]
        or not set(source_ids).issubset(declared)
        or set(source_ids).intersection(excluded)
    ):
        raise ValueError("P0 frames are not an evaluation-disjoint exact-MPR source subset")
    forbidden = (
        "benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened",
    )
    if any(bool(metadata.get(key, False)) for key in forbidden):
        raise ValueError("responsibility authority violates source-only provenance")

    primitive_path = Path(args.primitive_query_cache).expanduser().resolve()
    primitive = torch.load(primitive_path, map_location="cpu", weights_only=False)
    xyz = torch.as_tensor(primitive.get("xyz")).float().cpu()
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("primitive query cache has no canonical [N,3] xyz")
    if str(metadata.get("xyz_sha256", "")) != _xyz_sha256(xyz):
        raise ValueError("primitive query cache rows differ from exact-MPR authority")
    responsibility = validate_responsibility_authority(
        responsibility,
        authority_path=responsibility_path,
        num_gaussians=len(xyz),
        xyz_sha256=_xyz_sha256(xyz),
    )

    graph_path = Path(args.support_graph).expanduser().resolve()
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    graph_metadata = dict(graph.get("metadata", {}))
    capability_metadata = dict(graph_metadata.get("capability_metadata", {}))
    provenance = {**capability_metadata, **graph_metadata}
    if provenance.get("query_independent") is not True or any(
        bool(provenance.get(key, False)) for key in (
            "benchmark_masks_opened", "text_queries_opened", "labels_opened",
            "instances_opened", "masks_opened", "text_opened",
        )
    ):
        raise ValueError("support graph lacks explicit query-free provenance")
    global_rows = torch.as_tensor(graph.get("global_rows")).long().cpu().reshape(-1)
    graph_xyz = torch.as_tensor(graph.get("xyz")).float().cpu()
    if (
        global_rows.shape != (len(graph_xyz),)
        or int(graph.get("num_global_rows", -1)) != len(xyz)
        or global_rows.numel() == 0
        or int(global_rows.min()) < 0
        or int(global_rows.max()) >= len(xyz)
        or not torch.equal(graph_xyz, xyz[global_rows])
    ):
        raise ValueError("support graph is not an exact subset of canonical primitive rows")

    image_root = Path(args.source_image_root).expanduser().resolve()
    construction_manifest_path = Path(
        args.construction_frame_manifest
    ).expanduser().resolve()
    construction_manifest_bytes = construction_manifest_path.read_bytes()
    construction_manifest = json.loads(construction_manifest_bytes)
    if not isinstance(construction_manifest, dict):
        raise ValueError("construction frame manifest is not an object")
    if Path(str(construction_manifest.get("image_dir", ""))).resolve() != image_root:
        raise ValueError("construction frame manifest source root differs")
    construction_records: dict[int, dict[str, Any]] = {}
    for raw in construction_manifest.get("frames", []):
        if not isinstance(raw, dict):
            raise ValueError("construction frame record differs")
        frame_id = int(raw.get("frame_idx", -1))
        if frame_id <= 0 or frame_id in construction_records:
            raise ValueError("construction frame identity repeats or differs")
        construction_records[frame_id] = dict(raw)
    if not set(source_ids).issubset(construction_records):
        raise ValueError("P0 source frames are absent from construction frame manifest")
    images: list[dict[str, Any]] = []
    for frame_id in source_ids:
        candidates = [
            image_root / f"frame_{frame_id:05d}{suffix}"
            for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
        ]
        matches = [path for path in candidates if path.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(f"cannot uniquely resolve source RGB frame {frame_id}")
        source_bytes = matches[0].read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        with Image.open(io.BytesIO(source_bytes)) as image:
            width, height = image.size
        construction_record = construction_records[frame_id]
        if (
            str(construction_record.get("source_file", "")) != matches[0].name
            or str(construction_record.get("source_sha256", "")) != source_sha256
        ):
            raise ValueError("source RGB bytes differ from construction frame manifest")
        images.append({
            "frame_id": frame_id,
            "path": str(matches[0].resolve()),
            "sha256": source_sha256,
            "height": int(height),
            "width": int(width),
            "construction_frame_record": construction_record,
        })

    sam_path = Path(args.sam_checkpoint).expanduser().resolve()
    if not sam_path.is_file():
        raise FileNotFoundError(f"official SAM3 checkpoint is absent: {sam_path}")
    expected_grid_size = int(args.expected_grid_size)
    if expected_grid_size <= 0:
        raise ValueError("expected grid size must be positive")
    checkpoint_sha256 = sha256_file(sam_path)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"preflight receipt already exists: {output}")
    report = {
        "schema": SCHEMA,
        "status": "ready_for_query_free_sparse_p0_pilot_generation",
        "scene": str(args.scene),
        "source_frame_count": len(source_ids),
        "source_frame_ids": source_ids,
        "excluded_evaluation_frame_ids": sorted(excluded),
        "source_images": images,
        "authorities": {
            "responsibility": {
                "path": str(responsibility_path),
                "sha256": hashlib.sha256(responsibility_bytes).hexdigest(),
            },
            "construction_frame_manifest": {
                "path": str(construction_manifest_path),
                "sha256": hashlib.sha256(construction_manifest_bytes).hexdigest(),
            },
            "support_graph": {
                "path": str(graph_path), "sha256": sha256_file(graph_path),
            },
            "primitive_query_cache": {
                "path": str(primitive_path), "sha256": sha256_file(primitive_path),
            },
            "official_sam3_checkpoint": {
                "path": str(sam_path), "sha256": checkpoint_sha256,
            },
        },
        "row_contract": {
            "num_global_rows": len(xyz),
            "num_graph_rows": len(graph_xyz),
            "xyz_sha256": _xyz_sha256(xyz),
            "graph_mapping": "explicit_global_rows_exact_tensor_identity",
        },
        "generation_contract": {
            **EXPECTED_P0_GENERATION,
            "checkpoint_sha256": checkpoint_sha256,
            "grid_size": expected_grid_size,
        },
        "access_audit": {
            "source_rgb_opened": True,
            "evaluation_rgb_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_query_names_opened": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--responsibility-authority", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--primitive-query-cache", required=True)
    parser.add_argument("--source-image-root", required=True)
    parser.add_argument("--construction-frame-manifest", required=True)
    parser.add_argument("--source-frame-ids", required=True)
    parser.add_argument("--sam-checkpoint", default="checkpoints/sam3_modelscope/sam3.pt")
    parser.add_argument("--expected-grid-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
