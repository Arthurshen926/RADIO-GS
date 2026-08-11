#!/usr/bin/env python3
"""Open the one preregistered ScanNet development semantic sidecar after gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.data.scannet_source_region_semantics import (
    PREREGISTERED_DEVELOPMENT_SCENES,
    build_development_region_semantic_sidecar,
    load_scannet_raw_to_nyu40,
    official_vertex_nyu40_labels,
    sha256_file,
    validate_development_region_semantic_sidecar,
)
from radio_gs.scripts.build_scannet_source_region_semantic_sidecar import (
    _read_mesh_xyz,
    _write_json_noclobber,
    _write_torch_noclobber,
)
from radio_gs.scripts.validate_source_text_likelihood_source_gates import (
    GATE_SCHEMA,
)


RECEIPT_SCHEMA = "radio_gs.scannet_development_region_semantic_sidecar_receipt.v1"


def run(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    scene_id = str(args.scene_id)
    if scene_id not in PREREGISTERED_DEVELOPMENT_SCENES:
        raise PermissionError("scene is not the single preregistered development scene")
    gate_path = Path(args.source_gate_receipt).expanduser().resolve(strict=True)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("schema") != GATE_SCHEMA or gate.get("status") != (
        "source_gates_passed_single_development_open_authorized"
    ):
        raise PermissionError("source gates have not authorized development open")
    authorization = gate.get("development_authorization", {})
    if (
        authorization.get("scene_id") != scene_id
        or authorization.get("maximum_opens") != 1
        or authorization.get("parameter_callback_allowed") is not False
        or authorization.get("benchmark_metric_allowed") is not False
    ):
        raise PermissionError("development authorization differs")

    # Semantic files are resolved/read only after the source gate above passes.
    paths = {
        "accepted_region_authority": Path(args.accepted_region_authority)
        .expanduser()
        .resolve(strict=True),
        "factorized_field_authority": Path(args.factorized_field_authority)
        .expanduser()
        .resolve(strict=True),
        "official_mesh": Path(args.official_mesh).expanduser().resolve(strict=True),
        "official_segmentation": Path(args.official_segmentation)
        .expanduser()
        .resolve(strict=True),
        "official_aggregation": Path(args.official_aggregation)
        .expanduser()
        .resolve(strict=True),
        "official_label_tsv": Path(args.official_label_tsv)
        .expanduser()
        .resolve(strict=True),
        "official_train_split": Path(args.official_train_split)
        .expanduser()
        .resolve(strict=True),
        "source_gate_receipt": gate_path,
    }
    train_scenes = {
        line.strip()
        for line in paths["official_train_split"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if scene_id not in train_scenes:
        raise PermissionError("development scene is not in official ScanNet train")
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
    sidecar = build_development_region_semantic_sidecar(
        scene_id=scene_id,
        accepted_region_payload=torch.load(
            paths["accepted_region_authority"], map_location="cpu", weights_only=True
        ),
        factorized_field_payload=torch.load(
            paths["factorized_field_authority"], map_location="cpu", weights_only=True
        ),
        official_mesh_xyz=mesh_xyz,
        official_vertex_labels=vertex_labels,
        lineage_paths=paths,
        official_label_audit=label_audit,
    )
    validate_development_region_semantic_sidecar(sidecar)
    output = _write_torch_noclobber(args.output, sidecar)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": "complete_single_heldout_development_semantic_open",
        "scene_id": scene_id,
        "sidecar": {"path": str(output), "sha256": sha256_file(output)},
        "source_gate_receipt": {
            "path": str(gate_path),
            "sha256": sha256_file(gate_path),
        },
        "statistics": dict(sidecar["statistics"]),
        "source_access": dict(sidecar["source_access"]),
        "parameter_callback_allowed": False,
        "benchmark_metric_run": False,
    }
    receipt_path = _write_json_noclobber(args.receipt, receipt)
    return output, receipt_path, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True, choices=PREREGISTERED_DEVELOPMENT_SCENES)
    parser.add_argument("--source-gate-receipt", required=True)
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
    _output, path, receipt = run(args)
    print(json.dumps({"receipt": str(path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
