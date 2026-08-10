#!/usr/bin/env python3
"""Assemble one formal source-only V2 region co-membership authority."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.models.region_comembership_v2 import PAIR_FEATURE_NAMES
from radio_gs.scripts.audit_region_comembership_v2_scene1_to_scene2 import (
    capability_pair_features,
)
from radio_gs.scripts.build_source_region_comembership_v1 import (
    _load_exact_instance_mass,
)
from radio_gs.scripts.materialize_region_capability_descriptors_v2 import (
    validate_region_capability_descriptor_authority,
)
from radio_gs.scripts.train_source_region_comembership_v1 import (
    TRAIN_SCENES,
    VALIDATION_SCENES,
    load_scene_authority,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    validate_file_record,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.source_region_comembership_authority.v2"
SCHEMA_VERSION = 2
PREREGISTRATION = Path(
    "paper/artifacts/source_only_region_comembership_v2_preregistration_20260807.json"
)
EFFICIENCY_ADDENDUM = Path(
    "paper/artifacts/source_only_region_comembership_v2_formal_selection_efficiency_addendum_20260807.json"
)
TENSOR_CHANNELS = (
    "canonical_region_indices",
    "region_rows",
    "token_mask",
    "dominant_instance_ids",
    "instance_purity",
    "instance_label_coverage",
    "instance_observed",
    "pair_indices",
    "pair_features",
    "same_instance_targets",
    "pair_evidence_weights",
    "primitive_instance_flat_keys",
    "primitive_instance_mass",
)


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


def sparsify_primitive_instance_mass(
    dense_mass: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dense = torch.as_tensor(dense_mass).detach().float().cpu()
    if (
        dense.ndim != 2
        or dense.shape[0] <= 0
        or dense.shape[1] < 2
        or not bool(torch.isfinite(dense).all())
        or bool((dense < 0).any())
    ):
        raise ValueError("V2 dense primitive-instance mass differs")
    coordinates = torch.nonzero(dense > 0, as_tuple=False)
    if coordinates.numel() == 0:
        raise ValueError("V2 primitive-instance mass is empty")
    columns = int(dense.shape[1])
    keys = coordinates[:, 0] * columns + coordinates[:, 1]
    mass = dense[coordinates[:, 0], coordinates[:, 1]]
    return keys.long().contiguous(), mass.float().contiguous()


def densify_primitive_instance_mass(
    *,
    flat_keys: torch.Tensor,
    mass: torch.Tensor,
    primitive_count: int,
    instance_columns_including_zero: int,
) -> torch.Tensor:
    keys = torch.as_tensor(flat_keys).detach().long().cpu()
    values = torch.as_tensor(mass).detach().float().cpu()
    primitives = int(primitive_count)
    columns = int(instance_columns_including_zero)
    if (
        keys.ndim != 1
        or values.shape != keys.shape
        or keys.numel() <= 0
        or primitives <= 0
        or columns < 2
        or bool((keys < 0).any())
        or bool((keys >= primitives * columns).any())
        or (keys.numel() > 1 and not bool((keys[1:] > keys[:-1]).all()))
        or not bool(torch.isfinite(values).all())
        or bool((values <= 0).any())
    ):
        raise ValueError("V2 sparse primitive-instance mass differs")
    dense = torch.zeros(primitives, columns, dtype=torch.float32)
    dense[keys // columns, keys % columns] = values
    return dense.contiguous()


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "schema",
        "schema_version",
        "scene_id",
        "split",
        "producer",
        "preregistration",
        "efficiency_addendum",
        "input_authority",
        "pair_feature_names",
        "pair_feature_names_sha256",
        "primitive_metric_contract",
        "source_access",
    )
    return {name: payload[name] for name in names}


def validate_source_region_comembership_v2(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source V2 co-membership authority must be a mapping")
    payload = dict(value)
    required = {
        "schema",
        "schema_version",
        "scene_id",
        "split",
        "producer",
        "preregistration",
        "efficiency_addendum",
        "input_authority",
        "pair_feature_names",
        "pair_feature_names_sha256",
        "primitive_metric_contract",
        "source_access",
        "content_authority_sha256",
        "region_fingerprints",
        *TENSOR_CHANNELS,
        "primitive_count",
        "instance_columns_including_zero",
        "channel_sha256",
        "audit",
    }
    if (
        set(payload) != required
        or payload.get("schema") != SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("scene_id")
        not in set(TRAIN_SCENES) | set(VALIDATION_SCENES)
        or payload.get("split")
        != (
            "source_train"
            if payload.get("scene_id") in TRAIN_SCENES
            else "source_validation"
        )
        or payload.get("pair_feature_names") != list(PAIR_FEATURE_NAMES)
        or payload.get("pair_feature_names_sha256")
        != canonical_json_sha256(list(PAIR_FEATURE_NAMES))
        or payload.get("source_access") != source_access()
        or payload.get("primitive_metric_contract")
        != {
            "storage": "sorted_sparse_flat_primitive_times_instance_column",
            "instance_zero": "included_as_selected_contamination_never_target",
            "selected_union": "deduplicate_primitive_rows_before_mass_sum",
        }
        or payload.get("content_authority_sha256")
        != canonical_json_sha256(_identity(payload))
    ):
        raise ValueError("source V2 co-membership authority header differs")
    for name in ("producer", "preregistration", "efficiency_addendum"):
        validate_file_record(payload[name], label=f"source V2 {name}")
    if not isinstance(payload["input_authority"], Mapping) or set(
        payload["input_authority"]
    ) != {"v1_source_authority", "capability_descriptor", "exact_marginal", "instance_zip"}:
        raise ValueError("source V2 input authority differs")
    for name, record in payload["input_authority"].items():
        validate_file_record(record, label=f"source V2 {name}")

    canonical = torch.as_tensor(payload["canonical_region_indices"])
    rows = torch.as_tensor(payload["region_rows"])
    mask = torch.as_tensor(payload["token_mask"])
    pairs = torch.as_tensor(payload["pair_indices"])
    features = torch.as_tensor(payload["pair_features"])
    targets = torch.as_tensor(payload["same_instance_targets"])
    weights = torch.as_tensor(payload["pair_evidence_weights"])
    dominant = torch.as_tensor(payload["dominant_instance_ids"])
    purity = torch.as_tensor(payload["instance_purity"])
    coverage = torch.as_tensor(payload["instance_label_coverage"])
    observed = torch.as_tensor(payload["instance_observed"])
    count = int(canonical.numel())
    primitive_count = int(payload["primitive_count"])
    columns = int(payload["instance_columns_including_zero"])
    if (
        count <= 0
        or canonical.dtype != torch.int64
        or canonical.ndim != 1
        or rows.dtype != torch.int64
        or rows.ndim != 2
        or rows.shape[0] != count
        or mask.dtype != torch.bool
        or mask.shape != rows.shape
        or not bool(mask.any(dim=1).all())
        or bool((rows[mask] < 0).any())
        or bool((rows[mask] >= primitive_count).any())
        or pairs.dtype != torch.int64
        or pairs.ndim != 2
        or pairs.shape[0] != 2
        or features.dtype != torch.float32
        or features.shape != (pairs.shape[1], len(PAIR_FEATURE_NAMES))
        or targets.dtype != torch.bool
        or targets.shape != (pairs.shape[1],)
        or weights.dtype != torch.float32
        or weights.shape != targets.shape
        or bool((pairs < 0).any())
        or bool((pairs >= count).any())
        or bool((pairs[0] >= pairs[1]).any())
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or dominant.dtype != torch.int64
        or dominant.shape != (count,)
        or purity.dtype != torch.float32
        or purity.shape != (count,)
        or coverage.dtype != torch.float32
        or coverage.shape != (count,)
        or observed.dtype != torch.bool
        or observed.shape != (count,)
        or len(payload["region_fingerprints"]) != count
    ):
        raise ValueError("source V2 co-membership tensor axes differ")
    pair_keys = pairs[0] * count + pairs[1]
    if pair_keys.numel() <= 0 or (
        pair_keys.numel() > 1 and not bool((pair_keys[1:] > pair_keys[:-1]).all())
    ):
        raise ValueError("source V2 pair keys are not sorted unique")
    densify_primitive_instance_mass(
        flat_keys=payload["primitive_instance_flat_keys"],
        mass=payload["primitive_instance_mass"],
        primitive_count=primitive_count,
        instance_columns_including_zero=columns,
    )
    channels = payload["channel_sha256"]
    if not isinstance(channels, Mapping) or set(channels) != set(TENSOR_CHANNELS):
        raise ValueError("source V2 channel authority differs")
    for name in TENSOR_CHANNELS:
        if channels[name] != tensor_sha256(payload[name]):
            raise ValueError(f"source V2 channel changed: {name}")
    return payload


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"source V2 authority exists: {output}")
    v1_record = {
        "path": str(Path(args.v1_source_authority).expanduser().resolve()),
        "sha256": str(args.expected_v1_source_authority_sha256),
    }
    scene_id = str(args.scene_id)
    split = "source_train" if scene_id in TRAIN_SCENES else "source_validation"
    base = load_scene_authority(
        v1_record, expected_scene_id=scene_id, expected_split=split
    )
    v1_payload, _, _ = load_torch_mapping(
        v1_record["path"],
        expected_sha256=v1_record["sha256"],
        map_location="cpu",
        label="V2 input V1 source authority",
    )
    descriptor_record = {
        "path": str(Path(args.capability_descriptor).expanduser().resolve()),
        "sha256": str(args.expected_capability_descriptor_sha256),
    }
    descriptor_raw, descriptor_sha, descriptor_path = load_torch_mapping(
        descriptor_record["path"],
        expected_sha256=descriptor_record["sha256"],
        map_location="cpu",
        label="V2 capability descriptor",
    )
    descriptor = validate_region_capability_descriptor_authority(descriptor_raw)
    if (
        descriptor["scene_id"] != scene_id
        or descriptor["input_authority"]["accepted_v2"]["sha256"]
        != v1_payload["input_authority"]["accepted_v2"]["sha256"]
        or list(descriptor["region_fingerprints"])
        != list(v1_payload["region_fingerprints"])
        or not torch.equal(
            descriptor["canonical_region_indices"],
            v1_payload["canonical_region_indices"],
        )
    ):
        raise ValueError("V2 source inputs do not share the canonical axis")
    appended = capability_pair_features(
        pair_indices=base.pair_indices,
        appearance_direction=descriptor["appearance_direction"],
        boundary_direction=descriptor["boundary_direction"],
        appearance_concentration=descriptor["appearance_concentration"],
        boundary_concentration=descriptor["boundary_concentration"],
    )
    pair_features = torch.cat((base.pair_features, appended), dim=1).contiguous()
    exact = dict(v1_payload["input_authority"]["exact_marginal"])
    instance_zip = dict(v1_payload["input_authority"]["instance_zip"])
    dense_mass, instance_audit, exact_record = _load_exact_instance_mass(
        manifest_path=Path(exact["path"]),
        manifest_sha256=exact["sha256"],
        instance_zip=Path(instance_zip["path"]),
        instance_zip_sha256=instance_zip["sha256"],
    )
    if (
        int(descriptor["region_rows"][descriptor["token_mask"]].max())
        >= dense_mass.shape[0]
    ):
        raise ValueError("V2 primitive instance and canonical row axes differ")
    flat_keys, sparse_mass = sparsify_primitive_instance_mass(dense_mass)
    root = Path(__file__).resolve().parents[2]
    identity = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "split": split,
        "producer": file_record(Path(__file__).resolve()),
        "preregistration": file_record(root / PREREGISTRATION),
        "efficiency_addendum": file_record(root / EFFICIENCY_ADDENDUM),
        "input_authority": {
            "v1_source_authority": dict(base.record),
            "capability_descriptor": {
                "path": str(descriptor_path),
                "sha256": descriptor_sha,
            },
            "exact_marginal": exact_record,
            "instance_zip": instance_zip,
        },
        "pair_feature_names": list(PAIR_FEATURE_NAMES),
        "pair_feature_names_sha256": canonical_json_sha256(
            list(PAIR_FEATURE_NAMES)
        ),
        "primitive_metric_contract": {
            "storage": "sorted_sparse_flat_primitive_times_instance_column",
            "instance_zero": "included_as_selected_contamination_never_target",
            "selected_union": "deduplicate_primitive_rows_before_mass_sum",
        },
        "source_access": source_access(),
    }
    payload = {
        **identity,
        "content_authority_sha256": canonical_json_sha256(identity),
        "region_fingerprints": list(v1_payload["region_fingerprints"]),
        "canonical_region_indices": v1_payload["canonical_region_indices"]
        .long()
        .cpu()
        .contiguous(),
        "region_rows": descriptor["region_rows"].long().cpu().contiguous(),
        "token_mask": descriptor["token_mask"].bool().cpu().contiguous(),
        "dominant_instance_ids": base.dominant_instance_ids,
        "instance_purity": base.instance_purity,
        "instance_label_coverage": base.instance_label_coverage,
        "instance_observed": base.instance_observed,
        "pair_indices": base.pair_indices,
        "pair_features": pair_features,
        "same_instance_targets": base.targets,
        "pair_evidence_weights": base.evidence_weights,
        "primitive_instance_flat_keys": flat_keys,
        "primitive_instance_mass": sparse_mass,
        "primitive_count": int(dense_mass.shape[0]),
        "instance_columns_including_zero": int(dense_mass.shape[1]),
        "channel_sha256": {},
        "audit": {
            **instance_audit,
            "regions": base.region_count,
            "candidate_pairs": int(base.pair_indices.shape[1]),
            "pair_feature_dimension": len(PAIR_FEATURE_NAMES),
            "sparse_primitive_instance_cells": int(flat_keys.numel()),
            "sparse_storage_bytes": int(
                flat_keys.numel() * flat_keys.element_size()
                + sparse_mass.numel() * sparse_mass.element_size()
            ),
        },
    }
    payload["channel_sha256"] = {
        name: tensor_sha256(payload[name]) for name in TENSOR_CHANNELS
    }
    validate_source_region_comembership_v2(payload)
    written = write_torch_noclobber(output, payload)
    return {
        "status": "source_region_comembership_v2_complete",
        "scene_id": scene_id,
        "split": split,
        "output": file_record(written),
        "audit": payload["audit"],
        "target_metric_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--v1-source-authority", required=True)
    parser.add_argument("--expected-v1-source-authority-sha256", required=True)
    parser.add_argument("--capability-descriptor", required=True)
    parser.add_argument("--expected-capability-descriptor-sha256", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(materialize(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
