#!/usr/bin/env python3
"""Derive the deployment teacher scalars from an immutable canonical top-4 asset."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.scripts import materialize_lerf_o1_o2_streaming as _core
from radio_gs.scripts import materialize_lerf_o1_o2_teacher_agreement_streaming_v2 as _v2
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    validate_file_record,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.lerf_teacher_agreement_from_canonical_top4.v1"
SCHEMA_VERSION = 1


def contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "required_top4_schema": "radio_gs.lerf_source_teacher_view_siglip_authority.v1",
        "required_construction": "top4_exact_marginal_source_observation_then_official_siglip2_summary_v1",
        "required_retention_order": "marginal_mass_descending_then_frame_id_ascending",
        "source_view_count": 120,
        "maximum_views": 4,
        "teacher_mean": "fp16(normalize(sum(masked_fp16_top4),dim=descriptor))",
        "teacher_valid": "retained_view_count_gt_zero",
        "directional_resultant": "norm(sum(unit(masked_fp16_top4)))/retained_view_count",
        "query_independent": True,
        "target_data_or_metric_access": False,
    }


CONTRACT_SHA256 = canonical_json_sha256(contract())


def _record(path: str, digest: str, *, label: str) -> dict[str, str]:
    record = {"path": str(Path(path).expanduser().resolve()), "sha256": digest}
    validate_file_record(record, label=label)
    return record


def validate_payload(value: Mapping[str, Any]) -> None:
    required = {
        "schema", "schema_version", "scene_id", "global_rows", "teacher_mean",
        "teacher_valid", "retained_view_count", "teacher_view_directional_resultant",
        "producer", "top4_source", "base_descriptor", "contract", "contract_sha256",
        "teacher_mean_sha256", "teacher_view_directional_resultant_sha256",
        "access_audit", "metric_execution_authorized",
    }
    rows = value.get("global_rows")
    mean = value.get("teacher_mean")
    valid = value.get("teacher_valid")
    count = value.get("retained_view_count")
    resultant = value.get("teacher_view_directional_resultant")
    if (
        set(value) != required
        or value.get("schema") != SCHEMA
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("contract") != contract()
        or value.get("contract_sha256") != CONTRACT_SHA256
        or not torch.is_tensor(rows) or rows.ndim != 1 or rows.dtype != torch.int64
        or not torch.is_tensor(mean) or mean.shape != (rows.numel(), 1536) or mean.dtype != torch.float16
        or not torch.is_tensor(valid) or valid.shape != (rows.numel(),) or valid.dtype != torch.bool
        or not torch.is_tensor(count) or count.shape != (rows.numel(),) or count.dtype != torch.uint8
        or not torch.equal(valid, count > 0)
        or not torch.is_tensor(resultant) or resultant.shape != (rows.numel(),) or resultant.dtype != torch.float32
        or not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(resultant).all())
        or bool((mean[~valid] != 0).any()) or bool((resultant[~valid] != 0).any())
        or bool((resultant < 0).any()) or bool((resultant > 1).any())
        or value.get("teacher_mean_sha256") != _core.tensor_sha256_typed(mean)
        or value.get("teacher_view_directional_resultant_sha256") != _core.tensor_sha256_typed(resultant)
        or value.get("access_audit") != {
            "canonical_top4_source_opened": True,
            "accepted_v2_base_descriptor_opened": True,
            "query_embeddings_or_text_opened": False,
            "target_images_labels_masks_metrics_opened": False,
            "target_metric_executed": False,
        }
        or value.get("metric_execution_authorized") is not False
    ):
        raise ValueError("derived top4 teacher payload differs")
    for name in ("producer", "top4_source", "base_descriptor"):
        if not isinstance(value.get(name), Mapping):
            raise ValueError(f"derived top4 teacher {name} record differs")
        validate_file_record(value[name], label=f"derived top4 teacher {name}")


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if str(output) != str(args.output) or output.exists() or output.is_symlink():
        raise ValueError("derived teacher output differs")
    top4_record = _record(args.top4_source, args.top4_source_sha256, label="top4 source")
    base_record = _record(args.base_descriptor, args.base_descriptor_sha256, label="base descriptor")
    top4, _, _ = load_torch_mapping(
        top4_record["path"], expected_sha256=top4_record["sha256"], map_location="cpu",
        label="canonical top4 source",
    )
    base, rows = _core._validate_base_descriptor_general(
        Path(base_record["path"]), base_record["sha256"]
    )
    metadata = top4.get("metadata")
    descriptors = top4.get("teacher_view_descriptors")
    mask = top4.get("teacher_view_mask")
    frame_ids = top4.get("teacher_view_frame_ids")
    top4_rows = top4.get("global_rows")
    if (
        top4.get("schema") != "radio_gs.lerf_source_teacher_view_siglip_authority.v1"
        or top4.get("schema_version") != 1
        or not isinstance(metadata, Mapping)
        or metadata.get("construction") != contract()["required_construction"]
        or metadata.get("retention_order") != contract()["required_retention_order"]
        or metadata.get("maximum_views") != 4
        or metadata.get("source_view_count") != 120
        or metadata.get("base_descriptor") != base_record
        or metadata.get("query_independent") is not True
        or metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
        or metadata.get("text_queries_opened") is not False
        or metadata.get("target_frames_opened") is not False
        or not torch.equal(torch.as_tensor(top4_rows), rows)
        or not torch.is_tensor(descriptors) or descriptors.shape != (rows.numel(), 4, 1536)
        or descriptors.dtype != torch.float16
        or not torch.is_tensor(mask) or mask.shape != descriptors.shape[:2] or mask.dtype != torch.bool
        or not torch.is_tensor(frame_ids) or frame_ids.shape != mask.shape
        or not torch.equal(mask, frame_ids >= 0)
        or bool((descriptors[~mask] != 0).any())
        or base["features_by_scale"].shape != (rows.numel(), 3, 1536)
    ):
        raise ValueError("canonical top4/base lineage differs")
    count = mask.sum(dim=1).to(torch.uint8).contiguous()
    valid = (count > 0).contiguous()
    mean_float = F.normalize((descriptors.float() * mask[..., None]).sum(dim=1), dim=-1)
    mean_float[~valid] = 0
    mean = mean_float.half().contiguous()
    resultant, resultant_count = _v2.directional_resultant_from_canonical_top_views(
        descriptors, frame_ids
    )
    if not torch.equal(count, resultant_count):
        raise ValueError("derived teacher top4 count differs")
    payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": args.scene_id,
        "global_rows": rows.clone(),
        "teacher_mean": mean,
        "teacher_valid": valid,
        "retained_view_count": count,
        "teacher_view_directional_resultant": resultant,
        "producer": file_record(Path(__file__).resolve()),
        "top4_source": top4_record,
        "base_descriptor": base_record,
        "contract": contract(),
        "contract_sha256": CONTRACT_SHA256,
        "teacher_mean_sha256": _core.tensor_sha256_typed(mean),
        "teacher_view_directional_resultant_sha256": _core.tensor_sha256_typed(resultant),
        "access_audit": {
            "canonical_top4_source_opened": True,
            "accepted_v2_base_descriptor_opened": True,
            "query_embeddings_or_text_opened": False,
            "target_images_labels_masks_metrics_opened": False,
            "target_metric_executed": False,
        },
        "metric_execution_authorized": False,
    }
    validate_payload(payload)
    write_torch_noclobber(output, payload)
    return {
        "status": "complete_query_free_top4_teacher_derivation",
        "scene_id": args.scene_id,
        "accepted_rows": int(rows.numel()),
        "teacher_valid_rows": int(valid.sum()),
        "output": file_record(output),
        "metric_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--top4-source", required=True)
    parser.add_argument("--top4-source-sha256", required=True)
    parser.add_argument("--base-descriptor", required=True)
    parser.add_argument("--base-descriptor-sha256", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(materialize(parser.parse_args()), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

