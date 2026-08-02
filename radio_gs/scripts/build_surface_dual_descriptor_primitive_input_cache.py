#!/usr/bin/env python3
"""Build the CPU primitive-row input consumed by the dual descriptor.

This is a converter for the real ``SurfaceRegionContractV2`` row cache, not a
synthetic tensor generator.  Every output row remains bound to the source
scene/seed primitive, the frozen surface readout, and the official RADIO
checkpoint.  The converter also replays the frozen official branch once so
the later independent materializer can require bitwise equality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts.materialize_surface_dual_descriptor import (
    DESCRIPTOR_DIMENSION,
    INPUT_ARTIFACT_TYPE,
    _QUERY_FREE_FLAGS,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_surface_region_summary_readout_v2,
    load_torch_mapping,
    sha256_file,
    validate_file_record,
    write_torch_noclobber,
)


SCHEMA_VERSION = 1
SOURCE_SCHEMA_VERSION = 3


def _false_target_flags(metadata: Mapping[str, Any], *, label: str) -> None:
    for key in (
        "uses_benchmark_scenes",
        "uses_benchmark_test_vocabulary",
        "annotations_opened",
        "labels_opened",
        "instances_opened",
        "masks_opened",
        "text_opened",
    ):
        if metadata.get(key) is not False:
            raise ValueError(f"{label} must explicitly certify {key}=false")


def _finite_cpu_tensor(
    value: object,
    *,
    label: str,
    floating: bool | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError(f"{label} must be a CPU tensor")
    if floating is True and not value.is_floating_point():
        raise ValueError(f"{label} must be floating point")
    if floating is False and value.is_floating_point():
        raise ValueError(f"{label} must be integer/bool")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains non-finite values")
    return value


def _validate_source_cache(
    path: Path,
    *,
    radio_sha256: str,
    base_contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, digest, source = load_torch_mapping(
        path,
        map_location="cpu",
        label="dual-descriptor source SurfaceRegion cache",
    )
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source}: SurfaceRegion metadata is missing")
    _false_target_flags(metadata, label=str(source))
    if (
        metadata.get("schema_version") != SOURCE_SCHEMA_VERSION
        or metadata.get("complete_scene_regions") is not True
        or metadata.get("physical_space_disjoint") is not True
        or metadata.get("failed_scenes") not in ({}, None)
        or metadata.get("radio_checkpoint_sha256") != radio_sha256
    ):
        raise ValueError(f"{source}: source SurfaceRegion authority differs")
    raw_contract = metadata.get("region_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError(f"{source}: frozen region contract is missing")
    try:
        expanded_contract = {
            **dict(raw_contract),
            "radii_m": tuple(raw_contract["radii_m"]),
        }
        expanded_contract.setdefault(
            "token_candidate_limit",
            int(raw_contract["maximum_tokens"]),
        )
        contract = SurfaceRegionContractV2(
            **expanded_contract
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source}: frozen region contract is invalid") from error
    contract.assert_compatible(metadata)
    if contract.digest != base_contract_sha256:
        raise ValueError(f"{source}: source/base region contracts differ")

    required = (
        "radio_features",
        "geometry",
        "token_mask",
        "reliability",
        "anchor_index",
    )
    if any(key not in payload for key in required):
        raise ValueError(f"{source}: SurfaceRegion row tensors are incomplete")
    features = _finite_cpu_tensor(
        payload["radio_features"], label=f"{source} radio_features", floating=True
    )
    geometry = _finite_cpu_tensor(
        payload["geometry"], label=f"{source} geometry", floating=True
    )
    mask = _finite_cpu_tensor(
        payload["token_mask"], label=f"{source} token_mask"
    ).bool()
    reliability = _finite_cpu_tensor(
        payload["reliability"], label=f"{source} reliability", floating=True
    )
    anchor = _finite_cpu_tensor(
        payload["anchor_index"], label=f"{source} anchor_index", floating=False
    ).long()
    rows = int(features.shape[0]) if features.ndim == 3 else 0
    if (
        rows <= 0
        or geometry.ndim != 3
        or geometry.shape[:2] != features.shape[:2]
        or mask.shape != features.shape[:2]
        or reliability.shape not in (features.shape[:2], (*features.shape[:2], 1))
        or anchor.shape != (rows,)
        or bool((anchor < 0).any())
        or bool((anchor >= features.shape[1]).any())
        or not bool(mask.any(dim=1).all())
        or not bool(mask[torch.arange(rows), anchor].all())
        or bool((reliability < 0).any())
    ):
        raise ValueError(f"{source}: SurfaceRegion row tensors are malformed")
    if bool(features[~mask].count_nonzero()) or bool(geometry[~mask].count_nonzero()):
        raise ValueError(f"{source}: padded SurfaceRegion tensors must be zero")
    records = metadata.get("region_records")
    if not isinstance(records, list) or len(records) != rows:
        raise ValueError(f"{source}: primitive region records do not align")
    row_authority = metadata.get("production_primitive_row_authority")
    required_authority = {
        "contract",
        "scene_id",
        "geometry_checkpoint",
        "geometry_xyz_sha256",
        "total_geometry_rows",
        "row_start",
        "row_stop",
        "row_order",
    }
    if not isinstance(row_authority, Mapping) or set(row_authority) != required_authority:
        raise ValueError(f"{source}: full production primitive-row authority is missing")
    scene_id = str(row_authority.get("scene_id", ""))
    total_rows = row_authority.get("total_geometry_rows")
    row_start = row_authority.get("row_start")
    row_stop = row_authority.get("row_stop")
    geometry_record = row_authority.get("geometry_checkpoint")
    geometry_path = validate_file_record(
        geometry_record, label=f"{source} production geometry checkpoint"
    )
    geometry_xyz_sha256 = str(row_authority.get("geometry_xyz_sha256", ""))
    if (
        row_authority.get("contract")
        != "complete_single_scene_gaussian_checkpoint_row_order_v1"
        or not scene_id
        or not isinstance(total_rows, int)
        or isinstance(total_rows, bool)
        or total_rows <= 0
        or not isinstance(row_start, int)
        or isinstance(row_start, bool)
        or not isinstance(row_stop, int)
        or isinstance(row_stop, bool)
        or not 0 <= row_start < row_stop <= total_rows
        or row_stop - row_start != rows
        or len(geometry_xyz_sha256) != 64
        or row_authority.get("row_order")
        != "zero_based_geometry_checkpoint_row_order"
    ):
        raise ValueError(f"{source}: production primitive-row authority differs")
    primitive_ids: list[str] = []
    for row, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{source}: primitive region record is malformed")
        scene = str(record.get("scene", ""))
        seed = record.get("seed")
        region_id = str(record.get("region_id", ""))
        if (
            not scene
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed != row_start + row
            or not region_id
            or int(record.get("anchor_local_index", -1)) != int(anchor[row])
            or int(record.get("tokens", -1)) != int(mask[row].sum())
        ):
            raise ValueError(f"{source}: primitive scene/seed binding differs")
        if scene != scene_id:
            raise ValueError(f"{source}: primitive scene differs from geometry authority")
        primitive_ids.append(f"{scene}/primitive-{seed}")
    if len(set(primitive_ids)) != rows:
        raise ValueError("SurfaceRegion sources repeat a scene/seed primitive")
    return {
        "primitive_ids": primitive_ids,
        "radio_features": features.contiguous(),
        "geometry": geometry.contiguous(),
        "token_mask": mask.contiguous(),
        "reliability": reliability.contiguous(),
        "anchor_index": anchor.contiguous(),
    }, {
        "path": str(source),
        "sha256": digest,
        "rows": rows,
        "scenes": sorted({value.split("/", 1)[0] for value in primitive_ids}),
        "region_contract_sha256": contract.digest,
        "production_primitive_row_authority": {
            **dict(row_authority),
            "geometry_checkpoint": {
                "path": str(geometry_path),
                "sha256": str(geometry_record["sha256"]),
            },
        },
    }


def _load_official_head(path: Path) -> torch.nn.Module:
    return SigLIP2SummaryHead.from_radio_checkpoint(str(path)).cpu().eval()


@torch.inference_mode()
def build(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.device).lower() != "cpu":
        raise ValueError("primitive input builder is CPU-only")
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"primitive input output must be new: {output}")
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    base_path = Path(args.base_checkpoint).resolve(strict=True)
    radio_path = Path(args.radio_checkpoint).resolve(strict=True)
    base, base_payload, base_sha, base_path = load_surface_region_summary_readout_v2(
        base_path, map_location="cpu"
    )
    provenance = base_payload.get("provenance")
    architecture = base_payload.get("architecture")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("frozen") is not True
        or provenance.get("scene_disjoint") is not True
        or provenance.get("uses_benchmark_scenes") is not False
        or provenance.get("uses_benchmark_test_vocabulary") is not False
        or provenance.get("custom_text_projection") is not False
        or not isinstance(architecture, Mapping)
        or not isinstance(architecture.get("contract_sha256"), str)
    ):
        raise ValueError("base checkpoint is not frozen target-blind official path")
    radio_sha = sha256_file(radio_path)
    loaded = [
        _validate_source_cache(
            Path(path),
            radio_sha256=radio_sha,
            base_contract_sha256=str(architecture["contract_sha256"]),
        )
        for path in args.source_cache
    ]
    if not loaded:
        raise ValueError("at least one production primitive-row source is required")
    authorities = [record["production_primitive_row_authority"] for _, record in loaded]
    if (
        len({authority["scene_id"] for authority in authorities}) != 1
        or len({authority["total_geometry_rows"] for authority in authorities}) != 1
        or len({authority["geometry_xyz_sha256"] for authority in authorities}) != 1
        or len(
            {
                json.dumps(authority["geometry_checkpoint"], sort_keys=True)
                for authority in authorities
            }
        )
        != 1
    ):
        raise ValueError("source shards bind different production geometries")
    order = sorted(
        range(len(loaded)),
        key=lambda index: authorities[index]["row_start"],
    )
    loaded = [loaded[index] for index in order]
    authorities = [authorities[index] for index in order]
    total_geometry_rows = int(authorities[0]["total_geometry_rows"])
    if (
        [int(authority["row_start"]) for authority in authorities]
        != [0, *[int(authority["row_stop"]) for authority in authorities[:-1]]]
        or int(authorities[-1]["row_stop"]) != total_geometry_rows
    ):
        raise ValueError("source shards do not cover every production geometry row exactly")
    primitive_ids = [value for rows, _ in loaded for value in rows["primitive_ids"]]
    if len(primitive_ids) != len(set(primitive_ids)):
        raise ValueError("source caches contain duplicate scene/seed primitives")
    tensor_keys = (
        "radio_features",
        "geometry",
        "token_mask",
        "reliability",
        "anchor_index",
    )
    try:
        merged = {
            key: torch.cat([rows[key] for rows, _ in loaded], dim=0)
            for key in tensor_keys
        }
    except RuntimeError as error:
        raise ValueError("source SurfaceRegion tensor contracts differ") from error
    base = base.cpu().eval().requires_grad_(False)
    head = _load_official_head(radio_path).requires_grad_(False)
    tokens: list[torch.Tensor] = []
    descriptors: list[torch.Tensor] = []
    for start in range(0, len(primitive_ids), batch_size):
        stop = min(len(primitive_ids), start + batch_size)
        token = base(
            merged["radio_features"][start:stop],
            merged["geometry"][start:stop],
            anchor_index=merged["anchor_index"][start:stop],
            token_mask=merged["token_mask"][start:stop],
            reliability=merged["reliability"][start:stop],
        ).float()
        descriptor = F.normalize(head(token[:, None])[:, 0].float(), dim=-1)
        if descriptor.shape != (stop - start, DESCRIPTOR_DIMENSION):
            raise ValueError("official descriptor dimension differs")
        tokens.append(token.cpu())
        descriptors.append(descriptor.cpu())
    source_records = [record for _, record in loaded]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": INPUT_ARTIFACT_TYPE,
        "primitive_ids": primitive_ids,
        **merged,
        "official_summary_tokens": torch.cat(tokens).contiguous(),
        "official_descriptors": torch.cat(descriptors).contiguous(),
        "metadata": {
            **{key: False for key in _QUERY_FREE_FLAGS},
            "complete_primitive_rows": True,
            "target_blind": True,
            "benchmark_targets_or_metrics_used": False,
            "source_artifact_type": "scannet_surface_region_contract_v2_cache",
            "primitive_id_semantics": "scene/primitive-global-seed",
            "source_caches": source_records,
            "production_primitive_row_authority": {
                "contract": authorities[0]["contract"],
                "scene_id": authorities[0]["scene_id"],
                "geometry_checkpoint": authorities[0]["geometry_checkpoint"],
                "geometry_xyz_sha256": authorities[0]["geometry_xyz_sha256"],
                "total_geometry_rows": total_geometry_rows,
                "row_order": authorities[0]["row_order"],
                "complete_geometry_rows_present": True,
            },
            "primitive_input_builder_implementation": file_record(
                Path(__file__).resolve()
            ),
            "base_checkpoint": {"path": str(base_path), "sha256": base_sha},
            "radio_checkpoint": {"path": str(radio_path), "sha256": radio_sha},
            "official_replay_generated_by": (
                "frozen_surface_readout_then_official_siglip2_summary_head"
            ),
            "device": "cpu",
        },
    }
    write_torch_noclobber(output, payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": f"{INPUT_ARTIFACT_TYPE}_report",
        "output": file_record(output),
        "primitive_rows": len(primitive_ids),
        "source_caches": source_records,
        "base_checkpoint": {"path": str(base_path), "sha256": base_sha},
        "radio_checkpoint": {"path": str(radio_path), "sha256": radio_sha},
        "target_blind": True,
        "device": "cpu",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", action="append", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = build(build_arg_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
