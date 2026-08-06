#!/usr/bin/env python3
"""Materialize a bounded source-only LERF teacher-view SigLIP authority.

The producer replays the exact registered source observations already sealed
by the 120-view sparse exact-marginal authority.  For each AcceptedV2
primitive row it retains the four views with the largest exact marginal target
mass, then projects the per-view RADIO observation through the singleton
official SigLIP2 summary head.  Benchmark frames, masks, labels, queries, and
metrics are forbidden.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    _load_bundle_feature_maps,
    _validated_feature_bundle,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    accumulate_raster_contribution_features,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.lerf_source_teacher_view_siglip_authority.v1"
SCHEMA_VERSION = 1
TARGET_FRAME_IDS = (41, 105, 152, 195)
EXPECTED_VIEW_COUNT = 120
EXPECTED_POLICY = {
    "assignment_mode": "exact_front_to_back_sparse_marginal",
    "registration_weight_mode": "exact_front_to_back_marginal_responsibility",
    "benchmark_images_opened": False,
    "benchmark_masks_opened": False,
    "text_queries_opened": False,
}


def _validate_responsibility_authority(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_xyz_sha256: str,
) -> tuple[dict[str, Any], Path]:
    payload, _, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="120-view exact-marginal responsibility authority",
    )
    required = {
        "formula_contract", "formula_sha256", "frame_indices", "metadata",
        "num_gaussians", "num_pixels", "schema", "schema_version",
        "total_hits", "views",
    }
    metadata = payload.get("metadata")
    formula = payload.get("formula_contract")
    views = payload.get("views")
    frames = payload.get("frame_indices")
    if (
        set(payload) != required
        or payload.get("schema")
        != "radio_gs.sparse_exact_marginal_responsibility_authority.v1"
        or payload.get("schema_version") != 1
        or not isinstance(metadata, Mapping)
        or any(metadata.get(key) != expected for key, expected in EXPECTED_POLICY.items())
        or metadata.get("query_independent") is not True
        or metadata.get("xyz_sha256") != expected_xyz_sha256
        or metadata.get("excluded_frame_ids") != list(TARGET_FRAME_IDS)
        or metadata.get("selected_frame_indices") != frames
        or not isinstance(formula, Mapping)
        or formula.get("query_independent") is not True
        or formula.get("feature_independent") is not True
        or payload.get("formula_sha256") != canonical_json_sha256(formula)
        or not isinstance(views, list)
        or not isinstance(frames, list)
        or len(views) != EXPECTED_VIEW_COUNT
        or len(frames) != EXPECTED_VIEW_COUNT
        or len(frames) != len(set(frames))
        or set(frames).intersection(TARGET_FRAME_IDS)
        or frames != [int(record.get("frame_index", -1)) for record in views]
        or int(payload.get("num_pixels", -1)) <= 0
        or int(payload.get("num_gaussians", -1)) <= 0
        or int(payload.get("total_hits", -1))
        != sum(int(record.get("num_hits", -1)) for record in views)
    ):
        raise ValueError("120-view exact-marginal authority differs")
    last: tuple[int, int] | None = None
    for position, record in enumerate(views):
        if not isinstance(record, Mapping) or set(record) != {
            "frame_index", "num_hits", "relative_path", "sha256", "view_index"
        }:
            raise ValueError("responsibility view record differs")
        key = (int(record["frame_index"]), int(record["view_index"]))
        relative = Path(str(record["relative_path"]))
        if (
            int(record["num_hits"]) < 0
            or int(record["view_index"]) != position
            or relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or (last is not None and key <= last)
        ):
            raise ValueError("responsibility view order/path differs")
        last = key
    return dict(payload), source


def _load_responsibility_view(
    authority: Mapping[str, Any],
    authority_path: Path,
    record: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = (authority_path.parent / str(record["relative_path"])).resolve()
    if authority_path.parent not in path.parents:
        raise ValueError("responsibility view escapes its authority root")
    payload, _, _ = load_torch_mapping(
        path,
        expected_sha256=str(record["sha256"]),
        map_location="cpu",
        label="exact-marginal responsibility view",
    )
    required = {
        "schema", "schema_version", "formula_sha256", "view_index",
        "frame_index", "num_gaussians", "num_pixels", "gaussian_ids",
        "pixel_ids", "base_weights",
    }
    gaussian = torch.as_tensor(payload.get("gaussian_ids")).long().cpu()
    pixels = torch.as_tensor(payload.get("pixel_ids")).long().cpu()
    base = torch.as_tensor(payload.get("base_weights")).float().cpu()
    count = int(record["num_hits"])
    if (
        set(payload) != required
        or payload.get("schema")
        != "radio_gs.sparse_exact_marginal_responsibility_view.v1"
        or payload.get("schema_version") != 1
        or payload.get("formula_sha256") != authority["formula_sha256"]
        or int(payload.get("view_index", -1)) != int(record["view_index"])
        or int(payload.get("frame_index", -1)) != int(record["frame_index"])
        or int(payload.get("num_gaussians", -1)) != int(authority["num_gaussians"])
        or int(payload.get("num_pixels", -1)) != int(authority["num_pixels"])
        or gaussian.shape != (count,)
        or pixels.shape != (count,)
        or base.shape != (count,)
        or bool((gaussian < 0).any())
        or bool((gaussian >= int(authority["num_gaussians"])).any())
        or bool((pixels < 0).any())
        or bool((pixels >= int(authority["num_pixels"])).any())
        or not bool(torch.isfinite(base).all())
        or bool((base < 0).any())
    ):
        raise ValueError("responsibility view payload differs")
    pixel_mass = torch.zeros(int(authority["num_pixels"]), dtype=torch.float32)
    pixel_mass.index_add_(0, pixels, base)
    marginal = base * base / pixel_mass[pixels].clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return gaussian, pixels, marginal


def _require_sha(path: str | Path, expected: str, *, label: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    observed = sha256_file(source)
    if observed != str(expected):
        raise ValueError(f"{label} SHA-256 differs")
    return source


def _validate_base_descriptor(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], torch.Tensor]:
    payload, _, _ = load_torch_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="AcceptedV2 multiscale descriptor authority",
    )
    required = {
        "xyz", "features", "summary_features", "global_rows",
        "features_by_scale", "valid", "metadata", "primary_valid",
        "semantic_confidence",
    }
    if set(payload) != required:
        raise ValueError("AcceptedV2 multiscale descriptor fields differ")
    xyz = torch.as_tensor(payload["xyz"]).float().cpu()
    valid = torch.as_tensor(payload["valid"]).bool().cpu()
    rows = torch.as_tensor(payload["global_rows"]).long().cpu()
    descriptors = torch.as_tensor(payload["features_by_scale"])
    metadata = payload.get("metadata")
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or valid.shape != (xyz.shape[0],)
        or rows.shape != (int(valid.sum()),)
        or not torch.equal(rows, torch.where(valid)[0])
        or descriptors.shape != (rows.numel(), 3, 1536)
        or not descriptors.is_floating_point()
        or not bool(torch.isfinite(descriptors).all())
        or not isinstance(metadata, Mapping)
        or metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
    ):
        raise ValueError("AcceptedV2 multiscale descriptor authority differs")
    return dict(payload), rows


def _update_top_views(
    *,
    top_descriptors: torch.Tensor,
    top_mass: torch.Tensor,
    top_frame_ids: torch.Tensor,
    rows: torch.Tensor,
    descriptors: torch.Tensor,
    mass: torch.Tensor,
    frame_id: int,
) -> None:
    """Update top-4 tuples by descending mass, then ascending frame id."""

    selected = torch.as_tensor(rows).long().cpu()
    candidate = torch.as_tensor(descriptors).half().cpu()
    candidate_mass = torch.as_tensor(mass).float().cpu()
    if (
        selected.ndim != 1
        or candidate.shape != (selected.numel(), 1536)
        or candidate_mass.shape != (selected.numel(),)
        or not bool(torch.isfinite(candidate).all())
        or not bool(torch.isfinite(candidate_mass).all())
        or bool((candidate_mass <= 0).any())
    ):
        raise ValueError("teacher-view top-k update differs")
    existing_mass = top_mass[selected]
    existing_frames = top_frame_ids[selected]
    invalid = existing_frames < 0
    first_invalid = invalid.to(torch.int64).argmax(dim=1)
    has_invalid = invalid.any(dim=1)

    minimum_mass = existing_mass.min(dim=1).values
    tied = existing_mass == minimum_mass[:, None]
    worst_frame = existing_frames.masked_fill(~tied, -1).max(dim=1).values
    worst_slot = (
        tied & (existing_frames == worst_frame[:, None])
    ).to(torch.int64).argmax(dim=1)
    slot = torch.where(has_invalid, first_invalid, worst_slot)
    replace = has_invalid | (candidate_mass > minimum_mass) | (
        (candidate_mass == minimum_mass) & (int(frame_id) < worst_frame)
    )
    if bool(replace.any()):
        kept_rows = selected[replace]
        kept_slots = slot[replace]
        top_descriptors[kept_rows, kept_slots] = candidate[replace]
        top_mass[kept_rows, kept_slots] = candidate_mass[replace]
        top_frame_ids[kept_rows, kept_slots] = int(frame_id)


def _canonicalize_view_axis(
    descriptors: torch.Tensor,
    mass: torch.Tensor,
    frame_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort the retained axis by descending mass and ascending frame id."""

    safe_frames = frame_ids.masked_fill(frame_ids < 0, torch.iinfo(torch.int32).max)
    frame_order = torch.argsort(safe_frames, dim=1, stable=True)
    mass_by_frame = mass.gather(1, frame_order)
    mass_order = torch.argsort(mass_by_frame, dim=1, descending=True, stable=True)
    order = frame_order.gather(1, mass_order)
    descriptors = descriptors.gather(
        1, order[..., None].expand(-1, -1, descriptors.shape[-1])
    )
    mass = mass.gather(1, order)
    frame_ids = frame_ids.gather(1, order)
    descriptors[frame_ids < 0] = 0
    mass[frame_ids < 0] = 0
    return descriptors.contiguous(), mass.contiguous(), frame_ids.contiguous()


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to clobber teacher authority: {output}")
    preregistration = _require_sha(
        args.preregistration,
        args.expected_preregistration_sha256,
        label="oracle preregistration",
    )
    base_path = _require_sha(
        args.base_descriptor,
        args.expected_base_descriptor_sha256,
        label="AcceptedV2 descriptor authority",
    )
    base, accepted_rows = _validate_base_descriptor(
        base_path, args.expected_base_descriptor_sha256
    )
    base_metadata = dict(base["metadata"])
    authority, authority_path = _validate_responsibility_authority(
        args.responsibility_authority,
        args.expected_responsibility_authority_sha256,
        expected_xyz_sha256=str(base_metadata["field_geometry_xyz_sha256"]),
    )
    authority_metadata = dict(authority["metadata"])
    frames = [int(value) for value in authority["frame_indices"]]
    config_path = _require_sha(
        authority_metadata["config"],
        args.expected_config_sha256,
        label="scene config",
    )
    geometry_path = _require_sha(
        authority_metadata["checkpoint"],
        args.expected_geometry_checkpoint_sha256,
        label="renderer geometry checkpoint",
    )
    radio_path = _require_sha(
        args.official_radio_checkpoint,
        args.expected_official_radio_checkpoint_sha256,
        label="official RADIO checkpoint",
    )

    config = load_config(config_path)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("teacher-view materialization requires one CUDA device")
    xyz = torch.as_tensor(base["xyz"]).float().cpu()
    if int(authority["num_gaussians"]) != int(xyz.shape[0]):
        raise ValueError("AcceptedV2/responsibility primitive rows differ")
    feature_height = int(authority_metadata["feature_height"])
    feature_width = int(authority_metadata["feature_width"])
    if int(authority["num_pixels"]) != feature_height * feature_width:
        raise ValueError("responsibility feature grid differs")
    feature_dir = Path(str(getattr(config, "feature_dir", ""))).expanduser().resolve()
    _manifest, validation, tensor_records = _validated_feature_bundle(
        feature_dir,
        expected_output_bundle_sha256=args.expected_feature_output_bundle_sha256,
    )
    if validation["manifest_sha256"] != args.expected_feature_frame_manifest_sha256:
        raise ValueError("source teacher feature manifest differs")

    n_rows = int(accepted_rows.numel())
    global_to_accepted = torch.full(
        (xyz.shape[0],), -1, dtype=torch.long
    )
    global_to_accepted[accepted_rows] = torch.arange(n_rows)
    top_descriptors = torch.zeros(n_rows, 4, 1536, dtype=torch.float16)
    top_mass = torch.zeros(n_rows, 4, dtype=torch.float32)
    top_frame_ids = torch.full((n_rows, 4), -1, dtype=torch.int32)
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        radio_path, expected_sha256=args.expected_official_radio_checkpoint_sha256
    ).to(device).eval().requires_grad_(False)
    head_parameter = next(head.parameters())
    projection_batch = int(args.projection_batch_size)
    if projection_batch <= 0:
        raise ValueError("projection_batch_size must be positive")

    per_view_counts: list[dict[str, int]] = []
    for position, record in enumerate(authority["views"]):
        frame = int(record["frame_index"])
        feature_map = _load_bundle_feature_maps(
            feature_dir=feature_dir,
            selected_frame_indices=[frame],
            subdir="backbone",
            expected_dim=1280,
            feature_size=(feature_height, feature_width),
            tensor_records=tensor_records,
            normalize=False,
            output_dtype=torch.float32,
        )
        gaussian_ids, pixel_ids, marginal_weights = _load_responsibility_view(
            authority, authority_path, record
        )
        local_ids = global_to_accepted[gaussian_ids]
        keep = local_ids >= 0
        frame_sum, frame_mass = accumulate_raster_contribution_features(
            feature_map.to(device),
            local_ids[keep].to(device),
            pixel_ids[keep].to(device),
            marginal_weights[keep].to(device),
            n_gaussians=n_rows,
        )
        active = torch.where(frame_mass > 0)[0]
        projected_parts: list[torch.Tensor] = []
        for start in range(0, active.numel(), projection_batch):
            selected = active[start : start + projection_batch]
            raw = frame_sum[selected].float() / frame_mass[selected, None].clamp_min(1e-8)
            raw = raw.to(dtype=head_parameter.dtype)
            projected = F.normalize(head(raw[:, None])[:, 0].float(), dim=-1)
            projected_parts.append(projected.half().cpu())
        projected_cpu = (
            torch.cat(projected_parts, dim=0)
            if projected_parts
            else torch.empty(0, 1536, dtype=torch.float16)
        )
        _update_top_views(
            top_descriptors=top_descriptors,
            top_mass=top_mass,
            top_frame_ids=top_frame_ids,
            rows=active.cpu(),
            descriptors=projected_cpu,
            mass=frame_mass[active].float().cpu(),
            frame_id=frame,
        )
        per_view_counts.append({"frame_id": frame, "observed_rows": int(active.numel())})
        print(
            json.dumps(
                {
                    "view": position + 1,
                    "views": len(frames),
                    "frame_id": frame,
                    "observed_rows": int(active.numel()),
                }
            ),
            flush=True,
        )
        del (
            feature_map, gaussian_ids, pixel_ids, marginal_weights, local_ids,
            keep, frame_sum, frame_mass, active, projected_parts, projected_cpu,
        )
        torch.cuda.empty_cache()

    top_descriptors, top_mass, top_frame_ids = _canonicalize_view_axis(
        top_descriptors, top_mass, top_frame_ids
    )
    teacher_mask = top_frame_ids >= 0
    observed_counts = teacher_mask.sum(dim=1)
    if not bool((observed_counts > 0).any()):
        raise RuntimeError("teacher-view authority has no visible AcceptedV2 row")
    payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "global_rows": accepted_rows.contiguous(),
        "teacher_view_descriptors": top_descriptors,
        "teacher_view_mask": teacher_mask.contiguous(),
        "teacher_view_mass": top_mass,
        "teacher_view_frame_ids": top_frame_ids,
        "metadata": {
            "construction": "top4_exact_marginal_source_observation_then_official_siglip2_summary_v1",
            "retention_order": "marginal_mass_descending_then_frame_id_ascending",
            "maximum_views": 4,
            "source_view_count": len(frames),
            "source_frame_ids": frames,
            "excluded_target_frame_ids": list(TARGET_FRAME_IDS),
            "equal_view_mass_for_oracle": True,
            "base_descriptor": file_record(base_path),
            "responsibility_authority": file_record(authority_path),
            "scene_config": file_record(config_path),
            "geometry_checkpoint": file_record(geometry_path),
            "official_radio_checkpoint": file_record(radio_path),
            "feature_frame_manifest_sha256": validation["manifest_sha256"],
            "feature_frame_manifest": file_record(feature_dir / "frame_manifest.json"),
            "feature_output_bundle_sha256": args.expected_feature_output_bundle_sha256,
            "preregistration": file_record(preregistration),
            "per_view_observed_counts": per_view_counts,
            "query_independent": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "target_frames_opened": False,
        },
    }
    payload["authority_sha256"] = canonical_json_sha256(
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "global_rows": accepted_rows.tolist(),
            "source_frame_ids": frames,
            "excluded_target_frame_ids": list(TARGET_FRAME_IDS),
            "base_descriptor_sha256": args.expected_base_descriptor_sha256,
            "responsibility_authority_sha256": (
                args.expected_responsibility_authority_sha256
            ),
            "official_radio_checkpoint_sha256": args.expected_official_radio_checkpoint_sha256,
            "feature_frame_manifest_sha256": validation["manifest_sha256"],
            "retention_order": "marginal_mass_descending_then_frame_id_ascending",
        }
    )
    estimated_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            payload["global_rows"], payload["teacher_view_descriptors"],
            payload["teacher_view_mask"], payload["teacher_view_mass"],
            payload["teacher_view_frame_ids"],
        )
    )
    if estimated_bytes > int(args.maximum_tensor_bytes):
        raise RuntimeError("teacher-view tensor storage exceeds registered ceiling")
    write_torch_noclobber(output, payload)
    return {
        "status": "materialized",
        "output": file_record(output),
        "accepted_rows": n_rows,
        "rows_with_teacher": int((observed_counts > 0).sum()),
        "rows_with_four_views": int((observed_counts == 4).sum()),
        "tensor_storage_bytes": estimated_bytes,
        "authority_sha256": payload["authority_sha256"],
        "source_access": {
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "target_frames_opened": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--base-descriptor", required=True)
    parser.add_argument("--expected-base-descriptor-sha256", required=True)
    parser.add_argument("--responsibility-authority", required=True)
    parser.add_argument("--expected-responsibility-authority-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-geometry-checkpoint-sha256", required=True)
    parser.add_argument("--expected-feature-frame-manifest-sha256", required=True)
    parser.add_argument("--expected-feature-output-bundle-sha256", required=True)
    parser.add_argument("--official-radio-checkpoint", required=True)
    parser.add_argument("--expected-official-radio-checkpoint-sha256", required=True)
    parser.add_argument("--projection-batch-size", type=int, default=4096)
    parser.add_argument("--maximum-tensor-bytes", type=int, default=1200000000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    print(json.dumps(materialize(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
