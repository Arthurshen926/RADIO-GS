"""Shared fail-closed validation for all-available LERF source views."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from radio_gs.config import load_config
from radio_gs.querying.all_available_source_views import (
    SourceViewDomainAudit,
    audit_source_view_domain,
    validate_composite_frame_axis,
)
from radio_gs.rendering.sparse_marginal_authority import (
    SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA,
    SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
    sparse_exact_marginal_formula_contract,
)
from radio_gs.scripts import materialize_lerf_o1_o2_streaming as legacy
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    _validated_feature_bundle,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    validate_file_record,
)


REFERENCE_INPUT_NAMES = frozenset(
    {
        "base_descriptor",
        "responsibility_authority",
        "feature_manifest",
        "scene_config",
        "renderer_geometry_checkpoint",
        "official_radio_checkpoint",
        "positive_text",
        "negative_text",
        "o0_positive",
        "o0_negative",
        "frozen_metric_config",
    }
)


def file_record_value(
    value: object, *, label: str, validate_bytes: bool = True
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    result = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    if validate_bytes:
        validate_file_record(result, label=label)
    else:
        path = Path(result["path"])
        if (
            str(path.expanduser().resolve()) != result["path"]
            or path.is_symlink()
            or not path.is_file()
            or len(result["sha256"]) != 64
            or any(value not in "0123456789abcdef" for value in result["sha256"])
        ):
            raise ValueError(f"{label} lightweight file record differs")
    return result


def _validate_reference_header(authority: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "implementation",
        "method_contract",
        "method_contract_sha256",
        "feature_output_bundle_sha256",
        "inputs",
        "outputs",
        "execution",
        "query_free_materialization_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    method = authority.get("method_contract")
    execution = authority.get("execution")
    access = authority.get("access_audit")
    if (
        set(authority) != required
        or authority.get("status")
        != "authorized_source_only_premetric_o1_o2_streaming"
        or not str(authority.get("scene_id", ""))
        or authority.get("query_free_materialization_authorized") is not True
        or authority.get("metric_execution_authorized") is not False
        or not isinstance(method, Mapping)
        or int(method.get("source_view_count", -1)) != legacy.SOURCE_VIEW_COUNT
        or method.get("teacher_retention")
        != "top4_exact_marginal_mass_desc_then_frame_id_asc"
        or method.get("teacher_aggregation") != "equal_view_normalized_mean"
        or tuple(method.get("scale_ids", [])) != legacy.SCALE_IDS
        or tuple(method.get("scale_radii_m", [])) != legacy.SCALE_RADII_M
        or tuple(method.get("canonical_negative_queries", []))
        != legacy.NEGATIVE_QUERIES
        or not isinstance(execution, Mapping)
        or int(execution.get("physical_gpu", -1)) not in (0, 1)
        or str(execution.get("cuda_visible_devices"))
        != str(execution.get("physical_gpu"))
        or execution.get("program_device") != "cuda:0"
        or list(execution.get("projection_batch_candidates", []))
        != list(legacy.PREFLIGHT_BATCH_CANDIDATES)
        or float(execution.get("pacing_seconds_per_projection_batch", -1.0))
        not in (0.0, legacy.PACING_SECONDS_PER_PROJECTION_BATCH)
        or int(execution.get("thermal_poll_seconds", -1)) != 300
        or int(execution.get("soft_pause_temperature_c", -1)) != 0
        or int(execution.get("maximum_temperature_c", -1)) != 88
        or not isinstance(access, Mapping)
        or access.get("target_images_opened") is not False
        or access.get("target_ground_truth_opened") is not False
        or access.get("target_masks_opened") is not False
        or access.get("target_metrics_opened") is not False
        or access.get("target_quality_readout_executed") is not False
    ):
        raise ValueError("reference O1/O2 source-only authority header differs")
    implementation = file_record_value(
        authority["implementation"], label="reference O1/O2 implementation"
    )
    if implementation != file_record(implementation["path"]):
        raise ValueError("reference O1/O2 implementation bytes differ")


def load_reference_inputs(
    authority_path: str | Path,
    *,
    expected_sha256: str,
    load_tensor_payloads: bool = True,
) -> dict[str, Any]:
    """Validate the common frozen inputs of any sealed legacy-120 entrypoint."""

    raw, digest, authority_source = load_json_object(
        authority_path,
        expected_sha256=expected_sha256,
        label="reference O1/O2 source-only authority",
    )
    if not isinstance(raw, Mapping):
        raise ValueError("reference O1/O2 authority is not a mapping")
    authority = dict(raw)
    _validate_reference_header(authority)
    inputs = authority.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != REFERENCE_INPUT_NAMES:
        raise ValueError("reference O1/O2 inputs differ")
    records = {
        name: file_record_value(
            inputs[name],
            label=f"reference O1/O2 {name}",
            validate_bytes=load_tensor_payloads,
        )
        for name in sorted(REFERENCE_INPUT_NAMES)
    }

    base: dict[str, Any] | None = None
    rows: torch.Tensor | None = None
    if load_tensor_payloads:
        base, rows = legacy._validate_base_descriptor_general(
            Path(records["base_descriptor"]["path"]),
            records["base_descriptor"]["sha256"],
        )
    feature_path = Path(records["feature_manifest"]["path"])
    if feature_path.name != "frame_manifest.json":
        raise ValueError("reference feature manifest name differs")
    if load_tensor_payloads:
        feature_manifest, feature_validation, tensor_records = _validated_feature_bundle(
            feature_path.parent,
            expected_output_bundle_sha256=str(
                authority["feature_output_bundle_sha256"]
            ),
        )
        if (
            feature_validation["manifest_sha256"]
            != records["feature_manifest"]["sha256"]
        ):
            raise ValueError("reference feature manifest SHA differs")
    else:
        feature_manifest, manifest_digest, _ = load_json_object(
            feature_path,
            expected_sha256=records["feature_manifest"]["sha256"],
            label="reference feature manifest",
        )
        bundle = feature_manifest.get("output_bundle")
        if (
            not isinstance(bundle, Mapping)
            or feature_manifest.get("output_bundle_sha256")
            != authority["feature_output_bundle_sha256"]
            or canonical_json_sha256(bundle)
            != authority["feature_output_bundle_sha256"]
        ):
            raise ValueError("reference lightweight feature bundle differs")
        tensor_records = {}
        bundle_frames = bundle.get("frames")
        if not isinstance(bundle_frames, list):
            raise ValueError("reference lightweight feature tensor axis differs")
        for frame in bundle_frames:
            tensors = frame.get("tensors") if isinstance(frame, Mapping) else None
            if not isinstance(tensors, list):
                raise ValueError("reference lightweight feature tensor records differ")
            for tensor in tensors:
                relative = (
                    str(tensor.get("relative_path", ""))
                    if isinstance(tensor, Mapping)
                    else ""
                )
                if not relative or relative in tensor_records:
                    raise ValueError("reference lightweight feature tensor path differs")
                tensor_records[relative] = dict(tensor)
        feature_validation = {
            "manifest_sha256": manifest_digest,
            "output_bundle_sha256": authority["feature_output_bundle_sha256"],
            "lightweight_sealed_reference_validation": True,
        }
    feature_frames = [int(record["frame_idx"]) for record in feature_manifest["frames"]]
    if len(set(feature_frames)) != len(feature_frames):
        raise ValueError("reference feature frame axis repeats")

    responsibility_raw, _, responsibility_path = load_json_object(
        records["responsibility_authority"]["path"],
        expected_sha256=records["responsibility_authority"]["sha256"],
        label="reference exact responsibility authority",
    )
    raw_metadata = responsibility_raw.get("metadata")
    expected_xyz_sha = (
        str(base["metadata"]["field_geometry_xyz_sha256"])
        if base is not None
        else str(raw_metadata.get("xyz_sha256", ""))
        if isinstance(raw_metadata, Mapping)
        else ""
    )
    responsibility = legacy._validate_responsibility_payload(
        responsibility_raw,
        descriptor_xyz_sha256=expected_xyz_sha,
        feature_frame_ids=set(feature_frames),
    )
    responsibility_root = Path(responsibility_path).parent
    for view in responsibility["views"]:
        sidecar = (responsibility_root / str(view["relative_path"])).resolve()
        if responsibility_root not in sidecar.parents or not sidecar.is_file():
            raise ValueError("reference responsibility sidecar is missing or unsafe")
    metadata = responsibility["metadata"]
    if (
        records["scene_config"]["path"] != str(Path(metadata["config"]).resolve())
        or records["renderer_geometry_checkpoint"]["path"]
        != str(Path(metadata["checkpoint"]).resolve())
        or records["renderer_geometry_checkpoint"]["sha256"]
        != metadata["geometry_checkpoint_sha256"]
        or int(responsibility["num_pixels"])
        != int(metadata["feature_height"]) * int(metadata["feature_width"])
    ):
        raise ValueError("reference config/geometry alignment differs")
    if base is not None and int(responsibility["num_gaussians"]) != int(
        base["xyz"].shape[0]
    ):
        raise ValueError("reference Gaussian axis differs")
    if load_tensor_payloads:
        for name in (
            "scene_config",
            "renderer_geometry_checkpoint",
            "official_radio_checkpoint",
            "frozen_metric_config",
        ):
            if sha256_file(records[name]["path"]) != records[name]["sha256"]:
                raise ValueError(f"reference {name} SHA differs")
    scene_id = str(authority["scene_id"])
    source_config = load_config(records["scene_config"]["path"])
    frozen_config = load_config(records["frozen_metric_config"]["path"])
    if (
        Path(str(getattr(source_config, "feature_dir", ""))).expanduser().resolve()
        != feature_path.parent
        or Path(str(getattr(frozen_config, "scene_root", ""))).name != scene_id
    ):
        raise ValueError("reference scene/config/RADIO lineage differs")
    positive_queries: list[str] | None = None
    positive_embeddings: torch.Tensor | None = None
    negative_embeddings: torch.Tensor | None = None
    o0_positive: dict[str, Any] | None = None
    o0_negative: dict[str, Any] | None = None
    if load_tensor_payloads:
        if (
            base is None
            or rows is None
            or base["metadata"].get("official_radio_checkpoint_sha256")
            != records["official_radio_checkpoint"]["sha256"]
        ):
            raise ValueError("reference base/RADIO lineage differs")
        positive_raw, _, _ = load_torch_mapping(
            records["positive_text"]["path"],
            expected_sha256=records["positive_text"]["sha256"],
            map_location="cpu",
            label="reference positive text",
        )
        negative_raw, _, _ = load_torch_mapping(
            records["negative_text"]["path"],
            expected_sha256=records["negative_text"]["sha256"],
            map_location="cpu",
            label="reference negative text",
        )
        positive_queries = list(positive_raw.get("queries", []))
        positive_embeddings = legacy._validate_text_bank(
            positive_raw, expected_queries=positive_queries
        )
        negative_embeddings = legacy._validate_text_bank(
            negative_raw, expected_queries=list(legacy.NEGATIVE_QUERIES)
        )
        o0_positive_raw, _, _ = load_torch_mapping(
            records["o0_positive"]["path"],
            expected_sha256=records["o0_positive"]["sha256"],
            map_location="cpu",
            label="reference O0 positive",
        )
        o0_negative_raw, _, _ = load_torch_mapping(
            records["o0_negative"]["path"],
            expected_sha256=records["o0_negative"]["sha256"],
            map_location="cpu",
            label="reference O0 negative",
        )
        legacy._validate_o0_pair(
            o0_positive_raw,
            o0_negative_raw,
            base=base,
            positive_queries=positive_queries,
            renderer_sha256=records["renderer_geometry_checkpoint"]["sha256"],
        )
        o0_positive = dict(o0_positive_raw)
        o0_negative = dict(o0_negative_raw)
    audit = audit_source_view_domain(
        feature_frame_ids=feature_frames,
        excluded_frame_ids=metadata["excluded_frame_ids"],
        legacy_frame_ids=responsibility["frame_indices"],
    )
    return {
        "authority": authority,
        "authority_record": {"path": str(authority_source), "sha256": digest},
        "records": records,
        "base": base,
        "rows": rows,
        "responsibility": responsibility,
        "responsibility_path": responsibility_path,
        "feature_manifest": feature_manifest,
        "feature_validation": feature_validation,
        "tensor_records": tensor_records,
        "positive_queries": positive_queries,
        "positive_embeddings": positive_embeddings,
        "negative_embeddings": negative_embeddings,
        "o0_positive": o0_positive,
        "o0_negative": o0_negative,
        "domain_audit": audit,
    }


def validate_supplemental_responsibility(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path,
    audit: SourceViewDomainAudit,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a standard exact-marginal manifest over precisely omitted views."""

    required = {
        "formula_contract",
        "formula_sha256",
        "frame_indices",
        "metadata",
        "num_gaussians",
        "num_pixels",
        "schema",
        "schema_version",
        "total_hits",
        "views",
    }
    metadata = payload.get("metadata")
    frames = payload.get("frame_indices")
    views = payload.get("views")
    legacy_metadata = reference["responsibility"]["metadata"]
    if (
        set(payload) != required
        or payload.get("schema") != SPARSE_EXACT_MARGINAL_AUTHORITY_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("formula_contract") != sparse_exact_marginal_formula_contract()
        or payload.get("formula_sha256") != SPARSE_EXACT_MARGINAL_FORMULA_SHA256
        or not isinstance(metadata, Mapping)
        or metadata.get("supplement_contract")
        != "radio_gs.lerf_omitted_source_view_exact_marginal_supplement.v1"
        or metadata.get("query_independent") is not True
        or metadata.get("feature_independent") is not True
        or metadata.get("benchmark_images_opened") is not False
        or metadata.get("benchmark_masks_opened") is not False
        or metadata.get("text_queries_opened") is not False
        or metadata.get("target_metrics_opened") is not False
        or metadata.get("config") != legacy_metadata.get("config")
        or metadata.get("checkpoint") != legacy_metadata.get("checkpoint")
        or metadata.get("geometry_checkpoint_sha256")
        != legacy_metadata.get("geometry_checkpoint_sha256")
        or metadata.get("xyz_sha256") != legacy_metadata.get("xyz_sha256")
        or metadata.get("gaussian_state_sha256")
        != legacy_metadata.get("gaussian_state_sha256")
        or metadata.get("feature_height") != legacy_metadata.get("feature_height")
        or metadata.get("feature_width") != legacy_metadata.get("feature_width")
        or metadata.get("excluded_frame_ids")
        != legacy_metadata.get("excluded_frame_ids")
        or metadata.get("legacy_responsibility_authority")
        != reference["records"]["responsibility_authority"]
        or metadata.get("feature_manifest") != reference["records"]["feature_manifest"]
        or not isinstance(frames, list)
        or not isinstance(views, list)
        or frames != list(audit.omitted_frames)
        or frames != [int(record.get("frame_index", -1)) for record in views]
        or int(payload.get("num_gaussians", -1))
        != int(reference["responsibility"]["num_gaussians"])
        or int(payload.get("num_pixels", -1))
        != int(reference["responsibility"]["num_pixels"])
        or int(payload.get("total_hits", -1))
        != sum(int(record.get("num_hits", -1)) for record in views)
    ):
        raise ValueError("all-available supplemental responsibility differs")
    validate_composite_frame_axis(audit, frames)
    root = Path(source_path).expanduser().resolve().parent
    for index, record in enumerate(views):
        if not isinstance(record, Mapping) or set(record) != {
            "frame_index",
            "num_hits",
            "relative_path",
            "sha256",
            "view_index",
        }:
            raise ValueError("supplemental responsibility view record differs")
        sidecar = (root / str(record["relative_path"])).resolve()
        if (
            int(record["view_index"]) != index
            or int(record["num_hits"]) < 0
            or root not in sidecar.parents
            or not sidecar.is_file()
            or sha256_file(sidecar) != str(record["sha256"])
        ):
            raise ValueError("supplemental responsibility sidecar differs")
    return dict(payload)


__all__ = [
    "REFERENCE_INPUT_NAMES",
    "file_record_value",
    "load_reference_inputs",
    "validate_supplemental_responsibility",
]
