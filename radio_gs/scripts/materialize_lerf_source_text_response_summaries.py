#!/usr/bin/env python3
"""Render compact target-blind source text-response summaries.

The materializer is candidate agnostic.  It converts one hash-bound primitive
descriptor payload at a time into 101 primitive query-response rows, renders
those rows with the unchanged frozen geometry/camera/alpha compositor, and
immediately reduces each legal source-heldout frame to a small JSON summary.
No benchmark query, mask, label, or target metric is opened.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.evaluation.lerf_source_text_response_ranking import (
    FRAME_EVALUATOR_IMPLEMENTATION,
    build_scene_summary,
    evaluate_source_response_frame,
)
from radio_gs.evaluation.render_ceiling import normalize_premultiplied
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.training.feature_training_utils import SimpleRadioDataset
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    stable_descriptor_load,
    validate_file_record,
    write_frozen_json,
)


AUTHORITY_SCHEMA = "radio_gs.lerf_source_text_response_summary_execution.v1"
RESULT_SCHEMA = "radio_gs.lerf_source_text_response_summary_result.v1"
SCHEMA_VERSION = 1
PRIMITIVE_RESPONSE_CHUNK_ROWS = 8192
ALPHA_THRESHOLD = 0.02
SPARSE_TEACHER_MEAN_KIND = "sparse_teacher_mean_v2"
DENSE_OFFICIAL_CROP_SUMMARY_MPR_KIND = "dense_official_crop_summary_mpr_v1"
DESCRIPTOR_PAYLOAD_KINDS = {
    SPARSE_TEACHER_MEAN_KIND,
    DENSE_OFFICIAL_CROP_SUMMARY_MPR_KIND,
}
IMPLEMENTATION = file_record(Path(__file__).resolve())
THERMAL_GUARD = file_record(
    Path(__file__).resolve().with_name("run_with_gpu_thermal_guard.sh")
)


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path/SHA-256 record")
    record = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    if not record["path"] or len(record["sha256"]) != 64:
        raise ValueError(f"{label} differs")
    if any(value not in "0123456789abcdef" for value in record["sha256"]):
        raise ValueError(f"{label} SHA-256 differs")
    return record


def _integer_ids(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{label} must contain integers")
    result = [int(item) for item in value]
    if result != sorted(set(result)):
        raise ValueError(f"{label} must be sorted and unique")
    return result


def validate_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source response execution authority must be an object")
    authority = dict(value)
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != "authorized_source_only_compact_summary"
    ):
        raise ValueError("source response execution authority schema differs")
    if authority.get("implementation") != IMPLEMENTATION:
        raise ValueError("source response materializer implementation differs")
    if authority.get("frame_evaluator_implementation") != FRAME_EVALUATOR_IMPLEMENTATION:
        raise ValueError("source response frame evaluator implementation differs")
    if not str(authority.get("scene_id", "")):
        raise ValueError("source response scene identity differs")
    frames = _integer_ids(
        authority.get("source_heldout_frame_ids"), label="source-heldout frame IDs"
    )
    forbidden = _integer_ids(
        authority.get("forbidden_target_frame_ids"), label="forbidden target frame IDs"
    )
    if set(frames).intersection(forbidden):
        raise ValueError("source response frames overlap forbidden target frames")
    inputs = authority.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("source response input contract differs")
    for key in (
        "source_gate_preregistration",
        "source_view_preregistration",
        "scene_config",
        "geometry_checkpoint",
        "query_bank_artifact",
        "query_bank_manifest",
    ):
        _record(inputs.get(key), label=key)
    source_reseal = inputs.get("source_reseal")
    if not isinstance(source_reseal, Mapping) or set(source_reseal) != {
        "path",
        "sha256",
        "required_schema",
        "required_mode",
    }:
        raise ValueError("source response reseal contract differs")
    if (
        source_reseal.get("required_schema")
        != "radio_gs.lerf_official_crop_summary_reseal.v1"
        or source_reseal.get("required_mode")
        != "content_addressed_immutable_reseal"
        or not Path(str(source_reseal.get("path", ""))).is_absolute()
        or len(str(source_reseal.get("sha256", ""))) != 64
    ):
        raise ValueError("source response reseal requirement differs")
    methods = authority.get("methods")
    if not isinstance(methods, list) or len(methods) != 2:
        raise ValueError("source response authority requires control and candidate")
    roles: set[str] = set()
    method_ids: set[str] = set()
    for method in methods:
        if not isinstance(method, Mapping) or set(method) != {
            "method_id",
            "role",
            "descriptor_payload",
            "descriptor_payload_kind",
            "descriptor_payload_contract",
            "descriptor_provenance_authority",
            "descriptor_geometry_authority",
        }:
            raise ValueError("source response method contract differs")
        method_id = str(method["method_id"])
        role = str(method["role"])
        if not method_id or method_id in method_ids or role not in {"control", "candidate"}:
            raise ValueError("source response method identity/role differs")
        method_ids.add(method_id)
        roles.add(role)
        _record(method["descriptor_payload"], label=f"{method_id} descriptor")
        _record(
            method["descriptor_provenance_authority"],
            label=f"{method_id} descriptor provenance authority",
        )
        _record(
            method["descriptor_geometry_authority"],
            label=f"{method_id} descriptor geometry authority",
        )
        if method["descriptor_payload_kind"] not in DESCRIPTOR_PAYLOAD_KINDS:
            raise ValueError("source response descriptor payload kind differs")
        if not isinstance(method["descriptor_payload_contract"], Mapping):
            raise ValueError("source response descriptor payload contract differs")
    if roles != {"control", "candidate"}:
        raise ValueError("source response method roles differ")
    equivalence = authority.get("equivalence_smoke")
    same_descriptor = methods[0]["descriptor_payload"] == methods[1][
        "descriptor_payload"
    ]
    if not isinstance(equivalence, bool) or equivalence != same_descriptor:
        raise ValueError("source response equivalence-smoke declaration differs")
    execution = authority.get("execution")
    if not isinstance(execution, Mapping) or (
        execution.get("required_cuda_visible_devices") not in {"0", "1"}
        or execution.get("program_device") != "cuda:0"
        or execution.get("thermal_guard") != THERMAL_GUARD
        or execution.get("thermal_poll_seconds") != 300
        or execution.get("maximum_temperature_c") != 88
        or execution.get("soft_pause_temperature_c") != 0
    ):
        raise ValueError("source response execution/thermal contract differs")
    outputs = authority.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "control_summary",
        "candidate_summary",
        "result",
    }:
        raise ValueError("source response output contract differs")
    if any(not Path(str(path)).is_absolute() for path in outputs.values()):
        raise ValueError("source response output path is not absolute")
    access = authority.get("access_audit")
    if not isinstance(access, Mapping) or any(
        access.get(key) is not False
        for key in (
            "benchmark_queries_opened",
            "benchmark_masks_or_labels_opened",
            "target_metric_execution_authorized",
        )
    ):
        raise ValueError("source response authority is not target blind")
    bank = authority.get("query_bank")
    if not isinstance(bank, Mapping) or set(bank) != {
        "path", "sha256", "manifest_path", "manifest_sha256",
        "query_split", "queries", "embedding_tensor_sha256",
    }:
        raise ValueError("source response query-bank authority differs")
    if (
        bank.get("path") != inputs["query_bank_artifact"]["path"]
        or bank.get("sha256") != inputs["query_bank_artifact"]["sha256"]
        or bank.get("manifest_path") != inputs["query_bank_manifest"]["path"]
        or bank.get("manifest_sha256") != inputs["query_bank_manifest"]["sha256"]
        or bank.get("query_split") != "dev"
        or bank.get("queries") != 101
        or len(str(bank.get("embedding_tensor_sha256", ""))) != 64
    ):
        raise ValueError("source response query-bank binding differs")
    geometry = authority.get("geometry")
    if (
        not isinstance(geometry, Mapping)
        or set(geometry) != {"num_gaussians", "xyz_sha256"}
        or not isinstance(geometry.get("num_gaussians"), int)
        or isinstance(geometry.get("num_gaussians"), bool)
        or int(geometry["num_gaussians"]) <= 0
        or len(str(geometry.get("xyz_sha256", ""))) != 64
    ):
        raise ValueError("source response geometry authority differs")
    return authority


def load_authority(path: str | Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, str]]:
    authority, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="source response execution authority",
    )
    return validate_authority(authority), {"path": str(source), "sha256": digest}


def _load_query_bank(authority: Mapping[str, Any]) -> tuple[torch.Tensor, list[str], dict[str, object]]:
    inputs = authority["inputs"]
    artifact = _record(inputs["query_bank_artifact"], label="query bank artifact")
    manifest_record = _record(inputs["query_bank_manifest"], label="query bank manifest")
    payload, _, source = load_torch_mapping(
        artifact["path"],
        expected_sha256=artifact["sha256"],
        map_location="cpu",
        label="target-blind source query bank",
    )
    manifest, _, manifest_source = load_json_object(
        manifest_record["path"],
        expected_sha256=manifest_record["sha256"],
        label="target-blind source query bank manifest",
    )
    embeddings = torch.as_tensor(payload.get("embeddings"))
    queries = payload.get("synsets")
    if (
        payload.get("split") != "dev"
        or payload.get("benchmark_vocabulary_opened") is not False
        or payload.get("uses_benchmark_vocabulary_for_construction") is not False
        or manifest.get("split") != "dev"
        or manifest.get("benchmark_vocabulary_opened") is not False
        or not isinstance(queries, list)
        or len(queries) != 101
        or embeddings.device.type != "cpu"
        or embeddings.dtype != torch.float32
        or embeddings.shape != (101, 1536)
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("target-blind source query bank contract differs")
    expected_tensor_sha = str(authority["query_bank"]["embedding_tensor_sha256"])
    if payload.get("embedding_tensor_sha256") != expected_tensor_sha:
        raise ValueError("target-blind source query bank tensor authority differs")
    norms = torch.linalg.vector_norm(embeddings, dim=-1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=5e-5, rtol=5e-5)):
        raise ValueError("target-blind source query embeddings are not normalized")
    query_record = dict(authority["query_bank"])
    if (
        query_record["path"] != str(source)
        or query_record["manifest_path"] != str(manifest_source)
        or query_record["queries"] != 101
        or query_record["query_split"] != "dev"
    ):
        raise ValueError("source response query-bank record differs")
    return embeddings.contiguous(), [str(value) for value in queries], query_record


def _load_source_reseal(
    authority: Mapping[str, Any], source_view_preregistration: Mapping[str, str]
) -> tuple[dict[int, dict[str, object]], Path, dict[str, str]]:
    requirement = authority["inputs"]["source_reseal"]
    payload, digest, source = load_json_object(
        requirement["path"],
        expected_sha256=str(requirement["sha256"]),
        label="official crop-summary source reseal",
    )
    if (
        payload.get("schema") != requirement["required_schema"]
        or payload.get("mode") != requirement["required_mode"]
        or payload.get("scene") != authority["scene_id"]
        or payload.get("preregistration") != dict(source_view_preregistration)
        or payload.get("tensor_content_hashes_computed") is not True
        or payload.get("source_directory_modified") is not False
    ):
        raise ValueError("official crop-summary source reseal differs")
    records = payload.get("frame_records")
    if not isinstance(records, list):
        raise ValueError("official crop-summary source reseal lacks frame records")
    by_frame = {int(record["frame_id"]): dict(record) for record in records}
    required = authority["source_heldout_frame_ids"]
    if any(frame not in by_frame for frame in required):
        raise ValueError("source reseal lacks a required heldout frame")
    root = Path(str(payload["source_tensor_dir"])).resolve()
    return by_frame, root, {"path": str(source), "sha256": digest}


def _load_teacher_tensor(
    root: Path, record: Mapping[str, object]
) -> torch.Tensor:
    relative = Path(str(record["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source teacher tensor path is unsafe")

    def load(handle):
        try:
            return torch.load(handle, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError("source teacher loading requires weights_only=True") from exc

    value, digest, _ = stable_descriptor_load(
        root / relative,
        load,
        expected_sha256=str(record["sha256"]),
        label="official crop-summary source teacher tensor",
    )
    tensor = torch.as_tensor(value)
    if (
        digest != record["sha256"]
        or tensor.shape != (1536, 46, 62)
        or tensor.dtype != torch.float16
        or not bool(torch.isfinite(tensor).all())
    ):
        raise ValueError("official crop-summary source teacher tensor differs")
    return tensor.contiguous()


def _dataset(config: Any, frame_ids: Sequence[int]) -> SimpleRadioDataset:
    feature_dir = Path(str(getattr(config, "feature_dir", "")))
    pose_file_value = str(getattr(config, "pose_file", "") or "").strip()
    pose_file = pose_file_value if pose_file_value and Path(pose_file_value).is_file() else None
    pose_dir_value = str(getattr(config, "pose_dir", "") or "").strip()
    fallback = feature_dir / "poses_w2c"
    pose_dir = (
        pose_dir_value
        if pose_dir_value and Path(pose_dir_value).is_dir()
        else str(fallback) if fallback.is_dir() else None
    )
    return SimpleRadioDataset(
        feature_dir=str(feature_dir),
        pose_file=pose_file,
        pose_dir=pose_dir,
        feature_size=(46, 62),
        split="validation",
        dataset_type=str(getattr(config, "dataset_type", "lerf")),
        frame_ids=list(frame_ids),
    )


def _geometry_sha256(model: Any) -> str:
    xyz = model.get_xyz().detach().float().cpu().contiguous().numpy().astype(
        "<f4", copy=False
    )
    import hashlib

    return hashlib.sha256(xyz.tobytes()).hexdigest()


def load_descriptor_payload(
    method: Mapping[str, object],
    *,
    scene_id: str,
    num_gaussians: int,
    xyz_sha256: str,
) -> dict[str, Any]:
    record = _record(method["descriptor_payload"], label="primitive descriptor payload")
    provenance_record = _record(
        method["descriptor_provenance_authority"],
        label="primitive descriptor provenance authority",
    )
    geometry_authority_record = _record(
        method["descriptor_geometry_authority"],
        label="primitive descriptor geometry authority",
    )
    validate_file_record(provenance_record, label="primitive descriptor provenance authority")
    validate_file_record(
        geometry_authority_record, label="primitive descriptor geometry authority"
    )
    payload, _, _ = load_torch_mapping(
        record["path"],
        expected_sha256=record["sha256"],
        map_location="cpu",
        label="source primitive teacher descriptor payload",
    )
    kind = str(method["descriptor_payload_kind"])
    contract = dict(method["descriptor_payload_contract"])
    if kind == SPARSE_TEACHER_MEAN_KIND:
        provenance, _, _ = load_json_object(
            provenance_record["path"],
            expected_sha256=provenance_record["sha256"],
            label="sparse source teacher execution authority",
        )
        geometry_authority, _, _ = load_json_object(
            geometry_authority_record["path"],
            expected_sha256=geometry_authority_record["sha256"],
            label="sparse source teacher geometry authority",
        )
        geometry_metadata = geometry_authority.get("metadata")
        payload_inputs = payload.get("input_authority")
        if (
            contract != {
                "schema": "radio_gs.lerf_source_teacher_mean_siglip.v2",
                "schema_version": 2,
                "descriptor_dimension": 1536,
            }
            or payload.get("schema") != contract["schema"]
            or payload.get("schema_version") != contract["schema_version"]
            or payload.get("scene_id") != scene_id
            or payload.get("execution_authority")
            != method["descriptor_provenance_authority"]
            or not isinstance(provenance.get("inputs"), Mapping)
            or provenance["inputs"].get("responsibility_authority")
            != geometry_authority_record
            or not isinstance(payload_inputs, Mapping)
            or payload_inputs.get("responsibility_authority")
            != geometry_authority_record
            or not isinstance(geometry_metadata, Mapping)
            or geometry_authority.get("num_gaussians") != num_gaussians
            or geometry_metadata.get("xyz_sha256") != xyz_sha256
            or geometry_metadata.get("geometry_checkpoint_sha256")
            != provenance["inputs"].get("renderer_geometry_checkpoint", {}).get(
                "sha256"
            )
        ):
            raise ValueError(
                "sparse source teacher descriptor provenance/geometry differs"
            )
        rows = torch.as_tensor(payload.get("global_rows"))
        mean = torch.as_tensor(payload.get("teacher_mean"))
        valid = torch.as_tensor(payload.get("teacher_valid"))
    elif kind == DENSE_OFFICIAL_CROP_SUMMARY_MPR_KIND:
        if contract != {
            "metadata_schema_version": 1,
            "feature_space": "semantic_descriptor",
            "construction": "semantic_descriptor_raster_gaussian_top1_contribution_mean",
            "descriptor_dimension": 1536,
        }:
            raise ValueError("genuine crop-summary MPR authority contract differs")
        if geometry_authority_record != provenance_record:
            raise ValueError("genuine crop-summary MPR geometry authority differs")
        provenance, _, _ = load_json_object(
            provenance_record["path"],
            expected_sha256=provenance_record["sha256"],
            label="genuine crop-summary MPR execution authority",
        )
        metadata = payload.get("metadata")
        fingerprint = payload.get("geometry_fingerprint")
        xyz = torch.as_tensor(payload.get("xyz"))
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("schema_version") != 1
            or metadata.get("feature_space") != contract["feature_space"]
            or metadata.get("construction") != contract["construction"]
            or metadata.get("benchmark_masks_opened") is not False
            or metadata.get("text_queries_opened") is not False
            or not isinstance(fingerprint, Mapping)
            or fingerprint.get("num_gaussians") != num_gaussians
            or fingerprint.get("xyz_sha256") != xyz_sha256
            or provenance.get("scene") != scene_id
            or not isinstance(provenance.get("inputs"), Mapping)
            or provenance["inputs"].get("genuine_mpr") != record
            or xyz.device.type != "cpu"
            or xyz.dtype != torch.float32
            or xyz.shape != (num_gaussians, 3)
            or not bool(torch.isfinite(xyz).all())
            or _xyz_tensor_sha256(xyz) != xyz_sha256
        ):
            raise ValueError("genuine crop-summary MPR provenance/geometry differs")
        rows = torch.arange(num_gaussians, dtype=torch.long)
        mean = torch.as_tensor(payload.get("features"))
        valid = torch.as_tensor(payload.get("valid"))
    else:  # guarded by the sealed authority, retained as fail-closed defense.
        raise ValueError("source primitive descriptor payload kind differs")
    if (
        rows.device.type != "cpu"
        or rows.ndim != 1
        or rows.dtype == torch.bool
        or mean.device.type != "cpu"
        or mean.shape != (rows.numel(), 1536)
        or mean.dtype != torch.float16
        or valid.device.type != "cpu"
        or valid.shape != rows.shape
        or not bool(torch.isfinite(mean).all())
    ):
        raise ValueError("source primitive descriptor tensor contract differs")
    rows = rows.long().contiguous()
    valid = valid.bool().contiguous()
    if (
        not bool(valid.any())
        or bool((rows < 0).any())
        or bool((rows >= num_gaussians).any())
        or rows.unique().numel() != rows.numel()
        or bool((torch.linalg.vector_norm(mean[valid].float(), dim=-1) <= 0).any())
    ):
        raise ValueError("source primitive descriptor row authority differs")
    return {"global_rows": rows, "teacher_mean": mean.contiguous(), "teacher_valid": valid}


def _xyz_tensor_sha256(xyz: torch.Tensor) -> str:
    import hashlib

    array = torch.as_tensor(xyz).detach().float().cpu().contiguous().numpy().astype(
        "<f4", copy=False
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_primitive_descriptor_rows(
    payload: Mapping[str, torch.Tensor],
    *,
    num_gaussians: int,
    device: torch.device,
    chunk_rows: int = PRIMITIVE_RESPONSE_CHUNK_ROWS,
) -> torch.Tensor:
    """Place raw descriptor rows on the checkpoint-bound primitive axis.

    Text projection intentionally happens after descriptor compositing and
    per-pixel L2 normalization.  Normalizing/scoring primitive rows first is
    not equivalent when multiple primitives contribute to one pixel.
    """

    rows = torch.as_tensor(payload["global_rows"]).long()
    descriptors = torch.as_tensor(payload["teacher_mean"])
    valid = torch.as_tensor(payload["teacher_valid"]).bool()
    if chunk_rows <= 0:
        raise ValueError("primitive descriptor placement contract differs")
    descriptor_rows = torch.zeros(
        num_gaussians, 1536, dtype=torch.float32, device=device
    )
    for start in range(0, rows.numel(), chunk_rows):
        stop = min(rows.numel(), start + chunk_rows)
        active = valid[start:stop]
        if not bool(active.any()):
            continue
        selected_rows = rows[start:stop][active].to(device)
        selected = descriptors[start:stop][active].float().to(device)
        descriptor_rows[selected_rows] = selected
    return descriptor_rows.contiguous()


def descriptor_map_to_text_responses(
    descriptor_map: torch.Tensor, text_embeddings: torch.Tensor
) -> torch.Tensor:
    """Apply the sealed descriptor-first per-pixel response formula."""

    descriptors = torch.as_tensor(descriptor_map).float()
    text = F.normalize(torch.as_tensor(text_embeddings).float(), dim=-1)
    if descriptors.ndim != 3 or descriptors.shape[0] != 1536:
        raise ValueError("rendered descriptor map contract differs")
    if text.shape != (101, 1536):
        raise ValueError("source text embedding contract differs")
    unit = F.normalize(descriptors, dim=0, eps=1e-12)
    return torch.einsum("qd,dhw->qhw", text.to(unit.device), unit)


def _clone_equivalent_summary(
    summary: Mapping[str, object], *, method_id: str
) -> dict[str, object]:
    frames = copy.deepcopy(list(summary["frames"]))
    for frame in frames:
        frame["method_id"] = method_id
    return build_scene_summary(
        frames,
        scene_id=str(summary["scene_id"]),
        method_id=method_id,
        required_frame_ids=list(summary["source_heldout_frame_ids"]),
    )


@torch.inference_mode()
def materialize(
    authority_path: str | Path, expected_authority_sha256: str
) -> dict[str, Any]:
    authority, authority_record = load_authority(
        authority_path, expected_authority_sha256
    )
    visible = str(authority["execution"]["required_cuda_visible_devices"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != visible:
        raise RuntimeError("source response materializer CUDA visibility differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("source response materializer requires one visible GPU")
    device = torch.device("cuda:0")
    outputs = {key: Path(str(value)) for key, value in authority["outputs"].items()}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"refusing to replace source response output: {path}")

    inputs = authority["inputs"]
    for key in (
        "source_gate_preregistration",
        "source_view_preregistration",
        "scene_config",
        "geometry_checkpoint",
    ):
        validate_file_record(inputs[key], label=key)
    source_view_preregistration = _record(
        inputs["source_view_preregistration"], label="source view preregistration"
    )
    reseal_records, teacher_root, reseal_record = _load_source_reseal(
        authority, source_view_preregistration
    )
    text, query_ids, query_bank = _load_query_bank(authority)

    config_record = _record(inputs["scene_config"], label="scene config")
    geometry_record = _record(inputs["geometry_checkpoint"], label="geometry checkpoint")
    model, _codec, renderer, _sharpener, _refiner, config, _hybrid = load_render_pipeline(
        config_record["path"],
        geometry_record["path"],
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
        expected_checkpoint_sha256=geometry_record["sha256"],
    )
    if (
        int(model.get_xyz().shape[0]) != int(authority["geometry"]["num_gaussians"])
        or _geometry_sha256(model) != authority["geometry"]["xyz_sha256"]
    ):
        raise ValueError("source response frozen geometry fingerprint differs")
    frame_ids = list(authority["source_heldout_frame_ids"])
    dataset = _dataset(config, frame_ids)
    frame_to_index = {
        int(frame): index for index, frame in enumerate(dataset.frame_indices)
    }
    if sorted(frame_to_index) != frame_ids:
        raise ValueError("source response pose frame inventory differs")

    text_device = text.to(device)
    teacher_responses: dict[int, torch.Tensor] = {}
    teacher_nonzero: dict[int, torch.Tensor] = {}
    teacher_input_sha: dict[int, str] = {}
    for frame_id in frame_ids:
        record = reseal_records[frame_id]
        teacher = _load_teacher_tensor(teacher_root, record)
        teacher_float = teacher.float().to(device)
        unit = F.normalize(teacher_float, dim=0, eps=1e-12)
        teacher_responses[frame_id] = torch.einsum(
            "qd,dhw->qhw", text_device, unit
        ).cpu()
        teacher_nonzero[frame_id] = (
            torch.linalg.vector_norm(teacher.float(), dim=0) > 0
        )
        teacher_input_sha[frame_id] = str(record["sha256"])
        del teacher, teacher_float, unit

    summary_by_descriptor: dict[tuple[str, str], dict[str, object]] = {}
    output_records: dict[str, dict[str, str]] = {}
    role_to_output = {
        "control": outputs["control_summary"],
        "candidate": outputs["candidate_summary"],
    }
    for method in authority["methods"]:
        method_id = str(method["method_id"])
        descriptor = _record(method["descriptor_payload"], label=f"{method_id} descriptor")
        descriptor_key = (descriptor["path"], descriptor["sha256"])
        if descriptor_key in summary_by_descriptor:
            summary = _clone_equivalent_summary(
                summary_by_descriptor[descriptor_key], method_id=method_id
            )
        else:
            payload = load_descriptor_payload(
                method,
                scene_id=str(authority["scene_id"]),
                num_gaussians=int(model.get_xyz().shape[0]),
                xyz_sha256=str(authority["geometry"]["xyz_sha256"]),
            )
            primitive_descriptors = build_primitive_descriptor_rows(
                payload,
                num_gaussians=int(model.get_xyz().shape[0]),
                device=device,
            )
            del payload
            gc.collect()
            frame_summaries: list[dict[str, object]] = []
            for frame_id in frame_ids:
                pose = torch.from_numpy(
                    dataset.poses_w2c[frame_to_index[frame_id]]
                ).float().to(device)
                rendered = renderer.render_feature_rows(
                    model,
                    pose,
                    primitive_descriptors,
                    feature_height=46,
                    feature_width=62,
                    alpha_normalize=False,
                )
                descriptor_map = normalize_premultiplied(
                    rendered["feature_map"].float(), rendered["alpha_map"].float()
                )
                response_map = descriptor_map_to_text_responses(
                    descriptor_map, text_device
                ).cpu()
                alpha = rendered["alpha_map"].float().cpu()
                valid_mask = (alpha >= ALPHA_THRESHOLD) & teacher_nonzero[frame_id]
                frame_summaries.append(
                    evaluate_source_response_frame(
                        response_map,
                        teacher_responses[frame_id],
                        valid_mask,
                        scene_id=str(authority["scene_id"]),
                        frame_id=frame_id,
                        method_id=method_id,
                        query_ids=query_ids,
                        query_bank=query_bank,
                        method_input_sha256=descriptor["sha256"],
                        teacher_input_sha256=teacher_input_sha[frame_id],
                    )
                )
                print(
                    json.dumps(
                        {
                            "event": "source_response_frame_complete",
                            "scene_id": authority["scene_id"],
                            "method_id": method_id,
                            "frame_id": frame_id,
                            "valid_pixels": int(valid_mask.sum()),
                        }
                    ),
                    flush=True,
                )
                del rendered, descriptor_map, response_map, alpha, valid_mask
            summary = build_scene_summary(
                frame_summaries,
                scene_id=str(authority["scene_id"]),
                method_id=method_id,
                required_frame_ids=frame_ids,
            )
            summary_by_descriptor[descriptor_key] = summary
            del primitive_descriptors
            torch.cuda.empty_cache()
            gc.collect()
        output = role_to_output[str(method["role"])]
        write_frozen_json(output, summary)
        output_records[str(method["role"])] = file_record(output)

    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_source_only_compact_response_summaries",
        "scene_id": authority["scene_id"],
        "execution_authority": authority_record,
        "source_reseal": reseal_record,
        "outputs": output_records,
        "equivalence_smoke": authority["equivalence_smoke"],
        "descriptor_payloads_loaded": len(summary_by_descriptor),
        "source_frames": len(frame_ids),
        "query_count": len(query_ids),
        "access_audit": authority["access_audit"],
        "target_metric_execution_authorized": False,
        "target_metric_executed": False,
        "next_gate": "compact_cpu_source_text_response_ranking_selector",
    }
    write_frozen_json(outputs["result"], result)
    return {**result, "result": file_record(outputs["result"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--execution-authority-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                args.execution_authority, args.execution_authority_sha256
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "AUTHORITY_SCHEMA",
    "IMPLEMENTATION",
    "RESULT_SCHEMA",
    "build_primitive_descriptor_rows",
    "descriptor_map_to_text_responses",
    "load_authority",
    "load_descriptor_payload",
    "materialize",
    "validate_authority",
]
