#!/usr/bin/env python3
"""Materialize one target-blind semantic descriptor per primitive on CPU.

The production query contract is deliberately scalar-first: a normalized
primitive descriptor ``e_sem[i]`` is queried once as
``s_i(q) = dot(e_sem[i], q)`` and the frozen compositor consumes only those
scalar rows.  Rendering a 1536-D descriptor followed by normalization is not
part of the method.  A frozen CPU replay additionally evaluates the linear
render-then-query ordering, without normalization, solely to audit that both
orderings agree to at most 1e-6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.evaluation.text_response_fidelity import (
    canonical_json_sha256,
    tensor_sha256,
)
from radio_gs.interfaces.surface_region_summary import SurfaceRegionSummaryReadoutV2
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.models.surface_region_dual_descriptor import SurfaceRegionDualDescriptor
from radio_gs.scripts.select_query_free_scalar_compositor import (
    FIXED_SCALAR_OPERATOR_CONTRACT,
    SCREEN_NAME,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "surface_dual_descriptor_primitive_cache"
SCALAR_ARTIFACT_TYPE = "surface_dual_descriptor_scalar_replay_cache"
INPUT_ARTIFACT_TYPE = "surface_dual_descriptor_primitive_input_cache"
ADAPTER_ARTIFACT_TYPE = "surface_region_dual_descriptor_residual_seed0_pilot"
WEIGHTS_ARTIFACT_TYPE = "frozen_scalar_compositor_replay_weights"
POINT_RENDER_MAX_ABS_ERROR = 1e-6
DESCRIPTOR_DIMENSION = 1536

_QUERY_FREE_FLAGS = (
    "uses_benchmark_scenes",
    "uses_benchmark_test_vocabulary",
    "benchmark_targets_opened",
    "annotations_opened",
    "labels_opened",
    "instances_opened",
    "masks_opened",
    "text_opened",
)
_INPUT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "primitive_ids",
        "radio_features",
        "geometry",
        "token_mask",
        "reliability",
        "anchor_index",
        "official_summary_tokens",
        "official_descriptors",
        "metadata",
    }
)
_WEIGHTS_KEYS_V1 = frozenset(
    {
        "schema_version",
        "artifact_type",
        "primitive_ids",
        "contribution_weights",
        "query_bank",
        "scalar_compositor_manifest",
        "metadata",
    }
)
_WEIGHTS_KEYS_V2 = frozenset(
    {
        "schema_version",
        "artifact_type",
        "primitive_ids",
        "render_row_keys",
        "render_row_index",
        "primitive_row_index",
        "contribution_weights",
        "query_bank",
        "scalar_compositor_manifest",
        "source_contribution_caches",
        "metadata",
    }
)
_ADAPTER_STATE_KEYS = frozenset(
    {
        "context_norm.weight",
        "context_norm.bias",
        "context_projection.weight",
        "context_projection.bias",
        "film.weight",
        "film.bias",
        "gate.weight",
        "gate.bias",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu")
    except Exception as error:
        raise ValueError(f"cannot load {label}: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _canonical_file(raw: object, *, label: str) -> Path:
    if not isinstance(raw, (str, os.PathLike)) or not str(raw):
        raise ValueError(f"{label} path is missing")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"{label} is missing") from error
    if resolved != path or not resolved.is_file():
        raise ValueError(f"{label} must be a canonical non-symlink regular file")
    return resolved


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256_file(path)}


def _validate_file_record(
    raw: object,
    *,
    expected: Path,
    label: str,
) -> dict[str, str]:
    if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path/SHA256 record")
    path = _canonical_file(raw.get("path"), label=label)
    record = _file_record(path)
    if path != expected or dict(raw) != record:
        raise ValueError(f"{label} binding differs")
    return record


def _assert_query_free(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} target-blind provenance is missing")
    for key in _QUERY_FREE_FLAGS:
        if value.get(key) is not False:
            raise ValueError(f"{label} must explicitly certify {key}=false")
    return value


def _cpu_tensor(
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
        raise ValueError(f"{label} must not be floating point")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains a non-finite value")
    return value


def _validate_primitive_ids(value: object, *, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{label} must contain unique non-empty primitive IDs")
    return list(value)


def _validate_input_cache(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _torch_load(path, label="primitive input cache")
    if set(payload) != _INPUT_KEYS:
        raise ValueError(f"{path}: primitive input cache fields differ")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != INPUT_ARTIFACT_TYPE
    ):
        raise ValueError(f"{path}: unsupported primitive input cache")
    metadata = _assert_query_free(payload.get("metadata"), label=str(path))
    if (
        metadata.get("complete_primitive_rows") is not True
        or metadata.get("target_blind") is not True
        or metadata.get("benchmark_targets_or_metrics_used") is not False
    ):
        raise ValueError(f"{path}: input cache is not complete and target-blind")
    expected_builder = Path(__file__).resolve().with_name(
        "build_surface_dual_descriptor_primitive_input_cache.py"
    )
    if (
        not isinstance(metadata.get("primitive_input_builder_implementation"), Mapping)
        or dict(metadata["primitive_input_builder_implementation"])
        != _file_record(expected_builder)
    ):
        raise ValueError(f"{path}: primitive input builder binding differs")
    ids = _validate_primitive_ids(payload.get("primitive_ids"), label=str(path))
    rows = len(ids)
    row_authority = metadata.get("production_primitive_row_authority")
    required_row_authority = {
        "contract",
        "scene_id",
        "geometry_checkpoint",
        "geometry_xyz_sha256",
        "total_geometry_rows",
        "row_order",
        "complete_geometry_rows_present",
    }
    scene_id = str(row_authority.get("scene_id", "")) if isinstance(
        row_authority, Mapping
    ) else ""
    geometry_record = (
        row_authority.get("geometry_checkpoint")
        if isinstance(row_authority, Mapping)
        else None
    )
    if (
        not isinstance(row_authority, Mapping)
        or set(row_authority) != required_row_authority
        or row_authority.get("contract")
        != "complete_single_scene_gaussian_checkpoint_row_order_v1"
        or row_authority.get("row_order")
        != "zero_based_geometry_checkpoint_row_order"
        or row_authority.get("complete_geometry_rows_present") is not True
        or row_authority.get("total_geometry_rows") != rows
        or len(str(row_authority.get("geometry_xyz_sha256", ""))) != 64
        or ids != [f"{scene_id}/primitive-{index}" for index in range(rows)]
        or not isinstance(geometry_record, Mapping)
        or set(geometry_record) != {"path", "sha256"}
    ):
        raise ValueError(f"{path}: complete production primitive row order differs")
    geometry_path = _canonical_file(
        geometry_record.get("path"),
        label=f"{path} production geometry",
    )
    if dict(geometry_record) != _file_record(geometry_path):
        raise ValueError(f"{path}: production geometry binding differs")
    tensors = {
        "radio_features": _cpu_tensor(
            payload.get("radio_features"), label=f"{path} radio_features", floating=True
        ),
        "geometry": _cpu_tensor(
            payload.get("geometry"), label=f"{path} geometry", floating=True
        ),
        "token_mask": _cpu_tensor(
            payload.get("token_mask"), label=f"{path} token_mask"
        ).bool(),
        "reliability": _cpu_tensor(
            payload.get("reliability"), label=f"{path} reliability", floating=True
        ),
        "anchor_index": _cpu_tensor(
            payload.get("anchor_index"), label=f"{path} anchor_index", floating=False
        ).long(),
        "official_summary_tokens": _cpu_tensor(
            payload.get("official_summary_tokens"),
            label=f"{path} official_summary_tokens",
            floating=True,
        ),
        "official_descriptors": _cpu_tensor(
            payload.get("official_descriptors"),
            label=f"{path} official_descriptors",
            floating=True,
        ),
    }
    features = tensors["radio_features"]
    geometry = tensors["geometry"]
    mask = tensors["token_mask"]
    reliability = tensors["reliability"]
    anchor = tensors["anchor_index"]
    if features.ndim != 3 or features.shape[0] != rows:
        raise ValueError(f"{path}: radio_features row shape differs")
    if geometry.ndim != 3 or geometry.shape[:2] != features.shape[:2]:
        raise ValueError(f"{path}: geometry shape differs")
    if mask.shape != features.shape[:2] or not bool(mask.any(dim=1).all()):
        raise ValueError(f"{path}: token_mask shape/support differs")
    if reliability.ndim == 3 and reliability.shape[-1] == 1:
        reliability = reliability[..., 0]
        tensors["reliability"] = reliability
    if reliability.shape != features.shape[:2] or bool((reliability < 0).any()):
        raise ValueError(f"{path}: reliability shape/value differs")
    if anchor.shape != (rows,) or bool((anchor < 0).any()) or bool(
        (anchor >= features.shape[1]).any()
    ):
        raise ValueError(f"{path}: anchor_index shape/value differs")
    if not bool(mask[torch.arange(rows), anchor].all()):
        raise ValueError(f"{path}: every anchor must be active")
    if tensors["official_summary_tokens"].shape != (rows, features.shape[-1]):
        raise ValueError(f"{path}: official token replay shape differs")
    if tensors["official_descriptors"].shape != (rows, DESCRIPTOR_DIMENSION):
        raise ValueError(f"{path}: official descriptor replay shape differs")
    norms = tensors["official_descriptors"].float().norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=0.0):
        raise ValueError(f"{path}: official descriptor replay is not normalized")
    record = _file_record(path)
    record["rows"] = rows
    return {"primitive_ids": ids, **tensors}, record


def _merge_input_caches(
    paths: Sequence[Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not paths:
        raise ValueError("at least one input cache is required")
    loaded = [_validate_input_cache(path) for path in paths]
    ids = [item for data, _ in loaded for item in data["primitive_ids"]]
    if len(ids) != len(set(ids)):
        raise ValueError("input caches contain duplicate primitive IDs")
    tensor_names = (
        "radio_features",
        "geometry",
        "token_mask",
        "reliability",
        "anchor_index",
        "official_summary_tokens",
        "official_descriptors",
    )
    merged: dict[str, Any] = {"primitive_ids": ids}
    for name in tensor_names:
        try:
            merged[name] = torch.cat([data[name] for data, _ in loaded], dim=0)
        except RuntimeError as error:
            raise ValueError(f"input cache {name} tensors are incompatible") from error
    return merged, [record for _, record in loaded]


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {label}: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _validate_compositor_manifest(path: Path) -> dict[str, str]:
    payload = _load_json(path, label="scalar compositor manifest")
    if (
        payload.get("schema_version") != 1
        or payload.get("screen") != SCREEN_NAME
        or payload.get("scalar_operator_contract") != FIXED_SCALAR_OPERATOR_CONTRACT
        or payload.get("selection_uses_benchmark_scenes") is not False
        or payload.get("queries_opened") is not False
        or payload.get("masks_opened") is not False
        or payload.get("labels_opened") is not False
        or not isinstance(payload.get("selected_variant"), str)
    ):
        raise ValueError("scalar compositor manifest is not a frozen query-free selection")
    return {**_file_record(path), "selected_variant": str(payload["selected_variant"])}


def _validate_replay_weights(
    path: Path,
    *,
    primitive_ids: list[str],
    compositor: dict[str, str],
    input_cache_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], torch.Tensor, dict[str, str]]:
    payload = _torch_load(path, label="scalar compositor replay weights")
    version = payload.get("schema_version")
    expected_keys = _WEIGHTS_KEYS_V1 if version == 1 else _WEIGHTS_KEYS_V2
    if version not in (1, 2) or set(payload) != expected_keys:
        raise ValueError("scalar compositor replay-weight fields differ")
    if (
        payload.get("artifact_type") != WEIGHTS_ARTIFACT_TYPE
        or payload.get("primitive_ids") != primitive_ids
    ):
        raise ValueError("scalar compositor replay weights bind different primitives")
    _validate_file_record(
        payload.get("scalar_compositor_manifest"),
        expected=Path(compositor["path"]),
        label="replay scalar compositor manifest",
    )
    metadata = _assert_query_free(payload.get("metadata"), label=str(path))
    if (
        metadata.get("target_blind") is not True
        or metadata.get("benchmark_targets_or_metrics_used") is not False
        or metadata.get("frozen_before_materialization") is not True
        or metadata.get("query_bank_source") != "target_blind_replay_only"
        or metadata.get("selected_variant") != compositor["selected_variant"]
        or metadata.get("weights_semantics")
        != (
            "frozen_selected_normalized_contribution_weights"
            if version == 1
            else "frozen_selected_normalized_sparse_contribution_weights"
        )
    ):
        raise ValueError("scalar replay weights are not frozen and target-blind")
    weights = _cpu_tensor(
        payload.get("contribution_weights"),
        label="contribution_weights",
        floating=True,
    ).float().contiguous()
    queries = _cpu_tensor(
        payload.get("query_bank"), label="replay query_bank", floating=True
    ).float().contiguous()
    if version == 1:
        if (
            weights.ndim != 2
            or weights.shape[1] != len(primitive_ids)
            or weights.shape[0] == 0
            or bool((weights < 0).any())
            or not bool((weights.sum(dim=1) > 0).all())
        ):
            raise ValueError("contribution weights must be nonnegative [R,N] rows")
        weight_sums = weights.sum(dim=1)
        if not torch.allclose(
            weight_sums, torch.ones_like(weight_sums), atol=1e-6, rtol=0.0
        ):
            raise ValueError("contribution weights must be frozen normalized rows")
        operator = {"schema_version": 1, "layout": "dense_v1", "weights": weights}
    else:
        render_keys = payload.get("render_row_keys")
        render_rows = _cpu_tensor(
            payload.get("render_row_index"),
            label="render_row_index",
            floating=False,
        ).long().reshape(-1)
        primitive_rows = _cpu_tensor(
            payload.get("primitive_row_index"),
            label="primitive_row_index",
            floating=False,
        ).long().reshape(-1)
        if (
            not isinstance(render_keys, list)
            or not render_keys
            or any(not isinstance(value, str) or not value for value in render_keys)
            or len(set(render_keys)) != len(render_keys)
            or weights.ndim != 1
            or weights.numel() == 0
            or weights.shape != render_rows.shape
            or weights.shape != primitive_rows.shape
            or bool((weights < 0).any())
            or bool((render_rows < 0).any())
            or bool((render_rows >= len(render_keys)).any())
            or bool((primitive_rows < 0).any())
            or bool((primitive_rows >= len(primitive_ids)).any())
        ):
            raise ValueError("sparse compositor replay triplets differ")
        weight_sums = torch.zeros(len(render_keys), dtype=torch.float32)
        weight_sums.index_add_(0, render_rows, weights)
        if not torch.allclose(
            weight_sums, torch.ones_like(weight_sums), atol=1e-6, rtol=0.0
        ):
            raise ValueError("sparse contribution rows must be frozen normalized")
        source_records = payload.get("source_contribution_caches")
        if (
            not isinstance(source_records, list)
            or not source_records
            or any(
                not isinstance(record, Mapping)
                or set(record) != {"path", "sha256"}
                or len(str(record.get("sha256", ""))) != 64
                for record in source_records
            )
        ):
            raise ValueError("sparse replay source contribution bindings differ")
        for index, record in enumerate(source_records):
            source = _canonical_file(
                record.get("path"),
                label=f"sparse replay source contribution cache {index}",
            )
            if dict(record) != _file_record(source):
                raise ValueError("sparse replay source contribution binding differs")
        primitive_input = metadata.get("primitive_input_cache")
        if (
            len(input_cache_records) != 1
            or not isinstance(primitive_input, Mapping)
            or set(primitive_input) != {"path", "sha256"}
            or dict(primitive_input)
            != {
                "path": input_cache_records[0]["path"],
                "sha256": input_cache_records[0]["sha256"],
            }
        ):
            raise ValueError("sparse replay primitive-input binding differs")
        query_source = metadata.get("query_source")
        if (
            not isinstance(query_source, Mapping)
            or set(query_source)
            != {"artifact", "manifest", "selection", "selected_row_indices"}
        ):
            raise ValueError("sparse replay target-blind query binding differs")
        builder_record = metadata.get("replay_builder_implementation")
        expected_builder = Path(__file__).resolve().with_name(
            "build_frozen_scalar_compositor_replay.py"
        )
        if (
            not isinstance(builder_record, Mapping)
            or set(builder_record) != {"path", "sha256"}
            or dict(builder_record) != _file_record(expected_builder)
        ):
            raise ValueError("sparse replay builder implementation binding differs")
        if (
            query_source.get("selection")
            != "first_8_rows_of_frozen_order_or_all_if_fewer"
            or query_source.get("selected_row_indices")
            != list(range(int(queries.shape[0])))
        ):
            raise ValueError("sparse replay target-blind query selection differs")
        for key in ("artifact", "manifest"):
            raw_record = query_source[key]
            if not isinstance(raw_record, Mapping) or set(raw_record) != {
                "path",
                "sha256",
            }:
                raise ValueError("sparse replay target-blind query binding differs")
            source = _canonical_file(
                raw_record.get("path"), label=f"sparse replay query {key}"
            )
            if dict(raw_record) != _file_record(source):
                raise ValueError("sparse replay target-blind query binding differs")
        operator = {
            "schema_version": 2,
            "layout": "sparse_triplets_v2",
            "num_render_rows": len(render_keys),
            "render_row_index": render_rows,
            "primitive_row_index": primitive_rows,
            "weights": weights,
        }
    if queries.ndim != 2 or queries.shape[1] != DESCRIPTOR_DIMENSION or queries.shape[0] == 0:
        raise ValueError("replay query bank must be nonempty [Q,1536]")
    query_norms = queries.norm(dim=-1)
    if not torch.allclose(query_norms, torch.ones_like(query_norms), atol=1e-6, rtol=0.0):
        raise ValueError("replay query bank must already be normalized")
    return operator, queries, _file_record(path)


def _adapter_state(model: SurfaceRegionDualDescriptor) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    return {key: state[key] for key in sorted(_ADAPTER_STATE_KEYS)}


def _validate_adapter_checkpoint(
    path: Path,
    *,
    model: SurfaceRegionDualDescriptor,
    base_record: dict[str, str],
    radio_record: dict[str, str],
    cache_records: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = _torch_load(path, label="dual-descriptor adapter checkpoint")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != ADAPTER_ARTIFACT_TYPE
        or payload.get("training_complete") is not True
        or payload.get("gate_status")
        != "training_complete_pending_point_render_replay"
        or payload.get("pilot_advance_gate_passed") is not False
        or payload.get("continuation_authorized") is not False
        or payload.get("seed1_executed") is not False
        or payload.get("additional_seed_or_architecture_authorized") is not False
    ):
        raise ValueError("adapter checkpoint is not the pending seed-0 pilot")
    provenance = payload.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("external_benchmarks_opened") is not False
        or provenance.get("benchmark_vocabulary_opened") is not False
        or provenance.get("benchmark_images_opened") is not False
        or provenance.get("benchmark_masks_opened") is not False
        or provenance.get("benchmark_targets_opened") is not False
        or provenance.get("metric_continuation") is not False
        or provenance.get("fit_split_only_for_optimizer") is not True
        or provenance.get("dev_split_used_for_selection_only") is not True
    ):
        raise ValueError("adapter checkpoint is not target-blind fit-only")
    surface_control = provenance.get("surface_control")
    if not isinstance(surface_control, Mapping):
        raise ValueError("adapter checkpoint lacks its frozen surface control")
    _validate_file_record(
        {"path": surface_control.get("path"), "sha256": surface_control.get("sha256")},
        expected=Path(base_record["path"]),
        label="adapter base checkpoint",
    )
    _validate_file_record(
        provenance.get("radio_checkpoint"),
        expected=Path(radio_record["path"]),
        label="adapter RADIO checkpoint",
    )
    # The materialization inputs are independent primitive rows and are bound
    # by this output.  The training checkpoint must still bind its own frozen
    # fit/dev caches exactly; it must not claim that they are these inputs.
    for split in ("train_caches", "validation_caches"):
        bindings = provenance.get(split)
        if (
            not isinstance(bindings, list)
            or not bindings
            or any(
                not isinstance(record, Mapping)
                or set(record) != {"path", "sha256"}
                or len(str(record.get("sha256", ""))) != 64
                for record in bindings
            )
        ):
            raise ValueError(f"adapter checkpoint {split} bindings differ")

    live_architecture = model.architecture()
    if payload.get("dual_descriptor_architecture") != live_architecture:
        raise ValueError(
            "adapter checkpoint architecture differs: "
            f"checkpoint={payload.get('dual_descriptor_architecture')!r}, "
            f"live={live_architecture!r}"
        )
    base_state = payload.get("base_surface_state_dict")
    reference_base = model.summary_readout.state_dict()
    if not isinstance(base_state, Mapping) or set(base_state) != set(reference_base):
        raise ValueError("adapter checkpoint frozen base state fields differ")
    for key, reference in reference_base.items():
        value = _cpu_tensor(base_state[key], label=f"frozen base state {key}")
        if value.dtype != reference.dtype or not torch.equal(value, reference.cpu()):
            raise ValueError(f"adapter checkpoint frozen base state {key} differs")
    base_state_sha = _state_dict_sha256(base_state, label="frozen base state")
    if payload.get("base_surface_state_dict_sha256") != base_state_sha:
        raise ValueError("adapter checkpoint frozen base state SHA256 differs")

    state = payload.get("adapter_state_dict")
    if not isinstance(state, Mapping) or set(state) != _ADAPTER_STATE_KEYS:
        raise ValueError("adapter checkpoint state fields differ")
    adapter_state_sha = _state_dict_sha256(state, label="adapter state")
    if payload.get("adapter_state_dict_sha256") != adapter_state_sha:
        raise ValueError("adapter checkpoint state SHA256 differs")
    reference = _adapter_state(model)
    updated = dict(model.state_dict())
    for key, value in state.items():
        tensor = _cpu_tensor(value, label=f"adapter state {key}", floating=True)
        if tensor.shape != reference[key].shape:
            raise ValueError(f"adapter state {key} shape differs")
        updated[key] = tensor
    model.load_state_dict(updated, strict=True)
    report_path = path.with_suffix(path.suffix + ".json")
    if not report_path.is_file() or report_path.is_symlink():
        raise FileNotFoundError("dual-descriptor adapter checkpoint report is missing")
    report = _load_json(report_path, label="dual-descriptor adapter report")
    if (
        report.get("schema_version") != 1
        or report.get("artifact_type") != f"{ADAPTER_ARTIFACT_TYPE}_report"
        or Path(str(report.get("output", ""))).resolve() != path
        or report.get("checkpoint_sha256") != _sha256_file(path)
        or report.get("gate_status") != payload.get("gate_status")
        or report.get("pilot_advance_gate_passed") is not False
        or report.get("dual_descriptor_architecture") != live_architecture
        or report.get("base_surface_state_dict_sha256") != base_state_sha
        or report.get("adapter_state_dict_sha256") != adapter_state_sha
        or report.get("best_epoch") != payload.get("best_epoch")
    ):
        raise ValueError("dual-descriptor adapter report binding differs")
    return {
        "payload": payload,
        "report": _file_record(report_path.resolve()),
        "training_caches": {
            "train": list(provenance["train_caches"]),
            "validation": list(provenance["validation_caches"]),
        },
    }


def _assert_base_target_blind(payload: Mapping[str, Any], *, label: str) -> None:
    provenance = payload.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("uses_benchmark_scenes") is not False
        or provenance.get("uses_benchmark_test_vocabulary") is not False
        or provenance.get("frozen") is not True
        or provenance.get("scene_disjoint") is not True
        or provenance.get("custom_text_projection") is not False
    ):
        raise ValueError("base checkpoint is not the frozen scene-disjoint official path")


def _content_digest(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(payload)


def _state_dict_sha256(state: object, *, label: str) -> str:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{label} must be a non-empty state dict")
    records = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor):
            raise ValueError(f"{label} state fields differ")
        tensor = _cpu_tensor(value, label=f"{label} {name}")
        records.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "tensor_sha256": tensor_sha256(tensor),
            }
        )
    return canonical_json_sha256(records)


def _atomic_publish(files: Sequence[tuple[Path, bytes]]) -> None:
    outputs = [path for path, _ in files]
    if len(set(outputs)) != len(outputs):
        raise ValueError("output, scalar output, and report paths must be distinct")
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite materializer output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    temporaries: list[Path] = []
    published: list[Path] = []
    try:
        for path, content in files:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            os.close(descriptor)
            temporary = Path(name)
            temporary.write_bytes(content)
            temporaries.append(temporary)
        for (path, _), temporary in zip(files, temporaries):
            os.link(temporary, path)
            published.append(path)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporaries:
            path.unlink(missing_ok=True)


def _torch_bytes(value: object, *, parent: Path) -> tuple[bytes, str]:
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".dual-descriptor.", suffix=".pt", dir=parent)
    os.close(descriptor)
    path = Path(name)
    try:
        torch.save(value, path)
        content = path.read_bytes()
    finally:
        path.unlink(missing_ok=True)
    return content, hashlib.sha256(content).hexdigest()


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.device).lower() != "cpu":
        raise ValueError("this materializer is CPU-only; --device must be cpu")
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output = Path(args.output).resolve()
    scalar_output = Path(args.scalar_output).resolve()
    report = Path(args.report).resolve()
    for candidate in (output, scalar_output, report):
        if candidate.exists():
            raise FileExistsError(f"refusing to overwrite materializer output: {candidate}")

    cache_paths = sorted(
        (_canonical_file(raw, label="input cache") for raw in args.input_cache),
        key=str,
    )
    inputs, cache_records = _merge_input_caches(cache_paths)
    base_path = _canonical_file(args.base_checkpoint, label="base checkpoint")
    adapter_path = _canonical_file(
        args.adapter_checkpoint, label="adapter checkpoint"
    )
    radio_path = _canonical_file(args.radio_checkpoint, label="RADIO checkpoint")
    compositor_path = _canonical_file(
        args.scalar_compositor_manifest, label="scalar compositor manifest"
    )
    weights_path = _canonical_file(
        args.compositor_weights, label="compositor replay weights"
    )
    base_record = _file_record(base_path)
    adapter_record = _file_record(adapter_path)
    radio_record = _file_record(radio_path)
    compositor_record = _validate_compositor_manifest(compositor_path)

    base, base_payload = SurfaceRegionSummaryReadoutV2.from_checkpoint(
        base_path, map_location="cpu"
    )
    _assert_base_target_blind(base_payload, label=str(base_path))
    head = SigLIP2SummaryHead.from_radio_checkpoint(str(radio_path)).cpu().eval()
    model = SurfaceRegionDualDescriptor(base, head).cpu().eval()
    adapter_authority = _validate_adapter_checkpoint(
        adapter_path,
        model=model,
        base_record=base_record,
        radio_record=radio_record,
        cache_records=cache_records,
    )
    model.requires_grad_(False)
    replay_operator, queries, weights_record = _validate_replay_weights(
        weights_path,
        primitive_ids=inputs["primitive_ids"],
        compositor=compositor_record,
        input_cache_records=cache_records,
    )

    semantic_parts: list[torch.Tensor] = []
    token_parts: list[torch.Tensor] = []
    official_parts: list[torch.Tensor] = []
    rows = len(inputs["primitive_ids"])
    for start in range(0, rows, batch_size):
        stop = min(rows, start + batch_size)
        result = model(
            inputs["radio_features"][start:stop],
            inputs["geometry"][start:stop],
            anchor_index=inputs["anchor_index"][start:stop],
            token_mask=inputs["token_mask"][start:stop],
            reliability=inputs["reliability"][start:stop],
        )
        semantic_parts.append(result.semantic_descriptor.float().cpu())
        token_parts.append(result.official_token.float().cpu())
        official_parts.append(result.official_descriptor.float().cpu())
    semantic = torch.cat(semantic_parts).contiguous()
    official_tokens = torch.cat(token_parts).contiguous()
    official_descriptors = torch.cat(official_parts).contiguous()
    if semantic.shape != (rows, DESCRIPTOR_DIMENSION):
        raise ValueError("materialized semantic descriptor shape must be [N,1536]")
    norms = semantic.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=0.0):
        raise ValueError("materialized semantic descriptors are not normalized")
    token_bitwise = torch.equal(official_tokens, inputs["official_summary_tokens"].float())
    descriptor_bitwise = torch.equal(
        official_descriptors, inputs["official_descriptors"].float()
    )
    if not token_bitwise or not descriptor_bitwise:
        raise ValueError("official token/descriptor replay is not bitwise exact")

    # Production path: query each primitive once, then render only scalars.
    primitive_scores = semantic @ queries.T
    if replay_operator["layout"] == "dense_v1":
        weights = replay_operator["weights"]
        point_then_render = weights @ primitive_scores
        rendered_semantic = weights @ semantic
    else:
        render_rows = replay_operator["render_row_index"]
        primitive_rows = replay_operator["primitive_row_index"]
        sparse_weights = replay_operator["weights"]
        point_then_render = torch.zeros(
            replay_operator["num_render_rows"],
            queries.shape[0],
            dtype=torch.float32,
        )
        point_then_render.index_add_(
            0,
            render_rows,
            sparse_weights[:, None] * primitive_scores[primitive_rows],
        )
        rendered_semantic = torch.zeros(
            replay_operator["num_render_rows"],
            semantic.shape[1],
            dtype=torch.float32,
        )
        rendered_semantic.index_add_(
            0,
            render_rows,
            sparse_weights[:, None] * semantic[primitive_rows],
        )
    # Audit-only second path: linear high-D compositing without normalization.
    # It is never persisted as a descriptor cache or used for prediction.
    render_then_query = rendered_semantic @ queries.T
    replay_error = float((point_then_render - render_then_query).abs().max().item())
    if not math.isfinite(replay_error) or replay_error > POINT_RENDER_MAX_ABS_ERROR:
        raise ValueError(
            "point/render scalar replay exceeds 1e-6: " f"{replay_error:.9g}"
        )

    query_sha = tensor_sha256(queries)
    primitive_score_sha = tensor_sha256(primitive_scores)
    point_render_sha = canonical_json_sha256(
        {
            "query_bank_sha256": query_sha,
            "primitive_scalar_scores_sha256": primitive_score_sha,
            "point_then_render_scores_sha256": tensor_sha256(point_then_render),
            "render_then_query_scores_sha256": tensor_sha256(render_then_query),
            "point_render_replay_max_abs_error": replay_error,
        }
    )
    descriptor_content = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "primitive_ids_sha256": canonical_json_sha256(inputs["primitive_ids"]),
        "semantic_descriptors_sha256": tensor_sha256(semantic),
        "official_summary_tokens_sha256": tensor_sha256(official_tokens),
        "official_descriptors_sha256": tensor_sha256(official_descriptors),
        "input_caches": [
            {"path": row["path"], "sha256": row["sha256"]} for row in cache_records
        ],
        "base_checkpoint": base_record,
        "adapter_checkpoint": adapter_record,
        "adapter_checkpoint_report": adapter_authority["report"],
        "adapter_training_caches": adapter_authority["training_caches"],
        "radio_checkpoint": radio_record,
        "scalar_compositor_manifest": compositor_record,
        "compositor_weights": weights_record,
        "scalar_score_replay_sha256": point_render_sha,
    }
    descriptor_content_sha = _content_digest(descriptor_content)
    scalar_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": SCALAR_ARTIFACT_TYPE,
        "descriptor_cache_content_sha256": descriptor_content_sha,
        "primitive_ids": inputs["primitive_ids"],
        "query_bank": queries,
        "primitive_scalar_scores": primitive_scores.contiguous(),
        "point_then_render_scores": point_then_render.contiguous(),
        "render_then_query_scores": render_then_query.contiguous(),
        "query_bank_sha256": query_sha,
        "primitive_scalar_scores_sha256": primitive_score_sha,
        "point_then_render_scores_sha256": tensor_sha256(point_then_render),
        "render_then_query_scores_sha256": tensor_sha256(render_then_query),
        "scalar_score_replay_sha256": point_render_sha,
        "point_render_replay_max_abs_error": replay_error,
        "scalar_compositor_manifest": compositor_record,
        "compositor_weights": weights_record,
        "contract": {
            "primitive_scalar": "s_i(q)=dot(e_sem[i],q)",
            "direct_point_path": "primitive_scalar_scores",
            "rendered_production_path": "W@primitive_scalar_scores",
            "shared_between_paths": "primitive_scalar_scores_only",
            "render_1536d_then_renormalize": False,
            "render_then_query_is_audit_only": True,
            "render_then_query_normalization": "none",
            "tolerance": POINT_RENDER_MAX_ABS_ERROR,
            "replay_weights_schema_version": replay_operator["schema_version"],
            "replay_operator_layout": replay_operator["layout"],
        },
    }
    scalar_bytes, scalar_file_sha = _torch_bytes(scalar_payload, parent=scalar_output.parent)
    descriptor_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "primitive_ids": inputs["primitive_ids"],
        "semantic_descriptors": semantic,
        "official_summary_tokens": official_tokens,
        "official_descriptors": official_descriptors,
        "descriptor_dimension": DESCRIPTOR_DIMENSION,
        "descriptor_normalization": "l2_per_primitive_before_query",
        "descriptor_cache_content": descriptor_content,
        "descriptor_cache_content_sha256": descriptor_content_sha,
        "scalar_cache": {"path": str(scalar_output), "sha256": scalar_file_sha},
        "official_replay": {
            "official_token_bitwise_equal": token_bitwise,
            "official_descriptor_bitwise_equal": descriptor_bitwise,
            "official_summary_tokens_sha256": tensor_sha256(official_tokens),
            "official_descriptors_sha256": tensor_sha256(official_descriptors),
        },
        "provenance": {
            **{key: False for key in _QUERY_FREE_FLAGS},
            "target_blind": True,
            "benchmark_targets_or_metrics_used": False,
            "device": "cpu",
            "input_caches": descriptor_content["input_caches"],
            "base_checkpoint": base_record,
            "adapter_checkpoint": adapter_record,
            "adapter_checkpoint_report": adapter_authority["report"],
            "radio_checkpoint": radio_record,
            "scalar_compositor_manifest": compositor_record,
            "compositor_weights": weights_record,
        },
        "query_contract": scalar_payload["contract"],
    }
    descriptor_bytes, descriptor_file_sha = _torch_bytes(
        descriptor_payload, parent=output.parent
    )
    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "surface_dual_descriptor_materialization_report",
        "descriptor_cache": {"path": str(output), "sha256": descriptor_file_sha},
        "descriptor_cache_content_sha256": descriptor_content_sha,
        "scalar_cache": {"path": str(scalar_output), "sha256": scalar_file_sha},
        "scalar_score_replay_sha256": point_render_sha,
        "query_bank_sha256": query_sha,
        "primitive_scalar_scores_sha256": primitive_score_sha,
        "point_render_replay_max_abs_error": replay_error,
        "point_render_replay_passed": True,
        "formal_point_render_replay_evidence_eligible": (
            replay_operator["schema_version"] == 2
        ),
        "replay_weights_schema_version": replay_operator["schema_version"],
        "replay_operator_layout": replay_operator["layout"],
        "point_render_replay_evidence": {
            "schema_version": 1,
            "artifact_type": "dual_descriptor_point_render_replay_evidence",
            "candidate_adapter_state_dict_sha256": adapter_authority["payload"][
                "adapter_state_dict_sha256"
            ],
            "independent_materializer_replay": True,
            "frozen_scalar_compositor_replay": (
                replay_operator["schema_version"] == 2
            ),
            "point_render_replay_max_abs_error": replay_error,
        },
        "official_token_bitwise_equal": token_bitwise,
        "official_descriptor_bitwise_equal": descriptor_bitwise,
        "primitive_rows": rows,
        "descriptor_dimension": DESCRIPTOR_DIMENSION,
        "input_caches": descriptor_content["input_caches"],
        "base_checkpoint": base_record,
        "adapter_checkpoint": adapter_record,
        "adapter_checkpoint_report": adapter_authority["report"],
        "adapter_training_caches": adapter_authority["training_caches"],
        "radio_checkpoint": radio_record,
        "scalar_compositor_manifest": compositor_record,
        "compositor_weights": weights_record,
        "production_contract": scalar_payload["contract"],
        "target_blind": True,
        "benchmark_targets_or_metrics_used": False,
        "device": "cpu",
    }
    report_bytes = (json.dumps(report_payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _atomic_publish(
        (
            (scalar_output, scalar_bytes),
            (output, descriptor_bytes),
            (report, report_bytes),
        )
    )
    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", action="append", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--scalar-compositor-manifest", required=True)
    parser.add_argument("--compositor-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scalar-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(materialize(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
