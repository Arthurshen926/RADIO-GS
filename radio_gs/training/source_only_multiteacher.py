"""Strict source-only teachers for Stage-B canonical-field distillation.

The canonical factorized field already reconstructs raw RADIO direction/gauge
and the official DINO/SAM pointwise views.  This module adds the missing
contract for readout-aligned source descriptors and their local semantic
relations without admitting benchmark queries, target RGB, labels, masks, or
metrics into field training.

The first executable adapter is the query-free canonical-top4 LERF teacher.
The contract also names the clean ScanNet region-view authority accepted by
the same Stage-B method; region-view tensors remain a separate aggregation
level and are deliberately not mislabelled as primitive targets.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from radio_gs.interfaces.factorized_primitive_state import (
    FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256,
    FACTORIZED_PRIMITIVE_STATE_SCHEMA,
    FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION,
    load_factorized_primitive_state,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.training.gauge_separated_capability import gauge_separated_radio
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    validate_file_record,
)


SOURCE_ONLY_MULTITEACHER_SCHEMA = (
    "radio_gs.canonical_field_source_only_multiteacher.v2"
)
SOURCE_ONLY_MULTITEACHER_SCHEMA_VERSION = 2
LERF_QUERY_FREE_TOP4_TEACHER_SCHEMA = (
    "radio_gs.lerf_teacher_agreement_from_canonical_top4.v1"
)
SCANNET_REGION_VIEW_TEACHER_SCHEMA = (
    "radio_gs.surface_region_official_siglip2_multiview_teacher_authority.v2"
)
PRIMITIVE_TEACHER_KIND = "primitive_siglip_direction"
REGION_VIEW_TEACHER_KIND = "region_view_siglip_direction"
FACTORIZED_STATE_KIND = "factorized_primitive_state_v2"
RELATION_GRAPH_KIND = "canonical_source_support_graph_v1"

# Reuse the existing capability/relation scale.  These are method constants,
# not CLI knobs and not scene-selected values.
SOURCE_DESCRIPTOR_WEIGHT = 0.20
SOURCE_RELATION_WEIGHT = 0.05
SOURCE_RELATION_EDGE_CAP = 32_768
SOURCE_RELATION_BATCH_SIZE = 512

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def source_only_multiteacher_contract() -> dict[str, Any]:
    return {
        "schema": SOURCE_ONLY_MULTITEACHER_SCHEMA,
        "schema_version": SOURCE_ONLY_MULTITEACHER_SCHEMA_VERSION,
        "primary_field_contract": "canonical-factorized-radio-v1",
        "teacher_levels": {
            PRIMITIVE_TEACHER_KIND: {
                "accepted_schema": LERF_QUERY_FREE_TOP4_TEACHER_SCHEMA,
                "projection": (
                    "stop_gradient_raw_gauge_then_frozen_official_siglip2_"
                    "summary_head_on_primitive_radio_proxy"
                ),
                "target": "unit_source_multiview_siglip_direction",
            },
            REGION_VIEW_TEACHER_KIND: {
                "accepted_schema": SCANNET_REGION_VIEW_TEACHER_SCHEMA,
                "projection": (
                    "surface_region_aggregation_then_frozen_siglip2_summary_head"
                ),
                "target": "unit_source_rgb_region_view_siglip_direction",
            },
            FACTORIZED_STATE_KIND: {
                "accepted_schema": FACTORIZED_PRIMITIVE_STATE_SCHEMA,
                "accepted_schema_version": (
                    FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION
                ),
                "contract_sha256": FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256,
                "role": "separate_gauge_dispersion_evidence_state",
            },
        },
        "loss": {
            "descriptor": "reliability_weighted_one_minus_cosine",
            "descriptor_weight": SOURCE_DESCRIPTOR_WEIGHT,
            "relation": "reliability_weighted_smooth_l1_edge_cosine",
            "relation_weight": SOURCE_RELATION_WEIGHT,
            "relation_edge_cap": SOURCE_RELATION_EDGE_CAP,
            "relation_edge_selection": (
                "ascending_source_edge_index_evenly_spaced_floor_v1"
            ),
            "relation_batch_size": SOURCE_RELATION_BATCH_SIZE,
            "student_gauge_gradient": (
                "zero_by_detached_norm_direction_reparameterization_v1"
            ),
            "reliability": (
                "uniform_half(sqrt(clamp((retained_count-1)/3,0,1)"
                "*directional_resultant))"
            ),
        },
        "selection": {
            "query_free": True,
            "source_only": True,
            "per_scene_or_per_query_hyperparameters": False,
            "benchmark_metrics_may_select_field": False,
        },
        "forbidden_inputs": [
            "benchmark_query_or_text_bank",
            "benchmark_target_rgb",
            "benchmark_ground_truth",
            "benchmark_label_or_mask",
            "benchmark_metric_or_prediction",
            "query_score_cache",
        ],
    }


SOURCE_ONLY_MULTITEACHER_CONTRACT_SHA256 = canonical_json_sha256(
    source_only_multiteacher_contract()
)


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    result = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    validate_file_record(result, label=label)
    return result


def _tensor_rows_sha256(values: torch.Tensor) -> str:
    array = (
        torch.as_tensor(values)
        .detach()
        .float()
        .cpu()
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_source_only_multiteacher_manifest(value: object) -> dict[str, Any]:
    """Validate the small immutable Stage-B control plane.

    Heavy tensor validation is intentionally performed only by
    :func:`load_source_only_multiteacher_bundle`; this function is cheap enough
    for preregistration and unit tests.
    """

    if not isinstance(value, Mapping):
        raise ValueError("source-only multiteacher manifest must be a mapping")
    payload = dict(value)
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "status",
        "scene_id",
        "field_control",
        "factorized_radio_cache",
        "factorized_primitive_state",
        "primitive_descriptor_teacher",
        "relation_graph",
        "official_radio_checkpoint",
        "loss",
        "source_gates",
        "access_audit",
        "execution",
    }
    contract = source_only_multiteacher_contract()
    if (
        set(payload) != required
        or payload.get("schema") != SOURCE_ONLY_MULTITEACHER_SCHEMA
        or payload.get("schema_version")
        != SOURCE_ONLY_MULTITEACHER_SCHEMA_VERSION
        or payload.get("contract") != contract
        or payload.get("contract_sha256")
        != SOURCE_ONLY_MULTITEACHER_CONTRACT_SHA256
        or payload.get("status") != "preregistered_gpu_not_started"
        or not str(payload.get("scene_id", ""))
    ):
        raise ValueError("source-only multiteacher manifest contract differs")
    records = {
        name: _record(payload[name], label=f"Stage-B {name}")
        for name in (
            "field_control",
            "factorized_radio_cache",
            "factorized_primitive_state",
            "primitive_descriptor_teacher",
            "relation_graph",
            "official_radio_checkpoint",
        )
    }
    loss = payload.get("loss")
    if loss != contract["loss"]:
        raise ValueError("source-only multiteacher loss differs")
    access = payload.get("access_audit")
    expected_access = {
        "source_rgb_or_features_may_have_been_opened": True,
        "benchmark_query_or_text_opened": False,
        "benchmark_target_rgb_opened": False,
        "benchmark_ground_truth_opened": False,
        "benchmark_labels_or_masks_opened": False,
        "benchmark_metrics_or_predictions_opened": False,
    }
    if access != expected_access:
        raise ValueError("source-only multiteacher access audit differs")
    gates = payload.get("source_gates")
    if not isinstance(gates, Mapping) or set(gates) != {
        "raw_radio_no_regression",
        "gauge_no_regression",
        "dino_sam_no_regression",
        "descriptor_improvement",
        "descriptor_tail_no_regression",
        "semantic_relation_no_regression",
        "scalar_authority_unchanged",
        "benchmark_gate",
    }:
        raise ValueError("source-only multiteacher gate schema differs")
    if gates.get("benchmark_gate") != (
        "closed_until_every_source_gate_passes_then_one_shot_frozen_metric"
    ):
        raise ValueError("source-only multiteacher benchmark gate differs")
    execution = payload.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("gpu_started") is not False
        or execution.get("per_scene_tuning") is not False
        or execution.get("output_no_clobber") is not True
    ):
        raise ValueError("source-only multiteacher execution contract differs")
    implementation = execution.get("implementation")
    if not isinstance(implementation, Mapping) or set(implementation) != {
        "multiteacher",
        "trainer",
        "manifest_validator",
    }:
        raise ValueError("source-only multiteacher implementation binding differs")
    expected_implementation = {
        "multiteacher": Path(__file__).resolve(),
        "trainer": (
            Path(__file__).parents[1]
            / "scripts"
            / "train_canonical_radio_field.py"
        ).resolve(),
        "manifest_validator": (
            Path(__file__).parents[1]
            / "scripts"
            / "validate_source_only_multiteacher_manifest.py"
        ).resolve(),
    }
    for name, expected_path in expected_implementation.items():
        record = _record(
            implementation[name], label=f"Stage-B implementation {name}"
        )
        if Path(record["path"]).resolve() != expected_path:
            raise ValueError("source-only multiteacher implementation path differs")
    return {**payload, **records}


@dataclass(frozen=True)
class PrimitiveDescriptorTeacher:
    scene_id: str
    global_rows: torch.Tensor
    descriptor: torch.Tensor
    valid: torch.Tensor
    retained_view_count: torch.Tensor
    directional_resultant: torch.Tensor
    global_to_local: torch.Tensor
    source: Path
    sha256: str

    def batch(
        self, global_rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = torch.as_tensor(global_rows).long().cpu().reshape(-1)
        if rows.numel() and (
            int(rows.min()) < 0 or int(rows.max()) >= self.global_to_local.numel()
        ):
            raise ValueError("Stage-B primitive descriptor rows are out of range")
        local = self.global_to_local[rows]
        active = local >= 0
        safe = local.clamp_min(0)
        active &= self.valid[safe]
        reliability = torch.zeros(rows.numel(), dtype=torch.float32)
        if bool(active.any()):
            count = self.retained_view_count[safe[active]].float()
            resultant = self.directional_resultant[safe[active]].float()
            evidence = ((count - 1.0) / 3.0).clamp(0.0, 1.0)
            confidence = (evidence * resultant.clamp(0.0, 1.0)).sqrt()
            reliability[active] = 0.5 * (1.0 + confidence)
        return self.descriptor[safe], active, reliability


@dataclass(frozen=True)
class SourceRelationTeacher:
    global_edge_index: torch.Tensor
    teacher_local_edge_index: torch.Tensor
    edge_weight: torch.Tensor

    @property
    def num_edges(self) -> int:
        return int(self.global_edge_index.shape[1])


@dataclass(frozen=True)
class SourceOnlyMultiTeacherBundle:
    manifest: dict[str, Any]
    manifest_source: Path
    manifest_sha256: str
    primitive: PrimitiveDescriptorTeacher
    relation: SourceRelationTeacher
    primitive_state_summary: dict[str, Any]


def _load_primitive_teacher(
    record: Mapping[str, str], *, num_global_rows: int, scene_id: str
) -> PrimitiveDescriptorTeacher:
    from radio_gs.scripts.derive_lerf_teacher_agreement_from_top4 import (
        validate_payload as validate_lerf_query_free_top4_teacher,
    )

    payload, digest, source = load_torch_mapping(
        record["path"],
        expected_sha256=record["sha256"],
        map_location="cpu",
        label="Stage-B primitive descriptor teacher",
    )
    validate_lerf_query_free_top4_teacher(payload)
    rows = torch.as_tensor(payload["global_rows"]).long().cpu().contiguous()
    descriptor = torch.as_tensor(payload["teacher_mean"]).cpu().contiguous()
    valid = torch.as_tensor(payload["teacher_valid"]).bool().cpu().contiguous()
    count = torch.as_tensor(payload["retained_view_count"]).cpu().contiguous()
    resultant = (
        torch.as_tensor(payload["teacher_view_directional_resultant"])
        .float()
        .cpu()
        .contiguous()
    )
    if (
        str(payload.get("scene_id", "")) != str(scene_id)
        or rows.numel() == 0
        or int(rows.min()) < 0
        or int(rows.max()) >= int(num_global_rows)
        or rows.numel() != torch.unique(rows).numel()
        or (rows.numel() > 1 and not bool((rows[1:] > rows[:-1]).all()))
        or descriptor.shape != (rows.numel(), 1536)
        or valid.shape != (rows.numel(),)
        or count.shape != (rows.numel(),)
        or resultant.shape != (rows.numel(),)
        or not bool(torch.isfinite(descriptor.float()).all())
        or not bool(torch.isfinite(resultant).all())
        or not bool(valid.any())
    ):
        raise ValueError("Stage-B primitive descriptor row authority differs")
    lookup = torch.full((int(num_global_rows),), -1, dtype=torch.long)
    lookup[rows] = torch.arange(rows.numel(), dtype=torch.long)
    return PrimitiveDescriptorTeacher(
        scene_id=str(scene_id),
        global_rows=rows,
        descriptor=descriptor,
        valid=valid,
        retained_view_count=count,
        directional_resultant=resultant,
        global_to_local=lookup,
        source=source,
        sha256=digest,
    )


def _evenly_spaced_edge_selection(count: int, cap: int) -> torch.Tensor:
    if int(count) <= 0 or int(cap) <= 0:
        raise ValueError("Stage-B relation edge count/cap must be positive")
    if count <= cap:
        return torch.arange(count, dtype=torch.long)
    if cap == 1:
        return torch.zeros(1, dtype=torch.long)
    # Integer arithmetic fixes both endpoints and avoids float rounding drift.
    positions = torch.arange(cap, dtype=torch.int64)
    return torch.div(positions * (count - 1), cap - 1, rounding_mode="floor")


def _load_relation_teacher(
    record: Mapping[str, str],
    *,
    primitive: PrimitiveDescriptorTeacher,
    num_global_rows: int,
) -> SourceRelationTeacher:
    payload, _digest, _source = load_torch_mapping(
        record["path"],
        expected_sha256=record["sha256"],
        map_location="cpu",
        label="Stage-B source relation graph",
    )
    graph_rows = torch.as_tensor(payload.get("global_rows")).long().cpu()
    edge = torch.as_tensor(payload.get("edge_index")).long().cpu()
    metadata = payload.get("metadata")
    if (
        payload.get("schema_version") != 1
        or graph_rows.ndim != 1
        or graph_rows.numel() == 0
        or graph_rows.numel() != torch.unique(graph_rows).numel()
        or int(graph_rows.min()) < 0
        or int(graph_rows.max()) >= int(num_global_rows)
        or edge.ndim != 2
        or edge.shape[0] != 2
        or edge.shape[1] == 0
        or int(edge.min()) < 0
        or int(edge.max()) >= graph_rows.numel()
        or not isinstance(metadata, Mapping)
    ):
        raise ValueError("Stage-B source relation graph layout differs")
    capability = metadata.get("capability_metadata")
    if not isinstance(capability, Mapping) or any(
        capability.get(name) is not expected
        for name, expected in {
            "query_independent": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        }.items()
    ):
        raise ValueError("Stage-B source relation graph is not source-only")
    global_edge = graph_rows[edge]
    teacher_local = primitive.global_to_local[global_edge]
    keep = (teacher_local >= 0).all(dim=0)
    keep &= primitive.valid[teacher_local.clamp_min(0)].all(dim=0)
    eligible = torch.where(keep)[0]
    if eligible.numel() == 0:
        raise ValueError("Stage-B relation graph has no descriptor-aligned edges")
    selected = eligible[
        _evenly_spaced_edge_selection(
            int(eligible.numel()), SOURCE_RELATION_EDGE_CAP
        )
    ]
    global_edge = global_edge[:, selected].contiguous()
    teacher_local = teacher_local[:, selected].contiguous()
    endpoint_local = teacher_local.reshape(-1)
    endpoint_count = primitive.retained_view_count[endpoint_local].float()
    endpoint_resultant = primitive.directional_resultant[endpoint_local].float()
    endpoint_evidence = ((endpoint_count - 1.0) / 3.0).clamp(0.0, 1.0)
    endpoint_reliability = (
        0.5
        * (1.0 + (endpoint_evidence * endpoint_resultant).clamp_min(0.0).sqrt())
    ).reshape(2, -1)
    edge_weight = endpoint_reliability.prod(dim=0).sqrt().contiguous()
    return SourceRelationTeacher(
        global_edge_index=global_edge,
        teacher_local_edge_index=teacher_local,
        edge_weight=edge_weight,
    )


def load_source_only_multiteacher_bundle(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_xyz: torch.Tensor,
    expected_valid: torch.Tensor,
    expected_factorized_radio_cache_sha256: str,
    expected_field_checkpoint_sha256: str,
    expected_radio_checkpoint_sha256: str,
) -> SourceOnlyMultiTeacherBundle:
    """Load all Stage-B source tensors and prove a common primitive authority."""

    manifest, manifest_sha256, manifest_source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="Stage-B source-only multiteacher manifest",
    )
    manifest = validate_source_only_multiteacher_manifest(manifest)
    if (
        manifest["factorized_radio_cache"]["sha256"]
        != str(expected_factorized_radio_cache_sha256)
        or manifest["field_control"]["sha256"]
        != str(expected_field_checkpoint_sha256)
        or manifest["official_radio_checkpoint"]["sha256"]
        != str(expected_radio_checkpoint_sha256)
    ):
        raise ValueError("Stage-B caller/source lineage differs")
    xyz = torch.as_tensor(expected_xyz).float().cpu().contiguous()
    valid = torch.as_tensor(expected_valid).bool().cpu().contiguous()
    state = load_factorized_primitive_state(
        manifest["factorized_primitive_state"]["path"],
        expected_sha256=manifest["factorized_primitive_state"]["sha256"],
        expected_field_checkpoint_sha256=expected_field_checkpoint_sha256,
        expected_factorized_radio_cache_sha256=(
            expected_factorized_radio_cache_sha256
        ),
        expected_xyz=xyz,
        expected_valid=valid,
    )
    if state.schema != FACTORIZED_PRIMITIVE_STATE_SCHEMA or (
        state.schema_version != FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION
    ):
        raise ValueError("Stage-B requires factorized primitive state v2")
    primitive = _load_primitive_teacher(
        manifest["primitive_descriptor_teacher"],
        num_global_rows=int(xyz.shape[0]),
        scene_id=str(manifest["scene_id"]),
    )
    aligned_valid = primitive.valid & state.valid[primitive.global_rows]
    if not bool(aligned_valid.any()):
        raise ValueError(
            "Stage-B primitive descriptor teacher has no factorized-valid rows"
        )
    # The independent source teacher and exact-marginal factorized field need
    # not cover identical primitive sets.  Their immutable intersection is the
    # only legal distillation cohort; this never expands the field support.
    primitive = replace(primitive, valid=aligned_valid.contiguous())
    relation = _load_relation_teacher(
        manifest["relation_graph"],
        primitive=primitive,
        num_global_rows=int(xyz.shape[0]),
    )
    summary = {
        "schema": state.schema,
        "schema_version": state.schema_version,
        "contract_sha256": state.contract_sha256,
        "valid_rows": int(state.valid.sum()),
        "primitive_descriptor_rows": int(primitive.global_rows.numel()),
        "primitive_descriptor_valid_rows": int(primitive.valid.sum()),
        "visibility_purity_known_rows": int(
            state.visibility_purity_known.sum()
        ),
        "factorized_radio_cache_sha256": str(
            state.metadata["factorized_radio_cache_sha256"]
        ),
        "geometry_fingerprint": dict(state.metadata["geometry_fingerprint"]),
    }
    return SourceOnlyMultiTeacherBundle(
        manifest=manifest,
        manifest_source=manifest_source,
        manifest_sha256=manifest_sha256,
        primitive=primitive,
        relation=relation,
        primitive_state_summary=summary,
    )


class FrozenPrimitiveSiglipProxy(nn.Module):
    """Legacy pointwise projection used by the archived O2 source teacher.

    The official SigLIP summary head was trained for genuine summary tokens,
    not primitive or spatial tokens.  The class remains only to reproduce the
    immutable O2 artifact family; new core-method runs must use a genuine
    region-view summary teacher instead and therefore fail closed unless the
    caller explicitly opts into legacy reproduction.
    """

    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        self.head = head.eval()
        self.head.requires_grad_(False)

    @classmethod
    def from_radio_checkpoint(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        allow_legacy_primitive_proxy: bool = False,
    ) -> "FrozenPrimitiveSiglipProxy":
        if not bool(allow_legacy_primitive_proxy):
            raise ValueError(
                "primitive SigLIP summary projection is legacy-only; use a "
                "genuine source-region summary teacher for core-method runs"
            )
        actual = sha256_file(path)
        if actual != str(expected_sha256):
            raise ValueError("Stage-B official RADIO checkpoint differs")
        return cls(
            SigLIP2SummaryHead.from_radio_checkpoint(
                str(path), expected_sha256=actual
            )
        )

    def forward(self, radio: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(radio)
        if values.ndim != 2 or values.shape[1] != 1280:
            raise ValueError("Stage-B primitive RADIO rows must be [B,1280]")
        return F.normalize(self.head(values[:, None])[:, 0].float(), dim=-1)


def source_direction_only_radio(
    radio: torch.Tensor, *, norm_epsilon: float = 1e-8
) -> torch.Tensor:
    """Preserve the raw RADIO value while removing source-loss gauge gradient.

    Forward values equal the decoded RADIO rows up to floating-point roundoff,
    so the frozen teacher/readout interface is unchanged.  The detached norm
    makes the Jacobian tangent to each RADIO direction: source descriptor and
    relation losses cannot change log amplitude.  The primary factorized loss
    remains the sole radial/gauge authority.
    """

    values = torch.as_tensor(radio)
    if values.ndim != 2 or values.shape[1] != 1280:
        raise ValueError("Stage-B source RADIO rows must be [B,1280]")
    if bool(
        (torch.linalg.vector_norm(values.float(), dim=-1) <= norm_epsilon).any()
    ):
        raise ValueError("Stage-B source RADIO gauge must be positive")
    return gauge_separated_radio(
        values, feature_dim=-1, norm_epsilon=norm_epsilon
    )


def primitive_source_descriptor_loss(
    predicted_radio: torch.Tensor,
    global_rows: torch.Tensor,
    *,
    projector: nn.Module,
    teacher: PrimitiveDescriptorTeacher,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reliability-weighted source descriptor reconstruction on one row batch."""

    rows = torch.as_tensor(global_rows).long().cpu().reshape(-1)
    target, active, weight = teacher.batch(rows)
    if predicted_radio.shape != (rows.numel(), 1280):
        raise ValueError("Stage-B predicted RADIO batch differs")
    if not bool(active.any()):
        zero = predicted_radio.sum() * 0.0
        return zero, {
            "active_rows": torch.tensor(0, device=predicted_radio.device),
            "mean_cosine": torch.tensor(0.0, device=predicted_radio.device),
        }
    device = predicted_radio.device
    predicted = projector(
        source_direction_only_radio(predicted_radio[active.to(device)])
    )
    target_active = F.normalize(
        target[active].to(device).float(), dim=-1, eps=1e-8
    )
    errors = 1.0 - F.cosine_similarity(predicted, target_active, dim=-1)
    weights = weight[active].to(device).float().clamp_min(1e-4)
    loss = (errors * weights).sum() / weights.sum()
    return loss, {
        "active_rows": active.sum().to(device),
        "mean_cosine": (1.0 - errors).mean().detach(),
    }


def source_relation_batch_loss(
    field: nn.Module,
    *,
    projector: nn.Module,
    teacher: PrimitiveDescriptorTeacher,
    relation: SourceRelationTeacher,
    edge_indices: torch.Tensor,
) -> torch.Tensor:
    """Match local teacher cosines without storing a second relation field."""

    selected = torch.as_tensor(edge_indices).long().cpu().reshape(-1)
    if selected.numel() == 0:
        return field.local_codes.sum() * 0.0
    if int(selected.min()) < 0 or int(selected.max()) >= relation.num_edges:
        raise ValueError("Stage-B relation batch is outside its frozen edge axis")
    global_edge = relation.global_edge_index[:, selected]
    unique, inverse = torch.unique(
        global_edge.reshape(-1), sorted=True, return_inverse=True
    )
    device = field.local_codes.device
    predicted = projector(
        source_direction_only_radio(field.radio_features(unique.to(device)))
    )
    local_edge = inverse.reshape_as(global_edge).to(device)
    predicted_relation = (
        predicted[local_edge[0]] * predicted[local_edge[1]]
    ).sum(dim=-1)
    teacher_edge = relation.teacher_local_edge_index[:, selected]
    target_rows, target_inverse = torch.unique(
        teacher_edge.reshape(-1), sorted=True, return_inverse=True
    )
    target = F.normalize(
        teacher.descriptor[target_rows].float(), dim=-1, eps=1e-8
    )
    target_edge = target_inverse.reshape_as(teacher_edge)
    target_relation = (
        target[target_edge[0]] * target[target_edge[1]]
    ).sum(dim=-1).to(device)
    error = F.smooth_l1_loss(
        predicted_relation.float(), target_relation.float(), reduction="none"
    )
    weight = relation.edge_weight[selected].to(device).float().clamp_min(1e-4)
    return (error * weight).sum() / weight.sum()


@torch.no_grad()
def source_descriptor_metrics(
    field: nn.Module,
    *,
    projector: nn.Module,
    teacher: PrimitiveDescriptorTeacher,
    rows: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    """Compute preregistered reconstruction gates on a fixed global row set."""

    requested = torch.as_tensor(rows).long().cpu().reshape(-1)
    if int(batch_size) <= 0:
        raise ValueError("Stage-B metric batch size must be positive")
    parts: list[torch.Tensor] = []
    device = field.local_codes.device
    for start in range(0, requested.numel(), int(batch_size)):
        batch = requested[start : start + int(batch_size)]
        target, active, _weight = teacher.batch(batch)
        if not bool(active.any()):
            continue
        predicted = projector(
            field.radio_features(batch[active].to(device)).float()
        ).cpu()
        parts.append(
            F.cosine_similarity(
                predicted,
                F.normalize(target[active].float(), dim=-1, eps=1e-8),
                dim=-1,
                eps=1e-8,
            )
        )
    if not parts:
        return {
            "source_descriptor_rows": 0,
            "source_descriptor_mean_cosine": 0.0,
            "source_descriptor_p05_cosine": 0.0,
        }
    cosine = torch.cat(parts)
    return {
        "source_descriptor_rows": int(cosine.numel()),
        "source_descriptor_mean_cosine": float(cosine.mean()),
        "source_descriptor_p05_cosine": float(cosine.quantile(0.05)),
    }


@torch.no_grad()
def source_relation_metrics(
    field: nn.Module,
    *,
    projector: nn.Module,
    teacher: PrimitiveDescriptorTeacher,
    relation: SourceRelationTeacher,
    batch_size: int = SOURCE_RELATION_BATCH_SIZE,
) -> dict[str, float]:
    errors: list[torch.Tensor] = []
    device = field.local_codes.device
    for start in range(0, relation.num_edges, int(batch_size)):
        stop = min(start + int(batch_size), relation.num_edges)
        global_edge = relation.global_edge_index[:, start:stop]
        unique, inverse = torch.unique(
            global_edge.reshape(-1), sorted=True, return_inverse=True
        )
        predicted = projector(field.radio_features(unique.to(device))).cpu()
        local_edge = inverse.reshape_as(global_edge)
        predicted_relation = (
            predicted[local_edge[0]] * predicted[local_edge[1]]
        ).sum(dim=-1)
        teacher_edge = relation.teacher_local_edge_index[:, start:stop]
        target_rows, target_inverse = torch.unique(
            teacher_edge.reshape(-1), sorted=True, return_inverse=True
        )
        target = F.normalize(
            teacher.descriptor[target_rows].float(), dim=-1, eps=1e-8
        )
        target_edge = target_inverse.reshape_as(teacher_edge)
        target_relation = (
            target[target_edge[0]] * target[target_edge[1]]
        ).sum(dim=-1)
        errors.append((predicted_relation - target_relation).abs())
    error = torch.cat(errors)
    return {
        "source_relation_edges": int(error.numel()),
        "source_relation_mean_abs_error": float(error.mean()),
        "source_relation_p95_abs_error": float(error.quantile(0.95)),
    }


def evaluate_source_only_gates(
    *,
    manifest: Mapping[str, Any],
    control_source_metrics: Mapping[str, float],
    candidate_primary_metrics: Mapping[str, float],
    candidate_capability_metrics: Mapping[str, float],
    candidate_source_metrics: Mapping[str, float],
    primitive_state_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate only preregistered source gates; never consume a benchmark metric."""

    gates = manifest["source_gates"]

    def number(values: Mapping[str, Any], name: str) -> float:
        result = float(values[name])
        if not torch.isfinite(torch.tensor(result)):
            raise ValueError(f"Stage-B gate metric is non-finite: {name}")
        return result

    raw = gates["raw_radio_no_regression"]
    raw_pass = (
        number(candidate_primary_metrics, "mean_cosine")
        >= float(raw["candidate_mean_cosine_min"])
        and number(candidate_primary_metrics, "p05_cosine")
        >= float(raw["candidate_p05_cosine_min"])
    )
    gauge = gates["gauge_no_regression"]
    gauge_pass = (
        number(candidate_primary_metrics, "mean_abs_log_amplitude_error")
        <= float(gauge["candidate_mean_abs_log_amplitude_error_max"])
        and number(candidate_primary_metrics, "p95_abs_log_amplitude_error")
        <= float(gauge["candidate_p95_abs_log_amplitude_error_max"])
    )
    capability = gates["dino_sam_no_regression"]
    capability_pass = True
    for space in ("dino_v3", "sam3"):
        capability_pass &= number(
            candidate_capability_metrics, f"{space}_target_mean_cosine"
        ) >= float(capability[f"{space}_control_mean_cosine"]) - float(
            capability["maximum_mean_cosine_drop"]
        )
        capability_pass &= number(
            candidate_capability_metrics, f"{space}_target_p05_cosine"
        ) >= float(capability[f"{space}_control_p05_cosine"]) - float(
            capability["maximum_p05_cosine_drop"]
        )
    descriptor = gates["descriptor_improvement"]
    descriptor_delta = number(
        candidate_source_metrics, "source_descriptor_mean_cosine"
    ) - number(control_source_metrics, "source_descriptor_mean_cosine")
    descriptor_pass = descriptor_delta >= float(
        descriptor["candidate_mean_cosine_minus_hash_bound_control_min"]
    )
    tail = gates["descriptor_tail_no_regression"]
    descriptor_tail_delta = number(
        candidate_source_metrics, "source_descriptor_p05_cosine"
    ) - number(control_source_metrics, "source_descriptor_p05_cosine")
    descriptor_tail_pass = descriptor_tail_delta >= float(
        tail["candidate_p05_cosine_minus_hash_bound_control_min"]
    )
    relation = gates["semantic_relation_no_regression"]
    relation_mean_delta = number(
        candidate_source_metrics, "source_relation_mean_abs_error"
    ) - number(control_source_metrics, "source_relation_mean_abs_error")
    relation_p95_delta = number(
        candidate_source_metrics, "source_relation_p95_abs_error"
    ) - number(control_source_metrics, "source_relation_p95_abs_error")
    relation_pass = relation_mean_delta <= float(
        relation["candidate_mean_abs_error_minus_hash_bound_control_max"]
    ) and relation_p95_delta <= float(
        relation["candidate_p95_abs_error_minus_hash_bound_control_max"]
    )
    scalar = gates["scalar_authority_unchanged"]
    scalar_pass = (
        manifest["factorized_primitive_state"]["sha256"]
        == scalar["required_factorized_primitive_state_sha256"]
        and int(primitive_state_summary["valid_rows"])
        == int(scalar["required_valid_rows"])
    )
    support_pass = (
        int(candidate_source_metrics["source_descriptor_rows"])
        == int(descriptor["required_hash_bound_support_intersection_rows"])
        and int(candidate_source_metrics["source_relation_edges"])
        == int(relation["required_deterministic_relation_edges"])
    )
    results = {
        "raw_radio_no_regression": bool(raw_pass),
        "gauge_no_regression": bool(gauge_pass),
        "dino_sam_no_regression": bool(capability_pass),
        "descriptor_improvement": bool(descriptor_pass),
        "descriptor_tail_no_regression": bool(descriptor_tail_pass),
        "semantic_relation_no_regression": bool(relation_pass),
        "scalar_authority_unchanged": bool(scalar_pass),
        "source_support_unchanged": bool(support_pass),
    }
    all_passed = all(results.values())
    return {
        "results": results,
        "deltas": {
            "source_descriptor_mean_cosine": descriptor_delta,
            "source_descriptor_p05_cosine": descriptor_tail_delta,
            "source_relation_mean_abs_error": relation_mean_delta,
            "source_relation_p95_abs_error": relation_p95_delta,
        },
        "all_source_gates_passed": all_passed,
        "benchmark_gate_opened": all_passed,
        "benchmark_metric_read": False,
    }


def validate_scannet_region_view_teacher_record(
    record: Mapping[str, str]
) -> dict[str, Any]:
    """Expose the clean ScanNet teacher family under the shared Stage-B contract.

    The returned mapping is intentionally not converted to a primitive target;
    a region-view teacher must be consumed after the frozen region aggregation
    operator.  This function makes the cross-dataset contract executable while
    preventing the common semantic-level mismatch.
    """

    from radio_gs.scripts.materialize_full_scalar_clean_training_shard import (
        validate_teacher_observation_authority,
    )

    frozen = _record(record, label="Stage-B ScanNet region-view teacher")
    payload, digest, source = load_torch_mapping(
        frozen["path"],
        expected_sha256=frozen["sha256"],
        map_location="cpu",
        label="Stage-B ScanNet region-view teacher",
    )
    validated = validate_teacher_observation_authority(payload)
    return {
        "scene_id": str(validated["scene_id"]),
        "teacher_level": REGION_VIEW_TEACHER_KIND,
        "sampled_regions": int(validated["canonical_region_indices"].numel()),
        "region_view_pairs": int(validated["pair_descriptors"].shape[0]),
        "descriptor_dim": int(validated["pair_descriptors"].shape[1]),
        "path": str(source),
        "sha256": digest,
    }
