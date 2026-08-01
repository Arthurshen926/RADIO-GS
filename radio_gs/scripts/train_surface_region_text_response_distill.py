#!/usr/bin/env python3
"""Train a SurfaceRegion readout with target-blind fit-text responses.

The inference architecture and checkpoint schema are identical to
``train_surface_region_summary_readout.py``.  Training adds an independent
SmoothL1 loss and a scene-wise profile/ranking loss over responses to the
frozen *fit* vocabulary.  Their coefficients are not command-line
hyperparameters: each seed receives one immutable CPU calibration measured at
that seed's exact frozen Surface attention checkpoint.  Each response branch
is budgeted to 0.25 of the complete Surface-objective gradient norm; by the
triangle inequality their combined gradient is bounded by 0.5 of Surface.
Epoch
selection is target-blind and constrained: summary-token, mean-descriptor, and
all-view descriptor validation cosine must each remain within 0.002 of that
seed's Surface control before fit-query response metrics may select an epoch.
The frozen control is epoch 0, so selection cannot silently trade away the
recovered Surface feature field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.surface_region_summary import SurfaceRegionSummaryReadoutV2
from radio_gs.losses.direct_point_query_logit_distill_loss import (
    compute_independent_normalized_cosine_response_smooth_l1_loss,
    compute_scene_wise_text_response_profile_ranking_loss,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts import build_target_blind_siglip2_embedding_artifact as bank_builder
from radio_gs.scripts.train_surface_region_summary_readout import (
    _load,
    _paths,
    _seed_training,
    _sha256_file,
    _targets,
    inject_tangent_direction_noise,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_sha_bound_project_checkpoint_mapping,
    load_surface_region_summary_readout_v2,
    load_torch_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


CALIBRATION_SCHEMA_VERSION = 2
CALIBRATION_ARTIFACT_TYPE = "surface_text_response_gradient_calibration"
CALIBRATION_ALGORITHM_VERSION = (
    "per-seed-surface-warmstart-dual-response-gradient-budget-v2"
)
CALIBRATION_SCENE_COUNT = 4
CALIBRATION_SCENE_SELECTION = "lexicographically_first_complete_train_scenes_v1"
SHARED_TRAINING_SEEDS = (0, 1, 2)
GRADIENT_NORM_EPSILON = 1e-12
RESPONSE_BRANCH_GRADIENT_RATIO = 0.25
TOTAL_RESPONSE_GRADIENT_RATIO_UPPER_BOUND = 0.5
SCENE_PROFILE_WEIGHT = 1.0
SCENE_RANKING_WEIGHT = 1.0
SCENE_RANKING_TEMPERATURE = 0.1
SCENE_TIE_TOLERANCE = 1e-6
MAX_COMPLETE_SCENE_BATCH_ROWS = 64
FIT_SPLIT = "fit"
DISTILL_RUN_MANIFEST_SCHEMA_VERSION = 3
DISTILL_RUN_MANIFEST_ARTIFACT_TYPE = "surface_region_text_response_distill_run"
RESPONSE_EPOCH_SELECTION = (
    "surface_control_feasible_0p002_then_fit_support_response_relation_surface_v2"
)
SURFACE_CONTROL_NONINFERIORITY_TOLERANCE = 0.002
SURFACE_CONTROL_METRICS = (
    "summary_token_cosine",
    "mean_descriptor_cosine",
    "all_view_descriptor_cosine",
)
FROZEN_FORMAL_FIT_BANK_BUILDER = {
    "path": "/root/RADIO-GS/radio_gs/scripts/build_target_blind_siglip2_embedding_artifact.py",
    "sha256": "e8e815b5f15796c21205788769a48d1bef95e21b9eac4c2777cf6754b424d136",
}
FROZEN_FORMAL_FIT_BANK = {
    "artifact_path": "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260731/target_blind_siglip2_text_bank_v1/target_blind_siglip2_fit_embeddings.pt",
    "artifact_sha256": "d67b632e8ccce13d84479379e8f674f5ec31b729acf02a79ce6c4bb2a4f170f4",
    "manifest_path": "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260731/target_blind_siglip2_text_bank_v1/target_blind_siglip2_fit_embeddings.manifest.json",
    "manifest_sha256": "5d6c7fa167b63a4052f9be57389a2f5f0e469c75391916ef0b7fe562cce7e3f0",
}

_PAYLOAD_KEYS = {
    "schema_version",
    "artifact_type",
    "algorithm_version",
    "benchmark_vocabulary_opened",
    "uses_benchmark_vocabulary_for_construction",
    "split",
    "split_synset_tab_query_lf_sha256",
    "prompt_templates",
    "text_canonicalization",
    "records",
    "queries",
    "synsets",
    "ordered_records_sha256",
    "vocabulary_path",
    "vocabulary_sha256",
    "vocabulary_manifest_path",
    "vocabulary_manifest_sha256",
    "embeddings",
    "embedding_semantic_sha256",
    "embedding_tensor_sha256",
    "text_encoder",
}
_SIDECAR_KEYS = {
    "schema_version",
    "artifact_type",
    "algorithm_version",
    "benchmark_vocabulary_opened",
    "uses_benchmark_vocabulary_for_construction",
    "split",
    "split_synset_tab_query_lf_sha256",
    "prompt_templates",
    "text_canonicalization",
    "records",
    "queries",
    "synsets",
    "ordered_records_sha256",
    "vocabulary",
    "text_encoder",
    "embedding",
    "artifact",
    "builder",
}


def _read_json(path: Path) -> dict[str, Any]:
    value, _, _ = load_json_object(path, label="text-response JSON artifact")
    return value


def _torch_load_mapping(path: Path) -> Mapping[str, Any]:
    value, _, _ = load_torch_mapping(
        path,
        map_location="cpu",
        label="text-response torch artifact",
    )
    return value


def _resolve_file(raw: object, *, relative_to: Path, label: str) -> Path:
    value = Path(str(raw))
    path = (value if value.is_absolute() else relative_to / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"bound {label} does not exist: {path}")
    return path


def _semantic_embedding_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().to(torch.float32).contiguous()
    array = tensor.numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def load_fit_text_embedding_bank(path: Path, sidecar_path: Path) -> dict[str, Any]:
    """Load the exact builder-produced fit split and fail closed on drift."""

    path = Path(path).resolve()
    sidecar_path = Path(sidecar_path).resolve()
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("fit text embedding artifact and sidecar must exist")
    payload = _torch_load_mapping(path)
    sidecar = _read_json(sidecar_path)
    if (
        set(payload) != _PAYLOAD_KEYS
        or payload.get("schema_version") != bank_builder.SCHEMA_VERSION
        or payload.get("artifact_type") != bank_builder.ARTIFACT_TYPE
        or payload.get("algorithm_version") != bank_builder.ALGORITHM_VERSION
        or set(sidecar) != _SIDECAR_KEYS
        or sidecar.get("schema_version") != bank_builder.SCHEMA_VERSION
        or sidecar.get("artifact_type") != bank_builder.MANIFEST_ARTIFACT_TYPE
        or sidecar.get("algorithm_version") != bank_builder.ALGORITHM_VERSION
    ):
        raise ValueError("invalid target-blind split-v1 text embedding cache schema")
    for record in (payload, sidecar):
        if (
            record.get("benchmark_vocabulary_opened") is not False
            or record.get("uses_benchmark_vocabulary_for_construction") is not False
        ):
            raise ValueError("fit text bank must certify that benchmark vocabulary was not opened")
        if record.get("split") != FIT_SPLIT:
            raise ValueError("text-response training requires the target-blind fit split")
        if (
            record.get("prompt_templates") != ["{query}"]
            or record.get("text_canonicalization")
            != bank_builder.TEXT_CANONICALIZATION
        ):
            raise ValueError("fit text bank violates the frozen text policy")

    vocabulary_path = _resolve_file(
        payload.get("vocabulary_path"), relative_to=path.parent, label="vocabulary"
    )
    vocabulary_manifest_path = _resolve_file(
        payload.get("vocabulary_manifest_path"),
        relative_to=path.parent,
        label="vocabulary manifest",
    )
    vocabulary_contract = bank_builder._validate_vocabulary(
        vocabulary_path,
        vocabulary_manifest_path,
        FIT_SPLIT,
    )
    vocabulary_manifest = _read_json(vocabulary_manifest_path)
    split_hashes = vocabulary_manifest.get("split_synset_tab_query_lf_sha256")
    if not isinstance(split_hashes, Mapping) or set(split_hashes) != {
        "fit",
        "dev",
        "audit",
    }:
        raise ValueError("fit vocabulary manifest has an incomplete split hash index")
    expected_records = vocabulary_contract["records"]
    expected_queries = [record["query"] for record in expected_records]
    expected_synsets = [record["synset"] for record in expected_records]
    if (
        payload.get("records") != expected_records
        or payload.get("queries") != expected_queries
        or payload.get("synsets") != expected_synsets
        or payload.get("ordered_records_sha256")
        != vocabulary_contract["ordered_records_sha256"]
        or payload.get("split_synset_tab_query_lf_sha256")
        != vocabulary_contract["split_sha256"]
        or payload.get("vocabulary_sha256")
        != vocabulary_contract["vocabulary_sha256"]
        or payload.get("vocabulary_manifest_sha256")
        != vocabulary_contract["vocabulary_manifest_sha256"]
    ):
        raise ValueError("fit embedding rows differ from the bound canonical vocabulary")

    text_encoder = payload.get("text_encoder")
    if not isinstance(text_encoder, Mapping):
        raise ValueError("fit text bank lacks text_encoder provenance")
    snapshot = Path(str(text_encoder.get("snapshot_path", ""))).resolve()
    expected_encoder = bank_builder._validate_snapshot(
        snapshot,
        model_id=bank_builder.MODEL_ID,
        revision=bank_builder.MODEL_REVISION,
    )
    if dict(text_encoder) != expected_encoder:
        raise ValueError("fit text encoder differs from the current frozen snapshot")

    embeddings = torch.as_tensor(payload.get("embeddings"))
    expected_shape = (len(expected_records), bank_builder.OUTPUT_DIMENSION)
    if (
        embeddings.device.type != "cpu"
        or embeddings.dtype != torch.float32
        or tuple(embeddings.shape) != expected_shape
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("fit embeddings must be finite CPU float32 [Q,1536]")
    norms = torch.linalg.vector_norm(embeddings, dim=-1)
    if not bool(
        torch.allclose(norms, torch.ones_like(norms), atol=5e-5, rtol=5e-5)
    ):
        raise ValueError("fit embedding rows must be L2-normalized")
    semantic_sha = _semantic_embedding_sha256(embeddings)
    typed_sha = tensor_sha256(embeddings)
    if (
        payload.get("embedding_semantic_sha256") != semantic_sha
        or payload.get("embedding_tensor_sha256") != typed_sha
    ):
        raise ValueError("fit embedding tensor hash mismatch")

    common = (
        "split",
        "split_synset_tab_query_lf_sha256",
        "prompt_templates",
        "text_canonicalization",
        "records",
        "queries",
        "synsets",
        "ordered_records_sha256",
    )
    if any(sidecar.get(key) != payload.get(key) for key in common):
        raise ValueError("fit embedding sidecar differs from its tensor payload")
    if sidecar.get("text_encoder") != expected_encoder:
        raise ValueError("fit embedding sidecar text_encoder mismatch")
    expected_vocabulary = {
        "path": str(vocabulary_path),
        "sha256": vocabulary_contract["vocabulary_sha256"],
        "manifest_path": str(vocabulary_manifest_path),
        "manifest_sha256": vocabulary_contract["vocabulary_manifest_sha256"],
    }
    if sidecar.get("vocabulary") != expected_vocabulary:
        raise ValueError("fit embedding sidecar vocabulary binding mismatch")
    expected_embedding = {
        "shape": list(expected_shape),
        "dtype": "float32",
        "byte_order": "little_endian",
        "normalization": "l2",
        "semantic_sha256": semantic_sha,
        "tensor_sha256": typed_sha,
    }
    if sidecar.get("embedding") != expected_embedding:
        raise ValueError("fit embedding sidecar tensor binding mismatch")

    artifact_record = sidecar.get("artifact")
    if not isinstance(artifact_record, Mapping) or set(artifact_record) != {
        "path",
        "sha256",
    }:
        raise ValueError("fit embedding sidecar lacks artifact provenance")
    bound_artifact = _resolve_file(
        artifact_record.get("path"),
        relative_to=sidecar_path.parent,
        label="fit embedding artifact",
    )
    file_sha = _sha256_file(path)
    if bound_artifact != path or artifact_record.get("sha256") != file_sha:
        raise ValueError("fit embedding sidecar artifact binding mismatch")

    builder_record = sidecar.get("builder")
    if not isinstance(builder_record, Mapping) or set(builder_record) != {
        "path",
        "sha256",
    }:
        raise ValueError("fit embedding sidecar lacks builder provenance")
    current_builder = Path(bank_builder.__file__).resolve()
    current_builder_record = {
        "path": str(current_builder),
        "sha256": _sha256_file(current_builder),
    }
    sidecar_sha = _sha256_file(sidecar_path)
    exact_frozen_formal_bank = (
        dict(builder_record) == FROZEN_FORMAL_FIT_BANK_BUILDER
        and str(path) == FROZEN_FORMAL_FIT_BANK["artifact_path"]
        and file_sha == FROZEN_FORMAL_FIT_BANK["artifact_sha256"]
        and str(sidecar_path) == FROZEN_FORMAL_FIT_BANK["manifest_path"]
        and sidecar_sha == FROZEN_FORMAL_FIT_BANK["manifest_sha256"]
    )
    if dict(builder_record) != current_builder_record and not exact_frozen_formal_bank:
        raise ValueError("fit embedding sidecar builder binding mismatch")

    return {
        "path": path,
        "file_sha256": file_sha,
        "manifest_path": sidecar_path,
        "manifest_sha256": sidecar_sha,
        "embeddings": embeddings.contiguous(),
        "query_count": len(expected_records),
        "split_sha256": vocabulary_contract["split_sha256"],
        "ordered_records_sha256": vocabulary_contract["ordered_records_sha256"],
        "vocabulary_sha256": vocabulary_contract["vocabulary_sha256"],
        "vocabulary_manifest_sha256": vocabulary_contract[
            "vocabulary_manifest_sha256"
        ],
        "embedding_semantic_sha256": semantic_sha,
        "embedding_tensor_sha256": typed_sha,
        "text_encoder": expected_encoder,
    }


def _descriptor_loss(
    projected: torch.Tensor,
    all_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
) -> torch.Tensor:
    if projected.ndim != 2 or all_descriptors.ndim != 3:
        raise ValueError("projected/all_descriptors must be [B,D]/[B,V,D]")
    if (
        all_descriptors.shape[0] != projected.shape[0]
        or all_descriptors.shape[2] != projected.shape[1]
        or teacher_mask.shape != all_descriptors.shape[:2]
    ):
        raise ValueError("descriptor targets and mask are misaligned")
    mask = teacher_mask.bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every row requires at least one teacher descriptor")
    cosine = torch.einsum("bd,bvd->bv", projected, all_descriptors)
    return (1.0 - cosine)[mask].mean()


def compute_query_free_response_selection_metrics(
    student_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    text_bank: torch.Tensor,
    *,
    scene_ids: Sequence[str],
) -> dict[str, float]:
    """Measure text unary preservation along the within-scene support axis.

    Descriptor cosine is a row-local metric.  A text unary instead ranks
    regions for one query, so small row-local errors can still swap its peak.
    This query-free metric uses only the target-blind fit bank and groups rows
    by ScanNet scene before comparing teacher/student argmax support.
    """

    student = F.normalize(torch.as_tensor(student_descriptors).float(), dim=-1)
    teacher = F.normalize(
        torch.as_tensor(teacher_descriptors, device=student.device).float(),
        dim=-1,
    )
    text = F.normalize(
        torch.as_tensor(text_bank, device=student.device).float(),
        dim=-1,
    )
    if (
        student.ndim != 2
        or teacher.shape != student.shape
        or text.ndim != 2
        or text.shape[1] != student.shape[1]
        or len(scene_ids) != student.shape[0]
        or student.shape[0] < 2
        or text.shape[0] == 0
        or not bool(torch.isfinite(student).all())
        or not bool(torch.isfinite(teacher).all())
        or not bool(torch.isfinite(text).all())
    ):
        raise ValueError("query-free response selection inputs are misaligned")
    normalized_scenes = [str(value) for value in scene_ids]
    if any(not value for value in normalized_scenes):
        raise ValueError("query-free response selection requires scene IDs")

    student_response = student @ text.T
    teacher_response = teacher @ text.T
    support_agreements: list[torch.Tensor] = []
    valid_ratios: list[torch.Tensor] = []
    relation_errors: list[torch.Tensor] = []
    for scene in sorted(set(normalized_scenes)):
        rows = torch.tensor(
            [index for index, value in enumerate(normalized_scenes) if value == scene],
            device=student.device,
            dtype=torch.long,
        )
        if len(rows) < 2:
            continue
        local_student_response = student_response[rows]
        local_teacher_response = teacher_response[rows]
        top_two = local_teacher_response.topk(2, dim=0).values
        valid = (top_two[0] - top_two[1]).abs() > 1e-8
        valid_ratios.append(valid.float().mean())
        if bool(valid.any()):
            support_agreements.append(
                (
                    local_student_response[:, valid].argmax(dim=0)
                    == local_teacher_response[:, valid].argmax(dim=0)
                ).float().mean()
            )
        local_student = student[rows]
        local_teacher = teacher[rows]
        relation_errors.append(
            F.smooth_l1_loss(
                local_student @ local_student.T,
                local_teacher @ local_teacher.T,
            )
        )
    if not support_agreements or not relation_errors:
        raise ValueError(
            "query-free response selection needs a multi-region scene with "
            "non-tied teacher support"
        )
    return {
        "text_support_top1_agreement": float(
            torch.stack(support_agreements).mean().detach().cpu()
        ),
        "text_support_valid_query_ratio": float(
            torch.stack(valid_ratios).mean().detach().cpu()
        ),
        "text_response_smooth_l1": float(
            F.smooth_l1_loss(student_response, teacher_response).detach().cpu()
        ),
        "descriptor_relation_smooth_l1": float(
            torch.stack(relation_errors).mean().detach().cpu()
        ),
    }


def finalize_response_primary_epoch_selection(
    history: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int, float]:
    """Select fit-text response quality inside the Surface-control feasible set.

    The first row is the frozen control at epoch 0.  A later epoch is feasible
    only when *each* Surface validation component is at least control minus
    :data:`SURFACE_CONTROL_NONINFERIORITY_TOLERANCE`.  Within that feasible set
    the deterministic order is: support top-1 (higher), fit-response SmoothL1
    (lower), descriptor-relation SmoothL1 (lower), and legacy Surface score
    (higher).  The returned scalar remains the selected Surface score for
    compatibility with descriptor-fidelity reports.
    """

    if not history:
        raise ValueError("response-aware epoch selection requires history")
    rows = [dict(record) for record in history]
    epochs = [record.get("epoch") for record in rows]
    if epochs != list(range(len(rows))):
        raise ValueError(
            "response-aware history must start at control epoch 0 and be contiguous"
        )
    control = rows[0]
    control_surface: dict[str, float] = {}
    for field in SURFACE_CONTROL_METRICS:
        value = control.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"Surface control {field} must be finite")
        control_surface[field] = float(value)

    ranks: list[tuple[float, float, float, float] | None] = []
    for record in rows:
        finite_fields = (
            "surface_selection_score",
            "text_support_top1_agreement",
            "text_response_smooth_l1",
            "descriptor_relation_smooth_l1",
            *SURFACE_CONTROL_METRICS,
        )
        values: dict[str, float] = {}
        for field in finite_fields:
            raw = record.get(field)
            if (
                not isinstance(raw, (int, float))
                or isinstance(raw, bool)
                or not math.isfinite(float(raw))
            ):
                raise ValueError(
                    f"response-aware epoch selection {field} must be finite"
                )
            values[field] = float(raw)
        deltas = {
            field: values[field] - control_surface[field]
            for field in SURFACE_CONTROL_METRICS
        }
        feasible = all(
            delta >= -SURFACE_CONTROL_NONINFERIORITY_TOLERANCE
            or math.isclose(
                delta,
                -SURFACE_CONTROL_NONINFERIORITY_TOLERANCE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for delta in deltas.values()
        )
        record["surface_control_deltas"] = deltas
        record["surface_control_feasible"] = feasible
        record["surface_control_tolerance"] = (
            SURFACE_CONTROL_NONINFERIORITY_TOLERANCE
        )
        ranks.append(
            (
                values["text_support_top1_agreement"],
                -values["text_response_smooth_l1"],
                -values["descriptor_relation_smooth_l1"],
                values["surface_selection_score"],
            )
            if feasible
            else None
        )

    # Epoch 0 compares with itself, so absence of a feasible state indicates
    # corrupt/non-finite control metadata rather than a legitimate outcome.
    eligible = [index for index, rank in enumerate(ranks) if rank is not None]
    if not eligible:
        raise RuntimeError("Surface-control feasible set unexpectedly is empty")
    best_index = max(eligible, key=lambda index: ranks[index])
    for index, record in enumerate(rows):
        record["selection_score"] = (
            float(record["surface_selection_score"])
            if index == best_index
            else -1.0
        )
    return (
        rows,
        int(rows[best_index]["epoch"]),
        float(rows[best_index]["surface_selection_score"]),
    )


@torch.no_grad()
def _evaluate_response_aware(
    model: torch.nn.Module,
    head: torch.nn.Module,
    data: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    text_bank: torch.Tensor,
) -> tuple[dict[str, float], dict[str, float]]:
    scene_ids = data.get("scene_ids")
    row_count = len(data["radio_features"])
    if not isinstance(scene_ids, list) or len(scene_ids) != row_count:
        raise ValueError(
            "text-response selection requires exact cache row-to-scene bindings"
        )
    token_cos: list[float] = []
    descriptor_cos: list[float] = []
    multiview_cos: list[float] = []
    students: list[torch.Tensor] = []
    teachers: list[torch.Tensor] = []
    for start in range(0, row_count, int(batch_size)):
        rows = torch.arange(start, min(start + int(batch_size), row_count))
        token, descriptor, all_descriptors, teacher_mask = _targets(data, rows)
        predicted = model(
            data["radio_features"][rows].to(device),
            data["geometry"][rows].to(device),
            anchor_index=data["anchor_index"][rows].to(device),
            token_mask=data["token_mask"][rows].to(device),
            reliability=data["reliability"][rows].to(device),
        )
        projected = F.normalize(
            head(predicted[:, None])[:, 0].float(),
            dim=-1,
            eps=1e-8,
        )
        projected_cpu = projected.cpu()
        token_cos.extend(
            F.cosine_similarity(predicted.cpu(), token, dim=-1).tolist()
        )
        descriptor_cos.extend(
            F.cosine_similarity(projected_cpu, descriptor, dim=-1).tolist()
        )
        pair = torch.einsum("bd,bvd->bv", projected_cpu, all_descriptors)
        multiview_cos.extend(pair[teacher_mask].tolist())
        students.append(projected_cpu)
        teachers.append(descriptor)
    surface = {
        "summary_token_cosine": sum(token_cos) / len(token_cos),
        "mean_descriptor_cosine": sum(descriptor_cos) / len(descriptor_cos),
        "all_view_descriptor_cosine": sum(multiview_cos) / len(multiview_cos),
    }
    response = compute_query_free_response_selection_metrics(
        torch.cat(students),
        torch.cat(teachers),
        text_bank.detach().cpu(),
        scene_ids=scene_ids,
    )
    return surface, response


def compute_training_losses(
    predicted_token: torch.Tensor,
    projected: torch.Tensor,
    target_token: torch.Tensor,
    target_descriptor: torch.Tensor,
    all_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    fit_text_bank: torch.Tensor,
    scene_ids: Sequence[str],
    *,
    token_weight: float,
    relation_weight: float,
    independent_response_lambda: float,
    scene_response_lambda: float,
    scene_profile_weight: float = SCENE_PROFILE_WEIGHT,
    scene_ranking_weight: float = SCENE_RANKING_WEIGHT,
    scene_ranking_temperature: float = SCENE_RANKING_TEMPERATURE,
) -> dict[str, torch.Tensor]:
    """Compute Surface plus independently budgeted response branches."""

    for label, value in (
        ("independent_response_lambda", independent_response_lambda),
        ("scene_response_lambda", scene_response_lambda),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{label} must be the positive calibrated value")
    token_loss = (
        1.0 - F.cosine_similarity(predicted_token, target_token, dim=-1)
    ).mean()
    descriptor_loss = _descriptor_loss(projected, all_descriptors, teacher_mask)
    teacher_rel = target_descriptor @ target_descriptor.T
    predicted_rel = projected @ projected.T
    relation_loss = F.smooth_l1_loss(predicted_rel, teacher_rel)
    independent_response_loss = (
        compute_independent_normalized_cosine_response_smooth_l1_loss(
            projected,
            target_descriptor,
            fit_text_bank,
        )
    )
    scene_response_loss, scene_stats = (
        compute_scene_wise_text_response_profile_ranking_loss(
            projected,
            target_descriptor,
            fit_text_bank,
            scene_ids,
            profile_weight=float(scene_profile_weight),
            ranking_weight=float(scene_ranking_weight),
            ranking_temperature=float(scene_ranking_temperature),
            tie_tolerance=SCENE_TIE_TOLERANCE,
        )
    )
    total = (
        float(token_weight) * token_loss
        + descriptor_loss
        + float(relation_weight) * relation_loss
        + float(independent_response_lambda) * independent_response_loss
        + float(scene_response_lambda) * scene_response_loss
    )
    return {
        "total": total,
        "token": token_loss,
        "descriptor": descriptor_loss,
        "relation": relation_loss,
        "independent_response": independent_response_loss,
        "scene_response": scene_response_loss,
        "scene_profile": scene_stats["profile_loss"],
        "scene_ranking": scene_stats["ranking_loss"],
    }


def _validated_scene_rows(
    scene_ids: object,
    *,
    row_count: int,
) -> dict[str, list[int]]:
    if not isinstance(scene_ids, list) or len(scene_ids) != int(row_count):
        raise ValueError("training caches lack exact row-to-scene identities")
    grouped: dict[str, list[int]] = {}
    for row, raw_scene in enumerate(scene_ids):
        if not isinstance(raw_scene, str) or not raw_scene:
            raise ValueError("training cache scene identities must be non-empty strings")
        grouped.setdefault(raw_scene, []).append(row)
    if not grouped or any(len(rows) < 2 for rows in grouped.values()):
        raise ValueError("every training scene must contain at least two complete rows")
    return grouped


def fixed_calibration_scene_batch(
    scene_ids: object,
    *,
    row_count: int,
) -> tuple[list[str], torch.Tensor]:
    """Select all rows of the declared deterministic four-scene batch."""

    grouped = _validated_scene_rows(scene_ids, row_count=row_count)
    scenes = sorted(grouped)[:CALIBRATION_SCENE_COUNT]
    if len(scenes) != CALIBRATION_SCENE_COUNT:
        raise ValueError("calibration requires four complete training scenes")
    rows = [row for scene in scenes for row in grouped[scene]]
    return scenes, torch.tensor(rows, dtype=torch.long)


def complete_scene_batches(
    scene_ids: object,
    *,
    row_count: int,
    target_batch_rows: int,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    """Shuffle scene groups while never splitting a scene across batches."""

    if int(target_batch_rows) <= 0:
        raise ValueError("target_batch_rows must be positive")
    grouped = _validated_scene_rows(scene_ids, row_count=row_count)
    scenes = sorted(grouped)
    order = torch.randperm(len(scenes), generator=generator).tolist()
    batches: list[torch.Tensor] = []
    pending: list[int] = []
    for index in order:
        pending.extend(grouped[scenes[index]])
        if len(pending) >= int(target_batch_rows):
            batches.append(torch.tensor(pending, dtype=torch.long))
            pending = []
    if pending:
        batches.append(torch.tensor(pending, dtype=torch.long))
    if any(len(batch) > MAX_COMPLETE_SCENE_BATCH_ROWS for batch in batches):
        raise ValueError(
            "complete-scene batch exceeds the fail-closed row-memory limit"
        )
    flattened = [int(row) for batch in batches for row in batch.tolist()]
    if sorted(flattened) != list(range(int(row_count))):
        raise RuntimeError("complete-scene batching did not cover every row exactly once")
    return batches


def _surface_and_response_losses(
    predicted_token: torch.Tensor,
    projected: torch.Tensor,
    target_token: torch.Tensor,
    target_descriptor: torch.Tensor,
    all_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    fit_text_bank: torch.Tensor,
    scene_ids: Sequence[str],
    *,
    token_weight: float,
    relation_weight: float,
    scene_profile_weight: float,
    scene_ranking_weight: float,
    scene_ranking_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    token_loss = (
        1.0 - F.cosine_similarity(predicted_token, target_token, dim=-1)
    ).mean()
    descriptor_loss = _descriptor_loss(projected, all_descriptors, teacher_mask)
    relation_loss = F.smooth_l1_loss(
        projected @ projected.T,
        target_descriptor @ target_descriptor.T,
    )
    surface_loss = (
        float(token_weight) * token_loss
        + descriptor_loss
        + float(relation_weight) * relation_loss
    )
    independent_loss = compute_independent_normalized_cosine_response_smooth_l1_loss(
        projected,
        target_descriptor,
        fit_text_bank,
    )
    scene_loss, scene_stats = compute_scene_wise_text_response_profile_ranking_loss(
        projected,
        target_descriptor,
        fit_text_bank,
        scene_ids,
        profile_weight=float(scene_profile_weight),
        ranking_weight=float(scene_ranking_weight),
        ranking_temperature=float(scene_ranking_temperature),
        tie_tolerance=SCENE_TIE_TOLERANCE,
    )
    return surface_loss, independent_loss, scene_loss, {
        "surface": surface_loss,
        "token": token_loss,
        "descriptor": descriptor_loss,
        "relation": relation_loss,
        "independent_response": independent_loss,
        "scene_response": scene_loss,
        "scene_profile": scene_stats["profile_loss"],
        "scene_ranking": scene_stats["ranking_loss"],
    }


def _gradient_l2_norm(
    loss: torch.Tensor,
    parameters: Iterable[torch.Tensor],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    values = tuple(parameters)
    if not values:
        raise ValueError("gradient calibration requires trainable parameters")
    gradients = torch.autograd.grad(
        loss,
        values,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    if any(gradient is None for gradient in gradients):
        raise ValueError(
            "calibration loss leaves a declared trainable parameter unused"
        )
    squares = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    if not squares:
        raise ValueError("calibration loss is disconnected from trainable parameters")
    return torch.stack(squares).sum().sqrt()


def _trainable_parameter_inventory(
    named_parameters: Iterable[tuple[str, torch.Tensor]],
) -> tuple[tuple[torch.Tensor, ...], list[dict[str, Any]]]:
    entries = tuple(named_parameters)
    if not entries or any(
        not isinstance(name, str) or not name or not parameter.requires_grad
        for name, parameter in entries
    ):
        raise ValueError("calibration trainable-parameter inventory is invalid")
    names = [name for name, _ in entries]
    if len(set(names)) != len(names):
        raise ValueError("calibration trainable-parameter names are not unique")
    inventory = [
        {"name": name, "shape": list(parameter.shape)}
        for name, parameter in entries
    ]
    return tuple(parameter for _, parameter in entries), inventory


def calibrate_gradient_budgets(
    predicted_token: torch.Tensor,
    projected: torch.Tensor,
    target_token: torch.Tensor,
    target_descriptor: torch.Tensor,
    all_descriptors: torch.Tensor,
    teacher_mask: torch.Tensor,
    fit_text_bank: torch.Tensor,
    scene_ids: Sequence[str],
    named_parameters: Iterable[tuple[str, torch.Tensor]],
    *,
    token_weight: float,
    relation_weight: float,
    scene_profile_weight: float = SCENE_PROFILE_WEIGHT,
    scene_ranking_weight: float = SCENE_RANKING_WEIGHT,
    scene_ranking_temperature: float = SCENE_RANKING_TEMPERATURE,
) -> dict[str, Any]:
    """Budget both response gradients at a per-seed Surface warm start."""

    values, parameter_inventory = _trainable_parameter_inventory(named_parameters)
    surface_loss, independent_loss, scene_loss, loss_terms = _surface_and_response_losses(
        predicted_token,
        projected,
        target_token,
        target_descriptor,
        all_descriptors,
        teacher_mask,
        fit_text_bank,
        scene_ids,
        token_weight=float(token_weight),
        relation_weight=float(relation_weight),
        scene_profile_weight=float(scene_profile_weight),
        scene_ranking_weight=float(scene_ranking_weight),
        scene_ranking_temperature=float(scene_ranking_temperature),
    )
    surface_norm = _gradient_l2_norm(
        surface_loss,
        values,
        retain_graph=True,
    )
    independent_norm = _gradient_l2_norm(
        independent_loss,
        values,
        retain_graph=True,
    )
    scene_norm = _gradient_l2_norm(
        scene_loss,
        values,
        retain_graph=False,
    )
    norms = {
        "surface": float(surface_norm.cpu()),
        "independent_response": float(independent_norm.cpu()),
        "scene_response": float(scene_norm.cpu()),
    }
    if any(
        not math.isfinite(value) or value <= GRADIENT_NORM_EPSILON
        for value in norms.values()
    ):
        raise ValueError("warm-start Surface/response gradient norm is degenerate")
    independent_lambda = (
        RESPONSE_BRANCH_GRADIENT_RATIO
        * norms["surface"]
        / norms["independent_response"]
    )
    scene_lambda = (
        RESPONSE_BRANCH_GRADIENT_RATIO
        * norms["surface"]
        / norms["scene_response"]
    )
    weighted = {
        "independent_response": independent_lambda * norms["independent_response"],
        "scene_response": scene_lambda * norms["scene_response"],
    }
    bound = weighted["independent_response"] + weighted["scene_response"]
    return {
        "loss_values": {
            name: float(value.detach().cpu()) for name, value in loss_terms.items()
        },
        "trainable_parameter_count": len(parameter_inventory),
        "trainable_parameters": parameter_inventory,
        "gradient_l2": norms,
        "branch_target_ratio": RESPONSE_BRANCH_GRADIENT_RATIO,
        "response_lambdas": {
            "independent_response": independent_lambda,
            "scene_response": scene_lambda,
        },
        "weighted_response_gradient_l2": weighted,
        "combined_response_gradient_l2_upper_bound": bound,
        "combined_response_to_surface_upper_bound_ratio": bound / norms["surface"],
    }


def _cache_binding(paths: list[Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path.resolve()), "sha256": _sha256_file(path.resolve())}
        for path in paths
    ]


def _cache_bound_metadata(
    metadata: Mapping[str, Any],
    paths: list[Path],
) -> dict[str, Any]:
    """Attach exact file digests to the semantic cache provenance."""

    value = dict(metadata)
    resolved = [str(path.resolve()) for path in paths]
    if value.get("cache_paths") != resolved:
        raise ValueError("merged cache provenance paths differ from loaded caches")
    value["cache_bindings"] = _cache_binding(paths)
    return value


def load_surface_control_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    seed: int,
    train_paths: list[Path],
    validation_paths: list[Path],
    train_meta: Mapping[str, Any],
    validation_meta: Mapping[str, Any],
    hidden_dim: int,
    reliability_attention_mode: str,
    context_pooling_mode: str,
) -> tuple[SurfaceRegionSummaryReadoutV2, dict[str, Any]]:
    """Load one frozen Surface-only attention checkpoint, fail closed.

    The caller must supply an external SHA-256 rather than trusting a digest
    stored beside or inside the checkpoint.  Seed, architecture, and complete
    merged cache provenance are then matched to this invocation before the
    state dict is admitted as the trainable epoch-0 initialization.
    """

    checkpoint, digest, source = load_sha_bound_project_checkpoint_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="frozen Surface attention control checkpoint",
    )
    expected_keys = {
        "schema_version",
        "architecture",
        "state_dict",
        "provenance",
        "history",
        "best_epoch",
        "best_selection_score",
        "untrained_baseline",
        "untrained_baseline_score",
        "training_config",
    }
    if checkpoint.get("schema_version") != 3 or set(checkpoint) != expected_keys:
        raise ValueError("invalid frozen Surface attention control schema")

    model = SurfaceRegionSummaryReadoutV2(
        hidden_dim=int(hidden_dim),
        reliability_attention_mode=str(reliability_attention_mode),
        context_pooling_mode=str(context_pooling_mode),
    )
    expected_architecture = model.architecture(
        str(train_meta["region_contract_sha256"])
    )
    if checkpoint.get("architecture") != expected_architecture:
        raise ValueError("Surface control architecture differs from treatment")

    config = checkpoint.get("training_config")
    provenance = checkpoint.get("provenance")
    seed_contract = (
        provenance.get("random_seed_contract")
        if isinstance(provenance, Mapping)
        else None
    )
    if (
        not isinstance(config, Mapping)
        or config.get("seed") != int(seed)
        or config.get("hidden_dim") != int(hidden_dim)
        or config.get("reliability_attention_mode", "log_prior")
        != str(reliability_attention_mode)
        or config.get("context_pooling_mode", "joint_attention_v1")
        != str(context_pooling_mode)
        or not isinstance(provenance, Mapping)
        or provenance.get("frozen") is not True
        or provenance.get("uses_benchmark_scenes") is not False
        or provenance.get("uses_benchmark_test_vocabulary") is not False
        or "text_response_distillation" in provenance
        or seed_contract
        != {
            "seed": int(seed),
            "model_initialization": True,
            "data_order": True,
            "canonical_noise": True,
        }
    ):
        raise ValueError("Surface control seed/training provenance differs")
    if provenance.get("train") != dict(train_meta):
        raise ValueError("Surface control training-cache provenance differs")
    if provenance.get("validation") != dict(validation_meta):
        raise ValueError("Surface control validation-cache provenance differs")
    if train_meta.get("cache_paths") != [
        str(value.resolve()) for value in train_paths
    ] or validation_meta.get("cache_paths") != [
        str(value.resolve()) for value in validation_paths
    ]:
        raise ValueError("Surface control cache metadata/path binding differs")

    state_dict = checkpoint.get("state_dict")
    reference_state = model.state_dict()
    if not isinstance(state_dict, Mapping) or set(state_dict) != set(reference_state):
        raise ValueError("Surface control state_dict fields differ")
    for name, expected in reference_state.items():
        value = state_dict[name]
        if (
            not torch.is_tensor(value)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Surface control state tensor {name} differs")
    model.load_state_dict(state_dict, strict=True)

    best_epoch = checkpoint.get("best_epoch")
    best_score = checkpoint.get("best_selection_score")
    if (
        not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or best_epoch <= 0
        or not isinstance(best_score, (int, float))
        or isinstance(best_score, bool)
        or not math.isfinite(float(best_score))
    ):
        raise ValueError("Surface control best-state metadata differs")
    binding = {
        "path": str(source),
        "sha256": digest,
        "seed": int(seed),
        "architecture": expected_architecture,
        "train_caches": _cache_binding(train_paths),
        "validation_caches": _cache_binding(validation_paths),
        "source_best_epoch": best_epoch,
        "source_best_selection_score": float(best_score),
    }
    return model, binding


def _fit_bank_binding(bank: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": str(bank["path"]),
        "artifact_sha256": str(bank["file_sha256"]),
        "manifest_path": str(bank["manifest_path"]),
        "manifest_sha256": str(bank["manifest_sha256"]),
        "split": FIT_SPLIT,
        "query_count": int(bank["query_count"]),
        "split_synset_tab_query_lf_sha256": str(bank["split_sha256"]),
        "ordered_records_sha256": str(bank["ordered_records_sha256"]),
        "vocabulary_sha256": str(bank["vocabulary_sha256"]),
        "vocabulary_manifest_sha256": str(bank["vocabulary_manifest_sha256"]),
        "embedding_semantic_sha256": str(bank["embedding_semantic_sha256"]),
        "embedding_tensor_sha256": str(bank["embedding_tensor_sha256"]),
        "text_encoder_snapshot_files_sha256": str(
            bank["text_encoder"]["snapshot_files_sha256"]
        ),
    }


def _implementation_binding() -> list[dict[str, str]]:
    repo = Path(__file__).resolve().parents[2]
    relative_paths = (
        "radio_gs/scripts/train_surface_region_text_response_distill.py",
        "radio_gs/scripts/train_surface_region_summary_readout.py",
        "radio_gs/losses/direct_point_query_logit_distill_loss.py",
        "radio_gs/interfaces/surface_region_summary.py",
        "radio_gs/models/siglip_projection.py",
        "radio_gs/scripts/build_target_blind_siglip2_embedding_artifact.py",
    )
    return [
        {"path": str((repo / relative).resolve()), "sha256": _sha256_file(repo / relative)}
        for relative in relative_paths
    ]


def load_distill_run_manifest(
    path: Path,
    *,
    train_paths: list[Path],
    validation_paths: list[Path],
    fit_bank: Mapping[str, Any],
    radio_path: Path,
    calibration_path: Path,
    output_path: Path,
    seed: int,
    training_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable runner contract consumed by one training seed."""

    path = Path(path).resolve()
    payload = _read_json(path)
    if (
        payload.get("schema_version") != DISTILL_RUN_MANIFEST_SCHEMA_VERSION
        or payload.get("artifact_type") != DISTILL_RUN_MANIFEST_ARTIFACT_TYPE
    ):
        raise ValueError("invalid text-response distill run manifest schema")
    if payload.get("train_caches") != _cache_binding(train_paths):
        raise ValueError("distill run manifest binds different training caches")
    if payload.get("validation_caches") != _cache_binding(validation_paths):
        raise ValueError("distill run manifest binds different validation caches")
    expected_bank = {
        "artifact": {
            "path": str(fit_bank["path"]),
            "sha256": str(fit_bank["file_sha256"]),
        },
        "manifest": {
            "path": str(fit_bank["manifest_path"]),
            "sha256": str(fit_bank["manifest_sha256"]),
        },
    }
    if payload.get("fit_text_bank") != expected_bank:
        raise ValueError("distill run manifest binds a different fit text bank")
    expected_radio = {
        "path": str(radio_path.resolve()),
        "sha256": _sha256_file(radio_path.resolve()),
    }
    if payload.get("radio_checkpoint") != expected_radio:
        raise ValueError("distill run manifest binds a different RADIO checkpoint")
    calibrations = payload.get("calibrations")
    if not isinstance(calibrations, list) or len(calibrations) != len(
        SHARED_TRAINING_SEEDS
    ):
        raise ValueError("distill run manifest lacks per-seed calibrations")
    calibration_by_seed = {
        row.get("seed"): row for row in calibrations if isinstance(row, Mapping)
    }
    expected_calibration = {
        "path": str(calibration_path.resolve()),
        "sha256": _sha256_file(calibration_path.resolve()),
    }
    if (
        set(calibration_by_seed) != set(SHARED_TRAINING_SEEDS)
        or calibration_by_seed[int(seed)].get("manifest") != expected_calibration
        or not isinstance(calibration_by_seed[int(seed)].get("audit"), Mapping)
    ):
        raise ValueError("distill run manifest binds another seed calibration")
    if payload.get("training_contract") != dict(training_contract):
        raise ValueError("distill run manifest training contract differs")
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(SHARED_TRAINING_SEEDS):
        raise ValueError("distill run manifest has an incomplete seed output index")
    by_seed = {
        value.get("seed"): value
        for value in outputs
        if isinstance(value, Mapping)
    }
    if set(by_seed) != set(SHARED_TRAINING_SEEDS):
        raise ValueError("distill run manifest output seeds differ")
    expected_output = by_seed[int(seed)]
    if (
        expected_output.get("checkpoint") != str(output_path.resolve())
        or expected_output.get("report")
        != str(output_path.resolve().with_suffix(output_path.suffix + ".json"))
    ):
        raise ValueError("distill run manifest binds another seed output")
    implementation = payload.get("implementation_sources")
    if not isinstance(implementation, Mapping) or not implementation:
        raise ValueError("distill run manifest lacks implementation hashes")
    repo = Path(__file__).resolve().parents[2]
    required_sources = {
        "radio_gs/scripts/run_surface_region_text_response_distill.sh",
        "radio_gs/scripts/surface_text_response_distill_authority.py",
        "radio_gs/scripts/train_surface_region_text_response_distill.py",
        "radio_gs/scripts/materialize_surface_text_response_descriptors.py",
        "radio_gs/scripts/finalize_gpu_guard_receipt.py",
        "radio_gs/scripts/finalize_surface_text_response_promotion.py",
        "radio_gs/scripts/train_surface_region_summary_readout.py",
        "radio_gs/losses/direct_point_query_logit_distill_loss.py",
        "radio_gs/interfaces/surface_region_summary.py",
        "radio_gs/models/siglip_projection.py",
        "radio_gs/scripts/build_target_blind_siglip2_embedding_artifact.py",
        "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
        "radio_gs/scripts/run_repo_python.sh",
        "radio_gs/utils/immutable_artifacts.py",
    }
    if not required_sources.issubset(set(implementation)):
        raise ValueError("distill run manifest implementation source set differs")
    for relative in sorted(required_sources):
        source = repo / relative
        if not source.is_file() or implementation[relative] != _sha256_file(source):
            raise ValueError(f"distill implementation source changed: {relative}")
    runtime_closure = payload.get("runtime_closure")
    authority_contract = payload.get("authority_contract")
    if (
        payload.get("authority_status")
        != "query_free_three_seed_gpu1_run_frozen"
        or not isinstance(runtime_closure, Mapping)
        or not isinstance(runtime_closure.get("digest"), str)
        or not isinstance(authority_contract, Mapping)
        or authority_contract.get("seed_resume")
        != "skip_only_exact_guarded_terminal_v1"
    ):
        raise ValueError("distill run manifest lacks authority closure")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "candidate": str(payload.get("candidate", "")),
    }


def _validate_train_validation_contracts(
    train_meta: Mapping[str, Any],
    validation_meta: Mapping[str, Any],
) -> None:
    overlap = set(train_meta["scenes"]) & set(validation_meta["scenes"])
    if overlap:
        raise ValueError(f"train/validation scene leakage: {sorted(overlap)}")
    for field, label in (
        ("region_contract_sha256", "region contracts"),
        ("excluded_physical_spaces", "benchmark exclusion contracts"),
        ("teacher_region", "teacher protocols"),
        ("radio_checkpoint_sha256", "RADIO checkpoints"),
    ):
        if train_meta[field] != validation_meta[field]:
            raise ValueError(f"train/validation {label} differ")


def _verify_radio_checkpoint(path: Path, train_meta: Mapping[str, Any]) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RADIO checkpoint is missing: {path}")
    digest = _sha256_file(path)
    if digest != train_meta["radio_checkpoint_sha256"]:
        raise ValueError("training RADIO checkpoint differs from cache provenance")
    return digest


def _training_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "hidden_dim": int(args.hidden_dim),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "token_weight": float(args.token_weight),
        "relation_weight": float(args.relation_weight),
        "reliability_attention_mode": str(args.reliability_attention_mode),
        "context_pooling_mode": str(
            getattr(args, "context_pooling_mode", "joint_attention_v1")
        ),
        "canonical_noise_degrees": float(args.canonical_noise_degrees),
        "canonical_noise_calibration": str(args.canonical_noise_calibration),
        "seeds": list(SHARED_TRAINING_SEEDS),
        "response_lambda_source": (
            "per_seed_exact_surface_warmstart_gradient_budget"
        ),
        "response_branch_gradient_target_ratio": RESPONSE_BRANCH_GRADIENT_RATIO,
        "total_response_gradient_ratio_upper_bound": (
            TOTAL_RESPONSE_GRADIENT_RATIO_UPPER_BOUND
        ),
        "response_gradient_bound_scope": (
            "local_at_unaugmented_exact_warmstart_not_a_global_training_bound"
        ),
        "response_losses": [
            "independent_normalized_cosine_response_smooth_l1",
            "scene_wise_text_response_profile_ranking",
        ],
        "scene_profile_weight": SCENE_PROFILE_WEIGHT,
        "scene_ranking_weight": SCENE_RANKING_WEIGHT,
        "scene_ranking_temperature": SCENE_RANKING_TEMPERATURE,
        "scene_tie_tolerance": SCENE_TIE_TOLERANCE,
        "training_batching": (
            "shuffle_complete_scene_groups_no_partial_scenes_v1"
        ),
        "max_complete_scene_batch_rows": MAX_COMPLETE_SCENE_BATCH_ROWS,
        "epoch_selection": RESPONSE_EPOCH_SELECTION,
        "surface_control_initialization": "exact_seed_checkpoint_state_dict",
        "surface_control_noninferiority_tolerance": (
            SURFACE_CONTROL_NONINFERIORITY_TOLERANCE
        ),
    }


def _training_config(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "train_caches",
        "validation_caches",
        "fit_text_bank",
        "fit_text_bank_manifest",
        "calibration_manifest",
        "run_manifest",
        "surface_control_checkpoint",
        "surface_control_checkpoint_sha256",
        "output",
        "hidden_dim",
        "epochs",
        "patience",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "token_weight",
        "relation_weight",
        "reliability_attention_mode",
        "context_pooling_mode",
        "canonical_noise_degrees",
        "canonical_noise_calibration",
        "seed",
        "device",
        "radio_checkpoint",
    )
    return {
        key: (
            getattr(args, key, "joint_attention_v1")
            if key == "context_pooling_mode"
            else getattr(args, key)
        )
        for key in keys
    }


def _training_provenance(
    args: argparse.Namespace,
    *,
    train_paths: list[Path],
    validation_paths: list[Path],
    train_meta: Mapping[str, Any],
    validation_meta: Mapping[str, Any],
    fit_bank: Mapping[str, Any],
    calibration: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    surface_control: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "training_scope": "global_cross_scene_3d_surface_v2",
        "frozen": True,
        "uses_benchmark_scenes": False,
        "uses_benchmark_test_vocabulary": False,
        "train": _cache_bound_metadata(train_meta, train_paths),
        "validation": _cache_bound_metadata(validation_meta, validation_paths),
        "scene_disjoint": True,
        "official_summary_head": "c-radio_v4 siglip2-g",
        "custom_text_projection": False,
        "context_pooling_mode": str(
            getattr(args, "context_pooling_mode", "joint_attention_v1")
        ),
        "region_contract_sha256": train_meta["region_contract_sha256"],
        "region_contract": train_meta["region_contract"],
        "canonical_direction_noise_degrees": float(args.canonical_noise_degrees),
        "canonical_noise_calibration": str(args.canonical_noise_calibration),
        "random_seed_contract": {
            "seed": seed,
            "model_initialization": False,
            "model_initialization_source": "frozen_seed_surface_control",
            "data_order": True,
            "canonical_noise": True,
        },
        "surface_control_warm_start": {
            **dict(surface_control),
            "epoch": 0,
            "noninferiority_metrics": list(SURFACE_CONTROL_METRICS),
            "noninferiority_tolerance": (
                SURFACE_CONTROL_NONINFERIORITY_TOLERANCE
            ),
            "selection_policy": RESPONSE_EPOCH_SELECTION,
        },
        "text_response_distillation": {
            "fit_split_only": True,
            "benchmark_vocabulary_opened": False,
            "fit_text_bank": _fit_bank_binding(fit_bank),
            "calibration_manifest": str(calibration["path"]),
            "calibration_manifest_sha256": calibration["file_sha256"],
            "calibration_seed": seed,
            "response_lambdas": dict(calibration["response_lambdas"]),
            "response_branch_gradient_target_ratio": (
                RESPONSE_BRANCH_GRADIENT_RATIO
            ),
            "total_response_gradient_ratio_upper_bound": (
                TOTAL_RESPONSE_GRADIENT_RATIO_UPPER_BOUND
            ),
            "losses": [
                "independent_normalized_cosine_response_smooth_l1",
                "scene_wise_text_response_profile_ranking",
            ],
            "complete_scene_batching": True,
            "design_diagnostic": dict(calibration["design_diagnostic"]),
        },
        "distill_run_manifest": dict(run_manifest),
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        if path.is_file() and path.stat().st_size == 0:
            path.unlink()
        raise


def _calibration_objective_contract(
    *,
    token_weight: float,
    relation_weight: float,
) -> dict[str, Any]:
    return {
        "surface_objective": (
            "token_weight*(1-cosine_summary_token)"
            "+masked_mean_one_minus_all_view_cosine"
            "+relation_weight*smooth_l1_descriptor_relation"
        ),
        "token_weight": float(token_weight),
        "relation_weight": float(relation_weight),
        "independent_response_loss": (
            "independent_normalized_cosine_response_smooth_l1"
        ),
        "scene_response_loss": "scene_wise_text_response_profile_ranking",
        "scene_profile_weight": SCENE_PROFILE_WEIGHT,
        "scene_ranking_weight": SCENE_RANKING_WEIGHT,
        "scene_ranking_temperature": SCENE_RANKING_TEMPERATURE,
        "scene_tie_tolerance": SCENE_TIE_TOLERANCE,
        "branch_gradient_target_ratio": RESPONSE_BRANCH_GRADIENT_RATIO,
        "combined_response_gradient_ratio_upper_bound": (
            TOTAL_RESPONSE_GRADIENT_RATIO_UPPER_BOUND
        ),
        "upper_bound_derivation": (
            "triangle_inequality_sum_of_two_branch_l2_budgets"
        ),
        "gradient_bound_scope": (
            "local_at_unaugmented_exact_warmstart_not_a_global_training_bound"
        ),
        "training_batching": "shuffle_complete_scene_groups_no_partial_scenes_v1",
        "max_complete_scene_batch_rows": MAX_COMPLETE_SCENE_BATCH_ROWS,
    }


def _load_gradient_design_diagnostic(
    path: Path,
    *,
    expected_sha256: str,
    train_paths: list[Path],
    radio_path: Path,
    fit_bank: Mapping[str, Any],
    scenes: list[str],
    row_count: int,
) -> dict[str, Any]:
    source = Path(path).resolve()
    if _sha256_file(source) != str(expected_sha256):
        raise ValueError("gradient design diagnostic SHA-256 differs")
    payload = _read_json(source)
    if (
        set(payload)
        != {
            "schema_version",
            "artifact_type",
            "device",
            "rows",
            "scenes",
            "ranking_temperature",
            "losses",
            "gradient_l2",
            "equal_surface_gradient_lambdas",
            "bindings",
        }
        or payload.get("schema_version") != 1
        or payload.get("artifact_type")
        != "warmstart_surface_text_response_gradient_diagnostic"
        or payload.get("scenes") != scenes
        or payload.get("rows") != int(row_count)
        or payload.get("ranking_temperature") != SCENE_RANKING_TEMPERATURE
    ):
        raise ValueError("gradient design diagnostic contract differs")
    bindings = payload.get("bindings")
    expected_simple = {
        "radio_checkpoint": {
            "path": str(radio_path.resolve()),
            "sha256": _sha256_file(radio_path.resolve()),
        },
        "train_caches": _cache_binding(train_paths),
        "fit_text_bank": {
            "path": str(fit_bank["path"]),
            "sha256": str(fit_bank["file_sha256"]),
        },
        "fit_text_bank_manifest": {
            "path": str(fit_bank["manifest_path"]),
            "sha256": str(fit_bank["manifest_sha256"]),
        },
    }
    if not isinstance(bindings, Mapping) or any(
        bindings.get(key) != value for key, value in expected_simple.items()
    ):
        raise ValueError("gradient design diagnostic input bindings differ")
    if set(bindings) != {
        "surface_control",
        "radio_checkpoint",
        "train_caches",
        "fit_text_bank",
        "fit_text_bank_manifest",
        "implementation",
    }:
        raise ValueError("gradient design diagnostic binding fields differ")
    for key in ("surface_control", "implementation"):
        record = bindings.get(key)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or _sha256_file(Path(str(record.get("path", ""))).resolve())
            != record.get("sha256")
        ):
            raise ValueError(f"gradient design diagnostic {key} binding differs")
    norms = payload.get("gradient_l2")
    equal = payload.get("equal_surface_gradient_lambdas")
    if not isinstance(norms, Mapping) or set(norms) != {
        "surface",
        "independent_response",
        "scene_response",
    } or not isinstance(equal, Mapping) or set(equal) != {
        "independent_response",
        "scene_response",
    }:
        raise ValueError("gradient design diagnostic measurements differ")
    numeric = [*norms.values(), *equal.values()]
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= GRADIENT_NORM_EPSILON
        for value in numeric
    ):
        raise ValueError("gradient design diagnostic measurements are invalid")
    for branch in ("independent_response", "scene_response"):
        if not math.isclose(
            float(equal[branch]),
            float(norms["surface"]) / float(norms[branch]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("gradient design diagnostic ratio is inconsistent")
    return {
        "path": str(source),
        "sha256": str(expected_sha256),
        "role": "seed0_design_prior_only_per_seed_values_remeasured",
        "measured_seed": 0,
        "calibration_reuses_measured_values": False,
        "diagnostic_surface_control": dict(bindings["surface_control"]),
    }


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    """Measure one seed's response budgets at its exact Surface warm start."""

    if str(args.device).lower() != "cpu":
        raise ValueError("gradient calibration is CPU-only; --device must be cpu")
    seed = int(args.seed)
    if seed not in SHARED_TRAINING_SEEDS:
        raise ValueError("calibration seed must be one of 0/1/2")
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("per-seed calibration manifest already exists")
    train_paths = _paths(args.train_caches)
    validation_paths = _paths(args.validation_caches)
    train_data, train_meta = _load(train_paths, "train")
    _, validation_meta = _load(validation_paths, "validation")
    _validate_train_validation_contracts(train_meta, validation_meta)
    radio_path = Path(args.radio_checkpoint).resolve()
    radio_sha = _verify_radio_checkpoint(radio_path, train_meta)
    fit_bank = load_fit_text_embedding_bank(
        Path(args.fit_text_bank), Path(args.fit_text_bank_manifest)
    )
    model, surface_control = load_surface_control_checkpoint(
        Path(args.surface_control_checkpoint),
        expected_sha256=str(args.surface_control_checkpoint_sha256),
        seed=seed,
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_meta=train_meta,
        validation_meta=validation_meta,
        hidden_dim=int(args.hidden_dim),
        reliability_attention_mode=str(args.reliability_attention_mode),
        context_pooling_mode=str(args.context_pooling_mode),
    )
    row_count = len(train_data["radio_features"])
    scenes, rows = fixed_calibration_scene_batch(
        train_data.get("scene_ids"), row_count=row_count
    )
    selected_scene_ids = [train_data["scene_ids"][row] for row in rows.tolist()]
    diagnostic = _load_gradient_design_diagnostic(
        Path(args.gradient_diagnostic),
        expected_sha256=str(args.gradient_diagnostic_sha256),
        train_paths=train_paths,
        radio_path=radio_path,
        fit_bank=fit_bank,
        scenes=scenes,
        row_count=len(rows),
    )
    if seed == 0 and diagnostic["diagnostic_surface_control"] != {
        "path": surface_control["path"],
        "sha256": surface_control["sha256"],
    }:
        raise ValueError("seed-0 design diagnostic binds another Surface control")
    head = SigLIP2SummaryHead.from_radio_checkpoint(str(radio_path)).cpu().eval()
    head.requires_grad_(False)
    model = model.cpu().train().requires_grad_(True)
    target_token, target_descriptor, all_descriptors, teacher_mask = _targets(
        train_data, rows
    )
    predicted = model(
        train_data["radio_features"][rows].cpu(),
        train_data["geometry"][rows].cpu(),
        anchor_index=train_data["anchor_index"][rows].cpu(),
        token_mask=train_data["token_mask"][rows].cpu(),
        reliability=train_data["reliability"][rows].cpu(),
    )
    projected = F.normalize(head(predicted[:, None])[:, 0].float(), dim=-1, eps=1e-8)
    gradient = calibrate_gradient_budgets(
        predicted,
        projected,
        target_token.cpu(),
        target_descriptor.cpu(),
        all_descriptors.cpu(),
        teacher_mask.cpu(),
        fit_bank["embeddings"],
        selected_scene_ids,
        (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        token_weight=float(args.token_weight),
        relation_weight=float(args.relation_weight),
    )
    fixed_batch = {
        "split_role": "train",
        "scene_selection_algorithm": CALIBRATION_SCENE_SELECTION,
        "requested_scene_count": CALIBRATION_SCENE_COUNT,
        "scenes": scenes,
        "scene_row_counts": {
            scene: selected_scene_ids.count(scene) for scene in scenes
        },
        "row_indices": rows.tolist(),
        "effective_row_count": len(rows),
        "complete_scenes": True,
        "augmentation": "none",
    }
    payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "artifact_type": CALIBRATION_ARTIFACT_TYPE,
        "algorithm_version": CALIBRATION_ALGORITHM_VERSION,
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_scenes": False,
        "uses_benchmark_test_vocabulary": False,
        "seed": seed,
        "surface_control": surface_control,
        "fixed_calibration_scene_batch": fixed_batch,
        "objective_contract": _calibration_objective_contract(
            token_weight=float(args.token_weight),
            relation_weight=float(args.relation_weight),
        ),
        "gradient_contract": {
            "parameter_set": (
                "all_trainable_surface_region_summary_readout_v2_parameters"
            ),
            "measurement_point": "exact_seed_frozen_surface_control_state_dict",
            "norm": "joint_parameter_gradient_l2",
            "epsilon": GRADIENT_NORM_EPSILON,
            **gradient,
        },
        "design_diagnostic": diagnostic,
        "architecture": surface_control["architecture"],
        "train_caches": _cache_binding(train_paths),
        "validation_caches": _cache_binding(validation_paths),
        "train_contract": {
            "region_contract_sha256": train_meta["region_contract_sha256"],
            "radio_checkpoint_sha256": train_meta["radio_checkpoint_sha256"],
            "row_count": row_count,
            "split_hashes": train_meta["split_hashes"],
            "scenes": train_meta["scenes"],
            "excluded_physical_spaces": train_meta["excluded_physical_spaces"],
            "teacher_region": train_meta["teacher_region"],
        },
        "radio_checkpoint": {"path": str(radio_path), "sha256": radio_sha},
        "fit_text_bank": _fit_bank_binding(fit_bank),
        "implementation": _implementation_binding(),
    }
    _write_json_exclusive(output, payload)
    return {
        "output": str(output),
        "manifest_sha256": _sha256_file(output),
        "seed": seed,
        "response_lambdas": gradient["response_lambdas"],
        "gradient_l2": gradient["gradient_l2"],
        "fit_query_count": fit_bank["query_count"],
    }


def load_calibration_manifest(
    path: Path,
    *,
    seed: int,
    train_paths: list[Path],
    validation_paths: list[Path],
    train_meta: Mapping[str, Any],
    train_scene_ids: object,
    train_row_count: int,
    radio_path: Path,
    fit_bank: Mapping[str, Any],
    surface_control: Mapping[str, Any],
    trainable_parameters: Iterable[tuple[str, torch.Tensor]],
    token_weight: float,
    relation_weight: float,
) -> dict[str, Any]:
    """Fail closed on the per-seed warm-start gradient-budget schema."""

    source = Path(path).resolve()
    payload = _read_json(source)
    expected_keys = {
        "schema_version", "artifact_type", "algorithm_version",
        "benchmark_vocabulary_opened", "uses_benchmark_scenes",
        "uses_benchmark_test_vocabulary", "seed", "surface_control",
        "fixed_calibration_scene_batch", "objective_contract",
        "gradient_contract", "design_diagnostic", "architecture",
        "train_caches", "validation_caches", "train_contract",
        "radio_checkpoint", "fit_text_bank", "implementation",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION
        or payload.get("artifact_type") != CALIBRATION_ARTIFACT_TYPE
        or payload.get("algorithm_version") != CALIBRATION_ALGORITHM_VERSION
        or any(
            payload.get(key) is not False
            for key in (
                "benchmark_vocabulary_opened",
                "uses_benchmark_scenes",
                "uses_benchmark_test_vocabulary",
            )
        )
        or payload.get("seed") != int(seed)
        or payload.get("surface_control") != dict(surface_control)
        or payload.get("architecture") != surface_control["architecture"]
    ):
        raise ValueError("invalid per-seed warm-start calibration schema")
    scenes, rows_tensor = fixed_calibration_scene_batch(
        train_scene_ids, row_count=train_row_count
    )
    scene_ids = list(train_scene_ids)
    expected_fixed = {
        "split_role": "train",
        "scene_selection_algorithm": CALIBRATION_SCENE_SELECTION,
        "requested_scene_count": CALIBRATION_SCENE_COUNT,
        "scenes": scenes,
        "scene_row_counts": {
            scene: sum(value == scene for value in scene_ids) for scene in scenes
        },
        "row_indices": rows_tensor.tolist(),
        "effective_row_count": len(rows_tensor),
        "complete_scenes": True,
        "augmentation": "none",
    }
    if payload.get("fixed_calibration_scene_batch") != expected_fixed:
        raise ValueError("calibration complete-scene batch differs")
    if payload.get("objective_contract") != _calibration_objective_contract(
        token_weight=float(token_weight), relation_weight=float(relation_weight)
    ):
        raise ValueError("calibration objective contract differs")
    expected_train_contract = {
        "region_contract_sha256": train_meta["region_contract_sha256"],
        "radio_checkpoint_sha256": train_meta["radio_checkpoint_sha256"],
        "row_count": train_row_count,
        "split_hashes": train_meta["split_hashes"],
        "scenes": train_meta["scenes"],
        "excluded_physical_spaces": train_meta["excluded_physical_spaces"],
        "teacher_region": train_meta["teacher_region"],
    }
    if (
        payload.get("train_caches") != _cache_binding(train_paths)
        or payload.get("validation_caches") != _cache_binding(validation_paths)
        or payload.get("train_contract") != expected_train_contract
        or payload.get("radio_checkpoint")
        != {"path": str(radio_path.resolve()), "sha256": _sha256_file(radio_path)}
        or payload.get("fit_text_bank") != _fit_bank_binding(fit_bank)
        or payload.get("implementation") != _implementation_binding()
    ):
        raise ValueError("calibration immutable input binding differs")
    diagnostic = payload.get("design_diagnostic")
    if not isinstance(diagnostic, Mapping) or set(diagnostic) != {
        "path", "sha256", "role", "measured_seed",
        "calibration_reuses_measured_values", "diagnostic_surface_control",
    }:
        raise ValueError("calibration design diagnostic binding differs")
    expected_diagnostic = _load_gradient_design_diagnostic(
        Path(str(diagnostic.get("path", ""))),
        expected_sha256=str(diagnostic.get("sha256", "")),
        train_paths=train_paths,
        radio_path=radio_path,
        fit_bank=fit_bank,
        scenes=scenes,
        row_count=len(rows_tensor),
    )
    if dict(diagnostic) != expected_diagnostic:
        raise ValueError("calibration design diagnostic role differs")
    if int(seed) == 0 and diagnostic["diagnostic_surface_control"] != {
        "path": surface_control["path"],
        "sha256": surface_control["sha256"],
    }:
        raise ValueError("seed-0 calibration diagnostic control differs")
    gradient = payload.get("gradient_contract")
    if not isinstance(gradient, Mapping) or set(gradient) != {
        "parameter_set", "measurement_point", "norm", "epsilon",
        "loss_values", "gradient_l2", "branch_target_ratio",
        "trainable_parameter_count", "trainable_parameters",
        "response_lambdas", "weighted_response_gradient_l2",
        "combined_response_gradient_l2_upper_bound",
        "combined_response_to_surface_upper_bound_ratio",
    }:
        raise ValueError("calibration gradient fields differ")
    if (
        gradient.get("parameter_set")
        != "all_trainable_surface_region_summary_readout_v2_parameters"
        or gradient.get("measurement_point")
        != "exact_seed_frozen_surface_control_state_dict"
        or gradient.get("norm") != "joint_parameter_gradient_l2"
        or gradient.get("epsilon") != GRADIENT_NORM_EPSILON
        or gradient.get("branch_target_ratio") != RESPONSE_BRANCH_GRADIENT_RATIO
    ):
        raise ValueError("calibration gradient definition differs")
    _, expected_inventory = _trainable_parameter_inventory(trainable_parameters)
    if (
        gradient.get("trainable_parameter_count") != len(expected_inventory)
        or gradient.get("trainable_parameters") != expected_inventory
    ):
        raise ValueError("calibration trainable-parameter topology differs")
    loss_values = gradient.get("loss_values")
    if not isinstance(loss_values, Mapping) or set(loss_values) != {
        "surface", "token", "descriptor", "relation",
        "independent_response", "scene_response", "scene_profile",
        "scene_ranking",
    } or any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        or not math.isfinite(float(value)) or float(value) < 0.0
        for value in loss_values.values()
    ):
        raise ValueError("calibration loss measurements differ")
    norms = gradient.get("gradient_l2")
    lambdas = gradient.get("response_lambdas")
    weighted = gradient.get("weighted_response_gradient_l2")
    branches = {"independent_response", "scene_response"}
    if (
        not isinstance(norms, Mapping)
        or set(norms) != {"surface", *branches}
        or not isinstance(lambdas, Mapping)
        or set(lambdas) != branches
        or not isinstance(weighted, Mapping)
        or set(weighted) != branches
    ):
        raise ValueError("calibration gradient measurements differ")
    numeric = [*norms.values(), *lambdas.values(), *weighted.values()]
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        or not math.isfinite(float(value)) or float(value) <= GRADIENT_NORM_EPSILON
        for value in numeric
    ):
        raise ValueError("calibration gradient measurements are invalid")
    for branch in branches:
        expected_lambda = (
            RESPONSE_BRANCH_GRADIENT_RATIO
            * float(norms["surface"]) / float(norms[branch])
        )
        if not math.isclose(float(lambdas[branch]), expected_lambda, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("calibration branch lambda differs from its budget")
        if not math.isclose(
            float(weighted[branch]),
            RESPONSE_BRANCH_GRADIENT_RATIO * float(norms["surface"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("calibration weighted branch gradient differs")
    bound = sum(float(weighted[branch]) for branch in branches)
    ratio = bound / float(norms["surface"])
    if (
        not math.isclose(
            float(gradient.get("combined_response_gradient_l2_upper_bound")),
            bound,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(gradient.get("combined_response_to_surface_upper_bound_ratio")),
            ratio,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or ratio > TOTAL_RESPONSE_GRADIENT_RATIO_UPPER_BOUND + 1e-12
    ):
        raise ValueError("calibration combined response gradient exceeds its bound")
    return {
        "path": source,
        "file_sha256": _sha256_file(source),
        "payload": payload,
        "seed": int(seed),
        "response_lambdas": {
            branch: float(lambdas[branch]) for branch in sorted(branches)
        },
        "gradient_l2": {key: float(value) for key, value in norms.items()},
        "design_diagnostic": expected_diagnostic,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Run the paired-seed treatment without changing inference artifacts."""

    seed = int(args.seed)
    if seed not in SHARED_TRAINING_SEEDS:
        raise ValueError("text-response treatment requires one of seeds 0/1/2")
    train_paths = _paths(args.train_caches)
    validation_paths = _paths(args.validation_caches)
    train_data, train_meta = _load(train_paths, "train")
    validation_data, validation_meta = _load(validation_paths, "validation")
    _validate_train_validation_contracts(train_meta, validation_meta)
    radio_path = Path(args.radio_checkpoint).resolve()
    _verify_radio_checkpoint(radio_path, train_meta)
    fit_bank = load_fit_text_embedding_bank(
        Path(args.fit_text_bank),
        Path(args.fit_text_bank_manifest),
    )
    model, surface_control = load_surface_control_checkpoint(
        Path(args.surface_control_checkpoint),
        expected_sha256=str(args.surface_control_checkpoint_sha256),
        seed=seed,
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_meta=train_meta,
        validation_meta=validation_meta,
        hidden_dim=int(args.hidden_dim),
        reliability_attention_mode=str(args.reliability_attention_mode),
        context_pooling_mode=str(
            getattr(args, "context_pooling_mode", "joint_attention_v1")
        ),
    )
    calibration = load_calibration_manifest(
        Path(args.calibration_manifest),
        seed=seed,
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_meta=train_meta,
        train_scene_ids=train_data.get("scene_ids"),
        train_row_count=len(train_data["radio_features"]),
        radio_path=radio_path,
        fit_bank=fit_bank,
        surface_control=surface_control,
        trainable_parameters=(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        token_weight=float(args.token_weight),
        relation_weight=float(args.relation_weight),
    )
    response_lambdas = dict(calibration["response_lambdas"])
    output = Path(args.output).resolve()
    run_manifest = load_distill_run_manifest(
        Path(args.run_manifest),
        train_paths=train_paths,
        validation_paths=validation_paths,
        fit_bank=fit_bank,
        radio_path=radio_path,
        calibration_path=Path(args.calibration_manifest),
        output_path=output,
        seed=seed,
        training_contract=_training_contract(args),
    )

    device = torch.device(args.device)
    generator = _seed_training(seed, device=device)
    model = model.to(device)
    head = SigLIP2SummaryHead.from_radio_checkpoint(str(radio_path)).to(device).eval()
    head.requires_grad_(False)
    text_bank = fit_bank["embeddings"].to(device)

    model.eval()
    control_validation, control_response = _evaluate_response_aware(
        model,
        head,
        validation_data,
        device,
        int(args.batch_size),
        text_bank,
    )
    control_score = 0.5 * (
        control_validation["mean_descriptor_cosine"]
        + control_validation["all_view_descriptor_cosine"]
    )
    print(
        json.dumps(
            {
                "surface_control_epoch0": control_validation,
                "selection_score": control_score,
                **control_response,
                "response_lambdas": response_lambdas,
            }
        ),
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    best_score = control_score
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "initialization": "frozen_surface_control_checkpoint",
            "response_lambdas": response_lambdas,
            "surface_selection_score": control_score,
            "selection_score": control_score,
            **control_validation,
            **control_response,
        }
    ]
    history, _, _ = finalize_response_primary_epoch_selection(history)
    stale = 0
    observed_peak_batch_rows = 0
    observed_batch_count = 0
    for epoch in range(int(args.epochs)):
        batches = complete_scene_batches(
            train_data.get("scene_ids"),
            row_count=len(train_data["radio_features"]),
            target_batch_rows=int(args.batch_size),
            generator=generator,
        )
        epoch_terms = {
            name: []
            for name in (
                "total", "token", "descriptor", "relation",
                "independent_response", "scene_response",
                "scene_profile", "scene_ranking",
            )
        }
        epoch_peak_batch_rows = max(len(rows) for rows in batches)
        observed_peak_batch_rows = max(
            observed_peak_batch_rows, epoch_peak_batch_rows
        )
        observed_batch_count += len(batches)
        model.train()
        for rows in batches:
            target_token, target_descriptor, all_descriptors, teacher_mask = _targets(
                train_data,
                rows,
            )
            token_mask = train_data["token_mask"][rows].to(device)
            radio_features = inject_tangent_direction_noise(
                train_data["radio_features"][rows].to(device),
                token_mask,
                angle_degrees=float(args.canonical_noise_degrees),
            )
            predicted = model(
                radio_features,
                train_data["geometry"][rows].to(device),
                anchor_index=train_data["anchor_index"][rows].to(device),
                token_mask=token_mask,
                reliability=train_data["reliability"][rows].to(device),
            )
            projected = F.normalize(
                head(predicted[:, None])[:, 0].float(),
                dim=-1,
                eps=1e-8,
            )
            terms = compute_training_losses(
                predicted,
                projected,
                target_token.to(device),
                target_descriptor.to(device),
                all_descriptors.to(device),
                teacher_mask.to(device),
                text_bank,
                [train_data["scene_ids"][row] for row in rows.tolist()],
                token_weight=float(args.token_weight),
                relation_weight=float(args.relation_weight),
                independent_response_lambda=response_lambdas[
                    "independent_response"
                ],
                scene_response_lambda=response_lambdas["scene_response"],
            )
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            optimizer.step()
            for name, value in terms.items():
                epoch_terms[name].append(float(value.detach().cpu()))

        model.eval()
        metrics, response_metrics = _evaluate_response_aware(
            model,
            head,
            validation_data,
            device,
            int(args.batch_size),
            text_bank,
        )
        score = 0.5 * (
            metrics["mean_descriptor_cosine"] + metrics["all_view_descriptor_cosine"]
        )
        record = {
            "epoch": epoch + 1,
            "loss": sum(epoch_terms["total"]) / len(epoch_terms["total"]),
            "token_loss": sum(epoch_terms["token"]) / len(epoch_terms["token"]),
            "descriptor_loss": sum(epoch_terms["descriptor"])
            / len(epoch_terms["descriptor"]),
            "relation_loss": sum(epoch_terms["relation"])
            / len(epoch_terms["relation"]),
            "independent_response_loss": sum(epoch_terms["independent_response"])
            / len(epoch_terms["independent_response"]),
            "scene_response_loss": sum(epoch_terms["scene_response"])
            / len(epoch_terms["scene_response"]),
            "scene_profile_loss": sum(epoch_terms["scene_profile"])
            / len(epoch_terms["scene_profile"]),
            "scene_ranking_loss": sum(epoch_terms["scene_ranking"])
            / len(epoch_terms["scene_ranking"]),
            "response_lambdas": response_lambdas,
            "complete_scene_batch_count": len(batches),
            "max_complete_scene_batch_rows": epoch_peak_batch_rows,
            "surface_selection_score": score,
            "selection_score": score,
            **metrics,
            **response_metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        history, selected_epoch, selected_score = (
            finalize_response_primary_epoch_selection(history)
        )
        if selected_epoch == epoch + 1:
            best_score = selected_score
            best_epoch = selected_epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if int(args.patience) and stale >= int(args.patience):
            break

    history, selected_epoch, selected_score = (
        finalize_response_primary_epoch_selection(history)
    )
    if selected_epoch != best_epoch or not math.isclose(
        selected_score,
        best_score,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("online and finalized response-aware selection differ")
    model.load_state_dict(best_state)
    architecture = model.architecture(train_meta["region_contract_sha256"])
    training_config = _training_config(args)
    provenance = _training_provenance(
        args,
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_meta=train_meta,
        validation_meta=validation_meta,
        fit_bank=fit_bank,
        calibration=calibration,
        run_manifest=run_manifest,
        surface_control=surface_control,
        seed=seed,
    )
    payload = {
        "schema_version": 3,
        "architecture": architecture,
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "provenance": provenance,
        "history": history,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "surface_control_checkpoint": surface_control,
        "surface_control_validation": control_validation,
        "surface_control_score": control_score,
        "complete_scene_batching": {
            "algorithm": "shuffle_complete_scene_groups_no_partial_scenes_v1",
            "row_limit": MAX_COMPLETE_SCENE_BATCH_ROWS,
            "observed_peak_rows": observed_peak_batch_rows,
            "observed_batch_count": observed_batch_count,
        },
        "training_config": training_config,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(output, payload)
    digest = _sha256_file(output)
    validation, _validation_response = _evaluate_response_aware(
        model.to(device),
        head,
        validation_data,
        device,
        int(args.batch_size),
        text_bank,
    )
    report = {
        "output": str(output),
        "checkpoint_sha256": digest,
        "architecture": architecture,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "surface_control_checkpoint": surface_control,
        "surface_control_validation": control_validation,
        "surface_control_score": control_score,
        "selection_score_delta": best_score - control_score,
        "validation": validation,
        "response_lambdas": response_lambdas,
        "complete_scene_batching": payload["complete_scene_batching"],
        "calibration_manifest": str(calibration["path"]),
        "calibration_manifest_sha256": calibration["file_sha256"],
        "fit_text_bank_sha256": fit_bank["file_sha256"],
        "fit_query_count": fit_bank["query_count"],
        "distill_run_manifest": run_manifest["path"],
        "distill_run_manifest_sha256": run_manifest["sha256"],
        "validation_caches": _cache_binding(validation_paths),
        "train_scenes": len(train_meta["scenes"]),
        "validation_scenes": len(validation_meta["scenes"]),
        "scene_overlap": [],
    }
    report_path = output.with_suffix(output.suffix + ".json")
    write_frozen_json(report_path, report)
    return report


def audit_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Revalidate an existing calibration before a runner reuses it."""

    if str(args.device).lower() != "cpu":
        raise ValueError("calibration audit is CPU-only; --device must be cpu")
    train_paths = _paths(args.train_caches)
    validation_paths = _paths(args.validation_caches)
    train_data, train_meta = _load(train_paths, "train")
    _, validation_meta = _load(validation_paths, "validation")
    _validate_train_validation_contracts(train_meta, validation_meta)
    radio_path = Path(args.radio_checkpoint).resolve()
    _verify_radio_checkpoint(radio_path, train_meta)
    fit_bank = load_fit_text_embedding_bank(
        Path(args.fit_text_bank), Path(args.fit_text_bank_manifest)
    )
    surface_model, surface_control = load_surface_control_checkpoint(
        Path(args.surface_control_checkpoint),
        expected_sha256=str(args.surface_control_checkpoint_sha256),
        seed=int(args.seed),
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_meta=train_meta,
        validation_meta=validation_meta,
        hidden_dim=int(args.hidden_dim),
        reliability_attention_mode=str(args.reliability_attention_mode),
        context_pooling_mode=str(args.context_pooling_mode),
    )
    value = load_calibration_manifest(
        Path(args.calibration_manifest),
        seed=int(args.seed),
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_meta=train_meta,
        train_scene_ids=train_data.get("scene_ids"),
        train_row_count=len(train_data["radio_features"]),
        radio_path=radio_path,
        fit_bank=fit_bank,
        surface_control=surface_control,
        trainable_parameters=(
            (name, parameter)
            for name, parameter in surface_model.named_parameters()
            if parameter.requires_grad
        ),
        token_weight=float(args.token_weight),
        relation_weight=float(args.relation_weight),
    )
    if value["design_diagnostic"] != _load_gradient_design_diagnostic(
        Path(args.gradient_diagnostic),
        expected_sha256=str(args.gradient_diagnostic_sha256),
        train_paths=train_paths,
        radio_path=radio_path,
        fit_bank=fit_bank,
        scenes=value["payload"]["fixed_calibration_scene_batch"]["scenes"],
        row_count=value["payload"]["fixed_calibration_scene_batch"][
            "effective_row_count"
        ],
    ):
        raise ValueError("calibration audit design diagnostic differs")
    return {
        "status": "calibration_verified",
        "calibration_manifest": str(value["path"]),
        "calibration_manifest_sha256": value["file_sha256"],
        "seed": int(args.seed),
        "response_lambdas": value["response_lambdas"],
        "device": "cpu",
    }


def _finite_float(value: object, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def audit_training_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    """Fail closed before reusing one checkpoint/report pair."""

    seed = int(args.seed)
    if seed not in SHARED_TRAINING_SEEDS:
        raise ValueError("text-response treatment requires one of seeds 0/1/2")
    train_paths = _paths(args.train_caches)
    validation_paths = _paths(args.validation_caches)
    train_data, train_meta = _load(train_paths, "train")
    train_row_count = len(train_data["radio_features"])
    validation_data, validation_meta = _load(validation_paths, "validation")
    del validation_data
    _validate_train_validation_contracts(train_meta, validation_meta)
    radio_path = Path(args.radio_checkpoint).resolve()
    _verify_radio_checkpoint(radio_path, train_meta)
    fit_bank = load_fit_text_embedding_bank(
        Path(args.fit_text_bank), Path(args.fit_text_bank_manifest)
    )
    output = Path(args.output).resolve()
    run_manifest = load_distill_run_manifest(
        Path(args.run_manifest),
        train_paths=train_paths,
        validation_paths=validation_paths,
        fit_bank=fit_bank,
        radio_path=radio_path,
        calibration_path=Path(args.calibration_manifest),
        output_path=output,
        seed=seed,
        training_contract=_training_contract(args),
    )
    surface_model, surface_control = load_surface_control_checkpoint(
        Path(args.surface_control_checkpoint),
        expected_sha256=str(args.surface_control_checkpoint_sha256),
        seed=seed,
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_meta=train_meta,
        validation_meta=validation_meta,
        hidden_dim=int(args.hidden_dim),
        reliability_attention_mode=str(args.reliability_attention_mode),
        context_pooling_mode=str(
            getattr(args, "context_pooling_mode", "joint_attention_v1")
        ),
    )
    calibration = load_calibration_manifest(
        Path(args.calibration_manifest),
        seed=seed,
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_meta=train_meta,
        train_scene_ids=train_data.get("scene_ids"),
        train_row_count=train_row_count,
        radio_path=radio_path,
        fit_bank=fit_bank,
        surface_control=surface_control,
        trainable_parameters=(
            (name, parameter)
            for name, parameter in surface_model.named_parameters()
            if parameter.requires_grad
        ),
        token_weight=float(args.token_weight),
        relation_weight=float(args.relation_weight),
    )
    del train_data
    report_path = output.with_suffix(output.suffix + ".json")
    report = _read_json(report_path)
    expected_checkpoint_sha = str(report.get("checkpoint_sha256", ""))
    if len(expected_checkpoint_sha) != 64:
        raise ValueError("checkpoint report lacks an externally bound SHA-256")
    model, checkpoint, checkpoint_sha, _ = load_surface_region_summary_readout_v2(
        output,
        expected_sha256=expected_checkpoint_sha,
        map_location="cpu",
    )
    model.cpu().eval().requires_grad_(False)
    if checkpoint.get("training_config") != _training_config(args):
        raise ValueError("checkpoint training configuration differs from the runner")
    expected_provenance = _training_provenance(
        args,
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_meta=train_meta,
        validation_meta=validation_meta,
        fit_bank=fit_bank,
        calibration=calibration,
        run_manifest=run_manifest,
        surface_control=surface_control,
        seed=seed,
    )
    if checkpoint.get("provenance") != expected_provenance:
        raise ValueError("checkpoint provenance differs from the bound run inputs")
    history = checkpoint.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("checkpoint lacks training history")
    if any(not isinstance(row, Mapping) for row in history):
        raise ValueError("checkpoint history contains a non-object row")
    observed_batch_rows: list[int] = []
    observed_batch_counts: list[int] = []
    for index, row in enumerate(history):
        if row.get("response_lambdas") != calibration["response_lambdas"]:
            raise ValueError("checkpoint history response lambdas drifted")
        if index == 0:
            continue
        for field in (
            "independent_response_loss",
            "scene_response_loss",
            "scene_profile_loss",
            "scene_ranking_loss",
        ):
            if _finite_float(row.get(field), label=f"history {field}") < 0.0:
                raise ValueError(f"history {field} must be non-negative")
        peak = row.get("max_complete_scene_batch_rows")
        count = row.get("complete_scene_batch_count")
        if (
            not isinstance(peak, int) or isinstance(peak, bool) or peak <= 1
            or peak > MAX_COMPLETE_SCENE_BATCH_ROWS
            or not isinstance(count, int) or isinstance(count, bool) or count <= 0
        ):
            raise ValueError("checkpoint history complete-scene batching differs")
        observed_batch_rows.append(peak)
        observed_batch_counts.append(count)
    finalized_history, best_epoch, best_score = (
        finalize_response_primary_epoch_selection(history)
    )
    if finalized_history != history:
        raise ValueError("checkpoint history response-aware selection drifted")
    if (
        checkpoint.get("best_epoch") != best_epoch
        or not math.isclose(
            _finite_float(
                checkpoint.get("best_selection_score"),
                label="checkpoint best selection score",
            ),
            best_score,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("checkpoint best selection metadata is inconsistent")
    if checkpoint.get("surface_control_checkpoint") != surface_control:
        raise ValueError("checkpoint Surface control binding differs")
    control_validation = checkpoint.get("surface_control_validation")
    if not isinstance(control_validation, Mapping) or set(control_validation) != set(
        SURFACE_CONTROL_METRICS
    ):
        raise ValueError("checkpoint lacks its Surface control validation")
    control_score = 0.5 * (
        _finite_float(
            control_validation.get("mean_descriptor_cosine"),
            label="Surface control descriptor cosine",
        )
        + _finite_float(
            control_validation.get("all_view_descriptor_cosine"),
            label="Surface control all-view cosine",
        )
    )
    if not math.isclose(
        _finite_float(
            checkpoint.get("surface_control_score"),
            label="Surface control selection score",
        ),
        control_score,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("checkpoint Surface control score is inconsistent")

    expected_keys = {
        "output",
        "checkpoint_sha256",
        "architecture",
        "best_epoch",
        "best_selection_score",
        "surface_control_checkpoint",
        "surface_control_validation",
        "surface_control_score",
        "complete_scene_batching",
        "selection_score_delta",
        "validation",
        "response_lambdas",
        "calibration_manifest",
        "calibration_manifest_sha256",
        "fit_text_bank_sha256",
        "fit_query_count",
        "distill_run_manifest",
        "distill_run_manifest_sha256",
        "validation_caches",
        "train_scenes",
        "validation_scenes",
        "scene_overlap",
    }
    if set(report) != expected_keys:
        raise ValueError("checkpoint report fields differ from the fixed schema")
    expected_static = {
        "output": str(output),
        "checkpoint_sha256": checkpoint_sha,
        "architecture": checkpoint["architecture"],
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "surface_control_checkpoint": surface_control,
        "surface_control_validation": control_validation,
        "surface_control_score": control_score,
        "complete_scene_batching": {
            "algorithm": "shuffle_complete_scene_groups_no_partial_scenes_v1",
            "row_limit": MAX_COMPLETE_SCENE_BATCH_ROWS,
            "observed_peak_rows": max(observed_batch_rows, default=0),
            "observed_batch_count": sum(observed_batch_counts),
        },
        "selection_score_delta": best_score - control_score,
        "response_lambdas": calibration["response_lambdas"],
        "calibration_manifest": str(calibration["path"]),
        "calibration_manifest_sha256": calibration["file_sha256"],
        "fit_text_bank_sha256": fit_bank["file_sha256"],
        "fit_query_count": fit_bank["query_count"],
        "distill_run_manifest": run_manifest["path"],
        "distill_run_manifest_sha256": run_manifest["sha256"],
        "validation_caches": _cache_binding(validation_paths),
        "train_scenes": len(train_meta["scenes"]),
        "validation_scenes": len(validation_meta["scenes"]),
        "scene_overlap": [],
    }
    for key, value in expected_static.items():
        if key in {
            "best_selection_score",
            "surface_control_score",
            "selection_score_delta",
        }:
            if not math.isclose(
                _finite_float(report.get(key), label=f"report {key}"),
                float(value),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(f"checkpoint report {key} differs")
        elif report.get(key) != value:
            raise ValueError(f"checkpoint report {key} differs")
    validation = report.get("validation")
    if not isinstance(validation, Mapping) or set(validation) != {
        "summary_token_cosine",
        "mean_descriptor_cosine",
        "all_view_descriptor_cosine",
    }:
        raise ValueError("checkpoint report validation metrics differ")
    validation_score = 0.5 * (
        _finite_float(
            validation["mean_descriptor_cosine"], label="validation descriptor cosine"
        )
        + _finite_float(
            validation["all_view_descriptor_cosine"], label="validation all-view cosine"
        )
    )
    if not math.isclose(validation_score, best_score, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("checkpoint report validation does not reproduce its best score")
    return {
        "status": "checkpoint_report_verified",
        "seed": seed,
        "checkpoint": str(output),
        "checkpoint_sha256": checkpoint_sha,
        "report": str(report_path),
        "report_sha256": _sha256_file(report_path),
        "run_manifest_sha256": run_manifest["sha256"],
        "device": "cpu",
    }


def _add_model_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-caches", required=True)
    parser.add_argument("--fit-text-bank", type=Path, required=True)
    parser.add_argument("--fit-text-bank-manifest", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--reliability-attention-mode",
        choices=("log_prior", "input_only"),
        default="log_prior",
    )
    parser.add_argument(
        "--context-pooling-mode",
        choices=("joint_attention_v1", "core_context_separate_attention_v1"),
        default="joint_attention_v1",
    )
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    _add_model_contract_arguments(parser)
    parser.add_argument("--validation-caches", required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument(
        "--surface-control-checkpoint",
        type=Path,
        required=True,
        help="same-seed frozen Surface attention checkpoint used as epoch 0",
    )
    parser.add_argument(
        "--surface-control-checkpoint-sha256",
        required=True,
        help="external lowercase SHA-256 of --surface-control-checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--token-weight", type=float, default=0.25)
    parser.add_argument("--relation-weight", type=float, default=0.1)
    parser.add_argument("--canonical-noise-degrees", type=float, default=0.0)
    parser.add_argument("--canonical-noise-calibration", default="")
    parser.add_argument(
        "--seed", type=int, choices=SHARED_TRAINING_SEEDS, required=True
    )
    # Audit uses this value only as the expected immutable training config; it
    # never constructs a CUDA device or model.
    parser.add_argument("--device", default="cuda:0")


def _add_calibration_arguments(parser: argparse.ArgumentParser) -> None:
    _add_model_contract_arguments(parser)
    parser.add_argument("--validation-caches", required=True)
    parser.add_argument("--surface-control-checkpoint", type=Path, required=True)
    parser.add_argument("--surface-control-checkpoint-sha256", required=True)
    parser.add_argument("--gradient-diagnostic", type=Path, required=True)
    parser.add_argument("--gradient-diagnostic-sha256", required=True)
    parser.add_argument("--token-weight", type=float, default=0.25)
    parser.add_argument("--relation-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, choices=SHARED_TRAINING_SEEDS, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibration_parser = subparsers.add_parser(
        "calibrate",
        help="write one seed's CPU warm-start gradient-budget manifest",
    )
    _add_calibration_arguments(calibration_parser)
    calibration_parser.add_argument("--output", type=Path, required=True)
    calibration_parser.add_argument("--device", choices=("cpu",), default="cpu")

    calibration_audit_parser = subparsers.add_parser(
        "audit-calibration",
        help="CPU-only verification before reusing a calibration manifest",
    )
    _add_calibration_arguments(calibration_audit_parser)
    calibration_audit_parser.add_argument(
        "--calibration-manifest", type=Path, required=True
    )
    calibration_audit_parser.add_argument(
        "--device", choices=("cpu",), default="cpu"
    )

    train_parser = subparsers.add_parser(
        "train",
        help="train one of the paired seeds using the frozen calibration",
    )
    _add_training_arguments(train_parser)
    checkpoint_audit_parser = subparsers.add_parser(
        "audit-checkpoint",
        help="CPU-only verification before reusing a checkpoint/report pair",
    )
    _add_training_arguments(checkpoint_audit_parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "calibrate":
        result = calibrate(args)
    elif args.command == "audit-calibration":
        result = audit_calibration(args)
    elif args.command == "train":
        result = train(args)
    else:
        result = audit_training_artifacts(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
