"""Source-only query-response diagnostics and scene-global confuser mining.

The implementation is deliberately independent of benchmark queries and
targets.  It compares immutable AcceptedV2 descriptors with official
multi-view SigLIP2 observations on a frozen generic text bank, then mines
spatially-disjoint semantic confusers one scene at a time.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256


AUTHORITY_SCHEMA = "radio_gs.source_query_response_hard_negative_authority.v1"
AUTHORITY_SCHEMA_VERSION = 1
TEACHER_RESPONSE_TEMPERATURE = 0.05
TEACHER_COSINE_MIN = 0.8
TEACHER_COSINE_MAX = 0.98
PER_SOURCE_K = 4
MATRIX_BLOCK_ROWS = 128


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _finite_float_matrix(value: object, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if (
        tensor.device.type != "cpu"
        or tensor.ndim != 2
        or not tensor.is_floating_point()
        or tensor.shape[0] <= 0
        or tensor.shape[1] <= 0
    ):
        raise ValueError(f"{label} must be a nonempty floating CPU [N,D] tensor")
    result = tensor.detach().float().contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} contains nonfinite values")
    if bool((torch.linalg.vector_norm(result, dim=-1) <= 1e-12).any()):
        raise ValueError(f"{label} contains a zero-norm row")
    return result


def _validated_pair_rows(
    pair_region_indices: object,
    *,
    region_count: int,
    pair_count: int,
) -> torch.Tensor:
    rows = torch.as_tensor(pair_region_indices)
    if (
        rows.device.type != "cpu"
        or rows.dtype != torch.int64
        or rows.shape != (pair_count,)
        or bool((rows < 0).any())
        or bool((rows >= region_count).any())
    ):
        raise ValueError("pair_region_indices must be valid CPU int64 region rows")
    counts = torch.bincount(rows, minlength=region_count)
    if bool((counts <= 0).any()):
        raise ValueError("every region requires at least one teacher observation")
    return rows.contiguous()


def build_multiview_teacher_targets(
    pair_descriptors: torch.Tensor,
    pair_region_indices: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    region_count: int,
    temperature: float = TEACHER_RESPONSE_TEMPERATURE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build consensus descriptors and log-mean-exp query responses.

    Returns ``(consensus, response, view_counts)``.  Query responses preserve
    strong evidence in any official source view without increasing merely
    because a region has more selected observations.
    """

    descriptors = _finite_float_matrix(pair_descriptors, label="pair_descriptors")
    text = _finite_float_matrix(text_embeddings, label="text_embeddings")
    if descriptors.shape[1] != text.shape[1]:
        raise ValueError("teacher descriptor and text dimensions differ")
    if not isinstance(region_count, int) or region_count <= 0:
        raise ValueError("region_count must be a positive integer")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    rows = _validated_pair_rows(
        pair_region_indices,
        region_count=region_count,
        pair_count=int(descriptors.shape[0]),
    )
    unit_descriptors = F.normalize(descriptors, dim=-1)
    unit_text = F.normalize(text, dim=-1)
    counts = torch.bincount(rows, minlength=region_count).long()

    consensus_sum = torch.zeros(
        (region_count, descriptors.shape[1]), dtype=torch.float32
    )
    consensus_sum.index_add_(0, rows, unit_descriptors)
    consensus = F.normalize(consensus_sum, dim=-1)

    pair_response = unit_descriptors @ unit_text.T
    response = torch.empty(
        (region_count, text.shape[0]), dtype=torch.float32
    )
    # The sealed sparse teacher is sorted by region, but selecting by equality
    # keeps this helper correct and testable without relying on that detail.
    for region in range(region_count):
        values = pair_response[rows == region]
        response[region] = float(temperature) * (
            torch.logsumexp(values / float(temperature), dim=0)
            - math.log(int(values.shape[0]))
        )
    if not bool(torch.isfinite(response).all()):
        raise ValueError("multi-view teacher responses are nonfinite")
    return consensus.contiguous(), response.contiguous(), counts.contiguous()


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        stop = start + 1
        while stop < values.shape[0] and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(student: np.ndarray, teacher: np.ndarray) -> float | None:
    teacher_rank = _average_ranks(np.asarray(teacher, dtype=np.float64))
    teacher_rank -= teacher_rank.mean()
    teacher_norm = float(np.linalg.norm(teacher_rank))
    if teacher_norm <= 1e-12:
        return None
    student_rank = _average_ranks(np.asarray(student, dtype=np.float64))
    student_rank -= student_rank.mean()
    student_norm = float(np.linalg.norm(student_rank))
    if student_norm <= 1e-12:
        return 0.0
    return float(
        np.clip(
            np.dot(student_rank, teacher_rank) / (student_norm * teacher_norm),
            -1.0,
            1.0,
        )
    )


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "p05": None, "median": None, "p95": None}
    if not np.isfinite(array).all():
        raise ValueError("metric distribution contains nonfinite values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _stable_descending_order(values: np.ndarray) -> np.ndarray:
    indices = np.arange(values.shape[0], dtype=np.int64)
    return np.lexsort((indices, -np.asarray(values, dtype=np.float64)))


def evaluate_source_query_response(
    accepted_descriptors: torch.Tensor,
    teacher_response: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    scale_indices: torch.Tensor,
    view_counts: torch.Tensor,
) -> dict[str, Any]:
    """Evaluate source-only response profiles and within-scene rankings."""

    accepted = _finite_float_matrix(
        accepted_descriptors, label="accepted_descriptors"
    )
    text = _finite_float_matrix(text_embeddings, label="text_embeddings")
    target = _finite_float_matrix(teacher_response, label="teacher_response")
    if accepted.shape[1] != text.shape[1]:
        raise ValueError("AcceptedV2 and text dimensions differ")
    if target.shape != (accepted.shape[0], text.shape[0]):
        raise ValueError("teacher_response must have shape [regions,queries]")
    scales = torch.as_tensor(scale_indices).long().cpu().contiguous()
    views = torch.as_tensor(view_counts).long().cpu().contiguous()
    if scales.shape != (accepted.shape[0],) or views.shape != scales.shape:
        raise ValueError("scale_indices and view_counts must align with regions")
    if bool((views <= 0).any()):
        raise ValueError("view_counts must be positive")

    student = F.normalize(accepted, dim=-1) @ F.normalize(text, dim=-1).T
    error = student - target
    profile = F.cosine_similarity(student, target, dim=-1, eps=1e-12)
    student_centered = student - student.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    centered_profile = F.cosine_similarity(
        student_centered, target_centered, dim=-1, eps=1e-12
    )
    mae = error.abs().mean(dim=1)
    smooth_l1 = F.smooth_l1_loss(student, target, reduction="none").mean(dim=1)

    rankings: list[float] = []
    top1: list[float] = []
    reciprocal_ranks: list[float] = []
    overlaps: dict[str, list[float]] = {
        "top5": [],
        "top10": [],
        "top_decile": [],
    }
    student_np = student.numpy()
    target_np = target.numpy()
    region_count = int(student.shape[0])
    top_sizes = {
        "top5": min(5, region_count),
        "top10": min(10, region_count),
        "top_decile": max(1, int(math.ceil(0.10 * region_count))),
    }
    for query in range(text.shape[0]):
        student_values = student_np[:, query]
        target_values = target_np[:, query]
        rank = _spearman(student_values, target_values)
        if rank is not None:
            rankings.append(rank)
        student_order = _stable_descending_order(student_values)
        target_order = _stable_descending_order(target_values)
        teacher_top = int(target_order[0])
        top1.append(float(int(student_order[0]) == teacher_top))
        student_position = int(np.flatnonzero(student_order == teacher_top)[0])
        reciprocal_ranks.append(1.0 / float(student_position + 1))
        for name, count in top_sizes.items():
            overlap = len(
                set(student_order[:count].tolist())
                & set(target_order[:count].tolist())
            )
            overlaps[name].append(float(overlap / count))

    def rows_for(mask: torch.Tensor) -> dict[str, Any]:
        selected = torch.nonzero(mask, as_tuple=False).flatten()
        return {
            "regions": int(selected.numel()),
            "response_profile_cosine": _distribution(profile[selected].tolist()),
            "centered_response_profile_cosine": _distribution(
                centered_profile[selected].tolist()
            ),
            "response_mae": _distribution(mae[selected].tolist()),
            "response_smooth_l1": _distribution(smooth_l1[selected].tolist()),
        }

    by_scale = {
        str(int(scale)): rows_for(scales == int(scale))
        for scale in sorted(set(scales.tolist()))
    }
    by_view_count = {
        str(int(count)): rows_for(views == int(count))
        for count in sorted(set(views.tolist()))
    }
    return {
        "contract": {
            "student": "l2_normalize(immutable AcceptedV2 e0) @ l2_normalize(text).T",
            "teacher": "official multiview temperature-0.05 log-mean-exp response",
            "query_coupling": "independent_no_softmax",
            "rank_axis": "all regions in one scene per generic query",
            "rank_tie_break": "ascending canonical region row",
        },
        "counts": {
            "regions": region_count,
            "queries": int(text.shape[0]),
            "response_cells": int(region_count * text.shape[0]),
            "valid_rank_queries": len(rankings),
        },
        "row_metrics": rows_for(torch.ones(region_count, dtype=torch.bool)),
        "by_scale_index": by_scale,
        "by_view_count": by_view_count,
        "query_rank_metrics": {
            "spearman": _distribution(rankings),
            "top1_agreement": _distribution(top1),
            "top5_overlap": _distribution(overlaps["top5"]),
            "top10_overlap": _distribution(overlaps["top10"]),
            "top_decile_overlap": _distribution(overlaps["top_decile"]),
            "teacher_top1_student_reciprocal_rank": _distribution(
                reciprocal_ranks
            ),
        },
        "student_response_sha256": tensor_sha256(student),
        "teacher_response_sha256": tensor_sha256(target),
    }


def _active_region_tokens(
    region_rows: torch.Tensor, token_mask: torch.Tensor
) -> tuple[list[set[int]], dict[int, list[int]]]:
    rows = torch.as_tensor(region_rows).long().cpu()
    mask = torch.as_tensor(token_mask).bool().cpu()
    if rows.ndim != 2 or mask.shape != rows.shape or rows.shape[0] <= 0:
        raise ValueError("region_rows/token_mask must be aligned nonempty [R,T]")
    region_tokens: list[set[int]] = []
    inverted: dict[int, list[int]] = defaultdict(list)
    for region in range(rows.shape[0]):
        tokens = set(int(value) for value in rows[region][mask[region]].tolist())
        if not tokens or min(tokens) < 0:
            raise ValueError("every region requires nonnegative active primitive tokens")
        region_tokens.append(tokens)
        for token in sorted(tokens):
            inverted[token].append(region)
    return region_tokens, dict(inverted)


def _top_indices(
    scores: np.ndarray,
    allowed: np.ndarray,
    *,
    count: int,
) -> list[int]:
    candidates = np.flatnonzero(allowed)
    if candidates.size == 0:
        return []
    order = np.lexsort((candidates, -scores[candidates]))
    return [int(value) for value in candidates[order[:count]]]


def mine_scene_global_hard_negatives(
    teacher_consensus: torch.Tensor,
    teacher_fit_response: torch.Tensor,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    scale_indices: torch.Tensor,
    teacher_cosine_min: float = TEACHER_COSINE_MIN,
    teacher_cosine_max: float = TEACHER_COSINE_MAX,
    per_source_k: int = PER_SOURCE_K,
    block_rows: int = MATRIX_BLOCK_ROWS,
) -> dict[str, torch.Tensor]:
    """Mine directed spatially-disjoint confusers without a dense scene matrix."""

    consensus = F.normalize(
        _finite_float_matrix(teacher_consensus, label="teacher_consensus"),
        dim=-1,
    )
    response = _finite_float_matrix(
        teacher_fit_response, label="teacher_fit_response"
    )
    if response.shape[0] != consensus.shape[0]:
        raise ValueError("teacher response and consensus rows differ")
    response = response - response.mean(dim=1, keepdim=True)
    if bool((torch.linalg.vector_norm(response, dim=-1) <= 1e-12).any()):
        raise ValueError("a centered teacher response profile is constant")
    response = F.normalize(response, dim=-1)
    scales = torch.as_tensor(scale_indices).long().cpu().contiguous()
    if scales.shape != (consensus.shape[0],):
        raise ValueError("scale_indices must align with regions")
    if (
        not math.isfinite(float(teacher_cosine_min))
        or not math.isfinite(float(teacher_cosine_max))
        or not -1 <= teacher_cosine_min < teacher_cosine_max <= 1
    ):
        raise ValueError("teacher cosine interval is invalid")
    if not isinstance(per_source_k, int) or per_source_k <= 0:
        raise ValueError("per_source_k must be positive")
    if not isinstance(block_rows, int) or block_rows <= 0:
        raise ValueError("block_rows must be positive")

    region_tokens, inverted = _active_region_tokens(region_rows, token_mask)
    region_count = int(consensus.shape[0])
    records: list[tuple[int, int, int, int, int, float, float]] = []
    all_consensus_t = consensus.T.contiguous()
    all_response_t = response.T.contiguous()
    for start in range(0, region_count, block_rows):
        stop = min(start + block_rows, region_count)
        teacher_block = (consensus[start:stop] @ all_consensus_t).numpy()
        response_block = (response[start:stop] @ all_response_t).numpy()
        for local, anchor in enumerate(range(start, stop)):
            allowed = np.ones(region_count, dtype=bool)
            blocked = {anchor}
            for token in region_tokens[anchor]:
                blocked.update(inverted[token])
            allowed[np.fromiter(sorted(blocked), dtype=np.int64)] = False

            teacher_allowed = allowed & (
                teacher_block[local] >= float(teacher_cosine_min)
            ) & (teacher_block[local] <= float(teacher_cosine_max))
            teacher_top = _top_indices(
                teacher_block[local], teacher_allowed, count=per_source_k
            )
            response_top = _top_indices(
                response_block[local], allowed, count=per_source_k
            )
            selected: dict[int, list[int]] = {}
            for rank, negative in enumerate(teacher_top):
                selected.setdefault(negative, [-1, -1])[0] = rank
            for rank, negative in enumerate(response_top):
                selected.setdefault(negative, [-1, -1])[1] = rank
            for negative in sorted(selected):
                teacher_rank, response_rank = selected[negative]
                source_code = (1 if teacher_rank >= 0 else 0) | (
                    2 if response_rank >= 0 else 0
                )
                if region_tokens[anchor] & region_tokens[negative]:
                    raise AssertionError("miner selected a spatially-overlapping row")
                records.append(
                    (
                        anchor,
                        negative,
                        source_code,
                        teacher_rank,
                        response_rank,
                        float(teacher_block[local, negative]),
                        float(response_block[local, negative]),
                    )
                )

    if not records:
        raise ValueError("hard-negative miner found no spatially-distinct confusers")
    anchor = torch.tensor([row[0] for row in records], dtype=torch.int64)
    negative = torch.tensor([row[1] for row in records], dtype=torch.int64)
    source_code = torch.tensor([row[2] for row in records], dtype=torch.int64)
    teacher_rank = torch.tensor([row[3] for row in records], dtype=torch.int64)
    response_rank = torch.tensor([row[4] for row in records], dtype=torch.int64)
    teacher_cosine = torch.tensor([row[5] for row in records], dtype=torch.float32)
    response_cosine = torch.tensor([row[6] for row in records], dtype=torch.float32)
    counts = torch.bincount(anchor, minlength=region_count)
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.cumsum(counts, dim=0)]
    )
    return {
        "anchor_region_indices": anchor,
        "negative_region_indices": negative,
        "source_codes": source_code,
        "teacher_similarity_ranks": teacher_rank,
        "response_nearest_ranks": response_rank,
        "teacher_cosines": teacher_cosine,
        "response_profile_cosines": response_cosine,
        "anchor_scale_indices": scales[anchor],
        "negative_scale_indices": scales[negative],
        "row_offsets": offsets,
    }


def build_negative_authority(
    *,
    scene_id: str,
    canonical_region_indices: torch.Tensor,
    region_fingerprints: Sequence[str],
    channels: Mapping[str, torch.Tensor],
    input_authority: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = torch.as_tensor(canonical_region_indices).long().cpu().contiguous()
    fingerprints = [str(value) for value in region_fingerprints]
    if canonical.ndim != 1 or len(fingerprints) != canonical.numel():
        raise ValueError("canonical region identity is misaligned")
    contract = {
        "teacher_response_temperature": TEACHER_RESPONSE_TEMPERATURE,
        "teacher_cosine_interval_inclusive": [
            TEACHER_COSINE_MIN,
            TEACHER_COSINE_MAX,
        ],
        "per_source_k": PER_SOURCE_K,
        "matrix_block_rows": MATRIX_BLOCK_ROWS,
        "spatially_distinct": "zero_shared_active_primitive_tokens",
        "sources": {
            "1": "teacher_consensus_similarity_band",
            "2": "centered_fit_query_response_nearest",
            "3": "selected_by_both",
        },
        "directed_pairs": True,
        "dense_cross_scene_matrix": False,
    }
    source_access = {
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_text_queries_opened": False,
        "generic_target_blind_text_bank_opened": True,
        "target_heldout_opened": False,
        "target_metrics_computed": False,
        "text_queries_opened": False,
    }
    source_access_semantics = {
        "text_queries_opened": "legacy field meaning benchmark text queries only",
        "generic_target_blind_text_bank_opened": (
            "frozen non-benchmark ImageNet-primary text embeddings were opened"
        ),
    }
    tensor_channels = {
        str(key): torch.as_tensor(value).detach().cpu().contiguous()
        for key, value in channels.items()
    }
    channel_sha = {key: tensor_sha256(value) for key, value in tensor_channels.items()}
    metadata_for_hash = {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "scene_id": str(scene_id),
        "contract": contract,
        "canonical_region_indices_sha256": tensor_sha256(canonical),
        "region_fingerprints": fingerprints,
        "channel_sha256": channel_sha,
        "input_authority": dict(input_authority),
        "source_access": source_access,
        "source_access_semantics": source_access_semantics,
    }
    payload: dict[str, Any] = {
        **metadata_for_hash,
        "physical_space_id": str(scene_id).split("_")[0],
        "canonical_region_indices": canonical,
        "channels": tensor_channels,
        "content_authority_sha256": canonical_json_sha256(metadata_for_hash),
    }
    return validate_negative_authority(payload)


def validate_negative_authority(
    value: object,
    *,
    region_rows: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("negative authority must be a mapping")
    payload = dict(value)
    required = {
        "schema",
        "schema_version",
        "scene_id",
        "physical_space_id",
        "contract",
        "canonical_region_indices_sha256",
        "region_fingerprints",
        "channel_sha256",
        "input_authority",
        "canonical_region_indices",
        "channels",
        "content_authority_sha256",
        "source_access",
        "source_access_semantics",
    }
    if (
        set(payload) != required
        or payload.get("schema") != AUTHORITY_SCHEMA
        or payload.get("schema_version") != AUTHORITY_SCHEMA_VERSION
    ):
        raise ValueError("negative authority schema differs")
    canonical = torch.as_tensor(payload["canonical_region_indices"])
    fingerprints = payload.get("region_fingerprints")
    channels_value = payload.get("channels")
    if (
        canonical.device.type != "cpu"
        or canonical.dtype != torch.int64
        or canonical.ndim != 1
        or canonical.numel() <= 0
        or bool((canonical < 0).any())
        or (
            canonical.numel() > 1
            and not bool((canonical[1:] > canonical[:-1]).all())
        )
    ):
        raise ValueError("negative authority canonical rows differ")
    if (
        not isinstance(fingerprints, list)
        or len(fingerprints) != canonical.numel()
        or any(not isinstance(item, str) or not item for item in fingerprints)
        or len(set(fingerprints)) != len(fingerprints)
    ):
        raise ValueError("negative authority region fingerprints differ")
    scene_id = payload.get("scene_id")
    if (
        not isinstance(scene_id, str)
        or not scene_id
        or payload.get("physical_space_id") != scene_id.split("_")[0]
        or not isinstance(payload.get("input_authority"), Mapping)
        or not payload["input_authority"]
    ):
        raise ValueError("negative authority scene or input identity differs")
    if not isinstance(channels_value, Mapping):
        raise ValueError("negative authority channels must be a mapping")
    channel_names = {
        "anchor_region_indices",
        "negative_region_indices",
        "source_codes",
        "teacher_similarity_ranks",
        "response_nearest_ranks",
        "teacher_cosines",
        "response_profile_cosines",
        "anchor_scale_indices",
        "negative_scale_indices",
        "row_offsets",
    }
    if set(channels_value) != channel_names:
        raise ValueError("negative authority channel declaration differs")
    channels = {key: torch.as_tensor(item).detach().cpu().contiguous() for key, item in channels_value.items()}
    pair_count = int(channels["anchor_region_indices"].numel())
    for name in channel_names - {"row_offsets"}:
        if channels[name].ndim != 1 or channels[name].numel() != pair_count:
            raise ValueError("negative authority pair channels are misaligned")
    integer_names = channel_names - {
        "teacher_cosines",
        "response_profile_cosines",
    }
    if any(channels[name].dtype != torch.int64 for name in integer_names):
        raise ValueError("negative authority integer channel dtype differs")
    if any(
        channels[name].dtype != torch.float32
        for name in ("teacher_cosines", "response_profile_cosines")
    ):
        raise ValueError("negative authority score channel dtype differs")
    anchors = channels["anchor_region_indices"]
    negatives = channels["negative_region_indices"]
    if (
        pair_count <= 0
        or bool((anchors < 0).any())
        or bool((negatives < 0).any())
        or bool((anchors >= canonical.numel()).any())
        or bool((negatives >= canonical.numel()).any())
        or bool((anchors == negatives).any())
    ):
        raise ValueError("negative authority pair indices are invalid")
    if bool((anchors[1:] < anchors[:-1]).any()):
        raise ValueError("negative authority anchors are not sorted")
    directed_pairs = list(zip(anchors.tolist(), negatives.tolist()))
    if len(set(directed_pairs)) != pair_count:
        raise ValueError("negative authority contains duplicate directed pairs")
    for start in range(pair_count - 1):
        if anchors[start] == anchors[start + 1] and negatives[start] >= negatives[start + 1]:
            raise ValueError("negative authority rows are not canonically ordered")
    offsets = channels["row_offsets"]
    expected_counts = torch.bincount(anchors, minlength=canonical.numel())
    expected_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.cumsum(expected_counts, dim=0)]
    )
    if not torch.equal(offsets, expected_offsets):
        raise ValueError("negative authority row offsets differ")
    codes = channels["source_codes"]
    teacher_ranks = channels["teacher_similarity_ranks"]
    response_ranks = channels["response_nearest_ranks"]
    contract = payload.get("contract")
    if contract != {
        "teacher_response_temperature": TEACHER_RESPONSE_TEMPERATURE,
        "teacher_cosine_interval_inclusive": [TEACHER_COSINE_MIN, TEACHER_COSINE_MAX],
        "per_source_k": PER_SOURCE_K,
        "matrix_block_rows": MATRIX_BLOCK_ROWS,
        "spatially_distinct": "zero_shared_active_primitive_tokens",
        "sources": {
            "1": "teacher_consensus_similarity_band",
            "2": "centered_fit_query_response_nearest",
            "3": "selected_by_both",
        },
        "directed_pairs": True,
        "dense_cross_scene_matrix": False,
    }:
        raise ValueError("negative authority contract differs")
    if (
        bool(((codes < 1) | (codes > 3)).any())
        or bool(((codes & 1 > 0) != (teacher_ranks >= 0)).any())
        or bool(((codes & 2 > 0) != (response_ranks >= 0)).any())
        or bool((teacher_ranks >= PER_SOURCE_K).any())
        or bool((response_ranks >= PER_SOURCE_K).any())
    ):
        raise ValueError("negative authority source/rank channels differ")
    teacher_selected = (codes & 1) > 0
    teacher_scores = channels["teacher_cosines"][teacher_selected]
    if bool(
        ((teacher_scores < TEACHER_COSINE_MIN - 1e-6) | (teacher_scores > TEACHER_COSINE_MAX + 1e-6)).any()
    ):
        raise ValueError("teacher-similarity negative lies outside the frozen band")
    if not bool(torch.isfinite(channels["teacher_cosines"]).all()) or not bool(
        torch.isfinite(channels["response_profile_cosines"]).all()
    ):
        raise ValueError("negative authority scores are nonfinite")
    if bool((channels["teacher_cosines"].abs() > 1.0 + 1e-5).any()) or bool(
        (channels["response_profile_cosines"].abs() > 1.0 + 1e-5).any()
    ):
        raise ValueError("negative authority cosine score is out of range")
    if region_rows is not None or token_mask is not None:
        if region_rows is None or token_mask is None:
            raise ValueError("region_rows and token_mask must be supplied together")
        region_tokens, _ = _active_region_tokens(region_rows, token_mask)
        for anchor, negative in zip(anchors.tolist(), negatives.tolist()):
            if region_tokens[anchor] & region_tokens[negative]:
                raise ValueError("negative authority contains spatial overlap")
    declared_sha = payload.get("channel_sha256")
    actual_sha = {key: tensor_sha256(item) for key, item in channels.items()}
    if declared_sha != actual_sha:
        raise ValueError("negative authority channel hashes differ")
    metadata_for_hash = {
        "schema": payload["schema"],
        "schema_version": payload["schema_version"],
        "scene_id": payload["scene_id"],
        "contract": payload["contract"],
        "canonical_region_indices_sha256": payload[
            "canonical_region_indices_sha256"
        ],
        "region_fingerprints": payload["region_fingerprints"],
        "channel_sha256": payload["channel_sha256"],
        "input_authority": payload["input_authority"],
        "source_access": payload["source_access"],
        "source_access_semantics": payload["source_access_semantics"],
    }
    expected_source_access = {
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_text_queries_opened": False,
        "generic_target_blind_text_bank_opened": True,
        "target_heldout_opened": False,
        "target_metrics_computed": False,
        "text_queries_opened": False,
    }
    expected_source_access_semantics = {
        "text_queries_opened": "legacy field meaning benchmark text queries only",
        "generic_target_blind_text_bank_opened": (
            "frozen non-benchmark ImageNet-primary text embeddings were opened"
        ),
    }
    if (
        payload.get("canonical_region_indices_sha256") != tensor_sha256(canonical)
        or payload.get("content_authority_sha256")
        != canonical_json_sha256(metadata_for_hash)
        or payload.get("source_access") != expected_source_access
        or payload.get("source_access_semantics")
        != expected_source_access_semantics
    ):
        raise ValueError("negative authority identity or source-access seal differs")
    payload["canonical_region_indices"] = canonical.contiguous()
    payload["channels"] = channels
    return payload


__all__ = [
    "AUTHORITY_SCHEMA",
    "MATRIX_BLOCK_ROWS",
    "PER_SOURCE_K",
    "TEACHER_COSINE_MAX",
    "TEACHER_COSINE_MIN",
    "TEACHER_RESPONSE_TEMPERATURE",
    "build_multiview_teacher_targets",
    "build_negative_authority",
    "evaluate_source_query_response",
    "mine_scene_global_hard_negatives",
    "validate_negative_authority",
]
