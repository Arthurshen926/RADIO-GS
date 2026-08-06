#!/usr/bin/env python3
"""Audit FP16 LERF score-cache quantization against source-exact FP32 cosine.

This is a read-only, benchmark-closed numeric audit.  It reconstructs scores
from the descriptor cache and the text artifacts named by each score cache,
then reuses the frozen canonical-negative and VALA readout implementation.  It
does not expose a learnable parameter, threshold sweep, or scene calibration.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.evaluation.openclip_readout import cosine_logits
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    SelectionSpec,
    canonical_negative_relevancy_query_scores,
    select_gaussians_from_scores,
    vala_multiscale_knn_peak_select_scores,
)
from radio_gs.utils.immutable_artifacts import sha256_file, write_frozen_json


AUDIT_SCHEMA_VERSION = 1
SCORE_CONTRACT = "radio_gs.ours_lerf_direct3d_multiscale_query_scores.v2"
AUTHORITY_CONTRACT = "radio_gs.lerf_multiscale_query_score_authority.v2"
FORMULA = "l2_normalize(descriptor) @ l2_normalize(text_embedding).T"
RANK_ABSOLUTE_CUTOFFS = (1, 10, 100)
RANK_FRACTION_CUTOFFS = (0.01, 0.05)
RELEVANCY_LOGIT_SCALE = 10.0
VALA_THRESHOLD = 0.6


@lru_cache(maxsize=32)
def _cached_sha256(path: str) -> str:
    return sha256_file(Path(path))


def _observed_sha256(path: Path) -> str:
    return _cached_sha256(str(path.expanduser().resolve(strict=True)))


def _load_mapping(path: Path, *, expected_sha256: str, label: str) -> Mapping[str, Any]:
    observed = _observed_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"{label} SHA256 differs: expected {expected_sha256}, got {observed}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return payload


def _require_score_cache(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if payload.get("version") != 2 or payload.get("contract") != SCORE_CONTRACT:
        raise ValueError(f"{label} score-cache contract differs")
    authority = payload.get("authority")
    if not isinstance(authority, Mapping) or authority.get("contract") != AUTHORITY_CONTRACT:
        raise ValueError(f"{label} authority contract differs")
    if authority.get("score_semantics") != "raw_independent_normalized_cosine":
        raise ValueError(f"{label} score semantics differ")
    if authority.get("score_formula") != FORMULA:
        raise ValueError(f"{label} score formula differs")
    if authority.get("score_dtype") != "torch.float16":
        raise ValueError(f"{label} authority is not the formal FP16 cache")
    scores = payload.get("query_scores")
    xyz = payload.get("xyz")
    valid = payload.get("valid")
    query_ids = payload.get("query_ids")
    if not isinstance(scores, torch.Tensor) or scores.dtype != torch.float16:
        raise ValueError(f"{label} query_scores must be FP16")
    if scores.ndim != 3 or scores.shape[1] != 3:
        raise ValueError(f"{label} query_scores must be [N,3,Q]")
    if not isinstance(xyz, torch.Tensor) or tuple(xyz.shape) != (scores.shape[0], 3):
        raise ValueError(f"{label} xyz does not align")
    if (
        not isinstance(valid, torch.Tensor)
        or valid.dtype != torch.bool
        or tuple(valid.shape) != (scores.shape[0],)
        or not bool(valid.any())
    ):
        raise ValueError(f"{label} valid mask does not align")
    if (
        not isinstance(query_ids, (list, tuple))
        or len(query_ids) != scores.shape[2]
        or len(set(query_ids)) != len(query_ids)
        or not all(isinstance(value, str) and value for value in query_ids)
    ):
        raise ValueError(f"{label} query IDs do not align")
    sources = authority.get("source_artifacts")
    if not isinstance(sources, Mapping):
        raise ValueError(f"{label} authority lacks source artifacts")
    return {
        "scores": scores.detach().cpu().contiguous(),
        "xyz": xyz.detach().cpu().float().contiguous(),
        "valid": valid.detach().cpu().contiguous(),
        "query_ids": tuple(query_ids),
        "scale_ids": tuple(payload.get("scale_ids", ())),
        "scale_radii_m": tuple(payload.get("scale_radii_m", ())),
        "sources": sources,
        "field_checkpoint_sha256": payload.get("field_checkpoint_sha256"),
        "readout_checkpoint_sha256": payload.get("readout_checkpoint_sha256"),
        "renderer_geometry_checkpoint_sha256": payload.get(
            "renderer_geometry_checkpoint_sha256"
        ),
    }


def _source_record(cache: Mapping[str, Any], role: str) -> tuple[Path, str]:
    record = cache["sources"].get(role)
    if not isinstance(record, Mapping):
        raise ValueError(f"score-cache source artifact {role} is missing")
    path = record.get("path")
    digest = record.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise ValueError(f"score-cache source artifact {role} is malformed")
    source = Path(path).expanduser().resolve(strict=True)
    if _observed_sha256(source) != digest:
        raise ValueError(f"score-cache source artifact {role} SHA256 differs")
    return source, digest


def _require_paired_caches(positive: Mapping[str, Any], negative: Mapping[str, Any]) -> None:
    for field in (
        "valid",
        "xyz",
        "scale_ids",
        "scale_radii_m",
        "field_checkpoint_sha256",
        "readout_checkpoint_sha256",
        "renderer_geometry_checkpoint_sha256",
    ):
        left = positive[field]
        right = negative[field]
        equal = torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
        if not bool(equal):
            raise ValueError(f"positive/negative score-cache {field} differs")


def _require_descriptor(
    payload: Mapping[str, Any],
    *,
    expected_rows: int,
    expected_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    descriptors = payload.get("features_by_scale")
    global_rows = payload.get("global_rows")
    if not isinstance(descriptors, torch.Tensor) or descriptors.ndim != 3:
        raise ValueError("descriptor features_by_scale must be [M,3,D]")
    if descriptors.shape[1] != 3:
        raise ValueError("descriptor cache must have exactly three native scales")
    if not isinstance(global_rows, torch.Tensor) or global_rows.dtype != torch.int64:
        raise ValueError("descriptor global_rows must be int64")
    rows = global_rows.detach().cpu().reshape(-1)
    if descriptors.shape[0] != rows.numel():
        raise ValueError("descriptor features/global_rows differ")
    if not torch.equal(rows, torch.where(expected_valid)[0]):
        raise ValueError("descriptor global_rows differ from score-cache valid rows")
    xyz = payload.get("xyz")
    if not isinstance(xyz, torch.Tensor) or tuple(xyz.shape) != (expected_rows, 3):
        raise ValueError("descriptor xyz does not align with score cache")
    values = descriptors.detach().cpu().contiguous()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("descriptor contains non-finite values")
    return values, rows


def _select_text_rows(
    payload: Mapping[str, Any], query_ids: Sequence[str], *, label: str
) -> torch.Tensor:
    raw_ids = payload.get("queries")
    embeddings = payload.get("embeddings")
    if not isinstance(raw_ids, (list, tuple)) or not all(
        isinstance(value, str) and value for value in raw_ids
    ):
        raise ValueError(f"{label} text query IDs are malformed")
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError(f"{label} text query IDs are not unique")
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
        or embeddings.shape[0] != len(raw_ids)
    ):
        raise ValueError(f"{label} text embeddings do not align")
    row_by_id = {value: index for index, value in enumerate(raw_ids)}
    missing = [value for value in query_ids if value not in row_by_id]
    if missing:
        raise ValueError(f"{label} text cache misses queries: {missing}")
    selected = embeddings[[row_by_id[value] for value in query_ids]].detach().cpu()
    if not bool(torch.isfinite(selected).all()):
        raise ValueError(f"{label} text embeddings contain non-finite values")
    return selected.contiguous()


def compile_fp32_scores(
    descriptors: torch.Tensor,
    global_rows: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    total_rows: int,
    chunk_size: int,
) -> torch.Tensor:
    """Reproduce formal normalized cosine without its final FP16 cast."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output = torch.zeros(
        total_rows, 3, int(text_embeddings.shape[0]), dtype=torch.float32
    )
    for scale in range(3):
        for start in range(0, int(global_rows.numel()), chunk_size):
            stop = min(start + chunk_size, int(global_rows.numel()))
            output[global_rows[start:stop], scale] = cosine_logits(
                descriptors[start:stop, scale], text_embeddings
            )
    return output.contiguous()


def error_summary(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise ValueError("error-summary tensors differ in shape")
    difference = (candidate.detach().float() - reference.detach().float()).abs().reshape(-1)
    quantiles = torch.quantile(
        difference, torch.tensor([0.5, 0.95, 0.99], dtype=difference.dtype)
    )
    return {
        "elements": int(difference.numel()),
        "nonzero_elements": int(torch.count_nonzero(difference)),
        "mean_absolute_error": float(difference.mean()),
        "p50_absolute_error": float(quantiles[0]),
        "p95_absolute_error": float(quantiles[1]),
        "p99_absolute_error": float(quantiles[2]),
        "max_absolute_error": float(difference.max()),
    }


def _rank_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate = candidate.detach().float().reshape(-1)
    reference = reference.detach().float().reshape(-1)
    count = int(reference.numel())
    candidate_order = torch.argsort(candidate, descending=True, stable=True)
    reference_order = torch.argsort(reference, descending=True, stable=True)
    candidate_rank = torch.empty(count, dtype=torch.int64)
    reference_rank = torch.empty(count, dtype=torch.int64)
    positions = torch.arange(count, dtype=torch.int64)
    candidate_rank[candidate_order] = positions
    reference_rank[reference_order] = positions
    displacement = (candidate_rank - reference_rank).abs().float()
    centered_candidate = candidate_rank.float() - candidate_rank.float().mean()
    centered_reference = reference_rank.float() - reference_rank.float().mean()
    denominator = centered_candidate.norm() * centered_reference.norm()
    spearman_raw = (
        float((centered_candidate * centered_reference).sum() / denominator)
        if float(denominator) > 0.0
        else 1.0
    )
    spearman = min(max(spearman_raw, -1.0), 1.0)
    cutoffs = set(min(count, value) for value in RANK_ABSOLUTE_CUTOFFS)
    cutoffs.update(
        min(count, max(1, int(round(count * value))))
        for value in RANK_FRACTION_CUTOFFS
    )
    topk = {}
    for cutoff in sorted(cutoffs):
        candidate_top = torch.zeros(count, dtype=torch.bool)
        reference_top = torch.zeros(count, dtype=torch.bool)
        candidate_top[candidate_order[:cutoff]] = True
        reference_top[reference_order[:cutoff]] = True
        overlap = int((candidate_top & reference_top).sum())
        topk[str(cutoff)] = {
            "overlap": overlap,
            "membership_flips": int((candidate_top != reference_top).sum()),
            "overlap_fraction": float(overlap / cutoff),
        }
    return {
        "exact_stable_rank_fraction": float((displacement == 0).float().mean()),
        "mean_absolute_rank_displacement": float(displacement.mean()),
        "p95_absolute_rank_displacement": float(torch.quantile(displacement, 0.95)),
        "max_absolute_rank_displacement": int(displacement.max()),
        "stable_rank_spearman": spearman,
        "topk": topk,
    }


def per_query_scale_audit(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    valid: torch.Tensor,
    query_ids: Sequence[str],
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for scale in range(3):
        for query, query_id in enumerate(query_ids):
            left = candidate[valid, scale, query]
            right = reference[valid, scale, query]
            row: dict[str, Any] = {
                "query_id": query_id,
                "scale_index": scale,
                "error": error_summary(left, right),
                "rank": _rank_metrics(left, right),
            }
            if threshold is not None:
                left_selected = left > float(threshold)
                right_selected = right > float(threshold)
                row["threshold"] = {
                    "value": float(threshold),
                    "candidate_selected": int(left_selected.sum()),
                    "reference_selected": int(right_selected.sum()),
                    "membership_flips": int((left_selected != right_selected).sum()),
                }
            rows.append(row)
    return rows


def _text_bank_comparison(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    return {
        "raw_embedding_error": error_summary(candidate, reference),
        "bit_exact": bool(torch.equal(candidate, reference)),
        "fp32_normalized_bit_exact": bool(
            torch.equal(
                torch.nn.functional.normalize(candidate.float(), dim=-1),
                torch.nn.functional.normalize(reference.float(), dim=-1),
            )
        ),
    }


def downstream_audit(
    candidate_positive: torch.Tensor,
    candidate_negative: torch.Tensor,
    reference_positive: torch.Tensor,
    reference_negative: torch.Tensor,
    *,
    xyz: torch.Tensor,
    valid: torch.Tensor,
    query_ids: Sequence[str],
    knn_chunk_size: int,
) -> dict[str, Any]:
    candidate_relevancy = canonical_negative_relevancy_query_scores(
        candidate_positive,
        candidate_negative,
        logit_scale=RELEVANCY_LOGIT_SCALE,
    )
    reference_relevancy = canonical_negative_relevancy_query_scores(
        reference_positive,
        reference_negative,
        logit_scale=RELEVANCY_LOGIT_SCALE,
    )
    relevancy_report = {
        "error": error_summary(candidate_relevancy[valid], reference_relevancy[valid]),
        "per_query_scale": per_query_scale_audit(
            candidate_relevancy,
            reference_relevancy,
            valid=valid,
            query_ids=query_ids,
            threshold=VALA_THRESHOLD,
        ),
    }
    candidate_readout = vala_multiscale_knn_peak_select_scores(
        candidate_relevancy,
        xyz,
        k=10,
        chunk_size=knn_chunk_size,
        valid_mask=valid,
    )
    reference_readout = vala_multiscale_knn_peak_select_scores(
        reference_relevancy,
        xyz,
        k=10,
        chunk_size=knn_chunk_size,
        valid_mask=valid,
    )
    candidate_selected = select_gaussians_from_scores(
        candidate_readout.scores, SelectionSpec("score_threshold", VALA_THRESHOLD), min_select=0
    ).bool()
    reference_selected = select_gaussians_from_scores(
        reference_readout.scores, SelectionSpec("score_threshold", VALA_THRESHOLD), min_select=0
    ).bool()
    per_query_selection = []
    for query, query_id in enumerate(query_ids):
        left = candidate_selected[:, query]
        right = reference_selected[:, query]
        per_query_selection.append(
            {
                "query_id": query_id,
                "candidate_selected": int(left.sum()),
                "reference_selected": int(right.sum()),
                "membership_flips": int((left != right).sum()),
            }
        )
    selection_flips = int((candidate_selected != reference_selected).sum())
    selection_elements = int(valid.sum()) * len(query_ids)
    scale_flips = int(
        (
            candidate_readout.selected_scale_indices
            != reference_readout.selected_scale_indices
        ).sum()
    )
    return {
        "canonical_negative_relevancy": relevancy_report,
        "vala_knn10_peak_select": {
            "candidate_selected_scale_indices": [
                int(value) for value in candidate_readout.selected_scale_indices
            ],
            "reference_selected_scale_indices": [
                int(value) for value in reference_readout.selected_scale_indices
            ],
            "selected_scale_query_flips": scale_flips,
            "final_score_error": error_summary(
                candidate_readout.scores[valid], reference_readout.scores[valid]
            ),
            "threshold": VALA_THRESHOLD,
            "primitive_query_membership_flips": selection_flips,
            "valid_primitive_query_elements": selection_elements,
            "primitive_query_membership_flip_fraction": float(
                selection_flips / max(selection_elements, 1)
            ),
            "per_query_selection": per_query_selection,
            "selection_bit_exact": selection_flips == 0,
        },
        "exact_evaluator_consequence": (
            "identical: frozen evaluator receives the exact same primitive "
            "selection for every query, so selected-only alpha renders and metrics are identical"
            if selection_flips == 0
            else "not provably identical without rendering: at least one frozen primitive selection changed"
        ),
    }


def audit(
    *,
    descriptor_cache: Path,
    descriptor_cache_sha256: str,
    positive_score_cache: Path,
    positive_score_cache_sha256: str,
    negative_score_cache: Path,
    negative_score_cache_sha256: str,
    frozen_positive_text_cache: Path,
    frozen_positive_text_cache_sha256: str,
    frozen_negative_text_cache: Path,
    frozen_negative_text_cache_sha256: str,
    output: Path,
    score_chunk_size: int = 4096,
    knn_chunk_size: int = 65536,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable audit output already exists: {output}")
    positive_payload = _load_mapping(
        positive_score_cache,
        expected_sha256=positive_score_cache_sha256,
        label="positive score cache",
    )
    negative_payload = _load_mapping(
        negative_score_cache,
        expected_sha256=negative_score_cache_sha256,
        label="negative score cache",
    )
    positive = _require_score_cache(positive_payload, label="positive")
    negative = _require_score_cache(negative_payload, label="negative")
    _require_paired_caches(positive, negative)

    descriptor_source, descriptor_source_sha = _source_record(
        positive, "descriptor_cache"
    )
    negative_descriptor_source, negative_descriptor_sha = _source_record(
        negative, "descriptor_cache"
    )
    requested_descriptor = descriptor_cache.expanduser().resolve(strict=True)
    if descriptor_source != requested_descriptor or negative_descriptor_source != requested_descriptor:
        raise ValueError("score caches are not bound to the requested descriptor cache")
    if descriptor_source_sha != descriptor_cache_sha256 or negative_descriptor_sha != descriptor_cache_sha256:
        raise ValueError("score-cache descriptor SHA differs from requested SHA")
    descriptor_payload = _load_mapping(
        requested_descriptor,
        expected_sha256=descriptor_cache_sha256,
        label="descriptor cache",
    )
    descriptors, global_rows = _require_descriptor(
        descriptor_payload,
        expected_rows=int(positive["scores"].shape[0]),
        expected_valid=positive["valid"],
    )

    source_positive_path, source_positive_sha = _source_record(
        positive, "text_query_cache"
    )
    source_negative_path, source_negative_sha = _source_record(
        negative, "text_query_cache"
    )
    source_positive_payload = _load_mapping(
        source_positive_path,
        expected_sha256=source_positive_sha,
        label="authority positive text cache",
    )
    source_negative_payload = _load_mapping(
        source_negative_path,
        expected_sha256=source_negative_sha,
        label="authority negative text cache",
    )
    source_positive_text = _select_text_rows(
        source_positive_payload, positive["query_ids"], label="authority positive"
    )
    source_negative_text = _select_text_rows(
        source_negative_payload, negative["query_ids"], label="authority negative"
    )

    frozen_positive_payload = _load_mapping(
        frozen_positive_text_cache.expanduser().resolve(strict=True),
        expected_sha256=frozen_positive_text_cache_sha256,
        label="frozen positive text cache",
    )
    frozen_negative_payload = _load_mapping(
        frozen_negative_text_cache.expanduser().resolve(strict=True),
        expected_sha256=frozen_negative_text_cache_sha256,
        label="frozen negative text cache",
    )
    frozen_positive_text = _select_text_rows(
        frozen_positive_payload, positive["query_ids"], label="frozen positive"
    )
    frozen_negative_text = _select_text_rows(
        frozen_negative_payload, negative["query_ids"], label="frozen negative"
    )

    total_rows = int(positive["scores"].shape[0])
    source_positive_fp32 = compile_fp32_scores(
        descriptors,
        global_rows,
        source_positive_text,
        total_rows=total_rows,
        chunk_size=score_chunk_size,
    )
    source_negative_fp32 = compile_fp32_scores(
        descriptors,
        global_rows,
        source_negative_text,
        total_rows=total_rows,
        chunk_size=score_chunk_size,
    )
    frozen_positive_fp32 = compile_fp32_scores(
        descriptors,
        global_rows,
        frozen_positive_text,
        total_rows=total_rows,
        chunk_size=score_chunk_size,
    )
    frozen_negative_fp32 = compile_fp32_scores(
        descriptors,
        global_rows,
        frozen_negative_text,
        total_rows=total_rows,
        chunk_size=score_chunk_size,
    )
    del descriptor_payload, descriptors
    gc.collect()

    valid = positive["valid"]
    existing_positive = positive["scores"].float()
    existing_negative = negative["scores"].float()
    pure_dtype_roundtrip = {
        "positive_fp16_cast_bit_exact": bool(
            torch.equal(source_positive_fp32.half(), positive["scores"])
        ),
        "positive_fp16_cast_mismatch_elements": int(
            (source_positive_fp32.half() != positive["scores"]).sum()
        ),
        "negative_fp16_cast_bit_exact": bool(
            torch.equal(source_negative_fp32.half(), negative["scores"])
        ),
        "negative_fp16_cast_mismatch_elements": int(
            (source_negative_fp32.half() != negative["scores"]).sum()
        ),
    }
    raw_source_audit = {
        "positive_error": error_summary(
            existing_positive[valid], source_positive_fp32[valid]
        ),
        "negative_error": error_summary(
            existing_negative[valid], source_negative_fp32[valid]
        ),
        "positive_per_query_scale": per_query_scale_audit(
            existing_positive,
            source_positive_fp32,
            valid=valid,
            query_ids=positive["query_ids"],
        ),
        "negative_per_query_scale": per_query_scale_audit(
            existing_negative,
            source_negative_fp32,
            valid=valid,
            query_ids=negative["query_ids"],
        ),
        "fp16_roundtrip": pure_dtype_roundtrip,
    }
    pure_dtype_downstream = downstream_audit(
        existing_positive,
        existing_negative,
        source_positive_fp32,
        source_negative_fp32,
        xyz=positive["xyz"],
        valid=valid,
        query_ids=positive["query_ids"],
        knn_chunk_size=knn_chunk_size,
    )
    frozen_bank_downstream = downstream_audit(
        existing_positive,
        existing_negative,
        frozen_positive_fp32,
        frozen_negative_fp32,
        xyz=positive["xyz"],
        valid=valid,
        query_ids=positive["query_ids"],
        knn_chunk_size=knn_chunk_size,
    )

    pure_flips = pure_dtype_downstream["vala_knn10_peak_select"][
        "primitive_query_membership_flips"
    ]
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "artifact_type": "radio_gs_lerf_official_score_fp16_fp32_audit",
        "status": "complete_benchmark_closed_numeric_audit",
        "scope": {
            "scene": "figurines",
            "query_independent_method_change": True,
            "benchmark_closed_numeric_audit": True,
            "parameter_or_threshold_tuning": False,
            "benchmark_images_opened": False,
            "benchmark_annotations_opened": False,
            "benchmark_masks_opened": False,
            "execution_device": "cpu",
        },
        "artifacts": {
            "descriptor_cache": {
                "path": str(requested_descriptor),
                "sha256": descriptor_cache_sha256,
            },
            "positive_score_cache": {
                "path": str(positive_score_cache.resolve()),
                "sha256": positive_score_cache_sha256,
            },
            "negative_score_cache": {
                "path": str(negative_score_cache.resolve()),
                "sha256": negative_score_cache_sha256,
            },
            "authority_positive_text_cache": {
                "path": str(source_positive_path),
                "sha256": source_positive_sha,
            },
            "authority_negative_text_cache": {
                "path": str(source_negative_path),
                "sha256": source_negative_sha,
            },
            "frozen_positive_text_cache": {
                "path": str(frozen_positive_text_cache.resolve()),
                "sha256": frozen_positive_text_cache_sha256,
            },
            "frozen_negative_text_cache": {
                "path": str(frozen_negative_text_cache.resolve()),
                "sha256": frozen_negative_text_cache_sha256,
            },
        },
        "fixed_protocol": {
            "formula": FORMULA,
            "relevancy": "sigmoid((positive - max(generic_negatives)) * 10)",
            "vala": "independent kNN10 smoothing, raw-peak scale selection, per-scale min-max, clip(2*x-1,0,1)",
            "primitive_selection": "strict score > 0.6",
            "rank_absolute_cutoffs": list(RANK_ABSOLUTE_CUTOFFS),
            "rank_fraction_cutoffs": list(RANK_FRACTION_CUTOFFS),
        },
        "dimensions": {
            "total_primitives": total_rows,
            "valid_primitives": int(valid.sum()),
            "positive_queries": len(positive["query_ids"]),
            "negative_queries": len(negative["query_ids"]),
            "scales": 3,
        },
        "execution": {
            "torch_version": torch.__version__,
            "torch_num_threads": int(torch.get_num_threads()),
            "score_chunk_size": int(score_chunk_size),
            "knn_chunk_size": int(knn_chunk_size),
        },
        "authority_source_pure_dtype_audit": {
            "raw_cosine": raw_source_audit,
            "downstream": pure_dtype_downstream,
        },
        "current_frozen_text_bank_migration_audit": {
            "positive_text": _text_bank_comparison(
                frozen_positive_text, source_positive_text
            ),
            "negative_text": _text_bank_comparison(
                frozen_negative_text, source_negative_text
            ),
            "positive_fp32_score_error_vs_authority_source": error_summary(
                frozen_positive_fp32[valid], source_positive_fp32[valid]
            ),
            "negative_fp32_score_error_vs_authority_source": error_summary(
                frozen_negative_fp32[valid], source_negative_fp32[valid]
            ),
            "downstream_vs_existing_fp16": frozen_bank_downstream,
        },
        "conclusion": {
            "formal_cache_dtype_upgrade_recommended": bool(pure_flips > 0),
            "reason": (
                "FP16 alone changes at least one frozen final primitive decision"
                if pure_flips > 0
                else "FP16 alone changes no frozen final primitive decisions"
            ),
            "primitive_selection_input_changed": bool(pure_flips > 0),
            "exact_rendered_evaluator_metric_executed": False,
            "exact_rendered_metric_status": (
                "not_provably_identical_without_rendering because primitive selection changed"
                if pure_flips > 0
                else "provably_identical because primitive selection is bit-exact"
            ),
            "formal_materializer_changed": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_frozen_json(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-cache", type=Path, required=True)
    parser.add_argument("--descriptor-cache-sha256", required=True)
    parser.add_argument("--positive-score-cache", type=Path, required=True)
    parser.add_argument("--positive-score-cache-sha256", required=True)
    parser.add_argument("--negative-score-cache", type=Path, required=True)
    parser.add_argument("--negative-score-cache-sha256", required=True)
    parser.add_argument("--frozen-positive-text-cache", type=Path, required=True)
    parser.add_argument("--frozen-positive-text-cache-sha256", required=True)
    parser.add_argument("--frozen-negative-text-cache", type=Path, required=True)
    parser.add_argument("--frozen-negative-text-cache-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-chunk-size", type=int, default=4096)
    parser.add_argument("--knn-chunk-size", type=int, default=65536)
    args = parser.parse_args(argv)
    report = audit(**vars(args))
    print(json.dumps(report["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
