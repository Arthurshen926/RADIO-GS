#!/usr/bin/env python3
"""Rebind O1/O2 raw scores to the frozen evaluator text embedding bank.

The parent raw-score cache is immutable.  This CPU-only materializer emits a
new candidate and recomputes only query columns whose parent text embeddings
are not numerically equivalent to the frozen bank.  No image, mask, label, or
target metric is opened.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen
from radio_gs.scripts import materialize_lerf_o1_o2_streaming as o1o2
from radio_gs.utils.immutable_artifacts import (
    file_record,
    sha256_file,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.lerf_o1_o2_frozen_text_rebind.v1"
COSINE_EQUIVALENCE_ATOL = 1e-6
VECTOR_EQUIVALENCE_ATOL = 2e-6
SCORE_CHUNK_ROWS = 4096


def access_audit() -> dict[str, bool]:
    return {
        "parent_raw_score_caches_opened": True,
        "query_free_base_descriptor_opened": True,
        "query_free_teacher_mean_opened": True,
        "frozen_text_embedding_banks_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_opened": False,
        "target_metrics_computed": False,
        "gpu_used": False,
    }


def _load_mapping_mmap(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[Mapping[str, Any], dict[str, str]]:
    source = Path(path).expanduser().resolve()
    record = {"path": str(source), "sha256": str(expected_sha256)}
    validate_file_record(record, label=label)
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    try:
        value = torch.load(
            source, map_location="cpu", mmap=True, weights_only=False
        )
    except TypeError as error:
        # The repository's pinned torch release predates mmap support.  Keep
        # the new implementation low-copy where available, but retain an
        # explicit CPU-only compatibility path for the frozen environment.
        if "mmap" not in str(error):
            raise
        value = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value, record


def _normalized_bank(
    payload: Mapping[str, Any], *, label: str
) -> tuple[tuple[str, ...], torch.Tensor]:
    raw_queries = payload.get("queries")
    embeddings = payload.get("embeddings")
    if (
        not isinstance(raw_queries, (list, tuple))
        or not raw_queries
        or not all(isinstance(value, str) and value for value in raw_queries)
        or len(set(raw_queries)) != len(raw_queries)
        or not torch.is_tensor(embeddings)
        or embeddings.ndim != 2
        or embeddings.shape[0] != len(raw_queries)
        or not embeddings.is_floating_point()
    ):
        raise ValueError(f"{label} text bank axes differ")
    values = embeddings.detach().float().cpu().contiguous()
    norms = torch.linalg.vector_norm(values, dim=-1)
    if not bool(torch.isfinite(values).all()) or bool((norms <= 1e-8).any()):
        raise ValueError(f"{label} text bank has nonfinite/low-norm embeddings")
    return tuple(raw_queries), F.normalize(values, dim=-1)


def select_frozen_embeddings(
    payload: Mapping[str, Any],
    query_ids: Sequence[str],
) -> torch.Tensor:
    """Select normalized frozen embeddings in the exact requested query order."""

    frozen_queries, frozen_embeddings = _normalized_bank(payload, label="frozen")
    requested = tuple(str(value) for value in query_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("requested query_ids must be non-empty and unique")
    lookup = {query: index for index, query in enumerate(frozen_queries)}
    missing = [query for query in requested if query not in lookup]
    if missing:
        raise ValueError(f"frozen text bank is missing queries: {missing}")
    return frozen_embeddings[
        torch.tensor([lookup[query] for query in requested], dtype=torch.long)
    ].contiguous()


def compare_source_to_frozen(
    source_payload: Mapping[str, Any],
    frozen_payload: Mapping[str, Any],
    query_ids: Sequence[str],
) -> dict[str, Any]:
    """Return a query-wise, target-free equivalence report."""

    source_queries, source_embeddings = _normalized_bank(source_payload, label="source")
    expected = tuple(str(value) for value in query_ids)
    if source_queries != expected:
        raise ValueError("source text bank query axis differs from score cache")
    reference = select_frozen_embeddings(frozen_payload, expected)
    cosine = (source_embeddings * reference).sum(dim=-1)
    max_abs = (source_embeddings - reference).abs().amax(dim=-1)
    equivalent = (cosine >= 1.0 - COSINE_EQUIVALENCE_ATOL) & (
        max_abs <= VECTOR_EQUIVALENCE_ATOL
    )
    return {
        "query_ids": list(expected),
        "cosine": [float(value) for value in cosine.tolist()],
        "normalized_max_abs": [float(value) for value in max_abs.tolist()],
        "equivalent": [bool(value) for value in equivalent.tolist()],
        "mismatched_query_ids": [
            query for query, same in zip(expected, equivalent.tolist()) if not same
        ],
        "minimum_cosine": float(cosine.min()),
        "maximum_normalized_abs_difference": float(max_abs.max()),
        "all_equivalent": bool(equivalent.all()),
    }


def _validate_parent_pair(
    positive: Mapping[str, Any], negative: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    query_ids = tuple(str(value) for value in positive.get("query_ids", []))
    if not query_ids:
        raise ValueError("parent positive query axis is empty")
    xyz = torch.as_tensor(positive.get("xyz")).detach().float().cpu()
    renderer_sha = str(positive.get("renderer_geometry_checkpoint_sha256", ""))
    positive_cache = frozen.validate_ours_multiscale_query_score_cache(
        positive,
        expected_xyz=xyz,
        expected_query_ids=query_ids,
        expected_renderer_geometry_checkpoint_sha256=renderer_sha,
    )
    negative_cache = frozen.validate_ours_multiscale_query_score_cache(
        negative,
        expected_xyz=xyz,
        expected_query_ids=o1o2.NEGATIVE_QUERIES,
        expected_renderer_geometry_checkpoint_sha256=renderer_sha,
    )
    for field in (
        "valid",
        "scale_ids",
        "scale_radii_m",
        "xyz_sha256",
        "field_checkpoint_sha256",
        "readout_checkpoint_sha256",
        "renderer_geometry_checkpoint_sha256",
    ):
        left, right = getattr(positive_cache, field), getattr(negative_cache, field)
        same = torch.equal(left, right) if torch.is_tensor(left) else left == right
        if not bool(same):
            raise ValueError(f"parent positive/negative {field} differs")
    return (
        positive_cache.query_scores,
        negative_cache.query_scores,
        xyz,
        positive_cache.valid,
        query_ids,
    )


def _parent_text_payload(parent: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, str]]:
    authority = parent.get("authority")
    sources = authority.get("source_artifacts") if isinstance(authority, Mapping) else None
    record = sources.get("text_query_cache") if isinstance(sources, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError("parent score cache text_query_cache binding is missing")
    return _load_mapping_mmap(
        str(record.get("path", "")),
        str(record.get("sha256", "")),
        label="parent text-query cache",
    )


def _validate_descriptor_inputs(
    base: Mapping[str, Any], teacher: Mapping[str, Any], valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.as_tensor(base.get("global_rows")).detach().long().cpu()
    features = base.get("features_by_scale")
    teacher_rows = torch.as_tensor(teacher.get("global_rows")).detach().long().cpu()
    teacher_mean = teacher.get("teacher_mean")
    teacher_valid = torch.as_tensor(teacher.get("teacher_valid")).detach().bool().cpu()
    expected_rows = torch.where(valid)[0]
    if (
        not torch.equal(rows, expected_rows)
        or not torch.equal(teacher_rows, rows)
        or not torch.is_tensor(features)
        or features.shape != (rows.numel(), 3, 1536)
        or not torch.is_tensor(teacher_mean)
        or teacher_mean.shape != (rows.numel(), 1536)
        or teacher_valid.shape != (rows.numel(),)
    ):
        raise ValueError("base/teacher/score-cache row axes differ")
    for start in range(0, int(rows.numel()), SCORE_CHUNK_ROWS):
        stop = min(start + SCORE_CHUNK_ROWS, int(rows.numel()))
        base_chunk = features[start:stop].float()
        teacher_chunk = teacher_mean[start:stop].float()
        if not bool(torch.isfinite(base_chunk).all()) or bool(
            (torch.linalg.vector_norm(base_chunk, dim=-1) <= 1e-8).any()
        ):
            raise ValueError("base descriptor contains nonfinite/low-norm rows")
        if not bool(torch.isfinite(teacher_chunk).all()) or bool(
            (torch.linalg.vector_norm(teacher_chunk[teacher_valid[start:stop]], dim=-1) <= 1e-8).any()
        ):
            raise ValueError("teacher descriptor contains nonfinite/low-norm valid rows")
    return rows, features, teacher_mean, teacher_valid


def _recompute_mismatched_columns(
    parent_scores: torch.Tensor,
    *,
    embeddings: torch.Tensor,
    mismatched_indices: Sequence[int],
    oracle: str,
    rows: torch.Tensor,
    base_features: torch.Tensor,
    teacher_mean: torch.Tensor,
    teacher_valid: torch.Tensor,
) -> torch.Tensor:
    result = parent_scores.detach().float().cpu().clone().contiguous()
    indices = torch.as_tensor(list(mismatched_indices), dtype=torch.long)
    if not int(indices.numel()):
        return result
    selected_text = embeddings[indices]
    for start in range(0, int(rows.numel()), SCORE_CHUNK_ROWS):
        stop = min(start + SCORE_CHUNK_ROWS, int(rows.numel()))
        base = F.normalize(base_features[start:stop].float(), dim=-1)
        mean = teacher_mean[start:stop].float()
        active = teacher_valid[start:stop]
        mean_unit = torch.zeros_like(mean)
        if bool(active.any()):
            mean_unit[active] = F.normalize(mean[active], dim=-1)
        if oracle == "O1":
            descriptor = base.clone()
            if bool(active.any()):
                for scale in range(3):
                    descriptor[active, scale] = o1o2.geodesic_project(
                        base[active, scale],
                        mean_unit[active],
                        o1o2.O1_MAXIMUM_ANGLE_RADIANS,
                    )
        elif oracle == "O2":
            descriptor = base.clone()
            if bool(active.any()):
                descriptor[active] = mean_unit[active, None, :]
        else:
            raise ValueError(f"unsupported oracle: {oracle}")
        scores = torch.einsum("bsd,qd->bsq", descriptor, selected_text)
        global_rows = rows[start:stop]
        result[global_rows[:, None, None], torch.arange(3)[None, :, None], indices[None, None, :]] = scores
    if not bool(torch.isfinite(result).all()) or bool((result.abs() > 1.0001).any()):
        raise ValueError("rebound normalized cosine scores are invalid")
    return result


def _rebound_cache(
    parent: Mapping[str, Any],
    scores: torch.Tensor,
    *,
    selected_text_record: Mapping[str, str],
    report_record: Mapping[str, str],
    parent_record: Mapping[str, str],
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(parent))
    payload["query_scores"] = scores.float().contiguous()
    authority = payload["authority"]
    authority["query_scores_sha256"] = frozen.tensor_sha256_typed(
        payload["query_scores"]
    )
    authority["score_implementation"] = str(Path(__file__).resolve())
    sources = authority["source_artifacts"]
    sources["text_query_cache"] = dict(selected_text_record)
    sources["frozen_text_rebind_report"] = dict(report_record)
    sources["parent_raw_score_cache"] = dict(parent_record)
    sources["materializer_source"] = file_record(Path(__file__).resolve())
    return payload


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if str(args.output_dir) != str(output_dir):
        raise ValueError("output_dir must be canonical absolute")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "positive_text": output_dir / f"{args.scene_id}_frozen_positive_text.pt",
        "negative_text": output_dir / f"{args.scene_id}_frozen_negative_text.pt",
        "binding_report": output_dir / f"{args.scene_id}_{args.oracle.lower()}_text_rebind.json",
        "positive": output_dir / f"{args.scene_id}_{args.oracle.lower()}_positive.pt",
        "negative": output_dir / f"{args.scene_id}_{args.oracle.lower()}_negative.pt",
    }
    if any(path.exists() or path.is_symlink() for path in names.values()):
        raise FileExistsError("frozen-text rebind outputs already exist")

    parent_positive, parent_positive_record = _load_mapping_mmap(
        args.parent_positive, args.parent_positive_sha256, label="parent positive cache"
    )
    parent_negative, parent_negative_record = _load_mapping_mmap(
        args.parent_negative, args.parent_negative_sha256, label="parent negative cache"
    )
    positive_scores, negative_scores, xyz, valid, query_ids = _validate_parent_pair(
        parent_positive, parent_negative
    )
    for parent in (parent_positive, parent_negative):
        oracle = str(parent["authority"]["descriptor_axis"].get("oracle", ""))
        if oracle != args.oracle:
            raise ValueError(f"parent cache oracle {oracle!r} differs from {args.oracle!r}")

    source_positive, source_positive_record = _parent_text_payload(parent_positive)
    source_negative, source_negative_record = _parent_text_payload(parent_negative)
    frozen_positive, frozen_positive_record = _load_mapping_mmap(
        args.frozen_positive_bank,
        args.frozen_positive_bank_sha256,
        label="frozen positive bank",
    )
    frozen_negative, frozen_negative_record = _load_mapping_mmap(
        args.frozen_negative_bank,
        args.frozen_negative_bank_sha256,
        label="frozen negative bank",
    )
    positive_binding = compare_source_to_frozen(
        source_positive, frozen_positive, query_ids
    )
    negative_binding = compare_source_to_frozen(
        source_negative, frozen_negative, o1o2.NEGATIVE_QUERIES
    )
    positive_embeddings = select_frozen_embeddings(frozen_positive, query_ids)
    negative_embeddings = select_frozen_embeddings(
        frozen_negative, o1o2.NEGATIVE_QUERIES
    )
    selected_positive_payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "source_frozen_bank": frozen_positive_record,
        "queries": list(query_ids),
        "prompt_templates": ["{query}"],
        "text_encoder": "siglip2",
        "model_name": "google/siglip2-giant-opt-patch16-384",
        "embeddings": positive_embeddings,
        "normalized_embeddings_sha256": frozen.tensor_sha256_typed(positive_embeddings),
    }
    selected_negative_payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "source_frozen_bank": frozen_negative_record,
        "queries": list(o1o2.NEGATIVE_QUERIES),
        "prompt_templates": ["{query}"],
        "text_encoder": "siglip2",
        "model_name": "google/siglip2-giant-opt-patch16-384",
        "embeddings": negative_embeddings,
        "normalized_embeddings_sha256": frozen.tensor_sha256_typed(negative_embeddings),
    }
    write_torch_noclobber(names["positive_text"], selected_positive_payload)
    write_torch_noclobber(names["negative_text"], selected_negative_payload)
    selected_positive_record = file_record(names["positive_text"])
    selected_negative_record = file_record(names["negative_text"])

    base, base_record = _load_mapping_mmap(
        args.base_descriptor, args.base_descriptor_sha256, label="base descriptor"
    )
    teacher, teacher_record = _load_mapping_mmap(
        args.teacher_mean, args.teacher_mean_sha256, label="teacher mean"
    )
    rows, base_features, teacher_mean, teacher_valid = _validate_descriptor_inputs(
        base, teacher, valid
    )
    positive_mismatches = [
        index for index, same in enumerate(positive_binding["equivalent"]) if not same
    ]
    negative_mismatches = [
        index for index, same in enumerate(negative_binding["equivalent"]) if not same
    ]
    rebound_positive = _recompute_mismatched_columns(
        positive_scores,
        embeddings=positive_embeddings,
        mismatched_indices=positive_mismatches,
        oracle=args.oracle,
        rows=rows,
        base_features=base_features,
        teacher_mean=teacher_mean,
        teacher_valid=teacher_valid,
    )
    rebound_negative = _recompute_mismatched_columns(
        negative_scores,
        embeddings=negative_embeddings,
        mismatched_indices=negative_mismatches,
        oracle=args.oracle,
        rows=rows,
        base_features=base_features,
        teacher_mean=teacher_mean,
        teacher_valid=teacher_valid,
    )
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete_source_only_frozen_text_rebind",
        "scene_id": args.scene_id,
        "oracle": args.oracle,
        "producer": file_record(Path(__file__).resolve()),
        "inputs": {
            "parent_positive": parent_positive_record,
            "parent_negative": parent_negative_record,
            "parent_positive_text": source_positive_record,
            "parent_negative_text": source_negative_record,
            "frozen_positive_bank": frozen_positive_record,
            "frozen_negative_bank": frozen_negative_record,
            "base_descriptor": base_record,
            "teacher_mean": teacher_record,
        },
        "selected_text_caches": {
            "positive": selected_positive_record,
            "negative": selected_negative_record,
        },
        "positive_binding": positive_binding,
        "negative_binding": negative_binding,
        "recomputed_positive_query_ids": [query_ids[index] for index in positive_mismatches],
        "recomputed_negative_query_ids": [
            o1o2.NEGATIVE_QUERIES[index] for index in negative_mismatches
        ],
        "unchanged_positive_query_columns_bitwise_parent": [
            query for index, query in enumerate(query_ids) if index not in positive_mismatches
        ],
        "unchanged_negative_query_columns_bitwise_parent": [
            query
            for index, query in enumerate(o1o2.NEGATIVE_QUERIES)
            if index not in negative_mismatches
        ],
        "valid_primitives": int(valid.sum()),
        "finite_output_scores": bool(
            torch.isfinite(rebound_positive).all()
            and torch.isfinite(rebound_negative).all()
        ),
        "access_audit": access_audit(),
        "metric_execution_authorized": False,
    }
    write_frozen_json(names["binding_report"], report)
    report_record = file_record(names["binding_report"])
    positive_payload = _rebound_cache(
        parent_positive,
        rebound_positive,
        selected_text_record=selected_positive_record,
        report_record=report_record,
        parent_record=parent_positive_record,
    )
    negative_payload = _rebound_cache(
        parent_negative,
        rebound_negative,
        selected_text_record=selected_negative_record,
        report_record=report_record,
        parent_record=parent_negative_record,
    )
    write_torch_noclobber(names["positive"], positive_payload)
    write_torch_noclobber(names["negative"], negative_payload)
    return {
        **report,
        "binding_report": report_record,
        "outputs": {
            "positive": file_record(names["positive"]),
            "negative": file_record(names["negative"]),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--oracle", choices=("O1", "O2"), required=True)
    for name in (
        "parent_positive",
        "parent_negative",
        "base_descriptor",
        "teacher_mean",
        "frozen_positive_bank",
        "frozen_negative_bank",
    ):
        option = name.replace("_", "-")
        parser.add_argument(f"--{option}", required=True)
        parser.add_argument(f"--{option}-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    result = materialize(build_parser().parse_args())
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "COSINE_EQUIVALENCE_ATOL",
    "SCHEMA",
    "VECTOR_EQUIVALENCE_ATOL",
    "access_audit",
    "compare_source_to_frozen",
    "materialize",
    "select_frozen_embeddings",
]
