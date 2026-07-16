#!/usr/bin/env python3
"""Compile query-independent primitive semantic features into text unaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.querying.unified_query import cosine_relevancy_torch


def compile_scores(
    features: torch.Tensor,
    text: torch.Tensor,
    valid: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
    peak_normalize: bool,
    scoring: str = "cosine",
    canonical: torch.Tensor | None = None,
    peak_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    values = torch.as_tensor(features).float().cpu()
    queries = F.normalize(torch.as_tensor(text).float().cpu(), dim=-1, eps=1e-8)
    mask = torch.as_tensor(valid).bool().cpu()
    if values.ndim != 2 or queries.ndim != 2 or values.shape[1] != queries.shape[1]:
        raise ValueError("primitive features and text embeddings must be [N,D]/[Q,D]")
    if mask.shape != (values.shape[0],) or not bool(mask.any()):
        raise ValueError("valid must keep at least one primitive row")
    if scoring not in {"softmax_scene", "cosine", "relevancy"}:
        raise ValueError(f"unsupported scoring mode: {scoring}")
    negatives = None
    if scoring == "relevancy":
        if canonical is None:
            raise ValueError("relevancy scoring requires canonical embeddings")
        negatives = F.normalize(torch.as_tensor(canonical).float().cpu(), dim=-1, eps=1e-8)
        if negatives.ndim != 2 or negatives.shape[1] != queries.shape[1]:
            raise ValueError("canonical embeddings must be [M,D] in the query space")

    result = torch.zeros(values.shape[0], queries.shape[0], dtype=torch.float32)
    rows = torch.where(mask)[0]
    for start in range(0, rows.numel(), int(chunk_size)):
        selected = rows[start : start + int(chunk_size)]
        visual = F.normalize(values[selected], dim=-1, eps=1e-8)
        if scoring == "softmax_scene":
            batch_scores = torch.softmax(
                (visual @ queries.T) * float(temperature), dim=-1
            )
        elif scoring == "cosine":
            batch_scores = visual @ queries.T
        else:
            assert negatives is not None
            batch_scores = cosine_relevancy_torch(
                visual,
                queries,
                negatives,
                logit_scale=float(temperature),
                assume_normalized=True,
            )
        result[selected] = batch_scores
    if peak_normalize:
        peak_rows = mask
        if peak_mask is not None:
            peak_rows = torch.as_tensor(peak_mask).bool().cpu().reshape(-1)
            if peak_rows.shape != mask.shape or bool((peak_rows & ~mask).any()):
                raise ValueError("peak_mask must be a subset of valid rows")
            if not bool(peak_rows.any()):
                raise ValueError("peak_mask must keep at least one row")
        peaks = result[peak_rows].amax(dim=0, keepdim=True).clamp_min(1e-12)
        result[mask] = (result[mask] / peaks).clamp_(0.0, 1.0)
    return result.half()


def apply_completion_evidence(
    scores: torch.Tensor,
    valid: torch.Tensor,
    *,
    semantic_confidence: torch.Tensor | None = None,
    primary_valid: torch.Tensor | None = None,
    routing: str = "primary_first",
    primary_support_threshold: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Apply query-independent completion confidence and optional routing.

    ``semantic_confidence`` is a property of the reconstructed primitive row,
    not of any benchmark query.  ``primary_first`` additionally preserves the
    established primary field for queries that already cross the declared
    decision boundary anywhere in 3D; fallback rows are then reserved for
    genuinely unsupported queries.  The direct mode is kept as an explicit
    ablation because it can improve surface coverage but can also overwrite a
    previously correct primary selection.
    """

    values = torch.as_tensor(scores).float().cpu().clone()
    mask = torch.as_tensor(valid).bool().cpu().reshape(-1)
    if values.ndim != 2 or mask.shape != (values.shape[0],):
        raise ValueError("scores and valid must be row-aligned [N,Q]/[N]")
    if routing not in {"direct", "primary_first", "primary_only"}:
        raise ValueError(f"unsupported completion routing: {routing}")
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("primitive query scores are non-finite")
    values[~mask] = 0.0

    confidence_applied = semantic_confidence is not None
    if semantic_confidence is not None:
        confidence = torch.as_tensor(semantic_confidence).float().cpu().reshape(-1)
        if confidence.shape != mask.shape:
            raise ValueError("semantic_confidence must align with primitive rows")
        if not bool(torch.isfinite(confidence).all()):
            raise FloatingPointError("semantic_confidence is non-finite")
        if bool(((confidence < 0.0) | (confidence > 1.0)).any()):
            raise ValueError("semantic_confidence must lie in [0,1]")
        values *= confidence[:, None]

    primary_supported = torch.zeros(values.shape[1], dtype=torch.bool)
    fallback_count = 0
    primary_count = 0
    has_primary_partition = primary_valid is not None
    if primary_valid is not None:
        primary = torch.as_tensor(primary_valid).bool().cpu().reshape(-1)
        if primary.shape != mask.shape:
            raise ValueError("primary_valid must align with primitive rows")
        if bool((primary & ~mask).any()):
            raise ValueError("primary_valid rows must also be valid")
        primary_count = int(primary.sum())
        fallback = mask & ~primary
        fallback_count = int(fallback.sum())
        if bool(primary.any()):
            primary_supported = values[primary].amax(dim=0) >= float(
                primary_support_threshold
            )
        if routing == "primary_first" and bool(fallback.any()):
            values[fallback] *= (~primary_supported).to(values.dtype)[None, :]
        elif routing == "primary_only":
            values[fallback] = 0.0

    values[~mask] = 0.0
    stats = {
        "routing": routing,
        "semantic_confidence_applied": bool(confidence_applied),
        "has_primary_partition": bool(has_primary_partition),
        "primary_valid_count": int(primary_count),
        "fallback_valid_count": int(fallback_count),
        "primary_support_threshold": float(primary_support_threshold),
        "primary_supported_queries": int(primary_supported.sum()),
        "total_queries": int(values.shape[1]),
    }
    return values.half(), stats


def build(args: argparse.Namespace) -> dict:
    feature_cache = torch.load(args.feature_cache, map_location="cpu")
    feature_values = feature_cache["features"]
    if "global_rows" in feature_cache:
        global_rows = torch.as_tensor(feature_cache["global_rows"]).long().cpu()
        valid = torch.as_tensor(feature_cache["valid"]).bool().cpu()
        sparse = torch.as_tensor(feature_values).cpu()
        if sparse.ndim != 2 or sparse.shape[0] != global_rows.numel():
            raise ValueError("sparse feature cache does not align with global_rows")
        if not torch.equal(torch.where(valid)[0], global_rows):
            raise ValueError("sparse feature cache global_rows do not match valid")
        feature_values = torch.zeros(valid.numel(), sparse.shape[1], dtype=sparse.dtype)
        feature_values[global_rows] = sparse
    text_cache = torch.load(args.text_embedding_cache, map_location="cpu")
    canonical_cache = None
    if args.scoring == "relevancy":
        if not args.canonical_embedding_cache:
            raise ValueError(
                "--canonical-embedding-cache is required for relevancy scoring"
            )
        canonical_cache = torch.load(args.canonical_embedding_cache, map_location="cpu")
        if not isinstance(canonical_cache, dict) or "embeddings" not in canonical_cache:
            raise ValueError("canonical embedding cache lacks embeddings")
    primary_peak_mask = (
        feature_cache.get("primary_valid")
        if args.peak_domain == "primary"
        else None
    )
    scores = compile_scores(
        feature_values,
        text_cache["embeddings"],
        feature_cache["valid"],
        temperature=float(args.temperature),
        chunk_size=int(args.chunk_size),
        peak_normalize=bool(args.peak_normalize),
        scoring=str(args.scoring),
        canonical=(
            canonical_cache["embeddings"] if canonical_cache is not None else None
        ),
        peak_mask=primary_peak_mask,
    )
    scores, completion = apply_completion_evidence(
        scores,
        feature_cache["valid"],
        semantic_confidence=feature_cache.get("semantic_confidence"),
        primary_valid=feature_cache.get("primary_valid"),
        routing=str(args.completion_routing),
        primary_support_threshold=float(args.primary_support_threshold),
    )
    construction = {
        "softmax_scene": "scene_softmax",
        "cosine": "cosine_similarity",
        "relevancy": "generic_negative_binary_relevancy",
    }[str(args.scoring)]
    if args.peak_normalize:
        construction += f"_then_{args.peak_domain}_query_peak_normalize"
    metadata = {
        "schema_version": 2,
        "feature_space": "primitive_text_query_scores",
        "construction": construction,
        "feature_cache": str(Path(args.feature_cache).resolve()),
        "text_embedding_cache": str(Path(args.text_embedding_cache).resolve()),
        "canonical_embedding_cache": (
            str(Path(args.canonical_embedding_cache).resolve())
            if args.canonical_embedding_cache
            else ""
        ),
        "scoring": str(args.scoring),
        "temperature": float(args.temperature),
        "peak_domain": str(args.peak_domain),
        "completion": completion,
        "query_names": [str(value) for value in text_cache["queries"]],
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_valid = feature_cache["valid"]
    if args.completion_routing == "primary_only":
        if "primary_valid" not in feature_cache:
            raise ValueError("primary_only requires primary_valid in feature cache")
        output_valid = feature_cache["primary_valid"]
    payload = {
        "xyz": feature_cache["xyz"],
        "geometry_fingerprint": feature_cache.get("geometry_fingerprint", {}),
        "features": scores,
        "valid": output_valid,
        "metadata": metadata,
    }
    if "primary_valid" in feature_cache:
        payload["primary_valid"] = feature_cache["primary_valid"]
    if "semantic_confidence" in feature_cache:
        payload["semantic_confidence"] = feature_cache["semantic_confidence"]
    torch.save(payload, output)
    report = {
        "output": str(output),
        "num_queries": int(scores.shape[1]),
        "valid_gaussians": int(torch.as_tensor(output_valid).sum()),
        "completion": completion,
        "metadata": metadata,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--text-embedding-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--scoring",
        choices=("softmax_scene", "cosine", "relevancy"),
        default="cosine",
    )
    parser.add_argument("--canonical-embedding-cache", default="")
    parser.add_argument("--temperature", type=float, default=50.0)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--peak-normalize", action="store_true")
    parser.add_argument(
        "--peak-domain",
        choices=("primary", "valid"),
        default="primary",
        help=(
            "Rows defining a query peak. primary preserves an established "
            "field when fallback support is added; without primary_valid it "
            "is equivalent to valid."
        ),
    )
    parser.add_argument(
        "--completion-routing",
        choices=("primary_first", "direct", "primary_only"),
        default="primary_first",
        help=(
            "How a completed semantic cache uses fallback rows. primary_first "
            "preserves supported primary queries; direct completes all rows; "
            "primary_only materializes the frozen-base ablation."
        ),
    )
    parser.add_argument(
        "--primary-support-threshold",
        type=float,
        default=0.5,
        help="Fixed query-score boundary used by primary_first routing",
    )
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
