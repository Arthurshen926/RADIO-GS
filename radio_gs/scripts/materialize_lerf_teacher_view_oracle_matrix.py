#!/usr/bin/env python3
"""Materialize the preregistered Figurines O1--O4 teacher-view oracle matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.scripts.eval_lerf_direct_3d_selection import tensor_sha256_typed
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.lerf_teacher_view_oracle_matrix.v1"
RAW_CACHE_CONTRACT = "radio_gs.ours_lerf_direct3d_multiscale_query_scores_fp32.v4"
RAW_AUTHORITY_CONTRACT = "radio_gs.lerf_multiscale_query_score_fp32_authority.v4"
PROBABILITY_CACHE_CONTRACT = (
    "radio_gs.ours_lerf_direct3d_multiscale_query_probabilities.v3"
)
PROBABILITY_AUTHORITY_CONTRACT = (
    "radio_gs.lerf_multiscale_query_probability_authority.v3"
)
NEGATIVE_QUERIES = ("object", "things", "stuff", "texture")
SCALE_IDS = ("0.25", "0.45", "0.7")
SCALE_RADII = (0.25, 0.45, 0.7)
BETA = 10.0


def _require_file(path: str | Path, expected_sha256: str, label: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise ValueError(f"{label} is missing or has a different SHA-256")
    return source


def geodesic_project(
    base: torch.Tensor, teacher: torch.Tensor, maximum_angle: float
) -> torch.Tensor:
    """Closed-form unit-sphere projection from base toward teacher."""

    x = F.normalize(base.float(), dim=-1)
    y = F.normalize(teacher.float(), dim=-1)
    angle = torch.acos((x * y).sum(dim=-1).clamp(-1.0, 1.0))
    fraction = (float(maximum_angle) / angle.clamp_min(1e-8)).clamp(max=1.0)
    sin_angle = torch.sin(angle)
    left = torch.sin((1.0 - fraction) * angle) / sin_angle.clamp_min(1e-8)
    right = torch.sin(fraction * angle) / sin_angle.clamp_min(1e-8)
    projected = left[..., None] * x + right[..., None] * y
    near = angle < 1e-5
    if bool(near.any()):
        projected[near] = F.normalize(
            (1.0 - fraction[near, None]) * x[near]
            + fraction[near, None] * y[near],
            dim=-1,
        )
    return F.normalize(projected, dim=-1)


def fit_two_spherical_modes(
    views: torch.Tensor,
    mask: torch.Tensor,
    frame_ids: torch.Tensor,
    *,
    iterations: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permutation-invariant K=2 spherical fit with immutable-frame tie breaks."""

    values = F.normalize(views.float(), dim=-1)
    valid = mask.bool()
    frames = frame_ids.long()
    if values.ndim != 3 or valid.shape != values.shape[:2] or frames.shape != valid.shape:
        raise ValueError("teacher-view mode input differs")
    batch, count, _ = values.shape
    if count < 1:
        raise ValueError("teacher-view mode input is empty")
    pairs = [(left, right) for left in range(count) for right in range(left + 1, count)]
    best_cos = torch.full((batch,), float("inf"), device=values.device)
    best_key = torch.full(
        (batch,), torch.iinfo(torch.int64).max, dtype=torch.int64, device=values.device
    )
    best_left = torch.zeros(batch, dtype=torch.long, device=values.device)
    best_right = torch.zeros(batch, dtype=torch.long, device=values.device)
    for left, right in pairs:
        pair_valid = valid[:, left] & valid[:, right]
        cosine = (values[:, left] * values[:, right]).sum(dim=-1)
        low = torch.minimum(frames[:, left], frames[:, right]).clamp_min(0)
        high = torch.maximum(frames[:, left], frames[:, right]).clamp_min(0)
        key = low * 1_000_000 + high
        replace = pair_valid & (
            (cosine < best_cos) | ((cosine == best_cos) & (key < best_key))
        )
        best_cos[replace] = cosine[replace]
        best_key[replace] = key[replace]
        best_left[replace] = left
        best_right[replace] = right
    row = torch.arange(batch, device=values.device)
    first_valid = valid.to(torch.int64).argmax(dim=1)
    has_pair = torch.isfinite(best_cos)
    best_left = torch.where(has_pair, best_left, first_valid)
    best_right = torch.where(has_pair, best_right, first_valid)
    centers = torch.stack(
        (values[row, best_left], values[row, best_right]), dim=1
    )
    seed_frames = torch.stack(
        (frames[row, best_left], frames[row, best_right]), dim=1
    )
    assignment = torch.zeros(batch, count, dtype=torch.long, device=values.device)
    for _ in range(int(iterations)):
        similarity = torch.einsum("bvd,bkd->bvk", values, centers)
        prefer_zero = seed_frames[:, 0] <= seed_frames[:, 1]
        assignment = torch.where(
            similarity[:, :, 0] > similarity[:, :, 1],
            torch.zeros_like(assignment),
            torch.where(
                similarity[:, :, 1] > similarity[:, :, 0],
                torch.ones_like(assignment),
                (~prefer_zero)[:, None].long(),
            ),
        )
        for mode in range(2):
            members = valid & (assignment == mode)
            summed = (values * members[:, :, None]).sum(dim=1)
            nonempty = members.any(dim=1)
            updated = F.normalize(summed, dim=-1)
            centers[:, mode] = torch.where(
                nonempty[:, None], updated, centers[:, mode]
            )
    counts = torch.stack(
        [(valid & (assignment == mode)).sum(dim=1) for mode in range(2)], dim=1
    ).float()
    weights = counts / valid.sum(dim=1, keepdim=True).clamp_min(1)
    centers[~(weights > 0)] = 0
    return canonicalize_two_mode_axis(centers, weights)


def canonicalize_two_mode_axis(
    modes: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Order modes by membership descending, then float32 descriptor SHA."""

    values = modes.detach().float().cpu().contiguous()
    mixture = weights.detach().float().cpu().contiguous()
    if values.ndim != 3 or values.shape[1] != 2 or mixture.shape != values.shape[:2]:
        raise ValueError("two-mode canonicalization input differs")
    swap = mixture[:, 1] > mixture[:, 0]
    tied = torch.where(mixture[:, 1] == mixture[:, 0])[0].tolist()
    for row in tied:
        left = hashlib.sha256(values[row, 0].numpy().tobytes(order="C")).digest()
        right = hashlib.sha256(values[row, 1].numpy().tobytes(order="C")).digest()
        if right < left:
            swap[row] = True
    if bool(swap.any()):
        original_values = values[swap].clone()
        original_weights = mixture[swap].clone()
        values[swap, 0] = original_values[:, 1]
        values[swap, 1] = original_values[:, 0]
        mixture[swap, 0] = original_weights[:, 1]
        mixture[swap, 1] = original_weights[:, 0]
    return values, mixture


def normalized_logsumexp_response(
    descriptors: torch.Tensor,
    weights: torch.Tensor,
    queries: torch.Tensor,
    *,
    beta: float = BETA,
) -> torch.Tensor:
    """Weighted normalized LSE of cosine responses."""

    values = F.normalize(descriptors.float(), dim=-1)
    text = F.normalize(queries.float(), dim=-1)
    mixture = weights.float()
    cosine = torch.einsum("bkd,qd->bkq", values, text)
    log_weight = torch.where(
        mixture > 0,
        torch.log(mixture.clamp_min(torch.finfo(torch.float32).tiny)),
        torch.full_like(mixture, -torch.inf),
    )
    return torch.logsumexp(float(beta) * cosine + log_weight[:, :, None], dim=1) / float(beta)


def _teacher_mean(views: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    summed = (views.float() * mask[:, :, None]).sum(dim=1)
    return F.normalize(summed, dim=-1)


def _raw_cache(
    template: Mapping[str, Any],
    scores: torch.Tensor,
    *,
    representation: Mapping[str, str],
    text_cache: Mapping[str, str],
    producer: Mapping[str, str],
    representation_tensor_sha256: str,
    oracle: str,
) -> dict[str, Any]:
    payload = {key: value for key, value in template.items() if key != "authority"}
    payload["query_scores"] = scores.contiguous().float()
    authority = copy.deepcopy(template["authority"])
    authority["contract"] = RAW_AUTHORITY_CONTRACT
    authority["score_semantics"] = "raw_independent_normalized_cosine"
    authority["score_formula"] = (
        "l2_normalize(descriptor) @ l2_normalize(text_embedding).T"
    )
    authority["score_implementation"] = str(producer["path"])
    authority["score_dtype"] = "torch.float32"
    authority["query_scores_sha256"] = tensor_sha256_typed(payload["query_scores"])
    authority["descriptor_axis"]["features_by_scale_sha256"] = (
        representation_tensor_sha256
    )
    authority["descriptor_axis"]["oracle"] = oracle
    authority["source_artifacts"]["descriptor_cache"] = dict(representation)
    authority["source_artifacts"]["text_query_cache"] = dict(text_cache)
    authority["source_artifacts"]["materializer_source"] = dict(producer)
    authority["calibration_constraints"]["benchmark_metrics_opened"] = False
    payload["authority"] = authority
    return payload


def _probability_cache(
    template: Mapping[str, Any],
    scores: torch.Tensor,
    *,
    representation: Mapping[str, str],
    positive_text: Mapping[str, str],
    negative_text: Mapping[str, str],
    producer: Mapping[str, str],
    oracle: str,
    formula: str,
) -> dict[str, Any]:
    payload = {key: value for key, value in template.items() if key != "authority"}
    payload["version"] = 3
    payload["contract"] = PROBABILITY_CACHE_CONTRACT
    payload["query_scores"] = scores.contiguous().float()
    base_authority = template["authority"]
    sources = copy.deepcopy(base_authority["source_artifacts"])
    sources.update(
        {
            "text_query_cache": dict(positive_text),
            "generic_negative_text_cache": dict(negative_text),
            "residual_codebook_checkpoint": dict(representation),
            "query_router_checkpoint": dict(producer),
            "materializer_source": dict(producer),
        }
    )
    authority = {
        "schema_version": 3,
        "artifact_type": "radio_gs_lerf_multiscale_primitive_query_score_cache",
        "contract": PROBABILITY_AUTHORITY_CONTRACT,
        "score_semantics": "canonical_negative_bernoulli_probability",
        "score_formula": formula,
        "score_implementation": str(producer["path"]),
        "score_dtype": "torch.float32",
        "probability_route": "exact_frozen_v2_slot0_control",
        "value_range": [0.0, 1.0],
        "logit_scale": 10.0,
        "generic_negative_queries": list(NEGATIVE_QUERIES),
        "scale_axis": copy.deepcopy(base_authority["scale_axis"]),
        "query_axis": copy.deepcopy(base_authority["query_axis"]),
        "geometry_axis": copy.deepcopy(base_authority["geometry_axis"]),
        "descriptor_axis": {
            "dimension": 1536,
            "materialized": True,
            "execution_representation": "teacher_view_oracle_matrix_v1",
            "valid_rows": int(payload["valid"].sum()),
            "oracle": oracle,
        },
        "query_scores_sha256": tensor_sha256_typed(payload["query_scores"]),
        "source_artifacts": sources,
        "consumer_contracts": copy.deepcopy(base_authority["consumer_contracts"]),
        "calibration_constraints": copy.deepcopy(
            base_authority["calibration_constraints"]
        ),
    }
    authority["consumer_contracts"]["direct3d"]["contract"] = (
        PROBABILITY_CACHE_CONTRACT
    )
    authority["consumer_contracts"]["lerf2d_scalar_map_renderer"][
        "score_semantics"
    ] = "canonical_negative_bernoulli_probability"
    payload["authority"] = authority
    return payload


def _relation_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    left = F.normalize(reference.float(), dim=-1)
    right = F.normalize(candidate.float(), dim=-1)
    ref_gram = left @ left.t()
    candidate_gram = right @ right.t()
    indices = torch.triu_indices(left.shape[0], left.shape[0], offset=1)
    ref_values = ref_gram[indices[0], indices[1]]
    candidate_values = candidate_gram[indices[0], indices[1]]
    centered_ref = ref_values - ref_values.mean()
    centered_candidate = candidate_values - candidate_values.mean()
    correlation = (centered_ref * centered_candidate).mean() / (
        centered_ref.square().mean().sqrt()
        * centered_candidate.square().mean().sqrt()
    ).clamp_min(1e-8)
    return {
        "pearson": float(correlation),
        "mean_absolute_error": float((ref_values - candidate_values).abs().mean()),
    }


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir).expanduser().resolve()
    paths = {
        "representation": root / "figurines_o1_o3_representations.pt",
        "o1_positive": root / "figurines_o1_positive.pt",
        "o1_negative": root / "figurines_o1_negative.pt",
        "o2_positive": root / "figurines_o2_positive.pt",
        "o2_negative": root / "figurines_o2_negative.pt",
        "o3_probability": root / "figurines_o3_probability.pt",
        "o4_probability": root / "figurines_o4_probability.pt",
        "diagnostics": root / "figurines_query_free_diagnostics.json",
        "manifest": root / "figurines_oracle_matrix_manifest.json",
    }
    existing = [str(path) for path in paths.values() if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError("refuses to clobber oracle outputs: " + ", ".join(existing))
    prereg = _require_file(args.preregistration, args.expected_preregistration_sha256, "preregistration")
    producer = file_record(Path(__file__).resolve())
    base_path = _require_file(args.base_descriptor, args.expected_base_descriptor_sha256, "base descriptor")
    teacher_path = _require_file(args.teacher_authority, args.expected_teacher_authority_sha256, "teacher authority")
    pos_path = _require_file(args.positive_text_cache, args.expected_positive_text_cache_sha256, "positive text")
    neg_path = _require_file(args.negative_text_cache, args.expected_negative_text_cache_sha256, "negative text")
    o0_pos_path = _require_file(args.o0_positive_cache, args.expected_o0_positive_cache_sha256, "O0 positive cache")
    o0_neg_path = _require_file(args.o0_negative_cache, args.expected_o0_negative_cache_sha256, "O0 negative cache")

    base, _, _ = load_torch_mapping(base_path, expected_sha256=args.expected_base_descriptor_sha256, map_location="cpu", label="base descriptor")
    teacher, _, _ = load_torch_mapping(teacher_path, expected_sha256=args.expected_teacher_authority_sha256, map_location="cpu", label="teacher authority")
    positive, _, _ = load_torch_mapping(pos_path, expected_sha256=args.expected_positive_text_cache_sha256, map_location="cpu", label="positive text")
    negative, _, _ = load_torch_mapping(neg_path, expected_sha256=args.expected_negative_text_cache_sha256, map_location="cpu", label="negative text")
    o0_positive, _, _ = load_torch_mapping(o0_pos_path, expected_sha256=args.expected_o0_positive_cache_sha256, map_location="cpu", label="O0 positive")
    o0_negative, _, _ = load_torch_mapping(o0_neg_path, expected_sha256=args.expected_o0_negative_cache_sha256, map_location="cpu", label="O0 negative")

    rows = torch.as_tensor(base["global_rows"]).long().cpu()
    base_features = F.normalize(torch.as_tensor(base["features_by_scale"]).float(), dim=-1)
    views = torch.as_tensor(teacher["teacher_view_descriptors"]).float().cpu()
    view_mask = torch.as_tensor(teacher["teacher_view_mask"]).bool().cpu()
    frame_ids = torch.as_tensor(teacher["teacher_view_frame_ids"]).long().cpu()
    if (
        teacher.get("schema") != "radio_gs.lerf_source_teacher_view_siglip_authority.v1"
        or not torch.equal(rows, torch.as_tensor(teacher["global_rows"]).long())
        or views.shape != (rows.numel(), 4, 1536)
        or base_features.shape != (rows.numel(), 3, 1536)
    ):
        raise ValueError("base/teacher authority alignment differs")
    teacher_valid = view_mask.any(dim=1)
    teacher_mean = _teacher_mean(views, view_mask)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA oracle materialization without CUDA")
    o1_parts: list[torch.Tensor] = []
    mode_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    batch_size = int(args.batch_size)
    for start in range(0, rows.numel(), batch_size):
        stop = min(rows.numel(), start + batch_size)
        batch_base = base_features[start:stop].to(device)
        batch_mean = teacher_mean[start:stop].to(device)
        batch_valid = teacher_valid[start:stop].to(device)
        batch_o1 = batch_base.clone()
        for scale in range(3):
            projected = geodesic_project(
                batch_base[:, scale], batch_mean, 0.15
            )
            batch_o1[:, scale] = torch.where(
                batch_valid[:, None], projected, batch_base[:, scale]
            )
        modes, weights = fit_two_spherical_modes(
            views[start:stop].to(device),
            view_mask[start:stop].to(device),
            frame_ids[start:stop].to(device),
        )
        o1_parts.append(batch_o1.float().cpu())
        mode_parts.append(modes.float().cpu())
        weight_parts.append(weights.cpu())
    o1 = torch.cat(o1_parts, dim=0)
    modes = torch.cat(mode_parts, dim=0)
    mode_weights = torch.cat(weight_parts, dim=0)
    representation_payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "global_rows": rows,
        "teacher_valid": teacher_valid,
            "o1_features_by_scale": o1,
        "o2_teacher_mean": teacher_mean.float(),
        "o3_modes": modes,
        "o3_mode_weights": mode_weights,
        "metadata": {
            "angle_cap_radians": 0.15,
            "mode_count": 2,
            "mode_iterations": 4,
            "mode_initialization": "farthest_pair_with_frame_id_tie_break",
            "view_response_beta": BETA,
            "base_descriptor": file_record(base_path),
            "teacher_authority": file_record(teacher_path),
            "preregistration": file_record(prereg),
            "producer": producer,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened_for_representation": False,
        },
    }
    write_torch_noclobber(paths["representation"], representation_payload)
    representation = file_record(paths["representation"])

    pos_embedding = F.normalize(torch.as_tensor(positive["embeddings"]).float(), dim=-1)
    neg_embedding = F.normalize(torch.as_tensor(negative["embeddings"]).float(), dim=-1)
    pos_embedding_device = pos_embedding.to(device)
    neg_embedding_device = neg_embedding.to(device)
    if list(positive["queries"]) != list(o0_positive["query_ids"]) or tuple(negative["queries"]) != NEGATIVE_QUERIES:
        raise ValueError("frozen query axes differ")
    total = int(torch.as_tensor(base["xyz"]).shape[0])
    query_count = int(pos_embedding.shape[0])
    score_bank = {
        name: torch.zeros(total, 3, query_count if "positive" in name else len(NEGATIVE_QUERIES), dtype=torch.float32)
        for name in ("o1_positive", "o1_negative", "o2_positive", "o2_negative")
    }
    o3_probability = torch.zeros(total, 3, query_count, dtype=torch.float32)
    o4_probability = torch.zeros_like(o3_probability)
    o0_pos_scores = torch.as_tensor(o0_positive["query_scores"]).float()
    o0_neg_scores = torch.as_tensor(o0_negative["query_scores"]).float()
    for start in range(0, rows.numel(), batch_size):
        stop = min(rows.numel(), start + batch_size)
        global_rows = rows[start:stop]
        current_valid = teacher_valid[start:stop]
        o1_batch = F.normalize(o1[start:stop].float().to(device), dim=-1)
        o2_batch = teacher_mean[start:stop].to(device)
        score_bank["o1_positive"][global_rows] = torch.einsum(
            "bsd,qd->bsq", o1_batch, pos_embedding_device
        ).cpu()
        score_bank["o1_negative"][global_rows] = torch.einsum(
            "bsd,qd->bsq", o1_batch, neg_embedding_device
        ).cpu()
        o2_pos = torch.einsum("bd,qd->bq", o2_batch, pos_embedding_device)[:, None].expand(-1, 3, -1).clone()
        o2_neg = torch.einsum("bd,qd->bq", o2_batch, neg_embedding_device)[:, None].expand(-1, 3, -1).clone()
        if bool((~current_valid).any()):
            fallback = global_rows[~current_valid]
            o2_pos[~current_valid.to(device)] = o0_pos_scores[fallback].to(device)
            o2_neg[~current_valid.to(device)] = o0_neg_scores[fallback].to(device)
        score_bank["o2_positive"][global_rows] = o2_pos.cpu()
        score_bank["o2_negative"][global_rows] = o2_neg.cpu()

        mode_pos = normalized_logsumexp_response(
            modes[start:stop].to(device), mode_weights[start:stop].to(device), pos_embedding_device
        )
        mode_neg = normalized_logsumexp_response(
            modes[start:stop].to(device), mode_weights[start:stop].to(device), neg_embedding_device
        )
        view_weights = view_mask[start:stop].float().to(device)
        view_weights = view_weights / view_weights.sum(dim=1, keepdim=True).clamp_min(1)
        view_pos = normalized_logsumexp_response(
            views[start:stop].to(device), view_weights, pos_embedding_device
        )
        view_neg = normalized_logsumexp_response(
            views[start:stop].to(device), view_weights, neg_embedding_device
        )
        mode_probability = torch.sigmoid(BETA * (mode_pos - mode_neg.max(dim=1, keepdim=True).values))
        view_probability = torch.sigmoid(BETA * (view_pos - view_neg.max(dim=1, keepdim=True).values))
        mode_probability = mode_probability[:, None].expand(-1, 3, -1).clone()
        view_probability = view_probability[:, None].expand(-1, 3, -1).clone()
        if bool((~current_valid).any()):
            fallback = global_rows[~current_valid]
            fallback_probability = torch.sigmoid(
                BETA
                * (
                    o0_pos_scores[fallback]
                    - o0_neg_scores[fallback].max(dim=-1, keepdim=True).values
                )
            )
            mode_probability[~current_valid.to(device)] = fallback_probability.to(device)
            view_probability[~current_valid.to(device)] = fallback_probability.to(device)
        o3_probability[global_rows] = mode_probability.cpu()
        o4_probability[global_rows] = view_probability.cpu()

    representation_shas = {
        "o1": tensor_sha256_typed(representation_payload["o1_features_by_scale"]),
        "o2": tensor_sha256_typed(representation_payload["o2_teacher_mean"]),
        "o3": tensor_sha256_typed(representation_payload["o3_modes"]),
    }
    text_records = {"positive": file_record(pos_path), "negative": file_record(neg_path)}
    raw_specs = (
        ("o1_positive", o0_positive, "o1", text_records["positive"]),
        ("o1_negative", o0_negative, "o1", text_records["negative"]),
        ("o2_positive", o0_positive, "o2", text_records["positive"]),
        ("o2_negative", o0_negative, "o2", text_records["negative"]),
    )
    for name, template, oracle, text_record in raw_specs:
        payload = _raw_cache(
            template,
            score_bank[name],
            representation=representation,
            text_cache=text_record,
            producer=producer,
            representation_tensor_sha256=representation_shas[oracle],
            oracle=oracle.upper(),
        )
        write_torch_noclobber(paths[name], payload)
    o3_payload = _probability_cache(
        o0_positive,
        o3_probability,
        representation=representation,
        positive_text=text_records["positive"],
        negative_text=text_records["negative"],
        producer=producer,
        oracle="O3",
        formula="sigmoid(10*(K2_frequency_weighted_normalized_LSE_positive-max_negative_same_aggregation))",
    )
    o4_payload = _probability_cache(
        o0_positive,
        o4_probability,
        representation=representation,
        positive_text=text_records["positive"],
        negative_text=text_records["negative"],
        producer=producer,
        oracle="O4",
        formula="sigmoid(10*(per_view_equal_weight_normalized_LSE_positive-max_negative_same_aggregation))",
    )
    write_torch_noclobber(paths["o3_probability"], o3_payload)
    write_torch_noclobber(paths["o4_probability"], o4_payload)

    active_views = view_mask[:, None, :]
    all_view: dict[str, torch.Tensor] = {}
    all_view["O0"] = torch.einsum("bsd,bvd->bsv", base_features, views)[active_views.expand(-1, 3, -1)]
    all_view["O1"] = torch.einsum("bsd,bvd->bsv", o1.float(), views)[active_views.expand(-1, 3, -1)]
    o2_cos = torch.einsum("bd,bvd->bv", teacher_mean, views)
    all_view["O2"] = o2_cos[view_mask]
    mode_cos = torch.einsum("bkd,bvd->bvk", modes.float(), views).amax(dim=-1)
    all_view["O3"] = mode_cos[view_mask]
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "scene": "figurines",
        "query_free": {},
        "required_rotation_angle_quantiles_radians": {},
        "context_resultant": {"available": False, "reason": "not defined for the frozen Figurines sentinel"},
        "coverage": {
            "accepted_rows": int(rows.numel()),
            "rows_with_teacher": int(teacher_valid.sum()),
            "rows_without_teacher_using_bitwise_o0_fallback": int((~teacher_valid).sum()),
        },
        "representation": representation,
        "preregistration": file_record(prereg),
    }
    for oracle, values in all_view.items():
        diagnostics["query_free"][oracle] = {
            "mean_all_view_cosine": float(values.mean()),
            "p05_all_view_cosine": float(torch.quantile(values, 0.05)),
        }
    diagnostics["query_free"]["O4"] = {
        "available": False,
        "reason": (
            "O4 is a query-response aggregation over retained views and has no "
            "single query-free descriptor against which all-view cosine is defined"
        ),
        "mean_all_view_cosine": None,
        "p05_all_view_cosine": None,
        "per_scale_mean_all_view_cosine": None,
    }
    for scale in range(3):
        angle = torch.acos(
            (base_features[teacher_valid, scale] * teacher_mean[teacher_valid])
            .sum(dim=-1)
            .clamp(-1.0, 1.0)
        )
        diagnostics["required_rotation_angle_quantiles_radians"][SCALE_IDS[scale]] = {
            key: float(torch.quantile(angle, quantile))
            for key, quantile in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95))
        }
        for oracle, candidate in (("O0", base_features), ("O1", o1.float())):
            values = torch.einsum(
                "bd,bvd->bv", candidate[:, scale], views
            )[view_mask]
            diagnostics["query_free"][oracle].setdefault("per_scale_mean_all_view_cosine", {})[
                SCALE_IDS[scale]
            ] = float(values.mean())
    diagnostics["query_free"]["O2"]["per_scale_mean_all_view_cosine"] = {
        scale_id: float(all_view["O2"].mean()) for scale_id in SCALE_IDS
    }
    generator = torch.Generator().manual_seed(0)
    active_rows = torch.where(teacher_valid)[0]
    sample = active_rows[torch.randperm(active_rows.numel(), generator=generator)[: min(2048, active_rows.numel())]]
    reference = teacher_mean[sample]
    diagnostics["relation_fidelity"] = {
        "O0": {SCALE_IDS[s]: _relation_metrics(reference, base_features[sample, s]) for s in range(3)},
        "O1": {SCALE_IDS[s]: _relation_metrics(reference, o1[sample, s].float()) for s in range(3)},
        "O2": _relation_metrics(reference, teacher_mean[sample]),
        "O3": _relation_metrics(reference, F.normalize((modes[sample].float() * mode_weights[sample, :, None]).sum(dim=1), dim=-1)),
        "O4": {
            "available": False,
            "reason": (
                "O4 has no single query-free descriptor; relation fidelity is not imputed"
            ),
        },
        "sample_rows": int(sample.numel()),
        "seed": 0,
    }
    write_frozen_json(paths["diagnostics"], diagnostics)
    manifest = {
        "schema_version": 1,
        "scene": "figurines",
        "status": "complete_source_only_oracle_matrix_materialization",
        "o0": {
            "positive": file_record(o0_pos_path),
            "negative": file_record(o0_neg_path),
        },
        "o1": {"positive": file_record(paths["o1_positive"]), "negative": file_record(paths["o1_negative"])},
        "o2": {"positive": file_record(paths["o2_positive"]), "negative": file_record(paths["o2_negative"])},
        "o3": {"probability": file_record(paths["o3_probability"])},
        "o4": {"probability": file_record(paths["o4_probability"])},
        "representation": representation,
        "query_free_diagnostics": file_record(paths["diagnostics"]),
        "producer": producer,
        "preregistration": file_record(prereg),
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_metrics_opened": False,
    }
    manifest["matrix_contract_sha256"] = canonical_json_sha256(manifest)
    write_frozen_json(paths["manifest"], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--base-descriptor", required=True)
    parser.add_argument("--expected-base-descriptor-sha256", required=True)
    parser.add_argument("--teacher-authority", required=True)
    parser.add_argument("--expected-teacher-authority-sha256", required=True)
    parser.add_argument("--positive-text-cache", required=True)
    parser.add_argument("--expected-positive-text-cache-sha256", required=True)
    parser.add_argument("--negative-text-cache", required=True)
    parser.add_argument("--expected-negative-text-cache-sha256", required=True)
    parser.add_argument("--o0-positive-cache", required=True)
    parser.add_argument("--expected-o0-positive-cache-sha256", required=True)
    parser.add_argument("--o0-negative-cache", required=True)
    parser.add_argument("--expected-o0-negative-cache-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    print(json.dumps(materialize(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
