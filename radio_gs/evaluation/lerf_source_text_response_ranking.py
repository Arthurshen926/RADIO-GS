"""Target-blind LERF source-view text-response and spatial-ranking gate.

The evaluator operates on one source-heldout frame at a time.  Text queries
are independent cosine directions; pixels, not query IDs, are the ranking
axis.  This preserves the capability relevant to text selection without
opening benchmark vocabulary, masks, labels, or target metrics.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import (
    _spearman,
    _top_decile_overlap,
    canonical_json_sha256,
    tensor_sha256,
)


FRAME_SCHEMA = "radio_gs.lerf_source_text_response_frame_summary.v1"
SCENE_SCHEMA = "radio_gs.lerf_source_text_response_scene_summary.v1"
GATE_SCHEMA = "radio_gs.lerf_source_text_response_ranking_gate.v1"
SCHEMA_VERSION = 1
QUERY_BANK_KEYS = {
    "path",
    "sha256",
    "manifest_path",
    "manifest_sha256",
    "query_split",
    "queries",
    "embedding_tensor_sha256",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_IMPLEMENTATION_PATH = Path(__file__).resolve()
FRAME_EVALUATOR_IMPLEMENTATION = {
    "path": str(_IMPLEMENTATION_PATH),
    "sha256": _file_sha256(_IMPLEMENTATION_PATH),
}


def _query_bank(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != QUERY_BANK_KEYS:
        raise ValueError("source text-response query-bank binding differs")
    result = dict(value)
    if result["query_split"] not in {"dev", "audit"}:
        raise ValueError("source text-response query split differs")
    if not isinstance(result["queries"], int) or int(result["queries"]) <= 1:
        raise ValueError("source text-response query count differs")
    for key in ("sha256", "manifest_sha256", "embedding_tensor_sha256"):
        digest = str(result[key])
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"source text-response query-bank {key} differs")
    for key in ("path", "manifest_path"):
        if not str(result[key]).startswith("/"):
            raise ValueError(f"source text-response query-bank {key} is not absolute")
    return result


def _descriptor_map(value: torch.Tensor, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.device.type != "cpu":
        raise ValueError(f"{label} must remain on CPU")
    if tensor.ndim != 3 or not tensor.is_floating_point():
        raise ValueError(f"{label} must be floating [D,H,W]")
    tensor = tensor.detach().to(torch.float32).contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite")
    return tensor


def evaluate_source_frame(
    method_descriptor_map: torch.Tensor,
    teacher_descriptor_map: torch.Tensor,
    text_embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    scene_id: str,
    frame_id: int,
    method_id: str,
    query_ids: Sequence[str],
    query_bank: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate real query responses over one legal source-heldout frame."""

    method = _descriptor_map(method_descriptor_map, label="method descriptor map")
    teacher = _descriptor_map(teacher_descriptor_map, label="teacher descriptor map")
    if method.shape != teacher.shape:
        raise ValueError("method/teacher source-frame descriptor shapes differ")
    text = torch.as_tensor(text_embeddings)
    if text.device.type != "cpu" or text.ndim != 2 or not text.is_floating_point():
        raise ValueError("source text embeddings must be floating [Q,D] on CPU")
    text = text.detach().to(torch.float32).contiguous()
    if text.shape[1] != method.shape[0] or not bool(torch.isfinite(text).all()):
        raise ValueError("source text embedding axis differs")
    queries = [str(value) for value in query_ids]
    if (
        len(queries) != text.shape[0]
        or len(set(queries)) != len(queries)
        or any(not value for value in queries)
    ):
        raise ValueError("source query identities differ")
    bank = _query_bank(query_bank)
    if int(bank["queries"]) != len(queries):
        raise ValueError("source query-bank count differs from runtime embeddings")
    mask = torch.as_tensor(valid_mask)
    if mask.device.type != "cpu" or mask.shape != method.shape[1:]:
        raise ValueError("source-frame valid mask shape/device differs")
    mask = mask.bool().contiguous()
    if int(mask.sum()) < 10:
        raise ValueError("source-frame response ranking requires at least 10 pixels")
    if not str(scene_id) or not str(method_id):
        raise ValueError("source-frame scene/method identity is required")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError("source-frame ID must be a non-negative integer")

    method_unit = F.normalize(method, dim=0, eps=1e-12)
    teacher_unit = F.normalize(teacher, dim=0, eps=1e-12)
    text_unit = F.normalize(text, dim=-1, eps=1e-12)
    method_response = torch.einsum("qd,dhw->qhw", text_unit, method_unit)
    teacher_response = torch.einsum("qd,dhw->qhw", text_unit, teacher_unit)
    return evaluate_source_response_frame(
        method_response,
        teacher_response,
        mask,
        scene_id=scene_id,
        frame_id=frame_id,
        method_id=method_id,
        query_ids=queries,
        query_bank=bank,
        method_input_sha256=tensor_sha256(method[:, mask].T.contiguous()),
        teacher_input_sha256=tensor_sha256(teacher[:, mask].T.contiguous()),
    )


def evaluate_source_response_frame(
    method_response_map: torch.Tensor,
    teacher_response_map: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    scene_id: str,
    frame_id: int,
    method_id: str,
    query_ids: Sequence[str],
    query_bank: Mapping[str, object],
    method_input_sha256: str,
    teacher_input_sha256: str,
) -> dict[str, object]:
    """Evaluate precomputed independent query responses for one source frame."""

    method = torch.as_tensor(method_response_map)
    teacher = torch.as_tensor(teacher_response_map)
    if (
        method.device.type != "cpu"
        or teacher.device.type != "cpu"
        or method.ndim != 3
        or teacher.shape != method.shape
        or not method.is_floating_point()
        or not teacher.is_floating_point()
    ):
        raise ValueError("source response maps must be paired floating [Q,H,W] on CPU")
    method = method.detach().to(torch.float32).contiguous()
    teacher = teacher.detach().to(torch.float32).contiguous()
    if not bool(torch.isfinite(method).all()) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("source response maps must be finite")
    mask = torch.as_tensor(valid_mask)
    if mask.device.type != "cpu" or mask.shape != method.shape[1:]:
        raise ValueError("source response valid mask shape/device differs")
    mask = mask.bool().contiguous()
    if int(mask.sum()) < 10:
        raise ValueError("source-frame response ranking requires at least 10 pixels")
    queries = [str(value) for value in query_ids]
    bank = _query_bank(query_bank)
    if (
        len(queries) != method.shape[0]
        or len(queries) != int(bank["queries"])
        or len(set(queries)) != len(queries)
        or any(not value for value in queries)
    ):
        raise ValueError("source response query identities differ")
    for digest, label in (
        (method_input_sha256, "method input"),
        (teacher_input_sha256, "teacher input"),
    ):
        if len(str(digest)) != 64 or any(
            value not in "0123456789abcdef" for value in str(digest)
        ):
            raise ValueError(f"source response {label} SHA-256 differs")
    if not str(scene_id) or not str(method_id):
        raise ValueError("source-frame scene/method identity is required")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError("source-frame ID must be a non-negative integer")

    coordinates = mask.nonzero(as_tuple=False)
    pixel_ids = [f"y{int(y):04d}_x{int(x):04d}" for y, x in coordinates]
    selected_method = method[:, mask].T.contiguous()
    selected_teacher = teacher[:, mask].T.contiguous()
    absolute_error = (selected_method - selected_teacher).abs()
    smooth_l1 = F.smooth_l1_loss(
        selected_method, selected_teacher, reduction="none", beta=1.0
    )
    profile_cosine = F.cosine_similarity(
        selected_method, selected_teacher, dim=-1, eps=1e-12
    )
    method_numpy = selected_method.numpy()
    teacher_numpy = selected_teacher.numpy()
    query_units: list[dict[str, object]] = []
    for query_index, query_id in enumerate(queries):
        ranking = _spearman(
            method_numpy[:, query_index], teacher_numpy[:, query_index]
        )
        top_decile = _top_decile_overlap(
            method_numpy[:, query_index],
            teacher_numpy[:, query_index],
            pixel_ids,
        )
        query_units.append(
            {
                "query_id": query_id,
                "ranking_spearman": ranking,
                "top_decile_overlap": top_decile,
            }
        )
    response_cells = int(selected_method.numel())
    return {
        "schema": FRAME_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": str(scene_id),
        "frame_id": int(frame_id),
        "method_id": str(method_id),
        "producer": dict(FRAME_EVALUATOR_IMPLEMENTATION),
        "query_bank": bank,
        "query_ids_sha256": canonical_json_sha256(queries),
        "valid_pixel_count": len(pixel_ids),
        "pixel_ids_sha256": canonical_json_sha256(pixel_ids),
        "method_input_sha256": str(method_input_sha256),
        "teacher_input_sha256": str(teacher_input_sha256),
        "method_response_sha256": tensor_sha256(selected_method),
        "teacher_response_sha256": tensor_sha256(selected_teacher),
        "sufficient_statistics": {
            "response_cell_count": response_cells,
            "absolute_error_sum": float(absolute_error.double().sum()),
            "smooth_l1_sum": float(smooth_l1.double().sum()),
            "response_profile_cosine_sum": float(profile_cosine.double().sum()),
            "response_profile_count": len(pixel_ids),
        },
        "query_units": query_units,
        "access_audit": {
            "source_heldout_view_opened": True,
            "target_blind_generic_query_bank_opened": True,
            "benchmark_queries_opened": False,
            "benchmark_masks_or_labels_opened": False,
            "target_metric_executed": False,
            "device": "cpu",
        },
    }


def _finite(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def validate_frame_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("source text-response frame summary must be an object")
    frame = dict(value)
    if frame.get("schema") != FRAME_SCHEMA or frame.get("schema_version") != 1:
        raise ValueError("source text-response frame summary schema differs")
    if not str(frame.get("scene_id", "")) or not str(frame.get("method_id", "")):
        raise ValueError("source text-response frame identity differs")
    if frame.get("producer") != FRAME_EVALUATOR_IMPLEMENTATION:
        raise ValueError("source text-response frame producer differs")
    if not isinstance(frame.get("frame_id"), int) or isinstance(
        frame.get("frame_id"), bool
    ):
        raise ValueError("source text-response frame ID differs")
    bank = _query_bank(frame.get("query_bank"))
    for key in (
        "query_ids_sha256",
        "pixel_ids_sha256",
        "method_input_sha256",
        "teacher_input_sha256",
        "method_response_sha256",
        "teacher_response_sha256",
    ):
        digest = str(frame.get(key, ""))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"source text-response frame {key} differs")
    pixels = frame.get("valid_pixel_count")
    if not isinstance(pixels, int) or pixels < 10:
        raise ValueError("source text-response valid pixel count differs")
    statistics = frame.get("sufficient_statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("source text-response sufficient statistics differ")
    cells = statistics.get("response_cell_count")
    if cells != pixels * int(bank["queries"]):
        raise ValueError("source text-response cell count differs")
    for key in (
        "absolute_error_sum",
        "smooth_l1_sum",
        "response_profile_cosine_sum",
    ):
        _finite(statistics.get(key), label=key)
    if statistics.get("response_profile_count") != pixels:
        raise ValueError("source text-response profile count differs")
    units = frame.get("query_units")
    if not isinstance(units, list) or len(units) != int(bank["queries"]):
        raise ValueError("source text-response query units differ")
    query_ids: set[str] = set()
    ordered_query_ids: list[str] = []
    valid_rankings = 0
    for unit in units:
        if not isinstance(unit, Mapping) or set(unit) != {
            "query_id",
            "ranking_spearman",
            "top_decile_overlap",
        }:
            raise ValueError("source text-response query unit schema differs")
        query_id = str(unit["query_id"])
        if not query_id or query_id in query_ids:
            raise ValueError("source text-response query unit identity differs")
        query_ids.add(query_id)
        ordered_query_ids.append(query_id)
        ranking = unit["ranking_spearman"]
        if ranking is not None:
            value_rank = _finite(ranking, label="ranking_spearman")
            if not -1.0 <= value_rank <= 1.0:
                raise ValueError("source text-response ranking range differs")
            valid_rankings += 1
        top = _finite(unit["top_decile_overlap"], label="top_decile_overlap")
        if not 0.0 <= top <= 1.0:
            raise ValueError("source text-response top-decile range differs")
    if valid_rankings == 0:
        raise ValueError("source text-response frame has no identifiable ranking")
    if canonical_json_sha256(ordered_query_ids) != frame["query_ids_sha256"]:
        raise ValueError("source text-response query identity hash differs")
    access = frame.get("access_audit")
    if not isinstance(access, Mapping) or any(
        access.get(key) is not False
        for key in (
            "benchmark_queries_opened",
            "benchmark_masks_or_labels_opened",
            "target_metric_executed",
        )
    ):
        raise ValueError("source text-response frame is not target blind")
    return frame


def _aggregate_frames(frames: Sequence[Mapping[str, object]]) -> dict[str, object]:
    cells = sum(
        int(frame["sufficient_statistics"]["response_cell_count"])
        for frame in frames
    )
    profiles = sum(
        int(frame["sufficient_statistics"]["response_profile_count"])
        for frame in frames
    )
    rankings = [
        float(unit["ranking_spearman"])
        for frame in frames
        for unit in frame["query_units"]
        if unit["ranking_spearman"] is not None
    ]
    top_deciles = [
        float(unit["top_decile_overlap"])
        for frame in frames
        for unit in frame["query_units"]
    ]
    if not rankings or not top_deciles or cells <= 0 or profiles <= 0:
        raise ValueError("source text-response aggregation is empty")
    return {
        "frames": len(frames),
        "valid_pixels": profiles,
        "response_cells": cells,
        "ranking_valid_units": len(rankings),
        "ranking_total_units": len(top_deciles),
        "response_mae": sum(
            float(frame["sufficient_statistics"]["absolute_error_sum"])
            for frame in frames
        )
        / cells,
        "response_smooth_l1": sum(
            float(frame["sufficient_statistics"]["smooth_l1_sum"])
            for frame in frames
        )
        / cells,
        "response_profile_cosine_mean": sum(
            float(frame["sufficient_statistics"]["response_profile_cosine_sum"])
            for frame in frames
        )
        / profiles,
        "ranking_spearman_mean": float(np.mean(rankings)),
        "ranking_spearman_p05": float(np.quantile(rankings, 0.05)),
        "top_decile_overlap_mean": float(np.mean(top_deciles)),
        "top_decile_overlap_p05": float(np.quantile(top_deciles, 0.05)),
    }


def build_scene_summary(
    frame_summaries: Sequence[Mapping[str, object]],
    *,
    scene_id: str,
    method_id: str,
    required_frame_ids: Sequence[int],
) -> dict[str, object]:
    frames = [validate_frame_summary(value) for value in frame_summaries]
    required = [int(value) for value in required_frame_ids]
    if required != sorted(set(required)) or not required:
        raise ValueError("required source-heldout frame IDs must be sorted and unique")
    if [int(frame["frame_id"]) for frame in frames] != required:
        raise ValueError("source text-response frame set/order differs")
    if any(frame["scene_id"] != scene_id for frame in frames):
        raise ValueError("source text-response scene identity differs")
    if any(frame["method_id"] != method_id for frame in frames):
        raise ValueError("source text-response method identity differs")
    bank = frames[0]["query_bank"]
    query_sha = frames[0]["query_ids_sha256"]
    if any(
        frame["query_bank"] != bank or frame["query_ids_sha256"] != query_sha
        for frame in frames
    ):
        raise ValueError("source text-response query axis differs across frames")
    if any(
        frame["producer"] != FRAME_EVALUATOR_IMPLEMENTATION for frame in frames
    ):
        raise ValueError("source text-response frame producer differs across frames")
    return {
        "schema": SCENE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": str(scene_id),
        "method_id": str(method_id),
        "frame_evaluator_implementation": dict(FRAME_EVALUATOR_IMPLEMENTATION),
        "source_heldout_frame_ids": required,
        "query_bank": bank,
        "query_ids_sha256": query_sha,
        "frames": frames,
        "aggregate": _aggregate_frames(frames),
        "access_audit": {
            "source_only": True,
            "benchmark_queries_masks_or_labels_opened": False,
            "target_metric_executed": False,
        },
    }


def validate_scene_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("source text-response scene summary must be an object")
    scene = dict(value)
    if scene.get("schema") != SCENE_SCHEMA or scene.get("schema_version") != 1:
        raise ValueError("source text-response scene summary schema differs")
    rebuilt = build_scene_summary(
        scene.get("frames", []),
        scene_id=str(scene.get("scene_id", "")),
        method_id=str(scene.get("method_id", "")),
        required_frame_ids=scene.get("source_heldout_frame_ids", []),
    )
    if rebuilt != scene:
        raise ValueError("source text-response scene summary is not self-consistent")
    return scene


def _paired_metric_deltas(
    control: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, float]:
    return {
        "response_mae_improvement": float(control["response_mae"])
        - float(candidate["response_mae"]),
        "response_profile_cosine_delta": float(
            candidate["response_profile_cosine_mean"]
        )
        - float(control["response_profile_cosine_mean"]),
        "ranking_spearman_mean_delta": float(candidate["ranking_spearman_mean"])
        - float(control["ranking_spearman_mean"]),
        "ranking_spearman_p05_delta": float(candidate["ranking_spearman_p05"])
        - float(control["ranking_spearman_p05"]),
        "top_decile_overlap_mean_delta": float(candidate["top_decile_overlap_mean"])
        - float(control["top_decile_overlap_mean"]),
        "top_decile_overlap_p05_delta": float(candidate["top_decile_overlap_p05"])
        - float(control["top_decile_overlap_p05"]),
    }


def paired_source_gate(
    control_summaries: Sequence[Mapping[str, object]],
    candidate_summaries: Sequence[Mapping[str, object]],
    *,
    required_scene_ids: Sequence[str],
) -> dict[str, object]:
    """Apply one target-blind, globally shared response/ranking gate."""

    required = [str(value) for value in required_scene_ids]
    if len(required) < 2 or required != sorted(set(required)):
        raise ValueError("source text-response gate requires >=2 sorted unique scenes")
    controls = {
        scene["scene_id"]: scene
        for scene in map(validate_scene_summary, control_summaries)
    }
    candidates = {
        scene["scene_id"]: scene
        for scene in map(validate_scene_summary, candidate_summaries)
    }
    if sorted(controls) != required or sorted(candidates) != required:
        raise ValueError("source text-response gate scene cohort differs")
    control_methods = {str(scene["method_id"]) for scene in controls.values()}
    candidate_methods = {str(scene["method_id"]) for scene in candidates.values()}
    if len(control_methods) != 1 or len(candidate_methods) != 1:
        raise ValueError("source text-response methods differ across scenes")
    if control_methods == candidate_methods:
        raise ValueError("source text-response control and candidate must differ")
    all_summaries = list(controls.values()) + list(candidates.values())
    first_query_bank = all_summaries[0]["query_bank"]
    first_query_ids_sha256 = all_summaries[0]["query_ids_sha256"]
    if any(
        scene["query_bank"] != first_query_bank
        or scene["query_ids_sha256"] != first_query_ids_sha256
        for scene in all_summaries
    ):
        raise ValueError("source text-response query axis differs across scenes")

    scene_results: list[dict[str, object]] = []
    all_control_frames: list[Mapping[str, object]] = []
    all_candidate_frames: list[Mapping[str, object]] = []
    for scene_id in required:
        control = controls[scene_id]
        candidate = candidates[scene_id]
        if (
            control["source_heldout_frame_ids"]
            != candidate["source_heldout_frame_ids"]
            or control["query_bank"] != candidate["query_bank"]
            or control["query_ids_sha256"] != candidate["query_ids_sha256"]
        ):
            raise ValueError("paired source text-response scene axes differ")
        for control_frame, candidate_frame in zip(
            control["frames"], candidate["frames"]
        ):
            for key in (
                "frame_id",
                "valid_pixel_count",
                "pixel_ids_sha256",
                "teacher_input_sha256",
                "teacher_response_sha256",
                "query_ids_sha256",
            ):
                if control_frame[key] != candidate_frame[key]:
                    raise ValueError(f"paired source text-response frame {key} differs")
        deltas = _paired_metric_deltas(control["aggregate"], candidate["aggregate"])
        nonregression = all(value >= 0.0 for value in deltas.values())
        scene_results.append(
            {
                "scene_id": scene_id,
                "control": control["aggregate"],
                "candidate": candidate["aggregate"],
                "deltas": deltas,
                "all_metrics_nonregressing": nonregression,
            }
        )
        all_control_frames.extend(control["frames"])
        all_candidate_frames.extend(candidate["frames"])

    pooled_control = _aggregate_frames(all_control_frames)
    pooled_candidate = _aggregate_frames(all_candidate_frames)
    pooled_deltas = _paired_metric_deltas(pooled_control, pooled_candidate)
    strict_joint_improvement = all(
        pooled_deltas[key] > 0.0
        for key in (
            "response_mae_improvement",
            "ranking_spearman_mean_delta",
            "top_decile_overlap_mean_delta",
        )
    )
    every_scene_nonregressing = all(
        bool(scene["all_metrics_nonregressing"]) for scene in scene_results
    )
    passed = strict_joint_improvement and every_scene_nonregressing
    return {
        "schema": GATE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "rejected",
        "control_method_id": next(iter(control_methods)),
        "candidate_method_id": next(iter(candidate_methods)),
        "required_scene_ids": required,
        "query_bank": controls[required[0]]["query_bank"],
        "scene_results": scene_results,
        "pooled": {
            "control": pooled_control,
            "candidate": pooled_candidate,
            "deltas": pooled_deltas,
        },
        "decision": {
            "strict_pooled_response_ranking_and_top_decile_improvement": (
                strict_joint_improvement
            ),
            "every_scene_all_metrics_nonregressing": every_scene_nonregressing,
            "candidate_eligible_for_next_source_gate": passed,
        },
        "protocol": {
            "query_coupling": "independent_no_softmax",
            "ranking_axis": "valid_pixels_within_source_heldout_frame",
            "one_global_policy": True,
            "per_scene_or_per_query_tuning": False,
            "benchmark_queries_masks_or_labels_opened": False,
            "target_metric_execution_authorized": False,
        },
    }


__all__ = [
    "FRAME_EVALUATOR_IMPLEMENTATION",
    "FRAME_SCHEMA",
    "GATE_SCHEMA",
    "SCENE_SCHEMA",
    "build_scene_summary",
    "evaluate_source_frame",
    "evaluate_source_response_frame",
    "paired_source_gate",
    "validate_frame_summary",
    "validate_scene_summary",
]
