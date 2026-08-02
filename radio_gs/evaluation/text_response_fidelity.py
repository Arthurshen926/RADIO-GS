"""CPU-only fidelity metrics for descriptor-to-text cosine responses.

The evaluation axis is intentionally independent for every text query.  It
does not apply a softmax across the bank, so adding or removing one generic
query cannot change another query's target.  Promotion statistics are paired
by descriptor row, query, scene, and training seed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


REPORT_SCHEMA_VERSION = 1
REPORT_ARTIFACT_TYPE = "generic_text_response_fidelity_report"
FROZEN_VALIDATION_SCENE_FILE = (
    Path(__file__).resolve().parents[2]
    / "paper/artifacts/scannet_surface_region_query_free_validation_scenes_20260731.txt"
)
FROZEN_VALIDATION_SCENE_FILE_SHA256 = (
    "2e71b30363f1c9268fd403c32139290e89070478fd0f5badbc54f2dc64665ec9"
)
FROZEN_VALIDATION_SCENES = (
    "scene0164_03",
    "scene0187_00",
    "scene0423_02",
    "scene0553_00",
    "scene0593_00",
    "scene0690_01",
    "scene0699_00",
    "scene0702_00",
)

_REPORT_KEYS = {
    "schema_version",
    "artifact_type",
    "method_id",
    "seed",
    "split_role",
    "query_split",
    "selection_contract",
    "descriptor_artifact",
    "descriptor_rows_sha256",
    "teacher_descriptors_sha256",
    "query_bank",
    "metrics",
}
_QUERY_BANK_KEYS = {
    "path",
    "sha256",
    "manifest_path",
    "manifest_sha256",
    "vocabulary_sha256",
    "query_split",
    "selected_queries",
    "selected_records_sha256",
    "ordered_records_sha256",
    "embedding_tensor_sha256",
    "embedding_semantic_sha256",
    "text_encoder",
}
_SELECTION_CONTRACT = {
    "benchmark_vocabulary_opened": False,
    "uses_benchmark_vocabulary_for_construction": False,
    "queries": "target_blind_imagenet1k_primary_text_bank_v1",
    "query_axis": "heldout_generic_only",
    "device": "cpu",
}
_HOLDOUT_SELECTION_CONTRACT = {
    "benchmark_vocabulary_opened": False,
    "uses_benchmark_vocabulary_for_construction": False,
    "queries": "target_blind_imagenet12k_minus_imagenet1k_holdout_v1",
    "query_axis": "heldout_generic_only",
    "device": "cpu",
}
IMAGENET1K_PRIMARY_BANK_FAMILY = "imagenet1k_primary_v1"
IMAGENET12K_HOLDOUT_BANK_FAMILY = "imagenet12k_minus_imagenet1k_holdout_v1"
_SELECTION_CONTRACTS_BY_BANK_FAMILY = {
    IMAGENET1K_PRIMARY_BANK_FAMILY: _SELECTION_CONTRACT,
    IMAGENET12K_HOLDOUT_BANK_FAMILY: _HOLDOUT_SELECTION_CONTRACT,
}


def selection_contract_for_bank_family(bank_family: str) -> dict:
    """Return the exact frozen report contract for a known query-bank family."""

    if not isinstance(bank_family, str):
        raise ValueError(f"unknown text query-bank family: {bank_family!r}")
    contract = _SELECTION_CONTRACTS_BY_BANK_FAMILY.get(bank_family)
    if contract is None:
        raise ValueError(f"unknown text query-bank family: {bank_family!r}")
    return dict(contract)


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash a tensor in a serialization-independent, little-endian format."""

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    if tensor.is_floating_point():
        array = tensor.to(torch.float32).numpy().astype("<f4", copy=False)
        dtype = "float32-le"
    elif tensor.dtype == torch.bool:
        array = tensor.to(torch.uint8).numpy()
        dtype = "bool-u8"
    elif tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        array = tensor.to(torch.int64).numpy().astype("<i8", copy=False)
        dtype = "int64-le"
    else:
        raise ValueError(f"unsupported tensor dtype for hashing: {tensor.dtype}")
    header = json.dumps(
        {"dtype": dtype, "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_frozen_validation_scenes() -> tuple[str, ...]:
    if not FROZEN_VALIDATION_SCENE_FILE.is_file():
        raise FileNotFoundError(
            f"frozen validation scene file is missing: {FROZEN_VALIDATION_SCENE_FILE}"
        )
    raw = FROZEN_VALIDATION_SCENE_FILE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_VALIDATION_SCENE_FILE_SHA256:
        raise ValueError("frozen validation scene file SHA256 mismatch")
    scenes = tuple(
        line.strip()
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    )
    if scenes != FROZEN_VALIDATION_SCENES or len(scenes) < 2 or len(set(scenes)) != len(scenes):
        raise ValueError("frozen validation scene contents differ from the registered set")
    return scenes


def row_identity_sha256(scene_ids: Sequence[str], region_ids: Sequence[str]) -> str:
    if len(scene_ids) != len(region_ids):
        raise ValueError("scene_ids and region_ids must have the same length")
    return canonical_json_sha256(
        [
            {"scene_id": str(scene), "region_id": str(region)}
            for scene, region in zip(scene_ids, region_ids)
        ]
    )


def _float_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must remain on CPU")
    if tensor.ndim != 2 or not tensor.is_floating_point():
        raise ValueError(f"{name} must be a floating [N,D] tensor")
    if tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must have positive dimensions")
    tensor = tensor.detach().to(torch.float32).contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    if bool((torch.linalg.vector_norm(tensor, dim=-1) <= 1e-12).any()):
        raise ValueError(f"{name} contains a zero-norm row")
    return tensor


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _quantile(values: Sequence[float], q: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic ascending average ranks, including exact ties."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(student: np.ndarray, teacher: np.ndarray) -> float | None:
    teacher_rank = _average_ranks(teacher)
    teacher_rank -= teacher_rank.mean()
    teacher_norm = float(np.linalg.norm(teacher_rank))
    if teacher_norm <= 1e-12:
        # There is no teacher ordering to preserve, so this unit is excluded
        # from rank statistics rather than being assigned an arbitrary score.
        return None
    student_rank = _average_ranks(student)
    student_rank -= student_rank.mean()
    student_norm = float(np.linalg.norm(student_rank))
    if student_norm <= 1e-12:
        return 0.0
    value = float(np.dot(student_rank, teacher_rank) / (student_norm * teacher_norm))
    return float(np.clip(value, -1.0, 1.0))


def _top_decile_overlap(
    student: np.ndarray,
    teacher: np.ndarray,
    region_ids: Sequence[str],
) -> float:
    count = len(region_ids)
    top_count = max(1, int(math.ceil(0.10 * count)))
    tie_break = np.asarray([str(value) for value in region_ids], dtype=object)
    teacher_order = np.lexsort((tie_break, -np.asarray(teacher, dtype=np.float64)))
    student_order = np.lexsort((tie_break, -np.asarray(student, dtype=np.float64)))
    teacher_top = set(int(value) for value in teacher_order[:top_count])
    student_top = set(int(value) for value in student_order[:top_count])
    return float(len(teacher_top & student_top) / top_count)


def evaluate_response_fidelity(
    student_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    scene_ids: Sequence[str],
    region_ids: Sequence[str],
    query_ids: Sequence[str] | None = None,
) -> dict:
    """Compare independent cosine-response fields on a fixed text bank.

    Ranking and top-decile metrics treat regions inside a scene as the support
    axis for one query.  Rank units with a constant teacher response are
    excluded because no teacher ordering exists; top-decile remains defined by
    a deterministic region-ID tie break.
    """

    student = _float_matrix(student_descriptors, "student_descriptors")
    teacher = _float_matrix(teacher_descriptors, "teacher_descriptors")
    text = _float_matrix(text_embeddings, "text_embeddings")
    if student.shape != teacher.shape:
        raise ValueError(
            "student_descriptors and teacher_descriptors must have the same shape"
        )
    if student.shape[1] != text.shape[1]:
        raise ValueError(
            "descriptor/text dimension mismatch: "
            f"{student.shape[1]} vs {text.shape[1]}"
        )
    row_count, query_count = student.shape[0], text.shape[0]
    if len(scene_ids) != row_count or len(region_ids) != row_count:
        raise ValueError("scene_ids and region_ids must align with descriptor rows")
    scenes = [str(value) for value in scene_ids]
    regions = [str(value) for value in region_ids]
    if any(not value for value in scenes) or any(not value for value in regions):
        raise ValueError("scene_ids and region_ids cannot be empty")
    if len(set(zip(scenes, regions))) != row_count:
        raise ValueError("(scene_id, region_id) pairs must be unique")
    if query_ids is None:
        queries = [str(index) for index in range(query_count)]
    else:
        queries = [str(value) for value in query_ids]
        if len(queries) != query_count or len(set(queries)) != query_count:
            raise ValueError("query_ids must be unique and align with text embeddings")

    student_response = F.normalize(student, dim=-1) @ F.normalize(text, dim=-1).T
    teacher_response = F.normalize(teacher, dim=-1) @ F.normalize(text, dim=-1).T
    absolute_error = (student_response - teacher_response).abs()
    smooth_l1 = F.smooth_l1_loss(
        student_response,
        teacher_response,
        reduction="none",
        beta=1.0,
    )
    response_profile_cosine = F.cosine_similarity(
        student_response,
        teacher_response,
        dim=-1,
        eps=1e-12,
    )

    rows_by_scene: dict[str, list[int]] = defaultdict(list)
    for row, scene in enumerate(scenes):
        rows_by_scene[scene].append(row)

    row_metrics = [
        {
            "scene_id": scenes[row],
            "region_id": regions[row],
            "smooth_l1": float(smooth_l1[row].mean()),
            "mae": float(absolute_error[row].mean()),
            "response_profile_cosine": float(response_profile_cosine[row]),
        }
        for row in range(row_count)
    ]
    unit_metrics: list[dict] = []
    scene_metrics: list[dict] = []
    all_rankings: list[float] = []
    all_top_deciles: list[float] = []
    for scene in sorted(rows_by_scene):
        rows = rows_by_scene[scene]
        local_regions = [regions[row] for row in rows]
        local_student = student_response[rows].numpy()
        local_teacher = teacher_response[rows].numpy()
        local_rankings: list[float] = []
        local_top_deciles: list[float] = []
        for query_index, query_id in enumerate(queries):
            ranking = _spearman(
                local_student[:, query_index],
                local_teacher[:, query_index],
            )
            top_decile = _top_decile_overlap(
                local_student[:, query_index],
                local_teacher[:, query_index],
                local_regions,
            )
            if ranking is not None:
                local_rankings.append(ranking)
                all_rankings.append(ranking)
            local_top_deciles.append(top_decile)
            all_top_deciles.append(top_decile)
            unit_metrics.append(
                {
                    "scene_id": scene,
                    "query_id": query_id,
                    "ranking_spearman": ranking,
                    "top_decile_overlap": top_decile,
                }
            )
        local_profiles = response_profile_cosine[rows].tolist()
        scene_metrics.append(
            {
                "scene_id": scene,
                "regions": len(rows),
                "smooth_l1": float(smooth_l1[rows].mean()),
                "mae": float(absolute_error[rows].mean()),
                "response_profile_cosine_mean": _mean(local_profiles),
                "response_profile_cosine_p05": _quantile(local_profiles, 0.05),
                "ranking_spearman_mean": _mean(local_rankings),
                "ranking_spearman_p05": _quantile(local_rankings, 0.05),
                "ranking_valid_queries": len(local_rankings),
                "top_decile_overlap_mean": _mean(local_top_deciles),
                "top_decile_overlap_p05": _quantile(local_top_deciles, 0.05),
            }
        )

    profile_values = response_profile_cosine.tolist()
    return {
        "response_definition": {
            "student": "l2_normalize(student_descriptor) @ l2_normalize(text).T",
            "teacher": "l2_normalize(teacher_descriptor) @ l2_normalize(text).T",
            "query_coupling": "independent_no_softmax",
            "smooth_l1_beta": 1.0,
            "ranking": "scene_query_spearman_over_regions_average_ties",
            "top_decile": "scene_query_overlap_at_ceil_10_percent_regions",
            "quantile_method": "numpy_linear",
        },
        "counts": {
            "regions": row_count,
            "scenes": len(rows_by_scene),
            "queries": query_count,
            "response_cells": row_count * query_count,
            "ranking_valid_scene_queries": len(all_rankings),
            "ranking_total_scene_queries": len(rows_by_scene) * query_count,
        },
        "aggregate": {
            "smooth_l1": float(smooth_l1.mean()),
            "mae": float(absolute_error.mean()),
            "response_profile_cosine_mean": _mean(profile_values),
            "response_profile_cosine_p05": _quantile(profile_values, 0.05),
            "ranking_spearman_mean": _mean(all_rankings),
            "ranking_spearman_p05": _quantile(all_rankings, 0.05),
            "top_decile_overlap_mean": _mean(all_top_deciles),
            "top_decile_overlap_p05": _quantile(all_top_deciles, 0.05),
        },
        "row_metrics": row_metrics,
        "scene_metrics": scene_metrics,
        "unit_metrics": unit_metrics,
        "student_response_sha256": tensor_sha256(student_response),
        "teacher_response_sha256": tensor_sha256(teacher_response),
    }


_ERROR_METRICS = ("smooth_l1", "mae")
_QUALITY_METRICS = (
    "response_profile_cosine_mean",
    "response_profile_cosine_p05",
    "ranking_spearman_mean",
    "ranking_spearman_p05",
    "top_decile_overlap_mean",
    "top_decile_overlap_p05",
)


def _validate_report(report: Mapping) -> None:
    if (
        set(report) != _REPORT_KEYS
        or
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("artifact_type") != REPORT_ARTIFACT_TYPE
    ):
        raise ValueError("invalid text-response fidelity report schema")
    if not isinstance(report.get("seed"), int) or int(report["seed"]) < 0:
        raise ValueError("report seed must be a non-negative integer")
    if not str(report.get("method_id", "")):
        raise ValueError("report method_id is required")
    if report.get("split_role") != "query_free_validation":
        raise ValueError("report split_role must be query_free_validation")
    if report.get("query_split") not in ("dev", "audit"):
        raise ValueError("report query_split must be held-out dev or audit")
    if report.get("selection_contract") not in (
        _SELECTION_CONTRACT,
        _HOLDOUT_SELECTION_CONTRACT,
    ):
        raise ValueError("report selection_contract differs from the frozen policy")
    descriptor_artifact = report.get("descriptor_artifact")
    if (
        not isinstance(descriptor_artifact, Mapping)
        or set(descriptor_artifact) != {"path", "sha256"}
        or not Path(str(descriptor_artifact.get("path", ""))).is_absolute()
        or not str(descriptor_artifact.get("path", ""))
        or not _is_lower_sha256(descriptor_artifact.get("sha256"))
    ):
        raise ValueError("report descriptor_artifact binding is invalid")
    query_bank = report.get("query_bank")
    if not isinstance(query_bank, Mapping) or set(query_bank) != _QUERY_BANK_KEYS:
        raise ValueError("report query_bank binding differs from the frozen schema")
    if (
        query_bank.get("query_split") != report.get("query_split")
        or not isinstance(query_bank.get("selected_queries"), int)
        or int(query_bank.get("selected_queries", 0)) <= 0
        or not Path(str(query_bank.get("path", ""))).is_absolute()
        or not Path(str(query_bank.get("manifest_path", ""))).is_absolute()
    ):
        raise ValueError("report query_bank split/count/path binding is invalid")
    for name in (
        "sha256",
        "manifest_sha256",
        "vocabulary_sha256",
        "selected_records_sha256",
        "ordered_records_sha256",
        "embedding_tensor_sha256",
        "embedding_semantic_sha256",
    ):
        if not _is_lower_sha256(query_bank.get(name)):
            raise ValueError(f"report query_bank {name} is not a lowercase SHA256")
    pairing = _report_pairing_key(report)
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in pairing
    ):
            raise ValueError("report pairing hashes must be lowercase SHA256 values")
    aggregate = report.get("metrics", {}).get("aggregate", {})
    for metric in (*_ERROR_METRICS, *_QUALITY_METRICS):
        value = aggregate.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"report lacks finite aggregate metric {metric}")
    row_metrics = report.get("metrics", {}).get("row_metrics")
    scene_metrics = report.get("metrics", {}).get("scene_metrics")
    unit_metrics = report.get("metrics", {}).get("unit_metrics")
    if not isinstance(row_metrics, list) or not row_metrics:
        raise ValueError("report must contain non-empty row_metrics")
    if not isinstance(scene_metrics, list) or not scene_metrics:
        raise ValueError("report must contain non-empty scene_metrics")
    if not isinstance(unit_metrics, list) or not unit_metrics:
        raise ValueError("report must contain non-empty unit_metrics")

    row_values = {
        "smooth_l1": [],
        "mae": [],
        "response_profile_cosine": [],
    }
    rows_by_scene: dict[str, list[Mapping]] = defaultdict(list)
    row_keys = set()
    for row in row_metrics:
        key = (str(row.get("scene_id", "")), str(row.get("region_id", "")))
        values = {name: row.get(name) for name in row_values}
        if (
            not all(key)
            or key in row_keys
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values.values()
            )
            or float(values["smooth_l1"]) < 0.0
            or float(values["mae"]) < 0.0
            or not -1.000001
            <= float(values["response_profile_cosine"])
            <= 1.000001
        ):
            raise ValueError("row_metrics contain invalid identities or values")
        row_keys.add(key)
        rows_by_scene[key[0]].append(row)
        for name, value in values.items():
            row_values[name].append(float(value))

    ranking_values, top_values = [], []
    units_by_scene: dict[str, list[Mapping]] = defaultdict(list)
    unit_keys = set()
    for row in unit_metrics:
        key = (str(row.get("scene_id", "")), str(row.get("query_id", "")))
        ranking = row.get("ranking_spearman")
        top = row.get("top_decile_overlap")
        if (
            not all(key)
            or key in unit_keys
            or not isinstance(top, (int, float))
            or not math.isfinite(float(top))
            or not 0.0 <= float(top) <= 1.0
        ):
            raise ValueError("unit_metrics contain invalid identities or top-decile values")
        if ranking is not None:
            if (
                not isinstance(ranking, (int, float))
                or not math.isfinite(float(ranking))
                or not -1.0 <= float(ranking) <= 1.0
            ):
                raise ValueError("unit_metrics contain an invalid ranking value")
            ranking_values.append(float(ranking))
        top_values.append(float(top))
        unit_keys.add(key)
        units_by_scene[key[0]].append(row)
    if not ranking_values:
        raise ValueError("report has no valid ranking units")

    def assert_close(actual: object, expected: float, name: str) -> None:
        if (
            not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(
                # Torch reductions over [R,Q] and the same persisted per-row
                # float32 means can differ by a few ulps because their
                # accumulation order is different.  This tolerance is still
                # seven orders below the registered non-regression gates.
                float(actual), float(expected), rel_tol=0.0, abs_tol=1e-7
            )
        ):
            raise ValueError(f"{name} does not match detailed metrics")

    exact_aggregate = {
        "smooth_l1": float(np.mean(row_values["smooth_l1"])),
        "mae": float(np.mean(row_values["mae"])),
        "response_profile_cosine_mean": float(
            np.mean(row_values["response_profile_cosine"])
        ),
        "response_profile_cosine_p05": float(
            np.quantile(row_values["response_profile_cosine"], 0.05)
        ),
        "ranking_spearman_mean": float(np.mean(ranking_values)),
        "ranking_spearman_p05": float(np.quantile(ranking_values, 0.05)),
        "top_decile_overlap_mean": float(np.mean(top_values)),
        "top_decile_overlap_p05": float(np.quantile(top_values, 0.05)),
    }
    for metric, expected in exact_aggregate.items():
        assert_close(aggregate.get(metric), expected, f"aggregate {metric}")

    scenes: dict[str, Mapping] = {}
    for row in scene_metrics:
        scene = str(row.get("scene_id", ""))
        if not scene or scene in scenes:
            raise ValueError("scene_metrics contain duplicate/empty scene IDs")
        scenes[scene] = row
    if set(scenes) != set(rows_by_scene) or set(scenes) != set(units_by_scene):
        raise ValueError("scene_metrics do not cover the detailed scene identities")
    query_ids: set[str] | None = None
    for scene, scene_row in scenes.items():
        local_rows = rows_by_scene[scene]
        local_units = units_by_scene[scene]
        local_queries = {str(row["query_id"]) for row in local_units}
        if query_ids is None:
            query_ids = local_queries
        elif local_queries != query_ids:
            raise ValueError("every scene must contain the same query identities")
        if scene_row.get("regions") != len(local_rows):
            raise ValueError("scene_metrics region count does not match row_metrics")
        local_rankings = [
            float(row["ranking_spearman"])
            for row in local_units
            if row.get("ranking_spearman") is not None
        ]
        if not local_rankings:
            raise ValueError(f"scene {scene} has no valid ranking units")
        local_tops = [float(row["top_decile_overlap"]) for row in local_units]
        local_profiles = [
            float(row["response_profile_cosine"]) for row in local_rows
        ]
        local_expected = {
            "smooth_l1": float(
                np.mean([float(row["smooth_l1"]) for row in local_rows])
            ),
            "mae": float(np.mean([float(row["mae"]) for row in local_rows])),
            "response_profile_cosine_mean": float(np.mean(local_profiles)),
            "response_profile_cosine_p05": float(
                np.quantile(local_profiles, 0.05)
            ),
            "ranking_spearman_mean": float(np.mean(local_rankings)),
            "ranking_spearman_p05": float(np.quantile(local_rankings, 0.05)),
            "top_decile_overlap_mean": float(np.mean(local_tops)),
            "top_decile_overlap_p05": float(np.quantile(local_tops, 0.05)),
        }
        if scene_row.get("ranking_valid_queries") != len(local_rankings):
            raise ValueError("scene ranking_valid_queries does not match unit_metrics")
        for metric, expected in local_expected.items():
            assert_close(
                scene_row.get(metric), expected, f"scene {scene} {metric}"
            )

    counts = report.get("metrics", {}).get("counts", {})
    query_count = len(query_ids or ())
    expected_counts = {
        "regions": len(row_metrics),
        "scenes": len(scenes),
        "queries": query_count,
        "response_cells": len(row_metrics) * query_count,
        "ranking_valid_scene_queries": len(ranking_values),
        "ranking_total_scene_queries": len(unit_metrics),
    }
    for name, expected in expected_counts.items():
        if counts.get(name) != expected:
            raise ValueError(f"report count {name} does not match detailed metrics")


def _index_reports(reports: Iterable[Mapping], role: str) -> dict[int, Mapping]:
    result: dict[int, Mapping] = {}
    method_ids = set()
    for report in reports:
        _validate_report(report)
        seed = int(report["seed"])
        if seed in result:
            raise ValueError(f"duplicate {role} report for seed {seed}")
        result[seed] = report
        method_ids.add(str(report["method_id"]))
    if len(method_ids) != 1:
        raise ValueError(f"{role} reports must share one method_id")
    return result


def _scene_index(report: Mapping) -> dict[str, Mapping]:
    result = {}
    for row in report["metrics"]["scene_metrics"]:
        scene = str(row.get("scene_id", ""))
        if not scene or scene in result:
            raise ValueError("scene_metrics contain duplicate/empty scene IDs")
        result[scene] = row
    return result


def _unit_index(report: Mapping) -> dict[tuple[str, str], Mapping]:
    result = {}
    for row in report["metrics"]["unit_metrics"]:
        key = (str(row.get("scene_id", "")), str(row.get("query_id", "")))
        if not all(key) or key in result:
            raise ValueError("unit_metrics contain duplicate/empty identities")
        result[key] = row
    return result


def _row_index(report: Mapping) -> dict[tuple[str, str], Mapping]:
    result = {}
    for row in report["metrics"]["row_metrics"]:
        key = (str(row.get("scene_id", "")), str(row.get("region_id", "")))
        if not all(key) or key in result:
            raise ValueError("row_metrics contain duplicate/empty identities")
        result[key] = row
    return result


def _report_pairing_key(report: Mapping) -> tuple[str, str, str, str]:
    return (
        str(report.get("query_bank", {}).get("selected_records_sha256", "")),
        str(report.get("query_bank", {}).get("embedding_tensor_sha256", "")),
        str(report.get("descriptor_rows_sha256", "")),
        str(report.get("teacher_descriptors_sha256", "")),
    )


def _is_lower_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _recompute_report_from_bound_artifacts(
    report: Mapping,
    *,
    phase: str,
    hash_cache: dict[Path, str],
    descriptor_cache: dict[Path, dict],
    bank_cache: dict[tuple[Path, Path, str], dict],
) -> Mapping:
    # Lazy import avoids a module-import cycle: the CLI module imports this
    # metric module, while this strict gate needs its artifact validators only
    # after both modules are initialized.
    from radio_gs.scripts.eval_text_response_fidelity_gate import evaluate_artifacts

    descriptor_record = report["descriptor_artifact"]
    query_bank = report["query_bank"]
    return evaluate_artifacts(
        Path(str(descriptor_record["path"])),
        Path(str(query_bank["path"])),
        Path(str(query_bank["manifest_path"])),
        query_split=phase,
        _hash_cache=hash_cache,
        _descriptor_cache=descriptor_cache,
        _bank_cache=bank_cache,
    )


def _percentile_interval(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "low": float(np.quantile(array, 0.025)),
        "high": float(np.quantile(array, 0.975)),
    }


def _aggregate_paired_seed_gate_impl(
    control_reports: Iterable[Mapping],
    candidate_reports: Iterable[Mapping],
    *,
    required_seeds: Sequence[int] = (0, 1, 2),
    minimum_improved_seeds: int = 2,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260731,
    quality_noninferiority_tolerance: float = 0.0,
    phase: str | None = None,
    _test_expected_scene_ids: Sequence[str] | None = None,
    _test_report_recomputer=None,
    _verified_same_process_source_authority: Mapping | None = None,
) -> dict:
    """Build a scene-clustered paired multi-seed promotion decision."""

    if phase not in ("dev", "audit"):
        raise ValueError("phase must be explicitly set to dev or audit")
    if (_test_expected_scene_ids is None) != (_test_report_recomputer is None):
        raise ValueError(
            "internal test scene/recompute injections must be supplied together"
        )
    if (
        _verified_same_process_source_authority is not None
        and _test_report_recomputer is not None
    ):
        raise ValueError("verified same-process aggregation cannot use test injections")
    if _test_expected_scene_ids is None:
        expected_scene_ids = _load_frozen_validation_scenes()
    else:
        expected_scene_ids = tuple(str(value) for value in _test_expected_scene_ids)
    if len(expected_scene_ids) < 2 or len(set(expected_scene_ids)) != len(
        expected_scene_ids
    ):
        raise ValueError("the preregistered gate requires at least two unique scenes")
    expected_scene_set = set(expected_scene_ids)

    raw_controls = list(control_reports)
    raw_candidates = list(candidate_reports)
    for role, reports in (("control", raw_controls), ("candidate", raw_candidates)):
        for report in reports:
            _validate_report(report)
            if report.get("query_split") != phase:
                raise ValueError(f"{role} report query_split differs from phase {phase}")

    hash_cache: dict[Path, str] = {}
    descriptor_cache: dict[Path, dict] = {}
    bank_cache: dict[tuple[Path, Path, str], dict] = {}
    verified: dict[str, list[Mapping]] = {"control": [], "candidate": []}
    for role, reports in (("control", raw_controls), ("candidate", raw_candidates)):
        for report in reports:
            if _verified_same_process_source_authority is not None:
                recomputed = report
            elif _test_report_recomputer is None:
                recomputed = _recompute_report_from_bound_artifacts(
                    report,
                    phase=phase,
                    hash_cache=hash_cache,
                    descriptor_cache=descriptor_cache,
                    bank_cache=bank_cache,
                )
            else:
                recomputed = _test_report_recomputer(report, phase)
            if dict(report) != dict(recomputed):
                raise ValueError(
                    f"{role} report differs from strict source-artifact recomputation"
                )
            verified[role].append(recomputed)

    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if quality_noninferiority_tolerance < 0:
        raise ValueError("quality_noninferiority_tolerance must be non-negative")
    seeds = tuple(int(value) for value in required_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(value < 0 for value in seeds):
        raise ValueError("required_seeds must be unique non-negative integers")
    if not 1 <= int(minimum_improved_seeds) <= len(seeds):
        raise ValueError("minimum_improved_seeds is outside the seed count")

    controls = _index_reports(verified["control"], "control")
    candidates = _index_reports(verified["candidate"], "candidate")
    if set(controls) != set(seeds) or set(candidates) != set(seeds):
        raise ValueError(
            f"control/candidate reports must exactly cover required seeds {list(seeds)}"
        )
    if str(controls[seeds[0]]["method_id"]) == str(
        candidates[seeds[0]]["method_id"]
    ):
        raise ValueError("control and candidate method_id must be different")

    reference_pairing = _report_pairing_key(controls[seeds[0]])
    if any(not value for value in reference_pairing):
        raise ValueError("reports lack strict query/descriptor pairing hashes")
    scene_tables: dict[int, tuple[dict[str, Mapping], dict[str, Mapping]]] = {}
    row_tables: dict[int, tuple[dict[tuple[str, str], Mapping], dict[tuple[str, str], Mapping]]] = {}
    unit_tables: dict[int, tuple[dict[tuple[str, str], Mapping], dict[tuple[str, str], Mapping]]] = {}
    for seed in seeds:
        control = controls[seed]
        candidate = candidates[seed]
        if control.get("query_split") != phase or candidate.get("query_split") != phase:
            raise ValueError(f"seed {seed} report split differs from phase {phase}")
        if _report_pairing_key(control) != reference_pairing:
            raise ValueError("control reports do not share one paired evaluation contract")
        if _report_pairing_key(candidate) != reference_pairing:
            raise ValueError(f"candidate seed {seed} is not paired to the control rows/bank")
        control_scenes, candidate_scenes = _scene_index(control), _scene_index(candidate)
        if set(control_scenes) != set(candidate_scenes):
            raise ValueError(f"scene mismatch for seed {seed}")
        control_units, candidate_units = _unit_index(control), _unit_index(candidate)
        if set(control_units) != set(candidate_units):
            raise ValueError(f"scene/query unit mismatch for seed {seed}")
        scene_tables[seed] = (control_scenes, candidate_scenes)
        control_rows, candidate_rows = _row_index(control), _row_index(candidate)
        if set(control_rows) != set(candidate_rows):
            raise ValueError(f"scene/region row mismatch for seed {seed}")
        row_tables[seed] = (control_rows, candidate_rows)
        unit_tables[seed] = (control_units, candidate_units)

    scene_ids = sorted(scene_tables[seeds[0]][0])
    if set(scene_ids) != expected_scene_set or len(scene_ids) != len(expected_scene_ids):
        raise ValueError("reports do not cover the complete preregistered scene set")
    if any(sorted(scene_tables[seed][0]) != scene_ids for seed in seeds):
        raise ValueError("all seeds must cover exactly the same scenes")

    per_seed = []
    improvement_counts = {metric: 0 for metric in _ERROR_METRICS}
    for seed in seeds:
        control_aggregate = controls[seed]["metrics"]["aggregate"]
        candidate_aggregate = candidates[seed]["metrics"]["aggregate"]
        error_improvement = {
            metric: float(control_aggregate[metric] - candidate_aggregate[metric])
            for metric in _ERROR_METRICS
        }
        quality_delta = {
            metric: float(candidate_aggregate[metric] - control_aggregate[metric])
            for metric in _QUALITY_METRICS
        }
        for metric, value in error_improvement.items():
            improvement_counts[metric] += int(value > 0.0)
        per_seed.append(
            {
                "seed": seed,
                "error_improvement_control_minus_candidate": error_improvement,
                "quality_delta_candidate_minus_control": quality_delta,
            }
        )

    def sampled_metric(metric: str, sampled_scenes: Sequence[str]) -> float:
        values = []
        if metric in _ERROR_METRICS:
            for seed in seeds:
                control_scene, candidate_scene = scene_tables[seed]
                values.extend(
                    float(control_scene[scene][metric] - candidate_scene[scene][metric])
                    for scene in sampled_scenes
                )
            return float(np.mean(values))
        if metric in (
            "response_profile_cosine_mean",
            "response_profile_cosine_p05",
        ):
            control_values, candidate_values = [], []
            for seed in seeds:
                control_rows, candidate_rows = row_tables[seed]
                for scene in sampled_scenes:
                    keys = sorted(key for key in control_rows if key[0] == scene)
                    for key in keys:
                        control_values.append(
                            float(control_rows[key]["response_profile_cosine"])
                        )
                        candidate_values.append(
                            float(candidate_rows[key]["response_profile_cosine"])
                        )
            if metric.endswith("_p05"):
                return float(
                    np.quantile(candidate_values, 0.05)
                    - np.quantile(control_values, 0.05)
                )
            return float(np.mean(candidate_values) - np.mean(control_values))
        if metric.endswith("_p05"):
            unit_key = (
                "ranking_spearman"
                if metric == "ranking_spearman_p05"
                else "top_decile_overlap"
                if metric == "top_decile_overlap_p05"
                else None
            )
            if unit_key is not None:
                control_values, candidate_values = [], []
                for seed in seeds:
                    control_units, candidate_units = unit_tables[seed]
                    for scene in sampled_scenes:
                        keys = sorted(key for key in control_units if key[0] == scene)
                        for key in keys:
                            control_value = control_units[key].get(unit_key)
                            candidate_value = candidate_units[key].get(unit_key)
                            if control_value is None or candidate_value is None:
                                continue
                            control_values.append(float(control_value))
                            candidate_values.append(float(candidate_value))
                if not control_values:
                    raise ValueError(f"no paired valid units for {metric}")
                return float(
                    np.quantile(candidate_values, 0.05)
                    - np.quantile(control_values, 0.05)
                )
        # Mean quality metrics and response-profile p05 are already summarized
        # per scene.  Their scene-clustered delta is the mean paired scene delta.
        for seed in seeds:
            control_scene, candidate_scene = scene_tables[seed]
            values.extend(
                float(candidate_scene[scene][metric] - control_scene[scene][metric])
                for scene in sampled_scenes
            )
        return float(np.mean(values))

    all_metrics = (*_ERROR_METRICS, *_QUALITY_METRICS)
    point_estimates = {
        metric: sampled_metric(metric, scene_ids) for metric in all_metrics
    }
    rng = np.random.default_rng(int(bootstrap_seed))
    samples = {metric: [] for metric in all_metrics}
    for _ in range(int(bootstrap_samples)):
        sampled = [scene_ids[index] for index in rng.integers(0, len(scene_ids), len(scene_ids))]
        for metric in all_metrics:
            samples[metric].append(sampled_metric(metric, sampled))
    intervals = {metric: _percentile_interval(samples[metric]) for metric in all_metrics}

    checks = {
        "smooth_l1_improves_required_seeds": (
            improvement_counts["smooth_l1"] >= int(minimum_improved_seeds)
        ),
        "mae_improves_required_seeds": (
            improvement_counts["mae"] >= int(minimum_improved_seeds)
        ),
        "smooth_l1_scene_ci_strictly_positive": intervals["smooth_l1"]["low"] > 0.0,
        "mae_scene_ci_strictly_positive": intervals["mae"]["low"] > 0.0,
    }
    tolerance = float(quality_noninferiority_tolerance)
    for metric in _QUALITY_METRICS:
        checks[f"{metric}_scene_ci_noninferior"] = intervals[metric]["low"] >= -tolerance

    protocol = {
        "phase": phase,
        "query_split": phase,
        "phase_semantics": (
            "selection_only" if phase == "dev" else "confirmation_only_no_retuning"
        ),
        "preregistered_scene_file_sha256": (
            FROZEN_VALIDATION_SCENE_FILE_SHA256
            if _test_expected_scene_ids is None
            else "internal_test_contract"
        ),
        "required_seeds": list(seeds),
        "minimum_improved_seeds": int(minimum_improved_seeds),
        "bootstrap_unit": "scene",
        "bootstrap_pairing": "control_candidate_within_seed_scene",
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "ci": "percentile_95",
        "error_direction": "control_minus_candidate_positive_is_better",
        "quality_direction": "candidate_minus_control_positive_is_better",
        "quality_noninferiority_tolerance": tolerance,
    }
    if _verified_same_process_source_authority is not None:
        protocol["embedded_metrics_source_authority"] = dict(
            _verified_same_process_source_authority
        )
    return {
        "schema_version": 1,
        "artifact_type": "generic_text_response_paired_seed_gate",
        "decision": "promote" if all(checks.values()) else "reject",
        "checks": checks,
        "protocol": protocol,
        "pairing": {
            "selected_records_sha256": reference_pairing[0],
            "embedding_tensor_sha256": reference_pairing[1],
            "descriptor_rows_sha256": reference_pairing[2],
            "teacher_descriptors_sha256": reference_pairing[3],
            "scenes": scene_ids,
        },
        "control_method_id": str(controls[seeds[0]]["method_id"]),
        "candidate_method_id": str(candidates[seeds[0]]["method_id"]),
        "per_seed": per_seed,
        "improved_seed_counts": improvement_counts,
        "point_estimates": point_estimates,
        "scene_bootstrap_ci95": intervals,
    }


def aggregate_paired_seed_gate(
    control_reports: Iterable[Mapping],
    candidate_reports: Iterable[Mapping],
    *,
    required_seeds: Sequence[int] = (0, 1, 2),
    minimum_improved_seeds: int = 2,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260731,
    quality_noninferiority_tolerance: float = 0.0,
    phase: str | None = None,
    _test_expected_scene_ids: Sequence[str] | None = None,
    _test_report_recomputer=None,
) -> dict:
    """Build a gate from independently materialized descriptor artifacts."""

    return _aggregate_paired_seed_gate_impl(
        control_reports,
        candidate_reports,
        required_seeds=required_seeds,
        minimum_improved_seeds=minimum_improved_seeds,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        quality_noninferiority_tolerance=quality_noninferiority_tolerance,
        phase=phase,
        _test_expected_scene_ids=_test_expected_scene_ids,
        _test_report_recomputer=_test_report_recomputer,
    )


_EMBEDDED_RESPONSE_RECORD_KEYS = {
    "method_id",
    "seed",
    "query_split",
    "descriptor_rows_sha256",
    "teacher_descriptors_sha256",
    "query_bank",
    "metrics",
}
_SAME_PROCESS_AUTHORITY_KEYS = {
    "schema_version",
    "authority_type",
    "selection",
    "diagnostic",
    "validation_caches",
    "endpoints",
    "audit_bank",
    "construction",
}


def _validate_live_file_record(record: object, label: str) -> dict[str, str]:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} file record fields differ")
    path = Path(str(record.get("path", "")))
    digest = str(record.get("sha256", ""))
    if (
        not path.is_absolute()
        or path.resolve(strict=True) != path
        or not path.is_file()
        or not _is_lower_sha256(digest)
    ):
        raise ValueError(f"{label} must bind one canonical regular file")
    observed = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            observed.update(chunk)
    if observed.hexdigest() != digest:
        raise ValueError(f"{label} SHA256 mismatch")
    return {"path": str(path), "sha256": digest}


def _validate_same_process_source_authority(
    value: object,
    *,
    required_seeds: Sequence[int],
) -> dict:
    """Validate the immutable inputs behind same-process embedded metrics."""

    if (
        not isinstance(value, Mapping)
        or set(value) != _SAME_PROCESS_AUTHORITY_KEYS
        or value.get("schema_version") != 1
        or value.get("authority_type")
        != "same_process_interpolation_audit_response_metrics_v1"
        or value.get("construction")
        != "evaluate_response_fidelity_from_cpu_descriptors_recomputed_after_selection_validation"
    ):
        raise ValueError("same-process source authority schema differs")
    normalized = {
        "schema_version": 1,
        "authority_type": value["authority_type"],
        "selection": _validate_live_file_record(value["selection"], "selection"),
        "diagnostic": _validate_live_file_record(value["diagnostic"], "diagnostic"),
        "construction": value["construction"],
    }
    caches = value.get("validation_caches")
    if not isinstance(caches, list) or not caches:
        raise ValueError("source authority validation caches are missing")
    normalized["validation_caches"] = [
        _validate_live_file_record(record, f"validation cache {index}")
        for index, record in enumerate(caches)
    ]
    endpoints = value.get("endpoints")
    seeds = tuple(int(seed) for seed in required_seeds)
    if not isinstance(endpoints, list) or len(endpoints) != len(seeds):
        raise ValueError("source authority endpoints do not cover required seeds")
    normalized_endpoints = []
    observed_seeds = set()
    for row in endpoints:
        if not isinstance(row, Mapping) or set(row) != {"seed", "control", "candidate"}:
            raise ValueError("source authority endpoint row fields differ")
        seed = row.get("seed")
        if seed not in seeds or seed in observed_seeds:
            raise ValueError("source authority endpoint seed differs")
        observed_seeds.add(seed)
        normalized_endpoints.append(
            {
                "seed": seed,
                "control": _validate_live_file_record(
                    row["control"], f"seed {seed} control endpoint"
                ),
                "candidate": _validate_live_file_record(
                    row["candidate"], f"seed {seed} candidate endpoint"
                ),
            }
        )
    if observed_seeds != set(seeds):
        raise ValueError("source authority endpoint seeds differ")
    normalized["endpoints"] = normalized_endpoints
    audit = value.get("audit_bank")
    if (
        not isinstance(audit, Mapping)
        or set(audit) != {"artifact", "manifest", "split", "query_count"}
        or audit.get("split") != "audit"
        or not isinstance(audit.get("query_count"), int)
        or int(audit["query_count"]) <= 0
    ):
        raise ValueError("source authority audit-bank fields differ")
    normalized["audit_bank"] = {
        "artifact": _validate_live_file_record(audit["artifact"], "audit bank"),
        "manifest": _validate_live_file_record(audit["manifest"], "audit-bank manifest"),
        "split": "audit",
        "query_count": int(audit["query_count"]),
    }
    return normalized


def aggregate_paired_seed_gate_from_same_process_metrics(
    control_records: Iterable[Mapping],
    candidate_records: Iterable[Mapping],
    *,
    source_authority: Mapping,
    required_seeds: Sequence[int] = (0, 1, 2),
    minimum_improved_seeds: int = 2,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260731,
    quality_noninferiority_tolerance: float = 0.0,
    phase: str | None = None,
) -> dict:
    """Aggregate metrics computed from strictly bound tensors in this process.

    Unlike :func:`aggregate_paired_seed_gate`, this production entry does not
    claim that its embedded reports are independently materialized descriptor
    artifacts.  It instead binds and rehashes the frozen selection, diagnostic,
    endpoints, caches, and audit bank that produced the tensors.  Metric detail
    is still validated with the ordinary report invariants before aggregation.
    """

    if phase != "audit":
        raise ValueError("same-process metric aggregation is audit-only")
    authority = _validate_same_process_source_authority(
        source_authority,
        required_seeds=required_seeds,
    )
    selection_record = authority["selection"]

    def as_internal_report(record: Mapping, role: str) -> dict:
        if not isinstance(record, Mapping) or set(record) != _EMBEDDED_RESPONSE_RECORD_KEYS:
            raise ValueError(f"{role} embedded response record fields differ")
        if record.get("query_split") != phase:
            raise ValueError(f"{role} embedded response split differs")
        query_bank = record.get("query_bank")
        if (
            not isinstance(query_bank, Mapping)
            or query_bank.get("selected_queries")
            != authority["audit_bank"]["query_count"]
            or {
                "path": query_bank.get("path"),
                "sha256": query_bank.get("sha256"),
            }
            != authority["audit_bank"]["artifact"]
            or {
                "path": query_bank.get("manifest_path"),
                "sha256": query_bank.get("manifest_sha256"),
            }
            != authority["audit_bank"]["manifest"]
        ):
            raise ValueError(f"{role} embedded response audit-bank binding differs")
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "artifact_type": REPORT_ARTIFACT_TYPE,
            "method_id": record.get("method_id"),
            "seed": record.get("seed"),
            "split_role": "query_free_validation",
            "query_split": record.get("query_split"),
            "selection_contract": dict(_SELECTION_CONTRACT),
            # Internal validation requires the ordinary report shape.  This
            # binding is not exposed as a descriptor artifact in the embedded
            # record; protocol provenance names it as the derivation authority.
            "descriptor_artifact": selection_record,
            "descriptor_rows_sha256": record.get("descriptor_rows_sha256"),
            "teacher_descriptors_sha256": record.get("teacher_descriptors_sha256"),
            "query_bank": dict(query_bank),
            "metrics": record.get("metrics"),
        }
        _validate_report(report)
        return report

    controls = [as_internal_report(record, "control") for record in control_records]
    candidates = [
        as_internal_report(record, "candidate") for record in candidate_records
    ]
    return _aggregate_paired_seed_gate_impl(
        controls,
        candidates,
        required_seeds=required_seeds,
        minimum_improved_seeds=minimum_improved_seeds,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        quality_noninferiority_tolerance=quality_noninferiority_tolerance,
        phase=phase,
        _verified_same_process_source_authority=authority,
    )
