#!/usr/bin/env python3
"""Evaluate ScanNet-UQIS mesh-probability predictions.

The public seam is :func:`evaluate_predictions`, which consumes already
loaded manifests and arrays.  The filesystem wrapper verifies private mesh
bindings before calling that pure evaluator.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .method_fields import validate_method_field_inventory
from .metrics import evaluate_query_probabilities, scene_clustered_bootstrap
from .protocol import (
    BENCHMARK_VERSION,
    PREDICTION_DOMAIN,
    QUERY_MANIFEST_NAMES,
    UQISProtocolConfig,
    audit_release,
    canonical_json_sha256,
)


MODALITIES = ("text", "image", "point_2d", "point_3d")
METRIC_KEYS = (
    "average_precision",
    "oracle_iou",
    "fixed_iou_0.5",
    "acc_at_iou_0.25",
    "acc_at_iou_0.50",
    "selected_purity",
    "positive_coverage",
    "same_class_distractor_iou",
    "centroid_error_m",
)
def _records(value: Any, *, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a non-empty sequence")
    records = list(value)
    if not records or not all(isinstance(record, Mapping) for record in records):
        raise ValueError(f"{label} must be a non-empty sequence of records")
    return records


def _query_id(value: Any, *, target_id: str, modality: str) -> str:
    if isinstance(value, Mapping):
        value = value.get("query_id", "")
    query_id = str(value)
    if not query_id or Path(query_id).name != query_id or query_id in {".", ".."}:
        raise ValueError(f"{target_id}/{modality}: invalid query_id")
    return query_id


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    return {key: _mean(rows, key) for key in METRIC_KEYS}


def _cohort_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    modalities: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
    """Aggregate one evaluator-private cohort without exposing target pairing."""

    if not rows:
        raise ValueError("evaluation cohort must contain at least one query")
    selected = tuple(map(str, modalities))
    scene_ids = sorted({str(row["scene_id"]) for row in rows})
    modality_reports: dict[str, Any] = {}
    scene_metrics_by_modality: dict[str, dict[str, dict[str, float | None]]] = {}
    for modality in selected:
        modality_rows = [row for row in rows if row["modality"] == modality]
        if not modality_rows:
            raise ValueError(f"evaluation cohort lacks modality {modality}")
        by_scene = {
            scene_id: _aggregate(
                [
                    row
                    for row in modality_rows
                    if str(row["scene_id"]) == scene_id
                ]
            )
            for scene_id in sorted({str(row["scene_id"]) for row in modality_rows})
        }
        scene_metrics_by_modality[modality] = by_scene
        scene_macro: dict[str, float | None] = {}
        intervals: dict[str, Any] = {}
        for key in METRIC_KEYS:
            values = {
                scene_id: float(metrics[key])
                for scene_id, metrics in by_scene.items()
                if metrics[key] is not None
            }
            scene_macro[key] = float(np.mean(list(values.values()))) if values else None
            intervals[key] = (
                scene_clustered_bootstrap(
                    values,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                    confidence=confidence,
                )
                if values
                else None
            )
        modality_reports[modality] = {
            "query_count": len(modality_rows),
            "scene_count": len(by_scene),
            "same_class_distractor_query_count": sum(
                row["same_class_distractor_iou"] is not None
                for row in modality_rows
            ),
            "query_micro": _aggregate(modality_rows),
            "scene_macro": scene_macro,
            "scene_clustered_ci": intervals,
        }

    complete = set(selected) == set(MODALITIES)
    unified: dict[str, Any] = {}
    if complete:
        if any(set(scene_metrics_by_modality[m]) != set(scene_ids) for m in MODALITIES):
            raise ValueError("unified cohort modalities do not cover the same scenes")
        for name, metric in (("uq_rank", "average_precision"), ("uq_mask", "fixed_iou_0.5")):
            per_scene = {
                scene_id: float(
                    np.mean(
                        [
                            scene_metrics_by_modality[modality][scene_id][metric]
                            for modality in MODALITIES
                        ]
                    )
                )
                for scene_id in scene_ids
            }
            unified[name] = {
                "metric": metric,
                "value": float(np.mean(list(per_scene.values()))),
                "scene_clustered_ci": scene_clustered_bootstrap(
                    per_scene,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                    confidence=confidence,
                ),
            }
    return {
        "query_count": len(rows),
        "target_count": len(rows) // len(selected),
        "scene_count": len(scene_ids),
        "modalities": modality_reports,
        **unified,
    }


def _normalized_scene_arrays(
    values: Mapping[str, np.ndarray], *, label: str
) -> dict[str, np.ndarray]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{label} must be a mapping keyed by scene_id")
    normalized = {str(key): np.asarray(value) for key, value in values.items()}
    if len(normalized) != len(values):
        raise ValueError(f"{label} contains colliding scene ids")
    return normalized


def evaluate_predictions(
    evaluator_manifest: Mapping[str, Any],
    scene_manifest: Mapping[str, Any],
    predictions_by_query: Mapping[str, np.ndarray],
    mesh_xyz_by_scene: Mapping[str, np.ndarray],
    mesh_instance_ids_by_scene: Mapping[str, np.ndarray],
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260813,
    confidence: float = 0.95,
    modalities: Sequence[str] = MODALITIES,
) -> dict[str, Any]:
    """Evaluate a complete system or one declared modality comparator.

    This array-level seam is intentionally diagnostic and never declares a
    formal row.  Formal-looking evaluation must go through
    :func:`evaluate_release`, which seals predictions before opening private
    target identities and binds the release and method receipts.
    """

    if not isinstance(evaluator_manifest, Mapping) or not isinstance(
        scene_manifest, Mapping
    ):
        raise ValueError("evaluator and scene manifests must be mappings")
    evaluator_version = str(evaluator_manifest.get("benchmark_version", ""))
    scene_version = str(scene_manifest.get("benchmark_version", ""))
    if evaluator_version != BENCHMARK_VERSION or scene_version != BENCHMARK_VERSION:
        raise ValueError("evaluator and scene manifests must use the frozen UQIS version")
    selected_modalities = tuple(map(str, modalities))
    if (
        not selected_modalities
        or len(set(selected_modalities)) != len(selected_modalities)
        or any(value not in MODALITIES for value in selected_modalities)
    ):
        raise ValueError("modalities must be a unique non-empty UQIS modality subset")
    if len(selected_modalities) not in {1, len(MODALITIES)}:
        raise ValueError("row scope must be one modality comparator or all four modalities")

    scene_records = _records(scene_manifest.get("scene_domains"), label="scene_domains")
    declared_scenes: set[str] = set()
    for record in scene_records:
        scene_id = str(record.get("scene_id", ""))
        if not scene_id or scene_id in declared_scenes:
            raise ValueError("scene_domains contains an invalid or duplicate scene_id")
        declared_scenes.add(scene_id)

    target_records = _records(evaluator_manifest.get("targets"), label="targets")
    tier_presence = ["evaluation_tier" in target for target in target_records]
    if any(tier_presence) and not all(tier_presence):
        raise ValueError("evaluation_tier must be declared for every target or none")
    tiered_evaluation = bool(tier_presence and all(tier_presence))
    seen_targets: set[str] = set()
    seen_queries: set[str] = set()
    query_records: list[dict[str, Any]] = []
    target_scenes: set[str] = set()
    for target in target_records:
        target_id = str(target.get("target_id", ""))
        scene_id = str(target.get("scene_id", ""))
        if not target_id or target_id in seen_targets:
            raise ValueError("targets contains an invalid or duplicate target_id")
        if not scene_id or scene_id not in declared_scenes:
            raise ValueError(f"{target_id}: target references an undeclared scene")
        evaluation_tier = (
            str(target.get("evaluation_tier", "")) if tiered_evaluation else None
        )
        if tiered_evaluation and evaluation_tier not in {
            "unified_core",
            "relational_text_challenge",
        }:
            raise ValueError(f"{target_id}: invalid evaluation_tier")
        seen_targets.add(target_id)
        target_scenes.add(scene_id)
        try:
            instance_id = int(target["instance_id"])
            nyu40_class_id = int(target["nyu40_class_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{target_id}: invalid instance/class identity") from error
        distractor_source = target.get("same_class_distractor_instance_ids", [])
        if not isinstance(distractor_source, Sequence) or isinstance(
            distractor_source, (str, bytes)
        ):
            raise ValueError(f"{target_id}: distractor ids must be a sequence")
        try:
            distractors = [int(value) for value in distractor_source]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{target_id}: invalid distractor instance id") from error
        queries = target.get("queries")
        if not isinstance(queries, Mapping) or set(queries) != set(MODALITIES):
            raise ValueError(f"{target_id}: queries must contain exactly {MODALITIES}")
        for modality in selected_modalities:
            query_id = _query_id(queries[modality], target_id=target_id, modality=modality)
            if query_id in seen_queries:
                raise ValueError(f"duplicate query_id: {query_id}")
            seen_queries.add(query_id)
            query_records.append(
                {
                    "query_id": query_id,
                    "target_id": target_id,
                    "scene_id": scene_id,
                    "modality": modality,
                    "instance_id": instance_id,
                    "nyu40_class_id": nyu40_class_id,
                    "same_class_distractor_instance_ids": distractors,
                    "evaluation_tier": evaluation_tier,
                }
            )
    if target_scenes != declared_scenes:
        raise ValueError("scene domains and target scenes do not align exactly")

    if not isinstance(predictions_by_query, Mapping):
        raise ValueError("predictions must be a mapping keyed by query_id")
    prediction_keys = {str(key) for key in predictions_by_query}
    if len(prediction_keys) != len(predictions_by_query):
        raise ValueError("predictions contain colliding query ids")
    missing = sorted(seen_queries - prediction_keys)
    unexpected = sorted(prediction_keys - seen_queries)
    if missing or unexpected:
        raise ValueError(
            f"prediction set is incomplete: missing={missing}, unexpected={unexpected}"
        )

    xyz_by_scene = _normalized_scene_arrays(mesh_xyz_by_scene, label="mesh xyz")
    ids_by_scene = _normalized_scene_arrays(
        mesh_instance_ids_by_scene, label="mesh instance ids"
    )
    if set(xyz_by_scene) != declared_scenes or set(ids_by_scene) != declared_scenes:
        raise ValueError("loaded mesh domains and scene manifest do not align exactly")
    for scene_id in sorted(declared_scenes):
        xyz = xyz_by_scene[scene_id]
        instance_ids = ids_by_scene[scene_id]
        if xyz.ndim != 2 or xyz.shape[1:] != (3,) or not len(xyz):
            raise ValueError(f"{scene_id}: mesh xyz must have shape [V, 3]")
        if instance_ids.ndim != 1 or instance_ids.shape != (len(xyz),):
            raise ValueError(f"{scene_id}: mesh instance ids must have shape [V]")
        if not np.issubdtype(instance_ids.dtype, np.integer):
            raise ValueError(f"{scene_id}: mesh instance ids must be integers")
        if not np.isfinite(xyz).all():
            raise ValueError(f"{scene_id}: mesh xyz must be finite")

    rows: list[dict[str, Any]] = []
    for query in query_records:
        scene_id = str(query["scene_id"])
        query_id = str(query["query_id"])
        scores = np.asarray(predictions_by_query[query_id])
        metrics = evaluate_query_probabilities(
            scores,
            target_instance_id=int(query["instance_id"]),
            same_class_distractor_instance_ids=query[
                "same_class_distractor_instance_ids"
            ],
            mesh_instance_ids=ids_by_scene[scene_id],
            mesh_xyz=xyz_by_scene[scene_id],
        )
        rows.append({**query, **metrics})

    rows_by_scene_modality: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_scene_modality[(str(row["scene_id"]), str(row["modality"]))].append(row)
    per_scene: dict[str, dict[str, Any]] = {}
    complete_system = set(selected_modalities) == set(MODALITIES)
    for scene_id in sorted(declared_scenes):
        modality_metrics = {
            modality: _aggregate(rows_by_scene_modality[(scene_id, modality)])
            for modality in selected_modalities
        }
        per_scene[scene_id] = {
            **modality_metrics,
            "uq_mean": (
                float(
                    np.mean(
                        [
                            modality_metrics[value]["fixed_iou_0.5"]
                            for value in MODALITIES
                        ]
                    )
                )
                if complete_system
                else None
            ),
        }

    modalities: dict[str, Any] = {}
    for modality in selected_modalities:
        modality_rows = [row for row in rows if row["modality"] == modality]
        scene_metrics = {
            scene_id: per_scene[scene_id][modality] for scene_id in sorted(per_scene)
        }
        scene_macro: dict[str, float | None] = {}
        intervals: dict[str, Any] = {}
        for key in METRIC_KEYS:
            values = {
                scene_id: float(metrics[key])
                for scene_id, metrics in scene_metrics.items()
                if metrics[key] is not None
            }
            scene_macro[key] = float(np.mean(list(values.values()))) if values else None
            intervals[key] = (
                scene_clustered_bootstrap(
                    values,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                    confidence=confidence,
                )
                if values
                else None
            )
        modalities[modality] = {
            "query_count": len(modality_rows),
            "scene_count": len(scene_metrics),
            "same_class_distractor_query_count": sum(
                row["same_class_distractor_iou"] is not None for row in modality_rows
            ),
            "query_micro": _aggregate(modality_rows),
            "scene_macro": scene_macro,
            "scene_clustered_ci": intervals,
        }

    scene_uq = (
        {scene_id: float(value["uq_mean"]) for scene_id, value in per_scene.items()}
        if complete_system
        else {}
    )
    uq_interval = (
        scene_clustered_bootstrap(
            scene_uq,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            confidence=confidence,
        )
        if complete_system
        else None
    )
    core_summary = None
    relational_text = None
    if tiered_evaluation:
        core_rows = [row for row in rows if row["evaluation_tier"] == "unified_core"]
        if not core_rows:
            raise ValueError("tiered evaluation has no Unified-Query Core Cohort")
        core_summary = _cohort_summary(
            core_rows,
            modalities=selected_modalities,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            confidence=confidence,
        )
        if "text" in selected_modalities:
            relational_rows = [
                row
                for row in rows
                if row["evaluation_tier"] == "relational_text_challenge"
                and row["modality"] == "text"
            ]
            if relational_rows:
                relational_text = _cohort_summary(
                    relational_rows,
                    modalities=("text",),
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                    confidence=confidence,
                )["modalities"]["text"]
    return {
        "benchmark": BENCHMARK_VERSION,
        "evaluation_mode": "diagnostic_unsealed_arrays",
        "formal_benchmark_eligible": False,
        "row_scope": "universal_complete" if complete_system else "modality_comparator",
        "evaluated_modalities": list(selected_modalities),
        "protocol": {
            "prediction_domain": PREDICTION_DOMAIN,
            "probability_range": [0.0, 1.0],
            "fixed_threshold": 0.5,
            "average_precision": "non_interpolated_tie_aware_binary_ap",
            "oracle_iou": "maximum_over_complete_score_tie_thresholds",
            "same_class_distractor_iou": "maximum_iou_over_declared_same_class_instances",
            "empty_centroid_penalty": "maximum_scene_vertex_distance_from_target_centroid",
            "aggregation": "query_mean_within_scene_then_equal_scene_mean",
        },
        "query_count": len(rows),
        "target_count": len(target_records),
        "scene_count": len(declared_scenes),
        "core_target_count": (
            core_summary["target_count"] if core_summary is not None else None
        ),
        "core_modalities": (
            core_summary["modalities"] if core_summary is not None else None
        ),
        "relational_text_target_count": (
            relational_text["query_count"] if relational_text is not None else None
        ),
        "modalities": modalities,
        "uq_rank": core_summary.get("uq_rank") if core_summary is not None else None,
        "uq_mask": (
            {
                **core_summary["uq_mask"],
                "calibration_status": "diagnostic_unverified",
            }
            if core_summary is not None and "uq_mask" in core_summary
            else None
        ),
        "relational_text_challenge": relational_text,
        "uq_mean": (
            {
                "metric": "fixed_iou_0.5",
                "value": float(np.mean(list(scene_uq.values()))),
                "scene_clustered_ci": uq_interval,
            }
            if complete_system
            else None
        ),
        "per_scene": per_scene,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path, *, required_key: str) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON manifest: {path}") from error
    if not isinstance(document, Mapping):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    if required_key in document:
        return document
    payload = document.get("payload")
    if not isinstance(payload, Mapping) or required_key not in payload:
        raise ValueError(f"manifest does not contain {required_key}: {path}")
    normalized = dict(payload)
    if "benchmark_version" not in normalized and "benchmark_version" in document:
        normalized["benchmark_version"] = document["benchmark_version"]
    return normalized


def _asset_binding(
    record: Mapping[str, Any], *, role: str, manifest_dir: Path
) -> tuple[Path, str]:
    path_value = record.get(f"{role}_path")
    digest_value = record.get(f"{role}_sha256", record.get(f"{role}_hash"))
    nested = record.get(role)
    if isinstance(nested, Mapping):
        path_value = path_value or nested.get("path")
        digest_value = digest_value or nested.get("sha256", nested.get("hash"))
    path_text = str(path_value or "")
    expected = str(digest_value or "")
    if (
        len(expected) != 64
        or expected.lower() != expected
        or any(value not in "0123456789abcdef" for value in expected)
    ):
        raise ValueError(f"{role} requires a lowercase SHA-256 binding")
    if not path_text:
        raise ValueError(f"{role} requires a path")
    path = Path(path_text)
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{role} file does not exist: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{role} SHA-256 mismatch: {path}")
    return path, actual


def _load_npy(path: Path, *, label: str) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load {label}: {path}") from error
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{label} must be a single NPY array: {path}")
    return value


def _expected_query_ids(evaluator_manifest: Mapping[str, Any]) -> set[str]:
    expected: set[str] = set()
    for target in _records(evaluator_manifest.get("targets"), label="targets"):
        target_id = str(target.get("target_id", ""))
        queries = target.get("queries")
        if not isinstance(queries, Mapping) or set(queries) != set(MODALITIES):
            raise ValueError(f"{target_id}: queries must contain exactly {MODALITIES}")
        for modality in MODALITIES:
            query_id = _query_id(queries[modality], target_id=target_id, modality=modality)
            if query_id in expected:
                raise ValueError(f"duplicate query_id: {query_id}")
            expected.add(query_id)
    return expected


def _load_evaluation_inputs(
    evaluator_path: Path, scenes_path: Path
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Open public mesh geometry and evaluator-private instance identities."""

    evaluator_manifest = _load_manifest(evaluator_path, required_key="targets")
    scene_manifest = _load_manifest(scenes_path, required_key="scene_domains")
    public_records = _records(
        scene_manifest.get("scene_domains"), label="scene_domains"
    )
    public_by_scene: dict[str, Mapping[str, Any]] = {}
    for record in public_records:
        scene_id = str(record.get("scene_id", ""))
        if not scene_id or scene_id in public_by_scene:
            raise ValueError("scene_domains contains an invalid or duplicate scene_id")
        public_by_scene[scene_id] = record
    private_source = evaluator_manifest.get("scene_domains", public_records)
    private_records = _records(private_source, label="evaluator scene_domains")
    private_by_scene: dict[str, Mapping[str, Any]] = {}
    for record in private_records:
        scene_id = str(record.get("scene_id", ""))
        if not scene_id or scene_id in private_by_scene:
            raise ValueError(
                "evaluator scene_domains contains an invalid or duplicate scene_id"
            )
        private_by_scene[scene_id] = record
    if set(private_by_scene) != set(public_by_scene):
        raise ValueError("public and evaluator-private scene domains differ")

    xyz_by_scene: dict[str, np.ndarray] = {}
    ids_by_scene: dict[str, np.ndarray] = {}
    for scene_id in sorted(public_by_scene):
        public_xyz_path, public_xyz_hash = _asset_binding(
            public_by_scene[scene_id],
            role="mesh_xyz",
            manifest_dir=scenes_path.parent,
        )
        private_record = private_by_scene[scene_id]
        private_xyz_path, private_xyz_hash = _asset_binding(
            private_record, role="mesh_xyz", manifest_dir=evaluator_path.parent
        )
        if (
            public_xyz_path != private_xyz_path
            or public_xyz_hash != private_xyz_hash
        ):
            raise ValueError(f"{scene_id}: public/private mesh xyz bindings differ")
        ids_path, _ = _asset_binding(
            private_record,
            role="mesh_instance_ids",
            manifest_dir=evaluator_path.parent,
        )
        xyz_by_scene[scene_id] = _load_npy(
            public_xyz_path, label=f"{scene_id} mesh xyz"
        )
        ids_by_scene[scene_id] = _load_npy(
            ids_path, label=f"{scene_id} mesh instance ids"
        )
    return evaluator_manifest, scene_manifest, xyz_by_scene, ids_by_scene


def evaluate(
    evaluator_manifest_path: str | Path,
    scene_manifest_path: str | Path,
    prediction_dir: str | Path,
    output: str | Path | None = None,
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260813,
    confidence: float = 0.95,
    modalities: Sequence[str] = MODALITIES,
) -> dict[str, Any]:
    """Load two manifests for a result-ineligible diagnostic evaluation.

    This compatibility seam is useful for unit tests and private diagnostics.
    It does not bind ``release.json`` or a sealed method receipt, so its report
    always remains formally ineligible.  Use :func:`evaluate_release` for a
    release-bound evaluation.
    """

    evaluator_path = Path(evaluator_manifest_path).resolve()
    scenes_path = Path(scene_manifest_path).resolve()
    evaluator_manifest, scene_manifest, xyz_by_scene, ids_by_scene = (
        _load_evaluation_inputs(evaluator_path, scenes_path)
    )

    selected_modalities = tuple(map(str, modalities))
    if len(selected_modalities) not in {1, len(MODALITIES)}:
        raise ValueError("diagnostic row scope must contain one or four modalities")
    all_expected = _expected_query_ids(evaluator_manifest)
    expected: set[str] = set()
    for target in _records(evaluator_manifest.get("targets"), label="targets"):
        for modality in selected_modalities:
            expected.add(
                _query_id(
                    target["queries"][modality],
                    target_id=str(target.get("target_id", "")),
                    modality=modality,
                )
            )
    if not expected.issubset(all_expected):
        raise ValueError("selected prediction inventory is not part of the evaluator")
    predictions_root = Path(prediction_dir).resolve()
    if not predictions_root.is_dir():
        raise ValueError(f"prediction directory does not exist: {predictions_root}")
    paths_by_query = {
        path.name[: -len(".npy")]: path
        for path in predictions_root.iterdir()
        if path.is_file() and path.name.endswith(".npy")
    }
    missing = sorted(expected - set(paths_by_query))
    unexpected = sorted(set(paths_by_query) - expected)
    if missing or unexpected:
        raise ValueError(
            f"prediction file inventory is incomplete: missing={missing}, "
            f"unexpected={unexpected}"
        )
    predictions = {
        query_id: _load_npy(paths_by_query[query_id], label=f"{query_id} prediction")
        for query_id in sorted(expected)
    }
    report = evaluate_predictions(
        evaluator_manifest,
        scene_manifest,
        predictions,
        xyz_by_scene,
        ids_by_scene,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
        modalities=selected_modalities,
    )
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


_SEALED_BATCH_KEYS = {
    "schema_version",
    "status",
    "benchmark_version",
    "release_json_sha256",
    "row_scope",
    "modalities",
    "query_manifests",
    "method_run_manifest",
    "method_result_eligible",
    "method_formal_row_eligible",
    "method_field_inventory",
    "method_representation_scope",
    "predictions",
    "prediction_inventory_sha256",
}


def _bind_method_field_inventory(
    method: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any] | None, str]:
    """Bind a method's representation inventory without opening private labels."""

    binding = method.get("method_field_inventory")
    if binding is None:
        if method.get("formal_benchmark_row_eligible") is True:
            raise ValueError("formal method lacks a bound field inventory")
        return None, "unbound_noneligible"
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise ValueError("method field-inventory binding schema changed")
    inventory_path = Path(str(binding["path"])).resolve()
    if not inventory_path.is_file() or _sha256_file(inventory_path) != binding["sha256"]:
        raise ValueError("method field-inventory file binding changed")
    scene_manifest = _load_manifest(root / "scene_manifest.json", required_key="scene_domains")
    scene_ids = [str(row["scene_id"]) for row in scene_manifest["scene_domains"]]
    inventory = validate_method_field_inventory(
        _load_manifest(inventory_path, required_key="scenes"),
        expected_scene_ids=scene_ids,
    )
    if method.get("method_identity_sha256") != inventory["method_identity_sha256"]:
        raise ValueError("method and field-inventory identities disagree")
    return (
        {
            "path": str(inventory_path),
            "sha256": str(binding["sha256"]),
            "inventory_sha256": inventory["inventory_sha256"],
        },
        str(inventory["representation_scope"]),
    )


def _selected_query_manifests(
    root: Path, selected_modalities: Sequence[str]
) -> tuple[set[str], dict[str, dict[str, str]]]:
    expected: set[str] = set()
    bindings: dict[str, dict[str, str]] = {}
    release = _load_manifest(root / "release.json", required_key="manifest_sha256")
    release_hashes = release.get("manifest_sha256")
    if not isinstance(release_hashes, Mapping):
        raise ValueError("release.json lacks manifest bindings")
    for modality in selected_modalities:
        name = QUERY_MANIFEST_NAMES[modality]
        path = root / name
        expected_hash = str(release_hashes.get(name, ""))
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"release query manifest hash mismatch: {name}")
        payload = _load_manifest(path, required_key="queries")
        if payload.get("modality") != modality:
            raise ValueError(f"release query modality changed: {name}")
        for row in _records(payload.get("queries"), label=f"{modality} queries"):
            query_id = _query_id(row.get("query_id"), target_id="public", modality=modality)
            if query_id in expected:
                raise ValueError("public query IDs overlap across selected modalities")
            expected.add(query_id)
        bindings[modality] = {"path": name, "sha256": expected_hash}
    return expected, bindings


def seal_prediction_batch(
    benchmark_dir: str | Path,
    prediction_dir: str | Path,
    method_run_manifest_path: str | Path,
    output: str | Path,
    *,
    row_scope: str,
    modality: str | None = None,
) -> dict[str, Any]:
    """Seal a complete public prediction inventory before private GT opens."""

    if row_scope == "universal_complete":
        if modality is not None:
            raise ValueError("universal_complete must not select one modality")
        selected_modalities = MODALITIES
    elif row_scope == "modality_comparator":
        if modality not in MODALITIES:
            raise ValueError("modality_comparator requires one UQIS modality")
        selected_modalities = (str(modality),)
    else:
        raise ValueError("row_scope must be universal_complete or modality_comparator")

    root = Path(benchmark_dir).resolve()
    release_path = root / "release.json"
    release = _load_manifest(release_path, required_key="manifest_sha256")
    if release.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError("prediction batch benchmark version changed")
    expected, query_bindings = _selected_query_manifests(root, selected_modalities)

    predictions_root = Path(prediction_dir).resolve()
    if not predictions_root.is_dir():
        raise ValueError(f"prediction directory does not exist: {predictions_root}")
    paths_by_query = {
        path.name[: -len(".npy")]: path
        for path in predictions_root.iterdir()
        if path.is_file() and path.name.endswith(".npy")
    }
    missing = sorted(expected - set(paths_by_query))
    unexpected = sorted(set(paths_by_query) - expected)
    if missing or unexpected:
        raise ValueError(
            f"prediction file inventory is incomplete: missing={missing}, "
            f"unexpected={unexpected}"
        )
    records: list[dict[str, Any]] = []
    for query_id in sorted(expected):
        path = paths_by_query[query_id]
        array = _load_npy(path, label=f"{query_id} prediction")
        if (
            array.ndim != 1
            or array.dtype != np.float32
            or not np.isfinite(array).all()
            or bool(((array < 0.0) | (array > 1.0)).any())
        ):
            raise ValueError(
                f"{query_id}: prediction must be a finite float32 [0,1] vector"
            )
        records.append(
            {
                "query_id": query_id,
                "relative_path": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
        )

    method_path = Path(method_run_manifest_path).resolve()
    method = _load_manifest(method_path, required_key="status")
    if method.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError("method run manifest benchmark version changed")
    if not str(method.get("schema_version", "")) or not str(
        method.get("status", "")
    ):
        raise ValueError("method run manifest lacks schema/status identity")
    method_binding = {
        "path": str(method_path),
        "sha256": _sha256_file(method_path),
        "schema_version": str(method.get("schema_version", "")),
        "status": str(method.get("status", "")),
    }
    field_binding, representation_scope = _bind_method_field_inventory(method, root)
    sealed = {
        "schema_version": "scannet_uqis_prediction_batch_v1",
        "status": "sealed_before_private_evaluation",
        "benchmark_version": BENCHMARK_VERSION,
        "release_json_sha256": _sha256_file(release_path),
        "row_scope": row_scope,
        "modalities": list(selected_modalities),
        "query_manifests": query_bindings,
        "method_run_manifest": method_binding,
        "method_result_eligible": method.get("result_eligible") is True,
        "method_formal_row_eligible": method.get("formal_benchmark_row_eligible") is True,
        "method_field_inventory": field_binding,
        "method_representation_scope": representation_scope,
        "predictions": records,
        "prediction_inventory_sha256": canonical_json_sha256(records),
    }
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite sealed prediction batch: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(sealed, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return sealed


def evaluate_release(
    benchmark_dir: str | Path,
    prediction_dir: str | Path,
    sealed_batch_path: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a release-bound batch after verifying its public seal."""

    root = Path(benchmark_dir).resolve()
    release_path = root / "release.json"
    release = _load_manifest(release_path, required_key="manifest_sha256")
    if release.get("formal_benchmark_eligible") is not True:
        raise ValueError(
            "private release evaluation is disabled for result-ineligible pilots"
        )
    seal_path = Path(sealed_batch_path).resolve()
    seal = _load_manifest(seal_path, required_key="predictions")
    if set(seal) != _SEALED_BATCH_KEYS:
        raise ValueError("sealed prediction batch fields changed")
    if (
        seal.get("schema_version") != "scannet_uqis_prediction_batch_v1"
        or seal.get("status") != "sealed_before_private_evaluation"
        or seal.get("benchmark_version") != BENCHMARK_VERSION
        or seal.get("release_json_sha256") != _sha256_file(release_path)
    ):
        raise ValueError("sealed prediction batch identity changed")
    selected_modalities = tuple(map(str, seal.get("modalities", [])))
    if seal.get("row_scope") == "universal_complete":
        if selected_modalities != MODALITIES:
            raise ValueError("universal batch does not contain all four modalities")
    elif seal.get("row_scope") == "modality_comparator":
        if len(selected_modalities) != 1 or selected_modalities[0] not in MODALITIES:
            raise ValueError("modality comparator seal is invalid")
    else:
        raise ValueError("sealed prediction row scope changed")

    expected, query_bindings = _selected_query_manifests(root, selected_modalities)
    if seal.get("query_manifests") != query_bindings:
        raise ValueError("sealed query-manifest bindings changed")
    prediction_records = seal.get("predictions")
    if not isinstance(prediction_records, list) or seal.get(
        "prediction_inventory_sha256"
    ) != canonical_json_sha256(prediction_records):
        raise ValueError("sealed prediction inventory digest changed")
    by_query = {
        str(record.get("query_id", "")): record
        for record in prediction_records
        if isinstance(record, Mapping)
    }
    if set(by_query) != expected or len(by_query) != len(prediction_records):
        raise ValueError("sealed prediction inventory is incomplete")
    predictions_root = Path(prediction_dir).resolve()
    sealed_predictions: dict[str, np.ndarray] = {}
    for query_id, record in by_query.items():
        if set(record) != {
            "query_id",
            "relative_path",
            "bytes",
            "sha256",
            "dtype",
            "shape",
        }:
            raise ValueError(f"{query_id}: sealed prediction fields changed")
        if record.get("relative_path") != f"{query_id}.npy":
            raise ValueError(f"{query_id}: sealed path does not match query identity")
        path = predictions_root / str(record.get("relative_path", ""))
        if path.parent != predictions_root or not path.is_file():
            raise ValueError(f"{query_id}: sealed prediction path escapes its batch")
        if (
            path.stat().st_size != int(record.get("bytes", -1))
            or _sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"{query_id}: sealed prediction binding changed")
        array = _load_npy(path, label=f"{query_id} sealed prediction")
        if (
            array.dtype != np.float32
            or list(array.shape) != record.get("shape")
            or str(array.dtype) != record.get("dtype")
            or array.ndim != 1
            or not np.isfinite(array).all()
            or bool(((array < 0.0) | (array > 1.0)).any())
        ):
            raise ValueError(f"{query_id}: sealed prediction array changed")
        # Retain the verified in-memory snapshot.  Evaluator-private labels are
        # opened only after every prediction has been loaded and hash-checked.
        sealed_predictions[query_id] = array
    method_binding = seal.get("method_run_manifest")
    if not isinstance(method_binding, Mapping) or set(method_binding) != {
        "path",
        "sha256",
        "schema_version",
        "status",
    }:
        raise ValueError("sealed batch lacks a method run manifest")
    method_path = Path(str(method_binding.get("path", ""))).resolve()
    if not method_path.is_file() or _sha256_file(method_path) != method_binding.get("sha256"):
        raise ValueError("sealed method run manifest binding changed")
    method = _load_manifest(method_path, required_key="status")
    if (
        method.get("benchmark_version") != BENCHMARK_VERSION
        or str(method.get("schema_version", ""))
        != method_binding.get("schema_version")
        or str(method.get("status", "")) != method_binding.get("status")
        or (method.get("result_eligible") is True)
        != bool(seal.get("method_result_eligible"))
        or (method.get("formal_benchmark_row_eligible") is True)
        != bool(seal.get("method_formal_row_eligible"))
    ):
        raise ValueError("sealed method run identity changed")
    field_binding, representation_scope = _bind_method_field_inventory(method, root)
    if (
        field_binding != seal.get("method_field_inventory")
        or representation_scope != seal.get("method_representation_scope")
    ):
        raise ValueError("sealed method field-inventory identity changed")

    # Only now may the evaluator open private pairing and instance labels.
    release_audit = audit_release(root, check_files=True)
    if not release_audit.get("valid"):
        raise ValueError("release audit failed after prediction sealing")
    config = UQISProtocolConfig(**release["protocol_config"])
    evaluator_path = root / "target_manifest.evaluator.json"
    scenes_path = root / "scene_manifest.json"
    evaluator_manifest, scene_manifest, xyz_by_scene, ids_by_scene = (
        _load_evaluation_inputs(evaluator_path, scenes_path)
    )
    report = evaluate_predictions(
        evaluator_manifest,
        scene_manifest,
        sealed_predictions,
        xyz_by_scene,
        ids_by_scene,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
        confidence=0.95,
        modalities=selected_modalities,
    )
    formal_eligible = bool(
        release.get("formal_benchmark_eligible") is True
        and seal.get("method_result_eligible") is True
        and seal.get("method_formal_row_eligible") is True
    )
    report.update(
        {
            "evaluation_mode": "sealed_release",
            "formal_benchmark_eligible": formal_eligible,
            "release_status": release.get("status"),
            "release_json_sha256": _sha256_file(release_path),
            "sealed_prediction_batch_sha256": _sha256_file(seal_path),
            "method_run_manifest": dict(method_binding),
            "method_result_eligible": bool(seal.get("method_result_eligible")),
            "method_formal_row_eligible": bool(
                seal.get("method_formal_row_eligible")
            ),
            "method_field_inventory": field_binding,
            "method_representation_scope": representation_scope,
            "single_universal_field_claim_eligible": bool(
                formal_eligible and representation_scope == "single_universal_field"
            ),
        }
    )
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--sealed-prediction-batch", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate_release(
        args.benchmark_dir,
        args.prediction_dir,
        args.sealed_prediction_batch,
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
