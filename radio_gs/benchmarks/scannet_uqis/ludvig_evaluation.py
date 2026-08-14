"""Seal a complete LUDVIG multi-field execution for UQIS evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .method_fields import (
    FIELD_INVENTORY_SCHEMA,
    MODALITIES,
    validate_method_field_inventory,
)
from .protocol import (
    BENCHMARK_VERSION,
    QUERY_MANIFEST_NAMES,
    canonical_json_sha256,
    sha256_file,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    source = path.resolve()
    boundary = root.resolve()
    if not source.is_file() or not source.is_relative_to(boundary):
        raise ValueError(f"field artifact escapes mapping root: {source}")
    return {
        "relative_path": source.relative_to(boundary).as_posix(),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _bound_artifacts(
    directory: Path, bindings: Mapping[str, Any], mapping_root: Path
) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(bindings):
        binding = bindings[name]
        if not isinstance(binding, Mapping) or "relative_path" not in binding:
            raise ValueError(f"invalid field artifact binding: {name}")
        path = directory / str(binding["relative_path"])
        row = _artifact(path, mapping_root)
        if row["bytes"] != int(binding["bytes"]) or row["sha256"] != binding["sha256"]:
            raise ValueError(f"field artifact changed: {path}")
        rows.append(row)
    return rows


def _query_inventory(benchmark: Path) -> tuple[dict[str, tuple[str, str]], dict[str, set[str]]]:
    all_queries: dict[str, tuple[str, str]] = {}
    by_modality: dict[str, set[str]] = {}
    for modality in MODALITIES:
        payload = _load(benchmark / QUERY_MANIFEST_NAMES[modality])
        rows = payload.get("queries")
        if payload.get("benchmark_version") != BENCHMARK_VERSION or not isinstance(rows, list):
            raise ValueError(f"query manifest changed: {modality}")
        ids: set[str] = set()
        for row in rows:
            query_id = str(row["query_id"])
            scene_id = str(row["scene_id"])
            if query_id in all_queries:
                raise ValueError("query IDs overlap across modalities")
            ids.add(query_id)
            all_queries[query_id] = (modality, scene_id)
        by_modality[modality] = ids
    return all_queries, by_modality


def seal_ludvig_method_execution(
    benchmark_dir: str | Path,
    mapping_root: str | Path,
    evaluation_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify all fields/queries and emit inventory plus method-run authority."""

    benchmark = Path(benchmark_dir).resolve()
    mapping = Path(mapping_root).resolve()
    evaluation = Path(evaluation_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    plan_path = mapping / "mapping_plan.json"
    plan = _load(plan_path)
    if (
        plan.get("benchmark_version") != BENCHMARK_VERSION
        or plan.get("method_system_id") != "ludvig_uqis9_clip_dino_system_v1"
        or plan.get("representation_scope") != "modality_specific_multi_field"
    ):
        raise ValueError("LUDVIG mapping-plan identity changed")
    method_identity = str(plan["method_identity_sha256"])
    all_queries, by_modality = _query_inventory(benchmark)

    queue_bindings: dict[str, dict[str, Any]] = {}
    runtime_rows: list[dict[str, Any]] = []
    covered: set[str] = set()
    for modality in MODALITIES:
        queue_path = evaluation / "queues" / f"{modality}.json"
        queue = _load(queue_path)
        if (
            queue.get("status") != "complete"
            or queue.get("benchmark_version") != BENCHMARK_VERSION
            or queue.get("modality") != modality
            or queue.get("completed") != len(by_modality[modality])
            or queue.get("total") != len(by_modality[modality])
        ):
            raise ValueError(f"incomplete query queue: {modality}")
        queue_bindings[modality] = {
            "path": str(queue_path),
            "sha256": sha256_file(queue_path),
        }
        for query_id in sorted(by_modality[modality]):
            run = evaluation / "runs" / modality / query_id
            runtime_path = run / "runtime_receipt.json"
            runtime = _load(runtime_path)
            adapter = _load(run / "run_manifest.json")
            prediction = evaluation / "predictions" / f"{query_id}.npy"
            adapter_query_ids = []
            if adapter.get("query_id") is not None:
                adapter_query_ids.append(str(adapter["query_id"]))
            if isinstance(adapter.get("queries"), list):
                adapter_query_ids.extend(
                    str(row.get("query_id"))
                    for row in adapter["queries"]
                    if isinstance(row, Mapping)
                )
            if (
                runtime.get("status") not in {"complete", "recovered_after_child_success"}
                or runtime.get("benchmark_version") != BENCHMARK_VERSION
                or runtime.get("query_id") != query_id
                or runtime.get("modality") != modality
                or runtime.get("scene_id") != all_queries[query_id][1]
                or runtime.get("fresh_process") is not True
                or runtime.get("cross_query_state_retained") is not False
                or runtime.get("workspace_read_only") is not True
                or runtime.get("evaluator_private_inputs_opened") is not False
                or not prediction.is_file()
                or runtime.get("prediction_sha256") != sha256_file(prediction)
                or runtime.get("run_manifest_sha256") != sha256_file(run / "run_manifest.json")
                or adapter.get("status") != "exact_runtime_smoke_complete"
                or adapter.get("benchmark_version") != BENCHMARK_VERSION
                or adapter.get("modality") != modality
                or adapter.get("scene_id") != all_queries[query_id][1]
                or adapter_query_ids != [query_id]
                or adapter.get("benchmark_local_adapter") is not True
                or adapter.get("official_ludvig_reproduction") is not False
                or adapter.get("paper_metric_comparable") is not False
                or adapter.get("privacy_boundary", {}).get("evaluator_manifest_opened") is not False
                or adapter.get("privacy_boundary", {}).get("private_target_inputs_opened") is not False
            ):
                raise ValueError(f"invalid runtime receipt: {query_id}")
            covered.add(query_id)
            runtime_rows.append({
                "query_id": query_id,
                "modality": modality,
                "scene_id": all_queries[query_id][1],
                "runtime_receipt_sha256": sha256_file(runtime_path),
                "prediction_sha256": runtime["prediction_sha256"],
            })
    if covered != set(all_queries):
        raise ValueError("runtime receipt coverage is incomplete")

    scene_ids = sorted({scene for _modality, scene in all_queries.values()})
    scenes = []
    for scene_id in scene_ids:
        phase_b_dir = mapping / "dino_phase_b" / f"{scene_id}_v1"
        uplift_dir = mapping / "dino_uplift" / f"{scene_id}_v1"
        clip_dir = mapping / "clip_field" / f"{scene_id}_v1"
        phase_b = _load(phase_b_dir / "run_manifest.json")
        uplift = _load(uplift_dir / "run_manifest.json")
        clip = _load(clip_dir / "run_manifest.json")
        dino_artifacts = _bound_artifacts(
            phase_b_dir, phase_b["pca"]["artifacts"], mapping
        ) + _bound_artifacts(uplift_dir, uplift["artifacts"], mapping)
        clip_artifacts = _bound_artifacts(clip_dir, clip["artifacts"], mapping)
        geometry_path = Path(str(clip["geometry"]["path"]))
        geometry_artifact = _artifact(geometry_path, mapping)
        if (
            geometry_artifact["bytes"] != int(clip["geometry"]["bytes"])
            or geometry_artifact["sha256"] != clip["geometry"]["sha256"]
        ):
            raise ValueError(f"CLIP geometry changed: {scene_id}")
        clip_artifacts.append(geometry_artifact)
        fields = [
            {
                "field_id": "ludvig_clip_text_field",
                "field_family": "ludvig_clip_language_field",
                "modalities": ["text"],
                "mapping_receipt_sha256": sha256_file(clip_dir / "run_manifest.json"),
                "artifacts": clip_artifacts,
            },
            {
                "field_id": "ludvig_dino_prompt_image_field",
                "field_family": "ludvig_dinov2_visual_field",
                "modalities": ["image", "point_2d", "point_3d"],
                "mapping_receipt_sha256": sha256_file(uplift_dir / "run_manifest.json"),
                "artifacts": dino_artifacts,
            },
        ]
        unique = {row["sha256"]: row["bytes"] for field in fields for row in field["artifacts"]}
        scenes.append({
            "scene_id": scene_id,
            "fields": fields,
            "field_count": len(fields),
            "persistent_bytes": sum(unique.values()),
        })
    inventory_body = {
        "schema_version": FIELD_INVENTORY_SCHEMA,
        "benchmark_version": BENCHMARK_VERSION,
        "status": "complete",
        "method_system_id": plan["method_system_id"],
        "method_identity_sha256": method_identity,
        "representation_scope": "modality_specific_multi_field",
        "scene_count": len(scenes),
        "scenes": scenes,
        "totals": {
            "field_count": sum(row["field_count"] for row in scenes),
            "persistent_bytes": sum(row["persistent_bytes"] for row in scenes),
        },
    }
    inventory = {**inventory_body, "inventory_sha256": canonical_json_sha256(inventory_body)}
    validate_method_field_inventory(inventory, expected_scene_ids=scene_ids)
    inventory_path = output / "method_field_inventory.json"
    output.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    runtime_rows.sort(key=lambda row: row["query_id"])
    method = {
        "schema_version": "scannet_uqis_ludvig_method_run_v1",
        "status": "complete_before_private_evaluation",
        "benchmark_version": BENCHMARK_VERSION,
        "method_system_id": plan["method_system_id"],
        "method_identity_sha256": method_identity,
        "result_eligible": True,
        "formal_benchmark_row_eligible": False,
        "official_ludvig_reproduction": False,
        "paper_metric_comparable": False,
        "benchmark_local_adapter": True,
        "representation_scope": "modality_specific_multi_field",
        "method_field_inventory": {
            "path": str(inventory_path),
            "sha256": sha256_file(inventory_path),
        },
        "mapping_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "query_queues": queue_bindings,
        "runtime_receipt_count": len(runtime_rows),
        "runtime_receipt_inventory_sha256": canonical_json_sha256(runtime_rows),
        "all_predictions_completed_before_private_evaluation": True,
        "private_evaluator_inputs_opened": False,
    }
    method_path = output / "method_run_manifest.json"
    method_path.write_text(
        json.dumps(method, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"field_inventory": inventory, "method_run_manifest": method}
