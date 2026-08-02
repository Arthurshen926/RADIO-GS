#!/usr/bin/env python3
"""Fail-closed aggregate for the frozen Ours LERF direct-3D evaluation.

This command does not score masks.  It validates four completed scene results,
their immutable three-scale v2 query-score authorities, and every bound input
file before publishing the scene-equal frozen-protocol aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    validate_ours_multiscale_query_score_cache,
)
from radio_gs.scripts.validate_evaluation_protocol_freeze import (
    FreezeError,
    load_and_validate,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    sha256_file,
    stable_descriptor_load,
    write_frozen_json,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "radio_gs_ours_lerf_direct3d_frozen_aggregate"
STATUS = "complete_exact_frozen_protocol_evaluation"
TASK_ID = "concept_lerf3d_vala"
REGISTRY_ROW = "lerf3d_vala_occam_geometry_exact_protocol_20260801"
EXPECTED_FREEZE_SHA256 = (
    "af91f0861d3a15354063579e78f64898801c41f2543d1cf9b352a0a123820916"
)
EXPECTED_SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")
EXPECTED_OBJECTS = 208
EXPECTED_FRAMES = {
    "figurines": (41, 105, 152, 195),
    "ramen": (6, 24, 60, 65, 81, 119, 128),
    "teatime": (2, 25, 43, 107, 129, 140),
    "waldo_kitchen": (53, 66, 89, 140, 154),
}
EXPECTED_SCALE_IDS = ("0.25", "0.45", "0.7")
EXPECTED_SCALE_RADII_M = (0.25, 0.45, 0.7)
RESULT_NAME = "lerf_direct_3d_selection_results.json"
THRESHOLD_TAG = "thr0p6"


class FrozenDirect3DError(ValueError):
    """Raised before publication when any frozen authority check fails."""


@dataclass(frozen=True)
class FrozenDirect3DContract:
    freeze_path: str
    freeze_sha256: str
    freeze_id: str
    task_id: str
    registry_row: str
    scenes: tuple[str, ...]
    objects: int


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FrozenDirect3DError(f"{label} must be a mapping")
    return value


def _finite_metric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenDirect3DError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise FrozenDirect3DError(f"{label} must be finite and in [0,1]")
    return result


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.layout != torch.strided:
        raise FrozenDirect3DError("authority tensors must use strided layout")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    if tensor.ndim == 0:
        digest.update(tensor.contiguous().numpy().tobytes(order="C"))
    else:
        for start in range(0, int(tensor.shape[0]), 4096):
            digest.update(
                tensor[start : start + 4096]
                .contiguous()
                .numpy()
                .tobytes(order="C")
            )
    return digest.hexdigest()


def contract_from_validated_freeze(
    payload: Mapping[str, Any], *, freeze_path: Path, freeze_sha256: str
) -> FrozenDirect3DContract:
    if freeze_sha256 != EXPECTED_FREEZE_SHA256:
        raise FrozenDirect3DError("evaluation protocol freeze SHA256 differs")
    tasks = _mapping(payload.get("canonical_tasks"), "canonical_tasks")
    task = _mapping(tasks.get(TASK_ID), TASK_ID)
    expected_task_header = {
        "family": "concept_query",
        "benchmark": "LERF direct 3D",
        "method": "VALA",
        "status": "canonical_protocol_reproduction_with_compatible_rgb_geometry",
        "registry_row": REGISTRY_ROW,
    }
    for key, expected in expected_task_header.items():
        if task.get(key) != expected:
            raise FrozenDirect3DError(f"frozen task field {key} differs")
    cohort = _mapping(task.get("cohort"), "LERF direct-3D cohort")
    if set(cohort) != {"scenes", "queries", "extensionless_test_stems_required"}:
        raise FrozenDirect3DError("frozen LERF direct-3D cohort fields differ")
    if tuple(cohort.get("scenes", ())) != EXPECTED_SCENES:
        raise FrozenDirect3DError("frozen LERF direct-3D scene order differs")
    if cohort.get("queries") != EXPECTED_OBJECTS:
        raise FrozenDirect3DError("frozen LERF direct-3D object count differs")
    if cohort.get("extensionless_test_stems_required") is not True:
        raise FrozenDirect3DError("extensionless test stems are not required")
    frozen = _mapping(task.get("frozen_protocol"), "LERF direct-3D protocol")
    expected_protocol = {
        "source_commit": "48902a541333d65aeb0aebf64ad664777a27c3fc",
        "semantic_levels": 3,
        "significance": "official gsplat marginal alpha-times-transmittance",
        "robust_lift": {"tau_mass": 0.75, "tau_abs": 0.13},
        "significance_weight_threshold": 0.00001,
        "aggregation_batch_size": 50000,
        "level_selection": "highest raw relevance peak per query",
        "knn": 10,
        "relevance_mapping": "min-max map then clip",
        "mask_threshold": 0.6,
        "render": "selected-only alpha",
        "metrics": ["mIoU", "accuracy_at_iou_0p25", "accuracy_at_iou_0p5"],
        "aggregation": "per-object metrics then unweighted equal macro over four scenes",
    }
    if dict(frozen) != expected_protocol:
        raise FrozenDirect3DError("frozen LERF direct-3D protocol fields differ")
    freeze_id = payload.get("freeze_id")
    if not isinstance(freeze_id, str) or not freeze_id:
        raise FrozenDirect3DError("validated freeze lacks freeze_id")
    return FrozenDirect3DContract(
        freeze_path=str(freeze_path),
        freeze_sha256=freeze_sha256,
        freeze_id=freeze_id,
        task_id=TASK_ID,
        registry_row=REGISTRY_ROW,
        scenes=EXPECTED_SCENES,
        objects=EXPECTED_OBJECTS,
    )


def load_frozen_contract(freeze_path: Path, *, repo_root: Path) -> FrozenDirect3DContract:
    canonical = freeze_path.resolve(strict=True)
    digest = sha256_file(canonical)
    try:
        payload = load_and_validate(canonical, root=repo_root.resolve(), verify_hashes=True)
    except FreezeError as exc:
        raise FrozenDirect3DError(f"protocol freeze validation failed: {exc}") from exc
    if sha256_file(canonical) != digest:
        raise FrozenDirect3DError("protocol freeze changed during validation")
    return contract_from_validated_freeze(
        payload, freeze_path=canonical, freeze_sha256=digest
    )


def _resolve_bound_path(raw: Any, *, repo_root: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise FrozenDirect3DError(f"{label} path is missing")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise FrozenDirect3DError(f"{label} path cannot be resolved") from exc


def _verify_file(
    path: Path,
    expected_sha256: Any,
    *,
    label: str,
    verified: dict[Path, str],
) -> dict[str, str]:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise FrozenDirect3DError(f"{label} SHA256 is malformed")
    canonical = path.resolve(strict=True)
    prior = verified.get(canonical)
    if prior is None:
        _, digest, canonical = stable_descriptor_load(
            canonical,
            lambda handle: None,
            expected_sha256=expected_sha256,
            label=label,
        )
        verified[canonical] = digest
    elif prior != expected_sha256:
        raise FrozenDirect3DError(f"{label} conflicts with an earlier file binding")
    return {"path": str(canonical), "sha256": expected_sha256}


def _verify_record(
    raw: Any, *, label: str, verified: dict[Path, str]
) -> dict[str, str]:
    record = _mapping(raw, label)
    if set(record) != {"path", "sha256"}:
        raise FrozenDirect3DError(f"{label} file record fields differ")
    path = _resolve_bound_path(record["path"], repo_root=Path("/"), label=label)
    return _verify_file(path, record["sha256"], label=label, verified=verified)


def _validate_cache_authority(
    payload: Mapping[str, Any],
    *,
    categories: Sequence[str],
    renderer_sha256: str,
    verified: dict[Path, str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    xyz = payload.get("xyz")
    if not isinstance(xyz, torch.Tensor):
        raise FrozenDirect3DError("v2 cache lacks xyz tensor")
    validated = validate_ours_multiscale_query_score_cache(
        payload,
        expected_xyz=xyz,
        expected_query_ids=categories,
        expected_renderer_geometry_checkpoint_sha256=renderer_sha256,
    )
    if validated.scale_ids != EXPECTED_SCALE_IDS:
        raise FrozenDirect3DError("v2 cache native scale IDs differ")
    if validated.scale_radii_m != EXPECTED_SCALE_RADII_M:
        raise FrozenDirect3DError("v2 cache native scale radii differ")
    authority = _mapping(payload.get("authority"), "v2 cache authority")
    expected_calibration = {
        "softmax_applied": False,
        "temperature_applied": False,
        "peak_normalization_applied": False,
        "threshold_applied": False,
        "scale_reduction_applied": False,
        "benchmark_images_opened": False,
        "benchmark_annotations_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_metrics_opened": False,
    }
    if authority.get("calibration_constraints") != expected_calibration:
        raise FrozenDirect3DError("v2 cache calibration constraints differ")
    if authority.get("score_semantics") != "raw_independent_normalized_cosine":
        raise FrozenDirect3DError("v2 cache score semantics differ")
    if authority.get("score_dtype") != "torch.float16":
        raise FrozenDirect3DError("v2 cache score dtype differs")
    query_axis = _mapping(authority.get("query_axis"), "v2 cache query axis")
    if query_axis.get("order_sha256") != canonical_json_sha256(list(categories)):
        raise FrozenDirect3DError("v2 cache query order SHA256 differs")
    scores = payload.get("query_scores")
    valid = payload.get("valid")
    if not isinstance(scores, torch.Tensor) or not isinstance(valid, torch.Tensor):
        raise FrozenDirect3DError("v2 cache authority tensors are missing")
    if authority.get("query_scores_sha256") != _tensor_sha256(scores):
        raise FrozenDirect3DError("v2 cache query-score tensor SHA256 differs")
    geometry = _mapping(authority.get("geometry_axis"), "v2 cache geometry axis")
    if geometry.get("valid_sha256") != _tensor_sha256(valid):
        raise FrozenDirect3DError("v2 cache valid tensor SHA256 differs")
    direct = _mapping(
        _mapping(authority.get("consumer_contracts"), "consumer contracts").get(
            "direct3d"
        ),
        "direct3d consumer contract",
    )
    if dict(direct) != {
        "contract": "radio_gs.ours_lerf_direct3d_multiscale_query_scores.v2",
        "tensor_layout": "[primitive_row,scale,query]",
        "scale_selection": "downstream_frozen_VALA_readout_only",
    }:
        raise FrozenDirect3DError("v2 cache direct3d consumer contract differs")
    sources = _mapping(authority.get("source_artifacts"), "v2 source artifacts")
    expected_roles = {
        "descriptor_cache",
        "text_query_cache",
        "field_checkpoint",
        "readout_checkpoint",
        "renderer_geometry_checkpoint",
        "materializer_source",
    }
    if set(sources) != expected_roles:
        raise FrozenDirect3DError("v2 source artifact roles differ")
    records = [
        {"role": role, **_verify_record(sources[role], label=role, verified=verified)}
        for role in sorted(expected_roles)
    ]
    by_role = {record["role"]: record for record in records}
    expected_source_shas = {
        "field_checkpoint": payload.get("field_checkpoint_sha256"),
        "readout_checkpoint": payload.get("readout_checkpoint_sha256"),
        "renderer_geometry_checkpoint": renderer_sha256,
    }
    for role, expected in expected_source_shas.items():
        if by_role[role]["sha256"] != expected:
            raise FrozenDirect3DError(f"v2 {role} top-level SHA256 differs")
    return {
        "contract": authority.get("contract"),
        "query_scores_sha256": authority.get("query_scores_sha256"),
        "query_order_sha256": query_axis.get("order_sha256"),
        "xyz_sha256": validated.xyz_sha256,
        "valid_sha256": geometry.get("valid_sha256"),
        "field_checkpoint_sha256": validated.field_checkpoint_sha256,
        "readout_checkpoint_sha256": validated.readout_checkpoint_sha256,
        "renderer_geometry_checkpoint_sha256": (
            validated.renderer_geometry_checkpoint_sha256
        ),
        "scale_ids": list(validated.scale_ids),
        "scale_radii_m": list(validated.scale_radii_m),
        "queries": len(validated.query_ids),
        "primitives": int(validated.query_scores.shape[0]),
        "valid_primitives": int(validated.valid.sum().item()),
    }, records


def _require_protocol(protocol: Mapping[str, Any]) -> None:
    exact = {
        "protocol_preset": "vala_repo_3d",
        "feature_source": "frozen Ours row-aligned three-scale primitive query-score cache",
        "feature_level_count": 3,
        "scale_ids": list(EXPECTED_SCALE_IDS),
        "scale_radii_m": list(EXPECTED_SCALE_RADII_M),
        "level_selection": "highest_raw_knn_smoothed_peak_per_query",
        "vala_knn_k": 10,
        "vala_repo_score_remap": "clip(2 * per_query_minmax - 1, 0, 1)",
        "vala_repo_effective_pre_remap_threshold": 0.8,
        "score_postprocess": "vala_knn_minmax",
        "selection_refinement": "none",
        "mask_refinement": "none",
        "proposal_smoothing": "none",
        "render_role": "render physically selected primitives only for mask evaluation",
        "projection_mode": "selected_only_alpha",
        "projection_semantics": "physically subset selected primitives and render their alpha",
        "alpha_binarization": "png_uint8_gt10",
        "silhouette_threshold": 10.0 / 255.0,
        "diagnostic_oracle_prompt": False,
        "geometry_alignment_maps": False,
        "score_aggregation": "none",
        "rgb_refinement_source": "",
        "registered_feature_cache": "",
        "score_cache": "",
    }
    for key, expected in exact.items():
        if protocol.get(key) != expected:
            raise FrozenDirect3DError(f"result protocol {key} differs")
    if protocol.get("metrics") != [
        "mIoU",
        "Acc@0.25",
        "Acc@0.50",
        "boundary_f",
        "trimap_iou",
    ]:
        raise FrozenDirect3DError("result metric declaration differs")
    checkpoint_contract = _mapping(
        protocol.get("checkpoint_contract"), "checkpoint contract"
    )
    for key in (
        "model_missing_keys",
        "model_unexpected_keys",
        "codec_missing_keys",
        "codec_unexpected_keys",
        "sharpener_missing_keys",
        "sharpener_unexpected_keys",
        "refiner_missing_keys",
        "refiner_unexpected_keys",
        "errors",
    ):
        if checkpoint_contract.get(key) != []:
            raise FrozenDirect3DError(f"checkpoint contract {key} is nonempty")


def _validate_scene_metrics(scene: str, row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("selection_mode") != "score_threshold":
        raise FrozenDirect3DError(f"{scene}: selection mode differs")
    if row.get("selection_value") != 0.6 or row.get("selection_tag") != THRESHOLD_TAG:
        raise FrozenDirect3DError(f"{scene}: threshold differs")
    if row.get("selection_refinement") != "none" or row.get("mask_refinement") != "none":
        raise FrozenDirect3DError(f"{scene}: refinement is enabled")
    if row.get("projection_mode") != "selected_only_alpha":
        raise FrozenDirect3DError(f"{scene}: selected-only alpha is disabled")
    categories = _mapping(row.get("per_category"), f"{scene} per-category metrics")
    if not categories:
        raise FrozenDirect3DError(f"{scene}: category metrics are empty")
    totals = {"miou": 0.0, "acc025": 0.0, "acc050": 0.0}
    objects = 0
    for category, raw in categories.items():
        entry = _mapping(raw, f"{scene}/{category}")
        n = entry.get("n")
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise FrozenDirect3DError(f"{scene}/{category}: n must be positive")
        objects += n
        for metric in totals:
            totals[metric] += n * _finite_metric(
                entry.get(metric), f"{scene}/{category}/{metric}"
            )
    if row.get("n") != objects:
        raise FrozenDirect3DError(f"{scene}: object count differs from categories")
    output = {"objects": objects, "categories": len(categories)}
    for metric, total in totals.items():
        recomputed = total / objects
        reported = _finite_metric(row.get(metric), f"{scene}/{metric}")
        if not math.isclose(reported, recomputed, rel_tol=0.0, abs_tol=5e-7):
            raise FrozenDirect3DError(f"{scene}: {metric} differs from categories")
        output[metric] = reported
    return output


def aggregate_results_root(
    results_root: Path,
    *,
    repo_root: Path,
    contract: FrozenDirect3DContract,
) -> dict[str, Any]:
    root = results_root.resolve(strict=True)
    repo = repo_root.resolve(strict=True)
    expected_paths = {
        scene: root / scene / scene / RESULT_NAME for scene in contract.scenes
    }
    discovered = set(root.rglob(RESULT_NAME))
    if discovered != set(expected_paths.values()):
        raise FrozenDirect3DError("result-file cohort/layout differs")
    verified: dict[Path, str] = {}
    scene_results: dict[str, Any] = {}
    total_objects = 0
    repo_commits: set[str] = set()
    for scene in contract.scenes:
        path = expected_paths[scene]
        payload, result_sha, result_path = load_json_object(
            path, label=f"{scene} result"
        )
        verified[result_path] = result_sha
        scene_block = _mapping(payload.get("scene"), f"{scene} scene block")
        if scene_block.get("scene") != scene or payload.get("args", {}).get("scene") != scene:
            raise FrozenDirect3DError(f"{scene}: scene identity differs")
        if scene_block.get("official_frames") != list(EXPECTED_FRAMES[scene]):
            raise FrozenDirect3DError(f"{scene}: extensionless official frames differ")
        if scene_block.get("official_frames_only") is not True:
            raise FrozenDirect3DError(f"{scene}: non-test frames were evaluated")
        categories = scene_block.get("categories")
        if not isinstance(categories, list) or not categories or not all(
            isinstance(item, str) and item for item in categories
        ) or len(set(categories)) != len(categories):
            raise FrozenDirect3DError(f"{scene}: category order is invalid")
        protocol = _mapping(payload.get("protocol"), f"{scene} protocol")
        _require_protocol(protocol)
        args = _mapping(payload.get("args"), f"{scene} args")
        expected_args = {
            "protocol_preset": "vala_repo_3d",
            "score_source": "direct",
            "vala_knn_k": "10",
            "selection_mode": "score_threshold",
            "score_threshold": "0.6",
            "threshold_sweep": "",
            "mean_std_sweep": "",
            "ratio_sweep": "",
            "confidence_sweep": "",
            "selection_refinement": "none",
            "mask_refinement": "none",
            "proposal_smoothing": "none",
            "score_postprocess": "vala_knn_minmax",
            "projection_mode": "selected_only_alpha",
            "all_labeled_frames": "False",
            "registered_feature_cache": "",
            "score_cache": "",
            "external_query_score_cache": "",
            "external_query_feature_cache": "",
        }
        for key, expected in expected_args.items():
            if args.get(key) != expected:
                raise FrozenDirect3DError(f"{scene}: evaluator argument {key} differs")
        if scene_block.get("best_by_miou") != THRESHOLD_TAG:
            raise FrozenDirect3DError(f"{scene}: selected row differs")
        rows = _mapping(scene_block.get("results"), f"{scene} result rows")
        if set(rows) != {THRESHOLD_TAG}:
            raise FrozenDirect3DError(f"{scene}: sweep or extra result row detected")
        metrics = _validate_scene_metrics(scene, _mapping(rows[THRESHOLD_TAG], scene))
        total_objects += metrics["objects"]

        config = _resolve_bound_path(scene_block.get("config"), repo_root=repo, label="config")
        checkpoint = _resolve_bound_path(
            scene_block.get("checkpoint"), repo_root=repo, label="checkpoint"
        )
        if args.get("config") != scene_block.get("config") or args.get("checkpoint") != scene_block.get("checkpoint"):
            raise FrozenDirect3DError(f"{scene}: args/source path binding differs")
        config_record = _verify_file(
            config,
            protocol.get("config_sha256"),
            label=f"{scene} config",
            verified=verified,
        )
        checkpoint_record = _verify_file(
            checkpoint,
            protocol.get("checkpoint_sha256"),
            label=f"{scene} checkpoint",
            verified=verified,
        )
        cache = _resolve_bound_path(
            protocol.get("ours_multiscale_query_score_cache"),
            repo_root=repo,
            label=f"{scene} cache",
        )
        if args.get("ours_multiscale_query_score_cache") != protocol.get(
            "ours_multiscale_query_score_cache"
        ):
            raise FrozenDirect3DError(f"{scene}: cache path binding differs")
        cache_payload, cache_sha, cache_path = load_torch_mapping(
            cache,
            expected_sha256=protocol.get("ours_multiscale_query_score_cache_sha256"),
            map_location="cpu",
            label=f"{scene} v2 query-score cache",
        )
        verified[cache_path] = cache_sha
        cache_authority, source_records = _validate_cache_authority(
            cache_payload,
            categories=categories,
            renderer_sha256=checkpoint_record["sha256"],
            verified=verified,
        )
        sidecar_path = cache_path.with_suffix(cache_path.suffix + ".json")
        sidecar, sidecar_sha, sidecar_source = load_json_object(
            sidecar_path, label=f"{scene} cache sidecar"
        )
        verified[sidecar_source] = sidecar_sha
        if sidecar.get("status") != "complete_calibration_free_query_score_materialization":
            raise FrozenDirect3DError(f"{scene}: cache materialization is incomplete")
        if sidecar.get("query_score_cache") != {
            "path": str(cache_path),
            "sha256": cache_sha,
        }:
            raise FrozenDirect3DError(f"{scene}: cache sidecar file binding differs")
        if sidecar.get("shared_renderer_authority") != cache_payload.get("authority"):
            raise FrozenDirect3DError(f"{scene}: cache sidecar authority differs")
        repo_commit = protocol.get("repo_commit")
        if not isinstance(repo_commit, str) or len(repo_commit) != 40:
            raise FrozenDirect3DError(f"{scene}: evaluator commit is malformed")
        repo_commits.add(repo_commit)
        scene_results[scene] = {
            **metrics,
            "result": {"path": str(result_path), "sha256": result_sha},
            "config": config_record,
            "checkpoint": checkpoint_record,
            "query_score_cache": {"path": str(cache_path), "sha256": cache_sha},
            "query_score_cache_sidecar": {
                "path": str(sidecar_source),
                "sha256": sidecar_sha,
            },
            "cache_authority": cache_authority,
            "cache_source_artifacts": source_records,
        }
    if total_objects != contract.objects:
        raise FrozenDirect3DError(
            f"observed {total_objects} objects, expected {contract.objects}"
        )
    if len(repo_commits) != 1:
        raise FrozenDirect3DError("scene evaluator commits differ")
    macro = {
        metric: sum(scene_results[scene][metric] for scene in contract.scenes)
        / len(contract.scenes)
        for metric in ("miou", "acc025", "acc050")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": STATUS,
        "method": "RADIO-GS Ours native multiscale primitive query scores",
        "benchmark": "LERF direct 3D",
        "protocol_authority": {
            "path": contract.freeze_path,
            "sha256": contract.freeze_sha256,
            "freeze_id": contract.freeze_id,
            "canonical_task_id": contract.task_id,
            "registry_row": contract.registry_row,
        },
        "protocol_constraints": {
            "protocol_preset": "vala_repo_3d",
            "native_scales_m": list(EXPECTED_SCALE_RADII_M),
            "level_selection": "highest raw kNN-smoothed relevance peak per query",
            "knn": 10,
            "relevance_mapping": "per-query min-max then 2x-1 and clip",
            "mask_threshold": 0.6,
            "render": "physically selected primitives only; alpha mask",
            "smoothing": "none beyond frozen kNN10 score readout",
            "selection_refinement": "none",
            "mask_refinement": "none",
            "threshold_or_configuration_sweep": False,
            "aggregation": "per-object scene metrics then unweighted equal macro over four scenes",
        },
        "cohort": {
            "scenes": list(contract.scenes),
            "objects": total_objects,
            "labelled_frames": sum(len(EXPECTED_FRAMES[s]) for s in contract.scenes),
        },
        "evaluator_repo_commit": next(iter(repo_commits)),
        "scene_results": scene_results,
        "scene_equal_macro": {**macro, "scenes": len(contract.scenes)},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument(
        "--protocol-freeze",
        type=Path,
        default=repo / "paper/artifacts/evaluation_protocol_freeze_20260801.yaml",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    contract = load_frozen_contract(args.protocol_freeze, repo_root=args.repo_root)
    report = aggregate_results_root(
        args.results_root, repo_root=args.repo_root, contract=contract
    )
    write_frozen_json(args.output_json, report)
    macro = report["scene_equal_macro"]
    print(
        f"scene-equal mIoU={macro['miou']:.12f} "
        f"Acc@0.25={macro['acc025']:.12f} "
        f"Acc@0.50={macro['acc050']:.12f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
