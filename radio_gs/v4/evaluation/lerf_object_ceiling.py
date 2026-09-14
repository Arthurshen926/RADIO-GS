"""Cold-load object-memory ceilings for the four-scene LERF cohort.

This evaluator is intentionally diagnostic.  It opens benchmark polygons only
after a query-independent scene state has been loaded and validated, then asks
how well the stored object memberships could perform if object identity were
known.  It never feeds labels back into persistent scene state.

The reported oracle rows are not deployment results.  They isolate three
failure modes before another text or completion experiment is authorized:

* per-observation best token: carrier/extent ceiling with perfect identity;
* one fixed token per category: cross-view object-hypothesis consistency;
* up to three fixed tokens per category: recoverable association fragmentation.

All token-to-element composition goes through ``SparseObjectAssignments`` and
``QueryPacket``.  The legacy evaluator's independent ``max`` compositor is not
used here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radio_gs.data.lerf_dataset import _read_cameras_binary
from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.evaluation.lerf_common import (
    binary_iou as _binary_iou,
    camera as _camera,
    ground_truth_masks as _ground_truth_masks,
    load_labels as _load_labels,
)
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images
from radio_gs.v4.object_memory import SparseObjectAssignments
from radio_gs.v4.query import QueryPacket


STATE_SCHEMA = "radio_gs.surface_object_memory_v4.development_scene_state.v1"
REPORT_SCHEMA = "radio_gs.surface_object_memory_v4.lerf_object_ceiling.v1"


def _load_state(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Load and validate the legacy development state in a fresh process."""

    resolved = path.resolve(strict=True)
    actual_sha256 = sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise ValueError("scene-state SHA256 differs from the sealed expectation")
    state = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
        raise ValueError("scene state is not the expected v4 development schema")
    policy = state.get("information_policy", {})
    forbidden = (
        "target_rgb_opened_during_construction",
        "benchmark_labels_opened_during_construction",
        "text_queries_opened_during_construction",
    )
    if any(policy.get(key) is not False for key in forbidden):
        raise ValueError("scene state construction opened a forbidden information channel")

    centres = torch.as_tensor(state.get("centres"), dtype=torch.float32)
    normals = torch.as_tensor(state.get("normals"), dtype=torch.float32)
    confidence = torch.as_tensor(state.get("confidence"), dtype=torch.float32)
    if centres.ndim != 2 or centres.shape[1] != 3 or normals.shape != centres.shape:
        raise ValueError("scene-state surface geometry is malformed")
    if confidence.shape != (centres.shape[0],):
        raise ValueError("scene-state surface confidence is malformed")
    if not all(bool(torch.isfinite(value).all()) for value in (centres, normals, confidence)):
        raise ValueError("scene-state surface geometry contains non-finite values")

    configuration = state.get("method_configuration", {}).get("carrier", {})
    required_configuration = (
        "voxel_size",
        "maximum_splat_radius",
        "surface_band_voxels",
        "maximum_contributors_per_pixel",
        "camera_convention",
    )
    if any(key not in configuration for key in required_configuration):
        raise ValueError("scene state lacks the complete frozen carrier configuration")
    if configuration["camera_convention"] != "colmap_world_opencv_camera_feature_raster":
        raise ValueError("scene-state camera convention differs from the LERF evaluator")
    if int(configuration["maximum_splat_radius"]) != 1:
        raise ValueError("scene state does not use the frozen radius-one carrier")
    if abs(float(configuration["voxel_size"]) - float(state.get("voxel_size"))) > 1e-12:
        raise ValueError("scene-state voxel size differs from its frozen configuration")

    for key in ("observed_membership", "completed_membership"):
        membership = torch.as_tensor(state.get(key), dtype=torch.float32)
        if membership.ndim != 2 or membership.shape[0] != centres.shape[0]:
            raise ValueError(f"scene-state {key} is malformed")
        if not bool(torch.isfinite(membership).all()) or bool((membership < 0).any()):
            raise ValueError(f"scene-state {key} contains invalid probability mass")
    if state["observed_membership"].shape != state["completed_membership"].shape:
        raise ValueError("observed and completed membership axes differ")
    state["scene_state_sha256"] = actual_sha256
    return state


def _build_carrier(state: dict[str, Any]):
    configuration = state["method_configuration"]["carrier"]
    from radio_gs.v4.carrier import SurfaceVoxelCarrier

    return SurfaceVoxelCarrier(
        state["centres"],
        float(configuration["voxel_size"]),
        normals=state["normals"],
        confidence=state["confidence"],
        maximum_splat_radius=configuration["maximum_splat_radius"],
        surface_band_voxels=float(configuration["surface_band_voxels"]),
        maximum_contributors_per_pixel=configuration["maximum_contributors_per_pixel"],
        reference_raster_shape=configuration.get("reference_raster_shape"),
    )


def _canonical_membership(membership: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Compress to the declared top-2 deployment assignment and return it dense."""

    values = torch.as_tensor(membership, dtype=torch.float32).cpu()
    raw_overlap = (values > 0).sum(-1)
    assignments = SparseObjectAssignments.from_dense(values, top_k=2)
    canonical = assignments.to_dense()
    return canonical, {
        "raw_multi_token_element_fraction": float((raw_overlap > 1).float().mean()),
        "canonical_known_element_fraction": float((canonical.sum(-1) > 0).float().mean()),
        "canonical_unknown_mass_mean": float(assignments.unknown_weight.mean()),
    }


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _greedy_category_tokens(
    rendered: dict[int, torch.Tensor],
    targets: list[tuple[int, torch.Tensor]],
    *,
    maximum_tokens: int,
) -> tuple[list[int], float]:
    """Select a fixed oracle token set by category-level mean IoU."""

    if not targets:
        return [], float("nan")
    token_count = int(next(iter(rendered.values())).shape[-1])
    selected: list[int] = []
    current = {
        frame_id: torch.zeros_like(target, dtype=torch.bool)
        for frame_id, target in targets
    }
    current_score = 0.0
    for _ in range(min(maximum_tokens, token_count)):
        best_token = -1
        best_score = current_score
        for token_id in range(token_count):
            if token_id in selected:
                continue
            values = []
            for frame_id, target in targets:
                candidate = current[frame_id] | rendered[frame_id][..., token_id]
                values.append(_binary_iou(candidate, target)[0])
            score = _mean(values)
            if score is not None and score > best_score + 1e-12:
                best_token, best_score = token_id, score
        if best_token < 0:
            break
        selected.append(best_token)
        current = {
            frame_id: mask | rendered[frame_id][..., best_token]
            for frame_id, mask in current.items()
        }
        current_score = best_score
    return selected, current_score


def _evaluate_membership_ceiling_at_raster(
    *,
    carrier: Any,
    membership: torch.Tensor,
    views: dict[int, Any],
    annotations: dict[int, list[dict[str, Any]]],
    categories: list[str],
    source_height: int,
    source_width: int,
    height: int,
    width: int,
    pixel_threshold: float,
    assignment_mode: str = "canonical_top2",
    fragmentation_capacity: int = 3,
) -> dict[str, Any]:
    if not 0.0 < pixel_threshold < 1.0:
        raise ValueError("pixel threshold must lie strictly in (0,1)")
    if fragmentation_capacity <= 0:
        raise ValueError("fragmentation capacity must be positive")
    if assignment_mode == "canonical_top2":
        evaluated_membership, assignment_audit = _canonical_membership(membership)
        posterior_composition = "canonical_top2_sparse_assignment_mixture_sum"
    elif assignment_mode == "raw_independent_token":
        evaluated_membership = torch.as_tensor(membership, dtype=torch.float32).cpu()
        if bool((evaluated_membership > 1).any()):
            raise ValueError("raw independent token membership must lie in [0,1]")
        raw_overlap = (evaluated_membership > 0).sum(-1)
        assignment_audit = {
            "raw_multi_token_element_fraction": float((raw_overlap > 1).float().mean()),
            "canonical_known_element_fraction": None,
            "canonical_unknown_mass_mean": None,
        }
        posterior_composition = "nondeployable_raw_independent_token_extent_diagnostic"
    else:
        raise ValueError("assignment_mode must be canonical_top2 or raw_independent_token")
    query = QueryPacket("multi_instance")
    rendered: dict[int, torch.Tensor] = {}
    targets_by_category: dict[str, list[tuple[int, torch.Tensor]]] = {
        category: [] for category in categories
    }
    for frame_id, objects in sorted(annotations.items()):
        if frame_id not in views:
            raise KeyError(f"labeled frame {frame_id} is absent from COLMAP registration")
        camera = _camera(views[frame_id], frame_id, height, width)
        rendered[frame_id] = (
            carrier.render_posterior(evaluated_membership, camera) >= pixel_threshold
        )
        ground_truth = _ground_truth_masks(
            objects, categories, height, width, source_height, source_width
        )
        for category in categories:
            if bool(ground_truth[category].any()):
                targets_by_category[category].append((frame_id, ground_truth[category]))

    category_metrics: dict[str, Any] = {}
    observation_oracle: list[float] = []
    stable_single: list[float] = []
    stable_fragmented: list[float] = []
    loo_values: list[float] = []
    loo_observations = 0
    for category in categories:
        targets = targets_by_category[category]
        per_observation_best: list[float] = []
        per_observation_token: list[int] = []
        for frame_id, target in targets:
            scores = [
                _binary_iou(rendered[frame_id][..., token_id], target)[0]
                for token_id in range(evaluated_membership.shape[1])
            ]
            best_token = int(np.nanargmax(scores))
            per_observation_best.append(float(scores[best_token]))
            per_observation_token.append(best_token)
        observation_oracle.extend(per_observation_best)

        single_tokens, single_score = _greedy_category_tokens(
            rendered, targets, maximum_tokens=1
        )
        fragmented_tokens, fragmented_score = _greedy_category_tokens(
            rendered, targets, maximum_tokens=fragmentation_capacity
        )
        if np.isfinite(single_score):
            stable_single.append(single_score)
        if np.isfinite(fragmented_score):
            stable_fragmented.append(fragmented_score)

        category_loo: list[float] = []
        if len(targets) >= 2:
            for heldout_index, (frame_id, target) in enumerate(targets):
                training = [value for index, value in enumerate(targets) if index != heldout_index]
                selected, _ = _greedy_category_tokens(rendered, training, maximum_tokens=1)
                if selected:
                    category_loo.append(
                        _binary_iou(rendered[frame_id][..., selected[0]], target)[0]
                    )
            if category_loo:
                loo_values.append(float(np.mean(category_loo)))
                loo_observations += len(category_loo)

        category_metrics[category] = {
            "observation_count": len(targets),
            "per_observation_best_single_miou": _mean(per_observation_best),
            "per_observation_best_token_ids": per_observation_token,
            "stable_single_token_ids": single_tokens,
            "stable_single_token_miou": single_score,
            f"stable_top{fragmentation_capacity}_token_ids": fragmented_tokens,
            f"stable_top{fragmentation_capacity}_token_miou": fragmented_score,
            "fragmentation_recovery": (
                fragmented_score - single_score
                if np.isfinite(fragmented_score) and np.isfinite(single_score) else None
            ),
            "leave_one_observation_out_single_token_miou": _mean(category_loo),
        }

    return {
        "query_selection_mode": query.selection_mode.value,
        "posterior_composition": posterior_composition,
        "pixel_threshold": pixel_threshold,
        "assignment_audit": assignment_audit,
        "metrics": {
            "per_observation_best_single_token_miou": _mean(observation_oracle),
            "stable_category_single_token_miou": _mean(stable_single),
            f"stable_category_top{fragmentation_capacity}_token_miou": _mean(stable_fragmented),
            f"stable_top{fragmentation_capacity}_fragmentation_gain": (
                _mean(stable_fragmented) - _mean(stable_single)
                if stable_fragmented and stable_single else None
            ),
            "leave_one_observation_out_single_token_miou": _mean(loo_values),
            "leave_one_observation_out_category_count": len(loo_values),
            "leave_one_observation_out_observation_count": loo_observations,
        },
        "categories": category_metrics,
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    state_path = Path(args.scene_state)
    state = _load_state(state_path, expected_sha256=args.expected_scene_state_sha256)
    if state["scene_label"] != args.scene_label:
        raise ValueError("scene-state label differs from the requested scene")
    configuration = state["method_configuration"]["carrier"]
    height, width = (int(value) for value in args.raster_shape)
    if height <= 0 or width <= 0:
        raise ValueError("raster shape must be positive")
    carrier = _build_carrier(state)

    scene_root = Path(args.scene_root).resolve(strict=True)
    sparse = scene_root / "sparse" / "0"
    raw_cameras = _read_cameras_binary(sparse / "cameras.bin")
    views = _read_images(sparse / "images.bin", raw_cameras)
    # Benchmark labels are opened only after the complete state and carrier
    # have been cold-loaded and validated above.
    annotations, categories, source_height, source_width = _load_labels(
        Path(args.label_root).resolve(strict=True), args.scene_label
    )

    variants = {}
    for name in ("observed", "completed"):
        variants[name] = {}
        for assignment_mode in ("canonical_top2", "raw_independent_token"):
            variants[name][assignment_mode] = _evaluate_membership_ceiling_at_raster(
                carrier=carrier,
                membership=state[f"{name}_membership"],
                views=views,
                annotations=annotations,
                categories=categories,
                source_height=source_height,
                source_width=source_width,
                height=height,
                width=width,
                pixel_threshold=float(args.pixel_threshold),
                assignment_mode=assignment_mode,
            )
    report = {
        "schema": REPORT_SCHEMA,
        "development_only": True,
        "oracle_identity_used": True,
        "promotion_eligible": False,
        "scene_label": args.scene_label,
        "scene_state": str(state_path.resolve()),
        "scene_state_sha256": state["scene_state_sha256"],
        "cold_loaded_in_independent_process": True,
        "carrier_configuration": configuration,
        "raster_shape": [height, width],
        "variants": variants,
        "interpretation_contract": {
            "per_observation_best_is_extent_upper_bound": True,
            "stable_single_measures_object_hypothesis_consistency": True,
            "stable_top3_minus_single_measures_recoverable_fragmentation": True,
            "benchmark_labels_written_to_scene_state": False,
            "text_query_used": False,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--expected-scene-state-sha256", required=True)
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--raster-shape", nargs=2, type=int, required=True, metavar=("H", "W"))
    parser.add_argument("--pixel-threshold", type=float, default=0.2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "scene_label": report["scene_label"],
        "observed": {
            key: value["metrics"] for key, value in report["variants"]["observed"].items()
        },
        "completed": {
            key: value["metrics"] for key, value in report["variants"]["completed"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
