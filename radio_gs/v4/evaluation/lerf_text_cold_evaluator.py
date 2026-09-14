"""Cold-load LERF text evaluation with the canonical v4 posterior.

This module replaces the deprecated monolithic development evaluator.  It
cannot build or mutate scene state.  It loads one SHA-bound state, optionally
replaces its invalid per-pixel-summary descriptors with source-only official
region descriptors, constructs typed multi-instance ``QueryPacket`` objects,
and composes token probabilities through the sole top-2 mixture-sum path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radio_gs.data.lerf_dataset import _read_cameras_binary
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.contracts.build_lerf_fragment_language_memory import (
    FRAME_SCHEMA,
    MANIFEST_SCHEMA,
)
from radio_gs.v4.evaluation.lerf_common import (
    GENERIC_NEGATIVES,
    binary_iou as _binary_iou,
    camera as _camera,
    ground_truth_masks as _ground_truth_masks,
    load_labels as _load_labels,
    load_text_cache as _load_text_cache,
    prototype_max_token_posterior,
    retain_top_query_tokens,
)
from radio_gs.v4.evaluation.lerf_object_ceiling import _build_carrier, _load_state
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images
from radio_gs.v4.object_memory import SparseObjectAssignments
from radio_gs.v4.query import QueryPacket


REPORT_SCHEMA = "radio_gs.surface_object_memory_v4.lerf_text_cold_evaluation.v1"


def prototype_cosine_set_posterior(
    prototypes: torch.Tensor,
    prototype_token_ids: torch.Tensor,
    queries: torch.Tensor,
    *,
    num_tokens: int,
    temperature: float,
    set_mass: float,
    null_similarity: float = 0.0,
) -> torch.Tensor:
    """Convert prototype cosine scores to a bounded set-level token mass.

    Unlike the historical independent generic-negative sigmoid, this performs
    one query-level normalization over the complete candidate set.  ``set_mass``
    controls expected aggregate token support without a discontinuous top-k.
    """

    prototype = torch.nn.functional.normalize(
        torch.as_tensor(prototypes, dtype=torch.float32), dim=-1
    )
    query = torch.nn.functional.normalize(
        torch.as_tensor(queries, dtype=torch.float32, device=prototype.device), dim=-1
    )
    token_ids = torch.as_tensor(
        prototype_token_ids, dtype=torch.long, device=prototype.device
    )
    if prototype.ndim != 2 or query.ndim != 2 or prototype.shape[1] != query.shape[1]:
        raise ValueError("prototype and query descriptors must be aligned matrices")
    if token_ids.shape != (prototype.shape[0],) or num_tokens <= 0:
        raise ValueError("prototype token ids or token count are invalid")
    if temperature <= 0 or set_mass <= 0:
        raise ValueError("set-posterior temperature and mass must be positive")
    similarity = query @ prototype.T
    token_similarity = prototype.new_full((query.shape[0], num_tokens), -torch.inf)
    for token_id in range(num_tokens):
        selected = token_ids == token_id
        if bool(selected.any()):
            token_similarity[:, token_id] = similarity[:, selected].max(-1).values
    null = token_similarity.new_full((query.shape[0], 1), float(null_similarity))
    posterior = torch.softmax(
        torch.cat((token_similarity, null), dim=-1) / temperature, dim=-1
    )[:, :-1]
    return (posterior * float(set_mass)).clamp_max(1.0)


def compose_object_queries(
    membership: torch.Tensor,
    token_probability: torch.Tensor,
    *,
    maximum_query_tokens: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compose every query through one typed canonical element posterior."""

    probability = torch.as_tensor(token_probability, dtype=torch.float32).cpu()
    if probability.ndim != 2:
        raise ValueError("token probability must have shape [Q,K]")
    assignments = SparseObjectAssignments.from_dense(membership, top_k=2)
    if probability.shape[1] != assignments.num_tokens:
        raise ValueError("query token probabilities and object assignment axis differ")
    if maximum_query_tokens is not None:
        if maximum_query_tokens <= 0:
            raise ValueError("maximum query tokens must be positive when provided")
        probability = retain_top_query_tokens(probability, maximum_query_tokens)
    query_packet = QueryPacket("multi_instance")
    posterior = torch.stack([
        assignments.element_posterior(query_packet, row).foreground
        for row in probability
    ], dim=-1)
    return posterior, {
        "selection_mode": query_packet.selection_mode.value,
        "assignment_compression": "top2",
        "composition": "SparseObjectAssignments.element_posterior.mixture_sum",
        "query_token_capacity": maximum_query_tokens,
        "assignment_unknown_mass_mean": float(assignments.unknown_weight.mean()),
        "element_probability_maximum": float(posterior.max()),
    }


def _load_fragment_prototypes(
    manifest_path: Path,
    *,
    state: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = manifest_path.resolve(strict=True)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("fragment language manifest schema differs")
    if manifest.get("scene_label") != state["scene_label"]:
        raise ValueError("fragment language memory scene differs from scene state")
    policy = manifest.get("information_policy", {})
    required_false = (
        "benchmark_labels_opened",
        "benchmark_masks_opened",
        "text_queries_opened",
        "target_rgb_opened",
    )
    if any(policy.get(key) is not False for key in required_false):
        raise ValueError("fragment language memory opened a forbidden information channel")
    bound_sam = {
        value for key, value in state["source_input_digests"].items()
        if key.startswith("sam_manifest_")
    }
    manifest_sam = {item["sha256"] for item in manifest.get("sam_manifests", [])}
    if bound_sam != manifest_sam:
        raise ValueError("fragment language memory and scene state use different SAM manifests")

    records = manifest.get("outputs", [])
    by_frame = {int(item["frame_id"]): item for item in records}
    source_frames = list(map(int, state["source_frames"]))
    if len(by_frame) != len(records) or len(source_frames) != len(set(source_frames)) or set(by_frame) != set(source_frames):
        raise ValueError("fragment language frame inventory differs from scene state")
    # Token IDs were appended in mapping-view order, which need not be temporal.
    records = [by_frame[frame] for frame in source_frames]
    masked, context = [], []
    for record in records:
        output = Path(record["output"]).resolve(strict=True)
        if sha256_file(output) != record["output_sha256"]:
            raise ValueError("fragment language frame digest differs")
        payload = torch.load(output, map_location="cpu", weights_only=False)
        if payload.get("schema") != FRAME_SCHEMA:
            raise ValueError("fragment language frame schema differs")
        if int(payload.get("frame_id", -1)) != int(record["frame_id"]):
            raise ValueError("fragment descriptor frame identity differs")
        if payload["metadata"]["sam_fragment_payload_sha256"] != record[
            "sam_fragment_payload_sha256"
        ]:
            raise ValueError("fragment descriptor is not bound to the expected SAM payload")
        frame_masked = torch.as_tensor(payload["masked_crop_descriptor"]).float()
        frame_context = torch.as_tensor(payload["context_crop_descriptor"]).float()
        expected_shape = (int(record["proposal_count"]), 1536)
        if frame_masked.shape != expected_shape or frame_context.shape != expected_shape:
            raise ValueError("per-frame fragment descriptor counts or dimensions differ")
        if any(not bool(torch.isfinite(value).all()) or bool((value.norm(dim=-1) <= 1e-8).any())
               for value in (frame_masked, frame_context)):
            raise ValueError("fragment descriptors must be finite and nonzero")
        masked.append(frame_masked)
        context.append(frame_context)
    masked_tensor = torch.cat(masked)
    context_tensor = torch.cat(context)
    proposal_token_ids = torch.as_tensor(state["prototype_token_ids"], dtype=torch.long)
    if masked_tensor.shape != context_tensor.shape or masked_tensor.shape[0] != len(
        proposal_token_ids
    ):
        raise ValueError("fragment descriptors do not align with scene-state proposals")
    # Masked identity and context are separate prototypes; neither is averaged
    # away.  Prototype-max retrieval chooses the strongest legal source view.
    descriptors = torch.cat([masked_tensor, context_tensor], dim=0)
    token_ids = proposal_token_ids.repeat(2)
    return descriptors, token_ids, {
        "semantic_contract_valid": True,
        "descriptor_source": "official_masked_and_context_crop_summary_prototypes",
        "fragment_language_manifest": str(path),
        "fragment_language_manifest_sha256": sha256_file(path),
        "proposal_count": int(masked_tensor.shape[0]),
        "prototype_count": int(descriptors.shape[0]),
    }


def _evaluate_masks(
    *,
    carrier: Any,
    element_probability: torch.Tensor,
    views: dict[int, Any],
    annotations: dict[int, list[dict[str, Any]]],
    categories: list[str],
    source_height: int,
    source_width: int,
    height: int,
    width: int,
    pixel_threshold: float,
) -> dict[str, Any]:
    category_ious = {category: [] for category in categories}
    intersection_sum = 0
    union_sum = 0
    per_observation = []
    for frame_id, objects in sorted(annotations.items()):
        if frame_id not in views:
            raise KeyError(f"labeled frame {frame_id} is absent from COLMAP registration")
        prediction = carrier.render_posterior(
            element_probability, _camera(views[frame_id], frame_id, height, width)
        ) >= pixel_threshold
        target = _ground_truth_masks(
            objects, categories, height, width, source_height, source_width
        )
        for query_index, category in enumerate(categories):
            if not bool(target[category].any()):
                continue
            iou, intersection, union = _binary_iou(
                prediction[..., query_index], target[category]
            )
            category_ious[category].append(iou)
            intersection_sum += intersection
            union_sum += union
            per_observation.append({
                "frame_id": frame_id,
                "category": category,
                "iou": iou,
                "selected_element_fraction": float(
                    (element_probability[:, query_index] >= pixel_threshold).float().mean()
                ),
            })
    category_metrics = {
        category: {
            "mean_iou": float(np.mean(values)) if values else None,
            "observation_count": len(values),
        }
        for category, values in category_ious.items()
    }
    valid = [value["mean_iou"] for value in category_metrics.values() if value["mean_iou"] is not None]
    return {
        "macro_category_miou": float(np.mean(valid)),
        "micro_iou": float(intersection_sum / union_sum) if union_sum else None,
        "category_metrics": category_metrics,
        "per_observation": per_observation,
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    state_path = Path(args.scene_state)
    state = _load_state(state_path, expected_sha256=args.expected_scene_state_sha256)
    if state["scene_label"] != args.scene_label:
        raise ValueError("scene-state label differs from the requested scene")
    height, width = (int(value) for value in args.raster_shape)
    if height <= 0 or width <= 0 or not 0 < args.pixel_threshold < 1:
        raise ValueError("raster shape or pixel threshold is invalid")
    carrier = _build_carrier(state)
    scene_root = Path(args.scene_root).resolve(strict=True)
    sparse = scene_root / "sparse" / "0"
    views = _read_images(
        sparse / "images.bin", _read_cameras_binary(sparse / "cameras.bin")
    )
    annotations, categories, source_height, source_width = _load_labels(
        Path(args.label_root).resolve(strict=True), args.scene_label
    )
    device = torch.device(args.device)
    query_text = _load_text_cache(Path(args.text_query_cache), categories, device)
    negative_text = _load_text_cache(
        Path(args.negative_text_query_cache), list(GENERIC_NEGATIVES), device
    )

    if args.fragment_language_manifest:
        descriptors, token_ids, semantic_audit = _load_fragment_prototypes(
            Path(args.fragment_language_manifest), state=state
        )
    else:
        descriptors = torch.as_tensor(state["prototype_descriptors"]).float()
        token_ids = torch.as_tensor(state["prototype_token_ids"]).long()
        semantic_audit = {
            "semantic_contract_valid": False,
            "descriptor_source": "legacy_invalid_per_spatial_token_summary_head",
            "proposal_count": int(descriptors.shape[0]),
            "prototype_count": int(descriptors.shape[0]),
        }
    token_probability, _ = prototype_max_token_posterior(
        descriptors.to(device),
        token_ids.to(device),
        query_text,
        negative_text,
        num_tokens=int(state["completed_membership"].shape[1]),
        temperature=float(args.query_temperature),
        null_similarity=float(args.null_similarity),
    )
    token_probability = token_probability.cpu()

    assignment = state[f"{args.assignment_state}_membership"]
    evaluations = {}
    capacity_modes = (("all_tokens", None), ("top3_diagnostic", 3))
    for name, maximum_tokens in capacity_modes:
        element_probability, composition = compose_object_queries(
            assignment,
            token_probability,
            maximum_query_tokens=maximum_tokens,
        )
        evaluations[name] = {
            "composition": composition,
            "metrics": _evaluate_masks(
                carrier=carrier,
                element_probability=element_probability,
                views=views,
                annotations=annotations,
                categories=categories,
                source_height=source_height,
                source_width=source_width,
                height=height,
                width=width,
                pixel_threshold=float(args.pixel_threshold),
            ),
        }
    report = {
        "schema": REPORT_SCHEMA,
        "development_only": True,
        "promotion_eligible": bool(semantic_audit["semantic_contract_valid"]),
        "scene_label": args.scene_label,
        "scene_state": str(state_path.resolve()),
        "scene_state_sha256": state["scene_state_sha256"],
        "cold_loaded_in_independent_process": True,
        "assignment_state": args.assignment_state,
        "semantic_memory": semantic_audit,
        "query_contract": {
            "selection_mode": "multi_instance",
            "temperature": float(args.query_temperature),
            "null_similarity": float(args.null_similarity),
            "probability_calibration": "fixed_generic_negative_sigmoid_development",
            "benchmark_threshold_tuned": False,
        },
        "raster_shape": [height, width],
        "pixel_threshold": float(args.pixel_threshold),
        "evaluations": evaluations,
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
    parser.add_argument("--text-query-cache", required=True)
    parser.add_argument("--negative-text-query-cache", required=True)
    parser.add_argument("--fragment-language-manifest")
    parser.add_argument("--assignment-state", choices=("observed", "completed"), default="observed")
    parser.add_argument("--query-temperature", type=float, default=0.03)
    parser.add_argument("--null-similarity", type=float, default=0.0)
    parser.add_argument("--pixel-threshold", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="Bound intra-op CPU parallelism so independent scene jobs can coexist.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    report = run(args)
    print(json.dumps({
        key: value["metrics"]
        for key, value in report["evaluations"].items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
