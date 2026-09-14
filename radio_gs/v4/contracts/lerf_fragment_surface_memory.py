"""Query-free observed SAM fragments lifted onto a frozen LERF surface.

This is the first level of the v4.1 fragment-to-object memory.  It preserves
every source proposal as an observation fact and deliberately performs no
cross-view merge, object birth, text lookup, or completion.  Visibility is
stored once per view; positive/negative/unknown fragment evidence can then be
reconstructed without materializing three large ``F x E`` tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.data.lerf_dataset import _read_cameras_binary
from radio_gs.scripts.extract_official_crop_summary_teacher import _atomic_torch_save
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.evaluation.lerf_common import (
    camera as _camera,
    validate_source_authority as _validate_source_authority,
)
from radio_gs.v4.evaluation.lerf_object_ceiling import _build_carrier, _load_state
from radio_gs.v4.evaluation.lerf_source_mask_gate import (
    _load_sam_records,
    _masks,
)
from radio_gs.v4.evaluation.real_sam_token_association import _lift_masks
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images


SCHEMA = "radio_gs.surface_object_memory_v4.observed_fragment_surface_memory.v1"


def source_mask_raster(payload: dict[str, Any], reference: tuple[int, int], mode: str) -> tuple[int, int]:
    """Keep semantic feature resolution separate from source mask resolution."""
    if mode not in {"reference", "native"}:
        raise ValueError("unsupported source mask raster")
    shape = payload.get("mask_shape", ()) if mode == "native" else reference
    if len(shape) != 2 or any(not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError("source mask raster must contain positive integer dimensions")
    return tuple(shape)


def _hierarchy_depth(parent_index: torch.Tensor) -> torch.Tensor:
    """Return a checked local hierarchy depth for every proposal."""

    parent = torch.as_tensor(parent_index, dtype=torch.long).cpu()
    if parent.ndim != 1:
        raise ValueError("fragment parent indices must be a vector")
    count = int(parent.numel())
    if count and bool(((parent < -1) | (parent >= count)).any()):
        raise ValueError("fragment parent index is outside the local proposal axis")
    depth = torch.full((count,), -1, dtype=torch.long)
    for proposal_id in range(count):
        trail: list[int] = []
        current = proposal_id
        seen: set[int] = set()
        while current >= 0 and depth[current] < 0:
            if current in seen:
                raise ValueError("fragment hierarchy contains a cycle")
            seen.add(current)
            trail.append(current)
            current = int(parent[current])
        value = 0 if current < 0 else int(depth[current]) + 1
        for item in reversed(trail):
            depth[item] = value
            value += 1
    return depth


def validate_memory(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize an observed-fragment memory payload."""

    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("observed fragment memory schema differs")
    positive = torch.as_tensor(payload.get("fragment_positive"), dtype=torch.float32).cpu()
    visibility = torch.as_tensor(payload.get("view_visibility"), dtype=torch.bool).cpu()
    view_index = torch.as_tensor(payload.get("fragment_view_index"), dtype=torch.long).cpu()
    frame_id = torch.as_tensor(payload.get("fragment_frame_id"), dtype=torch.long).cpu()
    local_index = torch.as_tensor(payload.get("fragment_local_index"), dtype=torch.long).cpu()
    quality = torch.as_tensor(payload.get("quality"), dtype=torch.float32).cpu()
    stability = torch.as_tensor(payload.get("stability"), dtype=torch.float32).cpu()
    parent_index = torch.as_tensor(payload.get("parent_index"), dtype=torch.long).cpu()
    hierarchy_depth = torch.as_tensor(payload.get("hierarchy_depth"), dtype=torch.long).cpu()
    if positive.ndim != 2 or min(positive.shape) <= 0:
        raise ValueError("fragment positive evidence must have shape [F,E]")
    fragment_count, element_count = positive.shape
    if visibility.ndim != 2 or visibility.shape[1] != element_count:
        raise ValueError("view visibility must have shape [V,E]")
    vectors = (view_index, frame_id, local_index, quality, stability, parent_index, hierarchy_depth)
    if any(value.shape != (fragment_count,) for value in vectors):
        raise ValueError("fragment metadata does not align with the fragment axis")
    if not bool(torch.isfinite(positive).all()) or bool(((positive < 0) | (positive > 1)).any()):
        raise ValueError("fragment positive evidence must be finite and lie in [0,1]")
    if not bool(torch.isfinite(quality).all()) or not bool(torch.isfinite(stability).all()):
        raise ValueError("fragment quality metadata must be finite")
    if bool(((view_index < 0) | (view_index >= visibility.shape[0])).any()):
        raise ValueError("fragment view indices are outside the visibility axis")
    if bool((positive > visibility[view_index].float() + 1e-5).any()):
        raise ValueError("positive fragment evidence appears outside its source visibility")
    source_frames = [int(value) for value in payload.get("source_frames", [])]
    if len(source_frames) != visibility.shape[0] or len(source_frames) != len(set(source_frames)):
        raise ValueError("source frame inventory does not align with view visibility")
    if not torch.equal(frame_id, torch.tensor(source_frames, dtype=torch.long)[view_index]):
        raise ValueError("fragment frame ids differ from their view indices")
    policy = payload.get("information_policy", {})
    forbidden = (
        "benchmark_labels_opened",
        "benchmark_masks_opened",
        "text_queries_opened",
        "target_rgb_opened",
    )
    if any(policy.get(key) is not False for key in forbidden):
        raise ValueError("observed fragment memory opened a forbidden information channel")
    payload = dict(payload)
    payload.update({
        "fragment_positive": positive,
        "view_visibility": visibility,
        "fragment_view_index": view_index,
        "fragment_frame_id": frame_id,
        "fragment_local_index": local_index,
        "quality": quality,
        "stability": stability,
        "parent_index": parent_index,
        "hierarchy_depth": hierarchy_depth,
    })
    return payload


def fragment_evidence(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct positive/negative/unknown evidence in ``F x E`` layout."""

    memory = validate_memory(payload)
    positive = memory["fragment_positive"]
    visible = memory["view_visibility"][memory["fragment_view_index"]]
    unknown = ~visible
    negative = visible.float() * (1.0 - positive)
    return positive, negative, unknown.float()


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cpu_threads <= 0:
        raise ValueError("cpu thread count must be positive")
    torch.set_num_threads(int(args.cpu_threads))
    height, width = (int(value) for value in args.raster_shape)
    if height <= 0 or width <= 0:
        raise ValueError("fragment-memory raster shape must be positive")
    state_path = Path(args.scene_state)
    state = _load_state(state_path, expected_sha256=args.expected_scene_state_sha256)
    if state["scene_label"] != args.scene_label:
        raise ValueError("scene state and requested scene labels differ")
    if [int(value) for value in state["source_frames"]] != sorted(state["source_frames"]):
        raise ValueError("scene-state source frames are not canonical")

    authority_path = Path(args.source_rgb_authority).resolve(strict=True)
    authority = json.loads(authority_path.read_text())
    source_frames = _validate_source_authority(authority, args.scene_label)
    if source_frames != list(state["source_frames"]):
        raise ValueError("source RGB authority differs from scene-state frame inventory")
    if sha256_file(authority_path) != state["source_input_digests"]["source_rgb_authority"]:
        raise ValueError("source RGB authority digest differs from scene state")
    sam_paths = [Path(value).resolve(strict=True) for value in args.sam_manifest]
    expected_sam = {
        value for key, value in state["source_input_digests"].items()
        if key.startswith("sam_manifest_")
    }
    if {sha256_file(path) for path in sam_paths} != expected_sam:
        raise ValueError("SAM manifest digest set differs from scene state")
    sam_records = _load_sam_records(sam_paths)
    if set(sam_records) != set(source_frames):
        raise ValueError("SAM fragment frames differ from source authority")

    scene_root = Path(args.scene_root).resolve(strict=True)
    sparse = scene_root / "sparse" / "0"
    views = _read_images(
        sparse / "images.bin", _read_cameras_binary(sparse / "cameras.bin")
    )
    if not set(source_frames).issubset(views):
        raise ValueError("a source fragment frame is absent from COLMAP registration")
    carrier = _build_carrier(state)
    mask_raster_mode = getattr(args, "source_mask_raster", "reference")
    if mask_raster_mode == "native" and carrier.reference_raster_shape != (height, width):
        raise ValueError("native source masks require a sealed carrier reference raster matching --raster-shape")
    device = torch.device(args.device)
    positive_rows: list[torch.Tensor] = []
    visibility_rows: list[torch.Tensor] = []
    view_indices: list[torch.Tensor] = []
    frame_ids: list[torch.Tensor] = []
    local_indices: list[torch.Tensor] = []
    qualities: list[torch.Tensor] = []
    stabilities: list[torch.Tensor] = []
    parents: list[torch.Tensor] = []
    depths: list[torch.Tensor] = []
    areas: list[torch.Tensor] = []
    frame_records = []
    for view_index_value, frame_id_value in enumerate(source_frames):
        cache_path = Path(sam_records[frame_id_value]["output"]).resolve(strict=True)
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        mask_height, mask_width = source_mask_raster(payload, (height, width), mask_raster_mode)
        masks = _masks(cache_path, mask_height, mask_width)
        count = int(masks.shape[0])
        parent = torch.as_tensor(payload["parent_index"], dtype=torch.long).cpu()
        if parent.shape != (count,):
            raise ValueError("SAM hierarchy does not align with fragment masks")
        camera = _camera(views[frame_id_value], frame_id_value, mask_height, mask_width)
        positive, visible = _lift_masks(carrier, camera, masks, device)
        if visible.shape != positive.shape or positive.shape != (count, carrier.num_elements):
            raise ValueError("lifted fragment evidence has an invalid shape")
        if count and not torch.equal(visible, visible[:1].expand_as(visible)):
            raise ValueError("same-view fragments unexpectedly have different visibility")
        positive_rows.append(positive.half().cpu())
        visibility_rows.append(visible[0].bool().cpu())
        view_indices.append(torch.full((count,), view_index_value, dtype=torch.long))
        frame_ids.append(torch.full((count,), frame_id_value, dtype=torch.long))
        local_indices.append(torch.arange(count, dtype=torch.long))
        qualities.append(torch.as_tensor(payload["quality"], dtype=torch.float32).cpu())
        stabilities.append(torch.as_tensor(payload["stability"], dtype=torch.float32).cpu())
        parents.append(parent)
        depths.append(_hierarchy_depth(parent))
        areas.append(torch.as_tensor(payload["proposal_area_fraction"], dtype=torch.float32).cpu())
        frame_records.append({
            "frame_id": int(frame_id_value),
            "view_index": int(view_index_value),
            "sam_fragment_payload": str(cache_path),
            "sam_fragment_payload_sha256": sam_records[frame_id_value]["output_sha256"],
            "proposal_count": count,
            "mask_raster_shape": [mask_height, mask_width],
            "visible_element_count": int(visible[0].sum()),
        })
        # Each source frame is consumed once. Native projection tables must
        # not accumulate in RAM across the complete mapping sequence.
        carrier._projection_cache.clear()
        print(
            f"lifting observed fragments: {view_index_value + 1}/{len(source_frames)} "
            f"frame={frame_id_value} proposals={count}",
            flush=True,
        )

    memory = validate_memory({
        "schema": SCHEMA,
        "scene_label": args.scene_label,
        "scene_state": str(state_path.resolve()),
        "scene_state_sha256": state["scene_state_sha256"],
        "raster_shape": [height, width],
        "source_mask_raster": mask_raster_mode,
        "raster_shape_semantics": "carrier_reference_not_necessarily_mask_sampling",
        "source_frames": source_frames,
        "fragment_positive": torch.cat(positive_rows),
        "view_visibility": torch.stack(visibility_rows),
        "fragment_view_index": torch.cat(view_indices),
        "fragment_frame_id": torch.cat(frame_ids),
        "fragment_local_index": torch.cat(local_indices),
        "quality": torch.cat(qualities),
        "stability": torch.cat(stabilities),
        "parent_index": torch.cat(parents),
        "hierarchy_depth": torch.cat(depths),
        "proposal_area_fraction": torch.cat(areas),
        "source_rgb_authority": str(authority_path),
        "source_rgb_authority_sha256": sha256_file(authority_path),
        "sam_manifests": [
            {"path": str(path), "sha256": sha256_file(path)} for path in sam_paths
        ],
        "frame_records": frame_records,
        "representation": {
            "level": "observed_fragment_memory",
            "cross_view_merge_performed": False,
            "completion_performed": False,
            "fragment_positive_layout": "fragment_by_surface_element",
            "negative_reconstruction": "view_visibility_times_one_minus_positive",
            "unknown_reconstruction": "one_minus_view_visibility",
        },
        "information_policy": {
            "source_only": True,
            "query_free": True,
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "target_rgb_opened": False,
        },
    })
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"observed fragment memory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(memory, output)
    report = {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "scene_label": args.scene_label,
        "source_view_count": len(source_frames),
        "fragment_count": int(memory["fragment_positive"].shape[0]),
        "surface_element_count": int(memory["fragment_positive"].shape[1]),
        "observed_positive_fraction": float((memory["fragment_positive"] > 0).float().mean()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--expected-scene-state-sha256", required=True)
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--source-rgb-authority", required=True)
    parser.add_argument("--sam-manifest", action="append", required=True)
    parser.add_argument("--raster-shape", nargs=2, type=int, required=True, metavar=("H", "W"))
    parser.add_argument("--source-mask-raster", choices=("reference", "native"), default="reference")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
