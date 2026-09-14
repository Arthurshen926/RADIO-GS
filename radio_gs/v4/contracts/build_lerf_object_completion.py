"""Build query-free completed membership for a LERF object-hypothesis memory.

This path never reuses the historical online token identities.  It reconstructs
the sealed F71 surface observations from source RGB and frozen RADIO maps,
adapts the two-level fragment/object memory to the categorical completion
checkpoint without hardening ambiguous rows, and writes completion only where
the object memory has no observed evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from radio_gs.data.lerf_dataset import _read_cameras_binary
from radio_gs.scripts.extract_official_crop_summary_teacher import _atomic_torch_save
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.completion.lerf_adapter import (
    apply_noise_matched_completion_candidate,
    build_real_token_runtime,
)
from radio_gs.v4.completion.scannet import (
    _load_source_rgb,
    _radio_projection_matrix,
)
from radio_gs.v4.contracts.build_lerf_object_hypotheses import (
    validate_hypothesis_memory,
)
from radio_gs.v4.contracts.lerf_fragment_surface_memory import validate_memory
from radio_gs.v4.evaluation.lerf_common import (
    accumulate_surface_features,
    camera as _camera,
    frame_index as _frame_index,
    load_dense_feature as _load_dense_feature,
    validate_semantic_manifest as _validate_semantic_manifest,
    validate_source_authority as _validate_source_authority,
)
from radio_gs.v4.evaluation.lerf_fragment_ceiling import _load_fragment_memory
from radio_gs.v4.evaluation.lerf_object_ceiling import _build_carrier, _load_state
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images


COMPLETION_SCHEMA = "radio_gs.surface_object_memory_v4.lerf_object_completion.v1"


def merge_completion_into_unobserved_rows(
    observed_membership: torch.Tensor,
    active_completion: torch.Tensor,
    active_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve every nonzero observed fact and fill exactly-zero rows only."""

    observed = torch.as_tensor(observed_membership, dtype=torch.float32).cpu()
    active = torch.as_tensor(active_completion, dtype=torch.float32).cpu()
    token_ids = torch.as_tensor(active_token_ids, dtype=torch.long).cpu()
    if observed.ndim != 2 or active.shape != (observed.shape[0], token_ids.numel()):
        raise ValueError("observed and active completion axes do not align")
    if token_ids.ndim != 1 or token_ids.numel() == 0:
        raise ValueError("active completion requires a non-empty token axis")
    if bool(((token_ids < 0) | (token_ids >= observed.shape[1])).any()):
        raise ValueError("active completion token ids are outside the object axis")
    if token_ids.unique().numel() != token_ids.numel():
        raise ValueError("active completion token ids must be unique")
    if not bool(torch.isfinite(active).all()) or bool(((active < 0) | (active > 1)).any()):
        raise ValueError("active completion must be finite and lie in [0,1]")
    writable = observed.max(-1).values == 0
    expanded = torch.zeros_like(observed)
    expanded[:, token_ids] = active
    completed = observed.clone()
    completed[writable] = expanded[writable]
    if bool((completed + 1e-7 < observed).any()):
        raise RuntimeError("completion erased observed evidence")
    return completed, writable


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cpu_threads <= 0 or args.inference_element_batch_size <= 0:
        raise ValueError("completion CPU and element batch sizes must be positive")
    torch.set_num_threads(int(args.cpu_threads))
    device = torch.device(args.device)
    state_path = Path(args.scene_state)
    state = _load_state(state_path, expected_sha256=args.expected_scene_state_sha256)
    hypothesis_path = Path(args.hypothesis_memory).resolve(strict=True)
    if sha256_file(hypothesis_path) != args.expected_hypothesis_memory_sha256:
        raise ValueError("object-hypothesis memory SHA256 differs")
    hypothesis = validate_hypothesis_memory(
        torch.load(hypothesis_path, map_location="cpu", weights_only=False)
    )
    if "completed_membership" in hypothesis:
        raise ValueError("object-hypothesis memory is already completed")
    fragment_path = Path(hypothesis["fragment_memory"]).resolve(strict=True)
    fragment = validate_memory(
        _load_fragment_memory(
            fragment_path,
            expected_sha256=hypothesis["fragment_memory_sha256"],
            scene_state_sha256=state["scene_state_sha256"],
        )
    )
    if any(
        value != args.scene_label
        for value in (state["scene_label"], hypothesis["scene_label"], fragment["scene_label"])
    ):
        raise ValueError("scene labels differ across completion inputs")

    authority_path = Path(args.source_rgb_authority).resolve(strict=True)
    authority = json.loads(authority_path.read_text())
    source_frames = _validate_source_authority(authority, args.scene_label)
    if source_frames != list(fragment["source_frames"]):
        raise ValueError("completion source frames differ from fragment memory")
    if sha256_file(authority_path) != fragment["source_rgb_authority_sha256"]:
        raise ValueError("completion source RGB authority differs from fragment memory")
    source_image_paths = {
        _frame_index(str(item["image_id"])): Path(item["path"]).resolve(strict=True)
        for item in authority["images"]
    }
    if set(source_image_paths) != set(source_frames):
        raise ValueError("source image paths differ from the source frame inventory")

    feature_dir = Path(args.radio_feature_dir).resolve(strict=True)
    manifest_value = authority.get("construction", {}).get("frame_manifest", {}).get("path")
    if not manifest_value:
        raise ValueError("source RGB authority lacks a bound RADIO frame manifest")
    manifest_path = Path(manifest_value).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    manifest_frames = _validate_semantic_manifest(
        manifest, manifest_path, authority, feature_dir, source_frames
    )
    if not set(source_frames).issubset(manifest_frames):
        raise ValueError("RADIO manifest lacks completion source frames")
    if sha256_file(manifest_path) != state["source_input_digests"]["semantic_frame_manifest"]:
        raise ValueError("RADIO frame manifest differs from the frozen scene state")

    first = _load_dense_feature(
        feature_dir / f"rgb_{source_frames[0]}.pt", expected_channels=1280
    )
    _, height, width = first.shape
    if [height, width] != list(fragment["raster_shape"]):
        raise ValueError("RADIO feature grid differs from the fragment memory")
    scene_root = Path(args.scene_root).resolve(strict=True)
    sparse = scene_root / "sparse" / "0"
    views = _read_images(
        sparse / "images.bin", _read_cameras_binary(sparse / "cameras.bin")
    )
    if not set(source_frames).issubset(views):
        raise ValueError("source completion frames are absent from COLMAP registration")
    carrier = _build_carrier(state)
    if carrier.normals is None:
        raise ValueError("LERF F71 completion requires frozen surface normals")
    feature_sum = torch.zeros(
        carrier.num_elements, 67, dtype=torch.float32, device=device
    )
    feature_mass = torch.zeros(carrier.num_elements, dtype=torch.float32, device=device)
    projection_matrix = _radio_projection_matrix().to(device)
    cameras = []
    feature_receipts = []
    for frame_id in source_frames:
        camera = _camera(views[frame_id], frame_id, height, width)
        dense_path = feature_dir / f"rgb_{frame_id}.pt"
        dense = _load_dense_feature(
            dense_path, expected_channels=1280,
            expected_height=height, expected_width=width,
        ).to(device).float()
        normalized = F.normalize(dense.permute(1, 2, 0), dim=-1, eps=1e-12)
        projected = F.normalize(
            normalized @ projection_matrix, dim=-1, eps=1e-12
        ).permute(2, 0, 1)
        rgb_path = source_image_paths[frame_id]
        rgb = _load_source_rgb(rgb_path, height, width).to(device).permute(2, 0, 1)
        accumulate_surface_features(
            feature_sum,
            feature_mass,
            torch.cat((rgb, projected), dim=0),
            carrier.project(camera),
            channel_chunk_size=int(args.channel_chunk_size),
        )
        cameras.append(camera)
        feature_receipts.append({
            "frame_id": int(frame_id),
            "radio_feature": str(dense_path.resolve(strict=True)),
            "radio_feature_sha256": sha256_file(dense_path),
            "source_rgb": str(rgb_path),
            "source_rgb_sha256": sha256_file(rgb_path),
        })
    source_visible = feature_mass > 0
    average = feature_sum / feature_mass[:, None].clamp_min(1e-8)
    rgb = (average[:, :3] * 2.0 - 1.0) * source_visible[:, None]
    radio = F.normalize(average[:, 3:], dim=-1, eps=1e-12) * source_visible[:, None]
    local_features = torch.cat(
        (rgb, source_visible.float()[:, None], radio, carrier.normals.to(device).float()),
        dim=-1,
    ).cpu()
    if local_features.shape != (carrier.num_elements, 71):
        raise RuntimeError("constructed LERF completion features do not satisfy F71")

    fragment_assignment = hypothesis["fragment_assignment"]
    fragment_view = fragment["fragment_view_index"]
    view_hypothesis_ids = []
    for view_index in range(len(source_frames)):
        rows = fragment_assignment[fragment_view == view_index]
        view_hypothesis_ids.append(
            torch.where((rows > 0).any(0))[0] if rows.numel() else torch.empty(0, dtype=torch.long)
        )
    runtime, adapter_audit = build_real_token_runtime(
        carrier=carrier,
        local_features=local_features,
        source_visible=source_visible.cpu(),
        observed_membership=hypothesis["observed_membership"],
        observation_cameras=cameras,
        view_token_ids=view_hypothesis_ids,
        observed_threshold=float(args.observed_threshold),
    )
    active_completion, null, inference_audit = apply_noise_matched_completion_candidate(
        runtime,
        report_path=args.completion_report,
        checkpoint_path=args.completion_checkpoint,
        device=device,
        element_batch_size=int(args.inference_element_batch_size),
    )
    completed, writable = merge_completion_into_unobserved_rows(
        hypothesis["observed_membership"],
        active_completion,
        runtime["active_token_ids"],
    )
    payload = dict(hypothesis)
    payload.update({
        "completed_membership": completed.half(),
        "completion_schema": COMPLETION_SCHEMA,
        "completion_source_hypothesis_memory": str(hypothesis_path),
        "completion_source_hypothesis_memory_sha256": args.expected_hypothesis_memory_sha256,
        "completion_audit": {
            "adapter": adapter_audit,
            "inference": inference_audit,
            "exactly_zero_observed_row_count": int(writable.sum()),
            "completion_nonzero_row_count": int(
                (writable & (completed.max(-1).values > 0)).sum()
            ),
            "observed_evidence_preserved": True,
            "ambiguous_observed_rows_completed": False,
            "mean_unknown_null_probability": float(
                null[runtime["partial"].unknown.any(-1)].mean()
            ),
            "radio_frame_manifest": str(manifest_path),
            "radio_frame_manifest_sha256": sha256_file(manifest_path),
            "feature_receipts": feature_receipts,
        },
        "representation": {
            **hypothesis["representation"],
            "completion_performed": True,
            "completion_model": "scene_disjoint_coherent_fragment_noise_mlp",
            "completion_write_domain": "exactly_zero_observed_rows_only",
        },
    })
    payload = validate_hypothesis_memory(payload)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"completed hypothesis memory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(payload, output)
    report = {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "scene_label": args.scene_label,
        "object_hypothesis_count": int(completed.shape[1]),
        "observed_nonzero_element_fraction": float(
            (hypothesis["observed_membership"].max(-1).values > 0).float().mean()
        ),
        "completed_nonzero_element_fraction": float(
            (completed.max(-1).values > 0).float().mean()
        ),
        "completion_audit": payload["completion_audit"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--expected-scene-state-sha256", required=True)
    parser.add_argument("--hypothesis-memory", required=True)
    parser.add_argument("--expected-hypothesis-memory-sha256", required=True)
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--source-rgb-authority", required=True)
    parser.add_argument("--radio-feature-dir", required=True)
    parser.add_argument("--completion-report", required=True)
    parser.add_argument("--completion-checkpoint", required=True)
    parser.add_argument("--observed-threshold", type=float, default=0.05)
    parser.add_argument("--channel-chunk-size", type=int, default=67)
    parser.add_argument("--inference-element-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
