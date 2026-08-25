#!/usr/bin/env python3
"""Seal current formal LERF-2D posterior masks without opening RGB or GT."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.data.benchmark_paths import extract_feature_frame_index
from radio_gs.data.lerf_dataset import LERFDataset
from radio_gs.scripts.eval_lerf_grounding import (
    build_sam3_prompt_initial_mask,
    load_render_pipeline,
    neutralize_invalid_primitive_scores_for_render,
    normalize_primitive_scores_by_valid_mass,
    resolve_lerf_scene_root,
    validate_primitive_posterior_cache,
)
from radio_gs.scripts.eval_ours_lerf2d_scalar_maps import (
    ARTIFACT_TYPE as QUERY_AUTHORITY_TYPE,
    SCORE_SEMANTICS,
    _read_stable_regular_file,
    _require_mapping,
    _require_sha256,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    sha256_file,
    write_bytes_noclobber,
    write_frozen_json,
)


SCHEMA_VERSION = 1
COARSE_ARTIFACT_TYPE = "radio_gs_lerf2d_formal_posterior_coarse_prediction_receipt"
METHOD_NAME = "RADIO-GS retained identity_extent_posterior_v3 threshold-0.6"
FIXED_POLICY = {
    "posterior_threshold": 0.6,
    "threshold_mode": "fixed",
    "eval_at_image_resolution": True,
    "primitive_valid_normalization": True,
    "primitive_valid_coverage_power": 0.0,
    "feature_contribution_gamma": 1.0,
}


class FormalPosteriorProtocolError(ValueError):
    pass


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _load_json(path: Path, *, expected_sha256: str, label: str) -> tuple[dict[str, Any], str]:
    canonical = path.expanduser().resolve(strict=True)
    observed = sha256_file(canonical)
    if observed != _require_sha256(expected_sha256, label=f"{label} SHA256"):
        raise FormalPosteriorProtocolError(f"{label} SHA256 differs")
    try:
        payload = json.loads(_read_stable_regular_file(canonical, label=label))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalPosteriorProtocolError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise FormalPosteriorProtocolError(f"{label} must contain an object")
    return payload, observed


def materialize(
    *,
    scene: str,
    config: Path,
    checkpoint: Path,
    posterior_cache: Path,
    query_authority_manifest: Path,
    query_authority_manifest_sha256: str,
    output_dir: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    authority, authority_sha = _load_json(
        query_authority_manifest,
        expected_sha256=query_authority_manifest_sha256,
        label="query authority manifest",
    )
    if (
        authority.get("artifact_type") != QUERY_AUTHORITY_TYPE
        or authority.get("score_semantics") != SCORE_SEMANTICS
    ):
        raise FormalPosteriorProtocolError("query authority is not the frozen LERF-2D bundle")
    scenes = _require_mapping(authority.get("scenes"), label="authority scenes")
    scene_entry = _require_mapping(scenes.get(scene), label=f"authority scene {scene}")
    frames = _require_mapping(scene_entry.get("frames"), label=f"authority {scene}.frames")

    config_path = config.expanduser().resolve(strict=True)
    checkpoint_path = checkpoint.expanduser().resolve(strict=True)
    posterior_path = posterior_cache.expanduser().resolve(strict=True)
    output = output_dir.expanduser().absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"coarse output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device)
    pipeline = load_render_pipeline(
        str(config_path), str(checkpoint_path), torch_device, load_ply_rgb_features=False
    )
    model, _codec, renderer, _sharpener, _refiner, loaded_config, _hybrid = pipeline
    posterior = torch.load(posterior_path, map_location="cpu")
    query_names = posterior.get("metadata", {}).get("query_names")
    if not isinstance(query_names, list) or len(query_names) != len(set(query_names)):
        raise FormalPosteriorProtocolError("posterior query_names are malformed")
    query_to_index = {str(name): index for index, name in enumerate(query_names)}
    support_rows, support_valid = validate_primitive_posterior_cache(
        posterior, model.get_xyz(), query_names
    )
    metadata = _require_mapping(posterior.get("metadata"), label="posterior metadata")
    if (
        metadata.get("typed_posterior")
        != "official_sam3_siglip2_identity_extent_factorization_v3"
        or float(metadata.get("fixed_downstream_threshold", -1.0)) != 0.6
    ):
        raise FormalPosteriorProtocolError("posterior is not the retained formal v3 identity")
    support_rows = neutralize_invalid_primitive_scores_for_render(
        support_rows, support_valid
    ).to(device=torch_device, dtype=torch.float32)
    valid_column = torch.as_tensor(
        support_valid, device=torch_device, dtype=torch.float32
    )[:, None]

    scene_root = resolve_lerf_scene_root(scene, getattr(loaded_config, "scene_root", ""))
    dataset = LERFDataset(
        scene_root=str(scene_root),
        feature_dir=str(output / "absent_features"),
        annotation_dir=str(output / "absent_annotations"),
        feature_height=int(getattr(loaded_config, "feature_height", 30)),
        feature_width=int(getattr(loaded_config, "feature_width", 40)),
        allow_empty_features=True,
    )

    published_frames: dict[str, Any] = {}
    total_queries = 0
    for frame, raw_entry in frames.items():
        entry = _require_mapping(raw_entry, label=f"{scene}/{frame}")
        query_ids = entry.get("query_ids")
        query_texts = entry.get("query_texts")
        resolution = entry.get("map_resolution_hw")
        if (
            not isinstance(query_ids, list)
            or not isinstance(query_texts, list)
            or len(query_ids) != len(query_texts)
            or not isinstance(resolution, list)
            or len(resolution) != 2
        ):
            raise FormalPosteriorProtocolError(f"{scene}/{frame}: authority axes differ")
        missing = [query for query in query_texts if query not in query_to_index]
        if missing:
            raise FormalPosteriorProtocolError(f"{scene}/{frame}: posterior misses {missing}")
        frame_id = extract_feature_frame_index(Path(str(entry.get("camera_name", ""))))
        pose = dataset.pose_by_frame_idx.get(frame_id)
        if pose is None:
            raise FormalPosteriorProtocolError(f"{scene}/{frame}: registered pose is missing")
        indices = [query_to_index[str(query)] for query in query_texts]
        rows = torch.cat([support_rows[:, indices], valid_column], dim=1)
        rendered = renderer.render_feature_rows(
            model,
            torch.from_numpy(pose.copy()).float().to(torch_device),
            rows,
            alpha_normalize=True,
            contribution_gamma=1.0,
        )["feature_map"].float()
        heatmaps, _coverage = normalize_primitive_scores_by_valid_mass(
            rendered, coverage_power=0.0
        )
        height, width = int(resolution[0]), int(resolution[1])
        if tuple(heatmaps.shape[-2:]) != (height, width):
            heatmaps = F.interpolate(
                heatmaps[None], size=(height, width), mode="bilinear", align_corners=False
            )[0]
        masks = np.zeros((len(indices), height, width), dtype=np.uint8)
        reports = []
        for index, (query_id, query_text) in enumerate(zip(query_ids, query_texts)):
            mask = build_sam3_prompt_initial_mask(
                heatmaps[index],
                threshold_ratio=0.6,
                threshold_mode="fixed",
                threshold_mean_std_k=1.0,
                threshold_min_ratio=0.0,
                threshold_max_ratio=1.0,
                target_shape=(height, width),
                initial_refinement="none",
            )
            masks[index] = mask.astype(np.uint8)
            reports.append(
                {
                    "query_id": str(query_id),
                    "query_text": str(query_text),
                    "coarse_pixels": int(mask.sum()),
                }
            )
        relative = Path("coarse_masks") / scene / f"{frame}.npy"
        mask_path = output / relative
        write_bytes_noclobber(mask_path, _npy_bytes(masks))
        published_frames[frame] = {
            "annotation_sha256": entry.get("annotation_sha256"),
            "camera_name": entry.get("camera_name"),
            "query_ids": list(query_ids),
            "query_texts": list(query_texts),
            "resolution_hw": [height, width],
            "score_map": {"path": str(posterior_path), "sha256": sha256_file(posterior_path)},
            "coarse_prediction_file": str(relative),
            "coarse_prediction_sha256": sha256_file(mask_path),
            "prediction_shape_qhw": list(masks.shape),
            "queries": reports,
        }
        total_queries += len(indices)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COARSE_ARTIFACT_TYPE,
        "status": "coarse_predictions_sealed_before_target_rgb_or_gt_access",
        "method": METHOD_NAME,
        "policy": dict(FIXED_POLICY),
        "query_authority_manifest": {
            "path": str(query_authority_manifest.resolve(strict=True)),
            "sha256": authority_sha,
            "protocol_freeze": authority.get("protocol_freeze"),
        },
        "posterior_source": file_record(posterior_path),
        "config": file_record(config_path),
        "checkpoint": file_record(checkpoint_path),
        "implementation": file_record(Path(__file__).resolve()),
        "source_access": {
            "target_rgb_opened": False,
            "benchmark_annotation_json_opened": False,
            "benchmark_segmentation_opened": False,
            "benchmark_bboxes_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_metrics_opened": False,
            "candidate_selected_with_gt": False,
        },
        "cohort": {"scenes": [scene], "queries": total_queries},
        "scenes": {scene: {"frames": published_frames}},
    }
    write_frozen_json(output / "coarse_prediction_receipt.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--posterior-cache", type=Path, required=True)
    parser.add_argument("--query-authority-manifest", type=Path, required=True)
    parser.add_argument("--query-authority-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    payload = materialize(
        scene=args.scene,
        config=args.config,
        checkpoint=args.checkpoint,
        posterior_cache=args.posterior_cache,
        query_authority_manifest=args.query_authority_manifest,
        query_authority_manifest_sha256=args.query_authority_manifest_sha256,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps({"status": payload["status"], "cohort": payload["cohort"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
