#!/usr/bin/env python3
"""Seal frozen LERF-2D Occam 0.5 coarse masks without opening RGB or GT."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from radio_gs.scripts.eval_ours_lerf2d_scalar_maps import (
    ARTIFACT_TYPE as SCALAR_BUNDLE_TYPE,
    SCORE_SEMANTICS,
    _bundle_member,
    _load_json_bytes,
    _load_score_map,
    _read_stable_regular_file,
    _require_mapping,
    _require_sha256,
    _validate_scales,
)
from radio_gs.scripts.materialize_lerf2d_rgb_boundary_predictions import (
    RgbBoundaryPolicy,
    RgbBoundaryProtocolError,
    occam_coarse_prediction,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    sha256_file,
    write_bytes_noclobber,
    write_frozen_json,
)


SCHEMA_VERSION = 1
COARSE_ARTIFACT_TYPE = "radio_gs_lerf2d_occam_coarse_prediction_receipt"
METHOD_NAME = "RADIO-GS current-field exact scalar + frozen Occam 30/7 threshold-0.5"


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _parse_scenes(raw: str, available: Sequence[str]) -> tuple[str, ...]:
    if not raw.strip():
        return tuple(available)
    requested = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not requested or len(requested) != len(set(requested)):
        raise RgbBoundaryProtocolError("--scenes must be a non-empty unique comma list")
    unknown = set(requested) - set(available)
    if unknown:
        raise RgbBoundaryProtocolError(f"unknown scalar-manifest scenes: {sorted(unknown)}")
    # Preserve the frozen source order even if the CLI list is permuted.
    return tuple(scene for scene in available if scene in set(requested))


def materialize_coarse(
    *,
    score_manifest: Path,
    score_manifest_sha256: str,
    output_dir: Path,
    scenes: str = "",
) -> dict[str, Any]:
    manifest_path = score_manifest.resolve(strict=True)
    encoded = _read_stable_regular_file(manifest_path, label="score manifest")
    observed_sha = sha256_file(manifest_path)
    if observed_sha != _require_sha256(
        score_manifest_sha256, label="score manifest SHA256"
    ):
        raise RgbBoundaryProtocolError("score manifest SHA256 differs")
    scalar = _load_json_bytes(encoded, label="score manifest")
    if scalar.get("artifact_type") != SCALAR_BUNDLE_TYPE:
        raise RgbBoundaryProtocolError("input is not a frozen LERF2D scalar bundle")
    if scalar.get("score_semantics") != SCORE_SEMANTICS:
        raise RgbBoundaryProtocolError("input scalar semantics differ")
    scales = _validate_scales(scalar.get("scales"))
    raw_scenes = _require_mapping(scalar.get("scenes"), label="score scenes")
    selected_scenes = _parse_scenes(scenes, tuple(raw_scenes))
    output = output_dir.expanduser().absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"coarse output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    policy = RgbBoundaryPolicy()
    published_scenes: dict[str, Any] = {}
    total_queries = 0
    for scene in selected_scenes:
        scene_entry = _require_mapping(raw_scenes[scene], label=f"score scenes.{scene}")
        raw_frames = _require_mapping(scene_entry.get("frames"), label=f"{scene}.frames")
        published_frames: dict[str, Any] = {}
        for frame, frame_value in raw_frames.items():
            entry = _require_mapping(frame_value, label=f"{scene}/{frame}")
            shape = entry.get("map_shape_lqhw")
            if not isinstance(shape, list) or len(shape) != 4:
                raise RgbBoundaryProtocolError(f"{scene}/{frame}: malformed map shape")
            expected_shape = tuple(int(value) for value in shape)
            if expected_shape[0] != 3:
                raise RgbBoundaryProtocolError(f"{scene}/{frame}: expected three scales")
            query_ids, query_texts = entry.get("query_ids"), entry.get("query_texts")
            if (
                not isinstance(query_ids, list)
                or not isinstance(query_texts, list)
                or len(query_ids) != expected_shape[1]
                or len(query_texts) != expected_shape[1]
            ):
                raise RgbBoundaryProtocolError(f"{scene}/{frame}: query axis differs")
            if entry.get("scale_ids") != [scale["id"] for scale in scales]:
                raise RgbBoundaryProtocolError(f"{scene}/{frame}: scale axis differs")
            map_path = _bundle_member(
                manifest_path.parent,
                entry.get("map_file"),
                label=f"{scene}/{frame}.map_file",
            )
            score_map, map_sha = _load_score_map(
                map_path,
                expected_sha=_require_sha256(
                    entry.get("map_sha256"), label=f"{scene}/{frame}.map_sha256"
                ),
                expected_shape=expected_shape,
                label=f"{scene}/{frame} score map",
            )
            coarse_qhw = np.zeros(expected_shape[1:], dtype=np.uint8)
            query_reports = []
            for query_index in range(expected_shape[1]):
                coarse, _posterior, chosen, scores, loc_level, loc_coords = (
                    occam_coarse_prediction(score_map[:, query_index], policy=policy)
                )
                coarse_qhw[query_index] = coarse.astype(np.uint8)
                query_reports.append(
                    {
                        "query_id": str(query_ids[query_index]),
                        "query_text": str(query_texts[query_index]),
                        "chosen_level": int(chosen),
                        "level_scores": [float(value) for value in scores],
                        "localization_level": int(loc_level),
                        "localization_coords_yx": loc_coords,
                        "coarse_pixels": int(coarse.sum()),
                    }
                )
            relative = Path("coarse_masks") / scene / f"{frame}.npy"
            coarse_path = output / relative
            write_bytes_noclobber(coarse_path, _npy_bytes(coarse_qhw))
            camera_name = entry.get("camera_name")
            if not isinstance(camera_name, str) or Path(camera_name).stem != frame:
                raise RgbBoundaryProtocolError(f"{scene}/{frame}: camera name differs")
            published_frames[frame] = {
                "annotation_sha256": entry.get("annotation_sha256"),
                "camera_name": camera_name,
                "query_ids": list(query_ids),
                "query_texts": list(query_texts),
                "resolution_hw": list(expected_shape[-2:]),
                "score_map": {"path": str(map_path), "sha256": map_sha},
                "coarse_prediction_file": str(relative),
                "coarse_prediction_sha256": sha256_file(coarse_path),
                "prediction_shape_qhw": list(coarse_qhw.shape),
                "queries": query_reports,
            }
            total_queries += int(expected_shape[1])
        published_scenes[scene] = {"frames": published_frames}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COARSE_ARTIFACT_TYPE,
        "status": "coarse_predictions_sealed_before_target_rgb_or_gt_access",
        "method": METHOD_NAME,
        "policy": {
            "activation_kernel": 30,
            "mask_threshold": 0.5,
            "smooth_kernel": 7,
        },
        "score_manifest": {
            "path": str(manifest_path),
            "sha256": observed_sha,
            "method": scalar.get("method"),
            "protocol_freeze": scalar.get("protocol_freeze"),
        },
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
        "cohort": {"scenes": list(selected_scenes), "queries": total_queries},
        "scales": scales,
        "scenes": published_scenes,
    }
    write_frozen_json(output / "coarse_prediction_receipt.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-manifest", type=Path, required=True)
    parser.add_argument("--score-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenes", default="")
    args = parser.parse_args(argv)
    payload = materialize_coarse(
        score_manifest=args.score_manifest,
        score_manifest_sha256=args.score_manifest_sha256,
        output_dir=args.output_dir,
        scenes=args.scenes,
    )
    print(json.dumps({"status": payload["status"], "cohort": payload["cohort"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
