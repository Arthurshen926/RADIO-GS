#!/usr/bin/env python3
"""Materialize target-RGB-assisted LERF-2D masks without opening GT masks.

The input is the immutable three-scale scalar-map bundle produced by
``render_ours_lerf2d_scalar_maps.py``.  This adapter reproduces the frozen
Occam level selection and coarse-mask readout, proposes one RGB/GrabCut
boundary refinement, and accepts it only with fixed query-posterior gates.

This is deliberately a prediction-only boundary.  Annotation JSON files,
segmentation polygons, bounding boxes, GT masks, and metrics are not accepted
as inputs.  The emitted manifest binds every scalar map, target RGB image,
prediction mask, fixed policy value, and source implementation by SHA-256.
Scoring is performed separately by
``score_lerf2d_sealed_boundary_predictions.py``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

from radio_gs.scripts.eval_lerf_grounding import (
    heatmap_peak_in_shape,
    refine_mask_with_rgb_edges,
)
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
from radio_gs.scripts.eval_prerendered_lerf_features import (
    _box_filter,
    _localization_hit,
    _normalize_heatmap,
    _smooth_mask,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    sha256_file,
    write_bytes_noclobber,
    write_frozen_json,
)


SCHEMA_VERSION = 1
RGB_ROOT_AUTHORITY_TYPE = "radio_gs_lerf2d_target_rgb_root_authority"
PREDICTION_ARTIFACT_TYPE = "radio_gs_lerf2d_rgb_boundary_prediction_bundle"
METHOD_NAME = "RADIO-GS current-field exact scalar + target-RGB GrabCut boundary v1"


class RgbBoundaryProtocolError(ValueError):
    """Raised before publication when a prediction input is not authoritative."""


@dataclass(frozen=True)
class RgbBoundaryPolicy:
    """Frozen, scene-independent candidate construction and admission policy."""

    activation_kernel: int = 30
    mask_threshold: float = 0.5
    smooth_kernel: int = 7
    grabcut_iterations: int = 1
    grabcut_dilate_pixels: int = 5
    grabcut_erode_pixels: int = 2
    minimum_posterior_mean_ratio: float = 0.85
    minimum_posterior_mass_ratio: float = 0.25
    minimum_area_ratio: float = 0.70
    maximum_area_ratio: float = 1.30
    require_peak: bool = True


def _policy_from_mapping(value: Mapping[str, Any] | None) -> RgbBoundaryPolicy:
    if value is None:
        return RgbBoundaryPolicy()
    expected = set(asdict(RgbBoundaryPolicy()))
    if set(value) != expected:
        raise RgbBoundaryProtocolError(
            f"policy fields differ: expected {sorted(expected)}, got {sorted(value)}"
        )
    policy = RgbBoundaryPolicy(**dict(value))
    if policy.activation_kernel != 30 or policy.smooth_kernel != 7:
        raise RgbBoundaryProtocolError("policy must preserve the frozen Occam 30/7 readout")
    if not math.isclose(policy.mask_threshold, 0.5, abs_tol=0.0):
        raise RgbBoundaryProtocolError("policy must preserve the frozen 0.5 threshold")
    if policy.minimum_area_ratio < 0 or policy.maximum_area_ratio < policy.minimum_area_ratio:
        raise RgbBoundaryProtocolError("policy area-ratio bounds are invalid")
    return policy


def occam_coarse_prediction(
    relevance_lhw: np.ndarray,
    *,
    policy: RgbBoundaryPolicy,
) -> tuple[np.ndarray, np.ndarray, int, list[float], int, list[list[int]]]:
    """Return the exact frozen coarse mask and its selected posterior.

    ``relevance_lhw`` is one query over all levels.  Level selection follows
    the released Occam rule: maximum of ``0.5 * (raw + box-filter(raw))``.
    Localization remains the maximum of the box-filtered raw relevance.
    """

    relevance = torch.as_tensor(relevance_lhw).float()
    if relevance.ndim != 3 or relevance.shape[0] != 3:
        raise RgbBoundaryProtocolError(
            f"expected query relevance [3,H,W], got {tuple(relevance.shape)}"
        )
    if not bool(torch.isfinite(relevance).all()):
        raise RgbBoundaryProtocolError("query relevance contains non-finite values")
    activated_by_level: list[torch.Tensor] = []
    level_scores: list[float] = []
    localization_scores: list[float] = []
    localization_coords: list[torch.Tensor] = []
    for level in relevance:
        filtered = _box_filter(level, policy.activation_kernel, "opencv_filter2d")
        activated = 0.5 * (filtered + level)
        activated_by_level.append(activated)
        level_scores.append(float(activated.max()))
        localization_scores.append(float(filtered.max()))
        localization_coords.append(torch.nonzero(filtered == filtered.max()))
    chosen_level = int(np.argmax(level_scores))
    localization_level = int(np.argmax(localization_scores))
    posterior = _normalize_heatmap(activated_by_level[chosen_level])
    coarse = _smooth_mask(
        posterior > policy.mask_threshold,
        policy.smooth_kernel,
        "langsplat_legacy",
    )
    coords = localization_coords[localization_level].cpu().tolist()
    return (
        coarse.cpu().numpy().astype(bool),
        posterior.cpu().numpy().astype(np.float32),
        chosen_level,
        level_scores,
        localization_level,
        [[int(y), int(x)] for y, x in coords],
    )


def choose_rgb_candidate_without_gt(
    coarse: np.ndarray,
    candidate: np.ndarray,
    posterior: np.ndarray,
    *,
    policy: RgbBoundaryPolicy,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Accept an RGB candidate only from scalar support and fixed gates."""

    base = np.asarray(coarse, dtype=bool)
    refined = np.asarray(candidate, dtype=bool)
    score = np.asarray(posterior, dtype=np.float32)
    if base.ndim != 2 or refined.shape != base.shape or score.shape != base.shape:
        raise RgbBoundaryProtocolError("coarse/candidate/posterior shapes differ")
    if not np.isfinite(score).all():
        raise RgbBoundaryProtocolError("posterior contains non-finite values")
    peak_y, peak_x = heatmap_peak_in_shape(torch.as_tensor(score), base.shape)
    base_area = int(base.sum())
    refined_area = int(refined.sum())
    area_ratio = float(refined_area / max(base_area, 1))
    base_mass = float(score[base].sum()) if base_area else 0.0
    refined_mass = float(score[refined].sum()) if refined_area else 0.0
    base_mean = float(score[base].mean()) if base_area else 0.0
    refined_mean = float(score[refined].mean()) if refined_area else 0.0
    mass_ratio = float(refined_mass / max(base_mass, 1e-12))
    mean_ratio = float(refined_mean / max(base_mean, 1e-12))
    peak_in_refined = bool(refined[peak_y, peak_x]) if refined_area else False
    checks = {
        "coarse_nonempty": base_area > 0,
        "candidate_nonempty": refined_area > 0,
        "area_ratio": policy.minimum_area_ratio <= area_ratio <= policy.maximum_area_ratio,
        "posterior_mass_ratio": mass_ratio >= policy.minimum_posterior_mass_ratio,
        "posterior_mean_ratio": mean_ratio >= policy.minimum_posterior_mean_ratio,
        "peak_containment": peak_in_refined or not policy.require_peak,
    }
    accepted = all(checks.values())
    report: dict[str, Any] = {
        "candidate": "target_rgb_grabcut",
        "accepted": bool(accepted),
        "fallback": "coarse_occam_mask" if not accepted else None,
        "checks": checks,
        "coarse_area": base_area,
        "candidate_area": refined_area,
        "area_ratio": area_ratio,
        "posterior_mass_ratio": mass_ratio,
        "posterior_mean_ratio": mean_ratio,
        "peak_yx": [int(peak_y), int(peak_x)],
        "peak_in_candidate": peak_in_refined,
    }
    return (refined if accepted else base).copy(), report


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _load_rgb_authority(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str, Path]:
    encoded = _read_stable_regular_file(path.resolve(strict=True), label="RGB root authority")
    digest = sha256_file(path)
    if digest != _require_sha256(expected_sha256, label="RGB authority SHA256"):
        raise RgbBoundaryProtocolError("RGB root authority SHA256 differs")
    payload = _load_json_bytes(encoded, label="RGB root authority")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RgbBoundaryProtocolError("RGB authority schema_version differs")
    if payload.get("artifact_type") != RGB_ROOT_AUTHORITY_TYPE:
        raise RgbBoundaryProtocolError("RGB authority artifact_type differs")
    if payload.get("target_rgb_authorized") is not True:
        raise RgbBoundaryProtocolError("RGB authority does not authorize target RGB")
    for forbidden in (
        "benchmark_masks_opened",
        "benchmark_segmentation_opened",
        "benchmark_bboxes_opened",
        "benchmark_metrics_opened",
    ):
        if payload.get(forbidden) is not False:
            raise RgbBoundaryProtocolError(f"RGB authority {forbidden} must be false")
    return dict(payload), digest, path.resolve(strict=True)


def _resolve_rgb(scene: str, camera_name: str, authority: Mapping[str, Any]) -> Path:
    scenes = _require_mapping(authority.get("scenes"), label="RGB authority scenes")
    scene_entry = _require_mapping(scenes.get(scene), label=f"RGB authority {scene}")
    root = Path(str(scene_entry.get("scene_root", ""))).expanduser().resolve(strict=True)
    subdir = str(scene_entry.get("image_subdir", "images"))
    if not subdir or Path(subdir).is_absolute() or ".." in Path(subdir).parts:
        raise RgbBoundaryProtocolError(f"{scene}: invalid RGB image_subdir")
    candidate = root / subdir / camera_name
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RgbBoundaryProtocolError(f"{scene}: RGB path escaped scene root") from error
    if resolved != candidate.absolute() or not resolved.is_file():
        raise RgbBoundaryProtocolError(f"{scene}: RGB path must be a regular non-symlink file")
    return resolved


def materialize(
    *,
    score_manifest: Path,
    score_manifest_sha256: str,
    rgb_authority: Path,
    rgb_authority_sha256: str,
    output_dir: Path,
    policy: RgbBoundaryPolicy | None = None,
) -> dict[str, Any]:
    """Build and seal a target-RGB-assisted prediction bundle."""

    policy = policy or RgbBoundaryPolicy()
    manifest_path = score_manifest.resolve(strict=True)
    expected_manifest_sha = _require_sha256(
        score_manifest_sha256, label="score manifest SHA256"
    )
    manifest_encoded = _read_stable_regular_file(manifest_path, label="score manifest")
    observed_manifest_sha = sha256_file(manifest_path)
    if observed_manifest_sha != expected_manifest_sha:
        raise RgbBoundaryProtocolError("score manifest SHA256 differs")
    scalar = _load_json_bytes(manifest_encoded, label="score manifest")
    if scalar.get("artifact_type") != SCALAR_BUNDLE_TYPE:
        raise RgbBoundaryProtocolError("input is not a frozen LERF2D scalar bundle")
    if scalar.get("score_semantics") != SCORE_SEMANTICS:
        raise RgbBoundaryProtocolError("input scalar semantics differ")
    scales = _validate_scales(scalar.get("scales"))
    rgb_roots, rgb_digest, rgb_path = _load_rgb_authority(
        rgb_authority, rgb_authority_sha256
    )
    raw_scenes = _require_mapping(scalar.get("scenes"), label="score scenes")
    authority_scenes = _require_mapping(rgb_roots.get("scenes"), label="RGB scenes")
    if tuple(raw_scenes) != tuple(authority_scenes) or set(raw_scenes) != set(authority_scenes):
        raise RgbBoundaryProtocolError("RGB authority scene cohort/order differs")
    output = output_dir.expanduser().absolute()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"prediction output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    published_scenes: dict[str, Any] = {}
    total_queries = 0
    accepted_count = 0
    for scene, scene_value in raw_scenes.items():
        scene_entry = _require_mapping(scene_value, label=f"score scenes.{scene}")
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
            query_ids = entry.get("query_ids")
            query_texts = entry.get("query_texts")
            if not isinstance(query_ids, list) or not isinstance(query_texts, list):
                raise RgbBoundaryProtocolError(f"{scene}/{frame}: query axis is malformed")
            if len(query_ids) != expected_shape[1] or len(query_texts) != expected_shape[1]:
                raise RgbBoundaryProtocolError(f"{scene}/{frame}: query axis length differs")
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
            camera_name = entry.get("camera_name")
            if not isinstance(camera_name, str) or Path(camera_name).stem != frame:
                raise RgbBoundaryProtocolError(f"{scene}/{frame}: camera name differs")
            rgb_source = _resolve_rgb(scene, camera_name, rgb_roots)
            rgb_sha = sha256_file(rgb_source)
            rgb_bgr = cv2.imread(str(rgb_source), cv2.IMREAD_COLOR)
            if rgb_bgr is None or tuple(rgb_bgr.shape[:2]) != expected_shape[-2:]:
                raise RgbBoundaryProtocolError(
                    f"{scene}/{frame}: target RGB resolution differs from scalar map"
                )
            predictions = np.zeros(expected_shape[1:], dtype=np.uint8)
            query_reports: list[dict[str, Any]] = []
            for query_index in range(expected_shape[1]):
                coarse, posterior, chosen_level, level_scores, loc_level, loc_coords = (
                    occam_coarse_prediction(
                        score_map[:, query_index],
                        policy=policy,
                    )
                )
                candidate = refine_mask_with_rgb_edges(
                    rgb_bgr,
                    coarse,
                    iterations=policy.grabcut_iterations,
                    dilate_pixels=policy.grabcut_dilate_pixels,
                    erode_pixels=policy.grabcut_erode_pixels,
                )
                prediction, report = choose_rgb_candidate_without_gt(
                    coarse,
                    candidate,
                    posterior,
                    policy=policy,
                )
                predictions[query_index] = prediction.astype(np.uint8)
                accepted_count += int(report["accepted"])
                total_queries += 1
                query_reports.append(
                    {
                        "query_id": str(query_ids[query_index]),
                        "query_text": str(query_texts[query_index]),
                        "chosen_level": chosen_level,
                        "level_scores": level_scores,
                        "localization_level": loc_level,
                        "localization_coords_yx": loc_coords,
                        "boundary": report,
                    }
                )
            relative_mask = Path("masks") / scene / f"{frame}.npy"
            mask_path = output / relative_mask
            write_bytes_noclobber(mask_path, _npy_bytes(predictions))
            published_frames[frame] = {
                "annotation_sha256": entry.get("annotation_sha256"),
                "camera_name": camera_name,
                "query_ids": list(query_ids),
                "query_texts": list(query_texts),
                "resolution_hw": list(expected_shape[-2:]),
                "score_map": {"path": str(map_path), "sha256": map_sha},
                "target_rgb": {"path": str(rgb_source), "sha256": rgb_sha},
                "prediction_file": str(relative_mask),
                "prediction_sha256": sha256_file(mask_path),
                "prediction_shape_qhw": list(predictions.shape),
                "queries": query_reports,
            }
        published_scenes[scene] = {"frames": published_frames}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": PREDICTION_ARTIFACT_TYPE,
        "status": "sealed_before_benchmark_mask_or_metric_access",
        "method": METHOD_NAME,
        "policy": asdict(policy),
        "score_manifest": {
            "path": str(manifest_path),
            "sha256": observed_manifest_sha,
            "method": scalar.get("method"),
            "protocol_freeze": scalar.get("protocol_freeze"),
        },
        "rgb_root_authority": {"path": str(rgb_path), "sha256": rgb_digest},
        "implementation": file_record(Path(__file__)),
        "source_access": {
            "target_rgb_opened": True,
            "benchmark_annotation_json_opened": False,
            "benchmark_segmentation_opened": False,
            "benchmark_bboxes_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_metrics_opened": False,
            "candidate_selected_with_gt": False,
        },
        "cohort": {
            "scenes": list(raw_scenes),
            "queries": total_queries,
            "rgb_candidate_accepted": accepted_count,
        },
        "scales": scales,
        "scenes": published_scenes,
    }
    manifest_output = output / "prediction_manifest.json"
    write_frozen_json(manifest_output, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-manifest", type=Path, required=True)
    parser.add_argument("--score-manifest-sha256", required=True)
    parser.add_argument("--rgb-authority", type=Path, required=True)
    parser.add_argument("--rgb-authority-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--policy-json",
        type=Path,
        default=None,
        help="Optional exact RgbBoundaryPolicy JSON; defaults are frozen v1 values.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    policy = None
    if args.policy_json is not None:
        value = json.loads(args.policy_json.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise RgbBoundaryProtocolError("policy JSON must be an object")
        policy = _policy_from_mapping(value)
    payload = materialize(
        score_manifest=args.score_manifest,
        score_manifest_sha256=args.score_manifest_sha256,
        rgb_authority=args.rgb_authority,
        rgb_authority_sha256=args.rgb_authority_sha256,
        output_dir=args.output_dir,
        policy=policy,
    )
    print(json.dumps({"status": payload["status"], "cohort": payload["cohort"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
