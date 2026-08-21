#!/usr/bin/env python3
"""Remove SAM3 components disconnected from the sealed NVOS identity unary.

Official SAM3 supplies extent, but one decoded mask can contain several
disconnected instances.  A component survives only when it contains an
explicit positive point used by the frozen signed-evidence selector or
explains a fixed fraction of the coarse field-positive support.  The operation
can only delete disconnected extent; it cannot move the identity peak or add a
new component.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from radio_gs.evaluation.promptable_segmentation import resize_mask_nearest
from radio_gs.querying.transient_rgb_sam import (
    FROZEN_POLICY,
    deterministic_signed_point_trials,
)
from radio_gs.scripts.predict_nvos_method_v1_field_box_sam3 import (
    SIGNED_POINT_CANDIDATE_ID,
    _write_json,
    _write_numpy,
)
from radio_gs.scripts.predict_nvos_method_v1_transient_sam import _sha256
from radio_gs.scripts.build_nvos_identity_component_threshold_authority import (
    ARTIFACT_TYPE as THRESHOLD_AUTHORITY_TYPE,
)


CANDIDATE_ID = "nvos-method-v1-identity-supported-sam3-components-v1"
LOCAL_DENSITY_CANDIDATE_ID = (
    "nvos-method-v1-component-local-density-sam3-components-v1"
)
MINIMUM_COARSE_SUPPORT_FRACTION = 0.05


def identity_supported_components(
    extent_mask: np.ndarray,
    coarse_positive: np.ndarray,
    positive_points_xy: np.ndarray,
    *,
    minimum_coarse_support_fraction: float = MINIMUM_COARSE_SUPPORT_FRACTION,
) -> np.ndarray:
    """Keep prompt-anchored or identity-supported connected components."""

    extent = np.asarray(extent_mask, dtype=bool)
    coarse = np.asarray(coarse_positive, dtype=bool)
    points = np.asarray(positive_points_xy)
    threshold = float(minimum_coarse_support_fraction)
    if extent.ndim != 2 or coarse.shape != extent.shape:
        raise ValueError("extent and coarse identity axes differ")
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError("positive points must have shape [P,2]")
    if not np.isfinite(points).all() or not 0 <= threshold <= 1:
        raise ValueError("positive points or support threshold is invalid")
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        extent.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return extent.copy()
    xy = np.rint(points).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, extent.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, extent.shape[0] - 1)
    keep = {int(labels[y, x]) for x, y in xy if int(labels[y, x]) > 0}
    coarse_mass = max(1, int(coarse.sum()))
    for component in range(1, count):
        overlap = int(((labels == component) & coarse).sum())
        if overlap / coarse_mass >= threshold:
            keep.add(component)
    # The selector already accepted this extent.  Fail closed to the original
    # rather than emit an empty mask if upstream evidence axes unexpectedly
    # cease to touch it.
    return np.isin(labels, list(keep)) if keep else extent.copy()


def identity_supported_components_local_density(
    extent_mask: np.ndarray,
    coarse_positive: np.ndarray,
    positive_points_xy: np.ndarray,
    *,
    minimum_local_identity_density: float = MINIMUM_COARSE_SUPPORT_FRACTION,
) -> np.ndarray:
    """Keep anchored components or components locally dense in identity support."""

    extent = np.asarray(extent_mask, dtype=bool)
    coarse = np.asarray(coarse_positive, dtype=bool)
    points = np.asarray(positive_points_xy)
    threshold = float(minimum_local_identity_density)
    if extent.ndim != 2 or coarse.shape != extent.shape:
        raise ValueError("extent and coarse identity axes differ")
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError("positive points must have shape [P,2]")
    if not np.isfinite(points).all() or not 0 <= threshold <= 1:
        raise ValueError("positive points or support threshold is invalid")
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        extent.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return extent.copy()
    xy = np.rint(points).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, extent.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, extent.shape[0] - 1)
    keep = {int(labels[y, x]) for x, y in xy if int(labels[y, x]) > 0}
    for component in range(1, count):
        area = max(1, int(stats[component, cv2.CC_STAT_AREA]))
        overlap = int(((labels == component) & coarse).sum())
        if overlap / area >= threshold:
            keep.add(component)
    return np.isin(labels, list(keep)) if keep else extent.copy()


def _load_manifest(path: str | Path, expected_sha256: str, *, label: str) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve(strict=True)
    if len(str(expected_sha256)) != 64 or _sha256(source) != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 differs")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, source


def _prediction_path(manifest: Mapping[str, Any], source: Path, scene: str, frame: str) -> Path:
    root = Path(str(manifest.get("prediction_root", ".")))
    if not root.is_absolute():
        root = source.parent / root
    path = (root / str(manifest["predictions"][scene][frame])).resolve(strict=True)
    if _sha256(path) != str(manifest["prediction_sha256"][scene][frame]):
        raise ValueError(f"sealed prediction differs: {scene}/{frame}")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    extent, extent_manifest = _load_manifest(
        args.extent_manifest, args.expected_extent_manifest_sha256, label="SAM3 extent manifest"
    )
    unary, unary_manifest = _load_manifest(
        args.signed_unary_manifest, args.expected_signed_unary_manifest_sha256, label="signed unary manifest"
    )
    threshold_authority: dict[str, Any] | None = None
    threshold_authority_path: Path | None = None
    if args.threshold_authority:
        threshold_authority, threshold_authority_path = _load_manifest(
            args.threshold_authority,
            args.expected_threshold_authority_sha256,
            label="component threshold authority",
        )
        if (
            threshold_authority.get("artifact_type") != THRESHOLD_AUTHORITY_TYPE
            or threshold_authority.get("target_mask_opened") is not False
            or threshold_authority.get("target_metric_opened") is not False
            or float(threshold_authority.get("selected_threshold", -1))
            != float(args.minimum_coarse_support_fraction)
        ):
            raise ValueError("component threshold authority differs")
    scenes = [str(value) for value in extent.get("scene_order", [])]
    if (
        extent.get("candidate_id") != SIGNED_POINT_CANDIDATE_ID
        or extent.get("kind") != "promptable_nvs_method_v1_field_box_sam3_predictions"
        or len(scenes) != 8
        or unary.get("kind") != "promptable_nvs_continuous_score_predictions"
        or extent.get("protocol_hash") != unary.get("protocol_hash")
        or bool(extent.get("target_mask_opened", True))
        or bool(extent.get("target_metric_opened", True))
    ):
        raise ValueError("sealed extent/unary contracts differ")

    output = Path(args.output_dir).expanduser().resolve()
    predictions: dict[str, dict[str, str]] = {}
    hashes: dict[str, dict[str, str]] = {}
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        frames = list(extent["predictions"][scene])
        if len(frames) != 1:
            raise ValueError(f"NVOS target frame differs: {scene}")
        frame = str(frames[0])
        extent_path = _prediction_path(extent, extent_manifest, scene, frame)
        unary_path = _prediction_path(unary, unary_manifest, scene, frame)
        extent_mask = np.load(extent_path, allow_pickle=False) >= 0
        signed_margin = np.load(unary_path, allow_pickle=False).astype(np.float32, copy=False)
        coarse = resize_mask_nearest(signed_margin >= 0, extent_mask.shape).astype(bool)
        points, labels = deterministic_signed_point_trials(
            np.maximum(signed_margin, 0),
            np.maximum(-signed_margin, 0),
            image_shape=extent_mask.shape,
            policy=FROZEN_POLICY,
        )
        positive_points = points[labels == 1]
        if args.support_denominator == "component_area":
            filtered = identity_supported_components_local_density(
                extent_mask,
                coarse,
                positive_points,
                minimum_local_identity_density=args.minimum_coarse_support_fraction,
            )
        else:
            filtered = identity_supported_components(
                extent_mask,
                coarse,
                positive_points,
                minimum_coarse_support_fraction=args.minimum_coarse_support_fraction,
            )
        relative = Path("scores") / scene / f"{frame}.npy"
        target = output / relative
        digest = _write_numpy(target, np.where(filtered, 0.5, -0.5).astype(np.float32))
        predictions[scene] = {frame: relative.as_posix()}
        hashes[scene] = {frame: digest}
        rows.append(
            {
                "scene_id": scene,
                "frame_id": frame,
                "input_components": int(cv2.connectedComponents(extent_mask.astype(np.uint8), 8)[0] - 1),
                "output_components": int(cv2.connectedComponents(filtered.astype(np.uint8), 8)[0] - 1),
                "input_foreground_pixels": int(extent_mask.sum()),
                "output_foreground_pixels": int(filtered.sum()),
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "promptable_nvs_method_v1_identity_supported_sam3_predictions",
        "candidate_id": (
            LOCAL_DENSITY_CANDIDATE_ID
            if args.support_denominator == "component_area"
            else CANDIDATE_ID
        ),
        "protocol_hash": extent["protocol_hash"],
        "scene_order": scenes,
        "prediction_root": ".",
        "predictions": predictions,
        "prediction_sha256": hashes,
        "parents": {
            "extent": {"path": str(extent_manifest), "sha256": args.expected_extent_manifest_sha256},
            "signed_unary": {"path": str(unary_manifest), "sha256": args.expected_signed_unary_manifest_sha256},
            "threshold_authority": (
                {
                    "path": str(threshold_authority_path),
                    "sha256": args.expected_threshold_authority_sha256,
                }
                if threshold_authority_path is not None
                else None
            ),
        },
        "method": {
            "operation": "delete_disconnected_extent_components_only",
            "positive_anchor": "all frozen deterministic positive selector points",
            "minimum_component_overlap_over_coarse_identity_support": float(args.minimum_coarse_support_fraction),
            "support_denominator": args.support_denominator,
            "adds_foreground": False,
        },
        "rows": rows,
        "all_eight_predictions_sealed": True,
        "evaluation_performed": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
    }
    manifest_path = output / "prediction_manifest.json"
    _write_json(manifest_path, manifest)
    return {**manifest, "prediction_manifest": str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extent-manifest", required=True)
    parser.add_argument("--expected-extent-manifest-sha256", required=True)
    parser.add_argument("--signed-unary-manifest", required=True)
    parser.add_argument("--expected-signed-unary-manifest-sha256", required=True)
    parser.add_argument("--minimum-coarse-support-fraction", type=float, default=MINIMUM_COARSE_SUPPORT_FRACTION)
    parser.add_argument(
        "--support-denominator",
        choices=("coarse_identity_mass", "component_area"),
        default="coarse_identity_mass",
    )
    parser.add_argument("--threshold-authority", default="")
    parser.add_argument("--expected-threshold-authority-sha256", default="")
    parser.add_argument("--output-dir", required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({"prediction_manifest": report["prediction_manifest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
