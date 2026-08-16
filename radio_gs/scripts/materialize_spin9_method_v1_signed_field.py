#!/usr/bin/env python3
"""Seal one SPIn Method-v1 scene's scalar field prompts without target RGB.

The 1280-D feature maps are rendered into a scene-local temporary directory,
converted immediately to reference-prototype cosine margins, and removed.  A
successful scene leaves only small scalar maps plus a receipt binding them to
the complete factorized-v2 field and frozen camera/geometry authorities.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from radio_gs.data.promptable_nvs_manifest import (
    validate_manifest as validate_dataset_manifest,
)
from radio_gs.evaluation.promptable_feature_readout import (
    _prompt_prototypes,
    cosine_margin_scores,
    load_feature_map,
)
from radio_gs.five_benchmark_method_v1 import METHOD_ID, validate_method_authority
from radio_gs.scripts.render_promptable_nvs_features import render_protocol_scene
from radio_gs.scripts.run_spin9_method_v1_scene import (
    DATASET_MANIFEST,
    DEFAULT_RUN_ROOT,
    METHOD_AUTHORITY,
    SPIN_AUTHORITY,
    resolve_scene_assets,
)
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file


FINAL_FIELD_NAME = "generic_text_response_w005_s0_64.pth"
READOUT_PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "paper/artifacts/spin9_method_v1_transient_readout_preregistration_20260816.json"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_RUN_ROOT / "method_v1_readout/signed_field"


def _write_numpy(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _scene_row(rows: Sequence[Mapping[str, Any]], scene_id: str) -> Mapping[str, Any]:
    matches = [row for row in rows if str(row.get("scene_id")) == scene_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one frozen SPIn scene {scene_id!r}")
    return matches[0]


def _field_binding(scene_id: str, field_root: Path) -> dict[str, Any]:
    authority, authority_sha, authority_path = load_json_object(
        METHOD_AUTHORITY, label="Method-v1 authority"
    )
    validate_method_authority(authority)
    cohort = [
        str(value) for value in authority["frozen_cohorts"]["spin_nerf_available9"]
    ]
    if scene_id not in cohort:
        raise ValueError(f"scene is outside Method-v1 SPIn Available-Nine: {scene_id}")
    root = field_root / scene_id
    field = (root / FINAL_FIELD_NAME).resolve(strict=True)
    gate = (root / "method_v1_gate.json").resolve(strict=True)
    gate_payload, gate_sha, _gate_path = load_json_object(
        gate, label=f"{scene_id} Method-v1 field gate"
    )
    field_sha = sha256_file(field)
    if (
        gate_payload.get("status") != "pass"
        or gate_payload.get("benchmark") != "SPIn-NeRF Available-Nine"
        or gate_payload.get("scene") != scene_id
        or Path(str(gate_payload.get("field", ""))).resolve() != field
        or gate_payload.get("field_sha256") != field_sha
        or gate_payload.get("method_authority_sha256") != authority_sha
    ):
        raise ValueError(f"{scene_id} Method-v1 field gate differs")
    prereg, prereg_sha, prereg_path = load_json_object(
        READOUT_PREREGISTRATION, label="SPIn Method-v1 readout preregistration"
    )
    if prereg.get("status") != "frozen_before_first_method_v1_spin9_target_readout":
        raise ValueError("SPIn Method-v1 readout preregistration differs")
    return {
        "field": field,
        "field_sha256": field_sha,
        "gate": gate,
        "gate_sha256": gate_sha,
        "method_authority": authority_path,
        "method_authority_sha256": authority_sha,
        "readout_preregistration": prereg_path,
        "readout_preregistration_sha256": prereg_sha,
    }


def _camera_map(scene_id: str) -> tuple[Path, str]:
    authority, _digest, _source = load_json_object(
        SPIN_AUTHORITY, label="SPIn exact camera authority"
    )
    row = _scene_row(authority["scenes"], scene_id)
    record = row["assets"]["camera_map"]
    path = Path(record["path"]).resolve(strict=True)
    digest = sha256_file(path)
    if digest != record["sha256"]:
        raise ValueError(f"{scene_id} camera map SHA-256 differs")
    return path, digest


def _validate_existing(scene_root: Path, *, scene_id: str) -> dict[str, Any] | None:
    receipt_path = scene_root / "receipt.json"
    if not receipt_path.is_file():
        if scene_root.exists():
            raise RuntimeError(
                f"partial signed-field output must be audited: {scene_root}"
            )
        return None
    receipt, _digest, _source = load_json_object(
        receipt_path, label=f"{scene_id} signed-field receipt"
    )
    if (
        receipt.get("artifact_type") != "radio_gs_method_v1_spin9_signed_field_receipt"
        or receipt.get("method_id") != METHOD_ID
        or receipt.get("scene_id") != scene_id
        or receipt.get("safety", {}).get("target_rgb_opened") is not False
        or receipt.get("safety", {}).get("evaluation_masks_opened") is not False
        or receipt.get("safety", {}).get("target_metrics_opened") is not False
    ):
        raise ValueError(f"{scene_id} existing signed-field receipt differs")
    score_rows = [receipt["reference_score"], *receipt["target_scores"]]
    for row in score_rows:
        path = Path(str(row["path"])).resolve(strict=True)
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"{scene_id} signed-field score SHA-256 differs")
    field = Path(str(receipt["field"]["path"])).resolve(strict=True)
    if sha256_file(field) != receipt["field"]["sha256"]:
        raise ValueError(f"{scene_id} final field SHA-256 differs")
    return receipt


def materialize_scene(args: argparse.Namespace) -> dict[str, Any]:
    scene_id = str(args.scene)
    field_root = Path(args.field_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    final_scene_root = output_root / "scenes" / scene_id
    existing = _validate_existing(final_scene_root, scene_id=scene_id)
    if existing is not None:
        return existing

    assets = resolve_scene_assets(scene_id)
    binding = _field_binding(scene_id, field_root)
    camera_map, camera_map_sha = _camera_map(scene_id)
    runtime_config = (field_root / scene_id / "method_v1.yaml").resolve(strict=True)
    dataset, dataset_sha, dataset_path = load_json_object(
        DATASET_MANIFEST, label="SPIn Available-Nine dataset manifest"
    )
    normalized = validate_dataset_manifest(dataset, check_files=False)
    normalized_scene = _scene_row(normalized["scenes"], scene_id)
    raw_scene = _scene_row(dataset["scenes"], scene_id)
    prompt_ids = [str(value) for value in normalized_scene["prompt_frame_ids"]]
    if len(prompt_ids) != 1:
        raise ValueError(f"{scene_id} must have exactly one reference frame")
    reference_id = prompt_ids[0]

    output_root.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(args.scratch_root).expanduser().resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{scene_id}.", dir=output_root))
    try:
        with tempfile.TemporaryDirectory(
            prefix=f"spin9_{scene_id}_dense_", dir=scratch_root
        ) as dense_directory:
            render = render_protocol_scene(
                dataset_path,
                scene_id=scene_id,
                camera_map_path=camera_map,
                config_path=runtime_config,
                checkpoint_path=assets.geometry,
                canonical_field_checkpoint_path=binding["field"],
                canonical_field_checkpoint_schema="factorized-v2",
                expected_canonical_field_checkpoint_sha256=binding["field_sha256"],
                output_dir=dense_directory,
                device=args.device,
            )
            if (
                render.get("safety", {}).get("rgb_files_opened") is not False
                or render.get("safety", {}).get("segmentation_masks_opened")
                is not False
                or render.get("safety", {}).get("evaluation_ground_truth_opened")
                is not False
            ):
                raise ValueError(f"{scene_id} feature renderer safety contract differs")
            outputs = {str(row["frame_id"]): row for row in render["outputs"]}
            expected_ids = [
                reference_id,
                *map(str, normalized_scene["evaluation_frame_ids"]),
            ]
            if list(outputs) != expected_ids:
                raise ValueError(f"{scene_id} rendered protocol frame order differs")
            reference_feature = load_feature_map(
                outputs[reference_id]["feature_path"], layout="chw"
            )
            foreground, background, prompt_metadata = _prompt_prototypes(
                normalized_scene,
                reference_feature,
                base_dir=dataset_path.parent,
            )
            reference_margin = cosine_margin_scores(
                reference_feature, foreground, background
            )
            reference_final = final_scene_root / "reference_score.npy"
            reference_staging = staging / "reference_score.npy"
            reference_sha = _write_numpy(reference_staging, reference_margin)
            target_rows: list[dict[str, Any]] = []
            for frame_id in map(str, normalized_scene["evaluation_frame_ids"]):
                features = load_feature_map(
                    outputs[frame_id]["feature_path"], layout="chw"
                )
                margin = cosine_margin_scores(features, foreground, background)
                relative = Path("scores") / f"{frame_id}.npy"
                score_sha = _write_numpy(staging / relative, margin)
                target_rows.append(
                    {
                        "frame_id": frame_id,
                        "path": str(final_scene_root / relative),
                        "sha256": score_sha,
                        "shape": list(margin.shape),
                        "dtype": "float32",
                    }
                )
            dense_bytes = sum(
                Path(str(row["feature_path"])).stat().st_size
                for row in render["outputs"]
            )

        receipt = {
            "schema_version": 1,
            "artifact_type": "radio_gs_method_v1_spin9_signed_field_receipt",
            "method_id": METHOD_ID,
            "scene_id": scene_id,
            "protocol_hash": normalized["protocol_hash"],
            "dataset_manifest": {
                "path": str(dataset_path),
                "sha256": dataset_sha,
            },
            "field": {
                "path": str(binding["field"]),
                "sha256": binding["field_sha256"],
                "schema": "factorized-v2",
                "gate": str(binding["gate"]),
                "gate_sha256": binding["gate_sha256"],
            },
            "authorities": {
                "method": str(binding["method_authority"]),
                "method_sha256": binding["method_authority_sha256"],
                "readout_preregistration": str(binding["readout_preregistration"]),
                "readout_preregistration_sha256": binding[
                    "readout_preregistration_sha256"
                ],
                "camera_map": str(camera_map),
                "camera_map_sha256": camera_map_sha,
                "runtime_config": str(runtime_config),
                "runtime_config_sha256": sha256_file(runtime_config),
                "geometry_checkpoint": str(assets.geometry),
                "geometry_checkpoint_sha256": assets.geometry_sha256,
            },
            "prompt": prompt_metadata,
            "reference_frame_id": reference_id,
            "reference_score": {
                "frame_id": reference_id,
                "path": str(reference_final),
                "sha256": reference_sha,
                "shape": list(reference_margin.shape),
                "dtype": "float32",
            },
            "target_scores": target_rows,
            "readout": {
                "operator": "reference_prototype_cosine_margin",
                "score_semantics": "cosine_foreground_minus_cosine_background",
                "prototype_reduction": "mean_of_l2_normalized_prompt_pixel_embeddings_then_l2_normalize",
                "dense_feature_persistence": "ephemeral_per_scene_only",
                "ephemeral_dense_feature_bytes": dense_bytes,
                "ephemeral_dense_features_removed_before_receipt_seal": True,
            },
            "safety": {
                "reference_mask_opened": True,
                "target_rgb_opened": False,
                "evaluation_masks_opened": False,
                "target_metrics_opened": False,
                "all_scalar_margins_sealed_before_target_rgb": True,
            },
        }
        _write_json(staging / "receipt.json", receipt)
        final_scene_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final_scene_root)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--field-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--scratch-root",
        default=str(DEFAULT_RUN_ROOT / "method_v1_readout/scratch"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = materialize_scene(build_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "scene_id": report["scene_id"],
                "reference_score": report["reference_score"]["sha256"],
                "target_count": len(report["target_scores"]),
                "target_rgb_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
