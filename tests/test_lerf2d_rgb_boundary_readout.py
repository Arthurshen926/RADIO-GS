from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from radio_gs.scripts.eval_ours_lerf2d_scalar_maps import (
    CANONICAL_TASK_ID,
    EXPECTED_REGISTRY_ROW,
    SCORE_SEMANTICS,
    FrozenLerf2DContract,
    _occam_protocol_config,
    canonical_query_id,
)
from radio_gs.scripts.materialize_lerf2d_rgb_boundary_predictions import (
    PREDICTION_ARTIFACT_TYPE,
    RGB_ROOT_AUTHORITY_TYPE,
    RgbBoundaryPolicy,
    RgbBoundaryProtocolError,
    choose_rgb_candidate_without_gt,
    materialize,
    occam_coarse_prediction,
)
from radio_gs.scripts.materialize_lerf2d_coarse_prediction_receipt import (
    COARSE_ARTIFACT_TYPE,
    materialize_coarse,
)
from radio_gs.scripts.refine_lerf2d_coarse_receipt_official_sam3 import (
    FINAL_ARTIFACT_TYPE as LERF2D_SAM3_PREDICTION_TYPE,
    Sam3Lerf2DProtocolError,
    choose_candidate as choose_sam3_box_candidate,
    mask_to_box as lerf2d_mask_to_sam3_box,
)
from radio_gs.scripts.score_lerf2d_official_sam3_box_receipt import (
    FIXED_POLICY as LERF2D_SAM3_FIXED_POLICY,
    score as score_lerf2d_sam3_receipt,
)
import radio_gs.scripts.refine_lerf2d_coarse_receipt_official_sam3 as lerf2d_sam3_bridge
from radio_gs.utils.immutable_artifacts import file_record
from radio_gs.scripts.score_lerf2d_sealed_boundary_predictions import score


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_occam_coarse_prediction_preserves_frozen_three_scale_readout() -> None:
    value = np.zeros((3, 40, 40), dtype=np.float32)
    value[1, 10:25, 12:27] = 0.9
    coarse, posterior, chosen, scores, loc_level, coords = occam_coarse_prediction(
        value,
        policy=RgbBoundaryPolicy(),
    )
    assert chosen == 1
    assert loc_level == 1
    assert len(scores) == 3
    assert coarse.dtype == np.bool_
    assert posterior.dtype == np.float32
    assert posterior.shape == coarse.shape == (40, 40)
    assert coords


def test_rgb_candidate_gate_uses_only_posterior_peak_area_and_mass() -> None:
    posterior = np.zeros((20, 20), dtype=np.float32)
    posterior[5:15, 5:15] = 1.0
    coarse = posterior > 0.5
    accepted, report = choose_rgb_candidate_without_gt(
        coarse,
        coarse.copy(),
        posterior,
        policy=RgbBoundaryPolicy(),
    )
    assert report["accepted"] is True
    assert np.array_equal(accepted, coarse)

    misses_peak = np.zeros_like(coarse)
    misses_peak[12:17, 12:17] = True
    rejected, report = choose_rgb_candidate_without_gt(
        coarse,
        misses_peak,
        posterior,
        policy=RgbBoundaryPolicy(),
    )
    assert report["accepted"] is False
    assert report["checks"]["peak_containment"] is False
    assert np.array_equal(rejected, coarse)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, FrozenLerf2DContract]:
    scene = "scene_a"
    frame = "frame_00001"
    query = "red cup"
    query_id = canonical_query_id(scene, frame, 0, query)
    label_root = tmp_path / "labels"
    annotation = label_root / scene / f"{frame}.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps(
            {
                "info": {"name": f"{frame}.jpg", "height": 40, "width": 40},
                "objects": [
                    {
                        "category": query,
                        "bbox": [10, 10, 25, 25],
                        "segmentation": [[10, 10, 25, 10, 25, 25, 10, 25]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "scalar"
    map_path = bundle / "maps" / scene / f"{frame}.npy"
    map_path.parent.mkdir(parents=True)
    scalar = np.zeros((3, 1, 40, 40), dtype=np.float32)
    scalar[1, 0, 10:26, 10:26] = 0.95
    np.save(map_path, scalar, allow_pickle=False)
    freeze_sha = "f" * 64
    manifest = {
        "schema_version": 1,
        "artifact_type": "radio_gs_lerf2d_three_scale_scalar_query_map_bundle",
        "method": "RADIO-GS synthetic current field",
        "canonical_task_id": CANONICAL_TASK_ID,
        "registry_row": EXPECTED_REGISTRY_ROW,
        "protocol_freeze": {"freeze_id": "synthetic", "sha256": freeze_sha},
        "score_semantics": SCORE_SEMANTICS,
        "scales": [
            {"id": "0.25", "value": 0.25, "unit": "meter"},
            {"id": "0.45", "value": 0.45, "unit": "meter"},
            {"id": "0.7", "value": 0.7, "unit": "meter"},
        ],
        "source_artifacts": [],
        "scenes": {
            scene: {
                "frames": {
                    frame: {
                        "annotation_sha256": _sha(annotation),
                        "camera_name": f"{frame}.jpg",
                        "map_file": str(map_path.relative_to(bundle)),
                        "map_resolution_hw": [40, 40],
                        "map_sha256": _sha(map_path),
                        "map_shape_lqhw": [3, 1, 40, 40],
                        "query_ids": [query_id],
                        "query_texts": [query],
                        "scale_ids": ["0.25", "0.45", "0.7"],
                    }
                }
            }
        },
    }
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    scene_root = tmp_path / "dataset" / scene
    image_path = scene_root / "images" / f"{frame}.jpg"
    image_path.parent.mkdir(parents=True)
    rgb = np.full((40, 40, 3), 220, dtype=np.uint8)
    rgb[10:26, 10:26] = np.array([20, 20, 180], dtype=np.uint8)
    assert cv2.imwrite(str(image_path), rgb)
    rgb_authority = {
        "schema_version": 1,
        "artifact_type": RGB_ROOT_AUTHORITY_TYPE,
        "target_rgb_authorized": True,
        "benchmark_masks_opened": False,
        "benchmark_segmentation_opened": False,
        "benchmark_bboxes_opened": False,
        "benchmark_metrics_opened": False,
        "scenes": {scene: {"scene_root": str(scene_root), "image_subdir": "images"}},
    }
    rgb_authority_path = tmp_path / "rgb_authority.json"
    rgb_authority_path.write_text(json.dumps(rgb_authority), encoding="utf-8")
    contract = FrozenLerf2DContract(
        freeze_path="/synthetic/freeze.yaml",
        freeze_sha256=freeze_sha,
        freeze_id="synthetic",
        canonical_task_id=CANONICAL_TASK_ID,
        registry_row=EXPECTED_REGISTRY_ROW,
        scenes=(scene,),
        frames_by_scene={scene: (frame,)},
        labelled_frames=1,
        queries=1,
        protocol_config=_occam_protocol_config(),
    )
    return manifest_path, rgb_authority_path, label_root, contract


def test_prediction_and_scorer_are_separate_and_fail_closed(tmp_path: Path) -> None:
    scalar_manifest, rgb_authority, label_root, contract = _write_fixture(tmp_path)
    output = tmp_path / "predictions"
    prediction = materialize(
        score_manifest=scalar_manifest,
        score_manifest_sha256=_sha(scalar_manifest),
        rgb_authority=rgb_authority,
        rgb_authority_sha256=_sha(rgb_authority),
        output_dir=output,
    )
    assert prediction["artifact_type"] == PREDICTION_ARTIFACT_TYPE
    assert prediction["source_access"]["target_rgb_opened"] is True
    assert prediction["source_access"]["benchmark_masks_opened"] is False
    prediction_manifest = output / "prediction_manifest.json"
    result = score(
        prediction_manifest,
        prediction_manifest_sha256=_sha(prediction_manifest),
        label_root=label_root,
        contract=contract,
    )
    assert result["status"] == "complete_exact_frozen_protocol_evaluation"
    assert result["query_micro"]["objects"] == 1
    assert result["protocol_constraints"]["prediction_sealed_before_any_gt_open"] is True
    assert result["protocol_constraints"]["candidate_selected_with_gt"] is False

    payload = json.loads(prediction_manifest.read_text(encoding="utf-8"))
    payload["source_access"]["candidate_selected_with_gt"] = True
    prediction_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RgbBoundaryProtocolError, match="candidate_selected_with_gt"):
        score(
            prediction_manifest,
            prediction_manifest_sha256=_sha(prediction_manifest),
            label_root=label_root,
            contract=contract,
        )

    payload["source_access"]["candidate_selected_with_gt"] = False
    payload["implementation"]["sha256"] = "0" * 64
    prediction_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RgbBoundaryProtocolError, match="producer implementation changed"):
        score(
            prediction_manifest,
            prediction_manifest_sha256=_sha(prediction_manifest),
            label_root=label_root,
            contract=contract,
        )


def test_lerf2d_official_sam3_box_selector_is_coarse_only() -> None:
    coarse = np.zeros((12, 12), dtype=bool)
    coarse[3:8, 4:9] = True
    candidate_bad = np.zeros_like(coarse)
    candidate_bad[:3, :3] = True
    candidate_good = np.zeros_like(coarse)
    candidate_good[2:9, 3:10] = True
    selected, report = choose_sam3_box_candidate(
        coarse,
        np.stack([candidate_bad, candidate_good]),
        scores=np.array([0.99, 0.01], dtype=np.float32),
        min_initial_iou=0.05,
    )
    assert report["accepted"] is True
    assert report["selected_index"] == 1
    assert np.array_equal(selected, candidate_good)
    assert lerf2d_mask_to_sam3_box(coarse, padding_pixels=16) == [0.5, 0.5, 1.0, 1.0]


def test_lerf2d_coarse_receipt_opens_neither_rgb_nor_gt(tmp_path: Path) -> None:
    scalar_manifest, _rgb_authority, _label_root, _contract = _write_fixture(tmp_path)
    output = tmp_path / "coarse"
    payload = materialize_coarse(
        score_manifest=scalar_manifest,
        score_manifest_sha256=_sha(scalar_manifest),
        output_dir=output,
        scenes="scene_a",
    )
    assert payload["artifact_type"] == COARSE_ARTIFACT_TYPE
    assert payload["cohort"] == {"scenes": ["scene_a"], "queries": 1}
    assert payload["source_access"]["target_rgb_opened"] is False
    assert payload["source_access"]["benchmark_annotation_json_opened"] is False
    assert payload["source_access"]["benchmark_masks_opened"] is False
    coarse = np.load(output / "coarse_masks/scene_a/frame_00001.npy")
    assert coarse.dtype == np.uint8
    assert coarse.shape == (1, 40, 40)


def test_lerf2d_sam3_scorer_validates_receipts_before_gt(tmp_path: Path) -> None:
    scalar_manifest, rgb_authority, label_root, contract = _write_fixture(tmp_path)
    coarse_root = tmp_path / "coarse"
    materialize_coarse(
        score_manifest=scalar_manifest,
        score_manifest_sha256=_sha(scalar_manifest),
        output_dir=coarse_root,
        scenes="scene_a",
    )
    coarse_receipt = coarse_root / "coarse_prediction_receipt.json"
    coarse_payload = json.loads(coarse_receipt.read_text(encoding="utf-8"))
    coarse_entry = coarse_payload["scenes"]["scene_a"]["frames"]["frame_00001"]
    coarse_path = coarse_root / coarse_entry["coarse_prediction_file"]
    prediction = np.load(coarse_path, allow_pickle=False)
    final_root = tmp_path / "final"
    final_path = final_root / "final_masks/scene_a/frame_00001.npy"
    final_path.parent.mkdir(parents=True)
    np.save(final_path, prediction, allow_pickle=False)
    rgb_payload = json.loads(rgb_authority.read_text(encoding="utf-8"))
    rgb_path = (
        Path(rgb_payload["scenes"]["scene_a"]["scene_root"])
        / "images/frame_00001.jpg"
    )
    query = dict(coarse_entry["queries"][0])
    query["boundary"] = {
        "attempted": True,
        "accepted": True,
        "fallback_reason": "accepted",
    }
    final_entry = {
        **{
            key: coarse_entry[key]
            for key in (
                "annotation_sha256",
                "camera_name",
                "query_ids",
                "query_texts",
                "resolution_hw",
                "score_map",
                "coarse_prediction_file",
                "coarse_prediction_sha256",
                "prediction_shape_qhw",
            )
        },
        "coarse_prediction_receipt_root": str(coarse_root),
        "target_rgb": {"path": str(rgb_path), "sha256": _sha(rgb_path)},
        "prediction_file": "final_masks/scene_a/frame_00001.npy",
        "prediction_sha256": _sha(final_path),
        "queries": [query],
    }
    receipt_payload = {
        "schema_version": 1,
        "artifact_type": LERF2D_SAM3_PREDICTION_TYPE,
        "status": "sealed_before_benchmark_mask_or_metric_access",
        "method": "synthetic fixed SAM3 box",
        "policy": LERF2D_SAM3_FIXED_POLICY,
        "coarse_prediction_receipt": {
            "path": str(coarse_receipt),
            "sha256": _sha(coarse_receipt),
        },
        "rgb_root_authority": {"path": str(rgb_authority), "sha256": _sha(rgb_authority)},
        "implementation": file_record(Path(lerf2d_sam3_bridge.__file__).resolve()),
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
            "scenes": ["scene_a"],
            "queries": 1,
            "sam3_candidate_accepted": 1,
        },
        "scenes": {"scene_a": {"frames": {"frame_00001": final_entry}}},
    }
    receipt = final_root / "prediction_receipt.json"
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    result = score_lerf2d_sam3_receipt(
        receipt,
        prediction_receipt_sha256=_sha(receipt),
        label_root=label_root,
        contract=contract,
        scenes=("scene_a",),
    )
    assert result["scene_macro"]["delta_miou"] == 0.0
    assert result["query_micro"]["objects"] == 1
    assert result["protocol_constraints"]["prediction_sealed_before_any_gt_open"] is True

    receipt_payload["source_access"]["candidate_selected_with_gt"] = True
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    with pytest.raises(Sam3Lerf2DProtocolError, match="pre-GT contract"):
        score_lerf2d_sam3_receipt(
            receipt,
            prediction_receipt_sha256=_sha(receipt),
            label_root=label_root,
            contract=contract,
            scenes=("scene_a",),
        )
