from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from radio_gs.evaluation.promptable_feature_readout import (
    FeatureReadoutError,
    generate_feature_readout_predictions,
    load_feature_map,
)
from radio_gs.evaluation.promptable_segmentation import (
    ProtocolHashMismatchError,
    compute_protocol_hash,
    evaluate_manifest,
)
from radio_gs.models.radio_adaptors import RadioMLPAdaptor


def _mask(path: Path, values: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8) * 255).save(path)
    return path


def _protocol(benchmark: str) -> dict:
    return {
        "benchmark": benchmark,
        "dataset_version": "unit-test-v1",
        "task": "promptable_nvs_binary_segmentation",
        "prompt_type": "unit-test-reference-prompt",
        "metrics": ["foreground_iou", "pixel_accuracy"],
        "aggregation": "per_frame_then_per_scene_then_dataset_scene_macro",
        "resize": "nearest",
        "prediction_representation": "continuous_margin",
        "threshold_comparison": "greater_or_equal",
        "empty_union_value": 1.0,
        "allow_reference_scoring": False,
        "threshold": {"mode": "fixed", "value": 0.0},
    }


def _bind(manifest: dict) -> dict:
    manifest["protocol_hash"] = compute_protocol_hash(manifest)
    return manifest


def test_nvos_prediction_uses_only_scribbles_and_works_with_missing_target_gt(
    tmp_path: Path,
) -> None:
    positive = _mask(tmp_path / "prompts" / "positive.png", np.array([[1, 0], [0, 0]]))
    negative = _mask(tmp_path / "prompts" / "negative.png", np.array([[0, 0], [0, 1]]))
    prompt_feature = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    target_chw = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    prompt_path = tmp_path / "features" / "reference.npy"
    target_path = tmp_path / "features" / "target.npz"
    prompt_path.parent.mkdir(parents=True)
    np.save(prompt_path, prompt_feature)
    # Exercise HWC and a named npz payload on the target feature.
    np.savez(target_path, features=np.moveaxis(target_chw, 0, -1))

    missing_target_gt = tmp_path / "must_not_be_opened" / "target.png"
    manifest = _bind(
        {
            "schema_version": 1,
            "benchmark": "nvos",
            "protocol": _protocol("NVOS"),
            "scenes": [
                {
                    "scene_id": "fern",
                    "prompt_frame_ids": ["reference"],
                    "calibration_frame_ids": [],
                    "evaluation_frame_ids": ["target"],
                    "frames": [
                        {
                            "frame_id": "reference",
                            "ground_truth": None,
                            "feature_path": str(prompt_path),
                            "feature_layout": "chw",
                        },
                        {
                            "frame_id": "target",
                            "ground_truth": str(missing_target_gt),
                            "feature_path": str(target_path),
                            "feature_layout": "hwc",
                        },
                    ],
                    "prompt": {
                        "type": "positive_negative_scribbles",
                        "frame_id": "reference",
                        "positive_path": str(positive),
                        "negative_path": str(negative),
                    },
                }
            ],
        }
    )

    result = generate_feature_readout_predictions(manifest, tmp_path / "predictions")

    score_path = tmp_path / "predictions" / result["predictions"]["fern"]["target"]
    scores = np.load(score_path)
    assert scores.dtype == np.float32
    assert scores[0, 0] == pytest.approx(1.0)
    assert scores[0, 1] == pytest.approx(-1.0)
    assert scores[1, 0] == pytest.approx(0.0, abs=1e-6)
    assert result["protocol_hash"] == manifest["protocol_hash"]
    assert result["safety"]["evaluation_performed"] is False
    assert result["safety"]["evaluation_ground_truth_opened"] is False
    assert "dataset" not in result
    assert set(result["predictions"]["fern"]) == {"target"}
    assert not missing_target_gt.exists()

    persisted = json.loads(
        (tmp_path / "predictions" / "prediction_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["protocol_hash"] == manifest["protocol_hash"]
    assert persisted["prediction_root"] == "."
    assert "prediction_manifest_path" not in persisted

    # Scoring remains a separate, explicit stage.  Creating GT only after
    # prediction also demonstrates that the emitted manifest is directly
    # consumable by the independent evaluator.
    _mask(missing_target_gt, np.array([[1, 0], [1, 1]]))
    report = evaluate_manifest(
        manifest,
        prediction_manifest=tmp_path / "predictions" / "prediction_manifest.json",
    )
    assert report["dataset"]["foreground_iou"] == pytest.approx(1.0)

    np.save(score_path, np.zeros_like(scores))
    with pytest.raises(ProtocolHashMismatchError, match="Prediction SHA-256"):
        evaluate_manifest(
            manifest,
            prediction_manifest=tmp_path / "predictions" / "prediction_manifest.json",
        )


def test_readout_can_materialize_a_bound_scene_subset(tmp_path: Path) -> None:
    positive = _mask(tmp_path / "positive.png", np.array([[1, 0]]))
    negative = _mask(tmp_path / "negative.png", np.array([[0, 1]]))
    scenes = []
    for scene_id in ("fern", "flower"):
        root = tmp_path / scene_id
        root.mkdir()
        feature = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
        np.save(root / "reference.npy", feature)
        np.save(root / "target.npy", feature)
        scenes.append(
            {
                "scene_id": scene_id,
                "prompt_frame_ids": ["reference"],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": ["target"],
                "frames": [
                    {
                        "frame_id": "reference",
                        "ground_truth": None,
                        "feature_path": str(root / "reference.npy"),
                    },
                    {
                        "frame_id": "target",
                        "ground_truth": str(root / "missing.png"),
                        "feature_path": str(root / "target.npy"),
                    },
                ],
                "prompt": {
                    "type": "positive_negative_scribbles",
                    "frame_id": "reference",
                    "positive_path": str(positive),
                    "negative_path": str(negative),
                },
            }
        )
    manifest = _bind(
        {
            "schema_version": 1,
            "benchmark": "nvos",
            "protocol": _protocol("NVOS"),
            "scenes": scenes,
        }
    )
    for scene_id in ("fern", "flower"):
        root = tmp_path / scene_id
        outputs = []
        for frame_id in ("reference", "target"):
            path = root / f"{frame_id}.npy"
            outputs.append(
                {
                    "feature_path": str(path),
                    "feature_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        (root / "render_manifest.json").write_text(
            json.dumps(
                {
                    "kind": "promptable_nvs_gaussfm_render",
                    "scene_id": scene_id,
                    "protocol_hash": manifest["protocol_hash"],
                    "outputs": outputs,
                    "render_mode": "factorized_v2_affine_normalized_splat",
                    "canonical_field_checkpoint": f"/{scene_id}/field.pth",
                    "canonical_field_checkpoint_sha256": "a" * 64,
                    "canonical_field_checkpoint_schema": "factorized-v2",
                    "checkpoint": f"/{scene_id}/geometry.pth",
                    "checkpoint_sha256": "b" * 64,
                }
            ),
            encoding="utf-8",
        )

    result = generate_feature_readout_predictions(
        manifest,
        tmp_path / "out",
        scene_ids=["flower"],
        feature_layout="chw",
        require_render_authority=True,
    )

    assert set(result["predictions"]) == {"flower"}
    assert result["input"]["selected_scene_ids"] == ["flower"]
    assert result["scenes"][0]["feature_render_authority"][
        "field_checkpoint_schema"
    ] == "factorized-v2"
    with pytest.raises(FeatureReadoutError, match="scene_ids"):
        generate_feature_readout_predictions(
            manifest,
            tmp_path / "bad",
            scene_ids=["missing"],
            feature_layout="chw",
        )


def test_spin_reference_mask_and_feature_root_pattern_with_pt_features(
    tmp_path: Path,
) -> None:
    reference_mask = _mask(
        tmp_path / "annotations" / "reference.png",
        np.array([[1, 0], [0, 0]]),
    )
    feature_root = tmp_path / "field"
    feature_root.joinpath("room").mkdir(parents=True)
    reference_hwc = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    target_hwc = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    torch.save(torch.from_numpy(reference_hwc), feature_root / "room" / "cam_ref.pt")
    torch.save({"feature": torch.from_numpy(target_hwc)}, feature_root / "room" / "cam_eval.pt")

    manifest = _bind(
        {
            "schema_version": 1,
            "benchmark": "spin_nerf",
            "protocol": _protocol("SPIn-NeRF"),
            "scenes": [
                {
                    "scene_id": "room",
                    "prompt_frame_ids": ["image000"],
                    "calibration_frame_ids": [],
                    "evaluation_frame_ids": ["image001"],
                    "frames": [
                        {
                            "frame_id": "image000",
                            "ground_truth": str(reference_mask),
                            "camera_name": "cam_ref",
                        },
                        {
                            "frame_id": "image001",
                            "ground_truth": str(tmp_path / "missing_target.png"),
                            "camera_name": "cam_eval",
                        },
                    ],
                    "prompt": {
                        "type": "reference_binary_mask",
                        "frame_id": "image000",
                        "mask_path": str(reference_mask),
                    },
                }
            ],
        }
    )

    result = generate_feature_readout_predictions(
        manifest,
        tmp_path / "out",
        feature_root=feature_root,
        feature_pattern="{scene_id}/{camera_name}.pt",
        feature_layout="hwc",
    )

    score_path = tmp_path / "out" / result["predictions"]["room"]["image001"]
    scores = np.load(score_path)
    np.testing.assert_allclose(scores, np.array([[1.0, -1.0], [-1.0, 1.0]]))
    scene = result["scenes"][0]
    assert scene["prompt"]["foreground_pixels"] == 1
    assert scene["prompt"]["background_pixels"] == 3
    assert result["method"]["threshold"] == {
        "mode": "fixed",
        "value": 0.0,
        "source": "input_manifest",
    }


def test_radio_sam3_projection_is_labeled_as_adaptor_not_decoder(tmp_path: Path) -> None:
    reference_mask = _mask(tmp_path / "reference.png", np.array([[1, 0]]))
    reference = np.zeros((1280, 1, 2), dtype=np.float32)
    reference[0, 0, 0] = 1.0
    reference[1, 0, 1] = 1.0
    target = reference.copy()
    reference_path = tmp_path / "reference.npy"
    target_path = tmp_path / "target.npy"
    np.save(reference_path, reference)
    np.save(target_path, target)

    adaptor = RadioMLPAdaptor(
        input_dim=1280,
        hidden_dim=4,
        output_dim=1024,
        num_blocks=1,
    )
    checkpoint = tmp_path / "radio.pth"
    torch.save(
        {
            "state_dict": {
                f"_feature_projections.sam3.{key}": value
                for key, value in adaptor.state_dict().items()
            }
        },
        checkpoint,
    )
    manifest = _bind(
        {
            "schema_version": 1,
            "benchmark": "spin_nerf",
            "protocol": _protocol("SPIn-NeRF"),
            "scenes": [
                {
                    "scene_id": "toy",
                    "prompt_frame_ids": ["ref"],
                    "calibration_frame_ids": [],
                    "evaluation_frame_ids": ["eval"],
                    "frames": [
                        {
                            "frame_id": "ref",
                            "ground_truth": str(reference_mask),
                            "feature_path": str(reference_path),
                        },
                        {
                            "frame_id": "eval",
                            "ground_truth": str(tmp_path / "missing.png"),
                            "feature_path": str(target_path),
                        },
                    ],
                    "prompt": {
                        "type": "reference_binary_mask",
                        "frame_id": "ref",
                        "mask_path": str(reference_mask),
                    },
                }
            ],
        }
    )

    result = generate_feature_readout_predictions(
        manifest,
        tmp_path / "projected",
        radio_sam3_adaptor_checkpoint=checkpoint,
        projection_chunk_size=1,
    )

    embedding = result["method"]["embedding_space"]
    assert embedding["label"] == "RADIO SAM3 adaptor embedding"
    assert embedding["radio_sam3_adaptor_applied"] is True
    assert embedding["official_sam_decoder"] is False
    assert embedding["input_dim"] == 1280
    assert embedding["output_dim"] == 1024
    assert "not an official SAM/SAM2/SAM3 mask decoder" in embedding["clarification"]


def test_load_feature_map_fails_closed_on_ambiguous_layout(tmp_path: Path) -> None:
    path = tmp_path / "cube.npy"
    np.save(path, np.zeros((4, 4, 4), dtype=np.float32))
    with pytest.raises(FeatureReadoutError, match="Cannot infer CHW versus HWC"):
        load_feature_map(path)
    assert load_feature_map(path, layout="chw").shape == (4, 4, 4)


def test_readout_rejects_nonzero_or_calibrated_threshold(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "protocol": _protocol("NVOS"),
        "scenes": [
            {
                "scene_id": "toy",
                "prompt_frame_ids": ["ref"],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": ["target"],
                "frames": [
                    {"frame_id": "ref", "ground_truth": None},
                    {"frame_id": "target", "ground_truth": "missing.png"},
                ],
                "prompt": {
                    "type": "positive_negative_scribbles",
                    "frame_id": "ref",
                    "positive_path": "positive.png",
                    "negative_path": "negative.png",
                },
            }
        ],
    }
    manifest["protocol"]["threshold"] = {"mode": "fixed", "value": 0.5}
    with pytest.raises(FeatureReadoutError, match="threshold"):
        generate_feature_readout_predictions(manifest, tmp_path / "out")
