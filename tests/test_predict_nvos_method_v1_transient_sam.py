from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from radio_gs.scripts.predict_nvos_method_v1_transient_sam import run_sam_trials
from radio_gs.scripts import score_nvos_method_v1_full8 as scorer


class _Model:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, bool]] = []

    def predict_inst(
        self,
        state,
        *,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool,
    ):
        self.calls.append((point_coords.copy(), point_labels.copy(), multimask_output))
        height, width = state["shape"]
        mask = np.zeros((1, height, width), dtype=bool)
        mask[:, :, : width // 2] = True
        return mask, np.array([0.75], dtype=np.float32), np.zeros((1, 4, 4))


class _Processor:
    def __init__(self) -> None:
        self.model = _Model()

    def set_image(self, image: Image.Image):
        return {"shape": (image.height, image.width)}


def test_transient_sam_uses_ten_signed_trials_without_candidate_selection() -> None:
    margin = np.ones((10, 10), dtype=np.float32)
    margin[:, 5:] = -1.0
    processor = _Processor()

    result = run_sam_trials(
        processor,
        Image.new("RGB", (40, 30)),
        margin,
        device="cpu",
        amp_dtype=None,
    )

    assert len(processor.model.calls) == 10
    for points, labels, multimask in processor.model.calls:
        assert points.shape == (6, 2)
        np.testing.assert_array_equal(labels, np.array([1, 1, 1, 0, 0, 0]))
        assert multimask is False
    assert result["trial_masks"].shape == (10, 1, 30, 40)
    np.testing.assert_array_equal(result["binary_mask"][:, :20], True)
    np.testing.assert_array_equal(result["binary_mask"][:, 20:], False)
    np.testing.assert_allclose(result["continuous_margin"][:, :20], 0.5)
    np.testing.assert_allclose(result["continuous_margin"][:, 20:], -0.5)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full8_barrier_verifies_every_field_and_prediction_before_gt(
    tmp_path, monkeypatch
) -> None:
    scene_order = [
        "fern",
        "flower",
        "fortress",
        "horns_center",
        "horns_left",
        "leaves",
        "orchids",
        "trex",
    ]
    protocol_hash = "f" * 64
    scenes = []
    predictions = {}
    prediction_hashes = {}
    receipt_rows = []
    for scene_id in scene_order:
        frame_id = f"{scene_id}_target"
        scenes.append(
            {
                "scene_id": scene_id,
                "evaluation_frame_ids": [frame_id],
                "frames": [
                    {
                        "frame_id": frame_id,
                        "ground_truth": str(tmp_path / "must_not_open" / f"{scene_id}.png"),
                    }
                ],
            }
        )
        field = tmp_path / f"{scene_id}.pth"
        field.write_bytes(f"field-{scene_id}".encode())
        score = tmp_path / "scores" / scene_id / f"{frame_id}.npy"
        score.parent.mkdir(parents=True)
        np.save(score, np.zeros((2, 2), dtype=np.float32))
        score_sha = _sha(score)
        predictions[scene_id] = {frame_id: str(score)}
        prediction_hashes[scene_id] = {frame_id: score_sha}
        receipt = {
            "artifact_type": "radio_gs_method_v1_nvos_transient_sam_receipt",
            "method_id": "radio-gs-method-v1",
            "scene_id": scene_id,
            "frame_id": frame_id,
            "signed_field_prompt": {"sealed_before_target_rgb_open": True},
            "output": {
                "continuous_margin_path": str(score),
                "continuous_margin_sha256": score_sha,
            },
            "authorities": {
                "method_authority_sha256": None,
                "feature_render_authority": {
                    "field_checkpoint": str(field),
                    "field_checkpoint_sha256": _sha(field),
                    "field_checkpoint_schema": "factorized-v2",
                },
            },
            "safety": {
                "target_rgb_opened": True,
                "target_mask_opened": False,
                "target_metric_opened": False,
                "reference_mask_selection": False,
                "graph_used": False,
                "connected_component_used": False,
            },
        }
        receipt_path = tmp_path / "receipts" / f"{scene_id}.json"
        receipt_path.parent.mkdir(exist_ok=True)
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        receipt_rows.append(
            {"scene_id": scene_id, "path": str(receipt_path), "sha256": _sha(receipt_path)}
        )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
    authority = tmp_path / "authority.json"
    authority.write_text(
        json.dumps({"frozen_cohorts": {"nvos": scene_order}}), encoding="utf-8"
    )
    authority_sha = _sha(authority)
    for row in receipt_rows:
        path = Path(row["path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["authorities"]["method_authority_sha256"] = authority_sha
        path.write_text(json.dumps(payload), encoding="utf-8")
        row["sha256"] = _sha(path)
    prediction = tmp_path / "prediction.json"
    prediction.write_text(
        json.dumps(
            {
                "kind": "promptable_nvs_method_v1_transient_sam_predictions",
                "protocol_hash": protocol_hash,
                "prediction_root": ".",
                "predictions": predictions,
                "prediction_sha256": prediction_hashes,
                "method": {"id": "radio-gs-method-v1"},
                "receipts": receipt_rows,
                "evaluation_performed": False,
                "target_mask_opened": False,
                "target_metric_opened": False,
            }
        ),
        encoding="utf-8",
    )
    checks = []
    monkeypatch.setattr(
        scorer,
        "validate_dataset_manifest",
        lambda _dataset, *, check_files: checks.append(check_files)
        or {"protocol_hash": protocol_hash, "scenes": scenes},
    )
    monkeypatch.setattr(scorer, "validate_method_authority", lambda _value: None)

    barrier = scorer.verify_full8_before_gt(
        dataset_manifest_path=dataset,
        prediction_manifest_path=prediction,
        method_authority_path=authority,
    )

    assert checks == [False]
    assert barrier["scene_order"] == scene_order
    assert barrier[
        "all_eight_receipts_verified_before_first_target_ground_truth_open"
    ] is True
    assert not (tmp_path / "must_not_open").exists()

    prediction_payload = json.loads(prediction.read_text())
    prediction_payload["predictions"].pop("trex")
    prediction.write_text(json.dumps(prediction_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ordered full8"):
        scorer.verify_full8_before_gt(
            dataset_manifest_path=dataset,
            prediction_manifest_path=prediction,
            method_authority_path=authority,
        )
