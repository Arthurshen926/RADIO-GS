import json
from pathlib import Path

import numpy as np
from PIL import Image

from radio_gs.five_benchmark_method_v1 import METHOD_ID
from radio_gs.scripts.materialize_spin9_method_v1_signed_field import (
    READOUT_PREREGISTRATION,
)
from radio_gs.scripts.predict_spin9_method_v1_transient_sam import (
    run_candidate_trials,
    verify_signed_full9_before_rgb,
)
from radio_gs.scripts.run_spin9_method_v1_scene import (
    DATASET_MANIFEST,
    METHOD_AUTHORITY,
)
from radio_gs.scripts.run_spin9_method_v1_full9_readout import (
    frozen_scene_order,
    score_command,
    signed_command,
    transient_command,
)
from radio_gs.scripts.score_spin9_method_v1_full9 import (
    verify_full9_before_gt,
)
from radio_gs.utils.immutable_artifacts import sha256_file


class _FakeSamModel:
    def predict_inst(
        self,
        _state: object,
        *,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert point_coords.shape == (6, 2)
        assert point_labels.tolist() == [1, 1, 1, 0, 0, 0]
        assert multimask_output is True
        candidates = np.zeros((3, 6, 8), dtype=np.float32)
        candidates[1] = 1.0
        candidates[2, :, ::2] = 1.0
        return candidates, np.array([0.1, 0.9, 0.5], dtype=np.float32), candidates


class _FakeSamProcessor:
    model = _FakeSamModel()

    def set_image(self, image: Image.Image) -> object:
        assert image.size == (8, 6)
        return object()


def test_spin_multimask_trials_keep_three_candidate_axes() -> None:
    margin = np.zeros((4, 4), dtype=np.float32)
    margin[:2] = 1.0
    margin[2:] = -1.0

    result = run_candidate_trials(
        _FakeSamProcessor(),
        Image.new("RGB", (8, 6)),
        margin,
        device="cpu",
        amp_dtype=None,
    )

    assert result["probabilities"].shape == (3, 6, 8)
    assert float(result["probabilities"][0].mean()) == 0.0
    assert float(result["probabilities"][1].mean()) == 1.0
    assert float(result["probabilities"][2].mean()) == 0.5
    assert result["quality"].shape == (10, 3)


def test_spin_readout_preregistration_freezes_sam_only_reference_selection() -> None:
    payload = json.loads(READOUT_PREREGISTRATION.read_text(encoding="utf-8"))
    selection = payload["transient_sam"]["reference_selection"]

    assert payload["status"] == "frozen_before_first_method_v1_spin9_target_readout"
    assert payload["signed_field_prompt"][
        "reference_and_all_target_signed_margins_sealed_before_target_rgb"
    ]
    assert payload["transient_sam"]["multimask_output"] is True
    assert selection["candidate_indices"] == [0, 1, 2]
    assert selection["canonical_branch_fallback"] is False
    assert selection["target_frame_metric_used"] is False


def test_full9_commands_keep_target_masks_behind_final_barrier() -> None:
    output = Path("/tmp/spin9-full9")
    signed = signed_command(
        scene_id="orchids", field_root=Path("/tmp/fields"), output_root=output
    )
    transient = transient_command(output)
    scorer = score_command(output)

    assert frozen_scene_order() == (
        "orchids",
        "leaves",
        "fern",
        "room",
        "horns",
        "fortress",
        "pinecone",
        "truck",
        "lego",
    )
    assert "score_spin9_method_v1_full9.py" not in " ".join(signed)
    assert "score_spin9_method_v1_full9.py" not in " ".join(transient)
    assert scorer[0] == "radio_gs/scripts/score_spin9_method_v1_full9.py"


def test_signed_full9_barrier_verifies_every_scene_before_rgb(tmp_path: Path) -> None:
    dataset = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    method_sha = sha256_file(METHOD_AUTHORITY)
    prereg_sha = sha256_file(READOUT_PREREGISTRATION)
    signed_root = tmp_path / "signed"
    scene_order = list(dataset["protocol"]["cohort"])

    for scene in dataset["scenes"]:
        scene_id = str(scene["scene_id"])
        scene_root = signed_root / "scenes" / scene_id
        scene_root.mkdir(parents=True)
        field = scene_root / "field.pth"
        field.write_bytes(scene_id.encode("utf-8"))
        reference = scene_root / "reference.npy"
        np.save(reference, np.array([[1.0, -1.0]], dtype=np.float32))
        target_rows = []
        for frame_id in map(str, scene["evaluation_frame_ids"]):
            score = scene_root / f"{frame_id}.npy"
            np.save(score, np.array([[1.0, -1.0]], dtype=np.float32))
            target_rows.append(
                {
                    "frame_id": frame_id,
                    "path": str(score),
                    "sha256": sha256_file(score),
                }
            )
        receipt = {
            "artifact_type": "radio_gs_method_v1_spin9_signed_field_receipt",
            "method_id": METHOD_ID,
            "scene_id": scene_id,
            "protocol_hash": dataset["protocol_hash"],
            "authorities": {
                "method_sha256": method_sha,
                "readout_preregistration_sha256": prereg_sha,
            },
            "field": {"path": str(field), "sha256": sha256_file(field)},
            "reference_score": {
                "path": str(reference),
                "sha256": sha256_file(reference),
            },
            "target_scores": target_rows,
            "safety": {
                "target_rgb_opened": False,
                "evaluation_masks_opened": False,
                "target_metrics_opened": False,
            },
        }
        (scene_root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    barrier = verify_signed_full9_before_rgb(signed_root=signed_root)

    assert (
        barrier["all_signed_margins_and_fields_verified_before_first_rgb_open"] is True
    )
    assert barrier["scene_order"] == scene_order
    assert list(barrier["verified"]) == scene_order


def test_score_full9_barrier_verifies_predictions_before_gt(tmp_path: Path) -> None:
    dataset = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    output_root = tmp_path / "transient"
    output_root.mkdir()
    scene_order = list(dataset["protocol"]["cohort"])
    predictions = {}
    prediction_hashes = {}
    receipts = []

    for scene in dataset["scenes"]:
        scene_id = str(scene["scene_id"])
        scene_root = output_root / "scenes" / scene_id
        scene_root.mkdir(parents=True)
        field = scene_root / "field.pth"
        field.write_bytes(scene_id.encode("utf-8"))
        outputs = []
        scene_predictions = {}
        scene_hashes = {}
        for frame_id in map(str, scene["evaluation_frame_ids"]):
            score = scene_root / f"{frame_id}.npy"
            np.save(score, np.array([[0.1, -0.1]], dtype=np.float32))
            digest = sha256_file(score)
            relative = score.relative_to(output_root).as_posix()
            scene_predictions[frame_id] = relative
            scene_hashes[frame_id] = digest
            outputs.append({"frame_id": frame_id, "path": str(score), "sha256": digest})
        receipt = {
            "artifact_type": "radio_gs_method_v1_spin9_transient_sam_receipt",
            "method_id": METHOD_ID,
            "scene_id": scene_id,
            "field": {"path": str(field), "sha256": sha256_file(field)},
            "reference": {
                "selected_candidate": 0,
                "selected_threshold": 0.5,
                "selected_reference_iou": 0.9,
            },
            "candidate_policy": {
                "multimask_output": True,
                "candidate_count": 3,
                "reference_only_calibration": True,
                "canonical_branch_fallback": False,
            },
            "outputs": outputs,
            "safety": {
                "all_signed_margins_and_fields_verified_before_first_rgb_open": True,
                "reference_mask_opened": True,
                "target_rgb_opened": True,
                "target_mask_opened": False,
                "target_metric_opened": False,
                "reference_mask_selection": True,
                "target_metric_used_for_selection": False,
                "graph_used": False,
                "connected_component_used": False,
            },
        }
        receipt_path = scene_root / "receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        predictions[scene_id] = scene_predictions
        prediction_hashes[scene_id] = scene_hashes
        receipts.append(
            {
                "scene_id": scene_id,
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
            }
        )

    manifest = {
        "kind": "promptable_nvs_method_v1_spin9_transient_sam_predictions",
        "method_id": METHOD_ID,
        "protocol_hash": dataset["protocol_hash"],
        "scene_order": scene_order,
        "prediction_root": ".",
        "predictions": predictions,
        "prediction_sha256": prediction_hashes,
        "receipts": receipts,
        "evaluation_performed": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
        "all_nine_scene_predictions_sealed": True,
    }
    manifest_path = output_root / "prediction_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    barrier = verify_full9_before_gt(
        dataset_manifest_path=DATASET_MANIFEST,
        prediction_manifest_path=manifest_path,
        method_authority_path=METHOD_AUTHORITY,
    )

    assert (
        barrier[
            "all_nine_scene_receipts_and_predictions_verified_before_first_target_mask_open"
        ]
        is True
    )
    assert list(barrier["verified"]) == scene_order
