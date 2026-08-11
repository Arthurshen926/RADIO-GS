import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from radio_gs.querying.multiview_region_memory import method_contract
from radio_gs.scripts import materialize_multiview_region_memory as memory
from radio_gs.scripts import materialize_multiview_region_memory_coarse as coarse
from radio_gs.scripts import refine_multiview_region_memory_official_sam3 as bridge
from radio_gs.scripts.score_nvos_object_region_memory import (
    _verify_causal_change_domains,
)
from radio_gs.utils.immutable_artifacts import file_record, sha256_file, write_frozen_json


def _assignment(gaussian_ids, pixel_ids, weights):
    order = torch.argsort(torch.tensor(pixel_ids), stable=True)
    return {
        "gaussian_ids": torch.tensor(gaussian_ids, dtype=torch.int32)[order],
        "pixel_ids": torch.tensor(pixel_ids, dtype=torch.int32)[order],
        "weights": torch.tensor(weights, dtype=torch.float32)[order],
    }


def _pipeline_fixture(tmp_path: Path):
    rgb_root = tmp_path / "rgb"
    rgb_root.mkdir()
    frame_ids = ["reference", "view_a", "view_b", "view_c"]
    source_views = []
    for index, frame_id in enumerate(frame_ids):
        path = rgb_root / f"{frame_id}.png"
        Image.fromarray(
            np.full((24, 32, 3), 32 + index * 20, dtype=np.uint8), mode="RGB"
        ).save(path)
        source_views.append(
            {
                "assignment_view_index": index,
                "frame_id": frame_id,
                "source_file": path.name,
                "rgb_path": str(path),
                "rgb_sha256": sha256_file(path),
                "rgb_bytes": path.stat().st_size,
            }
        )
    responsibility = tmp_path / "responsibility.pt"
    assignments = [
        _assignment([0, 1, 2], [0, 1, 3], [0.9, 0.8, 0.5]),
        _assignment([0, 1, 2], [0, 1, 2], [0.8, 0.7, 0.4]),
        _assignment([0, 1, 3], [1, 2, 3], [0.7, 0.6, 0.5]),
        _assignment([0, 1, 3], [0, 2, 3], [0.6, 0.5, 0.4]),
    ]
    torch.save(
        {
            "schema_version": 1,
            "metadata": {
                "assignment_mode": "raster_gaussian_top1",
                "registration_weight_mode": "alpha_depth",
                "benchmark_images_opened": False,
                "benchmark_masks_opened": False,
                "feature_height": 2,
                "feature_width": 2,
            },
            "assignments": assignments,
        },
        responsibility,
    )
    inventory_path = tmp_path / "source_inventory.json"
    inventory = {
        "schema_version": 1,
        "artifact_type": "nvos_multiview_region_memory_source_inventory_v1",
        "status": "source_rgb_and_assignment_authority_sealed_before_sam3_or_target_access",
        "method_contract": method_contract(),
        "global_safety": {
            "target_rgb_content_opened_or_hashed": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
        },
        "scenes": {
            "scene": {
                "reference_frame_id": "reference",
                "reference_assignment_view_index": 0,
                "forbidden_target_frame_ids": ["target"],
                "forbidden_target_rgb_names": ["target.png"],
                "feature_grid_hw": [2, 2],
                "source_view_count": 4,
                "source_rgb_inventory_sha256": "a" * 64,
                "source_views": source_views,
                "assets": {
                    "responsibility": {
                        "path": str(responsibility),
                        "sha256": sha256_file(responsibility),
                    }
                },
                "safety": {
                    "target_rgb_content_opened_or_hashed": False,
                    "target_mask_opened": False,
                    "target_metric_opened": False,
                },
            }
        },
    }
    write_frozen_json(inventory_path, inventory)

    completed = torch.zeros(
        (coarse.DECODER_HEIGHT, coarse.DECODER_WIDTH), dtype=torch.bool
    )
    completed[: coarse.DECODER_HEIGHT // 2, :] = True
    raw_positive = torch.zeros_like(completed)
    raw_positive[10:20, 10:20] = True
    raw_negative = torch.zeros_like(completed)
    raw_negative[-20:-10, -20:-10] = True
    completion_path = tmp_path / "completion.pt"
    tensors = {
        "completed_positive": completed,
        "raw_positive": raw_positive,
        "raw_negative": raw_negative,
    }
    tensor_sha = {name: coarse._tensor_sha256(value) for name, value in tensors.items()}
    torch.save(
        {
            "artifact_type": "radio_gs.nvos_sam3_reference_completion",
            "schema_version": 1,
            "authority": {
                "scene_id": "scene",
                "frame_id": "reference",
                "target_rgb_opened": False,
                "target_mask_opened": False,
            },
            "tensor_sha256": tensor_sha,
            "tensors": tensors,
        },
        completion_path,
    )
    completion_receipt = tmp_path / "completion_receipt.json"
    write_frozen_json(
        completion_receipt,
        {
            "artifact_sha256": sha256_file(completion_path),
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
        },
    )
    coarse_root = tmp_path / "coarse"
    coarse_receipt = coarse_root / "coarse_receipt.json"
    report = coarse.run(
        argparse.Namespace(
            source_inventory=str(inventory_path),
            source_inventory_sha256=sha256_file(inventory_path),
            scene_id="scene",
            reference_completion=str(completion_path),
            reference_completion_sha256=sha256_file(completion_path),
            reference_completion_receipt=str(completion_receipt),
            reference_completion_receipt_sha256=sha256_file(completion_receipt),
            output_root=str(coarse_root),
            output_receipt=str(coarse_receipt),
        )
    )
    return report, coarse_receipt


def test_coarse_producer_selects_three_nonreference_views_and_is_noclobber(tmp_path):
    report, receipt_path = _pipeline_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    assert report["selected_frame_ids"] == ["view_a", "view_b", "view_c"]
    assert receipt["selection_count"] == 3
    assert receipt["source_access"] == {
        "source_rgb_opened": False,
        "official_sam3_loaded": False,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
        "candidate_selected_with_gt": False,
    }
    assert all(row["frame_id"] not in {"reference", "target"} for row in receipt["selection"])
    with pytest.raises(FileExistsError):
        coarse.run(
            argparse.Namespace(
                source_inventory=receipt["source_inventory"]["path"],
                source_inventory_sha256=receipt["source_inventory"]["sha256"],
                scene_id="scene",
                reference_completion=receipt["reference_anchor"]["source_completion"]["path"],
                reference_completion_sha256=receipt["reference_anchor"]["source_completion"]["sha256"],
                reference_completion_receipt=receipt["reference_anchor"]["source_completion_receipt"]["path"],
                reference_completion_receipt_sha256=receipt["reference_anchor"]["source_completion_receipt"]["sha256"],
                output_root=str(receipt_path.parent),
                output_receipt=str(receipt_path),
            )
        )


class _FakeProcessor:
    device = torch.device("cpu")

    def set_image(self, image):
        return {"image_size": image.size}

    def add_geometric_prompt(self, box, positive, state):
        mask = torch.zeros(
            (1, bridge.DECODER_HEIGHT, bridge.DECODER_WIDTH), dtype=torch.bool
        )
        x0 = max(int((box[0] - box[2] * 0.5) * bridge.DECODER_WIDTH), 0)
        x1 = min(int((box[0] + box[2] * 0.5) * bridge.DECODER_WIDTH), bridge.DECODER_WIDTH)
        y0 = max(int((box[1] - box[3] * 0.5) * bridge.DECODER_HEIGHT), 0)
        y1 = min(int((box[1] + box[3] * 0.5) * bridge.DECODER_HEIGHT), bridge.DECODER_HEIGHT)
        mask[:, y0:y1, x0:x1] = True
        return {"masks": mask, "scores": torch.tensor([0.75])}


def test_source_sam3_bridge_seals_each_view_and_final_receipt_without_gt(
    tmp_path, monkeypatch
):
    coarse_report, coarse_receipt = _pipeline_fixture(tmp_path)
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"fake-sam3")
    monkeypatch.setattr(bridge, "set_requested_cuda_device", lambda device: None)
    monkeypatch.setattr(bridge, "validate_sam3_resolution", lambda value, allow_unsafe: value)
    monkeypatch.setattr(bridge, "_load_sam3_model", lambda **kwargs: _FakeProcessor())
    monkeypatch.setattr(bridge, "resolve_sam3_amp_dtype", lambda device, dtype: torch.bfloat16)
    monkeypatch.setattr(bridge, "sam3_autocast_context", lambda device, dtype: nullcontext())
    monkeypatch.setattr(bridge.torch.cuda, "synchronize", lambda: None)
    output_root = tmp_path / "sam3"
    final_receipt = output_root / "prediction_receipt.json"
    result = bridge.run(
        argparse.Namespace(
            coarse_receipt=str(coarse_receipt),
            coarse_receipt_sha256=coarse_report["sha256"],
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256_file(checkpoint),
            output_root=str(output_root),
            output_receipt=str(final_receipt),
            device="cuda",
        )
    )
    payload = json.loads(final_receipt.read_text())
    assert result["views"] == 3
    assert result["accepted"] == 3
    assert len(payload["view_receipts"]) == 3
    assert payload["source_access"]["source_rgb_opened"] is True
    assert payload["source_access"]["target_rgb_opened"] is False
    assert payload["source_access"]["target_mask_opened"] is False
    assert payload["source_access"]["target_metric_opened"] is False
    for row in payload["view_receipts"]:
        view = json.loads(Path(row["path"]).read_text())
        assert sha256_file(row["path"]) == row["sha256"]
        assert view["source_access"]["candidate_selected_with_gt"] is False
        assert view["source_access"]["target_rgb_opened"] is False
        assert view["sam3_report"]["accepted"] is True
    with pytest.raises(FileExistsError):
        bridge.run(
            argparse.Namespace(
                coarse_receipt=str(coarse_receipt),
                coarse_receipt_sha256=coarse_report["sha256"],
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha256_file(checkpoint),
                output_root=str(output_root),
                output_receipt=str(final_receipt),
                device="cuda",
            )
        )


def test_bridge_rejects_target_identity_before_model_load(tmp_path, monkeypatch):
    coarse_report, coarse_receipt = _pipeline_fixture(tmp_path)
    payload = json.loads(coarse_receipt.read_text())
    payload["selection"][0]["frame_id"] = "target"
    forged = tmp_path / "forged_coarse.json"
    write_frozen_json(forged, payload)
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"fake-sam3")
    called = False

    def _forbidden_model(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("model must not load")

    monkeypatch.setattr(bridge, "_load_sam3_model", _forbidden_model)
    with pytest.raises(ValueError, match="reference or target"):
        bridge.run(
            argparse.Namespace(
                coarse_receipt=str(forged),
                coarse_receipt_sha256=sha256_file(forged),
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha256_file(checkpoint),
                output_root=str(tmp_path / "blocked"),
                output_receipt=str(tmp_path / "blocked" / "receipt.json"),
                device="cuda",
            )
        )
    assert called is False


def test_primitive_region_memory_is_sealed_before_target_and_noclobber(
    tmp_path, monkeypatch
):
    coarse_report, coarse_receipt = _pipeline_fixture(tmp_path)
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"fake-sam3")
    monkeypatch.setattr(bridge, "set_requested_cuda_device", lambda device: None)
    monkeypatch.setattr(bridge, "validate_sam3_resolution", lambda value, allow_unsafe: value)
    monkeypatch.setattr(bridge, "_load_sam3_model", lambda **kwargs: _FakeProcessor())
    monkeypatch.setattr(bridge, "resolve_sam3_amp_dtype", lambda device, dtype: torch.bfloat16)
    monkeypatch.setattr(bridge, "sam3_autocast_context", lambda device, dtype: nullcontext())
    monkeypatch.setattr(bridge.torch.cuda, "synchronize", lambda: None)
    sam3_root = tmp_path / "sam3"
    sam3_receipt = sam3_root / "prediction_receipt.json"
    sam3_report = bridge.run(
        argparse.Namespace(
            coarse_receipt=str(coarse_receipt),
            coarse_receipt_sha256=coarse_report["sha256"],
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256_file(checkpoint),
            output_root=str(sam3_root),
            output_receipt=str(sam3_receipt),
            device="cuda",
        )
    )
    base = tmp_path / "base_unary.pt"
    valid = torch.ones(4, dtype=torch.bool)
    torch.save(
        {
            "schema_version": 1,
            "artifact_type": "nvos_frozen_k16_primitive_unary_probability_v1",
            "scene_id": "scene",
            "valid": valid,
            "valid_rows": torch.where(valid)[0],
            "capability_cache": str(tmp_path / "capability.pt"),
            "capability_cache_sha256": "c" * 64,
            "written_before_target_ground_truth_open": True,
            "target_rgb_opened": False,
            "target_mask_opened": False,
        },
        base,
    )
    artifact = tmp_path / "memory.pt"
    receipt = tmp_path / "memory_receipt.json"
    report = memory.run(
        argparse.Namespace(
            source_sam3_receipt=str(sam3_receipt),
            source_sam3_receipt_sha256=sam3_report["sha256"],
            base_primitive_unary=str(base),
            base_primitive_unary_sha256=sha256_file(base),
            output=str(artifact),
            output_receipt=str(receipt),
        )
    )
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    sealed = json.loads(receipt.read_text())
    assert report["memory_observed_valid_rows"] > 0
    assert payload["membership_probability"].shape == (4,)
    assert payload["membership_confidence"].shape == (4,)
    assert payload["positive_mass_by_view"].shape == (3, 4)
    assert bool((payload["positive_mass_by_view"].sum(dim=1) > 0).all())
    assert sealed["source_access"]["target_rgb_opened"] is False
    assert sealed["source_access"]["target_mask_opened"] is False
    assert sealed["source_access"]["target_metric_opened"] is False
    with pytest.raises(FileExistsError):
        memory.run(
            argparse.Namespace(
                source_sam3_receipt=str(sam3_receipt),
                source_sam3_receipt_sha256=sam3_report["sha256"],
                base_primitive_unary=str(base),
                base_primitive_unary_sha256=sha256_file(base),
                output=str(artifact),
                output_receipt=str(receipt),
            )
        )


def test_causal_replay_keeps_logit_and_probability_change_counts_separate():
    base_logits = torch.tensor([10.0, 0.0], dtype=torch.float32)
    candidate_logits = torch.tensor([10.000002, 0.1], dtype=torch.float32)
    base_probability = torch.sigmoid(base_logits)
    candidate_probability = torch.sigmoid(candidate_logits)
    assert float((candidate_logits[0] - base_logits[0]).abs()) > 1e-6
    assert float((candidate_probability[0] - base_probability[0]).abs()) <= 1e-6
    report = _verify_causal_change_domains(
        candidate_probability=candidate_probability,
        base_probability=base_probability,
        valid=torch.ones(2, dtype=torch.bool),
        memory={
            "valid_rows": torch.arange(2),
            "membership_confidence": torch.ones(2),
        },
        diagnostics={"completed_rows": 2, "materially_changed_rows": 2},
    )
    assert report["logit_domain_materially_changed_rows"] == 2
    assert report["probability_domain_materially_changed_rows"] == 1
    assert report["counts_expected_to_match_across_sigmoid"] is False


def test_causal_replay_rejects_probability_change_outside_memory_authority():
    with pytest.raises(ValueError, match="escape memory-authorized"):
        _verify_causal_change_domains(
            candidate_probability=torch.tensor([0.6, 0.7]),
            base_probability=torch.tensor([0.5, 0.5]),
            valid=torch.ones(2, dtype=torch.bool),
            memory={
                "valid_rows": torch.arange(2),
                "membership_confidence": torch.tensor([1.0, 0.0]),
            },
            diagnostics={"completed_rows": 1, "materially_changed_rows": 1},
        )


def test_causal_replay_accepts_sub_envelope_non_memory_drift_without_bit_flip():
    report = _verify_causal_change_domains(
        candidate_probability=torch.tensor([0.7, 0.4001]),
        base_probability=torch.tensor([0.5, 0.4]),
        valid=torch.ones(2, dtype=torch.bool),
        memory={
            "valid_rows": torch.arange(2),
            "membership_confidence": torch.tensor([1.0, 0.0]),
        },
        diagnostics={"completed_rows": 1, "materially_changed_rows": 1},
    )
    assert report["non_memory_probability_decision_flips_at_0_5"] == 0
    assert report["maximum_probability_drift_outside_memory_authority"] < 1e-3
    assert report["final_material_write_mask_subset_of_memory_authorized_rows"] is True


def test_causal_replay_rejects_sub_envelope_non_memory_decision_flip():
    with pytest.raises(ValueError, match="non-memory decision"):
        _verify_causal_change_domains(
            candidate_probability=torch.tensor([0.7, 0.5002]),
            base_probability=torch.tensor([0.5, 0.4998]),
            valid=torch.ones(2, dtype=torch.bool),
            memory={
                "valid_rows": torch.arange(2),
                "membership_confidence": torch.tensor([1.0, 0.0]),
            },
            diagnostics={"completed_rows": 1, "materially_changed_rows": 1},
        )
