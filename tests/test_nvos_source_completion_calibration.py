from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.querying.nvos_source_completion_calibration import (
    PREREGISTRATION_RELATIVE_PATH,
    SCHEMA,
    canonical_sha256,
    compute_source_completion_loo_gate,
    file_sha256,
    load_source_completion_loo_gate,
    source_completion_loo_method_contract,
)


def _stable_masks() -> torch.Tensor:
    masks = torch.zeros((10, 8, 8), dtype=torch.bool)
    masks[:, 1:7, 1:7] = True
    masks[0, 1, 1] = False
    masks[1, 6, 6] = False
    return masks


def _completion_payload(masks: torch.Tensor) -> dict[str, object]:
    tensors = {"trial_masks": masks.contiguous()}
    digests = {name: tensor_sha256(value) for name, value in tensors.items()}
    return {
        "artifact_type": "radio_gs.nvos_sam3_reference_completion",
        "schema_version": 1,
        "authority": {
            "scene_id": "scene",
            "frame_id": "frame",
            "target_rgb_opened": False,
            "target_mask_opened": False,
        },
        "tensor_sha256": digests,
        "tensor_bundle_sha256": canonical_sha256(digests),
        "tensors": tensors,
    }


def _gate_receipt(
    completion: Path,
    completion_sha256: str,
    completion_receipt_sha256: str,
    masks: torch.Tensor,
) -> dict[str, object]:
    computed = compute_source_completion_loo_gate(masks)
    preregistration = (
        Path(__file__).resolve().parents[1] / PREREGISTRATION_RELATIVE_PATH
    ).resolve()
    contract = source_completion_loo_method_contract()
    payload = torch.load(completion, map_location="cpu", weights_only=True)
    return {
        "schema": SCHEMA,
        "scene_id": "scene",
        "prompt_frame_id": "frame",
        "preregistration": {
            "path": str(preregistration),
            "sha256": file_sha256(preregistration),
        },
        "source_completion": {
            "path": str(completion.resolve()),
            "sha256": completion_sha256,
            "receipt_path": "/unused/upstream.json",
            "receipt_sha256": completion_receipt_sha256,
            "tensor_bundle_sha256": payload["tensor_bundle_sha256"],
            "trial_masks_tensor_sha256": payload["tensor_sha256"]["trial_masks"],
        },
        "method_contract": contract,
        "method_contract_sha256": canonical_sha256(contract),
        "source_only_metrics": {
            "per_trial": computed["per_trial"],
            "summary": computed["summary"],
        },
        "decision": {
            "accept_source_completion": computed["accept_source_completion"],
            "action": computed["action"],
        },
        "safety": {
            "computed_before_target_rendering": True,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_opened": False,
        },
    }


def test_all_trial_loo_gate_accepts_stable_source_trials() -> None:
    result = compute_source_completion_loo_gate(_stable_masks())

    assert result["accept_source_completion"] is True
    assert result["summary"]["minimum_heldout_iou"] > 0.9
    assert result["summary"]["failed_trial_indices"] == []


def test_all_trial_loo_gate_abstains_on_one_collapsed_trial() -> None:
    masks = _stable_masks()
    masks[7].zero_()
    masks[7, 0, 0] = True

    result = compute_source_completion_loo_gate(masks)

    assert result["accept_source_completion"] is False
    assert result["summary"]["failed_trial_indices"] == [7]
    assert result["per_trial"][7]["heldout_iou"] == 0.0
    assert result["action"] == "zero_source_completion_reliability_abstain_to_field"


@pytest.mark.parametrize(
    "value",
    [
        torch.ones((9, 8, 8), dtype=torch.bool),
        torch.ones((10, 8, 8), dtype=torch.float32),
        torch.ones((10, 8, 8), dtype=torch.bool, device="meta"),
    ],
)
def test_all_trial_loo_gate_rejects_wrong_tensor_contract(value: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="CPU bool"):
        compute_source_completion_loo_gate(value)


def test_gate_loader_recomputes_and_rejects_forged_decision(tmp_path: Path) -> None:
    masks = _stable_masks()
    completion = tmp_path / "completion.pt"
    torch.save(_completion_payload(masks), completion)
    completion_sha256 = file_sha256(completion)
    upstream_sha256 = "a" * 64
    receipt = _gate_receipt(
        completion, completion_sha256, upstream_sha256, masks
    )
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    loaded = load_source_completion_loo_gate(
        gate,
        expected_gate_sha256=file_sha256(gate),
        completion_path=completion,
        expected_completion_sha256=completion_sha256,
        expected_completion_receipt_sha256=upstream_sha256,
        expected_scene_id="scene",
        expected_frame_id="frame",
    )
    assert loaded["decision"]["accept_source_completion"] is True

    receipt["decision"]["accept_source_completion"] = False
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="decision differs on replay"):
        load_source_completion_loo_gate(
            forged,
            expected_gate_sha256=file_sha256(forged),
            completion_path=completion,
            expected_completion_sha256=completion_sha256,
            expected_completion_receipt_sha256=upstream_sha256,
            expected_scene_id="scene",
            expected_frame_id="frame",
        )


def test_gate_loader_rejects_tampered_completion(tmp_path: Path) -> None:
    masks = _stable_masks()
    completion = tmp_path / "completion.pt"
    torch.save(_completion_payload(masks), completion)
    completion_sha256 = file_sha256(completion)
    receipt = _gate_receipt(completion, completion_sha256, "b" * 64, masks)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    payload = torch.load(completion, map_location="cpu", weights_only=True)
    payload["tensors"]["trial_masks"][0, 0, 0] = True
    torch.save(payload, completion)
    with pytest.raises(ValueError, match="source completion SHA256 differs"):
        load_source_completion_loo_gate(
            gate,
            expected_gate_sha256=file_sha256(gate),
            completion_path=completion,
            expected_completion_sha256=completion_sha256,
            expected_completion_receipt_sha256="b" * 64,
            expected_scene_id="scene",
            expected_frame_id="frame",
        )
