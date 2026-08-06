"""Strict source-only calibration for NVOS source completion.

The gate in this module never consumes an evaluation image, mask, or metric.
It asks a narrower question: is every frozen source-view SAM3 trial supported
by the majority of the other trials?  A failed gate means that the completion
operator abstains; it does not change the raw signed-scribble compiler.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256


SCHEMA = "radio_gs.nvos_source_completion_loo_gate.v1"
METHOD = "all_trial_loo_majority_iou_v1"
TRIAL_COUNT = 10
MAJORITY_THRESHOLD = 0.5
MINIMUM_HELDOUT_IOU = 0.5
PREREGISTRATION_RELATIVE_PATH = (
    "paper/artifacts/"
    "nvos_source_completion_loo_abstention_preregistration_20260805.json"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_completion_loo_method_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "method": METHOD,
        "input": "immutable_official_sam3_binary_trial_masks_before_overwrite",
        "trial_count": TRIAL_COUNT,
        "leave_one_out_consensus": "mean(other_nine)>=0.5",
        "heldout_metric": "binary_jaccard_iou",
        "minimum_each_heldout_iou": MINIMUM_HELDOUT_IOU,
        "acceptance": "all_heldout_trials_pass",
        "empty_policy": "fail_closed",
        "deployment_accept": "retain_probability_preserving_reliability",
        "deployment_abstain": "exact_zero_source_completion_reliability",
        "raw_signed_scribble_path_changed": False,
        "compact_field_path_changed": False,
        "uses_target_rgb_mask_or_metric": False,
        "learned_or_scene_tuned_constants": False,
    }


def compute_source_completion_loo_gate(
    trial_masks: torch.Tensor,
) -> dict[str, object]:
    """Compute the fixed all-trial leave-one-out majority-overlap gate."""

    masks = torch.as_tensor(trial_masks)
    if (
        masks.device.type != "cpu"
        or masks.dtype != torch.bool
        or masks.ndim != 3
        or masks.shape[0] != TRIAL_COUNT
        or masks.shape[1] <= 0
        or masks.shape[2] <= 0
    ):
        raise ValueError(
            f"trial_masks must be CPU bool [{TRIAL_COUNT},H,W] with nonzero H,W"
        )
    masks = masks.contiguous()
    total = masks.to(torch.int16).sum(dim=0)
    records: list[dict[str, object]] = []
    for index in range(TRIAL_COUNT):
        heldout = masks[index]
        other_count = total - heldout.to(torch.int16)
        consensus = (
            other_count.to(torch.float64) / float(TRIAL_COUNT - 1)
        ) >= MAJORITY_THRESHOLD
        intersection = int((heldout & consensus).sum())
        union = int((heldout | consensus).sum())
        heldout_pixels = int(heldout.sum())
        consensus_pixels = int(consensus.sum())
        nonempty = heldout_pixels > 0 and consensus_pixels > 0 and union > 0
        iou = float(intersection / union) if nonempty else 0.0
        passed = bool(nonempty and iou >= MINIMUM_HELDOUT_IOU)
        records.append(
            {
                "trial_index": index,
                "heldout_foreground_pixels": heldout_pixels,
                "loo_consensus_foreground_pixels": consensus_pixels,
                "intersection_pixels": intersection,
                "union_pixels": union,
                "heldout_iou": iou,
                "nonempty": nonempty,
                "passed": passed,
            }
        )
    ious = [float(record["heldout_iou"]) for record in records]
    decision = bool(all(bool(record["passed"]) for record in records))
    return {
        "method_contract": source_completion_loo_method_contract(),
        "per_trial": records,
        "summary": {
            "minimum_heldout_iou": min(ious),
            "mean_heldout_iou": math.fsum(ious) / len(ious),
            "maximum_heldout_iou": max(ious),
            "failed_trial_indices": [
                int(record["trial_index"])
                for record in records
                if not bool(record["passed"])
            ],
        },
        "accept_source_completion": decision,
        "action": (
            "retain_source_completion_reliability"
            if decision
            else "zero_source_completion_reliability_abstain_to_field"
        ),
    }


def _load_completion_trial_authority(
    completion_path: Path,
    *,
    expected_completion_sha256: str,
) -> tuple[Mapping[str, object], torch.Tensor, dict[str, str]]:
    if file_sha256(completion_path) != str(expected_completion_sha256):
        raise ValueError("source completion SHA256 differs")
    payload = torch.load(completion_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("source completion payload must be a mapping")
    tensors = payload.get("tensors")
    if (
        payload.get("artifact_type")
        != "radio_gs.nvos_sam3_reference_completion"
        or payload.get("schema_version") != 1
        or not isinstance(tensors, Mapping)
        or "trial_masks" not in tensors
    ):
        raise ValueError("source completion artifact authority differs")
    digests = {
        str(name): tensor_sha256(torch.as_tensor(value))
        for name, value in sorted(tensors.items())
    }
    if payload.get("tensor_sha256") != digests:
        raise ValueError("source completion tensor hashes differ")
    if payload.get("tensor_bundle_sha256") != canonical_sha256(digests):
        raise ValueError("source completion tensor bundle differs")
    return payload, torch.as_tensor(tensors["trial_masks"]), digests


def load_source_completion_loo_gate(
    path: str | Path,
    *,
    expected_gate_sha256: str,
    completion_path: str | Path,
    expected_completion_sha256: str,
    expected_completion_receipt_sha256: str,
    expected_scene_id: str,
    expected_frame_id: str,
) -> dict[str, object]:
    """Load a gate receipt and recompute its decision from frozen source masks."""

    gate_path = Path(path).expanduser().resolve()
    if file_sha256(gate_path) != str(expected_gate_sha256):
        raise ValueError("source completion LOO gate SHA256 differs")
    receipt = json.loads(gate_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema",
        "scene_id",
        "prompt_frame_id",
        "preregistration",
        "source_completion",
        "method_contract",
        "method_contract_sha256",
        "source_only_metrics",
        "decision",
        "safety",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise ValueError("source completion LOO gate schema keys differ")
    completion = Path(completion_path).expanduser().resolve()
    payload, trial_masks, digests = _load_completion_trial_authority(
        completion,
        expected_completion_sha256=expected_completion_sha256,
    )
    authority = payload.get("authority")
    if not isinstance(authority, Mapping):
        raise ValueError("source completion authority is missing")
    preregistration_path = (
        Path(__file__).resolve().parents[2] / PREREGISTRATION_RELATIVE_PATH
    ).resolve()
    contract = source_completion_loo_method_contract()
    source = receipt["source_completion"]
    preregistration = receipt["preregistration"]
    decision = receipt["decision"]
    safety = receipt["safety"]
    if (
        receipt["schema"] != SCHEMA
        or receipt["scene_id"] != str(expected_scene_id)
        or receipt["prompt_frame_id"] != str(expected_frame_id)
        or authority.get("scene_id") != str(expected_scene_id)
        or authority.get("frame_id") != str(expected_frame_id)
        or not isinstance(preregistration, Mapping)
        or Path(str(preregistration.get("path"))).expanduser().resolve()
        != preregistration_path
        or preregistration.get("sha256") != file_sha256(preregistration_path)
        or not isinstance(source, Mapping)
        or Path(str(source.get("path"))).expanduser().resolve() != completion
        or source.get("sha256") != str(expected_completion_sha256)
        or source.get("receipt_sha256")
        != str(expected_completion_receipt_sha256)
        or source.get("tensor_bundle_sha256")
        != payload.get("tensor_bundle_sha256")
        or source.get("trial_masks_tensor_sha256") != digests["trial_masks"]
        or receipt["method_contract"] != contract
        or receipt["method_contract_sha256"] != canonical_sha256(contract)
        or not isinstance(decision, Mapping)
        or not isinstance(safety, Mapping)
        or safety.get("computed_before_target_rendering") is not True
        or safety.get("target_rgb_opened") is not False
        or safety.get("target_mask_opened") is not False
        or safety.get("target_metric_opened") is not False
    ):
        raise ValueError("source completion LOO gate authority differs")
    recomputed = compute_source_completion_loo_gate(trial_masks)
    if receipt["source_only_metrics"] != {
        "per_trial": recomputed["per_trial"],
        "summary": recomputed["summary"],
    } or decision != {
        "accept_source_completion": recomputed["accept_source_completion"],
        "action": recomputed["action"],
    }:
        raise ValueError("source completion LOO gate decision differs on replay")
    return receipt

