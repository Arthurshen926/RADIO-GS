#!/usr/bin/env python3
"""Run one formal RegionCoMembershipV2 checkpoint without performing readout.

This authority contains only canonical pair probabilities and the globally
selected source-validation rule.  Query/O0 seed readout remains a separate
consumer and is deliberately not reimplemented here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.region_comembership_v2_formal import (
    validate_checkpoint,
    validate_target_execution_authority,
)
from radio_gs.models.region_comembership_v2 import RegionCoMembershipV2
from radio_gs.scripts.materialize_region_comembership_features_v2 import (
    validate_feature_authority,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    validate_file_record,
    write_torch_noclobber,
)


SCHEMA = "radio_gs.region_comembership_inference_authority.v2"
SCHEMA_VERSION = 2
CHANNEL_NAMES = (
    "canonical_region_indices",
    "pair_indices",
    "pair_probabilities",
    "accepted_edge_mask",
)
INFERENCE_BATCH_SIZE = 65536


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "schema",
        "schema_version",
        "scene_id",
        "domain",
        "producer",
        "target_execution_authority",
        "feature_authority",
        "checkpoint",
        "selected_rule",
        "region_fingerprints_sha256",
        "canonical_axis_sha256",
        "pair_axis_sha256",
        "inference_axis_sha256",
        "tensor_authority_sha256",
        "source_access",
    )
    return {name: payload[name] for name in names}


def _inference_axis_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "region_fingerprints_sha256": payload["region_fingerprints_sha256"],
            "canonical_region_indices_sha256": payload["channel_sha256"][
                "canonical_region_indices"
            ],
            "pair_indices_sha256": payload["channel_sha256"]["pair_indices"],
        }
    )


def validate_inference_authority(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "scene_id",
        "domain",
        "producer",
        "target_execution_authority",
        "feature_authority",
        "checkpoint",
        "selected_rule",
        "region_fingerprints_sha256",
        "canonical_axis_sha256",
        "pair_axis_sha256",
        "inference_axis_sha256",
        "tensor_authority_sha256",
        "source_access",
        "content_authority_sha256",
        "region_fingerprints",
        *CHANNEL_NAMES,
        "channel_sha256",
        "audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("RegionCoMembership V2 inference fields differ")
    payload = dict(value)
    if (
        payload.get("schema") != SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("domain") not in {"source_parity", "target"}
        or (payload.get("domain") == "target")
        != (payload.get("target_execution_authority") is not None)
        or payload.get("content_authority_sha256")
        != canonical_json_sha256(_identity(payload))
    ):
        raise ValueError("RegionCoMembership V2 inference identity differs")
    validate_file_record(payload["producer"], label="V2 inference producer")
    feature_path = validate_file_record(
        payload["feature_authority"], label="V2 feature authority"
    )
    checkpoint_path = validate_file_record(
        payload["checkpoint"], label="V2 promoted checkpoint"
    )
    if payload["domain"] == "target":
        validate_file_record(
            payload["target_execution_authority"], label="V2 target execution authority"
        )
    rule = payload.get("selected_rule")
    if (
        not isinstance(rule, Mapping)
        or set(rule) != {"method", "maximum_regions", "threshold"}
    ):
        raise ValueError("RegionCoMembership V2 selected rule differs")
    threshold = float(rule["threshold"])
    canonical = torch.as_tensor(payload["canonical_region_indices"])
    pairs = torch.as_tensor(payload["pair_indices"])
    probability = torch.as_tensor(payload["pair_probabilities"])
    accepted = torch.as_tensor(payload["accepted_edge_mask"])
    count = int(canonical.numel())
    fingerprints = payload.get("region_fingerprints")
    if (
        count <= 0
        or canonical.dtype != torch.int64
        or canonical.ndim != 1
        or bool((canonical < 0).any())
        or int(torch.unique(canonical).numel()) != count
        or not isinstance(fingerprints, list)
        or len(fingerprints) != count
        or len(set(fingerprints)) != count
        or pairs.dtype != torch.int64
        or pairs.ndim != 2
        or pairs.shape[0] != 2
        or bool((pairs < 0).any())
        or bool((pairs >= count).any())
        or bool((pairs[0] >= pairs[1]).any())
        or probability.dtype != torch.float32
        or probability.shape != (pairs.shape[1],)
        or not bool(torch.isfinite(probability).all())
        or bool((probability < 0).any())
        or bool((probability > 1).any())
        or accepted.dtype != torch.bool
        or accepted.shape != probability.shape
        or not torch.equal(accepted, probability >= threshold)
    ):
        raise ValueError("RegionCoMembership V2 inference tensors differ")
    pair_keys = pairs[0] * count + pairs[1]
    if pair_keys.numel() <= 0 or (
        pair_keys.numel() > 1 and not bool((pair_keys[1:] > pair_keys[:-1]).all())
    ):
        raise ValueError("RegionCoMembership V2 inference pairs are not sorted unique")
    if not isinstance(payload.get("channel_sha256"), Mapping) or set(
        payload["channel_sha256"]
    ) != set(CHANNEL_NAMES):
        raise ValueError("RegionCoMembership V2 inference channel mapping differs")
    for name in CHANNEL_NAMES:
        if payload["channel_sha256"][name] != tensor_sha256(payload[name]):
            raise ValueError(f"RegionCoMembership V2 inference changed: {name}")
    feature_raw, _, _ = load_torch_mapping(
        feature_path,
        expected_sha256=payload["feature_authority"]["sha256"],
        map_location="cpu",
        label="V2 inference-bound feature authority",
    )
    feature = validate_feature_authority(feature_raw)
    checkpoint_raw, _, _ = load_torch_mapping(
        checkpoint_path,
        expected_sha256=payload["checkpoint"]["sha256"],
        map_location="cpu",
        label="V2 inference-bound checkpoint",
    )
    checkpoint = validate_checkpoint(checkpoint_raw)
    if payload["selected_rule"] != checkpoint["selected_rule"]:
        raise ValueError("RegionCoMembership V2 inference selected rule differs")
    if payload["domain"] == "target":
        execution_record = payload["target_execution_authority"]
        execution = validate_target_execution_authority(
            execution_record["path"],
            expected_sha256=execution_record["sha256"],
            scene_id=str(payload["scene_id"]),
            expected_feature_output=feature_path,
        )
        if (
            execution["verified_record"] != execution_record
            or execution["verified_checkpoint_record"] != payload["checkpoint"]
            or execution["target_feature_inputs"] != feature["input_authority"]
        ):
            raise ValueError("RegionCoMembership V2 inference target chain differs")
    if (
        payload.get("region_fingerprints_sha256")
        != canonical_json_sha256(fingerprints)
        or payload.get("tensor_authority_sha256")
        != canonical_json_sha256(payload["channel_sha256"])
        or payload.get("inference_axis_sha256") != _inference_axis_sha256(payload)
        or payload.get("canonical_axis_sha256") != feature["canonical_axis_sha256"]
        or payload.get("pair_axis_sha256") != feature["pair_axis_sha256"]
        or payload.get("scene_id") != feature["scene_id"]
        or payload.get("domain") != feature["domain"]
        or payload.get("source_access") != feature["source_access"]
        or fingerprints != feature["region_fingerprints"]
        or not torch.equal(canonical, feature["canonical_region_indices"])
        or not torch.equal(pairs, feature["pair_indices"])
    ):
        raise ValueError("RegionCoMembership V2 inference SHA axis differs")
    return payload


def infer_probabilities(
    feature_payload: Mapping[str, Any], checkpoint_payload: Mapping[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    feature = validate_feature_authority(feature_payload)
    checkpoint = validate_checkpoint(checkpoint_payload)
    normalization = checkpoint["normalization"]
    model = RegionCoMembershipV2(
        normalization["median"], normalization["robust_scale"]
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, feature["pair_features"].shape[0], INFERENCE_BATCH_SIZE):
            stop = min(start + INFERENCE_BATCH_SIZE, feature["pair_features"].shape[0])
            chunks.append(model.probability(feature["pair_features"][start:stop]))
    probability = torch.cat(chunks).float().cpu().contiguous()
    return probability, dict(checkpoint["selected_rule"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"RegionCoMembership V2 inference exists: {output}")
    feature_raw, feature_sha, feature_path = load_torch_mapping(
        args.feature_authority,
        expected_sha256=args.expected_feature_authority_sha256,
        map_location="cpu",
        label="RegionCoMembership V2 feature authority",
    )
    feature = validate_feature_authority(feature_raw)
    checkpoint_raw, checkpoint_sha, checkpoint_path = load_torch_mapping(
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        map_location="cpu",
        label="RegionCoMembership V2 checkpoint",
    )
    checkpoint = validate_checkpoint(checkpoint_raw)
    checkpoint_record = {"path": str(checkpoint_path), "sha256": checkpoint_sha}
    target_execution = None
    if feature["domain"] == "target":
        execution_record = feature["target_execution_authority"]
        execution_path = validate_file_record(
            execution_record, label="V2 target execution authority"
        )
        execution = validate_target_execution_authority(
            execution_path,
            expected_sha256=execution_record["sha256"],
            scene_id=str(feature["scene_id"]),
            expected_feature_output=feature_path,
            expected_inference_output=output,
        )
        if (
            execution["verified_record"] != execution_record
            or execution["verified_checkpoint_record"] != checkpoint_record
            or execution["target_feature_inputs"] != feature["input_authority"]
        ):
            raise ValueError("V2 target feature/result/checkpoint chain differs")
        target_execution = execution["verified_record"]
    probability, selected_rule = infer_probabilities(feature, checkpoint)
    accepted = probability >= float(selected_rule["threshold"])
    identity = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": feature["scene_id"],
        "domain": feature["domain"],
        "producer": file_record(Path(__file__).resolve()),
        "target_execution_authority": target_execution,
        "feature_authority": {"path": str(feature_path), "sha256": feature_sha},
        "checkpoint": checkpoint_record,
        "selected_rule": selected_rule,
        "region_fingerprints_sha256": feature["region_fingerprints_sha256"],
        "canonical_axis_sha256": feature["canonical_axis_sha256"],
        "pair_axis_sha256": feature["pair_axis_sha256"],
        "inference_axis_sha256": "",
        "tensor_authority_sha256": "",
        "source_access": feature["source_access"],
    }
    payload = {
        **identity,
        "region_fingerprints": list(feature["region_fingerprints"]),
        "canonical_region_indices": feature["canonical_region_indices"],
        "pair_indices": feature["pair_indices"],
        "pair_probabilities": probability,
        "accepted_edge_mask": accepted.contiguous(),
        "channel_sha256": {},
        "audit": {
            "canonical_regions": int(feature["canonical_region_indices"].numel()),
            "candidate_pairs": int(probability.numel()),
            "accepted_edges": int(accepted.sum()),
            "selected_method": selected_rule["method"],
            "maximum_regions": int(selected_rule["maximum_regions"]),
            "probability_threshold": float(selected_rule["threshold"]),
            "readout_executed": False,
        },
    }
    payload["channel_sha256"] = {
        name: tensor_sha256(payload[name]) for name in CHANNEL_NAMES
    }
    payload["inference_axis_sha256"] = _inference_axis_sha256(payload)
    payload["tensor_authority_sha256"] = canonical_json_sha256(
        payload["channel_sha256"]
    )
    payload["content_authority_sha256"] = canonical_json_sha256(_identity(payload))
    validate_inference_authority(payload)
    written = write_torch_noclobber(output, payload)
    return {
        "status": "region_comembership_v2_checkpoint_inference_complete",
        "scene_id": feature["scene_id"],
        "domain": feature["domain"],
        "output": file_record(written),
        "selected_rule": selected_rule,
        "audit": payload["audit"],
        "readout_executed": False,
        "target_metric_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-authority", required=True)
    parser.add_argument("--expected-feature-authority-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
